# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import re
import sys
import time
import json
from absl import app
from absl import flags
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn.functional as F
import dataclasses
import random
import string
import os
import warnings

# LOCAL-BEGIN
from sentencepiece import sentencepiece_model_pb2
from sentencepiece import SentencePieceProcessor
from model import Transformer

# LOCAL-END

# GOOGLE-BEGIN
# import torch._dynamo as dynamo
# from torch._inductor.codecache import AsyncCompile
# from google3.experimental.users.suvinay.pasta.execution_trace import generate_trace
# from google3.experimental.users.suvinay.pasta.model import Transformer
# from google3.pyglib import gfile
# from google3.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
# from google3.third_party.sentencepiece.src import sentencepiece_model_pb2
# from google3.third_party.sentencepiece.src.python.sentencepiece_processor import SentencePieceProcessor
# from torch.google import distributed as gdist
# open = gfile.Open
# dynamo.config.cache_size_limit = 10000
# GOOGLE-END

FLAGS = flags.FLAGS
flags.DEFINE_string("prompt", "Hello, my name is", "Input prompt.")
flags.DEFINE_boolean("interactive", False, "Whether to launch in interactive mode")
flags.DEFINE_integer("num_samples", 2, "Number of samples.")
flags.DEFINE_integer("max_seq_len", 2048, "Maximum number of new tokens.")
flags.DEFINE_integer("top_k", 200, "Top-k for sampling.")
flags.DEFINE_float("temperature", 0.0, "Temperature for sampling.")
flags.DEFINE_string("checkpoint_path", None, "Model checkpoint path.")
flags.DEFINE_boolean("compile", True, "Whether to compile the model.")
flags.DEFINE_boolean(
    "compile_prefill", False, "Whether to compile the prefill (improves prefill perf, but higher compile times)"
)
flags.DEFINE_string("profile", None, "Profile path.")
flags.DEFINE_integer("speculate_k", 5, "Speculative execution depth.")
flags.DEFINE_string("draft_checkpoint_path", None, "Draft checkpoint path.")
flags.DEFINE_string("device", "cuda", "device to use")
flags.DEFINE_integer("resize_embedding", None, "resize embedding")
flags.DEFINE_string("input_file", None, "input file")
flags.DEFINE_string("model_name", None, "model name")
flags.DEFINE_string("output_file", None, "output file")
flags.DEFINE_string("positional_encoding_mode", "const-40", "positional encoding mode")
flags.DEFINE_boolean("sot", False, "Whether to use sot")

MAX_BATCH_SIZE = 8
MAX_OUTLINE_SEQ_LEN = 500
MAX_CONTENT_SEQ_LEN = 1500


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer


def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    # assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    tokenizer: SentencePieceProcessor,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    print_tokens=False,
    **sampling_kwargs,
):
    new_tokens = [cur_token.clone()]
    is_done = [False] * cur_token.size(0)

    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            is_done = [a or b[0] == tokenizer.eos_id() for a, b in zip(is_done, cur_token.clone().tolist())]
            if print_tokens:
                print("iteration:", i)
                print("starting tokens:", cur_token.clone().tolist())
                print("is_done:", is_done)
            # If all examples in batch reached EOS, break
            if all(is_done):
                # print("All examples reached EOS")
                break

            next_token, next_prob = decode_one_token(model, cur_token, input_pos, **sampling_kwargs)
            input_pos += 1
            # if cur token is 256001, then set next token to 256001
            next_token = torch.where(
                cur_token == tokenizer.eos_id(), tokenizer.eos_id(), next_token
            )
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            # new_probs.append(next_prob.clone())

            cur_token = next_token.clone()

    return new_tokens


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs
    )

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device),
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


@torch.no_grad()
def generate(
    model: Transformer,
    tokenizer: SentencePieceProcessor,
    prompt: torch.Tensor,
    max_seq_len: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    print_tokens=False,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    num_prompt_tokens = T
    batch_size = prompt.size(0)
    # batch_size = MAX_BATCH_SIZE

    # pad to batch size
    # if prompt.size(0) < batch_size:
    #     prompt = F.pad(prompt.clone(), (0, 0, 0, batch_size - prompt.size(0)), value=tokenizer.pad_id())

    # T_new = T + max_new_tokens
    # if interactive:
    #     max_seq_length = 350
    # else:
    #     max_seq_length = min(T_new, model.config.block_size)

    # print("batch_size", batch_size)
    # print("max_seq_len", max_seq_len)

    device, dtype = prompt.device, prompt.dtype
    # max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_len)
        # if is_speculative and draft_model is not model:
        #     draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.ones(batch_size, max_seq_len, dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    # print('prompt shape', prompt.shape)
    # print("prompt", prompt.view(batch_size, -1))
    # print("cache batch size", model.max_batch_size)
    # print("cache seq length", model.max_seq_length)

    next_token = prefill(model, prompt, input_pos, **sampling_kwargs).clone()
    # set to 256001 if the final prompt token is 256001
    prompt_last_tokens = prompt[:, -2:-1]
    next_token = torch.where(
        prompt_last_tokens == tokenizer.eos_id(), tokenizer.eos_id(), next_token
    )
    # print("Initial next_ids", next_token)
    # if is_speculative:
    #     prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    # seq[:, T] = next_token.squeeze()

    # accept_counts = [0] * (speculate_k + 1)

    # if is_speculative:
    #     input_pos = input_pos.item()  # for speculative decoding easier to keep on host
    #     while input_pos < T_new - 1:
    #         cur_token = next_token.view(())

    #         next_tokens = speculative_decode(
    #             model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
    #         )

    #         accept_counts[len(next_tokens) - 1] += 1
    #         num_added = min(T_new - input_pos - 1, len(next_tokens))
    #         seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
    #         for i in next_tokens[: num_added,]:
    #             callback(i)
    #         input_pos = input_pos + num_added
    #         next_token = next_tokens[-1]
    # else:

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    max_new_tokens = max_seq_len - num_prompt_tokens - 1
    time0 = time.time()
    generated_ids = decode_n_tokens(
        model,
        tokenizer,
        next_token.view(batch_size, -1),
        input_pos,
        max_new_tokens - 1,
        callback=callback,
        print_tokens=print_tokens,
        **sampling_kwargs,
    )
    time1 = time.time()

    # print('prompt', prompt)
    # print('generated_ids', generated_ids)
    # seq = torch.cat([prompt, torch.cat(generated_ids, dim=-1)], dim=-1)
    generated_ids = torch.cat(generated_ids, dim=-1)
    n_generated = generated_ids.size(-1)
    seq[:, T : T + n_generated] = generated_ids
    # print('seq', seq)

    # print("Time to generate batch", time1 - time0)

    # generate_stats = {
    #     'accept_counts': accept_counts
    # }
    return seq, (generated_ids, time1 - time0)


# Input prompt is already in chat format, so need to preprocess it to just the prompt
# Assumes it is in the chat template
def preprocess_prompt(prompt: str) -> str:
    return prompt.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]


def generate_outline(
    tokenizer: SentencePieceProcessor,
    model: Transformer,
    max_seq_len: int,
    device: str,
    draft_model: Transformer,
    speculate_k: int,
    interactive: bool,
    callback,
    temperature: float,
    top_k: int,
    prompt: str,
) -> Tuple[torch.Tensor, int]:
    outline_prompt = (
        f"You're an organizer responsible for only giving the skeleton (not the full content) for answering the question. "
        f"Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence, "
        f"each skeleton point should be very short with only 3~5 words. Generally, the skeleton should have 3~10 points.\n\n"
        f"Question:\nWhat are the typical types of Chinese dishes?\nSkeleton:\n1. Dumplings. \n2. Noodles. \n3. Dim Sum. \n4. Hot Pot. \n5. Wonton. \n6. Ma Po Tofu. \n7. Char Siu. \n8. Fried Rice. \n\n"
        f"Question:\nWhat are some practical tips for individuals to reduce their carbon emissions?\nSkeleton:\n1. Energy conservation. \n2. Efficient transportation. \n3. Home energy efficiency. \n4. Reduce water consumption. \n5. Sustainable diet. \n6. Sustainable travel. \n\n"
        f"Now, please provide the skeleton for the following question.\n{prompt}\n\nSkeleton:\n"
    )

    # outline_encoded = encode_tokens(tokenizer, outline_prompt, use_chat=True, bos=True, device=device)
    outline_encoded = encode_tokens(tokenizer, outline_prompt, use_chat=True, starter="1.", bos=True, device=device)

    print("Outline prompt: ", tokenizer.DecodeIds(outline_encoded.tolist()[0]))

    # print("outline shape", outline_encoded.shape)

    outline_seq, (outline_decode_tokens, outline_decode_time) = generate(
        model,
        tokenizer,
        outline_encoded,
        MAX_OUTLINE_SEQ_LEN,
        draft_model=draft_model,
        speculate_k=speculate_k,
        interactive=interactive,
        callback=callback,
        temperature=temperature,
        top_k=top_k,
        # print_tokens=True,
    )
    return outline_decode_tokens, outline_decode_time


# Breakdown the outline into individual points using regex (Second part)
def break_down_outline(outline: str):
    # Use regex to extract points
    # Added \. to the end of the regex to ensure that the last point is also captured
    # this is an improvement to SOT
    re_result = re.findall(r"(\d+)\.\s?([\s\S]+?)(?=\.|\n|\n*$)", outline)

    if len(re_result) > 0:
        points, point_outlines = zip(*re_result)
    else:
        points, point_outlines = [], []

    # print("Points extracted: ", points)
    # print("Point outlines extracted: ", point_outlines)
    return points, point_outlines


def generate_point_content(
    tokenizer: SentencePieceProcessor,
    model: Transformer,
    max_seq_len: int,
    device: str,
    draft_model: Transformer,
    speculate_k: int,
    interactive: bool,
    callback,
    temperature: float,
    top_k: int,
    prompt: str,
    outline: str,
    point_outlines: List[str],
) -> Tuple[torch.Tensor, int]:
    # print("point_outlines: ", point_outlines)

    # prepare prompts
    all_point_prompts = []
    all_point_starters = []
    max_prompt_len = 0
    for point_i, point_outline in enumerate(point_outlines):
        point = str(point_i + 1)
        point_outline = point_outline.strip()
        if point_outline[-1] != ".":
            point_outline = point_outline + "."

        point_prompt = (
            f"You're responsible for continuing the writing of one and only one point in the overall answer to the following question.\n\n"
            f"{prompt}\n\nThe skeleton of the answer is\n\n{outline}\n\n"
            f"Continue and only continue the writing of point {point}. Write it **very shortly** in 1~2 sentence and do not continue with other points!"
        )

        # point_prompt_encoded = encode_tokens(tokenizer, point_prompt, use_chat=True, bos=True, device=device)
        starter = f"{point}. {point_outline}"
        point_prompt_encoded = encode_tokens(
            tokenizer, point_prompt, use_chat=True, starter=starter, bos=True, device=device
        )

        all_point_starters.append(encode_tokens(tokenizer, starter, device=device)[0])
        all_point_prompts.append(point_prompt_encoded)

        max_prompt_len = max(max_prompt_len, point_prompt_encoded.size(1))

        if point == "1":
            print(f"Point {point} prompt: ", tokenizer.DecodeIds(point_prompt_encoded.tolist()[0]))

    # pad left
    for i in range(len(all_point_prompts)):
        all_point_prompts[i] = F.pad(
            all_point_prompts[i], (max_prompt_len - all_point_prompts[i].size(1), 0), value=tokenizer.pad_id()
        )

    all_point_prompts = torch.cat(all_point_prompts, dim=0)

    content_seq, (content_tokens, point_decode_time) = generate(
        model,
        tokenizer,
        all_point_prompts,
        MAX_CONTENT_SEQ_LEN,
        draft_model=draft_model,
        speculate_k=speculate_k,
        interactive=interactive,
        callback=callback,
        temperature=temperature,
        top_k=top_k,
        # print_tokens=True,
    )

    NEWLINE_TOKEN = torch.tensor(tokenizer.encode("\n"), dtype=torch.int, device=device).item()
    # print('newline token', NEWLINE_TOKEN)
    newline_tensor = torch.ones((content_tokens.size(0), 1), dtype=torch.int, device=device) * NEWLINE_TOKEN

    print("content_tokens shape", content_tokens.shape)
    print("newline_tensor shape", newline_tensor.shape)
    print("starter shape", len(all_point_starters))

    # print("all point starters: ")
    # for i in range(len(all_point_starters)):
    # print(f"starter {i+1}", tokenizer.DecodeIds(all_point_starters[i].tolist()))

    points_tokens = []
    point_strs = []
    # print("content tokens: ")
    for i in range(len(all_point_starters)):
        # replace 256001 with eos token
        # tokens_list = [x if x != 1 else 256001 for x in content_tokens[i].tolist()]
        tokens_list = content_tokens[i].tolist()
        print(f"point {i+1} content:", tokenizer.DecodeIds(tokens_list))
        # Extract content until <im_end>
        # if 256001 in content_tokens[i].tolist():
        #     real_content = content_tokens[i][: content_tokens[i].tolist().index(256001)]
        # else:
        #     real_content = content_tokens[i]
        real_content = content_tokens[i]

        print(f"point {i} all point starters: ", all_point_starters[i].shape)
        print(f"point {i} content tokens: ", real_content.shape)
        print(f"point {i} newline tensor: ", newline_tensor[i].shape)
        cur_point_tokens = torch.cat([all_point_starters[i], real_content, newline_tensor[i]], dim=-1)
        points_tokens.append(cur_point_tokens)

        cur_point_tokens_list: list = cur_point_tokens.tolist()
        # if 256001 in cur_point_tokens_list:
        #     cur_point_tokens_list = cur_point_tokens_list[: cur_point_tokens_list.index(256001) + 1]
        point_strs.append(tokenizer.DecodeIds(cur_point_tokens_list))

    points_tokens = torch.cat(points_tokens, dim=0)
    print("points shape", points_tokens.shape)
    # print("points generated: ", tokenizer.DecodeIds(points_tokens.tolist()))

    # replace the last newline token with eos token
    points_tokens[-1] = tokenizer.eos_id()

    return points_tokens.reshape(1, -1), point_decode_time, point_strs


def sot_generate(
    model: Transformer,
    tokenizer: SentencePieceProcessor,
    prompt: str,
    max_seq_len: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    device="cuda",
    **sampling_kwargs,
) -> torch.Tensor:
    torch.manual_seed(42)

    temperature = sampling_kwargs["temperature"]
    top_k = sampling_kwargs["top_k"]

    prompt = preprocess_prompt(prompt)

    t0 = time.time()

    # Step 1: Generate outline
    outline_tokens, outline_decode_time = generate_outline(
        tokenizer,
        model,
        max_seq_len,
        device,
        draft_model,
        speculate_k,
        interactive,
        callback,
        temperature,
        top_k,
        prompt,
    )
    # truncate outline tokens to 256001
    outline_tokens_list: list = outline_tokens.tolist()[0]
    # if 256001 in outline_tokens_list:
    #     outline_tokens_list = outline_tokens_list[: outline_tokens_list.index(256001) + 1]
    #     og_outline = "1." + tokenizer.DecodeIds(outline_tokens_list[:-1])
    #     og_outline_str = "1." + tokenizer.DecodeIds(outline_tokens_list)
    # else:
    og_outline = "1." + tokenizer.DecodeIds(outline_tokens_list)
    og_outline_str = "1." + tokenizer.DecodeIds(outline_tokens_list)
    # og_outline = tokenizer.DecodeIds(outline_tokens.tolist()[0])

    print("Outline time: ", outline_decode_time)
    print("Outline generated: ", og_outline)

    # Step 2: Break down outline into points
    points, point_outlines = break_down_outline(og_outline)

    print("Got a total of", len(points), "points")

    # Improvement from SOT: dedup by point outline not point indices
    # Deduplicate points
    point_outlines_filtered = []
    points_set = set([])
    for i in range(len(points)):
        # Get unique point outline
        if point_outlines[i] not in points_set:
            points_set.add(point_outlines[i])
            point_outlines_filtered.append(point_outlines[i])

    point_outlines = point_outlines_filtered
    print("After deduplication, got a total of", len(point_outlines), "points")

    # cap at max batch size
    if len(point_outlines) > MAX_BATCH_SIZE:
        print("Capping at max batch size")
    point_outlines = point_outlines[:MAX_BATCH_SIZE]

    # recontruct outline
    outline = "\n".join([f"{point+1}. {point_outline}" for point, point_outline in enumerate(point_outlines)])

    print("Reconstructed outline: ", outline)

    # Step 3: Generate content for each point in a batch
    points_tokens, points_decode_time, point_strs = generate_point_content(
        tokenizer,
        model,
        max_seq_len,
        device,
        draft_model,
        speculate_k,
        interactive,
        callback,
        temperature,
        top_k,
        prompt,
        outline,
        point_outlines,
    )

    t1 = time.time()

    # Format in the expected way
    prompt_tokens = encode_tokens(tokenizer, prompt, use_chat=True, bos=True, device=device)
    print("prompt_tokens shape", prompt_tokens.shape)
    print("points_tokens shape", points_tokens.shape)
    print("prompt tokens", prompt_tokens)
    print("point tokens", points_tokens)
    seq = torch.cat([prompt_tokens, points_tokens], dim=-1)
    print("seq", seq)
    print("seq text", tokenizer.DecodeIds(seq.tolist()[0]))

    return seq, (
        points_tokens,
        outline_tokens,
        points_decode_time,
        outline_decode_time,
        og_outline_str,
        point_strs,
        t1 - t0,
    )


def encode_tokens(tokenizer, string, starter=None, use_chat=False, bos=True, device="cuda"):
    if use_chat:
        # Hardcode it bc we will only have a single user prompt message
        system_prompt = "You are a helpful AI assistant"
        # string = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{string}<|im_end|>\n<|im_start|>assistant\n"
        string = f"<start_of_turn>user\n{string}<end_of_turn>\n<start_of_turn>model\n"
        if starter:
            string += starter

    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor([tokens], dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp, resize_embedding=None, model_name=None):
    with torch.device("meta"):
        model_name = model_name or checkpoint_path.parent.name
        model = Transformer.from_name(model_name)

    # if "int8" in str(checkpoint_path):
    #     print("Using int8 weight-only quantization!")
    #     from quantize import WeightOnlyInt8QuantHandler
    #     simple_quantizer = WeightOnlyInt8QuantHandler(model)
    #     model = simple_quantizer.convert_for_runtime()

    # if "int4" in str(checkpoint_path):
    #     print("Using int4 quantization!")
    #     path_comps = checkpoint_path.name.split(".")
    #     assert path_comps[-2].startswith("g")
    #     groupsize = int(path_comps[-2][1:])
    #     from quantize import WeightOnlyInt4QuantHandler
    #     simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
    #     model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if resize_embedding is not None:
        print(f"Resizing embedding dimension to {resize_embedding} ...")
        old_tok_embedding_weight = model.tok_embeddings.weight
        _, embed_dim = old_tok_embedding_weight.shape
        model.tok_embeddings = torch.nn.Embedding(resize_embedding, embed_dim)
        model.tok_embeddings.weight.data[: old_tok_embedding_weight.size(0)] = old_tok_embedding_weight

    # if use_tp:
    #     from tp import apply_tp
    #     print("Applying tensor parallel to model ...")
    #     apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main_fn(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_seq_len: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Optional[str] = None,
    _compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[str] = None,
    speculate_k: int = 5,
    device="cuda",
    input_file=None,
    resize_embedding=None,
    model_name=None,
    output_file=None,
    sot=False,
) -> None:
    global _debug_tokenizer
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    # assert checkpoint_path.is_file(), checkpoint_path

    checkpoint_path = Path(checkpoint_path)
    draft_checkpoint_path = Path(draft_checkpoint_path) if draft_checkpoint_path else None

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    # assert tokenizer_path.is_file(), tokenizer_path

    # global print
    # from tp import maybe_init_dist
    # rank = maybe_init_dist()
    rank = None
    use_tp = rank is not None
    # if use_tp:
    #     if rank != 0:
    #         # only print on rank 0
    #         print = lambda *args, **kwargs: None

    if "RANK" in os.environ:
        eval_rank = int(os.environ["RANK"])
        torch.cuda.set_device(eval_rank)
        # torch.set_default_device(f"cuda:{eval_rank}")
        # device = f"cuda:{eval_rank}"

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print(f"Loading model from {checkpoint_path}")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp, resize_embedding, model_name)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # LOCAL-BEGIN
    m = sentencepiece_model_pb2.ModelProto()
    m.ParseFromString(open(tokenizer_path, "rb").read())
    # LOCAL-END

    # GOOGLE-BEGIN
    #     m = sentencepiece_model_pb2.ModelProto()
    #     m.ParseFromString(
    #         gfile.Open(
    #             "/cns/sc-d/home/suvinay/checkpoints/gemma/gemma-7b/tokenizer.model",
    #             "rb",
    #         ).read()
    #     )
    # GOOGLE-END

    tokenizer = SentencePieceProcessor()
    tokenizer.LoadFromSerializedProto(m.SerializeToString())
    special_tokens = ["<|im_start|>", "<|im_end|>"]

    # for token in special_tokens:
    #     new_token = m.SentencePiece()
    #     new_token.piece = token
    #     new_token.score = 0
    #     new_token.type = 4 # type value for USER_DEFINED
    #     m.pieces.append(new_token)
    #     tokenizer.eos_id = lambda: 256001
    

    print("Vocab size:", len(m.pieces))

    # LOCAL-BEGIN
    tokenizer = SentencePieceProcessor(model_proto=m.SerializeToString())
    # LOCAL-END

    # GOOGLE-BEGIN
    #     tokenizer = SentencePieceProcessor()
    #     tokenizer.LoadFromSerializedProto(m.SerializeToString())
    # GOOGLE-END

    _debug_tokenizer = tokenizer

    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    if _compile:
        print("compiling...")
        if is_speculative and use_tp:  # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False  # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill

        fullgraph = True
        # CHECK-BEGIN
        #         fullgraph = False
        # CHECK-END

        decode_one_token = torch.compile(decode_one_token, mode="max-autotune", fullgraph=fullgraph)

        # Uncomment to squeeze more perf out of prefill
        # if args.compile_prefill:
        #    prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    name_to_configs = {}
    eval_rank = None
    if input_file:
        print(f"Loading input file: {input_file}")
        with open(input_file, "r") as f:
            all_configs = json.load(f)
            print(f"Loaded {len(all_configs)} configs")
            # Check if RANK is set
            if "RANK" in os.environ:
                eval_rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                print(f"Rank is {eval_rank}/{world_size}")
                # Take only the configs that are divisible by world_size
                all_configs = all_configs[eval_rank::world_size]

            for config in all_configs:
                name = config["name"]
                prompt = config["prompt"]
                name_to_configs[name] = prompt

    output_file = str(output_file)
    if eval_rank is not None:
        # insert .{eval_rank} before the extension
        output_file = Path(output_file)
        output_file = output_file.with_name(output_file.stem + f".{eval_rank}" + output_file.suffix)

    try:
        with open(output_file, "r") as f:
            pass
        raise ValueError(f"Output file {output_file} already exists! Exiting.")
    except:
        with open(output_file, "w") as f:
            pass

    log_file = open(output_file, "a")
    for name, config in name_to_configs.items():
        prompt = config
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
        prompt_length = encoded.size(0)

        aggregate_metrics = {
            "tokens_per_sec": [],
            "accept_counts": [],
        }
        start = -1 if _compile else 0

        torch.manual_seed(1234)
        for i in range(start, num_samples):
            print(f"Sample {i + 1} of {num_samples}")
            device_sync(device=device)  # MKG
            # if i >= 0 and interactive:
            #     prompt = input("What is your prompt? ")
            #     if is_chat:
            #         prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            #     encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

            # if interactive and i >= 0:
            #     buffer = []
            #     period_id = tokenizer.EncodeAsIds('.')[0]
            #     done_generating = False
            #     def callback(x):
            #         nonlocal done_generating
            #         if done_generating:
            #             return
            #         buffer.append(tokenizer.DecodeIds([period_id] + x.tolist())[1:])
            #         if x.item() == tokenizer.eos_id():
            #             done_generating = True
            #         if len(buffer) == 4 or done_generating:
            #             print(''.join(buffer), end='', flush=True)
            #             buffer.clear()
            #         # print(, end='', flush=True)
            # else:
            callback = lambda x: x
            t0 = time.perf_counter()
            import contextlib

            if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
                prof = contextlib.nullcontext()
            else:
                print("*****Profiling*****")
                torch.profiler._utils._init_for_cuda_graphs()
                prof = torch.profiler.profile()

            print("Generating...")

            with prof as p:
                if sot:
                    y, (
                        point_tokens,
                        outline_tokens,
                        points_decode_time,
                        outline_decode_time,
                        og_outline,
                        point_strs,
                        total_time,
                    ) = sot_generate(
                        model,
                        tokenizer,
                        prompt,
                        max_seq_len,
                        draft_model=draft_model,
                        speculate_k=speculate_k,
                        interactive=interactive,
                        callback=callback,
                        temperature=temperature,
                        top_k=top_k,
                        device=device,
                    )
                    decode_time = points_decode_time + outline_decode_time
                    tokens_generated = point_tokens.size(-1)
                else:
                    # Just do regular generation
                    y, (decode_tokens, decode_time) = generate(
                        model,
                        tokenizer,
                        encoded,
                        max_seq_len,
                        draft_model=draft_model,
                        speculate_k=speculate_k,
                        interactive=interactive,
                        callback=callback,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    tokens_generated = decode_tokens.size(-1)
            if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
                pass
            else:
                pass
                # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                # print("==========")
                # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
                # aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
            if i == -1:
                print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
                continue
            if hasattr(prof, "export_chrome_trace"):
                if use_tp:
                    prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
                else:
                    prof.export_chrome_trace(f"{profile}.json")
            device_sync(device=device)  # MKG
            t = time.perf_counter() - t0
            # print('ylist shape', f"({len(y.tolist())}, {len(y.tolist()[0])})")

            output = tokenizer.DecodeIds(y.tolist()[0])
            print(output)

            tokens_sec = tokens_generated / decode_time
            aggregate_metrics["tokens_per_sec"].append(tokens_sec)
            print(f"Tokens generated: {tokens_generated}, time taken: {t:.02f} sec")
            print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
            print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        print("==========")
        if is_speculative:
            counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
            acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

        print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        json_result = {
            "name": name,
            "tokens_per_sec": aggregate_metrics["tokens_per_sec"],
            "decode_time": decode_time,
            "output": output,
            # CHECK-BEGIN
            #             "inconsistency_with_reference": ctx.inconsistency_with_reference,
            # CHECK-END
        }
        if sot:
            json_result.update(
                {
                    "outline_time": outline_decode_time,
                    "points_time": points_decode_time,
                    "total_time": total_time,
                    "outline": og_outline,
                    "point_strs": point_strs,
                }
            )
        log_file.write(json.dumps(json_result) + "\n")
        print(f"JSON: {json.dumps(json_result)}")


def main(argv):
    del argv
    print("Compile flag is ", FLAGS.compile)
    main_fn(
        FLAGS.prompt,
        FLAGS.interactive,
        FLAGS.num_samples,
        FLAGS.max_seq_len,
        FLAGS.top_k,
        FLAGS.temperature,
        FLAGS.checkpoint_path,
        FLAGS.compile,
        FLAGS.compile_prefill,
        FLAGS.profile,
        FLAGS.draft_checkpoint_path,
        FLAGS.speculate_k,
        FLAGS.device,
        FLAGS.input_file,
        FLAGS.resize_embedding,
        FLAGS.model_name,
        FLAGS.output_file,
        FLAGS.sot,
    )


if __name__ == "__main__":
    # LOCAL-BEGIN
    app.run(main)
# LOCAL-END

# See: https://g3doc.corp.google.com/learning/pytorch/example/compiler/README.md?cl=head
# GOOGLE-BEGIN
#     g3_multiprocessing.handle_main(gdist.torchrun(main))
# GOOGLE-END
