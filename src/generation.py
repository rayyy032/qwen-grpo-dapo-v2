"""Batch generation and per-token log-prob extraction (hand-written, no vLLM).

Decoder-only models require **left padding** for correct batch generation; the
response span of each row is tracked explicitly so that log-probs cover exactly
the sampled tokens (pads / early-stop fill excluded, eos kept).
"""

import torch
from torch.amp import autocast


@torch.no_grad()
def batch_generate(model, tokenizer, prompts, max_new_tokens=256, temperature=1.0,
                   device="cuda", max_prompt_tokens=512):
    """Generate a batch of responses from chat-formatted prompts.

    Args:
        model: causal LM (policy).
        tokenizer: tokenizer owning ``apply_chat_template`` (left padding).
        prompts: list of message lists (``[{"role": ..., "content": ...}]``).
        max_new_tokens: generation budget for the completion.
        temperature: generation temperature.
        device: device string.
        max_prompt_tokens: prompt truncation limit.

    Returns:
        responses: list of decoded completion strings.
        full_ids: (batch, seq_len) prompt + completion tokens (left-padded).
        response_spans: list of (start, end) token positions per row — the
            exactly-sampled span used for log-prob extraction.
        completion_lengths: list of response token counts.
        truncated: list of bool, True when a completion hit ``max_new_tokens``.
    """
    texts = []
    for prompt in prompts:
        if isinstance(prompt, str):
            texts.append(prompt)
        else:
            texts.append(tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True))

    encodings = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    prompt_len = int(input_ids.size(1))  # uniform padded prompt width (left pad)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    pad_id = tokenizer.pad_token_id
    seq_len = int(full_ids.size(1))
    responses = []
    response_spans = []
    completion_lengths = []
    truncated = []
    for i in range(len(texts)):
        region = full_ids[i, prompt_len:]
        pad_positions = (region == pad_id).nonzero(as_tuple=True)[0]
        if pad_positions.numel() > 0:
            # stopped early (eos sampled, then pad fill): span ends before pads
            end = prompt_len + int(pad_positions[0])
            trunc = False
        else:
            end = seq_len
            trunc = True
        response_spans.append((prompt_len, end))
        completion_lengths.append(end - prompt_len)
        truncated.append(trunc)
        responses.append(tokenizer.decode(
            full_ids[i, prompt_len:end], skip_special_tokens=True))

    return responses, full_ids, response_spans, completion_lengths, truncated


def compute_log_probs_for_tokens(model, full_ids, response_spans,
                                 attention_mask=None, use_amp=False,
                                 device="cuda"):
    """Per-token log-probs of each row's response span.

    Args:
        response_spans: list of (start, end) positions from ``batch_generate``.

    Returns:
        List of 1-D float tensors (one per sample, variable length).
    """
    full_ids = full_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    device_type = "cuda" if "cuda" in str(device) else (
        "mps" if "mps" in str(device) else "cpu")
    with autocast(device_type=device_type, enabled=use_amp):
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits

    log_probs_all = torch.log_softmax(logits, dim=-1)

    log_probs_list = []
    for i, (start, end) in enumerate(response_spans):
        if end <= start or start < 1:
            log_probs_list.append(torch.tensor([], device=device))
            continue
        response_token_ids = full_ids[i, start:end]
        # logits at position t predict token t+1 -> shift by one
        logits_for_response = log_probs_all[i, start - 1: end - 1, :]
        token_log_probs = logits_for_response.gather(
            1, response_token_ids.unsqueeze(1)).squeeze(1)
        log_probs_list.append(token_log_probs)

    return log_probs_list
