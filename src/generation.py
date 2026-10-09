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
                                 device="cuda", row_chunk=8):
    """Per-token log-probs of each row's response span.

    Rows are forwarded in chunks of ``row_chunk`` so the (batch, seq, vocab)
    log-softmax activation is only ever materialized for a slice — with the
    v2 rollout budget (32 completions x up to 512+384 tokens, 152k vocab) a
    single full-batch forward would blow past 16GB.

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
    n = len(response_spans)
    log_probs_list: list = [None] * n  # type: ignore[assignment]

    # Left padding puts every response at the row tail, and every span shares
    # the same start; the earliest logits position we ever need is
    # min(start) - 1. Asking the model for only the last K positions
    # (``logits_to_keep``) keeps the vocab-sized, autograd-retained activation
    # at K x vocab instead of seq_len x vocab — the prompt-side lm_head
    # projection is never materialized. With 512-token prompts and 384-token
    # responses this is ~57% less retained memory, which is what lets the
    # training forward fit on an 18GB MPS budget.
    seq_len = full_ids.size(1)
    min_start = min((s for s, _ in response_spans), default=seq_len)
    logits_to_keep = min(max(1, seq_len - min_start + 1), seq_len)
    offset = seq_len - logits_to_keep  # position p -> sliced index p - offset

    for s in range(0, n, row_chunk):
        e = min(s + row_chunk, n)
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(
                input_ids=full_ids[s:e],
                attention_mask=attention_mask[s:e]
                if attention_mask is not None else None,
                logits_to_keep=logits_to_keep)
            logits = outputs.logits
        log_probs_all = torch.log_softmax(logits, dim=-1)
        for i in range(s, e):
            start, end = response_spans[i]
            if end <= start or start < 1:
                log_probs_list[i] = torch.tensor([], device=device)
                continue
            response_token_ids = full_ids[i, start:end]
            # logits at position t predict token t+1 -> shift by one
            logits_for_response = log_probs_all[
                i - s, start - 1 - offset: end - 1 - offset, :]
            token_log_probs = logits_for_response.gather(
                1, response_token_ids.unsqueeze(1)).squeeze(1)
            log_probs_list[i] = token_log_probs
        del outputs, logits, log_probs_all

    return log_probs_list


@torch.no_grad()
def compute_entropy_for_tokens(model, full_ids, response_spans,
                               attention_mask=None, use_amp=False,
                               device="cuda", row_chunk=2):
    """True policy entropy over the full vocabulary at response positions.

    ``H(p) = -sum_v p(v) log p(v)``, averaged over every sampled response
    token.  This complements the sampled-token surprisal proxy (which only
    sees the probabilities of tokens that were actually drawn) with the real
    distribution-level entropy used to detect policy collapse.

    Computed as ``H = logsumexp(z) - sum_v softmax(z)_v * z_v`` to avoid
    materializing an extra vocab-sized copy, in chunks of ``row_chunk``
    (small default: entropy probes run under no_grad but the vocab is 152k).
    """
    full_ids = full_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    device_type = "cuda" if "cuda" in str(device) else (
        "mps" if "mps" in str(device) else "cpu")
    total = 0.0
    count = 0

    # same trailing-slice trick as compute_log_probs_for_tokens: only the
    # response-side positions need vocab logits
    seq_len = full_ids.size(1)
    min_start = min((s for s, _ in response_spans), default=seq_len)
    logits_to_keep = min(max(1, seq_len - min_start + 1), seq_len)
    offset = seq_len - logits_to_keep

    for s in range(0, len(response_spans), row_chunk):
        e = min(s + row_chunk, len(response_spans))
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(
                input_ids=full_ids[s:e],
                attention_mask=attention_mask[s:e]
                if attention_mask is not None else None,
                logits_to_keep=logits_to_keep)
            logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1)
        pos_entropy = torch.logsumexp(logits, dim=-1) \
            - (probs * logits).sum(-1)
        for i in range(s, e):
            start, end = response_spans[i]
            if end <= start or start < 1:
                continue
            seg = pos_entropy[i - s, start - 1 - offset: end - 1 - offset]
            total += float(seg.sum().item())
            count += int(seg.numel())
        del outputs, logits, probs, pos_entropy

    return total / max(count, 1)
