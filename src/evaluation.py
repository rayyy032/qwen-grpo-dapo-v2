"""Greedy-decoding evaluation with per-source breakdown + optional pass@k."""

from __future__ import annotations

from collections import defaultdict

import torch

from src.rewards import score_answer


@torch.no_grad()
def evaluate_model(agent, eval_data, max_new_tokens=256, overlong_cache=128,
                   enable_overlong=False, format_weight=0.2,
                   pass_k=0, pass_k_samples=50, pass_k_temperature=1.0):
    """Evaluate accuracy / format rate on ``eval_data`` (near-greedy decode).

    When ``pass_k`` > 1, additionally estimates pass@k on the first
    ``pass_k_samples`` prompts: each prompt is sampled ``pass_k`` times at
    ``pass_k_temperature`` and counts as a hit if *any* sample is correct
    (k == n, so the estimator is exact rather than the hypergeometric bound).

    Returns a dict with ``accuracy``, ``format_rate``, ``per_source`` and,
    when enabled, ``pass@k``.
    """
    agent.policy_model.eval()
    n = len(eval_data)
    correct = 0
    valid_format = 0
    per_source = defaultdict(lambda: [0, 0])  # source -> [correct, total]

    # near-greedy: temperature ~0 under do_sample=True, batched for throughput
    chunk = 32
    for i in range(0, n, chunk):
        batch = eval_data[i:i + chunk]
        responses, _full_ids, _spans, comp_lens, truncated = \
            agent.generate_responses(
                [s["prompt"] for s in batch], max_new_tokens=max_new_tokens,
                temperature=0.01, max_prompt_tokens=512)
        for s, resp, cl, tr in zip(batch, responses, comp_lens, truncated):
            r = score_answer(resp, s["ground_truth"], cl, max_new_tokens,
                             overlong_cache, enable_overlong, tr,
                             format_weight)
            correct += int(r.correct)
            valid_format += int(r.valid_format)
            per_source[s.get("data_source", "unknown")][0] += int(r.correct)
            per_source[s.get("data_source", "unknown")][1] += 1
        print(f"[progress] eval greedy {min(i + chunk, n)}/{n}", flush=True)

    result = {
        "accuracy": round(correct / max(n, 1), 4),
        "format_rate": round(valid_format / max(n, 1), 4),
        "per_source": {k: round(v[0] / v[1], 4) for k, v in per_source.items()},
    }

    if pass_k > 1 and eval_data:
        subset = eval_data[:pass_k_samples]
        prompts, gts = [], []
        for s in subset:
            prompts.extend([s["prompt"]] * pass_k)
            gts.extend([s["ground_truth"]] * pass_k)
        any_correct = [False] * len(subset)
        chunk = 32
        for i in range(0, len(prompts), chunk):
            batch_p = prompts[i:i + chunk]
            batch_g = gts[i:i + chunk]
            responses, _f, _s, comp_lens, truncated = \
                agent.generate_responses(
                    batch_p, max_new_tokens=max_new_tokens,
                    temperature=pass_k_temperature, max_prompt_tokens=512)
            for j, (resp, gt) in enumerate(zip(responses, batch_g)):
                r = score_answer(resp, gt, comp_lens[j], max_new_tokens,
                                 overlong_cache, enable_overlong,
                                 truncated[j], format_weight)
                if r.correct:
                    any_correct[(i + j) // pass_k] = True
            print(f"[progress] eval pass@{pass_k} "
                  f"{min(i + chunk, len(prompts))}/{len(prompts)}", flush=True)
        result[f"pass@{pass_k}"] = round(
            sum(any_correct) / max(len(subset), 1), 4)

    return result
