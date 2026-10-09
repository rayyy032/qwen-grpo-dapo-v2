"""Evaluate the frozen base model (0-shot) as the pre-training baseline.

Reuses the exact same eval data + scoring pipeline as ``train.py``'s final
evaluation, so the numbers are directly comparable to the fine-tuned runs.
Passing no LoRA config means ``BaseAgent`` loads the raw checkpoint untouched.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import build_dataset
from src.evaluation import evaluate_model
from src.agents.base_agent import BaseAgent


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate base model 0-shot as the RL baseline")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--eval-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--pass-k", type=int, default=8, help="0 disables pass@k")
    p.add_argument("--pass-k-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = pick_device()
    print(f"device={device}", flush=True)

    t0 = time.time()
    agent = BaseAgent(args.base_model, device=str(device),
                      lora_config=None, use_8bit=False)
    print(f"loaded {args.base_model} in {time.time() - t0:.0f}s", flush=True)

    eval_data = build_dataset(args.dataset, args.eval_samples,
                              split="test", seed=args.seed)
    print(f"eval samples: {len(eval_data)}", flush=True)

    res = evaluate_model(
        agent, eval_data,
        max_new_tokens=args.max_new_tokens,
        overlong_cache=128,
        enable_overlong=False,
        format_weight=0.2,
        pass_k=args.pass_k,
        pass_k_samples=args.pass_k_samples,
    )

    print(json.dumps(res, indent=2), flush=True)
    out = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "eval_samples": args.eval_samples,
        "seed": args.seed,
        **res,
    }
    with open("base_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> base_baseline.json", flush=True)


if __name__ == "__main__":
    main()