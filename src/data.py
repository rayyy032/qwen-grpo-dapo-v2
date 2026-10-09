"""Dataset builders: GSM8K / MATH difficulty-gradient mix + offline synthetic.

``dataset`` config values:
  * ``synthetic``    -> tiny offline math QA list (smoke tests, no network).
  * ``gsm8k``         -> GSM8K (single difficulty level).
  * ``gsm8k_math``    -> GSM8K + MATH mixed, stratified by MATH level 1-5.
  * ``/path.jsonl``   -> local JSONL with ``{"question":..., "answer":...}`` and
    optional ``"data_source"`` / ``"level"`` fields.

Each sample is stored as a dict with ``prompt`` (list of chat messages),
``ground_truth`` (string), ``data_source`` and ``level``.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

_SYSTEM_PROMPT = (
    "Please solve the math problem step by step and put the final numeric "
    "answer inside \\boxed{}."
)

# Tiny offline fallback used by smoke tests so training runs with no dataset
# download and no HF authentication.
_SYNTHETIC = [
    ("John has 5 apples and buys 3 more. How many apples does he have?", "8", "synthetic", "1"),
    ("Compute 12 * 12.", "144", "synthetic", "1"),
    ("What is 7 + 6?", "13", "synthetic", "1"),
    ("Half of 50 is what number?", "25", "synthetic", "1"),
    ("A train travels 60 miles per hour for 2 hours. How far?", "120", "synthetic", "1"),
]


def _make_sample(question: str, answer: str, data_source: str, level) -> dict:
    gt = _extract_gt(answer)
    return {
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ground_truth": gt,
        "data_source": data_source,
        "level": level,
    }


def _extract_gt(answer: str) -> str:
    """Normalize a ground-truth string into a pairable numeric/LaTeX token."""
    text = answer.strip()
    m = re.search(r"####\s*(.+)$", text)
    if m:
        text = m.group(1).strip()
    return text


def build_synthetic(n: int) -> List[dict]:
    samples = [_make_sample(q, a, ds, lv) for q, a, ds, lv in _SYNTHETIC]
    # repeat to satisfy requested size if needed
    while len(samples) < n:
        samples.extend([_make_sample(q, a, ds, lv) for q, a, ds, lv in _SYNTHETIC])
    return samples[:n]


def _load_gsm8k(split: str = "train") -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for ex in ds:
        out.append(_make_sample(ex["question"], ex["answer"], "gsm8k", 1))
    return out


def _load_math(split: str = "train", levels: Optional[set] = None) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split=split)
    out = []
    for ex in ds:
        lv = ex.get("level")
        if levels is not None and str(lv) not in levels:
            continue
        out.append(_make_sample(ex["problem"], ex["solution"], "math", str(lv)))
    return out


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(_make_sample(
                rec["question"], rec["answer"],
                rec.get("data_source", "local"), rec.get("level", "1")))
    return out


def build_dataset(dataset: str, num_samples: int, split: str = "train",
                  seed: int = 42, levels: Optional[set] = None) -> List[dict]:
    """Build and shuffle a training list, truncated to ``num_samples``."""
    if dataset == "synthetic":
        samples = build_synthetic(num_samples)
    elif dataset == "gsm8k":
        samples = _load_gsm8k(split)
    elif dataset in ("gsm8k_math", "mix"):
        gsm = _load_gsm8k(split)
        math = _load_math(split, levels)
        samples = gsm + math
    elif os.path.isfile(dataset):
        samples = _load_jsonl(dataset)
    else:
        raise ValueError(f"Unknown dataset spec: {dataset!r}")

    import random
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]
