"""Dataset builders: GSM8K / MATH difficulty-gradient mix + offline synthetic.

``dataset`` config values:
  * ``synthetic``    -> tiny offline math QA list (smoke tests, no network).
  * ``gsm8k``        -> GSM8K (single difficulty tier).
  * ``gsm8k_math``   -> GSM8K + MATH mixed into a difficulty gradient:
                        GSM8K forms the easy tier, MATH is stratified by
                        level 1-5 with per-level quotas (``math_ratio`` sets
                        the gsm/math split).
  * ``/path.jsonl``  -> local JSONL with ``{"question":..., "answer":...}`` and
    optional ``"data_source"`` / ``"level"`` fields.

Each sample is a dict with ``prompt`` (list of chat messages), ``ground_truth``
(string), ``data_source`` and ``level``.

Ground-truth extraction is source-aware (v1 bug: a single extractor handled
only the GSM8K ``####`` marker, so MATH samples used the *entire solution
text* as the target):
  * GSM8K  -> text after the final ``####``.
  * MATH   -> content of the last ``\\boxed{...}`` in the reference solution;
              samples without one are skipped.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, List, Optional

from src.rewards import extract_last_boxed

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

# Fallback source for MATH: hendrycks/competition_math is gated on the Hub,
# EleutherAI/hendrycks_math mirrors the same data split into 7 subject configs.
_MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


# ---------------------------------------------------------------------------
# ground-truth extraction (per data source)
# ---------------------------------------------------------------------------
def _extract_gt_gsm8k(answer: str) -> Optional[str]:
    """GSM8K answers: the number after the final ``####`` marker."""
    m = re.search(r"####\s*(.+)$", answer.strip())
    return m.group(1).strip() if m else None


def _extract_gt_math(solution: str) -> Optional[str]:
    """MATH answers: the last well-formed ``\\boxed{...}`` in the solution."""
    return extract_last_boxed(solution)


def _extract_gt_auto(answer: str) -> Optional[str]:
    """JSONL input: GSM8K marker first, then boxed fallback, else raw text."""
    gt = _extract_gt_gsm8k(answer)
    if gt is None:
        gt = _extract_gt_math(answer)
    if gt is None:
        gt = answer.strip()
    return gt if gt else None


def _norm_level(level) -> Optional[str]:
    """Normalize a level field to '1'..'5'; None if unparseable."""
    s = str(level).strip().lower()
    m = re.match(r"level\s*([1-5])$", s)
    if m:
        return m.group(1)
    if s in ("1", "2", "3", "4", "5"):
        return s
    return None


def _make_sample(question: str, answer: str, data_source: str, level,
                 extractor: Callable[[str], Optional[str]]) -> Optional[dict]:
    gt = extractor(answer)
    if not gt:
        return None  # un-extractable ground truth -> skip the sample
    return {
        "prompt": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ground_truth": gt,
        "data_source": data_source,
        "level": str(level),
    }


# ---------------------------------------------------------------------------
# source loaders
# ---------------------------------------------------------------------------
def build_synthetic(n: int) -> List[dict]:
    samples = [_make_sample(q, a, ds, lv, lambda t: t.strip())
               for q, a, ds, lv in _SYNTHETIC]
    samples = [s for s in samples if s is not None]
    while len(samples) < n:
        samples.extend([_make_sample(q, a, ds, lv, lambda t: t.strip())
                        for q, a, ds, lv in _SYNTHETIC])
    return samples[:n]


def _load_gsm8k(split: str = "train") -> List[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/gsm8k", "main", split=split)
    except Exception:
        # older mirrors may still serve the pre-migration bare name
        ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for ex in ds:
        s = _make_sample(ex["question"], ex["answer"], "gsm8k", "gsm8k",
                         _extract_gt_gsm8k)
        if s is not None:
            out.append(s)
    return out


def _load_math(split: str = "train", levels: Optional[set] = None) -> List[dict]:
    """MATH with level 1-5 labels; boxed-less solutions are skipped.

    Tries the original ``hendrycks/competition_math`` first (gated), then
    falls back to ``EleutherAI/hendrycks_math`` (7 subject configs, same
    underlying data).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("hendrycks/competition_math", split=split)
    except Exception:
        parts = [load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
                 for cfg in _MATH_CONFIGS]
        ds = concatenate_datasets(parts)

    out = []
    wanted = {str(lv) for lv in levels} if levels is not None else None
    for ex in ds:
        lv = _norm_level(ex.get("level"))
        if lv is None or (wanted is not None and lv not in wanted):
            continue
        s = _make_sample(ex["problem"], ex["solution"], "math", lv,
                         _extract_gt_math)
        if s is not None:
            out.append(s)
    return out


def _load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            s = _make_sample(rec["question"], rec["answer"],
                             rec.get("data_source", "local"),
                             rec.get("level", "1"), _extract_gt_auto)
            if s is not None:
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# difficulty-gradient mix
# ---------------------------------------------------------------------------
def _stratified_gsm8k_math(split: str, n: int, seed: int,
                           math_ratio: float) -> List[dict]:
    """Quota-balanced mix: GSM8K easy tier + MATH levels 1-5.

    ``n * (1 - math_ratio)`` GSM8K samples plus ``n * math_ratio`` MATH
    samples split as evenly as possible across levels 1-5 (leftover quota
    goes to easier levels).  Both strata are shuffled with ``seed`` before
    quota selection so the gradient is reproducible.
    """
    import random
    rng = random.Random(seed)

    n_math = min(int(round(n * math_ratio)), n)
    n_gsm = n - n_math

    gsm = _load_gsm8k(split)
    math_all = _load_math(split)

    rng.shuffle(gsm)
    picked = gsm[:n_gsm]

    by_level: dict = {}
    for s in math_all:
        by_level.setdefault(s["level"], []).append(s)
    levels = sorted(by_level)  # '1' .. '5'
    if levels:
        base, extra = divmod(n_math, len(levels))
        for i, lv in enumerate(levels):
            pool = by_level[lv]
            rng.shuffle(pool)
            quota = base + (1 if i < extra else 0)  # extras -> easier levels
            picked.extend(pool[:quota])

    rng.shuffle(picked)
    return picked


def build_dataset(dataset: str, num_samples: int, split: str = "train",
                  seed: int = 42, levels: Optional[set] = None,
                  math_ratio: float = 0.5) -> List[dict]:
    """Build and shuffle a training list, truncated to ``num_samples``."""
    if dataset == "synthetic":
        return build_synthetic(num_samples)
    if dataset == "gsm8k":
        samples = _load_gsm8k(split)
    elif dataset in ("gsm8k_math", "mix"):
        # already shuffled, quota-balanced and sized to num_samples
        return _stratified_gsm8k_math(split, num_samples, seed, math_ratio)
    elif os.path.isfile(dataset):
        samples = _load_jsonl(dataset)
    else:
        raise ValueError(f"Unknown dataset spec: {dataset!r}")

    import random
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:num_samples]
