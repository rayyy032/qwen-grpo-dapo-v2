"""Rule-based verifiable reward for math tasks + reward-hacking diagnostics.

Correctness uses **numeric equivalence** via ``math-verify`` (handles ``1.0 == 1
== 01``, fractions, radicals) with exact string equality as a fallback.  The
DAPO **Overlong Reward Shaping** piecewise-linear penalty is applied to
truncated responses.  Reward-hacking detectors (length exploit / format-only /
equivalence bypass) are computed as diagnostics, never added to the reward.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

try:
    from math_verify import parse, verify, LatexExtractionConfig
    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - math-verify optional at import time
    _HAS_MATH_VERIFY = False


# ---------------------------------------------------------------------------
# \boxed{} extraction
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_boxed_content(text: str, open_brace_idx: int) -> Optional[str]:
    """Return inner content of ``\\boxed{...}`` handling nested braces.

    ``open_brace_idx`` is the index of the opening ``{``.
    """
    depth = 0
    for i in range(open_brace_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1:i]
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the content of the *last* well-formed ``\\boxed{...}``."""
    matches = list(_BOXED_RE.finditer(text))
    # fall back from the last occurrence until a well-formed brace is found
    for m in reversed(matches):
        content = _extract_boxed_content(text, m.end() - 1)
        if content is not None:
            return content.strip()
    return None


# ---------------------------------------------------------------------------
# numeric equivalence
# ---------------------------------------------------------------------------
def _clean_numeric(text: str):
    """Parse a pure-numeric answer to float, else None.

    Handles scientific notation (``1e3``), thousands separators (``1,000``),
    leading zeros (``01``) and surrounding math delimiters (``$``), which
    ``math-verify`` sometimes fails to parse.
    """
    t = text.strip().strip("$").replace(",", "").replace(" ", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def answers_equivalent(prediction: str, reference: str) -> bool:
    """True iff ``prediction`` and ``reference`` are numerically equivalent.

    Order: exact string match, then pure-numeric float comparison (scientific
    notation / separators / leading zeros), then math-verify (fractions,
    radicals, symbolic expressions).
    """
    if prediction is None or reference is None:
        return False
    p = prediction.strip()
    r = reference.strip()
    if p == r:
        return True
    # pure-numeric fallback before math-verify: cheaper and covers 1e3, 1,000, 01
    pf = _clean_numeric(p)
    rf = _clean_numeric(r)
    if pf is not None and rf is not None:
        return math.isclose(pf, rf, rel_tol=1e-9, abs_tol=1e-9)
    if not _HAS_MATH_VERIFY:
        return False
    try:
        parsed_ref = parse(r, extraction_mode="first_match")
        parsed_pred = parse(
            p, extraction_config=[LatexExtractionConfig()])
        # also try plain-text extraction for bare integer/float answers
        if parsed_pred == [] or parsed_pred is None:
            parsed_pred = parse(p, extraction_mode="first_match")
        if parsed_ref == [] or parsed_ref is None:
            return False
        return bool(verify(parsed_ref, parsed_pred))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DAPO overlong reward shaping
# ---------------------------------------------------------------------------
def soft_overlong_penalty(completion_tokens: int, max_tokens: int,
                          overlong_cache: int) -> float:
    """DAPO piecewise-linear length penalty, value in ``[-1, 0]``.

    .. math::
        p(|y|) = \\begin{cases}
            0 & |y| \\le L_{max} - L_{cache} \\\\
            -(|y| - (L_{max} - L_{cache}))/L_{cache} & \\text{mid} \\\\
            -1 & |y| > L_{max}
        \\end{cases}
    """
    if max_tokens <= 0 or overlong_cache <= 0:
        raise ValueError("max_tokens and overlong_cache must be positive")
    if overlong_cache >= max_tokens:
        raise ValueError("overlong_cache must be strictly smaller than max_tokens")
    if completion_tokens < 0:
        raise ValueError("completion_tokens must be non-negative")

    cache_start = max_tokens - overlong_cache
    if completion_tokens <= cache_start:
        return 0.0
    if completion_tokens <= max_tokens:
        return -(completion_tokens - cache_start) / overlong_cache
    return -1.0


# ---------------------------------------------------------------------------
# reward-hacking diagnostics (metrics only, not added to the reward)
# ---------------------------------------------------------------------------
@dataclass
class HackFlags:
    length_exploit: bool
    format_only_hack: bool
    equivalence_bypass: bool


def detect_reward_hacking(completion: str, ground_truth: str, correct: bool,
                          num_tokens: int, max_new_tokens: int,
                          min_reasoning_tokens: int = 0) -> HackFlags:
    """Flag reward-hacking patterns.

    * ``length_exploit``: correct but abnormally long (>= 0.9 * max_new_tokens).
    * ``format_only_hack``: correct *and* almost no reasoning before ``\boxed``.
    * ``equivalence_bypass``: exact string differs from truth but math-verify
      deems equivalent (a *legitimate* numeric rewrite, flagged to monitor
      whether the policy exploits loose parsing).
    """
    boxed = extract_last_boxed(completion)
    exact = (boxed is not None) and (boxed.strip() == ground_truth.strip())
    length_exploit = correct and num_tokens >= int(0.9 * max_new_tokens)
    format_only_hack = correct and (boxed is not None) and \
        (num_tokens <= min_reasoning_tokens or
         (len(_chars_before_box(completion)) < 10 and num_tokens <= 8))
    equivalence_bypass = correct and (not exact)
    return HackFlags(length_exploit, format_only_hack, equivalence_bypass)


def _chars_before_box(text: str) -> str:
    m = _BOXED_RE.search(text)
    return text[:m.start()] if m else text


# ---------------------------------------------------------------------------
# reward assembly
# ---------------------------------------------------------------------------
@dataclass
class ScoreResult:
    correct: bool
    base_reward: float          # 1.0 / 0.0 correctness
    length_penalty: float       # overlong shaping, 0 if disabled
    shaped_reward: float        # correctness + length_penalty (+ format)
    valid_format: bool
    hack: HackFlags


def score_answer(completion: str, ground_truth: str, completion_tokens: int,
                 max_tokens: int, overlong_cache: int, enable_overlong: bool,
                 is_truncated: bool, format_weight: float = 0.2) -> ScoreResult:
    """Score a single completion.

    Overlong Reward Shaping (DAPO): the piecewise-linear ``length_penalty``
    (``soft_overlong_penalty``) is applied as a smooth function of completion
    length.  Non-truncated responses shorter than the cache window are not
    penalized; responses inside the cache window receive a partial penalty;
    truncated responses (``is_truncated``, i.e. hit ``max_new_tokens``)
    saturate at ``-1``.  The truncation flag is passed in and used both to
    guarantee saturation and for the truncation-rate diagnostic.
    """
    boxed = extract_last_boxed(completion)
    valid_format = boxed is not None
    correct = valid_format and answers_equivalent(boxed, ground_truth.strip())

    if enable_overlong:
        if is_truncated:
            length_penalty = -1.0
        else:
            length_penalty = soft_overlong_penalty(completion_tokens, max_tokens,
                                                   overlong_cache)
    else:
        length_penalty = 0.0

    base_reward = 1.0 if correct else 0.0
    shaped_reward = base_reward + length_penalty + (format_weight if valid_format else 0.0)

    hack = detect_reward_hacking(completion, ground_truth, correct,
                                 num_tokens=completion_tokens,
                                 max_new_tokens=max_tokens)
    return ScoreResult(correct, base_reward, length_penalty, shaped_reward,
                       valid_format, hack)


def compute_rewards(completions: Sequence[str], ground_truths: Sequence[str],
                    completion_tokens: Sequence[int], truncated: Sequence[bool],
                    max_tokens: int, overlong_cache: int,
                    enable_overlong: bool, format_weight: float = 0.2
                    ) -> Tuple[List[float], List[ScoreResult]]:
    """Batch reward computation returning ``(combined, per_sample_results)``."""
    combined = []
    results = []
    for c, gt, nt, tr in zip(completions, ground_truths, completion_tokens, truncated):
        r = score_answer(c, gt, nt, max_tokens, overlong_cache, enable_overlong,
                         tr, format_weight)
        combined.append(r.shaped_reward)
        results.append(r)
    return combined, results
