"""Rules for \\boxed extraction, math-verify equivalence, overlong shaping.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards import (  # noqa: E402
    _HAS_MATH_VERIFY,
    answers_equivalent,
    extract_last_boxed,
    score_answer,
    soft_overlong_penalty,
)


# ---------------------------------------------------------------------------
# extract_last_boxed
# ---------------------------------------------------------------------------
def test_boxed_simple_integer():
    assert extract_last_boxed("blah \\boxed{42} done") == "42"


def test_boxed_nested_braces():
    assert extract_last_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_boxed_nested_deep():
    assert extract_last_boxed("\\boxed{\\sqrt{\\frac{3}{4}}}") == "\\sqrt{\\frac{3}{4}}"


def test_boxed_takes_last_of_multiple():
    assert extract_last_boxed("\\boxed{1} middle \\boxed{2}") == "2"


def test_boxed_missing_returns_none():
    assert extract_last_boxed("no answer here") is None


def test_boxed_unterminated_brace_ignored():
    assert extract_last_boxed("\\boxed{7} then \\boxed{broken") == "7"


# ---------------------------------------------------------------------------
# answers_equivalent
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAS_MATH_VERIFY, reason="math-verify not installed")
@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "42"),
        ("42.0", "42"),
        ("1,000", "1000"),
        ("0.5", "\\frac{1}{2}"),
        ("\\frac{2}{4}", "\\frac{1}{2}"),
        ("-3", "-3"),
        ("1e3", "1000"),
    ],
)
def test_answers_equivalent(prediction, reference):
    assert answers_equivalent(prediction, reference)


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("42", "43"),
        ("0.5", "\\frac{1}{3}"),
        ("-3", "3"),
    ],
)
def test_answers_not_equivalent(prediction, reference):
    assert not answers_equivalent(prediction, reference)


def test_answers_equivalent_garbage_is_safe():
    assert answers_equivalent("", "42") is False


# ---------------------------------------------------------------------------
# soft_overlong_penalty (piecewise boundaries)
# ---------------------------------------------------------------------------
def test_overlong_zero_before_cache():
    assert soft_overlong_penalty(500, max_tokens=1024, overlong_cache=256) == 0.0


def test_overlong_linear_inside_cache():
    value = soft_overlong_penalty(768 + 128, max_tokens=1024, overlong_cache=256)
    assert math.isclose(value, -0.5, abs_tol=1e-9)


def test_overlong_saturated_at_max_tokens():
    assert soft_overlong_penalty(1024, max_tokens=1024, overlong_cache=256) == -1.0


def test_overlong_beyond_max_tokens_stays_capped():
    assert soft_overlong_penalty(9999, max_tokens=1024, overlong_cache=256) == -1.0


def test_overlong_invalid_config_raises():
    with pytest.raises(ValueError):
        soft_overlong_penalty(10, max_tokens=100, overlong_cache=200)
    with pytest.raises(ValueError):
        soft_overlong_penalty(-1, max_tokens=100, overlong_cache=10)


# ---------------------------------------------------------------------------
# score_answer end-to-end
# ---------------------------------------------------------------------------
def test_score_answer_correct_without_overlong():
    result = score_answer("reasoning... \\boxed{72}", "72", completion_tokens=200,
                          max_tokens=1024, overlong_cache=256,
                          enable_overlong=False, is_truncated=False, format_weight=0.2)
    assert result.correct is True
    assert result.base_reward == 1.0
    assert result.length_penalty == 0.0
    assert result.shaped_reward == pytest.approx(1.2)


def test_score_answer_wrong_answer_zero_base():
    result = score_answer("reasoning... \\boxed{71}", "72", completion_tokens=200,
                          max_tokens=1024, overlong_cache=256,
                          enable_overlong=False, is_truncated=False, format_weight=0.2)
    assert result.correct is False
    assert result.base_reward == 0.0
    assert result.shaped_reward == pytest.approx(0.2)  # format bonus only


def test_score_answer_truncated_saturates_minus_one():
    result = score_answer("\\boxed{72}", "72", completion_tokens=1024,
                          max_tokens=1024, overlong_cache=256,
                          enable_overlong=True, is_truncated=True, format_weight=0.2)
    assert result.correct is True
    assert result.length_penalty == -1.0
    assert result.shaped_reward == pytest.approx(1.0 - 1.0 + 0.2)


def test_score_answer_non_truncated_inside_cache_partial():
    result = score_answer("\\boxed{72}", "72", completion_tokens=819,
                          max_tokens=1024, overlong_cache=256,
                          enable_overlong=True, is_truncated=False, format_weight=0.2)
    assert math.isclose(result.length_penalty, -51 / 256, abs_tol=1e-9)


def test_score_answer_missing_boxed_counts_wrong():
    result = score_answer("the answer is obviously 42", "42", completion_tokens=100,
                          max_tokens=1024, overlong_cache=256,
                          enable_overlong=False, is_truncated=False, format_weight=0.2)
    assert result.valid_format is False
    assert result.correct is False
