"""Unit tests for v2 additions: k3 KL estimator, true entropy probe, data.

Run: ``python -m pytest tests/ -q``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import grpo_loss  # noqa: E402
from src.generation import (  # noqa: E402
    compute_entropy_for_tokens,
    compute_log_probs_for_tokens,
)
from src.data import (  # noqa: E402
    _extract_gt_auto,
    _extract_gt_gsm8k,
    _extract_gt_math,
    _norm_level,
    build_synthetic,
    _make_sample,
)


# ---------------------------------------------------------------------------
# k3 KL estimate inside grpo_loss
# ---------------------------------------------------------------------------
def test_kl_estimate_zero_when_policies_match():
    lp = [torch.tensor([0.1, -0.2, 0.3])]
    _loss, _p, _kl_pen, kl_est, _clip, _ent = grpo_loss(
        lp, lp, lp, torch.tensor([1.0]), beta=0.0)
    assert kl_est == pytest.approx(0.0, abs=1e-6)


def test_kl_estimate_positive_when_policy_drifts():
    new = [torch.tensor([0.5, -0.1])]
    ref = [torch.tensor([0.0, 0.0])]
    _loss, _p, _kl_pen, kl_est, _clip, _ent = grpo_loss(
        new, new, ref, torch.tensor([1.0]), beta=0.0)
    assert kl_est > 0.0


def test_kl_penalty_applies_beta():
    new = [torch.tensor([0.5, -0.1])]
    ref = [torch.tensor([0.0, 0.0])]
    _loss, _p, kl_pen, kl_est, _clip, _ent = grpo_loss(
        new, new, ref, torch.tensor([1.0]), beta=0.1)
    assert kl_pen == pytest.approx(0.1 * kl_est, abs=1e-6)


def test_grpo_loss_returns_six_values():
    lp = [torch.tensor([0.1])]
    out = grpo_loss(lp, lp, lp, torch.tensor([1.0]))
    assert len(out) == 6


# ---------------------------------------------------------------------------
# true full-vocab entropy
# ---------------------------------------------------------------------------
class _StubOutput:
    def __init__(self, logits):
        self.logits = logits


class _UniformStub(torch.nn.Module):
    """Every position: uniform logits over `vocab` -> H = ln(vocab)."""

    def __init__(self, vocab=4):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return _StubOutput(logits)


class _PeakedStub(torch.nn.Module):
    """Every position: all mass on token 0 -> H ~ 0."""

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, 4)
        logits[..., 0] = 20.0
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return _StubOutput(logits)


def test_entropy_uniform_equals_log_vocab():
    full_ids = torch.tensor([[1, 5, 6, 7]])
    spans = [(1, 3)]  # tokens at positions 1..2, predicted from 0..1
    ent = compute_entropy_for_tokens(_UniformStub(vocab=4), full_ids, spans,
                                     device="cpu")
    assert ent == pytest.approx(math.log(4), abs=1e-5)


def test_entropy_peaked_near_zero():
    full_ids = torch.tensor([[1, 5, 6, 7]])
    spans = [(1, 3)]
    ent = compute_entropy_for_tokens(_PeakedStub(), full_ids, spans,
                                     device="cpu")
    assert ent < 1e-5


def test_entropy_ignores_non_response_positions():
    full_ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    spans = [(1, 2)]  # single response token only
    ent = compute_entropy_for_tokens(_UniformStub(vocab=8), full_ids, spans,
                                     device="cpu")
    assert ent == pytest.approx(math.log(8), abs=1e-5)


# ---------------------------------------------------------------------------
# logits_to_keep trailing-slice (MPS OOM fix): log probs must be identical to
# a full-sequence computation even with left padding + variable response ends
# ---------------------------------------------------------------------------
class _PositionalStub(torch.nn.Module):
    """Position-dependent logits so a wrong slice offset changes the answer."""

    def __init__(self, vocab=6):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0):
        b, t = input_ids.shape
        pos = torch.arange(t).view(1, t, 1).float()
        v = torch.arange(self.vocab).view(1, 1, self.vocab).float()
        logits = 0.01 * pos + 0.001 * v  # deterministic in (t, v)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return _StubOutput(logits.expand(b, -1, -1).clone())


def _manual_log_probs(full_ids, spans, vocab=6):
    """Full-sequence reference computation (no slicing)."""
    stub = _PositionalStub(vocab)
    out = []
    logits = stub(full_ids).logits
    logp = torch.log_softmax(logits, dim=-1)
    for i, (start, end) in enumerate(spans):
        if end <= start or start < 1:
            out.append(torch.tensor([]))
            continue
        toks = full_ids[i, start:end]
        out.append(logp[i, start - 1:end - 1, :].gather(
            1, toks.unsqueeze(1)).squeeze(1))
    return out


def test_log_probs_trailing_slice_matches_full():
    torch.manual_seed(0)
    # left-padded rows: pads, prompt (uniform start=6), variable-length responses
    full_ids = torch.tensor([
        [0, 0, 1, 2, 3, 4, 5, 2, 3, 4],   # response (6, 10): full-length
        [0, 0, 1, 2, 3, 4, 5, 1, 2, 0],   # response (6, 9): one trailing pad
        [0, 0, 1, 2, 3, 4, 5, 3, 0, 0],   # response (6, 8): early stop
    ])
    spans = [(6, 10), (6, 9), (6, 8)]
    got = compute_log_probs_for_tokens(
        _PositionalStub(), full_ids, spans, device="cpu", row_chunk=2)
    want = _manual_log_probs(full_ids, spans)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape
        assert torch.allclose(g, w, atol=1e-6)


def test_log_probs_empty_span_safe():
    full_ids = torch.tensor([[1, 2, 3]])
    got = compute_log_probs_for_tokens(
        _PositionalStub(), full_ids, [(2, 2)], device="cpu")
    assert got[0].numel() == 0


# ---------------------------------------------------------------------------
# ground-truth extraction (data.py v2 fix)
# ---------------------------------------------------------------------------
def test_extract_gt_gsm8k_marker():
    ans = "Natalia sold clips. #### 72"
    assert _extract_gt_gsm8k(ans) == "72"


def test_extract_gt_gsm8k_missing_marker():
    assert _extract_gt_gsm8k("no marker here") is None


def test_extract_gt_math_boxed():
    sol = r"We compute 2+2. \boxed{4} and continue."
    assert _extract_gt_math(sol) == "4"


def test_extract_gt_math_missing_boxed_is_none():
    assert _extract_gt_math("the answer is 4 without box") is None


def test_make_sample_skips_unextractable_answers():
    assert _make_sample("q", "no marker", "gsm8k", "gsm8k",
                       _extract_gt_gsm8k) is None


def test_make_sample_builds_labeled_sample():
    s = _make_sample("q?", "#### 8", "gsm8k", "gsm8k", _extract_gt_gsm8k)
    assert s["ground_truth"] == "8"
    assert s["data_source"] == "gsm8k"
    assert s["level"] == "gsm8k"
    assert s["prompt"][0]["role"] == "system"


def test_extract_gt_auto_prefers_gsm8k_marker():
    # real GSM8K shape: reasoning lines, final line "#### <number>"
    assert _extract_gt_auto("reasoning\n#### 12") == "12"


def test_extract_gt_auto_falls_back_to_boxed():
    assert _extract_gt_auto(r"answer \boxed{99}") == "99"


def test_norm_level_variants():
    assert _norm_level("Level 3") == "3"
    assert _norm_level("level 5") == "5"
    assert _norm_level(2) == "2"
    assert _norm_level("unknown") is None
    assert _norm_level("Level 6") is None


def test_build_synthetic_repeats_and_sizes():
    assert len(build_synthetic(12)) == 12
    assert all(s["ground_truth"] for s in build_synthetic(3))
