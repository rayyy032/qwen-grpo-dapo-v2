"""Loss functions: GRPO/DAPO clipped policy loss, advantages, entropy, KL.

DAPO modifications implemented here:
  * Clip-Higher : asymmetric ratio clip (upper bound ``1 + eps_high`` only for
    positive advantage, lower bound ``1 - eps_low`` for negative advantage).
  * Token-level loss : global mean over every completion token instead of
    sample-mean (``1/|o|`` dilution removed).
  * Entropy regularization : ``-ent_coef * H`` with ``H`` approximated by the
    mean sampled-token surprisal ``mean(-log p)`` (documented proxy).
"""

import torch
from torch import Tensor
from typing import List, Tuple


def compute_grpo_advantages(rewards_group, std_eps: float = 1e-8) -> Tensor:
    """GRPO group-relative advantages: ``A_i = (r_i - mean)/ (std + eps)``.

    Args:
        rewards_group: scalar rewards for one prompt group, shape (G,).
    Returns:
        Tensor of advantages, shape (G,).
    """
    rewards = torch.as_tensor(rewards_group, dtype=torch.float32)
    std = rewards.std()
    if std < std_eps:
        # Zero-variance group: no gradient signal, return zeros (group is
        # dropped before training when dynamic sampling is enabled).
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + std_eps)


def compute_surprisal(new_log_probs: List[Tensor]) -> float:
    """Mean sampled-token surprisal ``mean(-log p)`` as entropy proxy."""
    tensors = [t for t in new_log_probs if t.numel() > 0]
    if not tensors:
        return 0.0
    total = sum(int(t.numel()) for t in tensors)
    return float(-sum(t.sum().item() for t in tensors) / total)


def grpo_loss(
    old_log_probs: List[Tensor],
    new_log_probs: List[Tensor],
    ref_log_probs: List[Tensor],
    advantages: Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    beta: float = 0.0,
    token_level: bool = False,
    ent_coef: float = 0.0,
) -> Tuple[Tensor, float, float, float, float]:
    """GRPO / DAPO clipped surrogate objective.

    Args:
        old_log_probs: per-token log-probs under the collection policy (detached).
        new_log_probs: per-token log-probs under the current policy.
        ref_log_probs: per-token log-probs under the frozen reference.
        advantages: group-scoped advantage per sample, shape (G,).
        eps_low: lower clip bound factor -> ``1 - eps_low``.
        eps_high: upper clip bound factor -> ``1 + eps_high`` (clip-higher).
        beta: KL penalty coefficient (KL(ref || policy)).
        token_level: global token-mean reduction instead of sample-mean.
        ent_coef: entropy regularization strength (surprisal proxy).

    Returns:
        loss, policy_loss (float), kl_penalty (float), clip_fraction (float),
        entropy (float).
    """
    n = len(new_log_probs)
    if n == 0:
        return (torch.tensor(0.0), 0.0, 0.0, 0.0, 0.0)

    clip_low = 1.0 - eps_low
    clip_high = 1.0 + eps_high

    ratio_list = []
    surr1_list = []
    surr2_list = []
    kl_list = []

    for i in range(n):
        old = old_log_probs[i]
        new = new_log_probs[i]
        ref = ref_log_probs[i]
        if new.numel() == 0:
            continue
        adv = advantages[i].detach().to(dtype=new.dtype, device=new.device)

        ratio = torch.exp(new - old)
        surr1 = ratio * adv
        # Asymmetric clip: same clamp for both signs, but the bounds are
        # asymmetric in eps_high/eps_low. ``min(surr1, surr2)`` then correctly
        # caps positive-advantage gains at 1+eps_high and negative-advantage
        # gains at 1-eps_low.
        surr2 = torch.clamp(ratio, clip_low, clip_high) * adv
        ratio_list.append(ratio)
        surr1_list.append(surr1)
        surr2_list.append(surr2)
        kl_list.append(ref - new)

    if not surr1_list:
        return (torch.tensor(0.0), 0.0, 0.0, 0.0, 0.0)

    surr1_cat = torch.cat(surr1_list)
    surr2_cat = torch.cat(surr2_list)
    ratio_cat = torch.cat(ratio_list)
    kl_cat = torch.cat(kl_list)

    # min(clipped, unclipped) per token
    per_token_policy = -torch.minimum(surr1_cat, surr2_cat)

    if token_level:
        policy_loss = per_token_policy.mean()
    else:
        # sample-mean reduction: mean over tokens first, then over samples
        sample_means = [(-torch.minimum(s1, s2)).mean()
                        for s1, s2 in zip(surr1_list, surr2_list)]
        policy_loss = torch.stack(sample_means).mean()

    kl_penalty = beta * kl_cat.mean()

    # Entropy proxy = mean sampled-token surprisal ``mean(-log p)`` over every
    # completion token. Built from the in-graph tensor so the gradient flows
    # (the old version used compute_surprisal's detached float -> no-op).
    logp_cat = torch.cat([t for t in new_log_probs if t.numel() > 0])
    surprisal_t = -logp_cat.mean()
    entropy = float(surprisal_t.detach().item())
    entropy_term = -ent_coef * surprisal_t \
        if ent_coef > 0.0 else torch.tensor(0.0, device=policy_loss.device)

    loss = policy_loss + kl_penalty + entropy_term

    with torch.no_grad():
        clip_fraction = float((surr2_cat < surr1_cat).float().mean())

    return (loss, float(policy_loss.detach().item()),
            float(kl_penalty.detach().item()), clip_fraction, entropy)
