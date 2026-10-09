"""Unified GRPO / DAPO training entry point.

Usage:
    python scripts/train.py --algorithm dapo --max-steps 3 \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct --device mps \\
        --group-size 2 --groups-per-step 1 --max-new-tokens 32

``--algorithm`` picks a preset table (``src.config.PRESETS``); the seven
presets differ only in the DAPO / entropy toggles (single-variable ablation).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim

from src.config import load_config
from src.train_utils import get_device, set_seed, amp_backward_step
from src.data import build_dataset
from src.rewards import compute_rewards
from src.losses import compute_grpo_advantages, grpo_loss
from src.agents.grpo_agent import GRPOAgent
from src.diagnostics import Diagnostics, aggregate_hack_rates
from src.evaluation import evaluate_model


def make_lora_config(cfg):
    return {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": list(cfg.lora_target_modules),
        "bias": "none",
    }


def sample_groups(agent, dataset, n_prompts, cfg, device, cursor, use_amp):
    """Sample ``n_prompts`` prompt-groups (group_size completions each).

    Returns ``(groups, results, zero_variance_count)`` where ``groups`` holds
    only the *valid* (non-zero reward variance) groups when dynamic sampling is
    off; every group is returned otherwise.
    """
    idxs = [(cursor + i) % len(dataset) for i in range(n_prompts)]
    prompts = [dataset[i]["prompt"] for i in idxs]
    gts = [dataset[i]["ground_truth"] for i in idxs]

    # flatten: each prompt duplicated group_size times, batched generation
    flat_prompts = [p for p in prompts for _ in range(cfg.group_size)]
    flat_gts = [g for g in gts for _ in range(cfg.group_size)]

    agent.policy_model.eval()
    responses, full_ids, response_spans, comp_lens, truncated = \
        agent.generate_responses(
            flat_prompts, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, max_prompt_tokens=cfg.max_prompt_tokens)
    rewards, results = compute_rewards(
        responses, flat_gts, comp_lens, truncated,
        max_tokens=cfg.max_new_tokens, overlong_cache=cfg.overlong_cache,
        enable_overlong=cfg.overlong_shaping, format_weight=cfg.format_weight)

    groups = []
    zero_var = 0
    for i in range(n_prompts):
        s = i * cfg.group_size
        e = s + cfg.group_size
        group_rewards = rewards[s:e]
        adv = compute_grpo_advantages(group_rewards)
        group = {
            "full_ids": full_ids[s:e].detach().to(device),
            "response_spans": response_spans[s:e],
            "rewards": group_rewards,
            "advantages": adv.to(device),
            "completion_lengths": comp_lens[s:e],
            "truncated": truncated[s:e],
        }
        valid = not (cfg.dynamic_sampling and float(adv.std()) < 1e-6)
        if valid:
            groups.append(group)
        else:
            zero_var += 1

    return groups, results, zero_var


def train():
    cfg = load_config()
    device = get_device(cfg.device)
    set_seed(cfg.seed)
    use_amp = (device.type == "cuda")

    print("=" * 70)
    print(f"Algorithm: {cfg.algorithm}  ({cfg.base_model})")
    print(f"Device: {device}   Toggles: clip_higher={cfg.clip_higher} "
          f"dynamic={cfg.dynamic_sampling} token_level={cfg.token_level} "
          f"overlong={cfg.overlong_shaping} entropy_reg={cfg.entropy_reg}")
    print(f"eps=[{cfg.eps_low}, {cfg.eps_high}] beta={cfg.beta} "
          f"ent_coef={cfg.entropy_coef} group_size={cfg.group_size}")
    print(f"steps={cfg.max_steps} ppo_epochs={cfg.ppo_epochs} "
          f"lr={cfg.learning_rate} pass_at_k={cfg.pass_at_k}")
    print("--- FULL CONFIG (verify every value before leaving it running) ---")
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            print(f"  {k} = {v}")
    print("=" * 70)

    agent = GRPOAgent(cfg.base_model, device=str(device),
                      lora_config=make_lora_config(cfg), use_8bit=cfg.use_8bit,
                      ref_use_lora=cfg.ref_use_lora)
    trainable = [p for p in agent.policy_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable LoRA params: {n_trainable:,}")
    optimizer = optim.Adam(trainable, lr=cfg.learning_rate)

    train_data = build_dataset(cfg.dataset, cfg.train_samples,
                               split="train", seed=cfg.seed)
    eval_data = build_dataset(cfg.dataset, cfg.eval_samples,
                              split="test", seed=cfg.seed)
    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    diag = Diagnostics(
        project=cfg.swanlab_project, experiment_name=cfg.run_name,
        config={k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        enabled=cfg.use_swanlab, logdir=cfg.log_dir)

    cursor = 0
    global_step = 0

    for step in range(cfg.max_steps):
        step_start = time.time()
        # ---- collect valid groups (dynamic sampling) ----------------------
        budget = cfg.groups_per_step * cfg.dynamic_max_attempts
        valid_groups = []
        all_results = []
        zero_var_total = 0
        candidate_groups = 0

        while candidate_groups < budget and len(valid_groups) < cfg.groups_per_step:
            need = cfg.groups_per_step - len(valid_groups)
            groups, results, zero_var = sample_groups(
                agent, train_data, need, cfg, device, cursor, use_amp)
            cursor = (cursor + need) % len(train_data)
            valid_groups.extend(groups)
            all_results.extend(results)
            zero_var_total += zero_var
            candidate_groups += need
            if not cfg.dynamic_sampling:
                break

        effective_group_ratio = len(valid_groups) / max(candidate_groups, 1)
        oversample_ratio = candidate_groups / max(cfg.groups_per_step, 1)
        zero_var_ratio = zero_var_total / max(candidate_groups, 1)

        # ---- update over valid groups -------------------------------------
        p_losses, kl_losses, clip_fracs, surprises, comp_lens = [], [], [], [], []

        for group in valid_groups:
            if group["full_ids"].numel() == 0:
                continue
            agent.policy_model.eval()
            with torch.no_grad():
                old_lp = agent.get_policy_log_probs(
                    group["full_ids"], group["response_spans"], use_amp=use_amp)
                ref_lp = agent.get_reference_log_probs(
                    group["full_ids"], group["response_spans"], use_amp=use_amp)
            old_lp = [t.detach() for t in old_lp]

            agent.policy_model.train()
            for _ in range(cfg.ppo_epochs):
                new_lp = agent.get_policy_log_probs(
                    group["full_ids"], group["response_spans"], use_amp=use_amp)
                # NOTE: ent_coef must be gated on cfg.entropy_reg, otherwise the
                # entropy bonus leaks into ALL seven arms and the `entropy_reg`
                # toggle stops being a single variable (the whole ablation grid
                # would then be "everyone has entropy reg").
                _ent_coef = cfg.entropy_coef if cfg.entropy_reg else 0.0
                loss, p_loss, kl_pen, clip_frac, entropy = grpo_loss(
                    old_lp, new_lp, ref_lp, group["advantages"],
                    eps_low=cfg.eps_low, eps_high=cfg.eps_high, beta=cfg.beta,
                    token_level=cfg.token_level, ent_coef=_ent_coef)
                amp_backward_step(loss, optimizer, None, agent.policy_model,
                                  cfg.max_grad_norm)

                # per-token surprisal from the *collection* policy
                surprisal = -sum(t.sum().item()
                                 for t in old_lp) / max(sum(t.numel() for t in old_lp), 1)
                p_losses.append(p_loss)
                kl_losses.append(kl_pen)
                clip_fracs.append(clip_frac)
                surprises.append(surprisal)
                comp_lens.append(np.mean(group["completion_lengths"]))

        # ---- diagnostics ---------------------------------------------------
        hack_rates = aggregate_hack_rates(all_results)
        metrics = {
            "loss/policy_mean": float(np.mean(p_losses)) if p_losses else 0.0,
            "loss/kl_penalty": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "rollout/mean_sampled_token_surprisal": float(np.mean(surprises)) if surprises else 0.0,
            "rollout/mean_completion_tokens": float(np.mean(comp_lens)) if comp_lens else 0.0,
            "rollout/zero_variance_group_ratio": zero_var_ratio,
            "rollout/effective_group_ratio": effective_group_ratio,
            "rollout/oversample_ratio": oversample_ratio,
            "train/clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "reward/mean": float(np.mean([r.shaped_reward for r in all_results])) if all_results else 0.0,
            **hack_rates,
        }
        diag.log(global_step, metrics)
        print(f"[step {global_step:3d}] " + " | ".join(
            f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()
            if k.split('/')[0] in ("loss", "rollout", "train", "reward")) +
            f" | {time.time() - step_start:.1f}s")
        global_step += 1

    # ---- final evaluation + checkpoint -----------------------------------
    ckpt_dir = os.path.join(cfg.output_dir, f"ckpt_{cfg.run_name}")
    os.makedirs(ckpt_dir, exist_ok=True)
    adapter_path = os.path.join(ckpt_dir, "adapter")
    try:
        agent.policy_model.save_pretrained(adapter_path)
        agent.tokenizer.save_pretrained(adapter_path)
        print(f"[ckpt] LoRA adapter saved -> {adapter_path}", flush=True)
    except Exception as e:  # checkpointing must never kill the run
        print(f"[ckpt] save failed (non-fatal): {e}", flush=True)

    if eval_data:
        agent.policy_model.eval()
        eval_res = evaluate_model(
            agent, eval_data, max_new_tokens=cfg.max_new_tokens,
            overlong_cache=cfg.overlong_cache, enable_overlong=cfg.overlong_shaping,
            format_weight=cfg.format_weight, pass_k=cfg.pass_at_k,
            pass_k_samples=cfg.pass_k_samples)
        pass_str = (f" pass@{cfg.pass_at_k}={eval_res[f'pass@{cfg.pass_at_k}']:.4f}"
                    if cfg.pass_at_k > 1 else "")
        print(f"[eval] accuracy={eval_res['accuracy']:.4f} "
              f"format_rate={eval_res['format_rate']:.4f}{pass_str} "
              f"per_source={eval_res['per_source']}")
        with open(os.path.join(ckpt_dir, "eval_result.json"), "w") as f:
            import json
            json.dump(eval_res, f, indent=2)

    diag.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
