"""Unified GRPO / DAPO training entry point.

Usage:
    python scripts/train.py --algorithm dapo --max-steps 3 \\
        --base-model Qwen/Qwen2.5-0.5B-Instruct --device mps \\
        --group-size 2 --groups-per-step 1 --max-new-tokens 32

``--algorithm`` picks a preset table (``src.config.PRESETS``); the presets
differ only in the DAPO / entropy toggles (single-variable ablation).

v2 changes vs v1 (all aimed at making every ablation component measurable):
  * zero-variance groups are counted for **every** arm (v1 only counted them
    inside the dynamic-sampling branch, so the vanilla collapse trajectory
    was invisible); groups are only *filtered* in the dynamic arm.
  * true full-vocabulary policy entropy is probed on the first rollout group
    of every step (v1 only had the sampled-token surprisal proxy).
  * KL vs reference is monitored with the non-negative k3 estimator even
    though beta = 0 (v1 logged beta * KL, i.e. a constant zero).
  * per-step metrics are appended to ``metrics.jsonl`` incrementally and
    LoRA checkpoints are saved every ``save_every`` steps, so watchdog-killed
    runs keep their data; ``--resume-from`` restores adapter/optimizer/cursor.
  * stratified greedy eval every ``eval_every`` steps (by data source and
    difficulty level).
"""

import json
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


def sample_groups(agent, dataset, n_prompts, cfg, device, cursor):
    """Sample ``n_prompts`` prompt-groups (group_size completions each).

    Returns ``(groups, results, zero_var, probe)``:

    * ``zero_var`` counts groups whose rewards have zero variance — for
      every arm, since the zero-variance ratio *is* the collapse metric the
      ablation needs to show (all-correct groups count too).
    * Groups are only *dropped* when dynamic sampling is on.
    * ``probe`` holds the first prompt-group's tensors (ids + response spans)
      for the per-step true-entropy measurement, taken from the candidate
      batch before any filtering so every arm sees the same prompts.
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
        enable_overlong=cfg.overlong_shaping, format_weight=cfg.format_weight,
        empty_min_tokens=cfg.empty_min_tokens)

    probe = {
        "full_ids": full_ids[:cfg.group_size],
        "response_spans": response_spans[:cfg.group_size],
    }
    mean_len = float(np.mean(comp_lens)) if comp_lens else 0.0

    groups = []
    zero_var = 0
    for i in range(n_prompts):
        s = i * cfg.group_size
        e = s + cfg.group_size
        group_rewards = rewards[s:e]
        is_zero_var = float(np.std(group_rewards)) < 1e-6
        if is_zero_var:
            zero_var += 1
        if cfg.dynamic_sampling and is_zero_var:
            continue  # filter only in the dynamic-sampling arm
        adv = compute_grpo_advantages(group_rewards)
        groups.append({
            "full_ids": full_ids[s:e].detach().to(device),
            "response_spans": response_spans[s:e],
            "rewards": group_rewards,
            "advantages": adv.to(device),
            "completion_lengths": comp_lens[s:e],
            "truncated": truncated[s:e],
            "zero_adv": is_zero_var,
        })

    return groups, results, zero_var, probe, mean_len


def save_checkpoint(agent, optimizer, cfg, step, cursor, tag) -> str:
    """Save LoRA adapter + optimizer + loop state; never fatal."""
    ckpt_dir = os.path.join(cfg.output_dir, cfg.run_name, f"ckpt_{tag}")
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        adapter_path = os.path.join(ckpt_dir, "adapter")
        agent.policy_model.save_pretrained(adapter_path)
        agent.tokenizer.save_pretrained(adapter_path)
        torch.save(optimizer.state_dict(),
                   os.path.join(ckpt_dir, "optimizer.pt"))
        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump({"step": step, "cursor": cursor,
                       "algorithm": cfg.algorithm}, f)
        print(f"[ckpt] step {step} -> {ckpt_dir}", flush=True)
    except Exception as e:  # checkpointing must never kill the run
        print(f"[ckpt] save failed (non-fatal): {e}", flush=True)
    return ckpt_dir


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
    print(f"resume_from={cfg.resume_from}")
    print("--- FULL CONFIG (verify every value before leaving it running) ---")
    for k, v in vars(cfg).items():
        if not k.startswith("_"):
            print(f"  {k} = {v}")
    print("=" * 70)

    resume_adapter = None
    start_step = 0
    cursor = 0
    if cfg.resume_from:
        meta_path = os.path.join(cfg.resume_from, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        start_step = int(meta["step"])
        cursor = int(meta.get("cursor", 0))
        resume_adapter = os.path.join(cfg.resume_from, "adapter")
        print(f"[resume] from {cfg.resume_from}: step={start_step} "
              f"cursor={cursor}")

    agent = GRPOAgent(cfg.base_model, device=str(device),
                      lora_config=make_lora_config(cfg), use_8bit=cfg.use_8bit,
                      ref_use_lora=cfg.ref_use_lora,
                      resume_adapter=resume_adapter)
    trainable = [p for p in agent.policy_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable LoRA params: {n_trainable:,}")
    optimizer = optim.Adam(trainable, lr=cfg.learning_rate)
    if cfg.resume_from:
        opt_path = os.path.join(cfg.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=str(device)))
            print(f"[resume] optimizer state loaded")

    train_data = build_dataset(cfg.dataset, cfg.train_samples,
                               split="train", seed=cfg.seed,
                               math_ratio=cfg.math_ratio)
    eval_data = build_dataset(cfg.dataset, cfg.eval_samples,
                              split="test", seed=cfg.seed,
                              math_ratio=cfg.math_ratio)
    print(f"Train samples: {len(train_data)}, Eval samples: {len(eval_data)}")

    diag = Diagnostics(
        project=cfg.swanlab_project, experiment_name=cfg.run_name,
        config={k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        enabled=cfg.use_swanlab, logdir=cfg.log_dir)

    # incremental per-step metrics log (append-mode so resumes keep history)
    metrics_dir = os.path.join(cfg.log_dir, cfg.run_name)
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a", encoding="utf-8")

    global_step = start_step

    for step in range(start_step, cfg.max_steps):
        step_start = time.time()
        # ---- collect valid groups (dynamic sampling) ----------------------
        budget = cfg.groups_per_step * cfg.dynamic_max_attempts
        valid_groups = []
        all_results = []
        zero_var_total = 0
        candidate_groups = 0
        probe = None
        gen_len_sum = 0.0
        gen_len_n = 0

        while candidate_groups < budget and len(valid_groups) < cfg.groups_per_step:
            need = cfg.groups_per_step - len(valid_groups)
            groups, results, zero_var, probe_i, mlen = sample_groups(
                agent, train_data, need, cfg, device, cursor)
            if probe is None:
                probe = probe_i
            cursor = (cursor + need) % len(train_data)
            valid_groups.extend(groups)
            all_results.extend(results)
            zero_var_total += zero_var
            candidate_groups += need
            gen_len_sum += mlen * len(results)
            gen_len_n += len(results)
            if not cfg.dynamic_sampling:
                break

        effective_group_ratio = len(valid_groups) / max(candidate_groups, 1)
        oversample_ratio = candidate_groups / max(cfg.groups_per_step, 1)
        zero_var_ratio = zero_var_total / max(candidate_groups, 1)

        # ---- true policy entropy on the probe group (collection policy) ---
        agent.policy_model.eval()
        policy_entropy = agent.policy_entropy(
            probe["full_ids"], probe["response_spans"],
            use_amp=use_amp) if probe is not None else 0.0

        # ---- update over valid groups -------------------------------------
        # zero-advantage groups carry no policy gradient when beta=0 and the
        # entropy bonus is off; skip their forward/backward passes entirely
        # (late in vanilla training ~65% of groups are zero-variance).
        skip_zero_adv = (cfg.beta == 0.0) and (not cfg.entropy_reg)
        p_losses, kl_ests, clip_fracs, surprises = [], [], [], []

        for group in valid_groups:
            if group["full_ids"].numel() == 0:
                continue
            if skip_zero_adv and group["zero_adv"]:
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
                # entropy bonus leaks into ALL arms and the `entropy_reg`
                # toggle stops being a single variable (the whole ablation grid
                # would then be "everyone has entropy reg").
                _ent_coef = cfg.entropy_coef if cfg.entropy_reg else 0.0
                loss, p_loss, _kl_pen, kl_est, clip_frac, entropy = grpo_loss(
                    old_lp, new_lp, ref_lp, group["advantages"],
                    eps_low=cfg.eps_low, eps_high=cfg.eps_high, beta=cfg.beta,
                    token_level=cfg.token_level, ent_coef=_ent_coef)
                amp_backward_step(loss, optimizer, None, agent.policy_model,
                                  cfg.max_grad_norm)

                # per-token surprisal from the *collection* policy
                surprisal = -sum(t.sum().item()
                                 for t in old_lp) / max(sum(t.numel() for t in old_lp), 1)
                p_losses.append(p_loss)
                kl_ests.append(kl_est)
                clip_fracs.append(clip_frac)
                surprises.append(surprisal)

        # ---- diagnostics ---------------------------------------------------
        hack_rates = aggregate_hack_rates(all_results)
        metrics = {
            "loss/policy_mean": float(np.mean(p_losses)) if p_losses else 0.0,
            "rollout/kl_vs_ref": float(np.mean(kl_ests)) if kl_ests else 0.0,
            "rollout/mean_sampled_token_surprisal": float(np.mean(surprises)) if surprises else 0.0,
            "rollout/policy_entropy": float(policy_entropy),
            "rollout/mean_completion_tokens": (gen_len_sum / gen_len_n) if gen_len_n else 0.0,
            "rollout/zero_variance_group_ratio": zero_var_ratio,
            "rollout/effective_group_ratio": effective_group_ratio,
            "rollout/oversample_ratio": oversample_ratio,
            "train/clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            "reward/mean": float(np.mean([r.shaped_reward for r in all_results])) if all_results else 0.0,
            **hack_rates,
        }
        diag.log(global_step, metrics)
        metrics_file.write(json.dumps(
            {"step": global_step, "kind": "train",
             "wall_time": round(time.time() - step_start, 2), **metrics}) + "\n")
        metrics_file.flush()
        print(f"[step {global_step:3d}] " + " | ".join(
            f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()
            if k.split('/')[0] in ("loss", "rollout", "train", "reward")) +
            f" | {time.time() - step_start:.1f}s")
        global_step += 1

        # ---- periodic checkpoint -------------------------------------------
        if cfg.save_every > 0 and global_step % cfg.save_every == 0:
            save_checkpoint(agent, optimizer, cfg, global_step, cursor,
                            tag=f"step{global_step}")

        # ---- mid-run stratified eval ----------------------------------------
        if cfg.eval_every > 0 and global_step % cfg.eval_every == 0 and eval_data:
            eval_res = evaluate_model(
                agent, eval_data, max_new_tokens=cfg.max_new_tokens,
                overlong_cache=cfg.overlong_cache,
                enable_overlong=cfg.overlong_shaping,
                format_weight=cfg.format_weight, pass_k=0,
                empty_min_tokens=cfg.empty_min_tokens, verbose=False)
            flat = {"step": global_step, "kind": "eval",
                    "eval/accuracy": eval_res["accuracy"],
                    "eval/format_rate": eval_res["format_rate"]}
            flat.update({f"eval/src_{k}": v
                         for k, v in eval_res["per_source"].items()})
            flat.update({f"eval/level_{k}": v
                         for k, v in eval_res["per_level"].items()})
            metrics_file.write(json.dumps(flat) + "\n")
            metrics_file.flush()
            diag.log(global_step, {k: v for k, v in flat.items()
                                   if k.startswith("eval/")})
            print(f"[eval step {global_step}] accuracy={eval_res['accuracy']:.4f} "
                  f"format={eval_res['format_rate']:.4f} "
                  f"per_source={eval_res['per_source']} "
                  f"per_level={eval_res['per_level']}", flush=True)

    # ---- final evaluation + checkpoint -----------------------------------
    ckpt_dir = save_checkpoint(agent, optimizer, cfg, global_step, cursor,
                               tag="final")

    if eval_data:
        agent.policy_model.eval()
        eval_res = evaluate_model(
            agent, eval_data, max_new_tokens=cfg.max_new_tokens,
            overlong_cache=cfg.overlong_cache, enable_overlong=cfg.overlong_shaping,
            format_weight=cfg.format_weight, pass_k=cfg.pass_at_k,
            pass_k_samples=cfg.pass_k_samples,
            empty_min_tokens=cfg.empty_min_tokens)
        flat = {"step": global_step, "kind": "eval_final",
                "eval/accuracy": eval_res["accuracy"],
                "eval/format_rate": eval_res["format_rate"]}
        if cfg.pass_at_k > 1:
            flat[f"eval/pass@{cfg.pass_at_k}"] = eval_res[f"pass@{cfg.pass_at_k}"]
            flat.update({f"eval/pass{cfg.pass_at_k}_src_{k}": v
                         for k, v in eval_res[f"pass@{cfg.pass_at_k}_per_source"].items()})
        flat.update({f"eval/src_{k}": v for k, v in eval_res["per_source"].items()})
        flat.update({f"eval/level_{k}": v for k, v in eval_res["per_level"].items()})
        metrics_file.write(json.dumps(flat) + "\n")
        metrics_file.flush()
        pass_str = (f" pass@{cfg.pass_at_k}={eval_res[f'pass@{cfg.pass_at_k}']:.4f}"
                    if cfg.pass_at_k > 1 else "")
        print(f"[eval] accuracy={eval_res['accuracy']:.4f} "
              f"format_rate={eval_res['format_rate']:.4f}{pass_str} "
              f"per_source={eval_res['per_source']} per_level={eval_res['per_level']}")
        with open(os.path.join(ckpt_dir, "eval_result.json"), "w") as f:
            json.dump(eval_res, f, indent=2)

    metrics_file.close()
    diag.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
