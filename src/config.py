"""Configuration: dataclass + algorithm presets + argparse merge.

The whole project is driven by a single :class:`Config` dataclass.  The
:data:`PRESETS` table maps an ``--algorithm`` name to the four DAPO toggles
plus the entropy-regularization extension, so an ablation grid keeps every
hyper-parameter identical except for the algorithm switches.
"""

import argparse
import os
from dataclasses import dataclass, field, fields
from typing import Optional

# Make sure HF downloads happen through the mirror even before transformers
# is imported anywhere else.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ---------------------------------------------------------------------------
# Preset table (single-variable ablation: only toggles differ)
# ---------------------------------------------------------------------------
# clip_higher / dynamic_sampling / token_level / overlong_shaping / entropy_reg
PRESETS = {
    "grpo":        dict(clip_higher=False, dynamic_sampling=False, token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "clip_higher": dict(clip_higher=True,  dynamic_sampling=False, token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "dynamic":     dict(clip_higher=False, dynamic_sampling=True,  token_level=False,
                        overlong_shaping=False, entropy_reg=False),
    "token_level": dict(clip_higher=False, dynamic_sampling=False, token_level=True,
                        overlong_shaping=False, entropy_reg=False),
    "overlong":    dict(clip_higher=False, dynamic_sampling=False, token_level=False,
                        overlong_shaping=True,  entropy_reg=False),
    "dapo":        dict(clip_higher=True,  dynamic_sampling=True,  token_level=True,
                        overlong_shaping=True,  entropy_reg=False),
    "entropy_reg": dict(clip_higher=True,  dynamic_sampling=True,  token_level=True,
                        overlong_shaping=True,  entropy_reg=True),
}


@dataclass
class Config:
    # ---- model & device ----------------------------------------------------
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"                     # auto | cuda | mps | cpu
    use_8bit: bool = False                   # bitsandbytes 8-bit (CUDA only)

    # ---- LoRA --------------------------------------------------------------
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = field(default=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ))
    ref_use_lora: bool = False               # reference frozen base (no LoRA) by default

    # ---- algorithm (resolved from preset) ----------------------------------
    algorithm: str = "grpo"
    clip_higher: bool = False
    dynamic_sampling: bool = False
    token_level: bool = False
    overlong_shaping: bool = False
    entropy_reg: bool = False

    # ---- PPO / float / KL --------------------------------------------------
    epsilon: float = 0.2                     # base clip range = eps_low
    eps_high: float = 0.28                   # widened upper bound for clip-higher
    eps_low: float = 0.2
    beta: float = 0.0                        # KL penalty coefficient
    entropy_coef: float = 0.01               # entropy regularization strength

    # ---- training ----------------------------------------------------------
    seed: int = 42
    learning_rate: float = 1e-5
    ppo_epochs: int = 1
    max_grad_norm: float = 1.0
    group_size: int = 8
    groups_per_step: int = 2
    max_new_tokens: int = 256                # response generation budget
    max_prompt_tokens: int = 512             # prompt truncation limit
    temperature: float = 1.0
    max_steps: int = 100
    dynamic_max_attempts: int = 4            # max_candidate_multiplier for dynamic sampling

    # ---- reward ------------------------------------------------------------
    overlong_cache: int = 128                # soft zone width before response budget
    format_weight: float = 0.2

    # ---- data --------------------------------------------------------------
    dataset: str = "synthetic"               # synthetic | gsm8k | gsm8k_math | /path/to.jsonl
    train_samples: int = 500
    eval_samples: int = 100

    # ---- eval / logging / checkpoint --------------------------------------
    pass_at_k: int = 0                          # 0 disables; e.g. 8 -> pass@8
    pass_k_samples: int = 50                    # prompts used for the pass@k estimate
    eval_every: int = 10
    save_every: int = 50
    use_swanlab: bool = True
    swanlab_project: str = "qwen-math-grpo-dapo"
    run_name: str = ""
    output_dir: str = "./outputs"
    log_dir: str = "./outputs/runs"
    resume_from: Optional[str] = None


def apply_preset(cfg: Config, cli_set=frozenset()) -> Config:
    """Resolve the ``--algorithm`` name into DAPO toggles + adaptive clip range."""
    preset = PRESETS.get(cfg.algorithm, PRESETS["grpo"])
    for key, value in preset.items():
        if key not in cli_set:
            setattr(cfg, key, value)

    # Clip-Higher only changes eps_high; vanilla keeps a symmetric clip.
    # eps_high/eps_low are derived, so they may only be recomputed when the
    # user did not pass them explicitly on the CLI.
    if "epsilon" not in cli_set:
        if "eps_low" not in cli_set:
            cfg.eps_low = cfg.epsilon
        if "eps_high" not in cli_set:
            cfg.eps_high = 0.28 if cfg.clip_higher else cfg.epsilon
    elif "eps_low" not in cli_set:
        cfg.eps_low = cfg.epsilon
    # keep the soft-penalty window valid for small generation budgets (smoke)
    if cfg.overlong_cache >= cfg.max_new_tokens:
        cfg.overlong_cache = max(1, cfg.max_new_tokens // 4)
    if cfg.run_name == "":
        cfg.run_name = cfg.algorithm
    return cfg


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GRPO / DAPO from-scratch reproduction + ablation grid.")
    parser.add_argument("--algorithm", type=str, default="grpo",
                        choices=list(PRESETS.keys()))
    # Every remaining Config field gets an explicit CLI flag; unknown flags
    # now hard-error instead of being silently swallowed.
    for f in fields(Config):
        if f.name == "algorithm":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or f.type == "bool":
            parser.add_argument(flag, type=lambda s: s.lower() in ("true", "1", "yes"),
                                default=None)
        elif f.type in (int, "int"):
            parser.add_argument(flag, type=int, default=None)
        elif f.type in (float, "float"):
            parser.add_argument(flag, type=float, default=None)
        elif f.type is tuple:
            # comma-separated list, e.g. --lora-target-modules q_proj,v_proj
            parser.add_argument(flag, type=lambda s: tuple(s.split(",")), default=None)
        else:
            parser.add_argument(flag, type=str, default=None)
    return parser


def load_config(argv=None) -> Config:
    parser = build_parser()
    args = parser.parse_args(argv)  # unknown flags now raise SystemExit(2)
    cfg = Config()
    cli_set = {f.name for f in fields(Config)
               if getattr(args, f.name, None) is not None}
    for name in cli_set:
        setattr(cfg, name, getattr(args, name))
    cfg.algorithm = args.algorithm

    # Presets may only fill algorithm toggles that the CLI did not set
    # explicitly, so ``--clip-higher true`` survives a preset that wants it
    # off, and explicit flags always win over preset defaults.
    return apply_preset(cfg, cli_set)
