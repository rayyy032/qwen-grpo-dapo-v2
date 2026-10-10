"""Generate the run-4 Kaggle notebook (wave-based, dual-GPU, auto-resume).

Usage:
    python scripts/build_kaggle_notebook.py [--wave 1]

Writes ``kaggle_run4.ipynb`` + ``kernel-metadata.json`` at the repo root.
Push with:  kaggle kernels push -p .

Wave layout (5 official ablation arms, ~10h each, 240 steps):
    wave 1: grpo        + clip_higher     (one arm per GPU)
    wave 2: dynamic     + token_level
    wave 3: dapo                          (GPU 0; GPU 1 idle)

Each arm runs ``scripts/train.py`` with incremental metrics.jsonl + periodic
LoRA checkpoints; the runner kills on watchdog/deadline and relaunches with
``--resume-from <latest ckpt>`` until MAX_STEPS is reached or the wave
deadline passes. Cross-session resume: attach the previous kernel version's
output as an Add-On dataset and the runner copies its runs/* into working
before starting.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CELL0_MD = """\
# Qwen2.5 GRPO / DAPO 复现 run-4 — wave 制五组消融（Kaggle T4）

从零手写 GRPO + DAPO 组件消融，v2 修复版（真策略熵探针 / 全组零方差统计 /
k3 KL / MATH gt 提取 / 空输出下限 / 分层 eval / logits_to_keep + 梯度检查点）。

**运行方式**：Settings → Accelerator 选 **GPU T4 x2**（单卡 T4 也能跑，臂会串行）
→ Internet ON → Save & Run All (Commit)。

| wave | arms | 说明 |
|------|------|------|
| 1 | grpo + clip_higher | 基线熵坍缩 vs Clip-Higher 救熵 |
| 2 | dynamic + token_level | 零方差组过滤/补采 vs token 级损失归一 |
| 3 | dapo | 完整 DAPO 四组件 |

- 每 wave 一个 kernel 版本（`WAVE` 参数在 Cell 6 顶部）
- 中途崩了会自动从最近的 ckpt 续跑（`--resume-from`）
- 12h 会话上限：wave deadline 11h，超时保留部分曲线，重新 push 续跑
- 结果曲线：`outputs/runs/<arm>/metrics.jsonl` + `wave_results.json`
"""

CELL1_CODE = """\
# Cell 1: 环境检查（双卡则并行两臂，单卡自动串行）
import subprocess, torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
assert torch.cuda.is_available(), "Settings -> Accelerator 选 GPU T4 后重新运行"
N_GPU = torch.cuda.device_count()
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {p.name}, {p.total_memory/2**30:.1f} GiB")
print("mode:", "dual-GPU (两臂并行)" if N_GPU >= 2 else "single-GPU (臂串行)")
"""

CELL2_CODE = """\
# Cell 2: clone v2 仓库 + 安装依赖（pin 到本地验证过的版本）+ 完整性检查
import os, subprocess, sys

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["PYTHONUNBUFFERED"] = "1"

print("=== git clone ===")
r = subprocess.run(["git", "clone",
                   "https://github.com/rayyy032/qwen-grpo-dapo-v2.git"],
                  capture_output=True, text=True)
print(r.stdout or "", r.stderr or "", "returncode:", r.returncode)
assert r.returncode == 0, f"git clone 失败: {r.stderr}"

%cd /kaggle/working/qwen-grpo-dapo-v2

print("\\n=== 关键文件检查 ===")
critical = ["scripts/train.py", "src/config.py", "src/rewards.py",
            "src/losses.py", "src/generation.py", "src/data.py",
            "src/evaluation.py", "src/diagnostics.py", "src/train_utils.py",
            "src/agents/base_agent.py", "src/agents/grpo_agent.py",
            "tests/test_reward.py", "tests/test_train_units.py"]
missing = [f for f in critical if not (os.path.isfile(f) and os.path.getsize(f) > 0)]
print("missing:", missing or "none")
assert not missing, "文件缺失，检查 clone"

print("\\n=== pip install ===")
r = subprocess.run([sys.executable, "-m", "pip", "install",
                    "transformers==5.19.0", "datasets==5.1.0", "peft==0.21.2",
                    "accelerate==1.15.0", "math-verify", "pyyaml", "tqdm"],
                   capture_output=True, text=True)
print(r.stdout[-1500:] if len(r.stdout) > 1500 else r.stdout)
print("returncode:", r.returncode)

print("\\n=== 卸载 torchao（Kaggle 预装版与 peft 冲突） ===")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True, text=True)
print("done")

print("\\n=== import 检查 ===")
for mod in ["torch", "transformers", "datasets", "peft", "accelerate", "numpy", "yaml"]:
    try:
        m = __import__(mod)
        print(f"  [OK] {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}")
try:
    import math_verify
    print("  [OK] math_verify")
except Exception as e:
    print(f"  [FAIL] math_verify: {e}（数值等价 fallback 会退化，建议排查）")
"""

CELL3_CODE = """\
# Cell 3: 单测（50 项；全绿才继续）
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                   capture_output=True, text=True)
print(r.stdout[-3000:])
if r.stderr:
    print("STDERR:", r.stderr[-1500:])
assert r.returncode == 0, "单测有失败，先看输出"
"""

CELL4_CODE = """\
# Cell 4: 1 步 GPU 冒烟（~2min）——整条流水线（生成/奖励/损失/检查点）打通
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
r = subprocess.run([sys.executable, "scripts/train.py",
                    "--algorithm", "grpo", "--dataset", "synthetic",
                    "--train-samples", "8", "--eval-samples", "4",
                    "--group-size", "2", "--groups-per-step", "1",
                    "--max-new-tokens", "32", "--max-steps", "1",
                    "--ppo-epochs", "1", "--eval-every", "0",
                    "--save-every", "0", "--device", "cuda",
                    "--use-swanlab", "false", "--run-name", "gpu_smoke"],
                   capture_output=True, text=True, env=env, timeout=900)
print(r.stdout[-2500:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-3000:])
assert r.returncode == 0, "GPU 冒烟失败"
print("GPU smoke OK")
"""

CELL5_MD = """\
## Wave runner

- 双卡：两臂各绑一张卡（`CUDA_VISIBLE_DEVICES`）并行；单卡自动串行
- 看门狗：15 分钟无 stdout 判定挂死 → 杀进程
- 自动续跑：进程死亡/被杀后从最近 ckpt（`ckpt_step*`）`--resume-from` 重启，
  最多 8 次尝试，直到 240 步或 wave deadline（11h）
- 跨会话续跑：把上一版 kernel 的 output 作为 Add-On dataset 挂到
  `/kaggle/input/` 下，runner 会先把其中的 `outputs/runs/*` 拷回 working
- 每步曲线增量写 `outputs/runs/<arm>/metrics.jsonl`，不怕中途被杀
"""

CELL6_CODE = '''\
# ============================================================
# Cell 6: wave runner（唯一需要改的配置区）
# ============================================================
WAVE = __WAVE__
WAVES = {1: ["grpo", "clip_higher"],
          2: ["dynamic", "token_level"],
          3: ["dapo"]}
ARMS = WAVES[WAVE]

MAX_STEPS = 240          # 简历口径：150 步内两阶段坍缩 + 后续慢降
DATASET = "gsm8k_math"
TRAIN_SAMPLES = 500
EVAL_SAMPLES = 200
MATH_RATIO = 0.5
GROUP_SIZE = 8
GROUPS_PER_STEP = 2      # 16 rollouts/step（T4 步时 2.5-3.5min 的口径）
MAX_NEW_TOKENS = 384
PPO_EPOCHS = 4
LEARNING_RATE = "2e-5"
OVERLONG_CACHE = 96      # 384 的 25%，对齐论文 cache 比例
EVAL_EVERY = 40
SAVE_EVERY = 50
PASS_AT_K = 8
PASS_K_SAMPLES = 100
QUIET_KILL_MIN = 15      # 步时 ~3min + 中途 eval，15min 静默才判挂死
WAVE_DEADLINE_H = 11.0   # 12h 会话留 1h 给安装/冒烟/收尾
MAX_ATTEMPTS = 8
# ============================================================

import subprocess, threading, queue, json, time, os, sys, glob, shutil, traceback

REPO = "/kaggle/working/qwen-grpo-dapo-v2"
T0 = time.time()
results = {}
_results_lock = threading.Lock()

def save_results():
    with _results_lock:
        with open("/kaggle/working/wave_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

def load_metrics(arm):
    rows = []
    path = f"{REPO}/outputs/runs/{arm}/metrics.jsonl"
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows

def latest_ckpt(arm):
    best = None
    for d in glob.glob(f"{REPO}/outputs/runs/{arm}/ckpt_step*"):
        try:
            n = int(d.rsplit("step", 1)[1])
        except ValueError:
            continue
        if os.path.isfile(os.path.join(d, "meta.json")) and \\
                (best is None or n > best[0]):
            best = (n, d)
    return best[1] if best else None

def summarize(arm):
    rows = load_metrics(arm)
    # a watchdog kill + resume re-logs the overlapping steps; keep only the
    # last record per (kind, step) so curves stay monotone and unduplicated
    dedup = {}
    for r in rows:
        dedup[(r.get("kind"), r.get("step"))] = r
    rows = sorted(dedup.values(),
                  key=lambda r: (r.get("step") if r.get("step") is not None else 0,
                                 str(r.get("kind"))))
    train = [r for r in rows if r.get("kind") == "train"]
    out = {
        "steps_done": len(train),
        "wall_hours": round((time.time() - T0) / 3600, 2),
        "reward_curve": [r.get("reward/mean") for r in train],
        "entropy_curve": [r.get("rollout/policy_entropy") for r in train],
        "surprisal_curve": [r.get("rollout/mean_sampled_token_surprisal") for r in train],
        "zero_var_curve": [r.get("rollout/zero_variance_group_ratio") for r in train],
        "len_curve": [r.get("rollout/mean_completion_tokens") for r in train],
        "clip_fraction_curve": [r.get("train/clip_fraction") for r in train],
        "kl_curve": [r.get("rollout/kl_vs_ref") for r in train],
        "mid_evals": [{"step": r.get("step"), "accuracy": r.get("eval/accuracy"),
                     "src_gsm8k": r.get("eval/src_gsm8k"), "src_math": r.get("eval/src_math")}
                    for r in rows if r.get("kind") == "eval"],
    }
    finals = [r for r in rows if r.get("kind") == "eval_final"]
    if finals:
        out["final"] = finals[-1]
    return out

# ---- 跨会话续跑：把 Add-On dataset 里上一版的 runs/* 拷回 working ----------
for src in glob.glob("/kaggle/input/*/outputs/runs/*") + \\
           glob.glob("/kaggle/input/*/*/outputs/runs/*"):
    arm_dir = os.path.basename(src.rstrip("/"))
    dst = f"{REPO}/outputs/runs/{arm_dir}"
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(f"[resume] copied {src} -> {dst}")

# ---- 单进程预下载模型 + 数据集（避免双臂并发抢 HF cache/下载锁） -----------
print("=== pre-download (single process) ===", flush=True)
os.chdir(REPO)
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
from src.data import build_dataset
build_dataset(DATASET, TRAIN_SAMPLES, split="train", seed=42, math_ratio=MATH_RATIO)
build_dataset(DATASET, EVAL_SAMPLES, split="test", seed=42, math_ratio=MATH_RATIO)
print("pre-download done", flush=True)

def run_arm(arm, gpu_id):
    env_arm = os.environ.copy()
    env_arm["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env_arm["PYTHONUNBUFFERED"] = "1"
    env_arm["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    env_arm["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    status = "not_run"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if time.time() - T0 > WAVE_DEADLINE_H * 3600:
            status = status if status != "not_run" else "skipped_deadline"
            break
        resume = latest_ckpt(arm)
        if resume:
            print(f"[{arm}] attempt {attempt}: resume from {resume}", flush=True)
        else:
            print(f"[{arm}] attempt {attempt}: fresh start", flush=True)
        cmd = [sys.executable, "scripts/train.py",
               "--algorithm", arm, "--dataset", DATASET,
               "--train-samples", str(TRAIN_SAMPLES),
               "--eval-samples", str(EVAL_SAMPLES),
               "--math-ratio", str(MATH_RATIO),
               "--group-size", str(GROUP_SIZE),
               "--groups-per-step", str(GROUPS_PER_STEP),
               "--max-new-tokens", str(MAX_NEW_TOKENS),
               "--max-steps", str(MAX_STEPS),
               "--ppo-epochs", str(PPO_EPOCHS),
               "--learning-rate", LEARNING_RATE,
               "--overlong-cache", str(OVERLONG_CACHE),
               "--eval-every", str(EVAL_EVERY),
               "--save-every", str(SAVE_EVERY),
               "--pass-at-k", str(PASS_AT_K),
               "--pass-k-samples", str(PASS_K_SAMPLES),
               "--device", "cuda", "--use-swanlab", "false",
               "--run-name", arm]
        if resume:
            cmd += ["--resume-from", resume]
        t0, kill_reason = time.time(), None
        try:
            proc = subprocess.Popen(cmd, cwd=REPO,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   stdin=subprocess.DEVNULL,
                                   text=True, bufsize=1, env=env_arm)
            out_q, err_lines = queue.Queue(), []

            def pump_out():
                for line in proc.stdout:
                    out_q.put(line.rstrip("\\n"))
                out_q.put(None)

            def pump_err():
                for line in proc.stderr:
                    err_lines.append(line.rstrip("\\n"))
                    if len(err_lines) > 3000:  # bounded memory for chatty stderr
                        del err_lines[:1500]

            threading.Thread(target=pump_out, daemon=True).start()
            threading.Thread(target=pump_err, daemon=True).start()

            stdout_lines, last_active = [], time.time()
            prefix = f"[{arm}] "
            while True:
                try:
                    line = out_q.get(timeout=15)
                except queue.Empty:
                    line = "__timeout__"
                now = time.time()
                if line is None:
                    break
                if line != "__timeout__":
                    print(prefix + line, flush=True)
                    stdout_lines.append(line)
                    last_active = now
                    continue
                if now - last_active > QUIET_KILL_MIN * 60:
                    kill_reason = f"watchdog: {QUIET_KILL_MIN}min 无输出"
                    break
                if time.time() - T0 > WAVE_DEADLINE_H * 3600:
                    kill_reason = "wave deadline"
                    break

            if kill_reason:
                print(f"\\n=== [KILL] {arm}: {kill_reason} ===", flush=True)
                proc.kill()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            time.sleep(1.0)

            if kill_reason == "wave deadline":
                status = "partial: wave deadline"
                break
            if proc.returncode != 0 or kill_reason:
                print(f"=== [STDERR tail] {arm} ===", flush=True)
                print("\\n".join(err_lines[-25:]) or "(empty)", flush=True)
                status = f"retrying after: {kill_reason or 'exit ' + str(proc.returncode)}"
                continue  # resume loop 会从最近 ckpt 续跑
            status = "ok"
            break
        except Exception as e:
            print(f"=== [FATAL] {arm}: {e} ===", flush=True)
            traceback.print_exc()
            status = f"fatal: {e}"
            break

    results[arm] = summarize(arm)
    results[arm]["status"] = status
    save_results()
    print(f"[summary] {arm}: status={status} steps={results[arm]['steps_done']}", flush=True)

import torch
N_GPU = torch.cuda.device_count()
if N_GPU >= 2:
    threads = [threading.Thread(target=run_arm, args=(arm, i))
               for i, arm in enumerate(ARMS[:2])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
else:
    for i, arm in enumerate(ARMS):
        run_arm(arm, 0)

print(f"\\nwave {WAVE} 结束，总耗时 {(time.time() - T0)/3600:.2f}h")
print("结果 -> /kaggle/working/wave_results.json")
save_results()
'''

CELL7_CODE = """\
# Cell 7: 曲线 + 汇总表
import json, os
import matplotlib.pyplot as plt

with open("/kaggle/working/wave_results.json") as f:
    results = json.load(f)

def clean(xs):
    return [x for x in xs if x is not None]

for arm, r in results.items():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"wave {WAVE} — {arm} (steps={r['steps_done']}, {r['status']})")
    for ax, (key, title) in zip(axes.flat, [
            ("reward_curve", "reward/mean"),
            ("entropy_curve", "policy_entropy (true)"),
            ("zero_var_curve", "zero_var_group_ratio"),
            ("len_curve", "mean_completion_tokens"),
            ("clip_fraction_curve", "clip_fraction"),
            ("kl_curve", "kl_vs_ref (k3)")]):
        ys = clean(r.get(key) or [])
        ax.plot(ys, lw=1)
        ax.set_title(title)
        ax.set_xlabel("step")
    plt.tight_layout()
    plt.savefig(f"/kaggle/working/curves_{arm}.png", dpi=110)
    plt.show()

print("=== 汇总 ===")
for arm, r in results.items():
    fin = r.get("final") or {}
    print(f"{arm:14s} steps={r['steps_done']:3d} status={r['status'][:40]:42s} "
          f"acc={fin.get('eval/accuracy')} pass@8={fin.get(f'eval/pass@{PASS_AT_K}')}")
print("\\nper-source / per-level final eval:")
for arm, r in results.items():
    fin = r.get("final") or {}
    src = {k: v for k, v in fin.items() if k.startswith("eval/src_") or k.startswith("eval/level_")}
    print(f"  {arm}: {src}")
"""


def build(wave: int) -> None:
    cells = []
    for src, typ in [(CELL0_MD, "markdown"), (CELL1_CODE, "code"),
                     (CELL2_CODE, "code"), (CELL3_CODE, "code"),
                     (CELL4_CODE, "code"), (CELL5_MD, "markdown"),
                     (CELL6_CODE.replace("__WAVE__", str(wave)), "code"),
                     (CELL7_CODE, "code")]:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cells.append({"cell_type": typ, "metadata": {}, "source": source,
                      "execution_count": None,
                      **({"outputs": []} if typ == "code" else {})})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = REPO / "kaggle_run4.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out}")

    meta = {
        "id": "rickena/qwen-grpo-dapo-v2-run4",
        "title": "qwen-grpo-dapo-v2-run4",
        "code_file": "kaggle_run4.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # cold-push: version 1 runs on CPU and fails fast at the cell-1 GPU
        # assert (zero GPU quota burned). The user then opens the kernel in
        # the web UI, selects Accelerator = GPU T4 x2 (internet already on),
        # and Save & Run All to start the real wave.
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
    }
    meta_path = REPO / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")
    print("\npush with:  kaggle kernels push -p .")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args()
    build(args.wave)
