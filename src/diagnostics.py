"""Training diagnostics dashboard (SwanLab-backed, console-always).

Records: mean sampled-token surprisal (entropy proxy), KL divergence, mean
completion tokens, zero-variance group ratio, effective group ratio,
clip fraction, and the three reward-hacking rates.
"""

from __future__ import annotations

from typing import Dict, Optional


class Diagnostics:
    """SwanLab recorder with graceful no-op fallback (offline-safe)."""

    def __init__(self, project: str, experiment_name: str, config: dict,
                 enabled: bool = True, logdir: Optional[str] = None):
        self.enabled = enabled
        self._swanlab = None
        if enabled:
            try:
                import swanlab
                swanlab.init(
                    project=project,
                    experiment_name=experiment_name,
                    config=config,
                    logdir=logdir,
                    mode="local",
                )
                self._swanlab = swanlab
                print(f"[diagnostics] SwanLab initialized (project={project}, "
                      f"run={experiment_name})")
            except Exception as exc:  # noqa: BLE001 - offline/logout fallback
                print(f"[diagnostics] SwanLab unavailable ({exc}); "
                      f"falling back to console-only logging.")
                self.enabled = False

    def log(self, step: int, metrics: Dict[str, float]):
        if self.enabled and self._swanlab is not None:
            try:
                self._swanlab.log(metrics, step=step)
            except Exception:  # noqa: BLE001
                pass

    def finish(self):
        if self.enabled and self._swanlab is not None:
            try:
                self._swanlab.finish()
            except Exception:  # noqa: BLE001
                pass


def aggregate_hack_rates(results) -> Dict[str, float]:
    """Aggregate per-sample HackFlags into mean rates."""
    n = max(len(results), 1)
    length_exploit = sum(1 for r in results if r.hack.length_exploit) / n
    format_only = sum(1 for r in results if r.hack.format_only_hack) / n
    equiv_bypass = sum(1 for r in results if r.hack.equivalence_bypass) / n
    return {
        "reward_hack/length_exploit": length_exploit,
        "reward_hack/format_only": format_only,
        "reward_hack/equivalence_bypass": equiv_bypass,
    }
