"""Device selection, seeding, AMP-aware backward+step."""

import random

import numpy as np
import torch


def get_device(device: str = "auto"):
    """Resolve a device string; ``auto`` prefers cuda > mps > cpu."""
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_backward_step(loss, optimizer, scaler, model, max_grad_norm=1.0):
    """Backward + grad-clip + optimizer step, AMP-safe when ``scaler`` set.

    Frees the autograd graph + CUDA cache after stepping. On T4 the policy
    forward pass over full-length (256-token) completions plus LoRA gradients
    sits right at the memory ceiling; without this release the cache
    fragmentation accumulates and a later epoch hits OOM even though each
    single step fits.
    """
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    optimizer.zero_grad()
    del loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
