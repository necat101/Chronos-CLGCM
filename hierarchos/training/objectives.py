"""Auxiliary training objectives with explicit, testable contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaptivePonderResult:
    """Components of the adaptive-compute objective for logging and tests."""

    loss: torch.Tensor
    target_steps: torch.Tensor
    expected_steps: torch.Tensor


def resolve_ponder_objective(mode, *, architecture_revision: str | None) -> str:
    """Resolve ``auto`` without changing legacy checkpoint optimization.

    Coherent-v9 uses a symmetric objective: both needless over-computation and
    premature halting are penalized.  Legacy models retain their historical
    one-sided target unless the caller explicitly opts into the corrected loss.
    """

    normalized = str(mode or "auto").strip().lower().replace("_", "-")
    if normalized == "auto":
        revision = str(architecture_revision or "legacy-v8").strip().lower().replace("_", "-")
        return (
            "symmetric-huber"
            if revision in {"coherent", "coherent-v9", "v9", "v9-coherent"}
            else "legacy-one-sided"
        )
    aliases = {
        "symmetric": "symmetric-huber",
        "huber": "symmetric-huber",
        "legacy": "legacy-one-sided",
        "one-sided": "legacy-one-sided",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"symmetric-huber", "legacy-one-sided"}:
        raise ValueError(
            "ponder_objective must be one of "
            f"{{auto, symmetric-huber, legacy-one-sided}}, got {mode!r}"
        )
    return normalized


def adaptive_ponder_objective(
    expected_steps: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    max_steps: int,
    target_scale: float,
    min_steps: float = 1.0,
    mode: str = "symmetric-huber",
    huber_beta: float = 0.5,
    weight: torch.Tensor | None = None,
) -> AdaptivePonderResult:
    """Match expected ACT steps to a detached difficulty-derived target.

    ``expected_steps`` and ``difficulty`` may be scalars or token-shaped tensors.
    Broadcasting is allowed, which lets the same implementation serve the
    current scalar trainer and a token-level ACT path.  Difficulty is detached
    so the language model cannot lower this auxiliary by intentionally
    increasing/decreasing CE instead of learning an appropriate halt policy.
    """

    if not torch.is_tensor(expected_steps) or not torch.is_tensor(difficulty):
        raise TypeError("expected_steps and difficulty must be tensors")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if not 0.0 < float(min_steps) <= float(max_steps):
        raise ValueError(
            f"min_steps must be in (0, max_steps], got min_steps={min_steps}, "
            f"max_steps={max_steps}"
        )
    if float(target_scale) < 0.0:
        raise ValueError(f"target_scale must be non-negative, got {target_scale}")
    if float(huber_beta) <= 0.0:
        raise ValueError(f"huber_beta must be positive, got {huber_beta}")

    mode = resolve_ponder_objective(mode, architecture_revision="coherent-v9")
    expected = expected_steps.float()
    target = torch.clamp(
        difficulty.detach().float() * float(target_scale),
        min=float(min_steps),
        max=float(max_steps),
    )
    expected, target = torch.broadcast_tensors(expected, target)

    if mode == "legacy-one-sided":
        per_item = torch.relu(target - expected)
    else:
        per_item = F.smooth_l1_loss(
            expected,
            target,
            reduction="none",
            beta=float(huber_beta),
        )

    if weight is not None:
        item_weight = torch.broadcast_to(weight.float(), per_item.shape)
        finite = torch.isfinite(per_item) & torch.isfinite(item_weight) & (item_weight > 0)
        safe_weight = torch.where(finite, item_weight, torch.zeros_like(item_weight))
        safe_loss = torch.where(finite, per_item, torch.zeros_like(per_item))
        denom = safe_weight.sum().clamp_min(1e-8)
        loss = (safe_loss * safe_weight).sum() / denom
    else:
        finite = torch.isfinite(per_item)
        safe_loss = torch.where(finite, per_item, torch.zeros_like(per_item))
        loss = safe_loss.sum() / finite.sum().to(dtype=safe_loss.dtype).clamp_min(1.0)

    return AdaptivePonderResult(
        loss=loss,
        target_steps=target,
        expected_steps=expected,
    )
