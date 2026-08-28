#!/usr/bin/env python3
"""PyTorch parity for Vulkan hard ACT, context LERP, and drift recurrence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos.models.act import hard_act_depth_straight_through, hard_act_selection


TOL = 8.0e-6


def deterministic_values(length: int, scale: float, phase: int) -> torch.Tensor:
    return torch.tensor(
        [
            (((index * 37 + phase * 19) % 101) - 50.0) * scale / 50.0
            for index in range(length)
        ],
        dtype=torch.float32,
    )


def max_abs(actual, expected: torch.Tensor) -> float:
    actual_tensor = torch.tensor(actual, dtype=torch.float32).reshape(expected.shape)
    return float((actual_tensor - expected.detach().cpu()).abs().max().item())


def drift_reference(
    current: torch.Tensor,
    projected: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    add_current: bool,
    delta_scale: float,
    state_clamp: float,
    norm_clamp: float,
):
    current = current.clone().requires_grad_(True)
    projected = projected.clone().requires_grad_(True)
    delta = torch.tanh(projected)
    raw = current + delta * delta_scale if add_current else delta
    bounded = torch.clamp(raw, -state_clamp, state_clamp)
    norm = torch.linalg.vector_norm(bounded.float(), ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(norm_clamp / (norm.to(bounded.dtype) + 1.0e-6), max=1.0)
    output = bounded * scale
    (output * grad_output).sum().backward()
    current_grad = current.grad if current.grad is not None else torch.zeros_like(current)
    return output, current_grad, projected.grad


def main() -> None:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
            "--bin",
            "hierarchos-vulkan-control-parity",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Vulkan control parity binary failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    result = json.loads(completed.stdout)

    # ---- Hard ACT selection + straight-through depth gradient ----
    logits = torch.tensor(
        [
            [-4.0, -3.0, -2.0],
            [1.0, -0.7, 0.2],
            [0.5, 1.2, -0.5],
            [-0.1, 3.0, 4.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    outputs = deterministic_values(4 * 3 * 5, 0.37, 3).reshape(4, 3, 5)
    outputs.requires_grad_(True)
    grad_selected = deterministic_values(3 * 5, 0.29, 7).reshape(3, 5)
    grad_depth = torch.tensor([0.7, -0.4, 1.1], dtype=torch.float32)
    bounded_logits = torch.clamp(logits, -3.0, 3.0)
    assert int((logits.detach().abs() > 3.0).sum().item()) == 2
    probabilities = torch.sigmoid(bounded_logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    states = torch.zeros(4, 3, 1, dtype=torch.float32)
    selection = hard_act_selection(
        outputs,
        states,
        probabilities,
        threshold=0.72,
        min_steps=2,
    )
    depth = hard_act_depth_straight_through(
        probabilities,
        selection.executed_steps,
        threshold=0.72,
        min_steps=2,
        temperature=0.11,
    )
    ((selection.output * grad_selected).sum() + (depth * grad_depth).sum()).backward()

    checks = {
        "act_probabilities": max_abs(result["act"]["halt_probabilities"], probabilities),
        "act_executed_steps": max_abs(result["act"]["executed_steps"], selection.executed_steps),
        "act_selected_output": max_abs(result["act"]["selected_output"], selection.output),
        "act_grad_logits": max_abs(result["act"]["grad_halt_logits"], logits.grad),
        "act_grad_outputs": max_abs(result["act"]["grad_step_outputs"], outputs.grad),
    }
    assert result["act"]["selected_index"] == selection.selected_index.tolist(), result["act"]

    # ---- Context interpolation + worker concat backward ----
    enc = torch.tensor(
        [[0.2, -0.4, 0.1, 0.7, -0.3], [-0.5, 0.6, 0.25, -0.15, 0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    previous = torch.tensor(
        [[0.75, -0.9, 0.2, 0.1, -0.4], [-0.7, 0.45, 0.95, -0.2, 0.3]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor(
        [[1.1, 0.6, -0.5, 1.4, 0.2], [0.8, -1.2, 0.3, -1.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    drift = torch.tensor(
        [[0.12, -0.08, 0.05, -0.14, 0.09], [-0.11, 0.04, 0.13, -0.06, 0.02]],
        dtype=torch.float32,
        requires_grad=True,
    )
    grad_concat = deterministic_values(20, 0.31, 11).reshape(2, 10)
    sliding = torch.clamp(previous + 0.375 * (target - previous), -0.8, 0.8)
    concat = torch.cat([enc, sliding + drift], dim=-1)
    (concat * grad_concat).sum().backward()
    checks.update(
        {
            "context_output": max_abs(result["context"]["output"], concat),
            "context_grad_enc": max_abs(result["context"]["grad_enc"], enc.grad),
            "context_grad_previous": max_abs(result["context"]["grad_previous"], previous.grad),
            "context_grad_target": max_abs(result["context"]["grad_target"], target.grad),
            "context_grad_drift": max_abs(result["context"]["grad_drift"], drift.grad),
        }
    )

    # ---- State-derived drift seed and recurrent drift update ----
    current = torch.tensor(
        [[0.42, -0.51, 0.37, -0.28, 0.19], [-0.33, 0.26, -0.44, 0.39, -0.21]],
        dtype=torch.float32,
    )
    projected = torch.tensor(
        [[1.2, -0.7, 0.4, 1.8, -1.1], [-0.9, 1.4, -1.7, 0.8, 0.55]],
        dtype=torch.float32,
    )
    grad_drift_output = deterministic_values(10, 0.43, 17).reshape(2, 5)
    seed_output, seed_grad_current, seed_grad_projected = drift_reference(
        torch.zeros_like(current),
        projected,
        grad_drift_output,
        add_current=False,
        delta_scale=1.0,
        state_clamp=0.65,
        norm_clamp=0.9,
    )
    recur_output, recur_grad_current, recur_grad_projected = drift_reference(
        current,
        projected,
        grad_drift_output,
        add_current=True,
        delta_scale=0.4,
        state_clamp=0.65,
        norm_clamp=0.9,
    )
    checks.update(
        {
            "drift_seed_output": max_abs(result["drift_seed"]["output"], seed_output),
            "drift_seed_grad_current": max_abs(result["drift_seed"]["grad_current"], seed_grad_current),
            "drift_seed_grad_projected": max_abs(result["drift_seed"]["grad_projected"], seed_grad_projected),
            "drift_recur_output": max_abs(result["drift_recurrence"]["output"], recur_output),
            "drift_recur_grad_current": max_abs(result["drift_recurrence"]["grad_current"], recur_grad_current),
            "drift_recur_grad_projected": max_abs(result["drift_recurrence"]["grad_projected"], recur_grad_projected),
        }
    )

    for name, diff in checks.items():
        assert diff <= TOL, (name, diff, TOL)

    print(f"device={result['device']}")
    for name, diff in checks.items():
        print(f"{name}_max_abs_diff={diff:.3e}")
    print("Hierarchos Vulkan ACT/context/drift PyTorch parity: PASS")


if __name__ == "__main__":
    main()
