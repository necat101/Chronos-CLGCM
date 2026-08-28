#!/usr/bin/env python3
"""Verify Vulkan packed/clamped RWKV state ownership and clamp backward."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def run_case(mode: str) -> None:
    torch.manual_seed(20260818 if mode == "explicit-output" else 20260819)
    batch = 3
    width = 16
    head_size = 4
    matrix_offset = 4 if mode == "explicit-output" else 3
    state_size = matrix_offset + head_size
    state_clamp = 1.25

    packed_state = torch.randn(batch, width, state_size, dtype=torch.float32) * 0.7
    x_norm = (torch.randn(batch, width, dtype=torch.float32) * 1.6).requires_grad_()
    x_norm2 = (torch.randn(batch, width, dtype=torch.float32) * 1.5).requires_grad_()
    v_first = (torch.randn(batch, width, dtype=torch.float32) * 1.8).requires_grad_()
    output = (torch.randn(batch, width, dtype=torch.float32) * 1.7).requires_grad_()
    new_matrix_state = (
        torch.randn(batch, width, head_size, dtype=torch.float32) * 1.9
    ).requires_grad_()
    # Explicitly cover both inclusive clamp boundaries and both saturated
    # sides. Random values alone make boundary-derivative regressions invisible.
    with torch.no_grad():
        x_norm.view(-1)[:4] = torch.tensor([-2.0, -state_clamp, state_clamp, 2.0])
        x_norm2.view(-1)[:4] = torch.tensor([-2.0, -state_clamp, state_clamp, 2.0])
        v_first.view(-1)[:4] = torch.tensor([-2.0, -state_clamp, state_clamp, 2.0])
        output.view(-1)[:4] = torch.tensor([-2.0, -state_clamp, state_clamp, 2.0])
        new_matrix_state.view(-1)[:4] = torch.tensor(
            [-2.0, -state_clamp, state_clamp, 2.0]
        )
    grad_packed = torch.randn(batch, width, state_size, dtype=torch.float32) * 0.08

    pieces = [
        x_norm.unsqueeze(-1),
        x_norm2.unsqueeze(-1),
        v_first.unsqueeze(-1),
    ]
    if mode == "explicit-output":
        pieces.append(output.unsqueeze(-1))
    pieces.append(new_matrix_state)
    packed_new = torch.cat(pieces, dim=-1)
    packed_new = torch.clamp(packed_new, -state_clamp, state_clamp)
    (packed_new * grad_packed).sum().backward()

    expected_output_grad = (
        output.grad if mode == "explicit-output" else torch.zeros_like(output)
    )
    case = {
        "batch": batch,
        "width": width,
        "head_size": head_size,
        "mode": mode,
        "state_clamp": state_clamp,
        "packed_state": packed_state.flatten().tolist(),
        "x_norm": x_norm.detach().flatten().tolist(),
        "x_norm2": x_norm2.detach().flatten().tolist(),
        "v_first": v_first.detach().flatten().tolist(),
        "output": output.detach().flatten().tolist(),
        "new_matrix_state": new_matrix_state.detach().flatten().tolist(),
        "grad_packed_new_state": grad_packed.flatten().tolist(),
    }

    with tempfile.TemporaryDirectory(prefix=f"hierarchos-vulkan-packed-{mode}-") as temp_dir:
        case_path = Path(temp_dir) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-rwkv-packed-state-step",
                "--",
                "--case",
                str(case_path),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Vulkan packed-state {mode} runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    comparisons = {
        "previous_tm": (result["previous_tm"], packed_state[:, :, 0]),
        "previous_cm": (result["previous_cm"], packed_state[:, :, 1]),
        "previous_v_first": (result["previous_v_first"], packed_state[:, :, 2]),
        "matrix_state": (result["matrix_state"], packed_state[:, :, matrix_offset:]),
        "packed_new_state": (result["packed_new_state"], packed_new.detach()),
        "grad_x_norm": (result["grad_x_norm"], x_norm.grad),
        "grad_x_norm2": (result["grad_x_norm2"], x_norm2.grad),
        "grad_v_first": (result["grad_v_first"], v_first.grad),
        "grad_output": (result["grad_output"], expected_output_grad),
        "grad_matrix_state": (result["grad_matrix_state"], new_matrix_state.grad),
    }
    diffs: dict[str, float] = {}
    for name, (actual_values, expected) in comparisons.items():
        assert expected is not None
        actual = torch.tensor(actual_values, dtype=torch.float32).reshape_as(expected)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        diffs[name] = (actual - expected).abs().max().item()

    print(
        f"{mode}: device={result['device']} state_size={result['state_size']} "
        + " ".join(f"max_{name}_diff={value:.9g}" for name, value in diffs.items())
    )


def main() -> None:
    run_case("legacy-input-cache")
    run_case("explicit-output")
    print("Hierarchos Vulkan packed/clamped RWKV state parity: PASS")


if __name__ == "__main__":
    main()
