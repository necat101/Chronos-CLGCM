#!/usr/bin/env python3
"""Qualify 16-bit PyTorch checkpoints against Hierarchos FP16 Vulkan training storage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE_MIXED_ADAMW_EPS = 1.0e-6
PRODUCTION_DYNAMIC_LOSS_SCALE = 1024.0
PRODUCTION_PARAMETER_TOLERANCE = 8.0e-6
PRODUCTION_RECURRENT_STATE_TOLERANCE = 8.0e-6
AGGRESSIVE_FP16_PRECISIONS = (
    "fp16-storage-parity",
    "fp16-storage-fp16-lm-backward",
)


def _stdout_metric(stdout: str, prefix: str) -> float:
    for line in stdout.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].split(maxsplit=1)[0]
            return float(value)
    raise RuntimeError(f"qualification output is missing metric {prefix!r}")


def _run_case(checkpoint_dtype: str, precision: str, *, require_cuda: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "verify_vulkan_labeled_sequence_parity.py"),
        "--precision",
        precision,
        "--checkpoint-dtype",
        checkpoint_dtype,
        "--optimizer-eps",
        str(PORTABLE_MIXED_ADAMW_EPS),
        "--dynamic-loss-scale",
        str(PRODUCTION_DYNAMIC_LOSS_SCALE),
    ]
    if require_cuda:
        command.append("--require-cuda")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(
            f"{checkpoint_dtype} checkpoint + {precision} qualification failed "
            f"with exit code {completed.returncode}"
        )

    parameter_diff = _stdout_metric(completed.stdout, "parameter_max_abs_diff=")
    recurrent_state_diff = _stdout_metric(
        completed.stdout, "recurrent_state_max_abs_diff="
    )
    if parameter_diff > PRODUCTION_PARAMETER_TOLERANCE:
        raise SystemExit(
            f"{checkpoint_dtype} checkpoint + {precision} exceeded the production-scaled "
            f"parameter parity ceiling: {parameter_diff:.9g} > "
            f"{PRODUCTION_PARAMETER_TOLERANCE:.9g}"
        )
    if recurrent_state_diff > PRODUCTION_RECURRENT_STATE_TOLERANCE:
        raise SystemExit(
            f"{checkpoint_dtype} checkpoint + {precision} exceeded the production-scaled "
            f"recurrent-state parity ceiling: {recurrent_state_diff:.9g} > "
            f"{PRODUCTION_RECURRENT_STATE_TOLERANCE:.9g}"
        )


def _run_mid_window_case(precision: str, *, require_cuda: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "verify_vulkan_mid_window_cross_backend.py"),
        "--open-precision",
        precision,
        "--resume-precision",
        precision,
    ]
    if require_cuda:
        command.append("--require-cuda")
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"{precision} open-window + overflow/backoff qualification failed "
            f"with exit code {completed.returncode}"
        )


def _run_closed_window_trajectory(precision: str, *, require_cuda: bool) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "verify_vulkan_cuda_vulkan_trajectory.py"),
        "--precision",
        precision,
        "--cycles",
        "2",
    ]
    if require_cuda:
        command.append("--require-cuda")
    else:
        command.append("--cpu-fallback")
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"{precision} repeated Vulkan/PyTorch/Vulkan trajectory failed "
            f"with exit code {completed.returncode}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="also require the Vulkan-trained package to pass PyTorch/CUDA inference",
    )
    args = parser.parse_args()

    for precision in AGGRESSIVE_FP16_PRECISIONS:
        for checkpoint_dtype in ("fp16", "bf16"):
            print(
                f"=== qualifying checkpoint={checkpoint_dtype} "
                f"vulkan_precision={precision} adamw_eps={PORTABLE_MIXED_ADAMW_EPS:g} "
                f"dynamic_loss_scale={PRODUCTION_DYNAMIC_LOSS_SCALE:g} ===",
                flush=True,
            )
            _run_case(checkpoint_dtype, precision, require_cuda=args.require_cuda)

        print(
            f"=== qualifying open-window checkpoint/resume + dynamic-loss-scale backoff "
            f"vulkan_precision={precision} ===",
            flush=True,
        )
        _run_mid_window_case(precision, require_cuda=args.require_cuda)

        print(
            f"=== qualifying repeated Vulkan <-> PyTorch training trajectory "
            f"vulkan_precision={precision} ===",
            flush=True,
        )
        _run_closed_window_trajectory(precision, require_cuda=args.require_cuda)

    cuda_contract = "required" if args.require_cuda else "hardware-gated"
    print(
        "16-bit PyTorch checkpoint -> production-scaled FP16 Vulkan training -> "
        "portable checkpoint/resume -> repeated PyTorch training handoff -> "
        "native Rust inference qualification: "
        f"PASS (CUDA={cuda_contract})"
    )


if __name__ == "__main__":
    main()
