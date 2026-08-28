#!/usr/bin/env python3
"""A/B benchmark the width-aware SharedTokenAdapter up-projection fusion."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "hierarchos-vulkan" / "Cargo.toml"
BINARY = (
    ROOT
    / "hierarchos-vulkan"
    / "target"
    / "release"
    / "hierarchos-vulkan-adapter-step.exe"
)


def make_case(*, rows: int, input_dim: int, rank: int, output_dim: int, steps: int) -> dict:
    generator = torch.Generator().manual_seed(20260814 + output_dim)
    return {
        "rows": rows,
        "steps": steps,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": rank,
        "input": torch.randn(rows * input_dim, generator=generator).tolist(),
        "grad_output": (torch.randn(rows * output_dim, generator=generator) * 0.03).tolist(),
        "down_weight": (torch.randn(rank * input_dim, generator=generator) * 0.04).tolist(),
        "up_weight": (torch.randn(output_dim * rank, generator=generator) * 0.025).tolist(),
        "bias": (torch.randn(output_dim, generator=generator) * 0.01).tolist(),
        "lr": 1.0e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
        "matrix_weight_decay": 0.0,
    }


def run_case(case_path: Path, *, disable_fusion: bool) -> dict:
    env = os.environ.copy()
    if disable_fusion:
        env["HIERARCHOS_VULKAN_DISABLE_ADAPTER_UP_FUSION"] = "1"
    else:
        env.pop("HIERARCHOS_VULKAN_DISABLE_ADAPTER_UP_FUSION", None)
    completed = subprocess.run(
        [str(BINARY), "--case", str(case_path)],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "adapter fusion benchmark runner failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def max_abs(left: list[float], right: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def compare_results(fused: dict, legacy: dict) -> float:
    return max(
        max_abs(fused[name], legacy[name])
        for name in ("output", "input_grad", "down_weight", "up_weight", "bias")
    )


def benchmark_geometry(
    *, label: str, rows: int, input_dim: int, rank: int, output_dim: int, steps: int
) -> None:
    case = make_case(
        rows=rows,
        input_dim=input_dim,
        rank=rank,
        output_dim=output_dim,
        steps=steps,
    )
    with tempfile.TemporaryDirectory(prefix=f"hierarchos-adapter-fusion-{label}-") as temp_dir:
        case_path = Path(temp_dir) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        fused = run_case(case_path, disable_fusion=False)
        legacy = run_case(case_path, disable_fusion=True)

    drift = compare_results(fused, legacy)
    if drift > 2.0e-6:
        raise AssertionError(f"{label} fused/legacy drift {drift:.9g} exceeds tolerance")

    dispatch_delta = legacy["dispatch_count"] - fused["dispatch_count"]
    barrier_delta = legacy["shader_barrier_count"] - fused["shader_barrier_count"]
    speedup = legacy["elapsed_ms"] / fused["elapsed_ms"] if fused["elapsed_ms"] > 0.0 else 0.0
    print(
        f"{label}: device={fused['device']} geometry={rows}x{input_dim}->{rank}->{output_dim} "
        f"steps={steps} fused_dispatches={fused['dispatch_count']} "
        f"legacy_dispatches={legacy['dispatch_count']} dispatch_delta={dispatch_delta} "
        f"fused_barriers={fused['shader_barrier_count']} "
        f"legacy_barriers={legacy['shader_barrier_count']} barrier_delta={barrier_delta} "
        f"fused_ms={fused['elapsed_ms']:.3f} legacy_ms={legacy['elapsed_ms']:.3f} "
        f"speedup={speedup:.3f}x max_abs_drift={drift:.9g}"
    )

    if output_dim <= 512 and rank <= 64:
        if dispatch_delta != 1 or barrier_delta != 1:
            raise AssertionError(
                f"{label} expected the whole-adapter fast path to remove exactly one "
                f"dispatch/barrier, got dispatch_delta={dispatch_delta} barrier_delta={barrier_delta}"
            )
    else:
        if dispatch_delta != 0 or barrier_delta != 0:
            raise AssertionError(
                f"{label} expected width/rank fallback to preserve the legacy graph, got "
                f"dispatch_delta={dispatch_delta} barrier_delta={barrier_delta}"
            )


def main() -> None:
    subprocess.run(
        [
            "cargo",
            "build",
            "--quiet",
            "--release",
            "--manifest-path",
            str(MANIFEST),
            "--bin",
            "hierarchos-vulkan-adapter-step",
        ],
        cwd=ROOT,
        check=True,
    )
    benchmark_geometry(
        label="coherent-v9-default",
        rows=4,
        input_dim=448,
        rank=64,
        output_dim=448,
        steps=8,
    )
    benchmark_geometry(
        label="wide-output-fallback",
        rows=4,
        input_dim=448,
        rank=64,
        output_dim=768,
        steps=4,
    )
    print("Hierarchos Vulkan adapter up-projection fusion benchmark: PASS")


if __name__ == "__main__":
    main()
