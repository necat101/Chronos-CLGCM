#!/usr/bin/env python3
"""Verify that H/L/out-norm training fusion preserves the phased Vulkan math."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config


MAX_ABS_TOLERANCE = 5.0e-6


def main() -> None:
    torch.manual_seed(20260812)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-full-fusion-") as temp_dir:
        model_dir = Path(temp_dir) / "model"
        export_model(model, config, model_dir)
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-full-training-graph-fusion",
                "--",
                "--model",
                str(model_dir),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan full-training fusion verification failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["queue_submissions"] == 1, result
    assert result["h_optimizer_step_match"] is True, result
    assert result["l_optimizer_step_match"] is True, result
    assert result["lm_optimizer_step_match"] is True, result
    numeric_fields = (
        "h_sequence_max_abs_diff",
        "l_sequence_max_abs_diff",
        "h_parameter_max_abs_diff",
        "l_parameter_max_abs_diff",
        "lm_head_max_abs_diff",
        "out_norm_weight_max_abs_diff",
        "out_norm_bias_max_abs_diff",
        "loss_abs_diff",
    )
    for field in numeric_fields:
        assert result[field] <= MAX_ABS_TOLERANCE, (
            field,
            result[field],
            MAX_ABS_TOLERANCE,
            result,
        )

    print(f"device={result['device']}")
    print("queue_submissions=1")
    for field in numeric_fields:
        print(f"{field}={result[field]:.3e}")
    print("Hierarchos Vulkan H/L/out_norm execution fusion: PASS")


if __name__ == "__main__":
    main()
