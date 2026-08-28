#!/usr/bin/env python3
"""Verify CE-driven l_to_out coupling in the one-submit Vulkan token graph."""

from __future__ import annotations

import json
import math
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


def main() -> None:
    torch.manual_seed(20260812)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-loss-coupling-") as temp_dir:
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
                "full_training_graph_loss_coupling",
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
                "Vulkan loss-coupled graph verification failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["queue_submissions"] == 1, result
    assert result["h_optimizer_step"] == 1, result
    assert result["l_optimizer_step"] == 1, result
    assert result["projection_optimizer_step"] == 1, result
    assert result["lm_optimizer_step"] == 1, result
    assert math.isfinite(result["loss"]) and result["loss"] > 0.0, result
    # Weight decay is zero. Every changed tensor therefore received a real
    # graph gradient; l_to_out has no host-supplied gradient in this API.
    assert result["projection_changed_tensor_count"] == 10, result
    for field in (
        "l_to_out_max_abs_delta",
        "lm_head_max_abs_delta",
        "out_norm_max_abs_delta",
        "l_grad_x_max_abs",
    ):
        value = result[field]
        assert math.isfinite(value) and value > 0.0, (field, value, result)

    print(f"device={result['device']}")
    print("queue_submissions=1")
    print(f"loss={result['loss']:.7f}")
    print(f"l_to_out_max_abs_delta={result['l_to_out_max_abs_delta']:.3e}")
    print(f"lm_head_max_abs_delta={result['lm_head_max_abs_delta']:.3e}")
    print(f"out_norm_max_abs_delta={result['out_norm_max_abs_delta']:.3e}")
    print("Hierarchos Vulkan loss-coupled token graph: PASS")


if __name__ == "__main__":
    main()
