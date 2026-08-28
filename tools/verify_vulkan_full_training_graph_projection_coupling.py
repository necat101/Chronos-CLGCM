#!/usr/bin/env python3
"""Verify the first projection-coupled one-submit Hierarchos Vulkan slice."""

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

    with tempfile.TemporaryDirectory(
        prefix="hierarchos-vulkan-projection-coupling-"
    ) as temp_dir:
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
                "hierarchos-vulkan-full-training-graph-projection-coupling",
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
                "Vulkan projection-coupled graph verification failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["queue_submissions"] == 1, result
    assert result["h_optimizer_step"] == 1, result
    assert result["l_optimizer_step"] == 1, result
    assert result["projection_optimizer_step"] == 1, result
    assert result["projection_optimizer_tensor_count"] == 10, result
    assert result["shared_lm_step"] == 1, result
    assert result["optimizer_checkpoint_step"] == 1, result
    assert result["optimizer_checkpoint_max_abs_diff"] == 0.0, result
    # The verifier disables weight decay, so all ten changed tensors prove that
    # every one of the six projection nodes produced a real backward gradient.
    assert result["projection_changed_tensor_count"] == 10, result
    for field in (
        "projection_max_abs_delta",
        "h_grad_x_max_abs",
        "l_grad_x_max_abs",
        "l_initial_state_grad_max_abs",
    ):
        value = result[field]
        assert math.isfinite(value) and value > 0.0, (field, value, result)

    print(f"device={result['device']}")
    print("queue_submissions=1")
    print("projection_optimizer_tensor_count=10")
    print("projection_changed_tensor_count=10 (weight_decay=0)")
    print("projection_optimizer_checkpoint_roundtrip=bit-exact")
    print(
        "l_initial_state_grad_max_abs="
        f"{result['l_initial_state_grad_max_abs']:.3e}"
    )
    print("Hierarchos Vulkan projection-coupled recurrent graph: PASS")


if __name__ == "__main__":
    main()
