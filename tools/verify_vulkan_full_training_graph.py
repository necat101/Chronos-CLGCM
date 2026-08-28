#!/usr/bin/env python3
"""Verify construction of the shared Hierarchos Vulkan full-training graph."""

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


def main() -> None:
    torch.manual_seed(20260812)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-full-graph-") as temp_dir:
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
                "hierarchos-vulkan-full-training-graph-inspect",
                "--",
                "--model",
                str(model_dir),
                "--max-batch",
                "1",
                "--max-h-steps",
                "2",
                "--max-l-steps",
                "2",
                "--max-loss-rows",
                "2",
                "--plan-sequences",
                "1",
                "--plan-tokens",
                "8",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan full-training graph construction failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["context_dim"] == config.context_dim
    assert result["h_hidden"] == config.h_hidden
    assert result["l_hidden"] == config.l_hidden
    assert result["vocab_size"] == config.vocab_size
    assert result["projection_tensor_count"] == 10
    assert result["shared_lm_head_identity"] is True
    assert result["live_buffer_count"] > result["driver_allocation_count"]
    assert result["live_buffer_bytes"] > 0
    assert result["reserved_bytes"] >= result["live_buffer_bytes"]
    assert result["driver_allocation_count"] < result["max_driver_allocation_count"]
    assert result["training_working_set_logical_bytes"] > 0
    assert result["training_working_set_planned_bytes"] > 0
    assert (
        result["training_working_set_planned_bytes"]
        < result["training_working_set_logical_bytes"]
    )
    assert result["training_working_set_reused_bytes"] == (
        result["training_working_set_logical_bytes"]
        - result["training_working_set_planned_bytes"]
    )
    bindings = {entry["name"]: entry for entry in result["training_working_set_bindings"]}
    # The compiled working set now owns the complete reverse-sweep and bounded
    # optimizer transport scratch, not only the original five scratch buffers.
    # Keep this verifier tied to the aliasing contract rather than the obsolete
    # two-slot implementation detail: every logical range must bind to a valid
    # physical slot, multiple logical ranges must alias, and the representative
    # forward/backward/optimizer ranges must all be present.
    assert result["training_working_set_slot_count"] >= 5
    assert result["training_working_set_slot_count"] < len(bindings)
    assert {
        "worker.l_input_clamped",
        "worker.l_state_masked",
        "worker.l_output_adjoint",
        "worker.l_input_adjoint",
        "worker.drift_candidate",
        "worker.drift_passthrough",
        "worker.drift_commitment",
        "worker.drift_projected",
        "worker.drift_from_update",
        "worker.drift_from_input",
        "worker.vector_sum",
        "manager.h_clamped_output",
        "manager.h_candidate_output_step",
        "manager.h_recurrent_adjoint",
        "manager.h_selected_state_adjoint",
        "manager.h_candidate_state_step",
        "manager.state_grad_sum",
        "manager.vector_sum",
        "optimizer.transport_upload.0",
        "optimizer.transport_scratch.0",
        "optimizer.transport_upload.1",
        "optimizer.transport_scratch.1",
        "optimizer.nonfinite_flag",
    } <= set(bindings)
    for entry in bindings.values():
        assert entry["slot"] < result["training_working_set_slot_count"]
        assert entry["bytes"] == entry["f32_len"] * 4
    assert bindings["worker.l_input_clamped"]["begin"] == "forward"
    assert bindings["worker.l_state_masked"]["begin"] == "forward"
    assert bindings["worker.vector_sum"]["begin"] == "worker-backward"
    assert bindings["manager.state_grad_sum"]["begin"] == "manager-backward"
    assert bindings["manager.vector_sum"]["begin"] == "manager-backward"
    # Exact reuse pairings are geometry-dependent because the coloring pass is
    # size-aware. Only assert pairs that overlap in time and therefore may never
    # alias, plus the optimizer transport ranges that are all live together.
    assert bindings["manager.h_candidate_state_step"]["slot"] != bindings["manager.state_grad_sum"]["slot"]
    transport_slots = {
        bindings["optimizer.transport_upload.0"]["slot"],
        bindings["optimizer.transport_scratch.0"]["slot"],
        bindings["optimizer.transport_upload.1"]["slot"],
        bindings["optimizer.transport_scratch.1"]["slot"],
    }
    assert len(transport_slots) == 4
    assert result["estimated_vulkan_training_peak_bytes"] is not None
    assert result["estimated_vulkan_training_peak_bytes"] > result["live_buffer_bytes"]
    print(f"device={result['device']}")
    print("projection_tensor_count=10")
    print("shared_lm_head_identity=H-DeepEmbed=L-DeepEmbed=LM-loss")
    print(
        "vulkan_memory_pool="
        f"{result['live_buffer_count']} buffers / "
        f"{result['driver_allocation_count']} driver allocations / "
        f"{result['live_buffer_bytes']} live bytes / {result['reserved_bytes']} reserved bytes"
    )
    print(
        "training_working_set="
        f"{result['training_working_set_logical_bytes']} logical bytes -> "
        f"{result['training_working_set_planned_bytes']} planned bytes / "
        f"{result['training_working_set_reused_bytes']} reused bytes / "
        f"{result['training_working_set_slot_count']} slots"
    )
    print(
        "estimated_vulkan_training_peak_bytes="
        f"{result['estimated_vulkan_training_peak_bytes']}"
    )
    print("Hierarchos Vulkan full-training graph ownership: PASS")


if __name__ == "__main__":
    main()
