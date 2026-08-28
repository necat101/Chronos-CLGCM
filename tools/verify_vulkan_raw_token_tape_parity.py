#!/usr/bin/env python3
"""Verify multi-token raw-token Vulkan tape parity against the host-enc tape."""

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

from hierarchos import HierarchosCore, load_full_model_with_config
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
from tools.vulkan_optimizer_bridge import load_vulkan_training_package_into_torch


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def main() -> None:
    torch.manual_seed(20260814)
    config = tiny_coherent_config(32)
    config.max_h_steps = max(1, config.max_h_steps)
    config.max_l_steps = max(1, config.max_l_steps)
    # Keep native val_proj alignment live in this parity harness so raw dense,
    # sparse replay, and exact historical-TBPTT weighting all exercise the
    # sampled LTM backward path on the real Vulkan device.
    config.ltm_value_alignment_weight = 0.05
    config.ltm_value_alignment_stride = 2
    config.ltm_value_alignment_min_updates = 1
    model = HierarchosCore(config).eval()
    _make_nontrivial_memory_fixture(model, config)

    batch = 2
    tokens = [[2, 7], [5, 8], [2, 7]]
    resets = [[1, 1], [0, 0], [0, 1]]
    targets = [[5, 9], [2, 7], [8, 4]]
    steps: list[dict[str, object]] = []
    for index, (token_ids, reset_lanes, step_targets) in enumerate(
        zip(tokens, resets, targets, strict=True)
    ):
        scale = 1.0 + index * 0.15
        steps.append(
            {
                "token_ids": token_ids,
                "rosa_reset_lanes": reset_lanes,
                "previous_context": (
                    torch.randn(batch, config.context_dim) * (0.035 * scale)
                )
                .flatten()
                .tolist(),
                "target_context": (
                    torch.randn(batch, config.context_dim) * (0.03 * scale)
                )
                .flatten()
                .tolist(),
                "context_alpha": 0.25 + 0.125 * index,
                "h_token_ids": token_ids,
                "l_token_ids": token_ids,
                "h_to_context_grad": (
                    torch.randn(batch, config.context_dim) * 0.004
                )
                .flatten()
                .tolist(),
                "h_depth_grad": [0.003 + index * 0.0004, -0.0025],
                "final_drift_grad": (
                    torch.randn(batch, config.context_dim) * 0.0035
                )
                .flatten()
                .tolist(),
                "commitment_cost_grad": [0.05, 0.035 + index * 0.005],
                "targets": step_targets,
            }
        )

    h_state = (
        torch.randn(batch, config.h_hidden, model.h_rnn.state_size, dtype=torch.float32)
        * 0.01
    )
    l_state = (
        torch.randn(batch, config.l_hidden, model.l_rnn.state_size, dtype=torch.float32)
        * 0.01
    )
    case = {
        "h_initial_packed_state": h_state.flatten().tolist(),
        "l_initial_packed_state": l_state.flatten().tolist(),
        "steps": steps,
        "optimizer": {
            "lr": 3.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.0,
        },
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-raw-token-tape-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        case_path = temp / "case.json"
        export_model(model, config, model_dir)
        case_path.write_text(json.dumps(case), encoding="utf-8")
        result = json.loads(
            _run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--release",
                    "--manifest-path",
                    str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                    "--bin",
                    "hierarchos-vulkan-raw-token-tape-parity",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(case_path),
                ]
            ).stdout
        )

        # Consume the exact live-window package emitted above through the
        # PyTorch bridge before the temporary fixture disappears. This validates
        # the real Rust SafeTensors metadata/slot ordering, not a synthetic copy.
        open_checkpoint = temp / "open-accumulation-checkpoint"
        pytorch_resume_model, _ = load_full_model_with_config(
            str(open_checkpoint),
            torch.device("cpu"),
        )
        pytorch_resume_optimizer = torch.optim.AdamW(
            pytorch_resume_model.parameters(),
            lr=case["optimizer"]["lr"],
            weight_decay=case["optimizer"]["weight_decay"],
        )
        pytorch_resume = load_vulkan_training_package_into_torch(
            pytorch_resume_model,
            pytorch_resume_optimizer,
            open_checkpoint,
        )
        if pytorch_resume.pending_gradients is None:
            raise AssertionError("Rust open-window package lost pending gradients in PyTorch")
        if len(pytorch_resume.pending_gradients.slot_names) != result["raw_optimizer_tensor_count"]:
            raise AssertionError("Rust/PyTorch pending-gradient registry cardinality changed")
        if pytorch_resume.pytorch_accumulation_normalization != "weighted-token":
            raise AssertionError("Rust open-window normalization did not map to PyTorch")
        if pytorch_resume_model.lm_head.weight.grad is None:
            raise AssertionError("Rust canonical lm_head gradient did not reach tied PyTorch weight")
        if pytorch_resume_model.lm_head.weight.grad is not pytorch_resume_model.tok_emb.weight.grad:
            raise AssertionError("Rust lm_head gradient was duplicated instead of restoring tied topology")

    if result["batch"] != batch or result["tokens"] != len(steps):
        raise AssertionError(f"unexpected raw tape geometry: {result}")
    if result["raw_queue_submissions"] != 1:
        raise AssertionError("raw token tape must use exactly one Vulkan submission")
    if not result["control_match"]:
        raise AssertionError("raw token tape hard-control checkpoints diverged")
    if result["raw_optimizer_tensor_count"] != result["reference_optimizer_tensor_count"] + 16:
        raise AssertionError("raw token tape did not retain the full frontend optimizer registry")
    if result["dense_microbatch_queue_submissions"] != 1:
        raise AssertionError("raw dense two-sequence arena must use exactly one Vulkan submission")
    if result["sparse_microbatch_queue_submissions"] != 1:
        raise AssertionError("raw sparse two-sequence arena must use exactly one Vulkan submission")
    if result["exact_tbptt_dense_queue_submissions"] != 1:
        raise AssertionError("exact TBPTT dense arena must use exactly one Vulkan submission")
    if result["exact_tbptt_sparse_queue_submissions"] != 1:
        raise AssertionError("exact TBPTT sparse arena must use exactly one Vulkan submission")
    if not result["dense_microbatch_control_match"]:
        raise AssertionError("raw dense multi-sequence hard-control checkpoints diverged")
    if not result["sparse_raw_vs_dense_control_match"]:
        raise AssertionError("raw sparse replay hard-control checkpoints diverged from dense")
    if result["multi_device_stream_peak_host_gradient_bytes"] > 4096 * 4:
        raise AssertionError("multi-device gradient transport exceeded its bounded host chunk")
    if result["multi_device_stream_chunk_count"] < result["multi_device_shard_gradient_tensor_count"]:
        raise AssertionError("multi-device gradient transport did not visit every canonical tensor")
    if result["multi_device_stream_value_count"] <= 4096:
        raise AssertionError("multi-device parity fixture did not exercise a multi-chunk gradient stream")
    if result["multi_device_stream_backend"] == "opaque-external-memory":
        if result["multi_device_stream_persistent_transport_reused"]:
            raise AssertionError("first opaque-external gradient stream unexpectedly reported cache reuse")
        if not result["multi_device_second_stream_persistent_transport_reused"]:
            raise AssertionError("second opaque-external gradient stream did not reuse persistent slots")
    if result["multi_device_replica_state_stream_backend"] != "portable-host-snapshot":
        if result["multi_device_replica_state_stream_chunk_count"] < 3 * result["multi_device_shard_gradient_tensor_count"]:
            raise AssertionError("direct replica-state transport did not visit parameter and both AdamW moment planes")
        if result["multi_device_replica_state_stream_value_count"] <= 4096:
            raise AssertionError("replica-state parity fixture did not exercise a multi-chunk direct stream")
        if result["multi_device_replica_state_stream_pipeline_slots"] != 2:
            raise AssertionError("direct replica-state transport did not preserve the two-slot pipeline")
        if result["multi_device_replica_state_stream_backend"] == "opaque-external-memory":
            if not result["multi_device_replica_state_second_stream_persistent_transport_reused"]:
                raise AssertionError("second opaque-external replica-state broadcast did not reuse persistent slots")

    print(
        f"device={result['device']} batch={result['batch']} tokens={result['tokens']}"
    )
    print(f"queue_submissions={result['raw_queue_submissions']}")
    print(f"loss_max_abs_diff={result['loss_max_abs_diff']:.9g}")
    print(f"final_state_max_abs_diff={result['final_state_max_abs_diff']:.9g}")
    print(
        f"initial_adjoint_max_abs_diff={result['initial_adjoint_max_abs_diff']:.9g}"
    )
    print(f"control_match={result['control_match']}")
    print(
        f"common_optimizer_max_abs_diff={result['common_optimizer_max_abs_diff']:.9g}"
    )
    print(
        "optimizer_tensor_count="
        f"{result['reference_optimizer_tensor_count']} -> {result['raw_optimizer_tensor_count']}"
    )
    print(f"raw_frontend_moment_l1={result['raw_frontend_moment_l1']:.9g}")
    print(
        "dense_microbatch="
        f"loss:{result['dense_microbatch_loss_max_abs_diff']:.9g} "
        f"state:{result['dense_microbatch_state_max_abs_diff']:.9g} "
        f"adjoint:{result['dense_microbatch_initial_adjoint_max_abs_diff']:.9g} "
        f"controls:{result['dense_microbatch_control_match']} "
        f"submissions:{result['dense_microbatch_queue_submissions']}"
    )
    print(
        "sparse_raw_vs_dense="
        f"loss:{result['sparse_raw_vs_dense_loss_max_abs_diff']:.9g} "
        f"state:{result['sparse_raw_vs_dense_state_max_abs_diff']:.9g} "
        f"adjoint:{result['sparse_raw_vs_dense_initial_adjoint_max_abs_diff']:.9g} "
        f"optimizer:{result['sparse_raw_vs_dense_optimizer_max_abs_diff']:.9g} "
        f"controls:{result['sparse_raw_vs_dense_control_match']} "
        f"submissions:{result['sparse_microbatch_queue_submissions']}"
    )
    print(
        "exact_tbptt_sparse_vs_dense="
        f"loss:{result['exact_tbptt_sparse_vs_dense_loss_max_abs_diff']:.9g} "
        f"state:{result['exact_tbptt_sparse_vs_dense_state_max_abs_diff']:.9g} "
        f"adjoint:{result['exact_tbptt_sparse_vs_dense_initial_adjoint_max_abs_diff']:.9g} "
        f"optimizer:{result['exact_tbptt_sparse_vs_dense_optimizer_max_abs_diff']:.9g} "
        f"controller:{result['exact_tbptt_controller_last_abs_diff']:.9g} "
        f"rows:{result['exact_tbptt_controller_window_rows']}/"
        f"{result['exact_tbptt_controller_closing_microbatch_rows']} "
        f"checkpoint_resume:{result['open_accumulation_checkpoint_roundtripped']} "
        f"checkpoint_pending_grad:{result['open_accumulation_checkpoint_pending_gradient_max_abs_diff']:.9g} "
        f"checkpoint_optimizer:{result['open_accumulation_checkpoint_optimizer_max_abs_diff']:.9g} "
        f"checkpoint_parameters:{result['open_accumulation_checkpoint_parameter_max_abs_diff']:.9g} "
        f"controls:{result['exact_tbptt_sparse_vs_dense_control_match']} "
        f"submissions:{result['exact_tbptt_dense_queue_submissions']}/"
        f"{result['exact_tbptt_sparse_queue_submissions']}"
    )
    print(
        "multi_device_stream="
        f"tensors:{result['multi_device_shard_gradient_tensor_count']} "
        f"chunks:{result['multi_device_stream_chunk_count']} "
        f"values:{result['multi_device_stream_value_count']} "
        f"backend:{result['multi_device_stream_backend']} "
        f"persistent_reuse:{result['multi_device_second_stream_persistent_transport_reused']} "
        f"peak_host_gradient_bytes:{result['multi_device_stream_peak_host_gradient_bytes']}"
    )
    print(
        "replica_state_stream="
        f"backend:{result['multi_device_replica_state_stream_backend']} "
        f"chunks:{result['multi_device_replica_state_stream_chunk_count']} "
        f"values:{result['multi_device_replica_state_stream_value_count']} "
        f"slots:{result['multi_device_replica_state_stream_pipeline_slots']} "
        f"persistent_reuse:{result['multi_device_replica_state_second_stream_persistent_transport_reused']}"
    )
    print("Raw-token multi-token Vulkan tape parity: PASS")


if __name__ == "__main__":
    main()
