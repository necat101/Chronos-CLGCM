#!/usr/bin/env python3
"""Smoke the first raw-token -> loss one-submit Hierarchos Vulkan graph."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore, load_full_model_with_config
from hierarchos.training.trainer import build_hierarchos_optimizer
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
from tools.vulkan_optimizer_bridge import (
    load_vulkan_adamw_into_torch,
    read_vulkan_training_replay,
)

TRAINING_PRECISION_ENV = "HIERARCHOS_VULKAN_TRAINING_PRECISION"


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=(
            "fp32",
            "fp16-storage-fp32-compute",
            "fp16-storage-fp16-lm-backward",
        ),
        default="fp32",
        help="Vulkan trainable execution-storage precision arm",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=32,
        help="context/H/L width; 448 exercises the production 64/64/96 low-rank geometry",
    )
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("--width must be positive")
    torch.manual_seed(20260814)
    config = tiny_coherent_config(args.width)
    config.max_h_steps = max(1, config.max_h_steps)
    config.max_l_steps = max(1, config.max_l_steps)
    model = HierarchosCore(config).eval()
    _make_nontrivial_memory_fixture(model, config)

    batch = 2
    token_ids = [2, 7]
    previous_context = torch.randn(batch, config.context_dim, dtype=torch.float32) * 0.04
    target_context = torch.randn(batch, config.context_dim, dtype=torch.float32) * 0.035
    h_state = (
        torch.randn(batch, config.h_hidden, model.h_rnn.state_size, dtype=torch.float32)
        * 0.012
    )
    l_state = (
        torch.randn(batch, config.l_hidden, model.l_rnn.state_size, dtype=torch.float32)
        * 0.012
    )
    case = {
        "token_ids": token_ids,
        "previous_context": previous_context.flatten().tolist(),
        "target_context": target_context.flatten().tolist(),
        "context_alpha": 0.375,
        "h_token_ids": token_ids,
        "l_token_ids": token_ids,
        "h_initial_packed_state": h_state.flatten().tolist(),
        "l_initial_packed_state": l_state.flatten().tolist(),
        "h_to_context_grad": (torch.randn(batch, config.context_dim) * 0.006)
        .flatten()
        .tolist(),
        "h_depth_grad": [0.004, -0.003],
        "final_drift_grad": (torch.randn(batch, config.context_dim) * 0.005)
        .flatten()
        .tolist(),
        "commitment_cost_grad": [0.08, 0.05],
        "targets": [5, 9],
        "optimizer": {
            "lr": 3.0e-4,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-8,
            "weight_decay": 0.0,
        },
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-raw-token-graph-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        trained_package = temp / "trained-model"
        trained_model = trained_package / "model.safetensors"
        case_path = temp / "case.json"
        export_model(model, config, model_dir)
        case_path.write_text(json.dumps(case), encoding="utf-8")
        vulkan_env = os.environ.copy()
        vulkan_env[TRAINING_PRECISION_ENV] = args.precision
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
                    "hierarchos-vulkan-raw-token-training-graph-smoke",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(case_path),
                    "--trained-model",
                    str(trained_model),
                ],
                env=vulkan_env,
            ).stdout
        )

        original_state = load_file(str(model_dir / "model.safetensors"))
        trained_state = load_file(str(trained_model))
        reloaded, reloaded_config = load_full_model_with_config(
            str(trained_package), torch.device("cpu")
        )
        if float(trained_state["memory_gate_warmup_step"].item()) != 7.0:
            raise AssertionError("trained package did not persist memory_gate_warmup_step=7")
        if not torch.equal(trained_state["ltm.fast_vals"], original_state["ltm.fast_vals"]):
            raise AssertionError("full-model export changed untouched ltm.fast_vals")
        if torch.equal(trained_state["in_proj.weight"], original_state["in_proj.weight"]):
            raise AssertionError("full-model export omitted trained in_proj.weight")
        if torch.equal(trained_state["val_proj.weight"], original_state["val_proj.weight"]):
            raise AssertionError("full-model export omitted trained val_proj.weight")
        manifest_path = trained_package / "training_state.json"
        optimizer_path = trained_package / "optimizer.safetensors"
        if not manifest_path.is_file() or not optimizer_path.is_file():
            raise AssertionError("Vulkan package export omitted portable training state")
        training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if training_manifest["optimizer_file"] != "optimizer.safetensors":
            raise AssertionError(f"unexpected Vulkan optimizer member: {training_manifest}")
        replay_state = read_vulkan_training_replay(trained_package, training_manifest)
        if replay_state is None:
            raise AssertionError("native Vulkan export omitted its portable replay sidecar")
        native_session = training_manifest.get("training_session")
        if not isinstance(native_session, dict):
            raise AssertionError("native Vulkan export omitted typed training session state")
        if native_session.get("effective_training_config") != {"training_backend": "vulkan"}:
            raise AssertionError("native Vulkan session lost its effective trajectory config")
        if native_session.get("ltm_lr_scheduler", {}).get("step") != 1:
            raise AssertionError("native Vulkan session lost LTM scheduler progress")
        native_last_lrs = native_session.get("main_lr_scheduler", {}).get("last_lrs")
        if (
            not isinstance(native_last_lrs, list)
            or len(native_last_lrs) != 1
            or abs(float(native_last_lrs[0]) - case["optimizer"]["lr"] * 0.75) > 1.0e-10
        ):
            raise AssertionError("native Vulkan session lost the live next-update LR")
        raw_replay_document = json.loads(
            (trained_package / "training_replay.json").read_text(encoding="utf-8")
        )
        if "effective_training_config" in json.dumps(raw_replay_document):
            raise AssertionError(
                "native Vulkan trajectory config leaked back into host-only replay state"
            )
        probe = replay_state.get("native_replay_probe")
        if not torch.is_tensor(probe):
            raise AssertionError("native Vulkan replay tensor did not decode in the PyTorch bridge")
        torch.testing.assert_close(
            probe,
            torch.tensor([0.125, -0.25], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
        rng_probe = replay_state.get("native_rng_probe")
        if not torch.is_tensor(rng_probe) or rng_probe.dtype != torch.uint8:
            raise AssertionError("native Vulkan U8 replay tensor did not decode as torch.uint8")
        if not torch.equal(rng_probe, torch.tensor([1, 2, 3, 255], dtype=torch.uint8)):
            raise AssertionError("native Vulkan U8 replay tensor did not round-trip exactly")
        if replay_state.get("effective_training_config") != {"training_backend": "vulkan"}:
            raise AssertionError("native Vulkan replay JSON state did not round-trip through Python")
        controller = training_manifest["ltm_alignment_controller"]
        controller_config_fields = {
            "val_proj_alignment_updates": "updates",
            "val_proj_alignment_last": "last",
            "val_proj_alignment_ema": "ema",
            "val_proj_alignment_best": "best",
            "val_proj_writer_norm": "writer_norm",
            "val_proj_trained": "ready",
        }
        for config_name, controller_name in controller_config_fields.items():
            config_value = reloaded_config.get(config_name)
            controller_value = controller.get(controller_name)
            if isinstance(config_value, float) and isinstance(controller_value, float):
                matches = torch.tensor(config_value, dtype=torch.float32).item() == torch.tensor(
                    controller_value, dtype=torch.float32
                ).item()
            else:
                matches = config_value == controller_value
            if not matches:
                raise AssertionError(
                    "PyTorch config/controller checkpoint state diverged: "
                    f"{config_name}={config_value!r} "
                    f"manifest.{controller_name}={controller_value!r}"
                )
        # Build the continuation optimizer through the same grouping policy as
        # production PyTorch training. A flat AdamW group silently applies
        # PyTorch's default weight_decay=0.01 and misclassifies norms/vectors,
        # which makes a valid Vulkan no-decay checkpoint impossible to resume.
        optimizer_args = argparse.Namespace(
            starting_lr=case["optimizer"]["lr"],
            rwkv_weight_decay=case["optimizer"]["weight_decay"],
            adamw_eps=case["optimizer"]["eps"],
            _optimizer_grouping_version=2,
        )
        pytorch_optimizer = build_hierarchos_optimizer(
            reloaded,
            optimizer_args,
            torch.device("cpu"),
        )
        bridged_optimizer = load_vulkan_adamw_into_torch(
            reloaded,
            pytorch_optimizer,
            optimizer_path,
        )
        if bridged_optimizer.step != training_manifest["optimizer_step"]:
            raise AssertionError(
                "PyTorch optimizer bridge step disagrees with Vulkan training manifest"
            )
        if "lm_head.weight" not in bridged_optimizer.slot_names:
            raise AssertionError("full optimizer bridge lost canonical tied lm_head.weight")

        inference_tokens = [2, 5, 2]
        input_ids = torch.tensor([inference_tokens], dtype=torch.long)
        # The native runtime seeds each new RuntimeState from the checkpoint's
        # ltm.fast_vals. Do not call reset_memory() here: PyTorch's reset clears
        # that checkpoint state and would compare different initial memories.
        with torch.no_grad():
            expected_logits = reloaded(
                input_ids,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )["logits"][0].float().cpu()

        native_result = json.loads(
            _run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--release",
                    "--manifest-path",
                    str(ROOT / "hierarchos-inference" / "Cargo.toml"),
                    "--bin",
                    "hierarchos-infer",
                    "--",
                    "--model",
                    str(trained_package),
                    "--tokens",
                    ",".join(map(str, inference_tokens)),
                ]
            ).stdout
        )
        if native_result["tokens"] != inference_tokens or len(native_result["logits"]) != 3:
            raise AssertionError("native Rust inference failed to consume Vulkan-trained package")
        native_logits = torch.tensor(native_result["logits"], dtype=torch.float32)
        torch.testing.assert_close(native_logits, expected_logits, rtol=2.0e-4, atol=2.0e-5)
        native_vs_pytorch_max_abs = (native_logits - expected_logits).abs().max().item()

        if torch.cuda.is_available():
            cuda_model, cuda_config = load_full_model_with_config(
                str(trained_package), torch.device("cuda")
            )
            if dict(cuda_config) != dict(reloaded_config):
                raise AssertionError("CUDA and CPU package configs diverged")
            with torch.no_grad():
                cuda_logits = cuda_model(
                    input_ids.cuda(),
                    return_topk_values=False,
                    return_raw_topk_values=False,
                    return_topk_indices=False,
                    return_step_telemetry=False,
                    return_numerics=False,
                )["logits"][0].float().cpu()
            torch.testing.assert_close(cuda_logits, expected_logits, rtol=3.0e-4, atol=3.0e-5)
            cuda_status = "PASS"
        else:
            cuda_status = "SKIP(no CUDA device present)"

    if result["training_precision_policy"] != args.precision:
        raise AssertionError(
            "Vulkan training precision policy mismatch: "
            f"requested={args.precision!r} actual={result['training_precision_policy']!r}"
        )
    expects_fp16_parameter_storage = args.precision != "fp32"
    for tower in ("h", "l"):
        field = f"{tower}_low_rank_fp16_parameter_storage_active"
        if bool(result[field]) != expects_fp16_parameter_storage:
            raise AssertionError(
                f"Vulkan precision consumer mismatch: {field}={result[field]!r} "
                f"for requested precision {args.precision!r}"
            )
    if bool(result["projection_fp16_parameter_storage_active"]) != expects_fp16_parameter_storage:
        raise AssertionError(
            "Vulkan precision consumer mismatch: "
            f"projection_fp16_parameter_storage_active={result['projection_fp16_parameter_storage_active']!r} "
            f"for requested precision {args.precision!r}"
        )
    if bool(result["lm_head_fp16_parameter_storage_active"]) != expects_fp16_parameter_storage:
        raise AssertionError(
            "Vulkan precision consumer mismatch: "
            f"lm_head_fp16_parameter_storage_active={result['lm_head_fp16_parameter_storage_active']!r} "
            f"for requested precision {args.precision!r}"
        )
    expects_native_fp16_lm_backward = (
        args.precision == "fp16-storage-fp16-lm-backward"
    )
    if (
        bool(result["lm_head_native_fp16_backward_compute_active"])
        != expects_native_fp16_lm_backward
    ):
        raise AssertionError(
            "native-FP16 LM backward mode mismatch: "
            f"active={result['lm_head_native_fp16_backward_compute_active']!r} "
            f"for precision {args.precision!r}"
        )
    lm_execution_arm = result["lm_head_execution_arm"]
    expected_lm_arms = (
        {
            "fp16-packed",
            "fp16-ce-tape",
            "fp16-ce-tape-rows8",
            "fp16-ce-tape-rows16",
            "fp16-ce-tape-rows16-dot4",
            "fp16-ce-tape-rows16-fused-adjoints",
            "fp16-ce-tape-rows16-dot4-fused-adjoints",
            "fp16-ce-tape-rows16-cluster4-fused-adjoints",
            "fp16-native",
            "fp16-native-reuse64",
            "fp16-native-reuse128",
            "fp16-native-reuse224",
        }
        if expects_fp16_parameter_storage
        else {"fp32"}
    )
    if lm_execution_arm not in expected_lm_arms:
        raise AssertionError(
            "Vulkan LM execution-arm mismatch: "
            f"arm={lm_execution_arm!r} precision={args.precision!r}"
        )
    if expects_native_fp16_lm_backward and lm_execution_arm != "fp16-native":
        raise AssertionError(
            "native-FP16 LM backward currently requires fp16-native; "
            f"got {lm_execution_arm!r}"
        )
    lm_weight_grad_topology = result["lm_head_weight_grad_topology"]
    expected_lm_topologies = {"dw-vocab4", "dw-vocab8", "dw-vocab16"}
    if expects_fp16_parameter_storage:
        if lm_weight_grad_topology not in expected_lm_topologies:
            raise AssertionError(
                "Vulkan LM dW-topology mismatch: "
                f"topology={lm_weight_grad_topology!r} precision={args.precision!r}"
            )
    elif lm_weight_grad_topology is not None:
        raise AssertionError(
            f"FP32 run unexpectedly reported LM dW topology {lm_weight_grad_topology!r}"
        )
    fused_adjoint_topology = result["lm_head_fused_adjoint_topology"]
    fused_arms = {
        "fp16-ce-tape-rows16-fused-adjoints",
        "fp16-ce-tape-rows16-dot4-fused-adjoints",
        "fp16-ce-tape-rows16-cluster4-fused-adjoints",
    }
    if lm_execution_arm in fused_arms:
        if fused_adjoint_topology not in {
            "fused-shared-hidden",
            "fused-private-hidden",
            "fused-private-hidden-tile256-wg256",
        }:
            raise AssertionError(
                "Vulkan LM fused-adjoint topology mismatch: "
                f"topology={fused_adjoint_topology!r} arm={lm_execution_arm!r}"
            )
    elif fused_adjoint_topology is not None:
        raise AssertionError(
            "Vulkan non-fused LM arm unexpectedly reported a fused-adjoint topology: "
            f"{fused_adjoint_topology!r}"
        )
    required_frontend = {
        "persistent",
        "rosa_adapter.down.weight",
        "rosa_adapter.up.weight",
        "rosa_adapter.bias",
        "rosa_gate_logit",
        "rosa_router.weight",
        "rosa_router.bias",
        "qproj.weight",
        "val_proj.weight",
        "ltm.keys",
        "ltm.vals",
        "ltm_gate_logit",
        "ltm_router.weight",
        "ltm_router.bias",
        "in_proj.weight",
        "in_proj.bias",
    }
    names = result["raw_optimizer_names"]
    if not required_frontend.issubset(names):
        raise AssertionError(f"missing frontend registry names: {sorted(required_frontend - set(names))}")
    if names.count("lm_head.weight") != 1:
        raise AssertionError("raw graph must own exactly one canonical lm_head.weight optimizer slot")
    if result["raw_queue_submissions"] != 1:
        raise AssertionError(f"raw graph used {result['raw_queue_submissions']} queue submissions")
    if result["grad_previous_context_max_abs_diff"] > 2.0e-5:
        raise AssertionError(
            "raw graph dropped or distorted the frontend previous-context adjoint: "
            f"max_abs={result['grad_previous_context_max_abs_diff']:.9g}"
        )
    if result["batch"] != batch:
        raise AssertionError(f"raw graph reported batch {result['batch']} instead of {batch}")
    if not result["raw_optimizer_tensor_count"] == result["reference_optimizer_tensor_count"] + 16:
        raise AssertionError("raw graph optimizer registry did not add exactly 16 frontend tensors")
    if not result["frontend_moment_l1"] > 0.0:
        raise AssertionError("frontend gradients did not reach the canonical optimizer moments")
    if not result["val_proj_moment_l1"] > 0.0:
        raise AssertionError("LTM value-alignment gradient did not reach val_proj.weight moments")
    if result["training_step"] != 7 or result["checkpoint_warmup_step"] != 7.0:
        raise AssertionError("explicit Vulkan training-step schedule did not persist")
    if not result["checkpoint_in_proj_max_abs_delta"] > 0.0:
        raise AssertionError("trained Vulkan checkpoint omitted frontend parameter updates")
    if not result["checkpoint_val_proj_max_abs_delta"] > 0.0:
        raise AssertionError("trained Vulkan checkpoint omitted val_proj value-alignment update")
    if not result["checkpoint_fast_vals_exact"]:
        raise AssertionError("Vulkan checkpoint rewrite mutated untouched fast memory")
    if not result["training_checkpoint_manifest_roundtrip"]:
        raise AssertionError("portable Vulkan training checkpoint manifest did not round-trip")
    if not result["optimizer_checkpoint_roundtrip"]:
        raise AssertionError("full-model Vulkan optimizer checkpoint did not round-trip")
    if result["raw_tape_tokens"] != 3:
        raise AssertionError(f"raw token tape expected 3 tokens, got {result['raw_tape_tokens']}")
    if result["raw_tape_queue_submissions"] != 1 or result["raw_tape_optimizer_step"] != 1:
        raise AssertionError(
            "raw token tape must use one queue submission and one canonical AdamW step: "
            f"submissions={result['raw_tape_queue_submissions']} "
            f"step={result['raw_tape_optimizer_step']}"
        )
    expected_rosa = [[-1, -1], [-1, -1], [5, 3]]
    if result["raw_tape_rosa_predictions"] != expected_rosa:
        raise AssertionError(
            "raw token tape did not preserve discrete ROSA predictions across reverse replay: "
            f"expected={expected_rosa} actual={result['raw_tape_rosa_predictions']}"
        )
    if not result["raw_tape_losses_finite"]:
        raise AssertionError("raw token tape produced a non-finite loss")
    if not result["raw_tape_frontend_moment_l1"] > 0.0:
        raise AssertionError("raw token tape frontend gradients did not reach AdamW moments")

    print(
        f"device={result['device']} precision={result['training_precision_policy']} "
        f"h_fp16_packed={result['h_low_rank_fp16_parameter_storage_active']} "
        f"l_fp16_packed={result['l_low_rank_fp16_parameter_storage_active']} "
        f"projection_fp16_packed={result['projection_fp16_parameter_storage_active']} "
        f"lm_head_fp16_packed={result['lm_head_fp16_parameter_storage_active']} "
        f"lm_head_execution_arm={result['lm_head_execution_arm']} "
        f"lm_head_weight_grad_topology={result['lm_head_weight_grad_topology']} "
        f"lm_head_fused_adjoint_topology={result['lm_head_fused_adjoint_topology']} "
        f"width={args.width} batch={result['batch']}"
    )
    print(f"raw_queue_submissions={result['raw_queue_submissions']}")
    print(f"loss_abs_diff={result['loss_abs_diff']:.9g}")
    print(f"grad_enc_max_abs_diff={result['grad_enc_max_abs_diff']:.9g}")
    print(
        "grad_previous_context_max_abs_diff="
        f"{result['grad_previous_context_max_abs_diff']:.9g}"
    )
    print(f"h_output_max_abs_diff={result['h_output_max_abs_diff']:.9g}")
    print(f"l_output_max_abs_diff={result['l_output_max_abs_diff']:.9g}")
    print(
        "raw_token_tape="
        f"tokens={result['raw_tape_tokens']} "
        f"queue_submissions={result['raw_tape_queue_submissions']} "
        f"optimizer_step={result['raw_tape_optimizer_step']} "
        f"rosa={result['raw_tape_rosa_predictions']} "
        f"frontend_moment_l1={result['raw_tape_frontend_moment_l1']:.9g}"
    )
    print(
        "optimizer_tensor_count="
        f"{result['reference_optimizer_tensor_count']} -> {result['raw_optimizer_tensor_count']}"
    )
    print(f"frontend_moment_l1={result['frontend_moment_l1']:.9g}")
    print(f"val_proj_moment_l1={result['val_proj_moment_l1']:.9g}")
    print(f"checkpoint_in_proj_max_abs_delta={result['checkpoint_in_proj_max_abs_delta']:.9g}")
    print(f"checkpoint_val_proj_max_abs_delta={result['checkpoint_val_proj_max_abs_delta']:.9g}")
    print(f"training_step={result['training_step']}")
    print("training_checkpoint_package=PASS")
    print("full_optimizer_pytorch_bridge=PASS (val_proj.weight tracked)")
    print("pytorch_reload=PASS")
    print(f"native_vs_pytorch_max_abs={native_vs_pytorch_max_abs:.9g}")
    print("native_rust_inference_reload=PASS")
    print(f"cuda_runtime_check={cuda_status}")
    print("optimizer_checkpoint_roundtrip=PASS")
    print("Raw-token -> loss one-submit Vulkan training graph smoke: PASS")


if __name__ == "__main__":
    main()
