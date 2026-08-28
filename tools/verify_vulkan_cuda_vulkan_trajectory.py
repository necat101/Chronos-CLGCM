#!/usr/bin/env python3
"""Exercise repeated Vulkan <-> PyTorch training-state trajectory handoffs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore, load_full_model_with_config
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
from tools.vulkan_optimizer_bridge import (
    load_vulkan_training_package_into_torch,
    read_vulkan_adamw_checkpoint,
    read_vulkan_training_manifest,
    save_torch_adamw_as_vulkan,
    write_closed_torch_training_manifest_as_vulkan,
    write_vulkan_training_replay,
)

TRAINING_PRECISION_ENV = "HIERARCHOS_VULKAN_TRAINING_PRECISION"
VULKAN_PRECISION_CHOICES = (
    "fp32",
    "fp16-storage-fp32-compute",
    "fp16-storage-parity",
    "fp16-storage-fp16-lm-backward",
)


def _run(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=process_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _raw_case(model: HierarchosCore, config) -> dict[str, object]:
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
    return {
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


def _run_pytorch_objective_backward(
    model: HierarchosCore, trajectory_step: int
) -> dict[str, object]:
    """Run a real deterministic Hierarchos language-model training objective.

    The original trajectory probe assigned synthetic gradients directly to every
    parameter. That was useful for validating the portable AdamW/checkpoint
    mechanics, but it never exercised PyTorch autograd on the CUDA leg. This
    objective deliberately uses the ordinary Hierarchos forward path, including
    its auxiliary manager/commitment/LTM-alignment costs, and requires every
    unique trainable parameter to receive a finite gradient before the optimizer
    is allowed to advance.
    """

    vocab_size = int(model.config.vocab_size)
    if vocab_size < 8:
        raise AssertionError(
            f"trajectory autograd probe requires vocab_size >= 8, got {vocab_size}"
        )

    device = next(model.parameters()).device
    usable = vocab_size - 2

    def token(offset: int) -> int:
        return 2 + ((trajectory_step * 7 + offset * 5) % usable)

    input_ids = torch.tensor(
        [
            [token(0), token(1), token(2), token(3)],
            [token(4), token(5), token(6), token(7)],
        ],
        dtype=torch.long,
        device=device,
    )
    labels = input_ids.clone()
    outputs = model(
        input_ids,
        labels=labels,
        return_logits=False,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
        return_step_telemetry=False,
        return_numerics=False,
        compute_ltm_value_alignment=True,
    )

    lm_loss = outputs.get("loss")
    if not torch.is_tensor(lm_loss) or lm_loss.ndim != 0:
        raise AssertionError("PyTorch trajectory forward did not return a scalar LM loss")

    objective = lm_loss
    auxiliary_values: dict[str, float | None] = {}
    for name, weight in (
        ("ponder_cost", 1.0e-2),
        ("commitment_cost", 5.0e-2),
        ("ltm_value_alignment_cost", 2.0e-2),
    ):
        value = outputs.get(name)
        if value is None:
            auxiliary_values[name] = None
            continue
        if not torch.is_tensor(value) or value.ndim != 0:
            raise AssertionError(f"PyTorch trajectory {name} is not a scalar tensor")
        auxiliary_values[name] = float(value.detach().float().item())
        objective = objective + value * weight

    if not bool(torch.isfinite(objective.detach()).item()):
        raise AssertionError("PyTorch trajectory objective became non-finite")
    objective.backward()

    unique_parameters: list[tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        unique_parameters.append((name, parameter))

    missing = [name for name, parameter in unique_parameters if parameter.grad is None]
    if missing:
        raise AssertionError(
            "real PyTorch trajectory objective left trainable parameters without gradients: "
            + ", ".join(missing)
        )

    nonfinite = [
        name
        for name, parameter in unique_parameters
        if not bool(torch.isfinite(parameter.grad.detach()).all().item())
    ]
    if nonfinite:
        raise AssertionError(
            "real PyTorch trajectory objective produced non-finite gradients: "
            + ", ".join(nonfinite)
        )

    grad_max_abs = 0.0
    grad_l2_sq = 0.0
    for _, parameter in unique_parameters:
        grad = parameter.grad.detach().float()
        grad_max_abs = max(grad_max_abs, float(grad.abs().max().item()))
        grad_l2_sq += float(grad.square().sum().item())

    return {
        "loss": float(lm_loss.detach().float().item()),
        "objective": float(objective.detach().float().item()),
        "auxiliary": auxiliary_values,
        "parameter_count": len(unique_parameters),
        "gradient_parameter_count": len(unique_parameters) - len(missing),
        "grad_max_abs": grad_max_abs,
        "grad_l2": grad_l2_sq**0.5,
    }


def _checkpoint_max_abs_diff(lhs, rhs) -> tuple[float, float]:
    if lhs.step != rhs.step:
        raise AssertionError(f"optimizer outer step drifted: {lhs.step} != {rhs.step}")
    if lhs.slot_names != rhs.slot_names:
        raise AssertionError("optimizer slot registry changed across handoff")
    if lhs.slot_steps != rhs.slot_steps:
        raise AssertionError("optimizer per-slot steps changed across handoff")
    first_max = 0.0
    second_max = 0.0
    for name in lhs.slot_names:
        first_max = max(
            first_max,
            float((lhs.exp_avg[name] - rhs.exp_avg[name]).abs().max().item()),
        )
        second_max = max(
            second_max,
            float((lhs.exp_avg_sq[name] - rhs.exp_avg_sq[name]).abs().max().item()),
        )
    return first_max, second_max


def _model_parameter_snapshot(model: HierarchosCore) -> dict[str, torch.Tensor]:
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        named_parameters = model.named_parameters()
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in named_parameters
    }


def _model_gradient_snapshot(model: HierarchosCore) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing gradient for PyTorch parameter {name!r}")
        snapshot[name] = parameter.grad.detach().float().cpu().clone()
    return snapshot


def _parameter_snapshot_max_abs_diff(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]
) -> float:
    if lhs.keys() != rhs.keys():
        missing = sorted(lhs.keys() - rhs.keys())
        extra = sorted(rhs.keys() - lhs.keys())
        raise AssertionError(
            f"model parameter registry changed across handoff: missing={missing} extra={extra}"
        )
    maximum = 0.0
    for name in lhs:
        if lhs[name].shape != rhs[name].shape:
            raise AssertionError(f"model parameter shape changed for {name!r}")
        maximum = max(maximum, float((lhs[name] - rhs[name]).abs().max().item()))
    return maximum


def _safetensor_model_max_abs_diff(lhs: Path, rhs: Path) -> float:
    maximum = 0.0
    with safe_open(str(lhs), framework="pt", device="cpu") as lhs_handle, safe_open(
        str(rhs), framework="pt", device="cpu"
    ) as rhs_handle:
        lhs_keys = set(lhs_handle.keys())
        rhs_keys = set(rhs_handle.keys())
        if lhs_keys != rhs_keys:
            raise AssertionError(
                "model SafeTensors registry changed across PyTorch handoff: "
                f"missing={sorted(lhs_keys - rhs_keys)} extra={sorted(rhs_keys - lhs_keys)}"
            )
        for name in lhs_keys:
            lhs_tensor = lhs_handle.get_tensor(name)
            rhs_tensor = rhs_handle.get_tensor(name)
            if lhs_tensor.shape != rhs_tensor.shape or lhs_tensor.dtype != rhs_tensor.dtype:
                raise AssertionError(f"model SafeTensors metadata changed for {name!r}")
            if lhs_tensor.is_floating_point():
                maximum = max(
                    maximum,
                    float((lhs_tensor.float() - rhs_tensor.float()).abs().max().item()),
                )
            elif not torch.equal(lhs_tensor, rhs_tensor):
                raise AssertionError(f"non-floating model tensor changed for {name!r}")
    return maximum


def _new_optimizer(model: HierarchosCore, hyper) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(hyper["lr"]),
        betas=(float(hyper["beta1"]), float(hyper["beta2"])),
        eps=float(hyper["eps"]),
        weight_decay=float(hyper["weight_decay"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail instead of reporting SKIP when CUDA is unavailable",
    )
    parser.add_argument(
        "--cpu-fallback",
        action="store_true",
        help="exercise the identical cross-backend orchestration through PyTorch CPU when CUDA is unavailable",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="number of PyTorch -> Vulkan alternating step pairs after the initial Vulkan step (default: 3)",
    )
    parser.add_argument(
        "--precision",
        choices=VULKAN_PRECISION_CHOICES,
        default="fp32",
        help="Vulkan precision policy used on the initial Vulkan leg",
    )
    parser.add_argument(
        "--return-precision",
        choices=VULKAN_PRECISION_CHOICES,
        default=None,
        help=(
            "Vulkan precision policy used after each PyTorch handoff; defaults to --precision. "
            "Using a different value verifies destination-side execution-mirror rebuilding."
        ),
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.cpu_fallback:
        message = "Vulkan -> CUDA -> Vulkan trajectory: SKIP(no CUDA device present)"
        if args.require_cuda:
            raise RuntimeError(message)
        print(message)
        return

    torch.manual_seed(20260819)
    if cuda_available:
        torch.cuda.manual_seed_all(20260819)
    config = tiny_coherent_config(32)
    config.max_h_steps = max(1, config.max_h_steps)
    config.max_l_steps = max(1, config.max_l_steps)
    model = HierarchosCore(config).eval()
    _make_nontrivial_memory_fixture(model, config)
    case = _raw_case(model, config)
    hyper = case["optimizer"]
    return_precision = args.return_precision or args.precision
    initial_vulkan_env = {TRAINING_PRECISION_ENV: args.precision}
    return_vulkan_env = {TRAINING_PRECISION_ENV: return_precision}

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-cuda-vulkan-") as temp_dir:
        temp = Path(temp_dir)
        source_package = temp / "source"
        vulkan_package = temp / "after-vulkan"
        case_path = temp / "case.json"
        export_model(model, config, source_package)
        case_path.write_text(json.dumps(case), encoding="utf-8")

        first_vulkan = json.loads(
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
                    str(source_package),
                    "--case",
                    str(case_path),
                    "--trained-model",
                    str(vulkan_package / "model.safetensors"),
                ],
                env=initial_vulkan_env,
            ).stdout
        )
        if first_vulkan["training_precision_policy"] != args.precision:
            raise AssertionError(
                "initial Vulkan leg did not honor requested precision: "
                f"{first_vulkan['training_precision_policy']!r} != {args.precision!r}"
            )
        native_fp16_lm_backward_policies = {
            "fp16-storage-parity",
            "fp16-storage-fp16-lm-backward",
        }
        expected_native_fp16_lm_backward = (
            args.precision in native_fp16_lm_backward_policies
        )
        expected_return_native_fp16_lm_backward = (
            return_precision in native_fp16_lm_backward_policies
        )
        if (
            bool(first_vulkan["lm_head_native_fp16_backward_compute_active"])
            != expected_native_fp16_lm_backward
        ):
            raise AssertionError(
                "initial Vulkan leg reported the wrong native-FP16 LM backward state"
            )
        first_manifest = read_vulkan_training_manifest(vulkan_package)
        first_step = int(first_manifest["optimizer_step"])

        bridge_device = torch.device("cuda" if cuda_available else "cpu")
        backend_label = bridge_device.type
        current_package = vulkan_package
        current_step = first_step
        step_history = [first_step]
        parameter_handoff_max_abs = 0.0
        exp_avg_handoff_max_abs = 0.0
        exp_avg_sq_handoff_max_abs = 0.0
        pytorch_objective_losses: list[float] = []
        pytorch_objective_values: list[float] = []
        pytorch_grad_max_abs = 0.0
        pytorch_grad_l2_max = 0.0
        pytorch_parameter_count = 0
        pytorch_gradient_parameter_count = 0
        pytorch_cpu_oracle_loss_max_abs = 0.0
        pytorch_cpu_oracle_objective_max_abs = 0.0
        pytorch_cpu_oracle_grad_max_abs = 0.0

        for cycle in range(1, args.cycles + 1):
            pytorch_package = temp / f"after-{backend_label}-{cycle}"
            returned_package = temp / f"after-vulkan-{cycle + 1}"

            pytorch_model, pytorch_config = load_full_model_with_config(
                str(current_package), bridge_device
            )
            pytorch_model.train()
            optimizer = _new_optimizer(pytorch_model, hyper)
            loaded = load_vulkan_training_package_into_torch(
                pytorch_model,
                optimizer,
                current_package,
            )
            if loaded.optimizer.step != current_step:
                raise AssertionError(
                    f"PyTorch handoff did not restore optimizer step {current_step}"
                )

            # Measure Vulkan package -> PyTorch -> Vulkan-format state drift
            # before changing the trajectory. Model masters and both moment
            # tensors should survive this boundary bit-exactly.
            inbound_probe = temp / f"inbound-probe-{cycle}"
            export_model(pytorch_model, pytorch_config, inbound_probe)
            parameter_handoff_max_abs = max(
                parameter_handoff_max_abs,
                _safetensor_model_max_abs_diff(
                    current_package / "model.safetensors",
                    inbound_probe / "model.safetensors",
                ),
            )
            inbound_optimizer = save_torch_adamw_as_vulkan(
                pytorch_model,
                optimizer,
                inbound_probe / "optimizer.safetensors",
                template_checkpoint=current_package / "optimizer.safetensors",
            )
            inbound_checkpoint = read_vulkan_adamw_checkpoint(
                current_package / "optimizer.safetensors"
            )
            first_drift, second_drift = _checkpoint_max_abs_diff(
                inbound_checkpoint, inbound_optimizer
            )
            exp_avg_handoff_max_abs = max(exp_avg_handoff_max_abs, first_drift)
            exp_avg_sq_handoff_max_abs = max(exp_avg_sq_handoff_max_abs, second_drift)

            optimizer.zero_grad(set_to_none=True)
            pytorch_step = _run_pytorch_objective_backward(
                pytorch_model, current_step + 1
            )
            pytorch_objective_losses.append(float(pytorch_step["loss"]))
            pytorch_objective_values.append(float(pytorch_step["objective"]))
            pytorch_grad_max_abs = max(
                pytorch_grad_max_abs, float(pytorch_step["grad_max_abs"])
            )
            pytorch_grad_l2_max = max(
                pytorch_grad_l2_max, float(pytorch_step["grad_l2"])
            )
            pytorch_parameter_count = int(pytorch_step["parameter_count"])
            pytorch_gradient_parameter_count = int(
                pytorch_step["gradient_parameter_count"]
            )

            # A real CUDA leg needs a numerical oracle, not merely a finite-
            # gradient check. Re-run the exact same forward/backward from the
            # inbound package on CPU and expose the maximum device drift. On a
            # CPU fallback this is also an exact determinism check for the
            # shadow-oracle plumbing itself.
            primary_gradients = _model_gradient_snapshot(pytorch_model)
            cpu_oracle_model, _ = load_full_model_with_config(
                str(current_package), torch.device("cpu")
            )
            cpu_oracle_model.train()
            for parameter in cpu_oracle_model.parameters():
                parameter.grad = None
            cpu_oracle_step = _run_pytorch_objective_backward(
                cpu_oracle_model, current_step + 1
            )
            cpu_oracle_gradients = _model_gradient_snapshot(cpu_oracle_model)
            loss_drift = abs(
                float(pytorch_step["loss"]) - float(cpu_oracle_step["loss"])
            )
            objective_drift = abs(
                float(pytorch_step["objective"])
                - float(cpu_oracle_step["objective"])
            )
            gradient_drift = _parameter_snapshot_max_abs_diff(
                primary_gradients, cpu_oracle_gradients
            )
            pytorch_cpu_oracle_loss_max_abs = max(
                pytorch_cpu_oracle_loss_max_abs, loss_drift
            )
            pytorch_cpu_oracle_objective_max_abs = max(
                pytorch_cpu_oracle_objective_max_abs, objective_drift
            )
            pytorch_cpu_oracle_grad_max_abs = max(
                pytorch_cpu_oracle_grad_max_abs, gradient_drift
            )
            if bridge_device.type == "cpu" and (
                loss_drift != 0.0 or objective_drift != 0.0 or gradient_drift != 0.0
            ):
                raise AssertionError(
                    "CPU trajectory/autograd oracle is not deterministic: "
                    f"loss={loss_drift:.9g} objective={objective_drift:.9g} "
                    f"gradient={gradient_drift:.9g}"
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            in_memory_parameters = _model_parameter_snapshot(pytorch_model)

            export_model(pytorch_model, pytorch_config, pytorch_package)
            pytorch_optimizer = save_torch_adamw_as_vulkan(
                pytorch_model,
                optimizer,
                pytorch_package / "optimizer.safetensors",
                template_checkpoint=current_package / "optimizer.safetensors",
            )
            if pytorch_optimizer.step != current_step + 1:
                raise AssertionError(
                    f"PyTorch AdamW step did not advance exactly once: "
                    f"{current_step} -> {pytorch_optimizer.step}"
                )

            current_manifest = read_vulkan_training_manifest(current_package)
            pytorch_manifest = write_closed_torch_training_manifest_as_vulkan(
                pytorch_model,
                pytorch_package,
                current_manifest,
                pytorch_optimizer,
            )
            write_vulkan_training_replay(
                pytorch_package,
                {
                    "completed_epoch": int(current_manifest.get("completed_epoch") or 0),
                    "mid_epoch_step": 0,
                    "effective_training_config": {"training_backend": backend_label},
                },
            )

            pytorch_roundtrip = read_vulkan_adamw_checkpoint(
                pytorch_package / "optimizer.safetensors"
            )
            first_drift, second_drift = _checkpoint_max_abs_diff(
                pytorch_optimizer, pytorch_roundtrip
            )
            exp_avg_handoff_max_abs = max(exp_avg_handoff_max_abs, first_drift)
            exp_avg_sq_handoff_max_abs = max(exp_avg_sq_handoff_max_abs, second_drift)
            roundtrip_model, _ = load_full_model_with_config(
                str(pytorch_package), bridge_device
            )
            parameter_handoff_max_abs = max(
                parameter_handoff_max_abs,
                _parameter_snapshot_max_abs_diff(
                    in_memory_parameters, _model_parameter_snapshot(roundtrip_model)
                ),
            )
            step_history.append(pytorch_optimizer.step)

            returned = json.loads(
                _run(
                    [
                        "cargo",
                        "run",
                        "--quiet",
                        "--release",
                        "--manifest-path",
                        str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                        "--bin",
                        "hierarchos-vulkan-raw-token-resume-step",
                        "--",
                        "--model",
                        str(pytorch_package),
                        "--case",
                        str(case_path),
                        "--output-package",
                        str(returned_package),
                    ],
                    env=return_vulkan_env,
                ).stdout
            )
            if returned["training_precision_policy"] != return_precision:
                raise AssertionError(
                    "returning Vulkan leg did not honor destination precision policy: "
                    f"{returned['training_precision_policy']!r} != {return_precision!r}"
                )
            if (
                bool(returned["lm_head_native_fp16_backward_compute_active"])
                != expected_return_native_fp16_lm_backward
            ):
                raise AssertionError(
                    "returning Vulkan leg changed native-FP16 LM backward state"
                )
            if int(returned["optimizer_step_before"]) != current_step + 1:
                raise AssertionError(
                    "returning Vulkan graph did not restore the PyTorch optimizer step"
                )
            if int(returned["optimizer_step_after"]) != current_step + 2:
                raise AssertionError("returning Vulkan graph did not advance AdamW exactly once")

            returned_manifest = read_vulkan_training_manifest(returned_package)
            if int(returned_manifest["optimizer_step"]) != current_step + 2:
                raise AssertionError(
                    "Vulkan package manifest lost the backend-spanning trajectory"
                )
            if returned_manifest.get("training_precision_policy") != return_precision:
                raise AssertionError(
                    "returning Vulkan package did not record its destination runtime precision"
                )
            step_history.append(current_step + 2)
            current_package = returned_package
            current_step += 2

        final_manifest = read_vulkan_training_manifest(current_package)
        if int(final_manifest["optimizer_step"]) != current_step:
            raise AssertionError("final package optimizer step disagrees with trajectory")

        # The final Vulkan-written package must remain directly consumable on
        # NVIDIA for ordinary CUDA inference, not merely parse as optimizer state.
        final_cuda_model, final_config = load_full_model_with_config(
            str(current_package), bridge_device
        )
        final_cuda_model.eval()
        final_optimizer = _new_optimizer(final_cuda_model, hyper)
        final_loaded = load_vulkan_training_package_into_torch(
            final_cuda_model, final_optimizer, current_package
        )
        final_probe = temp / "final-pytorch-probe"
        export_model(final_cuda_model, final_config, final_probe)
        parameter_handoff_max_abs = max(
            parameter_handoff_max_abs,
            _safetensor_model_max_abs_diff(
                current_package / "model.safetensors",
                final_probe / "model.safetensors",
            ),
        )
        final_probe_optimizer = save_torch_adamw_as_vulkan(
            final_cuda_model,
            final_optimizer,
            final_probe / "optimizer.safetensors",
            template_checkpoint=current_package / "optimizer.safetensors",
        )
        first_drift, second_drift = _checkpoint_max_abs_diff(
            final_loaded.optimizer, final_probe_optimizer
        )
        exp_avg_handoff_max_abs = max(exp_avg_handoff_max_abs, first_drift)
        exp_avg_sq_handoff_max_abs = max(exp_avg_sq_handoff_max_abs, second_drift)
        inference_tokens = [2, 5, 2]
        input_ids = torch.tensor(
            [inference_tokens], dtype=torch.long, device=bridge_device
        )
        with torch.no_grad():
            logits = final_cuda_model(
                input_ids,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
            )["logits"]
        if not bool(torch.isfinite(logits).all().item()):
            raise AssertionError("final Vulkan package produced non-finite PyTorch inference logits")

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
                    str(current_package),
                    "--tokens",
                    ",".join(map(str, inference_tokens)),
                ]
            ).stdout
        )
        if native_result.get("tokens") != inference_tokens:
            raise AssertionError("native Rust inference consumed a different token sequence")
        native_logits = torch.tensor(native_result["logits"], dtype=torch.float32)
        pytorch_logits = logits[0].detach().float().cpu()
        torch.testing.assert_close(
            native_logits,
            pytorch_logits,
            rtol=3.0e-4,
            atol=3.0e-5,
        )
        native_vs_pytorch_max_abs = float(
            (native_logits - pytorch_logits).abs().max().item()
        )

    if cuda_available:
        print(f"nvidia_device={torch.cuda.get_device_name(0)}")
    else:
        print("bridge_device=cpu (CUDA orchestration fallback)")
    print(f"vulkan_device={first_vulkan['device']}")
    print(f"initial_vulkan_training_precision={args.precision}")
    print(f"return_vulkan_training_precision={return_precision}")
    print(
        "execution_mirror_destination_rebuild="
        + ("cross-precision" if return_precision != args.precision else "same-precision")
    )
    print(f"alternating_cycles={args.cycles}")
    print(f"optimizer_steps={'->'.join(str(step) for step in step_history)}")
    print(f"parameter_handoff_max_abs={parameter_handoff_max_abs:.9g}")
    print(f"exp_avg_handoff_max_abs={exp_avg_handoff_max_abs:.9g}")
    print(f"exp_avg_sq_handoff_max_abs={exp_avg_sq_handoff_max_abs:.9g}")
    print(
        "pytorch_objective_losses="
        + "->".join(f"{loss:.9g}" for loss in pytorch_objective_losses)
    )
    print(
        "pytorch_objective_values="
        + "->".join(f"{value:.9g}" for value in pytorch_objective_values)
    )
    print(
        "pytorch_autograd_coverage="
        f"{pytorch_gradient_parameter_count}/{pytorch_parameter_count}"
    )
    print(f"pytorch_grad_max_abs={pytorch_grad_max_abs:.9g}")
    print(f"pytorch_grad_l2_max={pytorch_grad_l2_max:.9g}")
    print(
        f"{bridge_device.type}_vs_cpu_autograd_loss_max_abs="
        f"{pytorch_cpu_oracle_loss_max_abs:.9g}"
    )
    print(
        f"{bridge_device.type}_vs_cpu_autograd_objective_max_abs="
        f"{pytorch_cpu_oracle_objective_max_abs:.9g}"
    )
    print(
        f"{bridge_device.type}_vs_cpu_autograd_gradient_max_abs="
        f"{pytorch_cpu_oracle_grad_max_abs:.9g}"
    )
    print(f"native_vs_{bridge_device.type}_max_abs={native_vs_pytorch_max_abs:.9g}")
    print("native_rust_inference=PASS")
    print(f"final_{bridge_device.type}_inference=PASS")
    if cuda_available:
        print("Vulkan -> CUDA -> Vulkan trajectory: PASS")
    else:
        print("Vulkan -> PyTorch CPU -> Vulkan trajectory: PASS (CUDA path structurally verified)")


if __name__ == "__main__":
    main()
