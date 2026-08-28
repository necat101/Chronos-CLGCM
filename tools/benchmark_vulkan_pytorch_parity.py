#!/usr/bin/env python3
"""Benchmark one parity-qualified Hierarchos training trajectory on Vulkan and PyTorch.

The native Vulkan leg stays completely PyTorch-free: this orchestration script
uses PyTorch only as the external numerical/performance oracle. Both legs start
from the same exported SafeTensors package and execute the same deterministic
labeled historical-TBPTT updates before the trained packages are compared.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import load_full_model_with_config
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config
from tools.verify_vulkan_labeled_sequence_parity import (
    TRAINING_PRECISION_ENV,
    UpdateFixture,
    _build_updates,
    _install_lm_execution_oracle,
    _max_abs,
    _native_inference,
    _optimizer,
    _pytorch_inference_logits,
    _rust_update_payload,
    _train_pytorch_update,
)
from tools.verify_vulkan_token_memory_frontend_parity import _make_nontrivial_memory_fixture
from tools.verify_vulkan_worker_refinement_loss_parity import (
    DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV,
    NATIVE_FP16_LOW_RANK_BACKWARD_ENV,
    NATIVE_FP16_OUT_NORM_BACKWARD_ENV,
    NATIVE_FP16_PROJECTION_BACKWARD_ENV,
    NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV,
    env_flag_enabled,
    install_fp16_execution_storage,
    install_native_fp16_backward_oracle,
)


PRECISION_CHOICES = (
    "fp32",
    "fp16-storage-fp32-compute",
    "fp16-storage-parity",
    "fp16-storage-fp16-lm-backward",
)


def _default_parameter_tolerance(precision: str) -> float:
    return 5.0e-4 if precision == "fp16-storage-fp16-lm-backward" else 2.0e-5


def _configure_pytorch_precision_oracle(
    model: torch.nn.Module,
    *,
    precision: str,
    dynamic_loss_scale: float | None,
) -> None:
    native_fp16_policy = precision == "fp16-storage-fp16-lm-backward"
    source_scaled_fp32_source_adjoint_guard = (
        native_fp16_policy and dynamic_loss_scale is not None
    )
    native_fp16_lm_input_grad = (
        native_fp16_policy
        and not source_scaled_fp32_source_adjoint_guard
        and not env_flag_enabled(DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV, default=False)
    )
    _install_lm_execution_oracle(
        model,
        precision,
        native_input_grad=native_fp16_lm_input_grad,
    )
    if not native_fp16_policy:
        return

    install_native_fp16_backward_oracle(
        model,
        include_out_norm=(
            not source_scaled_fp32_source_adjoint_guard
            and env_flag_enabled(NATIVE_FP16_OUT_NORM_BACKWARD_ENV, default=True)
        ),
        include_projections=(
            not source_scaled_fp32_source_adjoint_guard
            and env_flag_enabled(NATIVE_FP16_PROJECTION_BACKWARD_ENV, default=True)
        ),
        include_low_rank=env_flag_enabled(NATIVE_FP16_LOW_RANK_BACKWARD_ENV, default=True),
        include_recurrent_projection=(
            dynamic_loss_scale is not None
            and env_flag_enabled(NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV, default=False)
        ),
    )


def _objective(config: object) -> dict[str, float]:
    return {
        "z_loss_weight": float(config.z_loss_weight),
        "ponder_loss_weight": 0.013,
        "commitment_loss_weight": 0.37,
        "max_ce_loss_for_backward": 0.0,
        "max_ponder_cost_for_backward": 0.0,
        "max_commitment_cost_for_backward": 2.0,
    }


def _optimizer_case(args: argparse.Namespace) -> dict[str, float]:
    return {
        "lr": args.learning_rate,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": args.optimizer_eps,
        "weight_decay": args.weight_decay,
    }


def _fixture_to_device(fixture: UpdateFixture, device: torch.device) -> UpdateFixture:
    return replace(
        fixture,
        input_ids=fixture.input_ids.to(device),
        labels=fixture.labels.to(device),
        attention_mask=fixture.attention_mask.to(device),
        loss_weights=fixture.loss_weights.to(device),
        previous_context=fixture.previous_context.to(device),
        target_context=fixture.target_context.to(device),
        h_state=fixture.h_state.to(device),
        l_state=fixture.l_state.to(device),
    )


def _active_targets(fixture: UpdateFixture) -> int:
    labels = fixture.labels[:, 1:]
    mask = fixture.attention_mask[:, 1:] > 0
    return int(((labels != -100) & mask).sum().item())


def _build_benchmark_updates(
    model: torch.nn.Module,
    config: object,
    update_count: int,
    *,
    batch: int,
    tokens: int,
) -> list[UpdateFixture]:
    """Build deterministic parity fixtures at arbitrary batch/sequence geometry.

    The historical 2x6 fixture remains byte-for-byte unchanged so existing
    qualification numbers stay comparable. Larger shapes use a deterministic
    synthetic sequence generator whose masking, weighting, contexts, and packed
    recurrent states vary across optimizer updates without depending on random
    number generation.
    """

    if batch == 2 and tokens == 6:
        return _build_updates(model, config, update_count)
    if batch <= 0:
        raise ValueError("benchmark fixture batch must be positive")
    if tokens < 2:
        raise ValueError("benchmark fixture tokens must be at least 2")

    vocab_size = int(config.vocab_size)
    if vocab_size < 2:
        raise ValueError("benchmark fixture vocabulary must contain at least two tokens")

    positions = torch.arange(tokens, dtype=torch.long).unsqueeze(0)
    rows = torch.arange(batch, dtype=torch.long).unsqueeze(1)
    padding_span = max(1, min(tokens - 2, max(1, tokens // 4)))
    updates: list[UpdateFixture] = []

    for update_index in range(update_count):
        # Keep zero reserved while spreading active IDs across the complete
        # configured vocabulary. Large-vocabulary qualification therefore
        # exercises both low and high LM-head/embedding rows.
        input_ids = (
            (
                positions * 17
                + rows * 29
                + update_index * 43
                + (positions // 7) * 11
            )
            % (vocab_size - 1)
        ) + 1

        row_offsets = (torch.arange(batch, dtype=torch.long) + update_index) % (
            padding_span + 1
        )
        active_lengths = torch.clamp(tokens - row_offsets, min=2)
        attention_mask = (
            torch.arange(tokens, dtype=torch.long).unsqueeze(0)
            < active_lengths.unsqueeze(1)
        ).to(dtype=torch.float32)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if tokens >= 4 and update_index % 3 == 1:
            # Preserve the original oracle's interior ignored-label coverage at
            # scale while keeping the position inside every row's active span.
            ignored_row = update_index % batch
            ignored_col = max(1, tokens // 2)
            labels[ignored_row, ignored_col] = -100

        weight_phase = (
            rows.to(dtype=torch.float32) * 5.0
            + positions.to(dtype=torch.float32) * 3.0
            + float(update_index * 7)
        )
        loss_weights = 0.5 + torch.remainder(weight_phase, 11.0) * 0.125
        loss_weights *= 1.0 + 0.02 * float(update_index % 5)
        loss_weights[attention_mask == 0] = 0.0

        bounded_phase = update_index % 17 + 1
        context_base = 0.003 * bounded_phase
        previous_context = torch.arange(
            batch * int(config.context_dim), dtype=torch.float32
        ).reshape(batch, int(config.context_dim))
        previous_context = (previous_context + 1.0) * context_base
        target_context = previous_context * -0.35

        h_state = torch.arange(
            batch * int(config.h_hidden) * model.h_rnn.state_size,
            dtype=torch.float32,
        ).reshape(batch, int(config.h_hidden), model.h_rnn.state_size)
        h_state = (h_state + 1.0) * (0.0002 * bounded_phase)
        l_state = torch.arange(
            batch * int(config.l_hidden) * model.l_rnn.state_size,
            dtype=torch.float32,
        ).reshape(batch, int(config.l_hidden), model.l_rnn.state_size)
        l_state = (l_state + 1.0) * (-0.00015 * bounded_phase)

        updates.append(
            UpdateFixture(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                loss_weights=loss_weights,
                previous_context=previous_context,
                target_context=target_context,
                h_state=h_state,
                l_state=l_state,
            )
        )

    return updates


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute a median from an empty sample set")
    return float(statistics.median(values))


def _throughput_record(
    *,
    backend: str,
    device: str,
    step_ms: list[float],
    model_tokens_per_step: int,
    target_tokens_per_step: int,
    hourly_cost: float | None,
    memory_bytes: int | None,
    memory_metric: str | None,
) -> dict[str, object]:
    median_ms = _median(step_ms)
    model_tps = model_tokens_per_step / (median_ms / 1_000.0)
    target_tps = target_tokens_per_step / (median_ms / 1_000.0)
    return {
        "backend": backend,
        "device": device,
        "median_optimizer_step_ms": median_ms,
        "model_token_positions_per_second": model_tps,
        "supervised_target_positions_per_second": target_tps,
        "memory_bytes": memory_bytes,
        "memory_metric": memory_metric,
        "hourly_cost_usd": hourly_cost,
        "cost_per_billion_model_tokens_usd": (
            None
            if hourly_cost is None
            else hourly_cost * 1_000_000_000.0 / (model_tps * 3_600.0)
        ),
        "samples_ms": step_ms,
    }


def _parse_kernel_profile(stderr: str) -> dict[str, object] | None:
    total_dispatches = 0
    total_gpu_ms = 0.0
    sample_count = 0
    categories: dict[str, dict[str, float | int]] = {}
    shaders: dict[str, dict[str, float | int | str]] = {}

    def fields(line: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            parsed[key] = value.strip('"')
        return parsed

    for line in stderr.splitlines():
        if line.startswith("hierarchos_vulkan_kernel_profile "):
            values = fields(line)
            total_dispatches += int(values["dispatches"])
            total_gpu_ms += float(values["gpu_ms"])
            sample_count += 1
        elif line.startswith("hierarchos_vulkan_kernel_profile_category "):
            values = fields(line)
            category = values["category"]
            aggregate = categories.setdefault(category, {"dispatches": 0, "gpu_ms": 0.0})
            aggregate["dispatches"] = int(aggregate["dispatches"]) + int(values["dispatches"])
            aggregate["gpu_ms"] = float(aggregate["gpu_ms"]) + float(values["gpu_ms"])
        elif line.startswith("hierarchos_vulkan_kernel_profile_shader "):
            values = fields(line)
            shader = values["shader"]
            aggregate = shaders.setdefault(
                shader,
                {
                    "category": values["category"],
                    "dispatches": 0,
                    "gpu_ms": 0.0,
                },
            )
            aggregate["dispatches"] = int(aggregate["dispatches"]) + int(values["dispatches"])
            aggregate["gpu_ms"] = float(aggregate["gpu_ms"]) + float(values["gpu_ms"])

    if sample_count == 0:
        return None

    def ranked(values: dict[str, dict[str, float | int | str]], name_key: str) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for name, aggregate in values.items():
            gpu_ms = float(aggregate["gpu_ms"])
            item: dict[str, object] = {
                name_key: name,
                **aggregate,
                "pct_of_profiled_gpu_time": (
                    0.0 if total_gpu_ms == 0.0 else gpu_ms * 100.0 / total_gpu_ms
                ),
            }
            result.append(item)
        result.sort(key=lambda item: float(item["gpu_ms"]), reverse=True)
        return result

    return {
        "sample_count": sample_count,
        "dispatches": total_dispatches,
        "gpu_ms": total_gpu_ms,
        "categories": ranked(categories, "category"),
        # The runtime prints only its per-submission top-N shaders, so this list
        # is intentionally named as a reported-hot set rather than a complete
        # accounting of every shader dispatched in the trajectory.
        "reported_hot_shaders": ranked(shaders, "shader"),
    }


def _run_vulkan(
    *,
    model_dir: Path,
    case_path: Path,
    output_package: Path,
    device_index: int | None,
    precision: str,
    budgeted_windows: bool,
    sequence_microbatch_size: int | None,
    state_checkpoint_stride: int | None,
    profile_kernels: bool,
) -> dict[str, object]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "--release",
        "--manifest-path",
        str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
        "--bin",
        "hierarchos-vulkan-labeled-sequence-parity",
        "--",
        "--model",
        str(model_dir),
        "--case",
        str(case_path),
        "--output-package",
        str(output_package),
    ]
    if device_index is not None:
        command.extend(["--device-index", str(device_index)])
    if budgeted_windows:
        command.append("--budgeted-windows")
    if sequence_microbatch_size is not None:
        command.extend(
            [
                "--sequence-microbatch-size",
                str(sequence_microbatch_size),
                "--state-checkpoint-stride",
                str(state_checkpoint_stride),
            ]
        )
    env = os.environ.copy()
    env[TRAINING_PRECISION_ENV] = precision
    if profile_kernels:
        env["HIERARCHOS_VULKAN_PROFILE_KERNELS"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Vulkan benchmark failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}"
        )
    result = json.loads(completed.stdout)
    result["kernel_profile"] = _parse_kernel_profile(completed.stderr)
    return result


def _run_pytorch(
    *,
    initial_package: Path,
    updates: list[UpdateFixture],
    optimizer_case: dict[str, float],
    objective: dict[str, float],
    device: torch.device,
    warmup: int,
    grad_clip: float,
    precision: str,
    dynamic_loss_scale: float | None,
) -> tuple[torch.nn.Module, list[float], int | None]:
    # Reload through the canonical package boundary instead of deepcopying the
    # in-memory fixture. AttrDict deliberately aliases __dict__ to itself, and
    # Python deepcopy does not preserve that alias. Package reload is also the
    # stronger benchmark contract: it is exactly how a PyTorch CPU/CUDA worker
    # consumes weights produced for the native Vulkan path.
    model, config = load_full_model_with_config(str(initial_package), device)
    model = model.train()
    _configure_pytorch_precision_oracle(
        model,
        precision=precision,
        dynamic_loss_scale=dynamic_loss_scale,
    )
    fixtures = [_fixture_to_device(fixture, device) for fixture in updates]
    optimizer = _optimizer(model, optimizer_case)
    measured_ms: list[float] = []
    loss_scale = dynamic_loss_scale or 1.0
    gradient_divisor = dynamic_loss_scale

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for index, fixture in enumerate(fixtures):
        fp16_execution_masters = (
            install_fp16_execution_storage(model) if precision != "fp32" else None
        )
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _train_pytorch_update(
                model,
                optimizer,
                fixture,
                objective,
                config,
                grad_clip=grad_clip,
                loss_scale=loss_scale,
                gradient_divisor_before_step=gradient_divisor,
                fp16_execution_masters=fp16_execution_masters,
                host_diagnostics=False,
            )
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            _train_pytorch_update(
                model,
                optimizer,
                fixture,
                objective,
                config,
                grad_clip=grad_clip,
                loss_scale=loss_scale,
                gradient_divisor_before_step=gradient_divisor,
                fp16_execution_masters=fp16_execution_masters,
                host_diagnostics=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if index >= warmup:
            measured_ms.append(elapsed_ms)

    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return model, measured_ms, peak_memory_bytes


def _max_parameter_diff(
    pytorch_model: torch.nn.Module,
    vulkan_model: torch.nn.Module,
) -> tuple[str | None, float]:
    pytorch_state = pytorch_model.to("cpu").state_dict()
    vulkan_state = vulkan_model.to("cpu").state_dict()
    worst_name: str | None = None
    worst = 0.0
    for name, lhs in pytorch_state.items():
        rhs = vulkan_state.get(name)
        if rhs is None or not lhs.is_floating_point():
            continue
        diff = _max_abs(lhs.float(), rhs.float())
        if diff > worst:
            worst = diff
            worst_name = name
    return worst_name, worst


def _inference_reload_report(
    package: Path,
    require_cuda: bool,
    max_abs_diff: float,
) -> dict[str, object]:
    cpu_model, _ = load_full_model_with_config(str(package), torch.device("cpu"))
    cpu_logits = _pytorch_inference_logits(cpu_model, torch.device("cpu"))
    native_logits, native_payload = _native_inference(package)
    native_diff = _max_abs(native_logits, cpu_logits)
    if native_diff > max_abs_diff:
        raise RuntimeError(
            "native Rust inference reload parity failed: "
            f"max_abs_diff={native_diff:.9g} > {max_abs_diff:.9g}"
        )
    result: dict[str, object] = {
        "native_rust_status": "passed",
        "native_rust_max_abs_diff_vs_pytorch_cpu": native_diff,
        "native_architecture_contract_sha256": native_payload.get(
            "architecture_contract_sha256"
        ),
        "max_abs_diff_threshold": max_abs_diff,
    }
    if torch.cuda.is_available():
        cuda_model, _ = load_full_model_with_config(str(package), torch.device("cuda"))
        cuda_logits = _pytorch_inference_logits(cuda_model, torch.device("cuda"))
        cuda_diff = _max_abs(cuda_logits, cpu_logits)
        if cuda_diff > max_abs_diff:
            raise RuntimeError(
                "PyTorch CUDA inference reload parity failed: "
                f"max_abs_diff={cuda_diff:.9g} > {max_abs_diff:.9g}"
            )
        result.update(
            {
                "cuda_status": "passed",
                "cuda_max_abs_diff_vs_pytorch_cpu": cuda_diff,
            }
        )
    else:
        if require_cuda:
            raise RuntimeError("CUDA inference was required but torch.cuda.is_available() is false")
        result.update(
            {
                "cuda_status": "skipped-no-cuda-device",
                "cuda_max_abs_diff_vs_pytorch_cpu": None,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-width", type=int, default=32)
    parser.add_argument("--fixture-vocab", type=int, default=64)
    parser.add_argument(
        "--fixture-batch",
        type=int,
        default=2,
        help="training sequences per optimizer update (historical smoke default: 2)",
    )
    parser.add_argument(
        "--fixture-tokens",
        type=int,
        default=6,
        help=(
            "tokens per sequence; values above 6 exercise the scalable long-context "
            "parity fixture"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--optimizer-eps", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=0.07)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default="fp32",
        help=(
            "Vulkan/PyTorch trainable execution precision contract; FP32 remains the "
            "baseline while FP16 storage modes retain FP32 optimizer masters"
        ),
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        type=float,
        default=None,
        help="scale backward sources and unscale once before AdamW on both backends",
    )
    parser.add_argument("--budgeted-windows", action="store_true")
    parser.add_argument(
        "--profile-vulkan-kernels",
        action="store_true",
        help=(
            "enable Vulkan timestamp queries and attach aggregated category/hot-shader "
            "GPU time to the JSON report; profiled timings should not be treated as an "
            "unprofiled economics measurement"
        ),
    )
    parser.add_argument("--sequence-microbatch-size", type=int)
    parser.add_argument("--state-checkpoint-stride", type=int)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-inference-reload", action="store_true")
    parser.add_argument("--max-parameter-diff", type=float, default=None)
    parser.add_argument("--max-inference-diff", type=float, default=2.0e-5)
    parser.add_argument("--vulkan-usd-per-hour", type=float)
    parser.add_argument("--pytorch-cpu-usd-per-hour", type=float)
    parser.add_argument("--pytorch-cuda-usd-per-hour", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.fixture_width <= 0:
        parser.error("--fixture-width must be positive")
    if args.fixture_vocab < 20:
        parser.error("--fixture-vocab must be at least 20 for the deterministic fixture")
    if args.fixture_batch <= 0:
        parser.error("--fixture-batch must be positive")
    if args.fixture_tokens < 2:
        parser.error("--fixture-tokens must be at least 2")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    if not math.isfinite(args.max_inference_diff) or args.max_inference_diff < 0.0:
        parser.error("--max-inference-diff must be finite and non-negative")
    if args.dynamic_loss_scale is not None and (
        not math.isfinite(args.dynamic_loss_scale) or args.dynamic_loss_scale <= 0.0
    ):
        parser.error("--dynamic-loss-scale must be finite and positive")
    if args.max_parameter_diff is None:
        args.max_parameter_diff = _default_parameter_tolerance(args.precision)
    elif not math.isfinite(args.max_parameter_diff) or args.max_parameter_diff < 0.0:
        parser.error("--max-parameter-diff must be finite and non-negative")
    if (args.sequence_microbatch_size is None) != (args.state_checkpoint_stride is None):
        parser.error(
            "--sequence-microbatch-size and --state-checkpoint-stride must be supplied together"
        )
    if args.sequence_microbatch_size is not None and not args.budgeted_windows:
        parser.error("an explicit tape plan requires --budgeted-windows")
    for option in (
        "vulkan_usd_per_hour",
        "pytorch_cpu_usd_per_hour",
        "pytorch_cuda_usd_per_hour",
    ):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or value < 0.0):
            parser.error(f"--{option.replace('_', '-')} must be finite and non-negative")
    if args.require_cuda and args.skip_cuda:
        parser.error("--require-cuda and --skip-cuda are mutually exclusive")
    if (
        args.precision == "fp16-storage-fp16-lm-backward"
        and args.dynamic_loss_scale is None
    ):
        print(
            "WARNING: unscaled fp16-storage-fp16-lm-backward keeps native-FP16 "
            "projection parameter-gradient reductions enabled. Cancellation-sensitive "
            "dW can exceed the mixed-precision parity gate on some devices; use "
            "--dynamic-loss-scale (for example 1024) to retain aggressive FP16 "
            "execution while routing projection dW/db through the cancellation-safe "
            "FP32 source-scaled path.",
            file=sys.stderr,
        )

    torch.manual_seed(20260826)
    config = tiny_coherent_config(args.fixture_width)
    config.vocab_size = args.fixture_vocab
    # A long-sequence benchmark should qualify actual retained token history,
    # not repeatedly wrap the tiny fixture's eight-token ROSA history ring.
    config.rosa_max_context = max(int(config.rosa_max_context), args.fixture_tokens)
    from hierarchos import HierarchosCore

    model = HierarchosCore(config).train()
    _make_nontrivial_memory_fixture(model, config)
    total_updates = args.warmup + args.iterations
    updates = _build_benchmark_updates(
        model,
        config,
        total_updates,
        batch=args.fixture_batch,
        tokens=args.fixture_tokens,
    )
    optimizer_case = _optimizer_case(args)
    objective = _objective(config)
    batch = int(updates[0].input_ids.shape[0])
    tokens = int(updates[0].input_ids.shape[1])
    model_tokens_per_step = batch * tokens
    measured_targets = [
        _active_targets(fixture) for fixture in updates[args.warmup :]
    ]
    target_tokens_per_step = int(statistics.median(measured_targets))

    case = {
        "batch": batch,
        "tokens": tokens,
        "max_h_steps": config.max_h_steps,
        "max_l_steps": config.max_l_steps,
        "gradient_accumulation_steps": 1,
        "dynamic_loss_scale": args.dynamic_loss_scale,
        "grad_clip": args.grad_clip,
        "capture_pending_gradients": False,
        **_rust_update_payload(updates[0]),
        "additional_updates": [_rust_update_payload(fixture) for fixture in updates[1:]],
        "objective": objective,
        "optimizer": optimizer_case,
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-pytorch-bench-") as temp_dir:
        temp = Path(temp_dir)
        model_dir = temp / "model"
        case_path = temp / "case.json"
        vulkan_package = temp / "vulkan-trained"
        export_model(model, config, model_dir)
        case_path.write_text(json.dumps(case), encoding="utf-8")

        vulkan_result = _run_vulkan(
            model_dir=model_dir,
            case_path=case_path,
            output_package=vulkan_package,
            device_index=args.device_index,
            precision=args.precision,
            budgeted_windows=args.budgeted_windows,
            sequence_microbatch_size=args.sequence_microbatch_size,
            state_checkpoint_stride=args.state_checkpoint_stride,
            profile_kernels=args.profile_vulkan_kernels,
        )
        if vulkan_result["training_precision_policy"] != args.precision:
            raise RuntimeError(
                "Vulkan benchmark precision contract mismatch: "
                f"requested={args.precision!r} "
                f"actual={vulkan_result['training_precision_policy']!r}"
            )
        vulkan_samples = [float(value) for value in vulkan_result["optimizer_window_ms"]]
        if len(vulkan_samples) != total_updates:
            raise RuntimeError(
                "Vulkan benchmark did not return one optimizer timing per update: "
                f"expected {total_updates}, got {len(vulkan_samples)}"
            )
        measured_vulkan = vulkan_samples[args.warmup :]
        backends: list[dict[str, object]] = [
            _throughput_record(
                backend="vulkan-native",
                device=str(vulkan_result["device_name"]),
                step_ms=measured_vulkan,
                model_tokens_per_step=model_tokens_per_step,
                target_tokens_per_step=target_tokens_per_step,
                hourly_cost=args.vulkan_usd_per_hour,
                memory_bytes=int(vulkan_result["memory_reserved_bytes"]),
                memory_metric="vulkan-suballocator-reserved-after-training",
            )
        ]

        pytorch_reference_model = None
        if not args.skip_cpu:
            cpu_model, cpu_samples, _ = _run_pytorch(
                initial_package=model_dir,
                updates=updates,
                optimizer_case=optimizer_case,
                objective=objective,
                device=torch.device("cpu"),
                warmup=args.warmup,
                grad_clip=args.grad_clip,
                precision=args.precision,
                dynamic_loss_scale=args.dynamic_loss_scale,
            )
            pytorch_reference_model = cpu_model
            backends.append(
                _throughput_record(
                    backend="pytorch-cpu",
                    device="cpu",
                    step_ms=cpu_samples,
                    model_tokens_per_step=model_tokens_per_step,
                    target_tokens_per_step=target_tokens_per_step,
                    hourly_cost=args.pytorch_cpu_usd_per_hour,
                    memory_bytes=None,
                    memory_metric=None,
                )
            )

        if not args.skip_cuda:
            if torch.cuda.is_available():
                cuda_device = torch.device("cuda")
                cuda_model, cuda_samples, cuda_peak = _run_pytorch(
                    initial_package=model_dir,
                    updates=updates,
                    optimizer_case=optimizer_case,
                    objective=objective,
                    device=cuda_device,
                    warmup=args.warmup,
                    grad_clip=args.grad_clip,
                    precision=args.precision,
                    dynamic_loss_scale=args.dynamic_loss_scale,
                )
                if pytorch_reference_model is None:
                    pytorch_reference_model = cuda_model
                backends.append(
                    _throughput_record(
                        backend="pytorch-cuda",
                        device=torch.cuda.get_device_name(cuda_device),
                        step_ms=cuda_samples,
                        model_tokens_per_step=model_tokens_per_step,
                        target_tokens_per_step=target_tokens_per_step,
                        hourly_cost=args.pytorch_cuda_usd_per_hour,
                        memory_bytes=cuda_peak,
                        memory_metric="torch.cuda.max_memory_allocated",
                    )
                )
            elif args.require_cuda:
                raise RuntimeError("CUDA training benchmark was required but no CUDA device is available")

        if pytorch_reference_model is None:
            # Always retain one numerical oracle even when timing legs are skipped.
            pytorch_reference_model, _, _ = _run_pytorch(
                initial_package=model_dir,
                updates=updates,
                optimizer_case=optimizer_case,
                objective=objective,
                device=torch.device("cpu"),
                warmup=total_updates,
                grad_clip=args.grad_clip,
                precision=args.precision,
                dynamic_loss_scale=args.dynamic_loss_scale,
            )

        vulkan_model, _ = load_full_model_with_config(
            str(vulkan_package), torch.device("cpu")
        )
        worst_name, parameter_diff = _max_parameter_diff(
            pytorch_reference_model, vulkan_model
        )
        if parameter_diff > args.max_parameter_diff:
            raise RuntimeError(
                "Vulkan/PyTorch parameter parity failed: "
                f"{worst_name} max_abs_diff={parameter_diff:.9g} > "
                f"{args.max_parameter_diff:.9g}"
            )

        vulkan_tps = float(backends[0]["model_token_positions_per_second"])
        for backend in backends:
            backend["throughput_vs_vulkan"] = (
                float(backend["model_token_positions_per_second"]) / vulkan_tps
            )

        inference = (
            None
            if args.skip_inference_reload
            else _inference_reload_report(
                vulkan_package,
                args.require_cuda,
                args.max_inference_diff,
            )
        )
        report = {
            "schema": "hierarchos-cross-backend-training-benchmark-v1",
            "fixture": {
                "width": args.fixture_width,
                "vocab_size": args.fixture_vocab,
                "batch": batch,
                "tokens_per_sequence": tokens,
                "rosa_max_context": int(config.rosa_max_context),
                "scalable_geometry_fixture": not (batch == 2 and tokens == 6),
                "model_token_positions_per_step": model_tokens_per_step,
                "supervised_target_positions_per_step": target_tokens_per_step,
                "warmup_updates": args.warmup,
                "measured_updates": args.iterations,
            },
            "training_contract": {
                "precision": args.precision,
                "gradient_accumulation_steps": 1,
                "dynamic_loss_scale": args.dynamic_loss_scale,
                "grad_clip": args.grad_clip,
                "optimizer": optimizer_case,
                "pytorch_inside_vulkan_trainer": False,
                "portable_checkpoint": "SafeTensors FP32 canonical masters",
                "pytorch_precision_oracle": (
                    "fp32-reference"
                    if args.precision == "fp32"
                    else "fp16-rounded-execution-with-fp32-adamw-masters"
                ),
                "projection_parameter_gradient_policy": (
                    "fp32-cancellation-safe-source-scaled"
                    if (
                        args.precision == "fp16-storage-fp16-lm-backward"
                        and args.dynamic_loss_scale is not None
                    )
                    else (
                        "native-fp16"
                        if args.precision == "fp16-storage-fp16-lm-backward"
                        else "fp32"
                    )
                ),
                "optimizer_boundary_submissions_per_step": (
                    1 if args.dynamic_loss_scale is not None else 2
                ),
            },
            "parity": {
                "worst_parameter_name": worst_name,
                "max_abs_parameter_diff": parameter_diff,
                "threshold": args.max_parameter_diff,
                "status": "passed",
                "qualification_class": (
                    "fp32-baseline"
                    if args.precision == "fp32"
                    else "mixed-precision-training"
                ),
            },
            "inference_reload": inference,
            "vulkan_execution": {
                "device_index": vulkan_result["device_index"],
                "device_name": vulkan_result["device_name"],
                "queue_submissions": vulkan_result["queue_submissions"],
                "training_precision_policy": vulkan_result["training_precision_policy"],
                "dynamic_loss_scale": vulkan_result["dynamic_loss_scale"],
                "dynamic_loss_scale_after": vulkan_result["dynamic_loss_scale_after"],
                "dynamic_loss_scale_overflow_count": vulkan_result[
                    "dynamic_loss_scale_overflow_count"
                ],
                "h_low_rank_native_fp16_parameter_grad_compute_active": vulkan_result[
                    "h_low_rank_native_fp16_parameter_grad_compute_active"
                ],
                "l_low_rank_native_fp16_parameter_grad_compute_active": vulkan_result[
                    "l_low_rank_native_fp16_parameter_grad_compute_active"
                ],
                "h_low_rank_parameter_grad_arithmetic": vulkan_result[
                    "h_low_rank_parameter_grad_arithmetic"
                ],
                "l_low_rank_parameter_grad_arithmetic": vulkan_result[
                    "l_low_rank_parameter_grad_arithmetic"
                ],
                "memory_live_buffer_count": vulkan_result["memory_live_buffer_count"],
                "memory_live_buffer_bytes": vulkan_result["memory_live_buffer_bytes"],
                "memory_driver_allocation_count": vulkan_result[
                    "memory_driver_allocation_count"
                ],
                "memory_reserved_bytes": vulkan_result["memory_reserved_bytes"],
                "memory_max_driver_allocation_count": vulkan_result[
                    "memory_max_driver_allocation_count"
                ],
                "memory_budget_extension_supported": vulkan_result[
                    "memory_budget_extension_supported"
                ],
                "device_local_heap_size_bytes": vulkan_result[
                    "device_local_heap_size_bytes"
                ],
                "device_local_budget_bytes": vulkan_result[
                    "device_local_budget_bytes"
                ],
                "device_local_usage_bytes": vulkan_result[
                    "device_local_usage_bytes"
                ],
                "device_local_available_bytes": vulkan_result[
                    "device_local_available_bytes"
                ],
                "budgeted_plans": vulkan_result.get("budgeted_plans", []),
                "kernel_profile": vulkan_result.get("kernel_profile"),
            },
            "backends": backends,
        }

        payload = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()
