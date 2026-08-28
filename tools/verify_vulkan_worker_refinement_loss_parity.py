#!/usr/bin/env python3
"""PyTorch parity for the one-submit Vulkan worker-refinement/loss graph."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
import hierarchos.models.rwkv_cell as rwkv_cell_module
from hierarchos.models.act import hard_act_depth_straight_through, hard_act_selection
from hierarchos.models.core import _finite_clamp, _l2_norm_clamp
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config


LR = 3.0e-4
BETA1 = 0.9
BETA2 = 0.999
EPS = 1.0e-8
WEIGHT_DECAY = 0.0
TRAINING_PRECISION_ENV = "HIERARCHOS_VULKAN_TRAINING_PRECISION"
NATIVE_FP16_OUT_NORM_BACKWARD_ENV = "HIERARCHOS_VULKAN_NATIVE_FP16_OUT_NORM_BACKWARD"
NATIVE_FP16_PROJECTION_BACKWARD_ENV = "HIERARCHOS_VULKAN_NATIVE_FP16_PROJECTION_BACKWARD"
NATIVE_FP16_LOW_RANK_BACKWARD_ENV = "HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_BACKWARD"
NATIVE_FP16_LOW_RANK_PARAMETER_GRAD_ENV = (
    "HIERARCHOS_VULKAN_NATIVE_FP16_LOW_RANK_PARAMETER_GRAD"
)
NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD_ENV = (
    "HIERARCHOS_VULKAN_NATIVE_FP16_RECURRENT_PROJECTION_BACKWARD"
)
DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV = (
    "HIERARCHOS_VULKAN_DISABLE_NATIVE_FP16_LM_INPUT_GRAD"
)
SHADOW_STEPS = 3
H_STEPS = 4
ALPHA = 0.375
ACCUMULATION_REPEATS = int(
    os.environ.get("HIERARCHOS_VULKAN_ACCUMULATION_REPEATS", "1")
)


# Optional per-use low-rank dW trace shared with the labeled-sequence verifier.
# Each custom-autograd invocation contributes exactly one local parameter
# gradient before PyTorch's leaf accumulator combines repeated recurrent uses.
LOW_RANK_G2_BACKWARD_TRACE: list[tuple[str, torch.Tensor]] | None = None


def deepembed(model: HierarchosCore, token_ids: list[int], branch: str) -> torch.Tensor:
    ids = torch.tensor(token_ids, dtype=torch.long)
    tied = F.embedding(ids, model.lm_head.weight)
    adapter = (
        model.h_deepembed_adapter
        if branch == "h"
        else model.l_deepembed_adapter
    )
    return adapter(tied)


def fp16_storage_fp32_compute_weight(weight: torch.Tensor) -> torch.Tensor:
    """Round execution reads to FP16 while preserving the FP32 master gradient."""
    rounded = weight.to(torch.float16).to(torch.float32)
    return weight + (rounded - weight).detach()


class Fp16LmBackwardLinear(torch.autograd.Function):
    """FP32-forward/FP16-product LM backward oracle for the Vulkan AMP tranche."""

    VULKAN_DX_VOCAB_TILE = 64
    VULKAN_DW_ROW_CHUNK = 8

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        native_input_grad: bool,
    ) -> torch.Tensor:
        rounded_weight = weight.to(torch.float16).to(torch.float32)
        ctx.native_input_grad = bool(native_input_grad)
        ctx.save_for_backward(input, rounded_weight)
        return F.linear(input, rounded_weight)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        input, rounded_weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        native_input_grad = ctx.native_input_grad
        grad_half = grad_2d.to(torch.float16)
        weight_half = rounded_weight.to(torch.float16)
        input_half = input_2d.to(torch.float16)

        # Model the Vulkan shaders literally, including reduction topology.
        # dX walks vocabulary rows in 64-row tiles, accumulates each tile in
        # ascending vocabulary order, then adds completed tile sums in order.
        # Using torch.sum here is not equivalent: its CPU reduction tree can
        # change the sign of tiny near-cancelling adjoints that AdamW amplifies.
        grad_input_2d = torch.zeros(
            (grad_2d.shape[0], weight_half.shape[1]),
            dtype=torch.float32,
            device=grad_output.device,
        )
        for vocab_base in range(0, weight_half.shape[0], Fp16LmBackwardLinear.VULKAN_DX_VOCAB_TILE):
            vocab_end = min(
                vocab_base + Fp16LmBackwardLinear.VULKAN_DX_VOCAB_TILE,
                weight_half.shape[0],
            )
            tile_sum = torch.zeros_like(grad_input_2d)
            for out_col in range(vocab_base, vocab_end):
                if native_input_grad:
                    product = (
                        grad_half[:, out_col].unsqueeze(1)
                        * weight_half[out_col].unsqueeze(0)
                    ).to(torch.float32)
                else:
                    product = (
                        grad_2d[:, out_col].to(torch.float32).unsqueeze(1)
                        * rounded_weight[out_col].to(torch.float32).unsqueeze(0)
                    )
                tile_sum = tile_sum + product
            if vocab_base == 0:
                grad_input_2d = tile_sum
            else:
                grad_input_2d = grad_input_2d + tile_sum

        # dW uses the same fixed row order as the shader: rows are reduced in
        # chunks of eight, with each half product widened before FP32 addition.
        grad_weight = torch.zeros_like(rounded_weight, dtype=torch.float32)
        for row_base in range(0, grad_half.shape[0], Fp16LmBackwardLinear.VULKAN_DW_ROW_CHUNK):
            row_end = min(
                row_base + Fp16LmBackwardLinear.VULKAN_DW_ROW_CHUNK,
                grad_half.shape[0],
            )
            chunk_sum = torch.zeros_like(grad_weight)
            for row in range(row_base, row_end):
                product = (
                    grad_half[row].unsqueeze(1)
                    * input_half[row].unsqueeze(0)
                ).to(torch.float32)
                chunk_sum = chunk_sum + product
            if row_base == 0:
                grad_weight = chunk_sum
            else:
                grad_weight = grad_weight + chunk_sum

        grad_input = grad_input_2d.reshape_as(input)
        return grad_input, grad_weight, None


class Fp16BackwardLinear(torch.autograd.Function):
    """FP32 forward with native-FP16 projection parameter gradients."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input, weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_half = grad_2d.to(torch.float16)
        input_half = input_2d.to(torch.float16)
        grad_input = grad_output.to(torch.float32) @ weight.to(torch.float32)
        # Match `linear_weight_grad.comp` literally. Each output element walks
        # rows in ascending order (the shader stages them in 16-row tiles but
        # does not use a tree reduction), widening every FP16 product before
        # the FP32 accumulation. PyTorch's `sum(dim=0)` is free to choose a
        # different reduction tree and can change tiny near-cancelling dW signs.
        grad_weight = torch.zeros(
            (weight.shape[0], weight.shape[1]),
            dtype=torch.float32,
            device=grad_output.device,
        )
        grad_bias = (
            torch.zeros(weight.shape[0], dtype=torch.float32, device=grad_output.device)
            if ctx.has_bias
            else None
        )
        for row in range(grad_half.shape[0]):
            grad_weight = grad_weight + (
                grad_half[row].unsqueeze(1) * input_half[row].unsqueeze(0)
            ).to(torch.float32)
            if grad_bias is not None:
                grad_bias = grad_bias + grad_half[row].to(torch.float32)
        return grad_input, grad_weight, grad_bias


class Fp16InputGradLinear(torch.autograd.Function):
    """FP32 forward/dW with native-FP16 products only for upstream dX."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        return F.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input, weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        grad_half = grad_2d.to(torch.float16)
        weight_half = weight.to(torch.float16)

        # Match Vulkan literally: each dX product happens in half, then widens
        # into the canonical FP32 accumulation. dW remains the existing FP32
        # checkpoint-facing reduction.
        grad_input = (
            grad_half.unsqueeze(2) * weight_half.unsqueeze(0)
        ).to(torch.float32).sum(dim=1).reshape_as(input)
        grad_weight = grad_2d.to(torch.float32).transpose(0, 1) @ input_2d.to(
            torch.float32
        )
        return grad_input, grad_weight


class Fp16BackwardParameterMatmul(torch.autograd.Function):
    """Native-FP16 backward oracle for Hierarchos' row-vector `x @ parameter`."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        parameter: torch.Tensor,
        native_input_grad: bool,
        native_parameter_grad: bool,
        widened_parameter_grad_product: bool,
        compensated_parameter_grad_operands: bool,
        trace_name: str | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, parameter)
        ctx.native_input_grad = native_input_grad
        ctx.native_parameter_grad = native_parameter_grad
        ctx.widened_parameter_grad_product = widened_parameter_grad_product
        ctx.compensated_parameter_grad_operands = compensated_parameter_grad_operands
        ctx.trace_name = trace_name
        return input @ parameter

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        input, parameter = ctx.saved_tensors
        grad_2d_fp32 = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
        if ctx.native_input_grad:
            parameter_half = parameter.to(torch.float16)
            grad_half = grad_output.to(torch.float16)
            grad_2d = grad_half.reshape(-1, grad_half.shape[-1])
            grad_input = (
                grad_2d.unsqueeze(2) * parameter_half.transpose(0, 1).unsqueeze(0)
            ).to(torch.float32).sum(dim=1).reshape_as(input)
        else:
            # Parameter-gradient promotion is independent from dX promotion.
            # The Vulkan w2/a2/g2 path still consumes the FP16 execution mirror,
            # but widens it and performs dX in FP32. Reproduce ordinary PyTorch
            # matmul semantics here instead of accidentally promoting dX merely
            # because the same parameter also uses the experimental native dW.
            grad_input = (
                grad_output.to(torch.float32) @ parameter.to(torch.float32).transpose(0, 1)
            ).reshape_as(input)
        input_2d = input.reshape(-1, input.shape[-1])
        if ctx.native_parameter_grad:
            grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float16)
            # `parameter_matmul_weight_grad.comp` owns one parameter element per
            # invocation and accumulates rows strictly from 0..N. Reproduce that
            # scalar order exactly. The default oracle rounds each product in
            # FP16 before widening; the alternate widened-input oracle models
            # devices where the same Float16 SPIR-V multiply is observed to
            # retain the exact FP32 product of the two half-rounded operands.
            # A generic torch.sum reduction is not an exact oracle for
            # cancellation-heavy gradients because its tree is backend-dependent.
            input_half = input_2d.to(torch.float16)
            grad_parameter = torch.zeros(
                (input_2d.shape[1], grad_2d.shape[1]),
                dtype=torch.float32,
                device=grad_output.device,
            )
            for row in range(grad_2d.shape[0]):
                if ctx.compensated_parameter_grad_operands:
                    input_hi = input_half[row]
                    grad_hi = grad_2d[row]
                    input_lo = (
                        input_2d[row].to(torch.float32) - input_hi.to(torch.float32)
                    ).to(torch.float16)
                    grad_lo = (
                        grad_2d_fp32[row] - grad_hi.to(torch.float32)
                    ).to(torch.float16)
                    # This oracle models the observed AMD Vulkan arithmetic:
                    # Float16-typed products retain the exact FP32 product of
                    # their half-rounded operands before the FP32 accumulation.
                    product = (
                        input_hi.to(torch.float32).unsqueeze(1)
                        * grad_hi.to(torch.float32).unsqueeze(0)
                        + input_lo.to(torch.float32).unsqueeze(1)
                        * grad_hi.to(torch.float32).unsqueeze(0)
                        + input_hi.to(torch.float32).unsqueeze(1)
                        * grad_lo.to(torch.float32).unsqueeze(0)
                    )
                elif ctx.widened_parameter_grad_product:
                    product = input_half[row].to(torch.float32).unsqueeze(1) * grad_2d[
                        row
                    ].to(torch.float32).unsqueeze(0)
                else:
                    product = (
                        input_half[row].unsqueeze(1) * grad_2d[row].unsqueeze(0)
                    ).to(torch.float32)
                grad_parameter = grad_parameter + product
        else:
            grad_parameter = (
                input_2d.to(torch.float32).unsqueeze(2) * grad_2d_fp32.unsqueeze(1)
            ).sum(dim=0)
        if ctx.trace_name is not None and LOW_RANK_G2_BACKWARD_TRACE is not None:
            LOW_RANK_G2_BACKWARD_TRACE.append(
                (ctx.trace_name, grad_parameter.detach().float().cpu().clone())
            )
        return grad_input, grad_parameter, None, None, None, None, None


class Fp16BackwardLayerNorm(torch.autograd.Function):
    """Match Vulkan out_norm: FP16 products, FP32 stats/reductions/destinations."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mean = input.mean(dim=-1, keepdim=True)
        variance = ((input - mean) ** 2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        ctx.save_for_backward(input, weight, mean, rstd)
        return (input - mean) * rstd * weight + bias

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        input, weight, mean, rstd = ctx.saved_tensors
        grad_half = grad_output.to(torch.float16)
        weight_half = weight.to(torch.float16)
        xhat = (input - mean) * rstd

        dxhat = (grad_half * weight_half).to(torch.float32)
        mean_dxhat = dxhat.mean(dim=-1, keepdim=True)
        mean_dxhat_xhat = (dxhat * xhat).mean(dim=-1, keepdim=True)
        grad_input = rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)

        grad_weight = (
            grad_half * xhat.to(torch.float16)
        ).to(torch.float32).sum(dim=0)
        grad_bias = grad_half.to(torch.float32).sum(dim=0)
        return grad_input, grad_weight, grad_bias, None


def env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return False
    if value in ("1", "true", "on", "yes"):
        return True
    raise ValueError(f"{name} must be a boolean flag; got {value!r}")


def install_native_fp16_backward_oracle(
    model: HierarchosCore,
    *,
    include_out_norm: bool,
    include_projections: bool,
    include_low_rank: bool,
    include_low_rank_inter_stage: bool = False,
    include_low_rank_parameter_grad: bool = False,
    low_rank_parameter_grad_product_semantics: str = "fp16-product",
    include_recurrent_projection: bool = False,
) -> None:
    """Patch only the graph edges Vulkan has promoted to native FP16 backward."""

    if low_rank_parameter_grad_product_semantics not in (
        "fp16-product",
        "widened-fp16-inputs",
        "compensated-widened-fp16-inputs",
    ):
        raise ValueError(
            "low_rank_parameter_grad_product_semantics must be "
            "'fp16-product', 'widened-fp16-inputs', or "
            "'compensated-widened-fp16-inputs'; got "
            f"{low_rank_parameter_grad_product_semantics!r}"
        )

    def linear_forward(module: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
        return Fp16BackwardLinear.apply(input, module.weight, module.bias)

    if include_projections:
        for projection in (
            model.l_feedback_proj,
            model.h_to_context,
            model.h_halt_proj,
            model.l_input_proj,
            model.context_drift_proj,
            model.l_to_out,
        ):
            projection.forward = types.MethodType(linear_forward, projection)

    def out_norm_forward(
        module: torch.nn.LayerNorm, input: torch.Tensor
    ) -> torch.Tensor:
        return Fp16BackwardLayerNorm.apply(
            input,
            module.weight,
            module.bias,
            module.eps,
        )

    if include_out_norm:
        model.out_norm.forward = types.MethodType(out_norm_forward, model.out_norm)

    if include_low_rank:
        low_rank_input_grad_names = ["w1", "a1", "g1"]
        if include_low_rank_inter_stage:
            low_rank_input_grad_names.extend(("w2", "a2", "g2"))
        promoted_low_rank_input_grad_ids = {
            id(getattr(recurrent, name))
            for recurrent in (model.h_rnn, model.l_rnn)
            for name in low_rank_input_grad_names
        }
        promoted_low_rank_parameter_grad_ids = (
            {
                id(getattr(recurrent, name))
                for recurrent in (model.h_rnn, model.l_rnn)
                for name in ("w1", "w2", "a1", "a2", "g1", "g2")
            }
            if include_low_rank_parameter_grad
            else set()
        )
        g2_trace_names = {
            id(recurrent.g2): f"{tower}_rnn.g2"
            for tower, recurrent in (("h", model.h_rnn), ("l", model.l_rnn))
        }
        original_parameter_matmul = rwkv_cell_module._parameter_matmul

        def native_fp16_parameter_matmul(
            left: torch.Tensor, parameter: torch.Tensor
        ) -> torch.Tensor:
            parameter_id = id(parameter)
            native_input_grad = parameter_id in promoted_low_rank_input_grad_ids
            native_parameter_grad = parameter_id in promoted_low_rank_parameter_grad_ids
            if native_input_grad or native_parameter_grad:
                return Fp16BackwardParameterMatmul.apply(
                    left,
                    parameter,
                    native_input_grad,
                    native_parameter_grad,
                    low_rank_parameter_grad_product_semantics
                    == "widened-fp16-inputs",
                    low_rank_parameter_grad_product_semantics
                    == "compensated-widened-fp16-inputs",
                    g2_trace_names.get(parameter_id),
                )
            return original_parameter_matmul(left, parameter)

        rwkv_cell_module._parameter_matmul = native_fp16_parameter_matmul

    if include_recurrent_projection:
        def recurrent_projection_forward(
            module: torch.nn.Linear, input: torch.Tensor
        ) -> torch.Tensor:
            return Fp16InputGradLinear.apply(input, module.weight)

        for recurrent in (model.h_rnn, model.l_rnn):
            for projection in (recurrent.receptance, recurrent.key, recurrent.value):
                projection.forward = types.MethodType(recurrent_projection_forward, projection)


def install_fp16_execution_storage(
    model: HierarchosCore,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Install rounded execution values while retaining restorable FP32 masters."""
    masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for recurrent in (model.h_rnn, model.l_rnn):
        for name in ("w1", "w2", "a1", "a2", "g1", "g2"):
            parameter = getattr(recurrent, name)
            master = parameter.detach().clone()
            masters.append((parameter, master))
            with torch.no_grad():
                parameter.copy_(master.to(torch.float16).to(torch.float32))
    for projection in (
        model.l_feedback_proj,
        model.h_to_context,
        model.h_halt_proj,
        model.l_input_proj,
        model.context_drift_proj,
        model.l_to_out,
    ):
        parameter = projection.weight
        master = parameter.detach().clone()
        masters.append((parameter, master))
        with torch.no_grad():
            parameter.copy_(master.to(torch.float16).to(torch.float32))
    return masters


def restore_fp32_masters(
    masters: list[tuple[torch.nn.Parameter, torch.Tensor]],
) -> None:
    with torch.no_grad():
        for parameter, master in masters:
            parameter.copy_(master)


def relevant_parameters(model: HierarchosCore) -> list[torch.nn.Parameter]:
    modules = [
        model.h_deepembed_adapter,
        model.h_rnn,
        model.l_deepembed_adapter,
        model.l_rnn,
        model.l_feedback_proj,
        model.h_to_context,
        model.h_halt_proj,
        model.l_input_proj,
        model.context_drift_proj,
        model.l_to_out,
        model.out_norm,
    ]
    values: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                values.append(parameter)
    if id(model.lm_head.weight) not in seen:
        values.append(model.lm_head.weight)
    return values


def local_recurrent_name(branch: str, rust_name: str) -> str:
    if rust_name.startswith("deepembed."):
        return f"{branch}_deepembed_adapter.{rust_name.removeprefix('deepembed.')}"
    return f"{branch}_rnn.{rust_name}"


def tensor_from_result(values: list[float], reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape_as(reference)


def assert_close(
    label: str,
    actual_values: list[float],
    expected: torch.Tensor,
    *,
    rtol: float = 4.0e-3,
    atol: float = 5.0e-5,
) -> float:
    actual = tensor_from_result(actual_values, expected)
    expected = expected.detach().float()
    abs_diff = (actual - expected).abs()
    diff = abs_diff.max().item() if expected.numel() else 0.0
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        if expected.numel():
            flat_index = int(abs_diff.flatten().argmax().item())
            actual_at_max = float(actual.flatten()[flat_index].item())
            expected_at_max = float(expected.flatten()[flat_index].item())
            detail = (
                f" flat_index={flat_index} actual={actual_at_max:.9g} "
                f"expected={expected_at_max:.9g}"
            )
        else:
            detail = ""
        raise AssertionError(
            f"{label} parity failed; max_abs_diff={diff:.9g}{detail}\n{exc}"
        ) from exc
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        choices=(
            "fp32",
            "fp16-storage-fp32-compute",
            "fp16-storage-parity",
            "fp16-storage-fp16-lm-backward",
        ),
        default="fp32",
        help="Vulkan trainable execution-storage precision arm",
    )
    args = parser.parse_args()
    if args.precision == "fp16-storage-parity":
        os.environ[DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV] = "1"
    if ACCUMULATION_REPEATS <= 0:
        raise ValueError("HIERARCHOS_VULKAN_ACCUMULATION_REPEATS must be positive")
    torch.manual_seed(20260826)
    config = tiny_coherent_config(32)
    if config.max_l_steps < SHADOW_STEPS:
        config.max_l_steps = SHADOW_STEPS
    if config.max_h_steps < H_STEPS:
        config.max_h_steps = H_STEPS
    # Deliberately make the graph-level clamps active. This turns the parity
    # case into a backward-mask test instead of only checking the in-range
    # identity path of `_finite_clamp`.
    config.activation_clamp = 0.12
    model = HierarchosCore(config).eval()

    batch = 2
    enc = (torch.randn(batch, config.context_dim) * 0.08).requires_grad_()
    previous = (torch.randn(batch, config.context_dim) * 0.06).requires_grad_()
    target_context = (torch.randn(batch, config.context_dim) * 0.07).requires_grad_()
    h_state = (
        torch.randn(batch, config.h_hidden, model.h_rnn.state_size) * 0.025
    ).requires_grad_()
    l_state = (
        torch.randn(batch, config.l_hidden, model.l_rnn.state_size) * 0.025
    ).requires_grad_()
    h_context_grad = torch.randn(batch, config.context_dim) * 0.013
    h_depth_grad = torch.tensor([0.019, -0.013], dtype=torch.float32)
    h_selected_state_grad = (
        torch.randn(batch, config.h_hidden, model.h_rnn.state_size) * 0.003
    )
    final_drift_grad = torch.randn(batch, config.context_dim) * 0.011
    commitment_cost_grad = torch.tensor([0.31, -0.17], dtype=torch.float32)
    l_final_state_grad = (
        torch.randn(batch, config.l_hidden, model.l_rnn.state_size) * 0.004
    )
    h_token_ids = [3, 4]
    l_token_ids = [5, 6]
    target_ids = [7, 8]

    # Force a genuinely row-local convergence case. Choose the tolerance
    # between the two rows' first drift-delta magnitudes so exactly one row
    # freezes after accepting candidate 0 while its peer keeps refining.
    with torch.no_grad():
        l_deep_probe = deepembed(model, l_token_ids, "l")
        static_probe = _finite_clamp(
            previous + ALPHA * (target_context - previous),
            config.context_state_clamp,
        )
        seed_probe = model.context_drift_proj(model.l_rnn.state_hidden(l_state))
        drift_probe = _l2_norm_clamp(
            _finite_clamp(torch.tanh(seed_probe), config.drift_state_clamp),
            config.drift_norm_clamp,
        )
        input_probe = _finite_clamp(
            model.l_input_proj(torch.cat([enc, static_probe + drift_probe], dim=-1)),
            config.recurrent_state_clamp,
        )
        output_probe, _ = model.l_rnn(
            input_probe,
            l_state,
            timestep=None,
            deepemb_vec=l_deep_probe,
        )
        output_probe = _finite_clamp(output_probe, config.activation_clamp)
        delta_probe = (
            torch.tanh(model.context_drift_proj(output_probe))
            * config.drift_delta_scale
        )
        first_magnitudes = torch.mean(torch.abs(delta_probe), dim=-1)
        low = float(first_magnitudes.min().item())
        high = float(first_magnitudes.max().item())
        if not high > low + 1.0e-8:
            raise AssertionError(
                f"probe rows did not separate convergence magnitudes: {first_magnitudes.tolist()}"
            )
        config.l_conv_atol = (low + high) * 0.5

    with tempfile.TemporaryDirectory(
        prefix="hierarchos-vulkan-worker-refinement-parity-"
    ) as temp_dir:
        model_dir = Path(temp_dir) / "model"
        # Rust must start from the exact pre-update PyTorch checkpoint.
        export_model(model, config, model_dir)

        fp32_masters: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        if args.precision != "fp32":
            fp32_masters = install_fp16_execution_storage(model)
        native_fp16_low_rank_backward = env_flag_enabled(
            NATIVE_FP16_LOW_RANK_BACKWARD_ENV, default=True
        )
        native_fp16_out_norm_backward = env_flag_enabled(
            NATIVE_FP16_OUT_NORM_BACKWARD_ENV, default=True
        )
        native_fp16_projection_backward = env_flag_enabled(
            NATIVE_FP16_PROJECTION_BACKWARD_ENV, default=True
        )
        if args.precision == "fp16-storage-fp16-lm-backward":
            install_native_fp16_backward_oracle(
                model,
                include_out_norm=native_fp16_out_norm_backward,
                include_projections=native_fp16_projection_backward,
                include_low_rank=native_fp16_low_rank_backward,
            )

        optimizer = torch.optim.AdamW(
            relevant_parameters(model),
            lr=LR,
            betas=(BETA1, BETA2),
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
        )
        optimizer.zero_grad(set_to_none=True)

        h_deep = deepembed(model, h_token_ids, "h")
        l_deep = deepembed(model, l_token_ids, "l")

        # Manager slice: all hard-ACT candidates share one clamped residual,
        # while later candidates recurrently advance an isolated shadow state.
        # Selected state receives a future-token adjoint to verify the manager
        # commitment edge in addition to selected-output and depth gradients.
        l_hidden = model.l_rnn.state_hidden(l_state)
        raw_h_input = enc + model.l_feedback_proj(l_hidden)
        h_input = _finite_clamp(
            raw_h_input,
            config.activation_clamp,
        )
        if not bool((raw_h_input.abs() > config.activation_clamp).any().item()):
            raise AssertionError(
                "activation-clamp parity case no longer exercises the saturated manager-input path"
            )
        h_step_outputs: list[torch.Tensor] = []
        h_step_states: list[torch.Tensor] = []
        h_halt_probabilities: list[torch.Tensor] = []
        shadow_h_state = h_state
        h_first_raw_output = None
        h_first_state = None
        for h_step in range(H_STEPS):
            raw_h_output, shadow_h_state = model.h_rnn(
                h_input,
                shadow_h_state,
                timestep=None,
                deepemb_vec=h_deep,
            )
            if h_step == 0:
                h_first_raw_output = raw_h_output
                h_first_state = shadow_h_state
            h_output = _finite_clamp(raw_h_output, config.activation_clamp)
            halt_logit = _finite_clamp(
                model.h_halt_proj(h_output).squeeze(-1),
                config.halt_logit_clamp,
            )
            h_step_outputs.append(h_output)
            h_step_states.append(shadow_h_state)
            h_halt_probabilities.append(torch.sigmoid(halt_logit).clamp(1e-6, 1.0 - 1e-6))
        h_stack = torch.stack(h_step_outputs, dim=0)
        h_state_stack = torch.stack(h_step_states, dim=0)
        h_halt_stack = torch.stack(h_halt_probabilities, dim=0)
        selection = hard_act_selection(
            h_stack,
            h_state_stack,
            h_halt_stack,
            threshold=config.h_halt_thresh,
            min_steps=min(config.min_h_steps, H_STEPS),
        )
        h_depth = hard_act_depth_straight_through(
            h_halt_stack,
            selection.executed_steps,
            threshold=config.h_halt_thresh,
            min_steps=min(config.min_h_steps, H_STEPS),
            temperature=config.act_depth_temperature,
        )
        h_context = _finite_clamp(
            model.h_to_context(selection.output),
            config.context_state_clamp,
        )
        manager_objective = (
            (h_context * h_context_grad).sum()
            + (h_depth * h_depth_grad).sum()
            + (selection.state * h_selected_state_grad).sum()
        ).sum()

        static_context = _finite_clamp(
            previous + ALPHA * (target_context - previous),
            config.context_state_clamp,
        )
        drift_seed = model.context_drift_proj(model.l_rnn.state_hidden(l_state))
        current_drift = _l2_norm_clamp(
            _finite_clamp(torch.tanh(drift_seed), config.drift_state_clamp),
            config.drift_norm_clamp,
        )

        shadow_state = l_state
        active = torch.ones(batch, dtype=enc.dtype)
        drift_cost_sum = torch.zeros(batch, dtype=enc.dtype)
        drift_cost_count = torch.zeros(batch, dtype=enc.dtype)
        shadow_outputs: list[torch.Tensor] = []
        for _ in range(SHADOW_STEPS):
            dynamic_context = static_context + current_drift
            l_input = _finite_clamp(
                model.l_input_proj(torch.cat([enc, dynamic_context], dim=-1)),
                config.recurrent_state_clamp,
            )
            previous_shadow = shadow_state
            previous_drift = current_drift
            active_rows = active > 0
            shadow_output, candidate_shadow = model.l_rnn(
                l_input, previous_shadow, timestep=None, deepemb_vec=l_deep
            )
            shadow_output = _finite_clamp(shadow_output, config.activation_clamp)
            shadow_outputs.append(shadow_output)
            drift_delta = (
                torch.tanh(model.context_drift_proj(shadow_output))
                * config.drift_delta_scale
            )
            candidate_drift = _l2_norm_clamp(
                _finite_clamp(
                    previous_drift + drift_delta,
                    config.drift_state_clamp,
                ),
                config.drift_norm_clamp,
            )
            if config.commitment_cost_mode == "mean-square":
                drift_sq = torch.mean(candidate_drift**2, dim=-1)
            else:
                drift_sq = torch.sum(candidate_drift**2, dim=-1)
            hinge_cost = torch.clamp(
                torch.relu(drift_sq - config.commitment_threshold),
                max=100.0,
            )
            drift_cost_sum = drift_cost_sum + torch.where(
                active_rows, hinge_cost, torch.zeros_like(hinge_cost)
            )
            drift_cost_count = drift_cost_count + active
            shadow_state = torch.where(
                active_rows[:, None, None], candidate_shadow, previous_shadow
            )
            current_drift = torch.where(
                active_rows[:, None], candidate_drift, previous_drift
            )
            still_active = (
                torch.mean(torch.abs(drift_delta), dim=-1) >= config.l_conv_atol
            ).to(enc.dtype)
            active = active * still_active

        commitment_cost = drift_cost_sum / torch.clamp(drift_cost_count, min=1.0)
        effective_l_steps = drift_cost_count + 1.0

        committed_input = _finite_clamp(
            model.l_input_proj(torch.cat([enc, static_context + current_drift], dim=-1)),
            config.recurrent_state_clamp,
        )
        committed_raw_output, committed_state = model.l_rnn(
            committed_input, l_state, timestep=None, deepemb_vec=l_deep
        )
        committed_output = _finite_clamp(committed_raw_output, config.activation_clamp)
        final_enc = _finite_clamp(
            enc + model.l_to_out(committed_output),
            config.activation_clamp,
        )
        raw_normalized_final_enc = model.out_norm(final_enc)
        if not bool(
            (raw_normalized_final_enc.detach().abs() > config.activation_clamp)
            .any()
            .item()
        ):
            raise AssertionError(
                "worker parity fixture did not activate the final out_norm clamp"
            )
        normalized_final_enc = _finite_clamp(
            raw_normalized_final_enc,
            config.activation_clamp,
        )
        if args.precision in ("fp16-storage-parity", "fp16-storage-fp16-lm-backward"):
            logits = Fp16LmBackwardLinear.apply(
                normalized_final_enc,
                model.lm_head.weight,
                not env_flag_enabled(
                    DISABLE_NATIVE_FP16_LM_INPUT_GRAD_ENV,
                    default=False,
                ),
            )
        elif args.precision == "fp16-storage-fp32-compute":
            logits = F.linear(
                normalized_final_enc,
                fp16_storage_fp32_compute_weight(model.lm_head.weight),
            )
        else:
            logits = model.lm_head(normalized_final_enc)
        ce_loss = F.cross_entropy(
            logits,
            torch.tensor(target_ids, dtype=torch.long),
            reduction="mean",
        )
        objective = (
            ce_loss
            + manager_objective
            + (current_drift * final_drift_grad).sum()
            + (commitment_cost * commitment_cost_grad).sum()
            + (committed_state * l_final_state_grad).sum()
        )
        # N identical Vulkan microbatches accumulate into the one canonical
        # full-model gradient registry before AdamW advances. Multiplying the
        # scalar objective is the exact PyTorch reference for that summed-
        # gradient update while holding parameters fixed across microbatches.
        (objective * ACCUMULATION_REPEATS).backward()

        for name, tensor in [
            ("enc", enc),
            ("previous_context", previous),
            ("target_context", target_context),
            ("h_state", h_state),
            ("l_state", l_state),
        ]:
            if tensor.grad is None:
                raise AssertionError(f"PyTorch did not retain {name} gradient")

        expected = {
            "loss": ce_loss.detach().clone(),
            "h_outputs": h_first_raw_output.detach().clone(),
            "h_final_packed_state": h_first_state.detach().clone(),
            "h_grad_initial_packed_state": (
                h_state.grad.detach().clone() / ACCUMULATION_REPEATS
            ),
            "manager_halt_probabilities": h_halt_stack.detach().clone(),
            "manager_selected_index": selection.selected_index.detach().clone(),
            "manager_executed_steps": selection.executed_steps.detach().clone(),
            "manager_selected_output": selection.output.detach().clone(),
            "manager_selected_packed_state": selection.state.detach().clone(),
            "l_outputs": committed_raw_output.detach().clone(),
            "l_final_packed_state": committed_state.detach().clone(),
            "l_grad_initial_packed_state": (
                l_state.grad.detach().clone() / ACCUMULATION_REPEATS
            ),
            "final_drift": current_drift.detach().clone(),
            "commitment_cost": commitment_cost.detach().clone(),
            "effective_l_steps": effective_l_steps.detach().clone(),
            "grad_enc": enc.grad.detach().clone() / ACCUMULATION_REPEATS,
            "grad_previous_context": (
                previous.grad.detach().clone() / ACCUMULATION_REPEATS
            ),
            "grad_target_context": (
                target_context.grad.detach().clone() / ACCUMULATION_REPEATS
            ),
        }
        if fp32_masters:
            restore_fp32_masters(fp32_masters)
        optimizer.step()

        case = {
            "batch": batch,
            "h_steps": H_STEPS,
            "shadow_steps": SHADOW_STEPS,
            "enc": enc.detach().flatten().tolist(),
            "previous_context": previous.detach().flatten().tolist(),
            "target_context": target_context.detach().flatten().tolist(),
            "context_alpha": ALPHA,
            "h_token_ids": h_token_ids,
            "l_token_ids": l_token_ids,
            "h_initial_packed_state": h_state.detach().flatten().tolist(),
            "l_initial_packed_state": l_state.detach().flatten().tolist(),
            "l_final_packed_state_grad": l_final_state_grad.flatten().tolist(),
            "h_to_context_grad": h_context_grad.flatten().tolist(),
            "h_depth_grad": h_depth_grad.flatten().tolist(),
            "h_selected_packed_state_grad": h_selected_state_grad.flatten().tolist(),
            "final_drift_grad": final_drift_grad.flatten().tolist(),
            "commitment_cost_grad": commitment_cost_grad.flatten().tolist(),
            "targets": target_ids,
            "accumulation_repeats": ACCUMULATION_REPEATS,
            "optimizer": {
                "lr": LR,
                "beta1": BETA1,
                "beta2": BETA2,
                "eps": EPS,
                "weight_decay": WEIGHT_DECAY,
            },
        }
        case_path = Path(temp_dir) / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        vulkan_env = os.environ.copy()
        vulkan_env[TRAINING_PRECISION_ENV] = args.precision
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-worker-refinement-loss-parity",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(model_dir),
            ],
            cwd=ROOT,
            env=vulkan_env,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan worker-refinement parity runner failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if os.environ.get("HIERARCHOS_VULKAN_TRACE_DISPATCH_CHAINS"):
            sys.stderr.write(completed.stderr)
        result = json.loads(completed.stdout)

    if result["training_precision_policy"] != args.precision:
        raise AssertionError(
            "Vulkan training precision policy mismatch: "
            f"requested={args.precision!r} actual={result['training_precision_policy']!r}"
        )
    expects_fp16_parameter_storage = args.precision != "fp32"
    expects_native_fp16_lm_backward = args.precision in (
        "fp16-storage-parity",
        "fp16-storage-fp16-lm-backward",
    )
    expects_promoted_native_fp16_graph_backward = (
        args.precision == "fp16-storage-fp16-lm-backward"
    )
    expects_native_fp16_low_rank_backward = (
        expects_promoted_native_fp16_graph_backward
        and env_flag_enabled(NATIVE_FP16_LOW_RANK_BACKWARD_ENV, default=True)
    )
    expects_native_fp16_out_norm_backward = (
        expects_promoted_native_fp16_graph_backward
        and env_flag_enabled(NATIVE_FP16_OUT_NORM_BACKWARD_ENV, default=True)
    )
    expects_native_fp16_projection_backward = (
        expects_promoted_native_fp16_graph_backward
        and env_flag_enabled(NATIVE_FP16_PROJECTION_BACKWARD_ENV, default=True)
    )
    for tower in ("h", "l"):
        field = f"{tower}_low_rank_fp16_parameter_storage_active"
        if bool(result[field]) != expects_fp16_parameter_storage:
            raise AssertionError(
                f"Vulkan precision consumer mismatch: {field}={result[field]!r} "
                f"for requested precision {args.precision!r}"
            )
        native_backward_field = (
            f"{tower}_low_rank_native_fp16_backward_compute_active"
        )
        if bool(result[native_backward_field]) != expects_native_fp16_low_rank_backward:
            raise AssertionError(
                f"Vulkan precision consumer mismatch: "
                f"{native_backward_field}={result[native_backward_field]!r} "
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
    if (
        bool(result["lm_head_native_fp16_backward_compute_active"])
        != expects_native_fp16_lm_backward
    ):
        raise AssertionError(
            "Vulkan native-FP16 LM backward mode mismatch: "
            f"active={result['lm_head_native_fp16_backward_compute_active']!r} "
            f"for requested precision {args.precision!r}"
        )
    if (
        bool(result["out_norm_native_fp16_backward_compute_active"])
        != expects_native_fp16_out_norm_backward
    ):
        raise AssertionError(
            "Vulkan native-FP16 out_norm backward mode mismatch: "
            f"active={result['out_norm_native_fp16_backward_compute_active']!r} "
            f"for requested precision {args.precision!r}"
        )
    if (
        bool(result["projection_native_fp16_backward_compute_active"])
        != expects_native_fp16_projection_backward
    ):
        raise AssertionError(
            "Vulkan native-FP16 projection backward mode mismatch: "
            f"active={result['projection_native_fp16_backward_compute_active']!r} "
            f"for requested precision {args.precision!r}"
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
            "native-FP16 LM backward must use the explicit reuse32 native arm; "
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
    if expects_fp16_parameter_storage:
        packed_lm = tensor_from_result(
            result["lm_head_fp16_execution_weight"], model.lm_head.weight
        )
        rust_lm_master = tensor_from_result(result["lm_head_weight"], model.lm_head.weight)
        expected_packed_lm = rust_lm_master.to(torch.float16).to(torch.float32)
        torch.testing.assert_close(packed_lm, expected_packed_lm, rtol=0.0, atol=0.0)
    assert result["queue_submissions"] == ACCUMULATION_REPEATS, result
    assert result["microbatches"] == ACCUMULATION_REPEATS, result
    if abs(result["activation_clamp"] - config.activation_clamp) > 1.0e-7:
        raise AssertionError(
            "Rust model config activation_clamp mismatch: "
            f"rust={result['activation_clamp']} torch={config.activation_clamp}"
        )
    assert result["h_optimizer_step"] == 1, result
    assert result["l_optimizer_step"] == 1, result
    assert result["projection_optimizer_step"] == 1, result
    assert result["lm_optimizer_step"] == 1, result
    assert result["full_model_optimizer_step"] == 1, result
    assert result["full_model_optimizer_checkpoint_roundtrip"], result
    if result["token_tape_tokens"] != 2:
        raise AssertionError(
            f"token tape parity expected 2 tokens; got {result['token_tape_tokens']}"
        )
    if result["token_tape_optimizer_step"] != 1:
        raise AssertionError(
            "token tape must advance the canonical optimizer exactly once; "
            f"got step={result['token_tape_optimizer_step']}"
        )
    if result["token_tape_queue_submissions"] != 1:
        raise AssertionError(
            "token tape must record forward checkpoints, every reverse token, and readback "
            "into one Vulkan submission; "
            f"got queue_submissions={result['token_tape_queue_submissions']}"
        )
    if not result["token_tape_parity"]:
        raise AssertionError(
            "device-resident token tape diverged from the explicit host-adjoint reference: "
            f"state={result['token_tape_max_state_diff']:.9g} "
            f"adjoint={result['token_tape_max_adjoint_diff']:.9g} "
            f"loss={result['token_tape_max_loss_diff']:.9g} "
            f"optimizer={result['token_tape_max_optimizer_diff']:.9g} "
            f"control_match={result['token_tape_control_match']}"
        )
    if result["token_tape_microbatch_sequences"] != 2:
        raise AssertionError(
            "token-tape arena parity expected 2 independent sequences; "
            f"got {result['token_tape_microbatch_sequences']}"
        )
    if result["token_tape_microbatch_total_tokens"] != 4:
        raise AssertionError(
            "token-tape arena parity expected 4 accumulated tokens; "
            f"got {result['token_tape_microbatch_total_tokens']}"
        )
    if result["token_tape_microbatch_optimizer_step"] != 1:
        raise AssertionError(
            "token-tape arena must advance the canonical optimizer exactly once; "
            f"got step={result['token_tape_microbatch_optimizer_step']}"
        )
    if result["token_tape_microbatch_queue_submissions"] != 1:
        raise AssertionError(
            "token-tape arena must record all sequence forwards/reverses and the shared AdamW "
            "step into one Vulkan submission; "
            f"got queue_submissions={result['token_tape_microbatch_queue_submissions']}"
        )
    descriptor_pools = result["token_tape_microbatch_descriptor_pool_count"]
    descriptor_sets = result["token_tape_microbatch_descriptor_set_count"]
    dispatches = result["token_tape_microbatch_dispatch_count"]
    pipeline_binds = result["token_tape_microbatch_pipeline_bind_count"]
    descriptor_binds = result["token_tape_microbatch_descriptor_bind_count"]
    push_constant_writes = result["token_tape_microbatch_push_constant_write_count"]
    upload_count = result["token_tape_microbatch_upload_count"]
    upload_arena_buffers = result["token_tape_microbatch_upload_arena_buffer_count"]
    if descriptor_pools <= 0 or descriptor_sets <= descriptor_pools:
        raise AssertionError(
            "descriptor arena did not demonstrate pooled descriptor-set allocation: "
            f"pools={descriptor_pools} sets={descriptor_sets}"
        )
    if descriptor_sets >= dispatches:
        raise AssertionError(
            "descriptor binding cache did not reduce unique descriptor sets below dispatch count: "
            f"sets={descriptor_sets} dispatches={dispatches}"
        )
    if not (0 < pipeline_binds < dispatches):
        raise AssertionError(
            "pipeline bind accounting is outside the dispatch envelope: "
            f"pipeline_binds={pipeline_binds} dispatches={dispatches}"
        )
    if not (0 < descriptor_binds <= dispatches):
        raise AssertionError(
            "descriptor bind accounting is outside the dispatch envelope: "
            f"descriptor_binds={descriptor_binds} dispatches={dispatches}"
        )
    if not (0 < push_constant_writes < dispatches):
        raise AssertionError(
            "push-constant accounting is outside the dispatch envelope: "
            f"push_constant_writes={push_constant_writes} dispatches={dispatches}"
        )
    if upload_count <= 0 or not (0 < upload_arena_buffers < upload_count):
        raise AssertionError(
            "packed upload arena did not reduce staging-buffer allocation count: "
            f"arena_buffers={upload_arena_buffers} uploads={upload_count}"
        )
    if not result["token_tape_microbatch_parity"]:
        raise AssertionError(
            "multi-sequence token-tape arena diverged from the explicit shared-gradient reference: "
            f"state={result['token_tape_microbatch_max_state_diff']:.9g} "
            f"adjoint={result['token_tape_microbatch_max_adjoint_diff']:.9g} "
            f"loss={result['token_tape_microbatch_max_loss_diff']:.9g} "
            f"optimizer={result['token_tape_microbatch_max_optimizer_diff']:.9g} "
            f"control_match={result['token_tape_microbatch_control_match']}"
        )
    if result["token_tape_sparse_replay_tokens"] != 8:
        raise AssertionError(
            "sparse replay parity must cross several real segment boundaries with eight tokens; "
            f"got tokens={result['token_tape_sparse_replay_tokens']}"
        )
    sparse_stride = result["token_tape_sparse_replay_checkpoint_stride"]
    if not 2 <= sparse_stride < result["token_tape_sparse_replay_tokens"]:
        raise AssertionError(
            "sparse replay parity expected a multi-segment checkpoint stride; "
            f"got stride={sparse_stride} tokens={result['token_tape_sparse_replay_tokens']}"
        )
    if result["token_tape_sparse_replay_queue_submissions"] != 1:
        raise AssertionError(
            "sparse replay must keep checkpointing, segment rematerialization, reverse, and AdamW "
            "inside one Vulkan submission; "
            f"got queue_submissions={result['token_tape_sparse_replay_queue_submissions']}"
        )
    if not result["token_tape_sparse_replay_parity"]:
        raise AssertionError(
            "sparse segment replay diverged from the dense full-BPTT tape: "
            f"state={result['token_tape_sparse_replay_max_state_diff']:.9g} "
            f"adjoint={result['token_tape_sparse_replay_max_adjoint_diff']:.9g} "
            f"loss={result['token_tape_sparse_replay_max_loss_diff']:.9g} "
            f"optimizer={result['token_tape_sparse_replay_max_optimizer_diff']:.9g} "
            f"control_match={result['token_tape_sparse_replay_control_match']}"
        )
    if not result["dynamic_loss_scale_parity"]:
        raise AssertionError(
            "deferred dynamic loss-scale finish diverged from the ordinary full-model close: "
            f"parameter={result['dynamic_loss_scale_max_parameter_diff']:.9g} "
            f"moments={result['dynamic_loss_scale_max_moment_diff']:.9g} "
            f"step={result['dynamic_loss_scale_optimizer_step']} "
            f"submissions={result['dynamic_loss_scale_queue_submissions']} "
            f"scale_after={result['dynamic_loss_scale_scale_after']} "
            f"growth_tracker={result['dynamic_loss_scale_growth_tracker']}"
        )
    loss_diff = abs(result["loss"] - expected["loss"].item())
    if not math.isfinite(loss_diff) or loss_diff > 2.0e-5:
        raise AssertionError(
            f"CE loss parity failed: rust={result['loss']:.9g} "
            f"torch={expected['loss'].item():.9g} diff={loss_diff:.9g}"
        )

    graph_diffs: dict[str, float] = {}
    for field in (
        "manager_halt_probabilities",
        "manager_executed_steps",
        "manager_selected_output",
        "manager_selected_packed_state",
        "h_outputs",
        "h_final_packed_state",
        "h_grad_initial_packed_state",
        "l_outputs",
        "l_final_packed_state",
        "l_grad_initial_packed_state",
        "final_drift",
        "commitment_cost",
        "effective_l_steps",
        "grad_enc",
        "grad_previous_context",
        "grad_target_context",
    ):
        graph_diffs[field] = assert_close(field, result[field], expected[field])
    graph_diffs["sequence_state_h_packed_state"] = assert_close(
        "sequence_state_h_packed_state",
        result["sequence_state_h_packed_state"],
        expected["manager_selected_packed_state"],
    )
    graph_diffs["sequence_state_l_packed_state"] = assert_close(
        "sequence_state_l_packed_state",
        result["sequence_state_l_packed_state"],
        expected["l_final_packed_state"],
    )
    graph_diffs["sequence_state_h_packed_state_adjoint"] = assert_close(
        "sequence_state_h_packed_state_adjoint",
        result["sequence_state_h_packed_state_adjoint"],
        expected["h_grad_initial_packed_state"],
    )
    graph_diffs["sequence_state_l_packed_state_adjoint"] = assert_close(
        "sequence_state_l_packed_state_adjoint",
        result["sequence_state_l_packed_state_adjoint"],
        expected["l_grad_initial_packed_state"],
    )
    actual_selected_index = torch.tensor(result["manager_selected_index"], dtype=torch.long)
    if not torch.equal(actual_selected_index, expected["manager_selected_index"]):
        raise AssertionError(
            "manager_selected_index parity failed: "
            f"rust={actual_selected_index.tolist()} "
            f"torch={expected['manager_selected_index'].tolist()}"
        )

    named = dict(model.named_parameters())
    parameter_diffs: dict[str, float] = {}
    for branch in ("h", "l"):
        for snapshot in result[f"{branch}_parameters"]:
            pytorch_name = local_recurrent_name(branch, snapshot["name"])
            if pytorch_name not in named:
                raise AssertionError(
                    f"no PyTorch parameter matches Vulkan {branch} tensor {snapshot['name']!r}"
                )
            parameter_diffs[pytorch_name] = assert_close(
                pytorch_name, snapshot["values"], named[pytorch_name]
            )
    for snapshot in result["projection_parameters"]:
        name = snapshot["name"]
        if name not in named:
            raise AssertionError(f"no PyTorch projection parameter matches Vulkan {name!r}")
        parameter_diffs[name] = assert_close(name, snapshot["values"], named[name])

    parameter_diffs["lm_head.weight"] = assert_close(
        "lm_head.weight", result["lm_head_weight"], model.lm_head.weight
    )
    parameter_diffs["out_norm.weight"] = assert_close(
        "out_norm.weight", result["out_norm_weight"], model.out_norm.weight
    )
    parameter_diffs["out_norm.bias"] = assert_close(
        "out_norm.bias", result["out_norm_bias"], model.out_norm.bias
    )
    registry_names = result["full_model_optimizer_names"]
    if len(registry_names) != result["full_model_optimizer_tensor_count"]:
        raise AssertionError(
            "full-model AdamW tensor count does not match serialized slot names: "
            f"count={result['full_model_optimizer_tensor_count']} names={len(registry_names)}"
        )
    if len(registry_names) != len(set(registry_names)):
        raise AssertionError("full-model AdamW registry contains duplicate tensor names")
    expected_registry_names = set(parameter_diffs)
    if set(registry_names) != expected_registry_names:
        missing = sorted(expected_registry_names - set(registry_names))
        extra = sorted(set(registry_names) - expected_registry_names)
        raise AssertionError(
            "full-model AdamW registry does not match the trained PyTorch parameter set: "
            f"missing={missing} extra={extra}"
        )

    worst_graph = max(graph_diffs, key=graph_diffs.get)
    worst_parameter = max(parameter_diffs, key=parameter_diffs.get)
    print(
        f"device={result['device']} precision={result['training_precision_policy']} "
        f"h_fp16_packed={result['h_low_rank_fp16_parameter_storage_active']} "
        f"l_fp16_packed={result['l_low_rank_fp16_parameter_storage_active']} "
        f"projection_fp16_packed={result['projection_fp16_parameter_storage_active']} "
        f"lm_head_fp16_packed={result['lm_head_fp16_parameter_storage_active']} "
        f"lm_arm={result['lm_head_execution_arm']} "
        f"lm_dw={result['lm_head_weight_grad_topology']} "
        f"lm_fused={result['lm_head_fused_adjoint_topology']} "
        f"h_steps={H_STEPS} shadow_steps={SHADOW_STEPS} "
        f"microbatches={ACCUMULATION_REPEATS} "
        f"queue_submissions={result['queue_submissions']} "
        f"full_model_tensors={result['full_model_optimizer_tensor_count']}"
    )
    print(f"loss_abs_diff={loss_diff:.9g}")
    print(
        f"worst_graph_value={worst_graph} "
        f"max_abs_graph_diff={graph_diffs[worst_graph]:.9g}"
    )
    print(
        f"worst_parameter={worst_parameter} "
        f"max_abs_parameter_diff={parameter_diffs[worst_parameter]:.9g}"
    )
    print(
        "token_tape="
        f"tokens={result['token_tape_tokens']} "
        f"queue_submissions={result['token_tape_queue_submissions']} "
        f"state_diff={result['token_tape_max_state_diff']:.9g} "
        f"adjoint_diff={result['token_tape_max_adjoint_diff']:.9g} "
        f"loss_diff={result['token_tape_max_loss_diff']:.9g} "
        f"optimizer_diff={result['token_tape_max_optimizer_diff']:.9g} "
        f"control_match={result['token_tape_control_match']}"
    )
    print(
        "token_tape_microbatch="
        f"sequences={result['token_tape_microbatch_sequences']} "
        f"tokens={result['token_tape_microbatch_total_tokens']} "
        f"queue_submissions={result['token_tape_microbatch_queue_submissions']} "
        f"descriptor_pools={result['token_tape_microbatch_descriptor_pool_count']} "
        f"descriptor_sets={result['token_tape_microbatch_descriptor_set_count']} "
        f"dispatches={result['token_tape_microbatch_dispatch_count']} "
        f"shader_barriers={result['token_tape_microbatch_shader_barrier_count']} "
        f"pipeline_binds={result['token_tape_microbatch_pipeline_bind_count']} "
        f"descriptor_binds={result['token_tape_microbatch_descriptor_bind_count']} "
        f"push_constant_writes={result['token_tape_microbatch_push_constant_write_count']} "
        f"uploads={result['token_tape_microbatch_upload_count']} "
        f"uploaded_bytes={result['token_tape_microbatch_uploaded_bytes']} "
        f"upload_arena_buffers={result['token_tape_microbatch_upload_arena_buffer_count']} "
        f"state_diff={result['token_tape_microbatch_max_state_diff']:.9g} "
        f"adjoint_diff={result['token_tape_microbatch_max_adjoint_diff']:.9g} "
        f"loss_diff={result['token_tape_microbatch_max_loss_diff']:.9g} "
        f"optimizer_diff={result['token_tape_microbatch_max_optimizer_diff']:.9g} "
        f"control_match={result['token_tape_microbatch_control_match']}"
    )
    print(
        "token_tape_sparse_replay="
        f"tokens={result['token_tape_sparse_replay_tokens']} "
        f"checkpoint_stride={result['token_tape_sparse_replay_checkpoint_stride']} "
        f"queue_submissions={result['token_tape_sparse_replay_queue_submissions']} "
        f"state_diff={result['token_tape_sparse_replay_max_state_diff']:.9g} "
        f"adjoint_diff={result['token_tape_sparse_replay_max_adjoint_diff']:.9g} "
        f"loss_diff={result['token_tape_sparse_replay_max_loss_diff']:.9g} "
        f"optimizer_diff={result['token_tape_sparse_replay_max_optimizer_diff']:.9g} "
        f"control_match={result['token_tape_sparse_replay_control_match']}"
    )
    print(
        "dynamic_loss_scale_finish="
        f"optimizer_step={result['dynamic_loss_scale_optimizer_step']} "
        f"queue_submissions={result['dynamic_loss_scale_queue_submissions']} "
        f"scale_after={result['dynamic_loss_scale_scale_after']} "
        f"growth_tracker={result['dynamic_loss_scale_growth_tracker']} "
        f"parameter_diff={result['dynamic_loss_scale_max_parameter_diff']:.9g} "
        f"moment_diff={result['dynamic_loss_scale_max_moment_diff']:.9g}"
    )
    print(
        "Hierarchos Vulkan worker-refinement + committed-restart + LM loss "
        "PyTorch parity: PASS"
    )


if __name__ == "__main__":
    main()
