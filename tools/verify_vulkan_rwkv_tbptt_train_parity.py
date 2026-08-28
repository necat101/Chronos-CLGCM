#!/usr/bin/env python3
"""Verify all-cell Vulkan TBPTT gradient accumulation + persistent AdamW parity."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import verify_vulkan_rwkv_tbptt_parity as reference


ROOT = Path(__file__).resolve().parents[1]

LR = 3.0e-4
BETA1 = 0.9
BETA2 = 0.999
EPS = 1.0e-8
WEIGHT_DECAY = 0.1
UPDATES = 2
PARAMETER_RTOL = 2.5e-3
PARAMETER_ATOL = 2.5e-5

PARAMETER_MAP = {
    "ln1.weight": "ln1_w",
    "ln1.bias": "ln1_b",
    "x_r": "mix_r",
    "x_k": "mix_k",
    "x_v": "mix_v",
    "receptance.weight": "r_weight",
    "key.weight": "k_weight",
    "value.weight": "v_weight",
    "k_k": "k_k",
    "k_a": "k_a",
    "x_w": "mix_w",
    "x_a": "mix_a",
    "x_g": "mix_g",
    "w0": "w0",
    "w1": "w1",
    "w2": "w2",
    "a0": "a0",
    "a1": "a1",
    "a2": "a2",
    "g1": "g1",
    "g2": "g2",
    "ln_x.weight": "gn_w",
    "ln_x.bias": "gn_b",
    "r_k": "r_k",
    "output.weight": "out_weight",
    "ln2.weight": "ln2_w",
    "ln2.bias": "ln2_b",
    "x_k_cm": "mix_k_cm",
    "key_cm.weight": "key_cm",
    "value_cm.weight": "value_cm",
    "deepembed.down.weight": "adapter_down",
    "deepembed.up.weight": "adapter_up",
    "deepembed.bias": "adapter_bias",
}

NO_DECAY = {
    "ln1.weight",
    "ln1.bias",
    "ln_x.weight",
    "ln_x.bias",
    "ln2.weight",
    "ln2.bias",
    "deepembed.down.weight",
    "deepembed.up.weight",
    "deepembed.bias",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=reference.HEADS)
    parser.add_argument("--head-size", type=int, default=reference.HEAD_SIZE)
    parser.add_argument(
        "--kernel-geometry",
        choices=("rwkv-state-bwd-wg32", "rwkv-state-bwd-wg64", "rwkv-state-bwd-wg128"),
        default=None,
    )
    parser.add_argument(
        "--numerics",
        choices=(
            "strict",
            "fast-subgroup",
            "fast-recurrent-tree",
            "fast-recurrent-tiled",
            "fast-recurrent-subgroup",
        ),
        default="strict",
    )
    args = parser.parse_args()
    if args.heads <= 0 or args.head_size <= 0:
        parser.error("--heads and --head-size must be positive")
    reference.HEADS = args.heads
    reference.HEAD_SIZE = args.head_size
    reference.WIDTH = reference.HEADS * reference.HEAD_SIZE
    reference.HIDDEN = reference.WIDTH * 4
    reference.STATE_SIZE = reference.STATE_OFFSET + reference.HEAD_SIZE
    torch.manual_seed(20260822)
    p = reference.make_parameters()
    x_sequence = (
        torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH) * 0.22
    ).requires_grad_()
    token_sequence = (
        torch.randn(reference.STEPS, reference.BATCH, reference.INPUT_DIM) * 0.28
    ).requires_grad_()
    initial_state = (
        torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE) * 0.24
    ).requires_grad_()
    grad_output = (
        torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH) * 0.025
    )
    final_state_grad = (
        torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE) * 0.012
    )

    # Copy the pre-update package before AdamW mutates the PyTorch Parameters.
    initial_tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in reference.package_tensors(p).items()
    }
    untouched_tensor = torch.tensor([1.25, -3.5, 8.0], dtype=torch.float32)
    initial_tensors["unrelated.test_tensor"] = untouched_tensor.clone()

    decay_params = [
        p[PARAMETER_MAP[name]] for name in PARAMETER_MAP if name not in NO_DECAY
    ]
    no_decay_params = [
        p[PARAMETER_MAP[name]] for name in PARAMETER_MAP if name in NO_DECAY
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(BETA1, BETA2),
        eps=EPS,
    )
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        x_sequence.grad = None
        token_sequence.grad = None
        initial_state.grad = None
        state = initial_state
        outputs: list[torch.Tensor] = []
        for timestep in range(reference.STEPS):
            if timestep > 0 and timestep % reference.DETACH_EVERY == 0:
                state = state.detach()
            output, state = reference.cell_step(
                x_sequence[timestep], token_sequence[timestep], state, p
            )
            outputs.append(output)
        output_sequence = torch.stack(outputs, dim=0)
        objective = (output_sequence * grad_output).sum() + (
            state * final_state_grad
        ).sum()
        objective.backward()
        optimizer.step()

    case = {
        "batch": reference.BATCH,
        "steps": reference.STEPS,
        "width": reference.WIDTH,
        "head_size": reference.HEAD_SIZE,
        "input_dim": reference.INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": reference.STATE_CLAMP,
        "detach_every_n_steps": reference.DETACH_EVERY,
        "x_sequence": x_sequence.detach().flatten().tolist(),
        "token_feature_sequence": token_sequence.detach().flatten().tolist(),
        "initial_packed_state": initial_state.detach().flatten().tolist(),
        "grad_output_sequence": grad_output.flatten().tolist(),
        "final_packed_state_grad": final_state_grad.flatten().tolist(),
        "optimizer": {
            "lr": LR,
            "beta1": BETA1,
            "beta2": BETA2,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "updates": UPDATES,
        },
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-tbptt-train-") as temp_dir:
        temp = Path(temp_dir)
        package_dir = temp / "model-package"
        trained_package_dir = temp / "trained-model-package"
        package_dir.mkdir()
        save_file(
            initial_tensors,
            str(package_dir / "model.safetensors"),
            metadata={"format": "pt", "layout": "pytorch-row-major"},
        )
        case_path = temp / "case.json"
        case_path.write_text(json.dumps(case), encoding="utf-8")
        command = [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(ROOT / "hierarchos-vulkan" / "Cargo.toml"),
                "--bin",
                "hierarchos-vulkan-rwkv-tbptt-sequence-step",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(package_dir),
                "--cell-prefix",
                "h_rnn",
                "--adapter-prefix",
                "h_deepembed_adapter",
                "--output-model",
                str(trained_package_dir),
            ]
        if args.kernel_geometry is not None:
            command.extend(["--kernel-geometry", args.kernel_geometry])
        command.extend(["--numerics", args.numerics])
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan TBPTT training parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        trained_checkpoint = load_file(str(trained_package_dir / "model.safetensors"))

    expected_numerics = "strict-parity" if args.numerics == "strict" else args.numerics
    if result["numerics_policy"] != expected_numerics:
        raise AssertionError(
            f"Vulkan numerics policy mismatch: {result['numerics_policy']} != {expected_numerics}"
        )
    if args.kernel_geometry is not None and result["backward_kernel_geometry"] != args.kernel_geometry:
        raise AssertionError(
            "Vulkan backward geometry mismatch: "
            f"{result['backward_kernel_geometry']} != {args.kernel_geometry}"
        )

    assert result["optimizer_step"] == UPDATES, result["optimizer_step"]
    assert result["optimizer_tensor_count"] == len(PARAMETER_MAP), (
        result["optimizer_tensor_count"],
        len(PARAMETER_MAP),
    )
    actual_parameters = {
        item["name"]: torch.tensor(item["values"], dtype=torch.float32)
        for item in result["parameters"]
    }
    if set(actual_parameters) != set(PARAMETER_MAP):
        missing = sorted(set(PARAMETER_MAP) - set(actual_parameters))
        extra = sorted(set(actual_parameters) - set(PARAMETER_MAP))
        raise AssertionError(f"parameter registry mismatch: missing={missing} extra={extra}")

    max_diff = 0.0
    worst_name = ""
    first_close_failure: tuple[str, str, int, float, float, float, float] | None = None
    for rust_name, pytorch_name in PARAMETER_MAP.items():
        expected = p[pytorch_name].detach().flatten()
        actual = actual_parameters[rust_name].reshape_as(expected)
        abs_diff = (actual - expected).abs()
        flat_index = int(abs_diff.argmax().item())
        diff = abs_diff[flat_index].item()
        if diff > max_diff:
            max_diff = diff
            worst_name = rust_name
        try:
            torch.testing.assert_close(
                actual,
                expected,
                rtol=PARAMETER_RTOL,
                atol=PARAMETER_ATOL,
            )
        except AssertionError:
            if first_close_failure is None:
                expected_value = float(expected[flat_index].item())
                actual_value = float(actual[flat_index].item())
                relative_diff = diff / max(abs(expected_value), 1.0e-30)
                first_close_failure = (
                    rust_name,
                    pytorch_name,
                    flat_index,
                    expected_value,
                    actual_value,
                    diff,
                    relative_diff,
                )

    if first_close_failure is not None:
        (
            rust_name,
            pytorch_name,
            flat_index,
            expected_value,
            actual_value,
            diff,
            relative_diff,
        ) = first_close_failure
        raise AssertionError(
            "Vulkan/PyTorch trained-parameter parity failed: "
            f"rust={rust_name} pytorch={pytorch_name} flat_index={flat_index} "
            f"expected={expected_value:.9g} actual={actual_value:.9g} "
            f"abs_diff={diff:.9g} rel_diff={relative_diff:.9g}; "
            f"global_worst={worst_name} global_max_abs_diff={max_diff:.9g}"
        )

    expected_checkpoint = reference.package_tensors(p)
    for rust_name in PARAMETER_MAP:
        checkpoint_name = (
            f"h_deepembed_adapter.{rust_name.removeprefix('deepembed.')}"
            if rust_name.startswith("deepembed.")
            else f"h_rnn.{rust_name}"
        )
        expected = expected_checkpoint[checkpoint_name]
        actual = trained_checkpoint[checkpoint_name]
        if actual.shape != expected.shape:
            raise AssertionError(
                f"exported shape mismatch for {checkpoint_name}: {actual.shape} != {expected.shape}"
            )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=PARAMETER_RTOL,
            atol=PARAMETER_ATOL,
        )
    torch.testing.assert_close(
        trained_checkpoint["unrelated.test_tensor"], untouched_tensor, rtol=0.0, atol=0.0
    )

    print(
        f"device={result['device']} numerics={result['numerics_policy']} "
        f"geometry={result['backward_kernel_geometry']} optimizer_step={result['optimizer_step']} "
        f"tensor_count={result['optimizer_tensor_count']} worst={worst_name} "
        f"max_abs_parameter_diff={max_diff:.9g}"
    )
    print(
        "Hierarchos Vulkan TBPTT 33-tensor gradient accumulation + AdamW PyTorch parity: PASS"
    )
    print("Hierarchos Vulkan trained-cell SafeTensors interchange: PASS")


if __name__ == "__main__":
    main()
