#!/usr/bin/env python3
"""Verify token-ID -> tied embedding -> Vulkan TBPTT training parity."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import verify_vulkan_rwkv_tbptt_parity as reference
import verify_vulkan_rwkv_tbptt_train_parity as train_reference


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 29
UPDATES = 2


def main() -> None:
    torch.manual_seed(20260823)
    p = reference.make_parameters()
    x_sequence = (
        torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH) * 0.22
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

    # Repeat IDs deliberately so the Vulkan CAS-based FP32 scatter is tested,
    # not merely the trivial one-token-per-row case.
    token_ids = torch.tensor([[5], [11], [5], [5]], dtype=torch.long)
    if token_ids.shape != (reference.STEPS, reference.BATCH):
        raise AssertionError(token_ids.shape)

    initial_lm = torch.randn(
        VOCAB_SIZE, reference.INPUT_DIM, dtype=torch.float32
    ) * 0.05
    lm_weight = torch.nn.Parameter(initial_lm.clone())

    initial_tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in reference.package_tensors(p).items()
    }
    initial_tensors["lm_head.weight"] = initial_lm.clone().contiguous()
    untouched_tensor = torch.tensor([2.5, -1.0, 7.25], dtype=torch.float32)
    initial_tensors["unrelated.test_tensor"] = untouched_tensor.clone()

    decay_params = [
        p[train_reference.PARAMETER_MAP[name]]
        for name in train_reference.PARAMETER_MAP
        if name not in train_reference.NO_DECAY
    ]
    decay_params.append(lm_weight)
    no_decay_params = [
        p[train_reference.PARAMETER_MAP[name]]
        for name in train_reference.PARAMETER_MAP
        if name in train_reference.NO_DECAY
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_reference.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_reference.LR,
        betas=(train_reference.BETA1, train_reference.BETA2),
        eps=train_reference.EPS,
    )

    last_token_grad = None
    for _ in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        x_sequence.grad = None
        initial_state.grad = None
        token_features = F.embedding(token_ids, lm_weight)
        token_features.retain_grad()
        state = initial_state
        outputs: list[torch.Tensor] = []
        for timestep in range(reference.STEPS):
            if timestep > 0 and timestep % reference.DETACH_EVERY == 0:
                state = state.detach()
            output, state = reference.cell_step(
                x_sequence[timestep], token_features[timestep], state, p
            )
            outputs.append(output)
        output_sequence = torch.stack(outputs, dim=0)
        objective = (output_sequence * grad_output).sum() + (
            state * final_state_grad
        ).sum()
        objective.backward()
        last_token_grad = token_features.grad.detach().clone()
        optimizer.step()
    assert last_token_grad is not None

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
        "token_id_sequence": token_ids.flatten().tolist(),
        "initial_packed_state": initial_state.detach().flatten().tolist(),
        "grad_output_sequence": grad_output.flatten().tolist(),
        "final_packed_state_grad": final_state_grad.flatten().tolist(),
        "optimizer": {
            "lr": train_reference.LR,
            "beta1": train_reference.BETA1,
            "beta2": train_reference.BETA2,
            "eps": train_reference.EPS,
            "weight_decay": train_reference.WEIGHT_DECAY,
            "updates": UPDATES,
        },
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-tbptt-tied-") as temp_dir:
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
        completed = subprocess.run(
            [
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
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan tied-embedding TBPTT parity runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)
        trained_checkpoint = load_file(str(trained_package_dir / "model.safetensors"))

    assert result["optimizer_step"] == UPDATES, result["optimizer_step"]
    assert result["optimizer_tensor_count"] == len(train_reference.PARAMETER_MAP)
    assert result["tied_embedding_optimizer_step"] == UPDATES
    assert result["tied_embedding_optimizer_tensor_count"] == 1

    actual_parameters = {
        item["name"]: torch.tensor(item["values"], dtype=torch.float32)
        for item in result["parameters"]
    }
    expected_names = set(train_reference.PARAMETER_MAP) | {"lm_head.weight"}
    if set(actual_parameters) != expected_names:
        missing = sorted(expected_names - set(actual_parameters))
        extra = sorted(set(actual_parameters) - expected_names)
        raise AssertionError(f"parameter registry mismatch: missing={missing} extra={extra}")

    max_diff = 0.0
    max_relative_diff = 0.0
    worst_name = ""
    for rust_name, pytorch_name in train_reference.PARAMETER_MAP.items():
        expected = p[pytorch_name].detach().flatten()
        actual = actual_parameters[rust_name].reshape_as(expected)
        diff = (actual - expected).abs().max().item()
        relative = ((actual - expected).abs() / expected.abs().clamp_min(1.0e-6)).max().item()
        if diff > max_diff:
            max_diff = diff
            worst_name = rust_name
        max_relative_diff = max(max_relative_diff, relative)

    actual_lm = actual_parameters["lm_head.weight"].reshape_as(lm_weight)
    lm_diff = (actual_lm - lm_weight.detach()).abs().max().item()
    lm_relative = (
        (actual_lm - lm_weight.detach()).abs()
        / lm_weight.detach().abs().clamp_min(1.0e-6)
    ).max().item()
    if lm_diff > max_diff:
        max_diff = lm_diff
        worst_name = "lm_head.weight"
    max_relative_diff = max(max_relative_diff, lm_relative)

    # The token-ID path inserts one additional Vulkan gather before the same
    # recurrent graph. The operation itself is an exact FP32 copy, but the
    # changed activation distribution can amplify normal shader/PyTorch order
    # differences near zeros. Keep a strict absolute bound over the complete
    # 34-tensor registry rather than failing on a meaningless relative error at
    # a ~1e-4 parameter value.
    if max_diff > 3.0e-5:
        raise AssertionError(
            f"parameter parity exceeded absolute tolerance: worst={worst_name} "
            f"max_abs={max_diff:.9g} max_relative={max_relative_diff:.9g}"
        )

    actual_token_grad = torch.tensor(
        result["token_feature_grad"], dtype=torch.float32
    ).reshape_as(last_token_grad)
    torch.testing.assert_close(
        actual_token_grad, last_token_grad, rtol=3.0e-3, atol=3.0e-5
    )

    expected_checkpoint = reference.package_tensors(p)
    for rust_name in train_reference.PARAMETER_MAP:
        checkpoint_name = (
            f"h_deepembed_adapter.{rust_name.removeprefix('deepembed.')}"
            if rust_name.startswith("deepembed.")
            else f"h_rnn.{rust_name}"
        )
        torch.testing.assert_close(
            trained_checkpoint[checkpoint_name],
            expected_checkpoint[checkpoint_name],
            rtol=3.0e-3,
            atol=3.0e-5,
        )
    torch.testing.assert_close(
        trained_checkpoint["lm_head.weight"],
        lm_weight.detach(),
        rtol=3.0e-3,
        atol=3.0e-5,
    )
    torch.testing.assert_close(
        trained_checkpoint["unrelated.test_tensor"], untouched_tensor, rtol=0.0, atol=0.0
    )

    print(
        f"device={result['device']} cell_step={result['optimizer_step']} "
        f"tied_step={result['tied_embedding_optimizer_step']} worst={worst_name} "
        f"max_abs_parameter_diff={max_diff:.9g}"
    )
    print(
        "max_abs_token_feature_grad_diff="
        f"{(actual_token_grad - last_token_grad).abs().max().item():.9g}"
    )
    print("Hierarchos Vulkan token-ID -> tied embedding -> TBPTT PyTorch parity: PASS")
    print("Hierarchos Vulkan tied embedding SafeTensors interchange: PASS")


if __name__ == "__main__":
    main()
