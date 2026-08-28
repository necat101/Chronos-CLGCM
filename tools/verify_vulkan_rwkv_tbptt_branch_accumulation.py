#!/usr/bin/env python3
"""Verify deferred multi-branch RWKV/Tied-LM AdamW accumulation parity."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import verify_vulkan_rwkv_tbptt_parity as reference
import verify_vulkan_rwkv_tbptt_train_parity as train_reference


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 31
BRANCH_COUNT = 2


def main() -> None:
    torch.manual_seed(20260824)
    p = reference.make_parameters()
    lm_weight = torch.nn.Parameter(
        torch.randn(VOCAB_SIZE, reference.INPUT_DIM, dtype=torch.float32) * 0.05
    )

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

    initial_tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in reference.package_tensors(p).items()
    }
    initial_tensors["lm_head.weight"] = lm_weight.detach().clone().contiguous()

    branch_cases: list[dict[str, object]] = []
    final_expected: dict[str, torch.Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    for branch_index in range(BRANCH_COUNT):
        x = (
            torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH)
            * (0.18 + branch_index * 0.025)
        ).requires_grad_()
        initial_state = (
            torch.randn(
                reference.BATCH, reference.WIDTH, reference.STATE_SIZE
            )
            * (0.20 + branch_index * 0.02)
        ).requires_grad_()
        grad_output = (
            torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH)
            * (0.018 + branch_index * 0.004)
        )
        final_state_grad = (
            torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE)
            * (0.009 + branch_index * 0.002)
        )
        token_ids = torch.tensor(
            [[5 + branch_index], [11], [5 + branch_index], [17]], dtype=torch.long
        )
        token_features = F.embedding(token_ids, lm_weight)
        token_features.retain_grad()

        state = initial_state
        outputs: list[torch.Tensor] = []
        for timestep in range(reference.STEPS):
            if timestep > 0 and timestep % reference.DETACH_EVERY == 0:
                state = state.detach()
            output, state = reference.cell_step(
                x[timestep], token_features[timestep], state, p
            )
            outputs.append(output)
        output_sequence = torch.stack(outputs, dim=0)
        objective = (output_sequence * grad_output).sum() + (
            state * final_state_grad
        ).sum()
        objective.backward()

        branch_cases.append(
            {
                "batch": reference.BATCH,
                "steps": reference.STEPS,
                "detach_every_n_steps": reference.DETACH_EVERY,
                "x_sequence": x.detach().flatten().tolist(),
                "token_id_sequence": token_ids.flatten().tolist(),
                "initial_packed_state": initial_state.detach().flatten().tolist(),
                "grad_output_sequence": grad_output.flatten().tolist(),
                "final_packed_state_grad": final_state_grad.flatten().tolist(),
            }
        )
        if branch_index + 1 == BRANCH_COUNT:
            if token_features.grad is None or x.grad is None or initial_state.grad is None:
                raise AssertionError("final PyTorch branch gradients were not retained")
            final_expected = {
                "outputs": output_sequence.detach().clone(),
                "final_state": state.detach().clone(),
                "grad_x": x.grad.detach().clone(),
                "token_grad": token_features.grad.detach().clone(),
                "initial_state_grad": initial_state.grad.detach().clone(),
            }

    optimizer.step()
    if final_expected is None:
        raise AssertionError("no final branch reference")

    case = {
        "width": reference.WIDTH,
        "head_size": reference.HEAD_SIZE,
        "input_dim": reference.INPUT_DIM,
        "state_mode": "explicit-output",
        "state_clamp": reference.STATE_CLAMP,
        "branches": branch_cases,
        "optimizer": {
            "lr": train_reference.LR,
            "beta1": train_reference.BETA1,
            "beta2": train_reference.BETA2,
            "eps": train_reference.EPS,
            "weight_decay": train_reference.WEIGHT_DECAY,
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="hierarchos-vulkan-tbptt-branches-"
    ) as temp_dir:
        temp = Path(temp_dir)
        package_dir = temp / "model-package"
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
                "hierarchos-vulkan-rwkv-tbptt-branch-accumulation",
                "--",
                "--case",
                str(case_path),
                "--model-dir",
                str(package_dir),
                "--cell-prefix",
                "h_rnn",
                "--adapter-prefix",
                "h_deepembed_adapter",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Vulkan deferred-branch runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["branch_count"] == BRANCH_COUNT, result
    assert result["optimizer_step"] == 1, result
    assert result["optimizer_tensor_count"] == len(train_reference.PARAMETER_MAP), result
    assert result["tied_embedding_optimizer_step"] == 1, result
    assert result["tied_embedding_optimizer_tensor_count"] == 1, result

    actual_parameters = {
        item["name"]: torch.tensor(item["values"], dtype=torch.float32)
        for item in result["parameters"]
    }
    expected_names = set(train_reference.PARAMETER_MAP) | {"lm_head.weight"}
    if set(actual_parameters) != expected_names:
        missing = sorted(expected_names - set(actual_parameters))
        extra = sorted(set(actual_parameters) - expected_names)
        raise AssertionError(f"parameter registry mismatch: missing={missing} extra={extra}")

    max_parameter_diff = 0.0
    worst_name = ""
    for rust_name, pytorch_name in train_reference.PARAMETER_MAP.items():
        expected = p[pytorch_name].detach().flatten()
        actual = actual_parameters[rust_name].reshape_as(expected)
        diff = (actual - expected).abs().max().item()
        if diff > max_parameter_diff:
            max_parameter_diff = diff
            worst_name = rust_name
        torch.testing.assert_close(actual, expected, rtol=4.0e-3, atol=4.0e-5)
    actual_lm = actual_parameters["lm_head.weight"].reshape_as(lm_weight)
    lm_diff = (actual_lm - lm_weight.detach()).abs().max().item()
    if lm_diff > max_parameter_diff:
        max_parameter_diff = lm_diff
        worst_name = "lm_head.weight"
    torch.testing.assert_close(
        actual_lm, lm_weight.detach(), rtol=4.0e-3, atol=4.0e-5
    )

    comparisons = {
        "final_outputs": "outputs",
        "final_packed_state": "final_state",
        "final_grad_x": "grad_x",
        "final_token_feature_grad": "token_grad",
        "final_grad_initial_packed_state": "initial_state_grad",
    }
    max_sequence_diff = 0.0
    worst_sequence = ""
    for actual_name, expected_name in comparisons.items():
        expected = final_expected[expected_name].flatten()
        actual = torch.tensor(result[actual_name], dtype=torch.float32).reshape_as(expected)
        diff = (actual - expected).abs().max().item()
        if diff > max_sequence_diff:
            max_sequence_diff = diff
            worst_sequence = actual_name
        torch.testing.assert_close(actual, expected, rtol=4.0e-3, atol=4.0e-5)

    print(f"device={result['device']} branches={BRANCH_COUNT} optimizer_step=1")
    print(
        f"worst_parameter={worst_name} max_abs_parameter_diff={max_parameter_diff:.9g}"
    )
    print(
        f"worst_final_branch_value={worst_sequence} "
        f"max_abs_final_branch_diff={max_sequence_diff:.9g}"
    )
    print("Hierarchos Vulkan deferred recurrent branch accumulation parity: PASS")


if __name__ == "__main__":
    main()
