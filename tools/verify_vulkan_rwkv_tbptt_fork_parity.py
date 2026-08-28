#!/usr/bin/env python3
"""Verify shadow-chain + committed-restart recurrent fork parity."""

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
VOCAB_SIZE = 37
COMMITTED_STEPS = 1


def recurrent_branch(
    p: dict[str, torch.Tensor],
    lm_weight: torch.nn.Parameter,
    x: torch.Tensor,
    token_ids: torch.Tensor,
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    final_state_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_features = F.embedding(token_ids, lm_weight)
    token_features.retain_grad()
    state = initial_state
    outputs: list[torch.Tensor] = []
    for timestep in range(x.shape[0]):
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
    return objective, output_sequence, state, token_features


def main() -> None:
    torch.manual_seed(20260825)
    p = reference.make_parameters()
    lm_weight = torch.nn.Parameter(
        torch.randn(VOCAB_SIZE, reference.INPUT_DIM, dtype=torch.float32) * 0.05
    )
    initial_tensors = {
        name: tensor.detach().clone().contiguous()
        for name, tensor in reference.package_tensors(p).items()
    }
    initial_tensors["lm_head.weight"] = lm_weight.detach().clone().contiguous()

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

    shared_state_value = (
        torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE) * 0.21
    )
    shadow_state = shared_state_value.clone().requires_grad_()
    committed_state = shared_state_value.clone().requires_grad_()
    shadow_x = (
        torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH) * 0.19
    ).requires_grad_()
    committed_x = (
        torch.randn(COMMITTED_STEPS, reference.BATCH, reference.WIDTH) * 0.23
    ).requires_grad_()
    shadow_token_ids = torch.tensor([[4], [9], [4], [21]], dtype=torch.long)
    committed_token_ids = torch.tensor([[13]], dtype=torch.long)
    shadow_grad_output = (
        torch.randn(reference.STEPS, reference.BATCH, reference.WIDTH) * 0.021
    )
    committed_grad_output = (
        torch.randn(COMMITTED_STEPS, reference.BATCH, reference.WIDTH) * 0.027
    )
    shadow_final_grad = (
        torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE) * 0.010
    )
    committed_final_grad = (
        torch.randn(reference.BATCH, reference.WIDTH, reference.STATE_SIZE) * 0.013
    )

    optimizer.zero_grad(set_to_none=True)
    shadow_objective, _, _, shadow_features = recurrent_branch(
        p,
        lm_weight,
        shadow_x,
        shadow_token_ids,
        shadow_state,
        shadow_grad_output,
        shadow_final_grad,
    )
    committed_objective, committed_outputs, committed_final_state, committed_features = (
        recurrent_branch(
            p,
            lm_weight,
            committed_x,
            committed_token_ids,
            committed_state,
            committed_grad_output,
            committed_final_grad,
        )
    )
    (shadow_objective + committed_objective).backward()
    if (
        committed_x.grad is None
        or committed_state.grad is None
        or committed_features.grad is None
        or shadow_features.grad is None
    ):
        raise AssertionError("PyTorch fork gradients were not retained")
    committed_expected = {
        "outputs": committed_outputs.detach().clone(),
        "final_state": committed_final_state.detach().clone(),
        "grad_x": committed_x.grad.detach().clone(),
        "token_grad": committed_features.grad.detach().clone(),
        "initial_state_grad": committed_state.grad.detach().clone(),
    }
    optimizer.step()

    def branch_case(
        x: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
        grad_output: torch.Tensor,
        final_grad: torch.Tensor,
    ) -> dict[str, object]:
        return {
            "batch": reference.BATCH,
            "steps": x.shape[0],
            "detach_every_n_steps": reference.DETACH_EVERY,
            "x_sequence": x.detach().flatten().tolist(),
            "token_id_sequence": token_ids.flatten().tolist(),
            "initial_packed_state": state.detach().flatten().tolist(),
            "grad_output_sequence": grad_output.flatten().tolist(),
            "final_packed_state_grad": final_grad.flatten().tolist(),
        }

    case = {
        "width": reference.WIDTH,
        "head_size": reference.HEAD_SIZE,
        "state_mode": "explicit-output",
        "state_clamp": reference.STATE_CLAMP,
        "shadow": branch_case(
            shadow_x,
            shadow_token_ids,
            shadow_state,
            shadow_grad_output,
            shadow_final_grad,
        ),
        "committed": branch_case(
            committed_x,
            committed_token_ids,
            committed_state,
            committed_grad_output,
            committed_final_grad,
        ),
        "optimizer": {
            "lr": train_reference.LR,
            "beta1": train_reference.BETA1,
            "beta2": train_reference.BETA2,
            "eps": train_reference.EPS,
            "weight_decay": train_reference.WEIGHT_DECAY,
        },
    }

    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-tbptt-fork-") as temp_dir:
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
                "hierarchos-vulkan-rwkv-tbptt-fork-parity",
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
                "Vulkan recurrent fork runner failed:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        result = json.loads(completed.stdout)

    assert result["optimizer_step"] == 1, result
    assert result["optimizer_tensor_count"] == len(train_reference.PARAMETER_MAP), result
    assert result["tied_embedding_optimizer_step"] == 1, result
    actual_parameters = {
        item["name"]: torch.tensor(item["values"], dtype=torch.float32)
        for item in result["parameters"]
    }
    expected_names = set(train_reference.PARAMETER_MAP) | {"lm_head.weight"}
    if set(actual_parameters) != expected_names:
        raise AssertionError(
            f"fork parameter names mismatch: {sorted(set(actual_parameters) ^ expected_names)}"
        )

    max_parameter_diff = 0.0
    worst_parameter = ""
    for rust_name, pytorch_name in train_reference.PARAMETER_MAP.items():
        expected = p[pytorch_name].detach().flatten()
        actual = actual_parameters[rust_name].reshape_as(expected)
        diff = (actual - expected).abs().max().item()
        if diff > max_parameter_diff:
            max_parameter_diff = diff
            worst_parameter = rust_name
        torch.testing.assert_close(actual, expected, rtol=4.0e-3, atol=4.0e-5)
    actual_lm = actual_parameters["lm_head.weight"].reshape_as(lm_weight)
    lm_diff = (actual_lm - lm_weight.detach()).abs().max().item()
    if lm_diff > max_parameter_diff:
        max_parameter_diff = lm_diff
        worst_parameter = "lm_head.weight"
    torch.testing.assert_close(
        actual_lm, lm_weight.detach(), rtol=4.0e-3, atol=4.0e-5
    )

    result_map = {
        "committed_outputs": "outputs",
        "committed_final_packed_state": "final_state",
        "committed_grad_x": "grad_x",
        "committed_token_feature_grad": "token_grad",
        "committed_grad_initial_packed_state": "initial_state_grad",
    }
    max_committed_diff = 0.0
    worst_committed = ""
    for actual_name, expected_name in result_map.items():
        expected = committed_expected[expected_name].flatten()
        actual = torch.tensor(result[actual_name], dtype=torch.float32).reshape_as(expected)
        diff = (actual - expected).abs().max().item()
        if diff > max_committed_diff:
            max_committed_diff = diff
            worst_committed = actual_name
        torch.testing.assert_close(actual, expected, rtol=4.0e-3, atol=4.0e-5)

    print(f"device={result['device']} shadow_steps={reference.STEPS} committed_steps=1")
    print(
        f"worst_parameter={worst_parameter} max_abs_parameter_diff={max_parameter_diff:.9g}"
    )
    print(
        f"worst_committed_value={worst_committed} "
        f"max_abs_committed_diff={max_committed_diff:.9g}"
    )
    print("Hierarchos Vulkan shadow-chain/committed-restart fork parity: PASS")


if __name__ == "__main__":
    main()
