#!/usr/bin/env python3
"""Verify ROSA -> qproj -> LTM top-k/gating -> in_proj on Vulkan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos import HierarchosCore
from hierarchos.models.core import _finite_clamp
from hierarchos.utils.rosa import ROSAState, _rosa_incremental
from tools.export_rust_inference import export_model
from tools.verify_rust_inference_parity import tiny_coherent_config


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


def _make_nontrivial_memory_fixture(
    model: HierarchosCore,
    config,
    *,
    clamp_stress: bool = False,
) -> None:
    config.memory_gate_warmup_steps = 10
    config.memory_gate_warmup_floor = 0.35
    with torch.no_grad():
        model.memory_gate_warmup_step.fill_(3.0)
        model.rosa_adapter.up.weight.normal_(mean=0.0, std=0.035)
        model.rosa_adapter.bias.normal_(mean=0.0, std=0.01)
        model.rosa_router.weight.normal_(mean=0.0, std=0.025)
        model.rosa_router.bias.fill_(0.07)
        model.rosa_gate_logit.fill_(-0.35)
        model.ltm_router.weight.normal_(mean=0.0, std=0.02)
        model.ltm_router.bias.fill_(-0.04)
        model.ltm_gate_logit.fill_(-0.6)
        model.ltm.fast_vals.normal_(mean=0.0, std=0.012)
        if clamp_stress:
            # Exercise the exact finite-preserving safety bounds used by the
            # training graph.  Router weights are zeroed so the gate preimages
            # are deterministically outside +/-50 for every row.  qproj is
            # driven exclusively by prev_context[:, 0], which the fixture pins
            # to +1 below, making every query coordinate exactly +/-24 before
            # the PyTorch/Vulkan +/-12 clamp.
            model.rosa_router.weight.zero_()
            model.rosa_router.bias.zero_()
            model.rosa_gate_logit.fill_(80.0)
            model.ltm_router.weight.zero_()
            model.ltm_router.bias.zero_()
            model.ltm_gate_logit.fill_(-80.0)
            model.qproj.weight.zero_()
            source_column = int(config.context_dim)
            for row in range(model.qproj.weight.shape[0]):
                model.qproj.weight[row, source_column] = 24.0 if row % 2 == 0 else -24.0


def _reference(
    model: HierarchosCore,
    token_ids: torch.Tensor,
    prev_context: torch.Tensor,
    grad_enc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    prev_context = prev_context.detach().clone().requires_grad_(True)
    raw_token = F.embedding(token_ids, model.lm_head.weight)

    rosa_state = ROSAState.new()
    rosa_cap = (
        int(model.config.rosa_max_context)
        if bool(model.config.enforce_rosa_max_context)
        else 0
    )
    predictions = _rosa_incremental(rosa_state, token_ids.tolist(), rosa_cap)
    prediction_tensor = torch.tensor(predictions, dtype=torch.long, device=token_ids.device)
    rosa_valid = (prediction_tensor >= 0) & (prediction_tensor < model.config.vocab_size)
    safe_predictions = prediction_tensor.clamp(min=0, max=model.config.vocab_size - 1)
    rosa_raw = F.embedding(safe_predictions, model.lm_head.weight)
    rosa_features = model.rosa_adapter(rosa_raw)
    rosa_gate_pre = model.rosa_gate_logit + model.rosa_router(raw_token)
    rosa_gate = torch.sigmoid(
        _finite_clamp(
            rosa_gate_pre,
            50.0,
        )
    )
    rosa_gate = model._apply_memory_gate_warmup(rosa_gate)
    token_x = raw_token + rosa_gate * rosa_features * rosa_valid.unsqueeze(-1)

    q_input = torch.cat([token_x, prev_context], dim=-1)
    raw_query = model.qproj(q_input)
    query = _finite_clamp(raw_query, 12.0)
    similarity = (query @ model.ltm.keys.t()) * (model.config.ltm_key_dim ** -0.5)
    topk_indices = torch.topk(
        similarity.detach(),
        k=model.config.ltm_topk,
        dim=-1,
    ).indices

    memory = model.ltm.vals + model.ltm.fast_vals
    gathered = memory.index_select(0, topk_indices.reshape(-1)).view(
        token_ids.numel(), model.config.ltm_topk, model.config.ltm_val_dim
    )
    selected_sim = torch.gather(similarity, dim=-1, index=topk_indices)
    score_signal = (selected_sim - selected_sim.detach()).unsqueeze(-1)
    gathered = (
        gathered
        + score_signal.to(dtype=gathered.dtype)
        * gathered.detach()
        * float(model.ltm.score_grad_scale)
    )
    ltm_gate_pre = model.ltm_gate_logit + model.ltm_router(token_x)
    ltm_gate = torch.sigmoid(
        _finite_clamp(
            ltm_gate_pre,
            50.0,
        )
    )
    ltm_gate = model._apply_memory_gate_warmup(ltm_gate).unsqueeze(1)
    gated_ltm = gathered * ltm_gate
    persistent = model.persistent.unsqueeze(0).expand(token_ids.numel(), -1)
    mac_input = torch.cat([token_x, persistent, gated_ltm.flatten(1)], dim=-1)
    enc = _finite_clamp(F.gelu(model.in_proj(mac_input)), 30.0)
    torch.autograd.backward(enc, grad_enc)

    return {
        "predictions": torch.tensor(predictions, dtype=torch.long, device=token_ids.device),
        "raw_token": raw_token.detach(),
        "token_x": token_x.detach(),
        "rosa_gate_pre": rosa_gate_pre.detach(),
        "raw_query": raw_query.detach(),
        "query": query.detach(),
        "ltm_gate_pre": ltm_gate_pre.detach(),
        "topk": topk_indices.detach(),
        "gated_ltm": gated_ltm.detach(),
        "enc": enc.detach(),
        "grad_prev_context": prev_context.grad.detach(),
        "grad_persistent": model.persistent.grad.detach(),
        "grad_lm_head_weight": model.lm_head.weight.grad.detach(),
        "grad_rosa_adapter_down_weight": model.rosa_adapter.down.weight.grad.detach(),
        "grad_rosa_adapter_up_weight": model.rosa_adapter.up.weight.grad.detach(),
        "grad_rosa_adapter_bias": model.rosa_adapter.bias.grad.detach(),
        "grad_rosa_gate_logit": model.rosa_gate_logit.grad.detach(),
        "grad_rosa_router_weight": model.rosa_router.weight.grad.detach(),
        "grad_rosa_router_bias": model.rosa_router.bias.grad.detach(),
        "grad_qproj_weight": model.qproj.weight.grad.detach(),
        "grad_ltm_keys": model.ltm.keys.grad.detach(),
        "grad_ltm_vals": model.ltm.vals.grad.detach(),
        "grad_ltm_gate_logit": model.ltm_gate_logit.grad.detach(),
        "grad_ltm_router_weight": model.ltm_router.weight.grad.detach(),
        "grad_ltm_router_bias": model.ltm_router.bias.grad.detach(),
        "grad_in_proj_weight": model.in_proj.weight.grad.detach(),
        "grad_in_proj_bias": model.in_proj.bias.grad.detach(),
    }


def _alignment_reference(
    model: HierarchosCore,
    target_hidden: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    target = target_hidden.detach()
    memory_offset = model.config.context_dim + model.config.persistent_dim
    memory_width = model.config.ltm_topk * model.config.ltm_val_dim
    readout = (
        model.in_proj.weight[:, memory_offset : memory_offset + memory_width]
        .view(model.config.context_dim, model.config.ltm_topk, model.config.ltm_val_dim)
        .sum(dim=1)
        .detach()
    )
    value_to_store = model.val_proj(target)
    memory_readback = F.linear(value_to_store, readout)
    sqerr = (memory_readback - target).square().mean(dim=-1)
    target_energy = target.square().mean(dim=-1).clamp_min(1.0e-4)
    row_cost = sqerr / target_energy
    row_cost.mean().backward()
    if model.in_proj.weight.grad is not None:
        raise AssertionError("PyTorch LTM value-alignment reference leaked gradient into in_proj")
    if model.val_proj.weight.grad is None:
        raise AssertionError("PyTorch LTM value-alignment reference produced no val_proj gradient")
    return {
        "row_cost": row_cost.detach(),
        "grad_val_proj_weight": model.val_proj.weight.grad.detach().clone(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clamp-stress",
        action="store_true",
        help=(
            "force saturated ROSA/LTM gate preimages and qproj outputs to qualify "
            "the +/-50 and +/-12 forward/backward clamp semantics"
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(20260814)
    config = tiny_coherent_config(32)
    model = HierarchosCore(config).eval()
    _make_nontrivial_memory_fixture(model, config, clamp_stress=args.clamp_stress)

    token_ids = torch.tensor([2, 5, 2, 5, 2, 5, 7, 2, 5, 2, 5], dtype=torch.long)
    rows = token_ids.numel()
    prev_context = torch.randn(rows, config.context_dim, dtype=torch.float32) * 0.035
    if args.clamp_stress:
        prev_context[:, 0] = 1.0
    grad_enc = torch.randn(rows, config.context_dim, dtype=torch.float32) * 0.04
    alignment_target = torch.randn(rows, config.context_dim, dtype=torch.float32) * 0.06
    expected = _reference(model, token_ids, prev_context, grad_enc)
    expected_alignment = _alignment_reference(model, alignment_target)

    if args.clamp_stress:
        if not bool((expected["rosa_gate_pre"] > 50.0).all().item()):
            raise AssertionError("ROSA gate clamp-stress preimages are no longer above +50")
        if not bool((expected["ltm_gate_pre"] < -50.0).all().item()):
            raise AssertionError("LTM gate clamp-stress preimages are no longer below -50")
        if not bool((expected["raw_query"].abs() > 12.0).all().item()):
            raise AssertionError("qproj clamp-stress preimages are no longer outside +/-12")
        for name in (
            "grad_rosa_gate_logit",
            "grad_rosa_router_weight",
            "grad_rosa_router_bias",
            "grad_qproj_weight",
            "grad_ltm_gate_logit",
            "grad_ltm_router_weight",
            "grad_ltm_router_bias",
        ):
            if not torch.equal(expected[name], torch.zeros_like(expected[name])):
                raise AssertionError(
                    f"PyTorch clamp-stress reference did not fully mask saturated {name}"
                )

    case = {
        "token_ids": token_ids.tolist(),
        "prev_context": prev_context.flatten().tolist(),
        "grad_enc": grad_enc.flatten().tolist(),
        "alignment_target": alignment_target.flatten().tolist(),
    }
    with tempfile.TemporaryDirectory(prefix="hierarchos-vulkan-memory-frontend-") as temp_dir:
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
                    "hierarchos-vulkan-token-memory-frontend-parity",
                    "--",
                    "--model",
                    str(model_dir),
                    "--case",
                    str(case_path),
                ]
            ).stdout
        )

    actual_predictions = torch.tensor(result["rosa_prediction_ids"], dtype=torch.long)
    actual_raw_token = torch.tensor(result["raw_token_features"]).reshape_as(expected["raw_token"])
    actual_token_x = torch.tensor(result["token_features"]).reshape_as(expected["token_x"])
    actual_query = torch.tensor(result["query"]).reshape_as(expected["query"])
    actual_topk = torch.tensor(result["topk_indices"], dtype=torch.long).reshape_as(expected["topk"])
    actual_gated_ltm = torch.tensor(result["gated_ltm_values"]).reshape_as(expected["gated_ltm"])
    actual_enc = torch.tensor(result["enc"]).reshape_as(expected["enc"])
    actual_alignment_cost = torch.tensor(result["ltm_value_alignment_row_cost"]).reshape_as(
        expected_alignment["row_cost"]
    )
    actual_val_proj_grad = torch.tensor(result["grad_val_proj_weight"]).reshape_as(
        expected_alignment["grad_val_proj_weight"]
    )

    torch.testing.assert_close(actual_predictions, expected["predictions"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_raw_token, expected["raw_token"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_token_x, expected["token_x"], rtol=8e-4, atol=8e-6)
    torch.testing.assert_close(actual_query, expected["query"], rtol=1.2e-3, atol=1.2e-5)
    torch.testing.assert_close(actual_topk, expected["topk"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_gated_ltm, expected["gated_ltm"], rtol=1.5e-3, atol=1.5e-5)
    torch.testing.assert_close(actual_enc, expected["enc"], rtol=1.5e-3, atol=1.5e-5)
    torch.testing.assert_close(
        actual_alignment_cost,
        expected_alignment["row_cost"],
        rtol=2.0e-3,
        atol=2.0e-5,
    )
    torch.testing.assert_close(
        actual_val_proj_grad,
        expected_alignment["grad_val_proj_weight"],
        rtol=3.0e-3,
        atol=3.0e-6,
    )

    gradient_shapes = {
        "grad_prev_context": expected["grad_prev_context"],
        "grad_persistent": expected["grad_persistent"],
        "grad_lm_head_weight": expected["grad_lm_head_weight"],
        "grad_rosa_adapter_down_weight": expected["grad_rosa_adapter_down_weight"],
        "grad_rosa_adapter_up_weight": expected["grad_rosa_adapter_up_weight"],
        "grad_rosa_adapter_bias": expected["grad_rosa_adapter_bias"],
        "grad_rosa_router_weight": expected["grad_rosa_router_weight"],
        "grad_rosa_router_bias": expected["grad_rosa_router_bias"],
        "grad_qproj_weight": expected["grad_qproj_weight"],
        "grad_ltm_keys": expected["grad_ltm_keys"],
        "grad_ltm_vals": expected["grad_ltm_vals"],
        "grad_ltm_router_weight": expected["grad_ltm_router_weight"],
        "grad_ltm_router_bias": expected["grad_ltm_router_bias"],
        "grad_in_proj_weight": expected["grad_in_proj_weight"],
        "grad_in_proj_bias": expected["grad_in_proj_bias"],
    }
    gradient_diffs: dict[str, float] = {}
    for name, expected_grad in gradient_shapes.items():
        actual_grad = torch.tensor(result[name]).reshape_as(expected_grad)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3.0e-3, atol=3.0e-6)
        gradient_diffs[name] = (actual_grad - expected_grad).abs().max().item()
    for name in ("grad_rosa_gate_logit", "grad_ltm_gate_logit"):
        actual_grad = torch.tensor(result[name]).reshape_as(expected[name])
        torch.testing.assert_close(actual_grad, expected[name], rtol=3.0e-3, atol=3.0e-6)
        gradient_diffs[name] = (actual_grad - expected[name]).abs().max().item()

    cuda_status = "SKIP(no CUDA device present)"
    if torch.cuda.is_available():
        cuda_model = HierarchosCore(config).cuda().eval()
        cuda_model.load_state_dict(model.state_dict())
        cuda_reference = _reference(
            cuda_model,
            token_ids.cuda(),
            prev_context.cuda(),
            grad_enc.cuda(),
        )
        torch.testing.assert_close(cuda_reference["token_x"].cpu(), expected["token_x"], rtol=8e-4, atol=8e-6)
        torch.testing.assert_close(cuda_reference["enc"].cpu(), expected["enc"], rtol=1.5e-3, atol=1.5e-5)
        torch.testing.assert_close(
            cuda_reference["grad_qproj_weight"].cpu(),
            expected["grad_qproj_weight"],
            rtol=3.0e-3,
            atol=3.0e-6,
        )
        cuda_alignment = _alignment_reference(cuda_model, alignment_target.cuda())
        torch.testing.assert_close(
            cuda_alignment["row_cost"].cpu(),
            expected_alignment["row_cost"],
            rtol=2.0e-3,
            atol=2.0e-5,
        )
        torch.testing.assert_close(
            cuda_alignment["grad_val_proj_weight"].cpu(),
            expected_alignment["grad_val_proj_weight"],
            rtol=3.0e-3,
            atol=3.0e-6,
        )
        cuda_status = "PASS"

    print(f"device={result['device']}")
    print(f"queue_submissions={result['queue_submissions']}")
    print(f"rosa_predictions={actual_predictions.tolist()}")
    print(f"token_x_max_abs={(actual_token_x - expected['token_x']).abs().max().item():.9g}")
    print(f"query_max_abs={(actual_query - expected['query']).abs().max().item():.9g}")
    print(f"topk_exact={bool(torch.equal(actual_topk, expected['topk']))}")
    print(f"gated_ltm_max_abs={(actual_gated_ltm - expected['gated_ltm']).abs().max().item():.9g}")
    print(f"enc_max_abs={(actual_enc - expected['enc']).abs().max().item():.9g}")
    print(f"gradient_max_abs={max(gradient_diffs.values()):.9g}")
    print(f"qproj_grad_max_abs={gradient_diffs['grad_qproj_weight']:.9g}")
    print(f"ltm_key_grad_max_abs={gradient_diffs['grad_ltm_keys']:.9g}")
    print(f"rosa_adapter_grad_max_abs={gradient_diffs['grad_rosa_adapter_down_weight']:.9g}")
    print(
        "ltm_value_alignment_cost_max_abs="
        f"{(actual_alignment_cost - expected_alignment['row_cost']).abs().max().item():.9g}"
    )
    print(
        "val_proj_grad_max_abs="
        f"{(actual_val_proj_grad - expected_alignment['grad_val_proj_weight']).abs().max().item():.9g}"
    )
    print(f"clamp_stress={'PASS' if args.clamp_stress else 'disabled'}")
    print(f"cuda_reference={cuda_status}")
    print("Vulkan ROSA/qproj/LTM front-end + val_proj value-alignment PyTorch parity: PASS")


if __name__ == "__main__":
    main()
