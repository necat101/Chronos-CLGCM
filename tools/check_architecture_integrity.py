#!/usr/bin/env python
"""Architecture integrity checks for the active Hierarchos code paths.

This is intentionally broader than a single unit test. It probes the pieces
that can silently drift apart during rescue runs: RWKV state shape, LTM
train/chat mode alignment, DeepEmbed clamp/decay behavior, checkpoint guards,
quantized-loader architecture status, and the coherent-v9 compute budget.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hierarchos.inference.chat import (
    advance_chat_model_state,
    boundary_drift_seed,
    tbptt_chunk_ranges,
)
from hierarchos.models.core import HierarchosCore
from hierarchos.models.quantized import (
    detect_quantized_rwkv_format,
    validate_quantized_rwkv_format,
)
from hierarchos.training.trainer import (
    build_hierarchos_optimizer,
    train_step,
)
from hierarchos.utils.checkpoint import (
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
)


class Audit:
    def __init__(self, strict: bool = False):
        self.strict = bool(strict)
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes = 0

    def ok(self, message: str) -> None:
        self.passes += 1
        print(f"PASS: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        prefix = "FAIL" if self.strict else "WARN"
        print(f"{prefix}: {message}")
        if self.strict:
            self.failures.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL: {message}")

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.ok(message)
        else:
            self.fail(message)

    def finish(self) -> int:
        print()
        print(f"Architecture audit: {self.passes} passed, {len(self.warnings)} warning(s), {len(self.failures)} failure(s).")
        if self.failures:
            print("Failures:")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        return 0


def tiny_config(**overrides):
    base = dict(
        vocab_size=64,
        max_length=32,
        context_dim=16,
        h_hidden=16,
        l_hidden=16,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_stride=2,
        max_h_steps=3,
        max_l_steps=3,
        rwkv_head_size=4,
        rwkv_n_layer_hint=2,
        use_deepembed=True,
        use_rosa=True,
        memory_token_routers=True,
        rosa_max_context=32,
        compile=False,
        gradient_checkpointing=False,
        training_chunk_size=4,
        detach_every_n_steps=4,
        ltm_lr=1e-3,
        min_ltm_lr=1e-8,
        ltm_momentum=0.9,
        ltm_weight_decay=1e-4,
        val_proj_trained=True,
        ltm_forget_rate=0.01,
        ltm_score_grad_scale=1.0,
        ltm_cpu_gather_retrieval=True,
        ltm_cpu_sparse_update=True,
        isolate_batch_ltm=True,
        memory_gate_warmup_steps=0,
        memory_gate_warmup_floor=0.0,
        activation_clamp=30.0,
        recurrent_state_clamp=20.0,
        context_state_clamp=20.0,
        drift_state_clamp=2.0,
        drift_norm_clamp=1.25,
        drift_delta_scale=0.35,
        commitment_threshold=0.05,
        max_commitment_cost_for_backward=4.0,
        halt_logit_clamp=10.0,
        h_halt_thresh=0.9,
        l_conv_atol=1e-4,
        rwkv_channel_mix_key_clamp=3.0,
        rwkv_channel_mix_deepembed_clamp=2.0,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        z_loss_weight=1e-4,
        debug_numerics=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def tensor_tree_is_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensor_tree_is_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensor_tree_is_finite(v) for v in value if v is not None)
    return True


def make_batch(batch_size=2, length=8, vocab_size=64):
    ids = torch.arange(batch_size * length, dtype=torch.long).reshape(batch_size, length)
    ids = (ids % (vocab_size - 1)) + 1
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
    }


def make_train_args(mode: str):
    return SimpleNamespace(
        amp=False,
        training_chunk_size=4,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode=mode,
        ltm_lr=1e-4,
        min_ltm_lr=1e-8,
        min_lr=1e-8,
        disable_ltm_lr_schedule=True,
        max_ce_loss_for_backward=0.0,
        max_ponder_cost_for_backward=0.0,
        max_commitment_cost_for_backward=4.0,
        commitment_loss_weight=0.5,
        ponder_loss_weight=0.003,
        adaptive_ponder=True,
        ponder_target_scale=0.65,
        max_h_steps=3,
        max_sanitized_gradient_values=0,
    )


def check_core_forward(audit: Audit) -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = HierarchosCore(config)

    audit.require(model.tok_emb.weight is model.lm_head.weight, "token embedding and LM head weights are tied")
    audit.require(model.h_rnn.state_size == 3 + model.h_rnn.head_size, "H RWKV state size is 3 + head_size")
    audit.require(model.l_rnn.state_size == 3 + model.l_rnn.head_size, "L RWKV state size is 3 + head_size")
    audit.require(float(model.h_deepemb.weight.mean().item()) == 1.0, "DeepEmbed starts as identity multiplier")

    batch = make_batch(batch_size=2, length=8, vocab_size=config.vocab_size)
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_raw_topk_values=False,
        return_topk_indices=False,
    )
    audit.require(outputs["raw_topk_vals"] is None, "read-only forward can omit retained LTM tensors")
    audit.require(outputs["topk_idx"] is None, "read-only forward can omit LTM top-k indices")
    audit.require(outputs["logits"].shape == (2, 8, config.vocab_size), "forward logits shape is stable")
    audit.require(outputs["loss"] is not None and torch.isfinite(outputs["loss"]), "forward loss is finite")
    audit.require(tensor_tree_is_finite(outputs), "forward output tensor tree is finite")
    audit.require(
        len(outputs["ltm_memory_state"]) == 7,
        "LTM runtime state carries fast/momentum/ROSA/token-time/source/wall-time metadata",
    )
    drift_norm = outputs["drift_state"].float().norm(dim=-1).max().item()
    audit.require(drift_norm <= config.drift_norm_clamp + 1e-4, "drift state respects L2 norm clamp")

    inner_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_raw_topk_values=True,
        return_topk_indices=True,
    )
    audit.require(isinstance(inner_outputs["raw_topk_vals"], list), "inner-update forward exposes raw retrieval tensors")
    audit.require(inner_outputs["topk_idx"] is not None, "inner-update forward exposes top-k indices")


def check_coherent_v9_efficiency_contract(audit: Audit) -> None:
    """Keep the reference v9 model compact and its cheap inference path exact."""
    reference_overrides = dict(
        vocab_size=50_257,
        context_dim=448,
        h_hidden=448,
        l_hidden=448,
        persistent_dim=128,
        ltm_slots=1_024,
        ltm_key_dim=128,
        ltm_val_dim=128,
        ltm_topk=4,
        max_h_steps=5,
        max_l_steps=5,
        h_stride=4,
        rwkv_head_size=64,
    )
    with torch.device("meta"):
        legacy = HierarchosCore(
            tiny_config(
                **reference_overrides,
                architecture_revision="legacy-v8",
            )
        )
        coherent = HierarchosCore(
            tiny_config(
                **reference_overrides,
                architecture_revision="coherent-v9",
            )
        )
    legacy_count = sum(parameter.numel() for parameter in legacy.parameters())
    coherent_count = sum(
        parameter.numel() for parameter in coherent.parameters()
    )
    audit.require(
        legacy_count == 232_516_229,
        "reference legacy-v8 parameter budget is stable",
    )
    audit.require(
        coherent_count == 30_227_653,
        "reference coherent-v9 parameter budget is stable",
    )
    audit.require(
        coherent_count / legacy_count < 0.131,
        "coherent-v9 removes more than 86.9% of legacy parameters",
    )

    torch.manual_seed(9)
    model = HierarchosCore(
        tiny_config(
            architecture_revision="coherent-v9",
            use_rosa=False,
            isolate_batch_ltm=True,
        )
    )
    model.eval()
    model.suppress_hebbian = True
    ids = torch.tensor(
        [
            [3, 4, 5, 6],
            [7, 8, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    common_kwargs = dict(
        attention_mask=attention_mask,
        suppress_hebbian=True,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
        return_step_telemetry=False,
        return_numerics=False,
    )
    with torch.inference_mode():
        full = model(ids, **common_kwargs)
        last_only = model(
            ids,
            return_last_logit_only=True,
            **common_kwargs,
        )
    expected = torch.stack(
        [
            full["logits"][0, 3],
            full["logits"][1, 1],
        ],
        dim=0,
    ).unsqueeze(1)
    audit.require(
        last_only["logits"].shape == (2, 1, model.config.vocab_size),
        "last-logit inference allocates one vocabulary row per batch row",
    )
    last_only_diff = (last_only["logits"] - expected).abs().max().item()
    audit.require(
        last_only_diff <= 2e-6,
        (
            "last-logit inference selects each right-padded row's final active "
            f"token within FP32 kernel tolerance (max diff {last_only_diff:.3e})"
        ),
    )
    ltm_state = last_only["ltm_memory_state"]
    audit.require(
        ltm_state[0].ndim == 2 and ltm_state[1].ndim == 2,
        "read-only inference shares LTM fast/momentum storage across batch rows",
    )


def check_ltm_train_modes(audit: Audit) -> None:
    torch.manual_seed(11)
    batch = make_batch(batch_size=2, length=8)

    read_only_model = HierarchosCore(tiny_config())
    read_only_model.train()
    optimizer = torch.optim.AdamW(read_only_model.parameters(), lr=1e-3)

    def forbidden_inner_update(*_args, **_kwargs):
        raise AssertionError("read-only mode called LTM inner_update")

    read_only_model.ltm.inner_update = forbidden_inner_update
    outputs, states = train_step(
        read_only_model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=make_train_args("read-only"),
        running_states=(None, None, None, None, None, None),
    )
    audit.require(outputs is not None, "read-only train_step completed")
    audit.require(
        states[5] is not None and len(states[5]) == 7,
        "read-only train_step carries complete LTM/ROSA state",
    )

    inner_model = HierarchosCore(tiny_config())
    inner_model.train()
    optimizer = torch.optim.AdamW(inner_model.parameters(), lr=1e-3)
    original_inner_update = inner_model.ltm.inner_update
    calls = {"count": 0}

    def counted_inner_update(*args, **kwargs):
        calls["count"] += 1
        return original_inner_update(*args, **kwargs)

    inner_model.ltm.inner_update = counted_inner_update
    outputs, _states = train_step(
        inner_model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=make_train_args("inner-update"),
        running_states=(None, None, None, None, None, None),
    )
    audit.require(outputs is not None, "inner-update train_step completed")
    audit.require(calls["count"] > 0, "inner-update mode calls LTM inner_update")


def check_inference_memory_contract(audit: Audit) -> None:
    torch.manual_seed(13)
    model = HierarchosCore(tiny_config(isolate_batch_ltm=False))
    ids = make_batch(batch_size=1, length=6)["input_ids"]

    model.eval()
    model.suppress_hebbian = True
    before = model.ltm.fast_vals.detach().clone()
    with torch.no_grad():
        outputs = model(ids, suppress_hebbian=True)
    after = model.ltm.fast_vals.detach()
    audit.require(torch.equal(before, after), "normal inference suppresses Hebbian LTM writes")
    audit.require(outputs["logits"].shape == (1, 6, model.config.vocab_size), "normal inference logits shape is stable")

    with torch.no_grad():
        model(ids, allow_hebbian_update=True)
    changed = not torch.equal(before, model.ltm.fast_vals.detach())
    audit.require(changed, "explicit validation path can write fast LTM memory")

    model.reset_memory()
    model.config.val_proj_trained = False
    legacy_before = model.ltm.fast_vals.detach().clone()
    with torch.no_grad():
        model(ids, allow_hebbian_update=True)
    audit.require(
        torch.equal(legacy_before, model.ltm.fast_vals.detach()),
        "legacy checkpoints cannot write random val_proj vectors into fast LTM",
    )


def check_hebbian_writer_training_contract(audit: Audit) -> None:
    torch.manual_seed(131)
    model = HierarchosCore(tiny_config(use_rosa=False, use_deepembed=False))
    layout = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    model.eval()
    outputs = model(
        ids,
        labels=ids,
        compute_ltm_value_alignment=True,
        return_logits=False,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
    )
    cost = outputs["ltm_value_alignment_cost"]
    audit.require(cost is not None and torch.isfinite(cost), "Hebbian writer alignment cost is finite")
    cost.backward()

    writer_grad = model.val_proj.weight.grad
    audit.require(
        writer_grad is not None and torch.isfinite(writer_grad).all() and writer_grad.norm().item() > 0,
        "Hebbian writer alignment reaches val_proj",
    )
    other_grads = [
        name
        for name, parameter in model.named_parameters()
        if name != "val_proj.weight" and parameter.grad is not None
    ]
    audit.require(other_grads == [], "Hebbian writer auxiliary is isolated from language weights")
    audit.require(
        layout == {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()},
        "Hebbian writer training preserves checkpoint layout",
    )


def check_streaming_generation_parity(audit: Audit) -> None:
    torch.manual_seed(14)
    model = HierarchosCore(tiny_config(isolate_batch_ltm=False, h_stride=2))
    model.eval()
    model.suppress_hebbian = True

    prompt = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
    continuation = torch.tensor([[7, 8, 9, 10, 11]], dtype=torch.long)

    with torch.no_grad():
        prefill = model(prompt, suppress_hebbian=True, global_pos_offset=0)
        h_state = prefill["h_state"]
        l_state = prefill["l_state"]
        prev_context = prefill["prev_context"]
        target_context = prefill["target_context"]
        ltm_state = prefill["ltm_memory_state"]
        stream_logits = [prefill["logits"][:, -1, :].float()]

        for pos in range(continuation.shape[1]):
            token = continuation[:, pos:pos + 1]
            out = model(
                token,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=None,
                ltm_memory_state=ltm_state,
                suppress_hebbian=True,
                global_pos_offset=prompt.shape[1] + pos,
            )
            h_state = out["h_state"]
            l_state = out["l_state"]
            prev_context = out["prev_context"]
            target_context = out["target_context"]
            ltm_state = out["ltm_memory_state"]
            stream_logits.append(out["logits"][:, -1, :].float())

        full_ids = torch.cat([prompt, continuation], dim=1)
        for k in range(continuation.shape[1] + 1):
            full_prefix = full_ids[:, :prompt.shape[1] + k]
            full_out = model(full_prefix, suppress_hebbian=True, global_pos_offset=0)
            diff = (full_out["logits"][:, -1, :].float() - stream_logits[k]).abs().max().item()
            if diff > 1e-4:
                audit.fail(f"streaming logits diverge from full forward at continuation step {k}: max diff {diff:.3e}")
                return
    audit.ok("streaming one-token inference matches full forward without per-token drift reseeding")


def check_epoch13_tbptt_boundary_parity(audit: Audit) -> None:
    torch.manual_seed(15)
    chunk_size = 4
    model = HierarchosCore(tiny_config(isolate_batch_ltm=False, training_chunk_size=chunk_size, h_stride=2))
    model.eval()
    model.suppress_hebbian = True

    prompt = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    continuation = torch.tensor([[9, 10, 11, 12, 13]], dtype=torch.long)
    full_ids = torch.cat([prompt, continuation], dim=1)

    def run_chunked_reference(ids: torch.Tensor):
        h_state = l_state = prev_context = target_context = drift_state = ltm_state = None
        logits_by_pos = []
        with torch.no_grad():
            for start in range(0, ids.shape[1], chunk_size):
                out = model(
                    ids[:, start:start + chunk_size],
                    h_state=h_state,
                    l_state=l_state,
                    prev_context=prev_context,
                    target_context=target_context,
                    drift_state=drift_state,
                    ltm_memory_state=ltm_state,
                    suppress_hebbian=True,
                    global_pos_offset=start,
                )
                logits_by_pos.extend(out["logits"].float().unbind(dim=1))
                h_state = out["h_state"]
                l_state = out["l_state"]
                prev_context = out["prev_context"]
                target_context = out["target_context"]
                drift_state = out["drift_state"]
                ltm_state = out["ltm_memory_state"]
        return logits_by_pos

    reference_logits = run_chunked_reference(full_ids)

    with torch.no_grad():
        h_state = l_state = prev_context = target_context = drift_state = ltm_state = None
        chunk_drift_state = None
        for start in range(0, prompt.shape[1], chunk_size):
            out = model(
                prompt[:, start:start + chunk_size],
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=chunk_drift_state,
                ltm_memory_state=ltm_state,
                suppress_hebbian=True,
                global_pos_offset=start,
            )
            h_state = out["h_state"]
            l_state = out["l_state"]
            prev_context = out["prev_context"]
            target_context = out["target_context"]
            drift_state = out["drift_state"]
            ltm_state = out["ltm_memory_state"]
            chunk_drift_state = drift_state

        total_seen = int(prompt.shape[1])
        comparisons = [(prompt.shape[1] - 1, out["logits"][:, -1, :].float())]
        for pos in range(continuation.shape[1]):
            token = continuation[:, pos:pos + 1]
            boundary_drift = (
                drift_state
                if total_seen > 0 and total_seen % chunk_size == 0
                else None
            )
            out = model(
                token,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=boundary_drift,
                ltm_memory_state=ltm_state,
                suppress_hebbian=True,
                global_pos_offset=total_seen,
            )
            h_state = out["h_state"]
            l_state = out["l_state"]
            prev_context = out["prev_context"]
            target_context = out["target_context"]
            drift_state = out["drift_state"]
            ltm_state = out["ltm_memory_state"]
            comparisons.append((total_seen, out["logits"][:, -1, :].float()))
            total_seen += 1

        for pos, logits in comparisons:
            diff = (reference_logits[pos] - logits.squeeze(0)).abs().max().item()
            if diff > 1e-4:
                audit.fail(f"boundary-only generation diverges from TBPTT chunk reference at position {pos}: max diff {diff:.3e}")
                return
    audit.ok("boundary-only chat generation matches epoch-13 TBPTT chunk recurrence")


def check_carried_chat_state_parity(audit: Audit) -> None:
    """A turn split inside a TBPTT chunk must not create a new drift boundary."""
    torch.manual_seed(151)
    chunk_size = 4
    model = HierarchosCore(tiny_config(isolate_batch_ltm=False, training_chunk_size=chunk_size, h_stride=2))
    model.eval()
    model.suppress_hebbian = True
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.long)

    def run_fragments(fragment_widths):
        h_state = l_state = prev_context = target_context = drift_state = ltm_state = None
        logits_by_pos = []
        global_offset = 0
        with torch.no_grad():
            for fragment_width in fragment_widths:
                fragment = ids[:, global_offset:global_offset + fragment_width]
                for local_start, local_end in tbptt_chunk_ranges(
                    fragment.shape[1], chunk_size, global_offset=global_offset
                ):
                    absolute_start = global_offset + local_start
                    outputs, state = advance_chat_model_state(
                        model,
                        fragment[:, local_start:local_end],
                        device=torch.device("cpu"),
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        drift_seed=boundary_drift_seed(drift_state, absolute_start, chunk_size),
                        ltm_state=ltm_state,
                        global_pos_offset=absolute_start,
                    )
                    logits_by_pos.extend(outputs["logits"].float().unbind(dim=1))
                    h_state, l_state, prev_context, target_context, drift_state, ltm_state = state
                global_offset += fragment_width
        return logits_by_pos

    reference = run_fragments([ids.shape[1]])
    carried = run_fragments([6, ids.shape[1] - 6])
    max_diff = max((left - right).abs().max().item() for left, right in zip(reference, carried))
    audit.require(
        max_diff <= 1e-4,
        f"carried chat state preserves absolute TBPTT recurrence across an in-chunk turn split (max diff {max_diff:.3e})",
    )


def check_train_eval_parity_after_memory_warmup(audit: Audit) -> None:
    torch.manual_seed(16)
    warmup_steps = 5
    model = HierarchosCore(
        tiny_config(
            architecture_revision="coherent-v9",
            use_deepembed=False,
            use_rosa=False,
            isolate_batch_ltm=False,
            memory_gate_warmup_steps=warmup_steps,
            memory_gate_warmup_floor=0.10,
        )
    )
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)

    with torch.no_grad():
        model.train()
        model.suppress_hebbian = True
        model.set_training_step(0)
        train_warmup_logits = model(
            ids,
            suppress_hebbian=True,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
        )["logits"].float()

        model.eval()
        eval_warmup_logits = model(
            ids,
            suppress_hebbian=True,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
        )["logits"].float()

        model.train()
        model.suppress_hebbian = True
        model.set_training_step(warmup_steps)
        train_done_logits = model(
            ids,
            suppress_hebbian=True,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
        )["logits"].float()

        model.eval()
        model.suppress_hebbian = True
        eval_done_logits = model(
            ids,
            suppress_hebbian=True,
            return_topk_values=False,
            return_raw_topk_values=False,
            return_topk_indices=False,
        )["logits"].float()

    warmup_parity_diff = (
        train_warmup_logits - eval_warmup_logits
    ).abs().max().item()
    audit.require(
        warmup_parity_diff <= 1e-6,
        "coherent-v9 train/eval logits match at memory-gate schedule step 0",
    )
    done_diff = (train_done_logits - eval_done_logits).abs().max().item()
    audit.require(done_diff <= 1e-6, "train/eval logits match after memory-gate warmup is complete")
    schedule_diff = (
        eval_warmup_logits - eval_done_logits
    ).abs().max().item()
    audit.require(
        schedule_diff > 1e-5,
        "persisted memory-gate schedule position changes both train and eval coherently",
    )


def check_clamps_and_optimizer(audit: Audit) -> None:
    torch.manual_seed(17)
    model = HierarchosCore(tiny_config(rwkv_channel_mix_key_clamp=4.0, rwkv_channel_mix_deepembed_clamp=1.5))
    model.config.rwkv_channel_mix_key_clamp = 5.0
    model.config.rwkv_channel_mix_deepembed_clamp = 2.0
    model.refresh_runtime_config()
    audit.require(model.h_rnn.channel_mix_key_clamp == 5.0, "runtime refresh updates H channel-mix key clamp")
    audit.require(model.l_rnn.channel_mix_deepembed_clamp == 2.0, "runtime refresh updates L DeepEmbed clamp")

    x = torch.randn(2, model.config.h_hidden)
    state = model.h_rnn.initial_state(2)
    extreme_deepemb = torch.full((2, model.config.h_hidden * 4), 1e6)
    y, next_state = model.h_rnn(x, state, deepemb_vec=extreme_deepemb)
    audit.require(torch.isfinite(y).all() and torch.isfinite(next_state).all(), "RWKV channel-mix clamps keep extreme DeepEmbed finite")

    opt = build_hierarchos_optimizer(
        model,
        SimpleNamespace(starting_lr=1e-3, rwkv_weight_decay=0.1),
        torch.device("cpu"),
    )
    decay_params = {id(p) for p in opt.param_groups[0]["params"]}
    no_decay_params = {id(p) for p in opt.param_groups[1]["params"]}
    audit.require(id(model.h_deepemb.weight) in no_decay_params, "H DeepEmbed is excluded from AdamW decay")
    audit.require(id(model.l_deepemb.weight) in no_decay_params, "L DeepEmbed is excluded from AdamW decay")
    audit.require(id(model.h_deepemb.weight) not in decay_params, "H DeepEmbed is not in decay group")
    audit.require(id(model.val_proj.weight) in no_decay_params, "LTM val_proj is excluded from AdamW decay")
    audit.require(id(model.val_proj.weight) not in decay_params, "LTM val_proj is not in decay group")


def check_checkpoint_and_quantized_guards(audit: Audit) -> None:
    config = {}
    state = {
        "h_deepemb.weight": torch.empty(4, 16),
        "l_deepemb.weight": torch.empty(4, 16),
        "rosa_emb.weight": torch.empty(65, 16),
        "rosa_gate_logit": torch.empty(()),
        "h_rnn.r_k": torch.empty(4, 4),
    }
    _infer_arch_flags_from_state_dict(config, state)
    audit.require(config["use_deepembed"] is True, "checkpoint backfills DeepEmbed flag")
    audit.require(config["use_rosa"] is True, "checkpoint backfills ROSA flag")
    audit.require(config["rwkv_channel_mix_key_clamp"] == 12.0, "checkpoint backfills channel-mix key clamp")
    audit.require(config["rwkv_channel_mix_deepembed_clamp"] == 4.0, "checkpoint backfills DeepEmbed clamp")

    try:
        _reject_unsupported_rwkv_state_dict({"h_rnn.time_decay": torch.empty(4)}, "legacy.pt")
    except ValueError:
        audit.ok("legacy scalar-RWKV checkpoint is rejected")
    else:
        audit.fail("legacy scalar-RWKV checkpoint was accepted")

    legacy_q = {"h_rnn.time_decay": None, "h_rnn.time_mix_k": None}
    v8_q = {"h_rnn.x_r": None, "h_rnn.r_k": None}
    mixed_q = {"h_rnn.time_decay": None, "h_rnn.x_r": None}
    audit.require(detect_quantized_rwkv_format(legacy_q) == "legacy-scalar", "quantized legacy RWKV format is detected")
    for archive, source, label in (
        (legacy_q, "fake-legacy.npz", "legacy scalar"),
        (v8_q, "fake-v8.npz", "v8 matrix-state"),
        (mixed_q, "fake-mixed.npz", "mixed"),
    ):
        try:
            validate_quantized_rwkv_format(archive, source)
        except ValueError:
            audit.ok(f"{label} quantized archive is intentionally refused")
        else:
            audit.fail(f"{label} quantized archive was accepted")


def check_static_source_contracts(audit: Audit) -> None:
    cli_source = (ROOT / "hierarchos_cli.py").read_text(encoding="utf-8")
    chat_source = (ROOT / "hierarchos" / "inference" / "chat.py").read_text(encoding="utf-8")
    core_source = (ROOT / "hierarchos" / "models" / "core.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "hierarchos" / "training" / "trainer.py").read_text(encoding="utf-8")
    quantized_source = (ROOT / "hierarchos" / "models" / "quantized.py").read_text(encoding="utf-8")
    legacy_monolith_source = (ROOT / "hierarchos.py").read_text(encoding="utf-8")
    benchmarks_source = (ROOT / "hierarchos" / "evaluation" / "benchmarks.py").read_text(encoding="utf-8")
    lm_eval_source = (ROOT / "hierarchos" / "evaluation" / "lm_eval_wrapper.py").read_text(encoding="utf-8")
    expand_source = (ROOT / "expand_model.py").read_text(encoding="utf-8")

    audit.require("model.suppress_hebbian = True" in chat_source, "chat explicitly suppresses normal Hebbian writes")
    audit.require("allow_hebbian_update=True" in chat_source, "chat validation path is the explicit Hebbian write gate")
    audit.require("--carry-chat-state" in cli_source, "chat recurrent state carry is opt-in")
    audit.require("--chat-prefill-chunk-size" in cli_source, "chat exposes TBPTT prefill chunk sizing")
    audit.require("getattr(config, \"training_chunk_size\", 0)" in chat_source, "chat defaults prefill chunking from checkpoint training_chunk_size")
    audit.require("boundary_drift_seed(" in chat_source and "tbptt_chunk_ranges(" in chat_source, "chat feeds drift only at absolute TBPTT boundaries")
    audit.require("if pending_state_token and current_ids is not None:" in chat_source, "chat consumes the final emitted token before carrying or saving state")
    audit.require(
        "_set_online_update_warmup_complete" not in chat_source
        and "set_training_step(" not in chat_source,
        "chat preserves the checkpoint's persisted memory-gate schedule position",
    )
    audit.require(
        '"memory_gate_warmup_step"' in core_source
        and "persistent=True" in core_source,
        "memory-gate schedule position is checkpoint-persistent",
    )
    audit.require("_merge_ltm_state_into_model_state(model, ltm_state)" in chat_source, "chat save path persists active LTM fast-memory state")
    audit.require("--benchmark-preset" in cli_source and "rog-ally" in cli_source, "CLI exposes bounded ROG Ally benchmark preset")
    audit.require('"rog-ally": ("arc_easy", "hellaswag", "truthfulqa_mc1")' in benchmarks_source, "ROG Ally suite stays light and runnable")
    audit.require("for start in range(0, input_ids.shape[1], chunk_size)" in lm_eval_source, "lm-eval wrapper uses chunked TBPTT-style forward")
    audit.require("clear_ltm_working_memory(model)" in cli_source and "model.suppress_hebbian = True" in cli_source, "benchmark mode clears transient LTM and suppresses Hebbian writes")
    audit.require("suppress_hebbian=True" in lm_eval_source, "lm-eval wrapper explicitly suppresses Hebbian writes")
    audit.require("target_context=target_ctx, drift_state=drift_state, ltm_memory_state=ltm_state" in trainer_source, "trainer preserves epoch-13 TBPTT boundary drift default")
    audit.require("ltm_inner_updates_enabled(args)" in trainer_source, "trainer gates supervised LTM inner updates")
    audit.require(
        "advance_cached_rosa_history=rosa_ids is None" in trainer_source
        and 'kwargs.pop("advance_cached_rosa_history", True)' in core_source,
        "cached ROSA training skips duplicate host-history reconstruction",
    )
    audit.require(
        'not hasattr(args, "persist_state")' in trainer_source
        and 'bool(getattr(args, "persist_state", False))' in trainer_source,
        "independent-sample checkpoints omit unused terminal recurrent state",
    )
    audit.require("--ltm-value-alignment-weight" in cli_source, "CLI exposes opt-in Hebbian writer training")
    finetune_source = trainer_source.split("def finetune(", 1)[-1]
    audit.require(
        "outputs, running_states = train_step(" in finetune_source,
        "LoRA finetune delegates to the canonical TBPTT training step",
    )
    audit.require(
        'postfix["ltm_lr"] = "off"' in finetune_source,
        "LoRA finetune reports LTM LR as off in read-only mode",
    )
    audit.require(
        "source_artifact = _load_source_artifact(" in expand_source
        and "source_artifact=source_artifact" in expand_source,
        "model expansion loads and authenticates the source checkpoint only once",
    )
    audit.require(
        'if name == "qproj.weight"' in expand_source
        and 'if name == "l_input_proj.weight"' in expand_source
        and 'if name == "in_proj.weight"' in expand_source,
        "model expansion maps concatenated projections by semantic feature blocks",
    )
    audit.require(
        'source_artifact["state_dict"] = {}' in expand_source
        and "gc.collect()" in expand_source,
        "model expansion releases source and resume-only state before publication",
    )
    audit.require(
        "save_checkpoint_safely(checkpoint_payload" in expand_source
        and "_publish_directory_atomically(" in expand_source
        and "validate_inference_tokenizer_identity(" in expand_source,
        "model expansion stages, authenticates, and atomically publishes a complete package",
    )
    audit.require(
        "Model expansion cannot change vocabulary size" in expand_source
        and '"expansion_provenance": provenance' in expand_source,
        "model expansion binds tokenizer IDs and persists hashed source provenance",
    )
    audit.require(
        'output_dir = path.with_suffix("")' in expand_source
        and "recognized Hierarchos model package" in expand_source,
        "model expansion cannot reinterpret a legacy file output as a broad parent-directory overwrite",
    )
    audit.require(
        '"expansion_provenance"' in chat_source
        and '"expansion_provenance"' in cli_source,
        "inference re-exports preserve expansion provenance",
    )
    audit.require(
        "Quantized .npz inference is intentionally unsupported" in quantized_source,
        "all legacy .npz quantized inference paths fail closed",
    )
    audit.require(
        "Quantized export is intentionally disabled" in cli_source,
        "CLI quantized export fails closed",
    )
    audit.require(
        "ignored the stale/unsupported .npz artifact" in chat_source
        and "load_full_model_with_config" in chat_source,
        "chat falls back to a same-directory full-precision checkpoint",
    )
    audit.require(
        "_LEGACY_QUANTIZATION_UNSUPPORTED" in legacy_monolith_source,
        "legacy monolith marks its obsolete quantized path unsupported",
    )
    audit.require(
        "hierarchos.py is an unsupported historical monolith" in legacy_monolith_source
        and "SystemExit(2)" in legacy_monolith_source,
        "direct execution of the stale monolith fails closed",
    )
    audit.ok(
        "quantized CPU/Vulkan inference is disabled until it implements the "
        "full coherent-v9 recurrence and logit contract"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hierarchos architecture integrity checks.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    audit = Audit(strict=args.strict)
    checks = [
        check_core_forward,
        check_coherent_v9_efficiency_contract,
        check_ltm_train_modes,
        check_inference_memory_contract,
        check_hebbian_writer_training_contract,
        check_streaming_generation_parity,
        check_epoch13_tbptt_boundary_parity,
        check_carried_chat_state_parity,
        check_train_eval_parity_after_memory_warmup,
        check_clamps_and_optimizer,
        check_checkpoint_and_quantized_guards,
        check_static_source_contracts,
    ]
    for check in checks:
        print()
        print(f"== {check.__name__} ==")
        try:
            check(audit)
        except Exception as exc:
            audit.fail(f"{check.__name__} crashed: {exc}")
            traceback.print_exc()
    return audit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
