from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hierarchos import AttrDict, HierarchosCore
from hierarchos.training.trainer import (
    HIERARCHOS_PEFT_TARGET_MODULES,
    _apply_runtime_model_config_overrides,
    account_skipped_training_batch,
    build_exact_resume_identity,
    build_training_checkpoint,
    configure_checkpoint_rng_policy,
    ensure_finetune_training_mode,
    restore_model_grad_state,
    train_step,
    train_step_skip_reason,
    validate_exact_resume_identity,
)
from hierarchos.utils.checkpoint import load_model_state_dict_compatible


def _step_args(**overrides):
    values = {
        "amp": False,
        "amp_dtype": "float32",
        "training_chunk_size": 2,
        "full_sample_bptt": False,
        "full_sample_activation_checkpointing": False,
        "full_sample_checkpoint_segment_size": 2,
        "persist_state": False,
        "compile": False,
        "compile_pad_to_chunk_size": False,
        "pad_token_id": 0,
        "padding_metrics": False,
        "cpu_chunked_lm_loss": True,
        "cuda_chunked_lm_loss": False,
        "grad_clip": 100.0,
        "ltm_training_mode": "read-only",
        "accumulation_normalization": "microbatch",
        "max_ce_loss_for_backward": 0.0,
        "max_ponder_cost_for_backward": 0.0,
        "max_commitment_cost_for_backward": 2.0,
        "ponder_loss_weight": 0.0,
        "adaptive_ponder": False,
        "ponder_target_scale": 0.5,
        "ponder_objective": "symmetric-huber",
        "ponder_huber_beta": 0.5,
        "max_h_steps": 5,
        "min_h_steps": 1,
        "encourage_thinking": False,
        "commitment_loss_weight": 0.0,
        "ltm_value_alignment_weight": 0.0,
        "recurrent_state_clamp": 50.0,
        "context_state_clamp": 50.0,
        "drift_state_clamp": 5.0,
        "drift_norm_clamp": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _RecurrentAdapterModel(nn.Module):
    """Small causal recurrence with one trainable adapter and frozen base."""

    def __init__(self):
        super().__init__()
        self.frozen_base = nn.Parameter(
            torch.tensor(0.125),
            requires_grad=False,
        )
        self.adapter = nn.Parameter(torch.tensor(0.2))
        self.config = SimpleNamespace(
            vocab_size=32,
            h_stride=2,
            architecture_revision="coherent-v9",
            cpu_chunked_lm_loss=True,
            cuda_chunked_lm_loss=False,
        )
        self.calls = []
        self.reset_count = 0

    def reset_memory(self):
        self.reset_count += 1

    def forward(self, **kwargs):
        assert self.training
        assert kwargs["return_logits"] is False
        input_ids = kwargs["input_ids"]
        labels = kwargs["labels"]
        loss_weights = kwargs.get("loss_weights")
        h_state = kwargs.get("h_state")
        state = (
            h_state
            if h_state is not None
            else self.adapter.new_zeros((input_ids.shape[0], 1))
        )
        initial_state = state.detach().clone()
        numerator = self.adapter * 0.0
        denominator = self.adapter.new_zeros(())
        for token_index in range(input_ids.shape[1]):
            token = input_ids[:, token_index : token_index + 1].float()
            state = (
                state
                + self.adapter * (token + 1.0) * 0.03
                + self.frozen_base * 0.01
            )
            label_index = token_index + 1
            if label_index >= labels.shape[1]:
                continue
            target = labels[:, label_index]
            valid = target != -100
            weight = valid.float()
            if loss_weights is not None:
                weight = weight * loss_weights[:, label_index].float()
            error = (
                state.squeeze(-1)
                - target.clamp_min(0).float() / 10.0
            ).square()
            numerator = numerator + (error * weight).sum()
            denominator = denominator + weight.sum()
        loss = numerator / denominator.clamp_min(1.0)
        self.calls.append(
            {
                "offset": int(kwargs["global_pos_offset"]),
                "length": int(input_ids.shape[1]),
                "initial_h": initial_state,
                "final_h": state.detach().clone(),
            }
        )
        zero = self.adapter * 0.0
        return {
            "loss": loss,
            "ponder_cost": zero,
            "commitment_cost": zero,
            "ltm_value_alignment_cost": None,
            "raw_topk_vals": None,
            "topk_idx": None,
            "ltm_memory_state": None,
            "h_state": state,
            "l_state": state * 0.5,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _PonderAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_depth = nn.Parameter(torch.tensor(3.0))
        self.config = SimpleNamespace(
            vocab_size=16,
            h_stride=2,
            architecture_revision="coherent-v9",
            cpu_chunked_lm_loss=True,
            cuda_chunked_lm_loss=False,
        )

    def reset_memory(self):
        return None

    def forward(self, **kwargs):
        assert self.training
        zero = self.adapter_depth * 0.0
        state = zero.expand(kwargs["input_ids"].shape[0], 1)
        return {
            "loss": zero + 2.0,
            "ponder_cost": self.adapter_depth,
            "commitment_cost": zero,
            "ltm_value_alignment_cost": None,
            "raw_topk_vals": None,
            "topk_idx": None,
            "ltm_memory_state": None,
            "h_state": state,
            "l_state": state,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


def _batch(length=6, vocab_size=32):
    tokens = torch.arange(1, length + 1).remainder(vocab_size)
    return {
        "input_ids": tokens.unsqueeze(0).long(),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "labels": tokens.unsqueeze(0).long(),
    }


def _backward_without_step(model, args, batch):
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=2,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
        force_optimizer_step=False,
        accumulation_divisor=1,
    )
    return outputs, states


def test_finetune_adapter_dense_and_checkpointed_chunk_gradients_match():
    dense = _RecurrentAdapterModel().eval()
    segmented = _RecurrentAdapterModel().eval()
    segmented.load_state_dict(dense.state_dict())
    ensure_finetune_training_mode(dense)
    ensure_finetune_training_mode(segmented)

    dense_outputs, _ = _backward_without_step(
        dense,
        _step_args(
            full_sample_bptt=True,
            full_sample_activation_checkpointing=False,
        ),
        _batch(),
    )
    segmented_outputs, _ = _backward_without_step(
        segmented,
        _step_args(
            full_sample_bptt=True,
            full_sample_activation_checkpointing=True,
            full_sample_checkpoint_segment_size=2,
        ),
        _batch(),
    )

    assert dense.training and segmented.training
    torch.testing.assert_close(
        dense_outputs["loss"],
        segmented_outputs["loss"],
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        dense.adapter.grad,
        segmented.adapter.grad,
        rtol=1e-6,
        atol=1e-7,
    )
    assert dense.frozen_base.grad is None
    assert segmented.frozen_base.grad is None


def test_finetune_tbptt_uses_exact_chunk_geometry_and_carries_state():
    model = ensure_finetune_training_mode(_RecurrentAdapterModel().eval())
    outputs, terminal_states = _backward_without_step(
        model,
        _step_args(training_chunk_size=2),
        _batch(),
    )

    assert outputs is not None
    assert [call["offset"] for call in model.calls] == [0, 2, 4]
    assert [call["length"] for call in model.calls] == [2, 2, 2]
    for previous, current in zip(model.calls, model.calls[1:]):
        torch.testing.assert_close(
            current["initial_h"],
            previous["final_h"],
        )
    torch.testing.assert_close(
        terminal_states[0],
        model.calls[-1]["final_h"],
    )


@pytest.mark.parametrize(
    ("encourage_thinking", "adaptive_ponder", "expected_sign"),
    [
        (False, True, 1),
        (True, True, -1),
    ],
)
def test_finetune_canonical_objective_controls_adapter_depth_direction(
    encourage_thinking,
    adaptive_ponder,
    expected_sign,
):
    model = ensure_finetune_training_mode(_PonderAdapterModel().eval())
    _backward_without_step(
        model,
        _step_args(
            training_chunk_size=8,
            ponder_loss_weight=1.0,
            adaptive_ponder=adaptive_ponder,
            encourage_thinking=encourage_thinking,
        ),
        _batch(length=3, vocab_size=16),
    )

    assert model.adapter_depth.grad is not None
    assert int(torch.sign(model.adapter_depth.grad).item()) == expected_sign


def test_shared_finetune_skip_budget_fails_closed():
    args = SimpleNamespace(
        _train_step_had_nonfinite=False,
        _train_step_had_oom=False,
        _train_step_had_empty_supervision=True,
        _train_step_had_backward=False,
        _accumulation_weighted_token_mass=7.0,
        _skipped_train_batches=0,
        max_skipped_train_batches=1,
    )
    reason = train_step_skip_reason(args)
    assert reason == "empty supervised-token mass"
    assert account_skipped_training_batch(
        args,
        reason=reason,
        epoch=1,
        step=1,
        scope="Fine-tune",
    )
    assert args._accumulation_weighted_token_mass == 0.0
    with pytest.raises(RuntimeError, match="Fine-tune skip/error budget"):
        account_skipped_training_batch(
            args,
            reason=reason,
            epoch=1,
            step=2,
            scope="Fine-tune",
        )


class _TinyTokenizer:
    def __len__(self):
        return 32


class _TinyLoader:
    def __init__(self):
        self.dataset = [0, 1]
        self.num_workers = 0


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("grad_clip", 0.5),
        ("max_ce_loss_for_backward", 4.0),
        ("amp_dtype", "float16"),
        ("_resolved_training_backend", "cpu"),
        ("cpu_loss_chunk_rows", 17),
        ("compile_mode", "reduce-overhead"),
    ],
)
def test_exact_resume_binds_gradient_and_numerical_policy(field, changed):
    args = SimpleNamespace(
        grad_clip=1.0,
        max_ce_loss_for_backward=0.0,
        amp=True,
        amp_dtype="bfloat16",
        _resolved_training_backend="cuda",
        cpu_chunked_lm_loss=True,
        cpu_loss_chunk_rows=0,
        cuda_chunked_lm_loss=True,
        cuda_loss_chunk_rows=0,
        compile=True,
        force_compile=True,
        compile_mode="max-autotune-no-cudagraphs",
        compile_static_worker_loop=True,
        compile_pad_to_chunk_size=True,
        _tokenizer_identity={"vocab_size": 32, "sha256": "a" * 64},
    )
    loader = _TinyLoader()
    saved_identity = build_exact_resume_identity(
        args,
        _TinyTokenizer(),
        loader,
        dataloader_len=2,
    )
    changed_args = SimpleNamespace(**deepcopy(vars(args)))
    setattr(changed_args, field, changed)
    current_identity = build_exact_resume_identity(
        changed_args,
        _TinyTokenizer(),
        loader,
        dataloader_len=2,
    )

    with pytest.raises(RuntimeError, match=field):
        validate_exact_resume_identity(
            {
                "run_identity": saved_identity,
                "mid_epoch_step": 0,
            },
            current_identity,
            "checkpoint.pt",
        )


def _tiny_core_config():
    return AttrDict(
        vocab_size=32,
        model_type="hierarchos",
        context_dim=8,
        persistent_dim=4,
        ltm_slots=8,
        ltm_key_dim=4,
        ltm_val_dim=4,
        ltm_topk=2,
        h_hidden=8,
        l_hidden=8,
        max_h_steps=1,
        max_l_steps=1,
        h_stride=2,
        l_conv_atol=1e-4,
        use_deepembed=True,
        use_rosa=False,
        rosa_max_context=16,
        compile=False,
        gradient_checkpointing=False,
        detach_every_n_steps=None,
        cpu_chunked_lm_loss=True,
        cpu_loss_chunk_rows=0,
        cuda_chunked_lm_loss=False,
        architecture_revision="coherent-v9",
        ltm_training_mode="read-only",
        rwkv_head_size=4,
        full_sample_bptt=False,
    )


def _real_peft_model(*, dropout=0.0):
    peft = pytest.importorskip("peft")
    config = _tiny_core_config()
    base = HierarchosCore(config)
    lora_config = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=list(HIERARCHOS_PEFT_TARGET_MODULES),
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"],
    )
    return peft.get_peft_model(base, lora_config)


def test_real_peft_wrapper_runs_canonical_step_and_resumes_exact_state():
    model = ensure_finetune_training_mode(_real_peft_model().eval())
    assert model.config.model_type == "hierarchos"
    assert model.config.commitment_cost_mode == "mean-square"
    assert model.config.commitment_threshold == pytest.approx(0.1 / 8)
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(".lora_A." in name for name in trainable_names)
    assert any("ltm.modules_to_save.default.keys" in name for name in trainable_names)

    ltm_copy = model.base_model.model.ltm.modules_to_save["default"]
    fast_before = ltm_copy.fast_vals.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    args = _step_args()
    outputs, states = train_step(
        model,
        _batch(length=5),
        optimizer,
        scaler=None,
        accumulation_steps=2,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
        force_optimizer_step=False,
        accumulation_divisor=1,
    )
    assert outputs is not None
    assert args._train_step_had_backward
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if ".lora_B." in name
    )
    # Read-only LTM still trains slow keys/values but never performs a
    # label-derived fast-memory write.
    torch.testing.assert_close(ltm_copy.fast_vals, fast_before)

    args._run_identity = None
    args._optimizer_grouping_version = 2
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=1,
        running_states=states,
    )
    resumed = ensure_finetune_training_mode(_real_peft_model().eval())
    load_result = load_model_state_dict_compatible(
        resumed,
        checkpoint["model_state_dict"],
        "in-memory fine-tune checkpoint",
    )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    assert restore_model_grad_state(
        resumed,
        checkpoint["grad_state_dict"],
        torch.device("cpu"),
    )
    original_grads = {
        name.replace("_orig_mod.", ""): parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    resumed_grads = {
        name.replace("_orig_mod.", ""): parameter.grad
        for name, parameter in resumed.named_parameters()
        if parameter.grad is not None
    }
    assert original_grads.keys() == resumed_grads.keys()
    for name in original_grads:
        torch.testing.assert_close(
            original_grads[name],
            resumed_grads[name],
        )


def test_real_lora_dropout_enables_checkpoint_rng_replay():
    model = ensure_finetune_training_mode(
        _real_peft_model(dropout=0.05).eval()
    )
    assert configure_checkpoint_rng_policy(model) is True
    assert model._hierarchos_checkpoint_preserve_rng_state is True


def test_runtime_finetune_threshold_preserves_checkpoint_or_resolves_default():
    calibrated = AttrDict(
        architecture_revision="coherent-v9",
        context_dim=448,
        commitment_threshold=0.1 / 448,
    )
    _apply_runtime_model_config_overrides(
        calibrated,
        SimpleNamespace(commitment_threshold=None),
    )
    assert calibrated.commitment_threshold == pytest.approx(0.1 / 448)

    unresolved = AttrDict(
        architecture_revision="coherent-v9",
        context_dim=448,
    )
    _apply_runtime_model_config_overrides(
        unresolved,
        SimpleNamespace(commitment_threshold=None),
    )
    assert unresolved.commitment_threshold == pytest.approx(0.1 / 448)

    explicit_ablation = AttrDict(
        architecture_revision="coherent-v9",
        context_dim=448,
        commitment_threshold=0.1 / 448,
    )
    _apply_runtime_model_config_overrides(
        explicit_ablation,
        SimpleNamespace(commitment_threshold=0.002),
    )
    assert explicit_ablation.commitment_threshold == pytest.approx(0.002)


def test_peft_targets_cover_coherent_shared_adapters_and_memory_routers():
    peft = pytest.importorskip("peft")
    config = _tiny_core_config()
    config.use_rosa = True
    base = HierarchosCore(config)
    wrapped = peft.get_peft_model(
        base,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=list(HIERARCHOS_PEFT_TARGET_MODULES),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["ltm"],
        ),
    )
    trainable_names = {
        name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    }
    expected_targets = (
        "h_deepembed_adapter.down",
        "h_deepembed_adapter.up",
        "l_deepembed_adapter.down",
        "l_deepembed_adapter.up",
        "rosa_adapter.down",
        "rosa_adapter.up",
        "rosa_router",
        "ltm_router",
    )
    for target in expected_targets:
        assert any(
            target in name and ".lora_A." in name
            for name in trainable_names
        ), target
