"""Cheap response-boundary rescue fine-tuning for Hierarchos checkpoints.

This script keeps an existing checkpoint, masks prompt tokens, freezes most of
the model, and trains a small response-facing subset of parameters. It is meant
for salvage passes when a model learned token statistics but free-runs poorly at
the `### Response:` boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
from torch.amp import GradScaler
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos.inference.chat_state import clear_ltm_working_memory
from hierarchos.inference.chat import (
    advance_chat_model_state,
    boundary_drift_seed,
    resolve_inference_prefill_chunk_size,
    tbptt_chunk_ranges,
    uses_full_sample_inference_recurrence,
)
from hierarchos.models.revisions import (
    architecture_contract,
    architecture_contract_hash,
    architecture_default_training_chunk_size,
)
from hierarchos.training.datasets import (
    _compose_prompt_response_sample,
    _format_alpaca_prompt,
)
from hierarchos.training.trainer import (
    account_skipped_training_batch,
    accumulation_divisor_for_step,
    build_hierarchos_optimizer,
    configure_checkpoint_rng_policy,
    ensure_finetune_training_mode,
    get_model_training_step,
    resolve_training_step_offset,
    save_training_checkpoint_if_finite,
    should_step_accumulation,
    train_step,
    train_step_skip_reason,
)
from hierarchos.utils.checkpoint import (
    load_full_model_with_config,
    sanitize_model_state_dict,
)
from hierarchos.utils.tokenizer import (
    tokenizer_identity,
    validate_inference_tokenizer_identity,
)


DEFAULT_PROBES = [
    "Hello",
    "what is 4 + 4",
    "what is 8 + 8",
    "I was thinking about confidence versus arrogance. How does that apply to learning a hard skill?",
    "Write a simple Hello World program in Rust.",
]


TRAINABLE_PRESETS = {
    "head": (
        "tok_emb.weight",
        "out_norm.",
    ),
    "lite": (
        "tok_emb.weight",
        "out_norm.",
        "l_to_out.",
        "context_drift_proj.",
        "l_feedback_proj.",
        "l_input_proj.",
        "l_rnn.output.",
        "l_rnn.value_cm.",
    ),
    "worker": (
        "tok_emb.weight",
        "out_norm.",
        "l_to_out.",
        "context_drift_proj.",
        "l_feedback_proj.",
        "l_input_proj.",
        "l_rnn.",
    ),
    "all": ("",),
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            instruction = str(row.get("instruction") or row.get("Instruction") or "").strip()
            output = str(row.get("output") or row.get("Output") or "").strip()
            if instruction and output:
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable instruction/output rows found in {path}")
    return rows


def encode_row(
    tokenizer,
    row: dict,
    max_length: int,
    train_prompt_tokens: bool = False,
    prompt_loss_weight: float = 0.0,
    response_loss_weight: float = 1.0,
    response_boundary_loss_weight: float = 5.0,
    response_boundary_tokens: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instruction = str(row.get("instruction") or row.get("Instruction") or "")
    input_text = str(row.get("input") or row.get("Input") or "").strip()
    output = str(row.get("output") or row.get("Output") or "")

    prompt = _format_alpaca_prompt(instruction, input_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(output, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "Response rescue requires a tokenizer EOS token so training and chat "
            "share an explicit terminal transition."
        )
    if not prompt_ids:
        raise ValueError("Formatted Alpaca prompt encoded to zero tokens.")
    if not response_ids:
        raise ValueError("A rescue response encoded to zero non-EOS tokens.")

    composed = _compose_prompt_response_sample(
        prompt_ids,
        response_ids,
        int(eos_id),
        max_length=max_length,
        train_prompt_tokens=bool(train_prompt_tokens),
        prompt_loss_weight=prompt_loss_weight,
        response_loss_weight=response_loss_weight,
        response_boundary_loss_weight=response_boundary_loss_weight,
        response_boundary_tokens=response_boundary_tokens,
        min_response_tokens=1,
    )
    if composed is None:
        raise ValueError(
            "Rescue row cannot preserve one response token plus EOS within "
            f"max_length={max_length}."
        )
    input_ids, labels, loss_weights = composed
    if loss_weights is None:
        loss_weights = [1.0] * len(input_ids)

    supervised = [
        label != -100 and float(weight) > 0.0
        for label, weight in zip(labels[1:], loss_weights[1:])
    ]
    if not any(supervised):
        raise ValueError(
            "Rescue row has no positive-weight next-token supervision after "
            "truncation and masking."
        )

    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        torch.tensor(loss_weights, dtype=torch.float32).unsqueeze(0),
    )


def iter_batches(encoded_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_size: int, pad_token_id: int):
    for start in range(0, len(encoded_rows), batch_size):
        chunk = encoded_rows[start:start + batch_size]
        max_len = max(row[0].shape[1] for row in chunk)
        input_batch = torch.full((len(chunk), max_len), int(pad_token_id), dtype=torch.long)
        label_batch = torch.full((len(chunk), max_len), -100, dtype=torch.long)
        mask_batch = torch.zeros((len(chunk), max_len), dtype=torch.long)
        weight_batch = torch.zeros((len(chunk), max_len), dtype=torch.float32)
        for idx, (input_ids, labels, loss_weights) in enumerate(chunk):
            length = input_ids.shape[1]
            input_batch[idx, :length] = input_ids[0]
            label_batch[idx, :length] = labels[0]
            mask_batch[idx, :length] = 1
            weight_batch[idx, :length] = loss_weights[0]
        yield input_batch, label_batch, mask_batch, weight_batch


def select_trainable(model, preset: str, extra_patterns: Iterable[str] = ()) -> tuple[int, int]:
    patterns = tuple(TRAINABLE_PRESETS[preset]) + tuple(extra_patterns)
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        enabled = any(name.startswith(pattern) for pattern in patterns)
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError(f"Preset {preset!r} selected no trainable parameters.")
    return total, trainable


def _config_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _finite_float(value, name: str, *, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}, got {parsed:g}.")
    return parsed


def validate_cli_args(args) -> None:
    positive_ints = (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("accumulation_steps", args.accumulation_steps),
    )
    for name, value in positive_ints:
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero.")
    if int(args.max_length) < 2:
        raise ValueError("--max-length must be at least 2 (one answer token plus EOS).")
    if int(args.max_steps) < 0:
        raise ValueError("--max-steps cannot be negative.")
    if int(args.max_skipped_train_batches) < 0:
        raise ValueError("--max-skipped-train-batches cannot be negative.")
    if int(args.response_boundary_tokens) < 0:
        raise ValueError("--response-boundary-tokens cannot be negative.")

    if _finite_float(args.lr, "--lr", minimum=0.0) <= 0.0:
        raise ValueError("--lr must be greater than zero.")
    if _finite_float(args.grad_clip, "--grad-clip", minimum=0.0) <= 0.0:
        raise ValueError("--grad-clip must be greater than zero.")
    _finite_float(args.weight_decay, "--weight-decay", minimum=0.0)
    _finite_float(args.prompt_loss_weight, "--prompt-loss-weight", minimum=0.0)
    if (
        _finite_float(
            args.response_loss_weight,
            "--response-loss-weight",
            minimum=0.0,
        )
        <= 0.0
    ):
        raise ValueError("--response-loss-weight must be greater than zero.")
    _finite_float(
        args.response_boundary_loss_weight,
        "--response-boundary-loss-weight",
        minimum=0.0,
    )


def validate_training_geometry(config) -> None:
    """Reject a checkpoint whose persisted train/inference geometry conflicts."""
    recurrence_mode = _config_value(config, "inference_recurrence_mode", None)
    if recurrence_mode is None:
        return
    normalized = str(recurrence_mode).strip().lower().replace("_", "-")
    if normalized not in {"tbptt", "full-sample"}:
        raise ValueError(
            "Checkpoint inference_recurrence_mode must be 'tbptt' or "
            f"'full-sample', got {recurrence_mode!r}."
        )
    expected = (
        "full-sample"
        if bool(_config_value(config, "full_sample_bptt", False))
        else "tbptt"
    )
    if normalized != expected:
        raise ValueError(
            "Checkpoint recurrence metadata is internally inconsistent: "
            f"full_sample_bptt implies {expected!r}, while "
            f"inference_recurrence_mode is {normalized!r}."
        )


def validate_loaded_tokenizer(model, config, tokenizer) -> bool:
    model_vocab = _config_value(config, "vocab_size", None)
    tokenizer_vocab = len(tokenizer)
    if model_vocab is not None and int(model_vocab) != int(tokenizer_vocab):
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer_vocab}) does not match checkpoint "
            f"vocabulary ({int(model_vocab)})."
        )
    return validate_inference_tokenizer_identity(
        tokenizer,
        getattr(model, "_hierarchos_checkpoint_metadata", {}),
    )


def build_rescue_train_args(config, cli_args, tokenizer, device) -> SimpleNamespace:
    """Build the narrow canonical trainer contract used by this utility."""
    validate_training_geometry(config)

    use_amp = (
        bool(cli_args.amp)
        if cli_args.amp is not None
        else getattr(device, "type", None) == "cuda"
    )
    if use_amp and getattr(device, "type", None) != "cuda":
        raise ValueError(
            "Rescue AMP is supported only on CUDA; pass --no-amp on CPU."
        )
    amp_dtype = str(_config_value(config, "amp_dtype", "float16")).lower()
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError(
            f"Checkpoint amp_dtype must be float16 or bfloat16, got {amp_dtype!r}."
        )
    if (
        use_amp
        and amp_dtype == "bfloat16"
        and not bool(torch.cuda.is_bf16_supported())
    ):
        amp_dtype = "float16"

    return SimpleNamespace(
        amp=use_amp,
        amp_dtype=amp_dtype,
        accumulation_normalization="weighted-token",
        accumulation_steps=int(cli_args.accumulation_steps),
        adaptive_ponder=bool(_config_value(config, "adaptive_ponder", False)),
        encourage_thinking=bool(_config_value(config, "encourage_thinking", False)),
        ponder_loss_weight=float(_config_value(config, "ponder_loss_weight", 0.01)),
        ponder_target_scale=float(_config_value(config, "ponder_target_scale", 0.5)),
        ponder_objective=str(_config_value(config, "ponder_objective", "auto")),
        ponder_huber_beta=float(_config_value(config, "ponder_huber_beta", 0.5)),
        max_ponder_cost_for_backward=float(
            _config_value(config, "max_ponder_cost_for_backward", 0.0)
        ),
        commitment_loss_weight=float(
            _config_value(config, "commitment_loss_weight", 0.5)
        ),
        max_commitment_cost_for_backward=float(
            _config_value(config, "max_commitment_cost_for_backward", 2.0)
        ),
        max_ce_loss_for_backward=float(
            _config_value(config, "max_ce_loss_for_backward", 0.0)
        ),
        max_h_steps=int(_config_value(config, "max_h_steps", 5)),
        min_h_steps=int(_config_value(config, "min_h_steps", 1)),
        training_chunk_size=int(
            _config_value(
                config,
                "training_chunk_size",
                architecture_default_training_chunk_size(config),
            )
            or 0
        ),
        full_sample_bptt=bool(_config_value(config, "full_sample_bptt", False)),
        full_sample_activation_checkpointing=bool(
            _config_value(config, "full_sample_activation_checkpointing", False)
        ),
        full_sample_checkpoint_segment_size=int(
            _config_value(config, "full_sample_checkpoint_segment_size", 128)
            or 128
        ),
        persist_state=False,
        ltm_training_mode="read-only",
        ltm_value_alignment_weight=0.0,
        compile=False,
        compile_pad_to_chunk_size=False,
        cuda_chunked_lm_loss=bool(
            _config_value(config, "cuda_chunked_lm_loss", True)
        ),
        cpu_chunked_lm_loss=bool(
            _config_value(config, "cpu_chunked_lm_loss", True)
        ),
        pad_token_id=int(tokenizer.pad_token_id),
        vocab_size=int(_config_value(config, "vocab_size", len(tokenizer))),
        grad_clip=float(cli_args.grad_clip),
        max_sanitized_gradient_values=0,
        recurrent_state_clamp=float(
            _config_value(config, "recurrent_state_clamp", 50.0)
        ),
        context_state_clamp=float(
            _config_value(config, "context_state_clamp", 50.0)
        ),
        drift_state_clamp=float(
            _config_value(config, "drift_state_clamp", 5.0)
        ),
        drift_norm_clamp=float(_config_value(config, "drift_norm_clamp", 0.0)),
        padding_metrics=False,
        padding_metric_steps=0,
        train_prompt_tokens=bool(cli_args.train_prompt_tokens),
        strict_all_token_loss=False,
        alpaca=True,
        kayla=False,
        prompt_column=None,
        completion_column=None,
        max_skipped_train_batches=int(cli_args.max_skipped_train_batches),
        _skipped_train_batches=0,
        _optimizer_grouping_version=2,
        starting_lr=float(cli_args.lr),
        rwkv_weight_decay=float(cli_args.weight_decay),
    )


def apply_rescue_runtime_contract(model) -> None:
    """Make the model config match the canonical read-only rescue path.

    ``train_step`` receives a read-only LTM mode, but the core also consults
    ``model.config`` while it creates LTM state.  Keeping a legacy base's
    inner-update flag would make the actual rescue objective and the exported
    architecture contract disagree.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Response rescue model has no serialized config.")
    if isinstance(config, dict):
        config["ltm_training_mode"] = "read-only"
        config["inference_like_ltm_training"] = True
    else:
        config.ltm_training_mode = "read-only"
        config.inference_like_ltm_training = True
    refresh = getattr(model, "refresh_runtime_config", None)
    if callable(refresh):
        refresh()


def format_prompt(text: str) -> str:
    return _format_alpaca_prompt(text, "")


def probe_model(model, tokenizer, device, prompts: Iterable[str], max_tokens: int = 48) -> None:
    if int(max_tokens) < 0:
        raise ValueError("Probe max_tokens cannot be negative.")

    was_training = bool(model.training)
    previous_suppress_hebbian = getattr(model, "suppress_hebbian", False)
    model.eval()
    model.suppress_hebbian = True
    model_config = getattr(model, "config", None)
    exact_full_sample = uses_full_sample_inference_recurrence(model_config)
    prefill_chunk_size = resolve_inference_prefill_chunk_size(model_config)

    try:
        with torch.inference_mode():
            for prompt_text in prompts:
                clear_ltm_working_memory(model)
                prompt = format_prompt(prompt_text)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if not prompt_ids:
                    raise ValueError("Probe prompt encoded to zero tokens.")
                ids = torch.tensor(
                    prompt_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                state = (None, None, None, None, None, None)
                outputs = None
                ranges = (
                    [(0, int(ids.shape[1]))]
                    if exact_full_sample or prefill_chunk_size <= 0
                    else tbptt_chunk_ranges(
                        int(ids.shape[1]),
                        prefill_chunk_size,
                        global_offset=0,
                    )
                )
                for start, end in ranges:
                    (
                        h_state,
                        l_state,
                        prev_context,
                        target_context,
                        drift_state,
                        ltm_state,
                    ) = state
                    outputs, state = advance_chat_model_state(
                        model,
                        ids[:, start:end],
                        device=device,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        drift_seed=boundary_drift_seed(
                            drift_state,
                            start,
                            prefill_chunk_size,
                            exact_full_sample=exact_full_sample,
                        ),
                        ltm_state=ltm_state,
                        global_pos_offset=start,
                        return_last_logit_only=True,
                    )

                if outputs is None:
                    raise RuntimeError("Probe prefill produced no model output.")
                logits = outputs["logits"][:, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                top_vals, top_idx = torch.topk(
                    probs,
                    min(5, int(probs.shape[-1])),
                )
                top = [
                    (tokenizer.decode([int(index)]), round(float(value), 4))
                    for value, index in zip(top_vals[0], top_idx[0])
                ]

                current = top_idx[:, :1]
                generated: list[int] = []
                total_tokens_seen = int(ids.shape[1])
                for _ in range(int(max_tokens)):
                    token_id = int(current.item())
                    (
                        h_state,
                        l_state,
                        prev_context,
                        target_context,
                        drift_state,
                        ltm_state,
                    ) = state
                    outputs, state = advance_chat_model_state(
                        model,
                        current,
                        device=device,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        drift_seed=boundary_drift_seed(
                            drift_state,
                            total_tokens_seen,
                            prefill_chunk_size,
                            exact_full_sample=exact_full_sample,
                        ),
                        ltm_state=ltm_state,
                        global_pos_offset=total_tokens_seen,
                        return_last_logit_only=True,
                    )
                    total_tokens_seen += 1
                    # Terminal tokens are consumed exactly once before stopping,
                    # matching interactive chat's recurrent/LTM state semantics.
                    if token_id == tokenizer.eos_token_id:
                        break
                    generated.append(token_id)
                    current = outputs["logits"][:, -1, :].argmax(
                        dim=-1,
                        keepdim=True,
                    )

                print(f"\nPROMPT: {prompt_text}")
                print(f"TOP5: {top}")
                print(f"GREEDY: {tokenizer.decode(generated).strip()!r}")
    finally:
        clear_ltm_working_memory(model)
        model.suppress_hebbian = previous_suppress_hebbian
        model.train(was_training)


def _identity_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def build_salvage_run_identity(model, tokenizer, rescue_metadata: dict) -> dict:
    contract = architecture_contract(model.config)
    source_metadata = getattr(model, "_hierarchos_checkpoint_metadata", {})
    source_identity = (
        source_metadata.get("run_identity")
        if isinstance(source_metadata, dict)
        else None
    )
    identity = {
        "version": 1,
        "kind": "response-salvage-finetune",
        "tokenizer": tokenizer_identity(tokenizer),
        "architecture_contract": contract,
        "architecture_contract_sha256": architecture_contract_hash(model.config),
        "objective": copy.deepcopy(rescue_metadata),
    }
    if isinstance(source_identity, dict):
        source_digest = source_identity.get("sha256")
        identity["parent_run_sha256"] = (
            str(source_digest)
            if isinstance(source_digest, str)
            else _identity_digest(
                {key: value for key, value in source_identity.items() if key != "sha256"}
            )
        )
    identity["sha256"] = _identity_digest(identity)
    return identity


def save_inference_dir(
    model,
    config,
    tokenizer,
    out_dir: Path,
    *,
    rescue_metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    saved_config["architecture_contract_sha256"] = contract_hash
    saved_config["salvage_response_finetune"] = copy.deepcopy(rescue_metadata)
    source_metadata = getattr(model, "_hierarchos_checkpoint_metadata", {})
    if not isinstance(source_metadata, dict):
        source_metadata = {}
    completed_epoch = int(
        source_metadata.get(
            "completed_epoch",
            _config_value(config, "completed_epoch", 0),
        )
        or 0
    )
    checkpoint = {
        "checkpoint_version": max(
            4,
            int(source_metadata.get("checkpoint_version", 0) or 0),
        ),
        "checkpoint_kind": "inference-response-salvaged",
        "derived_from_checkpoint_kind": str(
            source_metadata.get("checkpoint_kind") or "unknown"
        ),
        "model_state_dict": sanitize_model_state_dict(model),
        "config": saved_config,
        "architecture_contract": contract,
        "architecture_contract_sha256": contract_hash,
        "completed_epoch": completed_epoch,
        "run_identity": build_salvage_run_identity(
            model,
            tokenizer,
            rescue_metadata,
        ),
        "optimizer_grouping_version": 2,
        "training_complete": True,
    }
    save_training_checkpoint_if_finite(
        checkpoint,
        str(out_dir / "hierarchos.pt"),
        model,
        optimizer=None,
    )
    tokenizer.save_pretrained(str(out_dir))
    with (out_dir / "hierarchos_config.json").open("w", encoding="utf-8") as handle:
        json.dump(saved_config, handle, indent=2, default=str)


def validate_output_destination(model_path: str, out_dir: Path) -> None:
    source = Path(model_path).expanduser().resolve()
    source_dir = source if source.is_dir() else source.parent
    destination = out_dir.expanduser().resolve()
    if destination == source_dir:
        raise ValueError(
            "--out-dir must differ from the source model directory. Rescue "
            "exports are derived artifacts and cannot overwrite their only base."
        )


def run_rescue_training(
    model,
    config,
    tokenizer,
    device,
    encoded_rows,
    cli_args,
) -> dict:
    train_args = build_rescue_train_args(
        config,
        cli_args,
        tokenizer,
        device,
    )
    optimizer = build_hierarchos_optimizer(model, train_args, device)
    scaler = None
    if train_args.amp and train_args.amp_dtype == "float16":
        scaler = GradScaler()

    batches_per_epoch = math.ceil(len(encoded_rows) / int(cli_args.batch_size))
    if batches_per_epoch <= 0:
        raise ValueError("Rescue training has no batches.")

    saved_training_step = get_model_training_step(model)
    training_step_offset = resolve_training_step_offset(model, 0)
    if saved_training_step is None:
        print(
            "WARNING: Base checkpoint has no persisted memory-gate curriculum "
            "step; rescue starts that curriculum at local step zero."
        )
    else:
        print(
            "INFO: Continuing memory-gate curriculum without re-warmup: "
            f"saved_step={saved_training_step}, "
            f"next_step={training_step_offset}."
        )

    rng = random.Random(int(cli_args.seed))
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    microbatches_seen = 0
    logged_losses: list[float] = []
    stop = False

    for epoch in range(int(cli_args.epochs)):
        ensure_finetune_training_mode(model)
        model.suppress_hebbian = True
        configure_checkpoint_rng_policy(model)
        rng.shuffle(encoded_rows)

        for batch_index, (
            input_ids,
            labels,
            attention_mask,
            loss_weights,
        ) in enumerate(
            iter_batches(
                encoded_rows,
                int(cli_args.batch_size),
                int(tokenizer.pad_token_id),
            )
        ):
            batch = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_weights": loss_weights,
            }
            # Rescue JSONL rows are independent samples. Recurrent, ROSA, and
            # transient LTM state may cross temporal chunks inside this call,
            # but never crosses a minibatch boundary.
            running_states = (None, None, None, None, None, None)
            train_args._current_global_step = (
                training_step_offset + microbatches_seen
            )
            force_optimizer_step = should_step_accumulation(
                batch_index,
                batches_per_epoch,
                train_args.accumulation_steps,
            )
            divisor = accumulation_divisor_for_step(
                batch_index,
                batches_per_epoch,
                train_args.accumulation_steps,
            )
            collect_metrics = (
                microbatches_seen == 0
                or (microbatches_seen + 1) % 10 == 0
                or force_optimizer_step
            )
            outputs, _ = train_step(
                model,
                batch,
                optimizer,
                scaler,
                train_args.accumulation_steps,
                batch_index,
                train_args,
                running_states,
                collect_metrics=collect_metrics,
                force_optimizer_step=force_optimizer_step,
                accumulation_divisor=divisor,
            )
            account_skipped_training_batch(
                train_args,
                reason=train_step_skip_reason(train_args),
                epoch=epoch + 1,
                step=batch_index + 1,
                scope="Response rescue",
            )
            microbatches_seen += 1

            if outputs is not None:
                loss_value = float(outputs["loss"].detach().float().cpu())
                logged_losses.append(loss_value)
                if (
                    microbatches_seen == 1
                    or microbatches_seen % 10 == 0
                    or force_optimizer_step
                ):
                    window = logged_losses[-10:]
                    fields = [
                        f"epoch={epoch + 1}",
                        f"batch={batch_index + 1}/{batches_per_epoch}",
                        f"loss={loss_value:.4f}",
                        f"avg10={sum(window) / len(window):.4f}",
                    ]
                    if outputs.get("ponder_cost") is not None:
                        fields.append(
                            f"ponder={float(outputs['ponder_cost']):.2f}"
                        )
                    if outputs.get("commitment_cost") is not None:
                        fields.append(
                            f"commit={float(outputs['commitment_cost']):.2e}"
                        )
                    print(" ".join(fields))

            if bool(getattr(train_args, "_optimizer_step_was_taken", False)):
                optimizer_steps += 1
                if (
                    int(cli_args.max_steps) > 0
                    and optimizer_steps >= int(cli_args.max_steps)
                ):
                    stop = True
                    break

        if stop:
            break

    if optimizer_steps <= 0:
        raise RuntimeError(
            "Rescue completed without a verified optimizer update; refusing to "
            "export an unchanged or failed model."
        )
    if any(
        parameter.requires_grad and parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise RuntimeError(
            "Rescue ended with an incomplete accumulation window; refusing to "
            "export before its gradients are applied."
        )

    return {
        "version": 1,
        "preset": str(cli_args.preset),
        "include": [str(value) for value in cli_args.include],
        "epochs_requested": int(cli_args.epochs),
        "optimizer_steps": int(optimizer_steps),
        "microbatches_seen": int(microbatches_seen),
        "batch_size": int(cli_args.batch_size),
        "accumulation_steps": int(cli_args.accumulation_steps),
        "max_length": int(cli_args.max_length),
        "starting_lr": float(cli_args.lr),
        "weight_decay": float(cli_args.weight_decay),
        "grad_clip": float(cli_args.grad_clip),
        "train_prompt_tokens": bool(cli_args.train_prompt_tokens),
        "prompt_loss_weight": float(cli_args.prompt_loss_weight),
        "response_loss_weight": float(cli_args.response_loss_weight),
        "response_boundary_loss_weight": float(
            cli_args.response_boundary_loss_weight
        ),
        "response_boundary_tokens": int(cli_args.response_boundary_tokens),
        "training_chunk_size": int(train_args.training_chunk_size),
        "full_sample_bptt": bool(train_args.full_sample_bptt),
        "full_sample_activation_checkpointing": bool(
            train_args.full_sample_activation_checkpointing
        ),
        "ltm_training_mode": "read-only",
        "adaptive_ponder": bool(train_args.adaptive_ponder),
        "ponder_loss_weight": float(train_args.ponder_loss_weight),
        "commitment_loss_weight": float(train_args.commitment_loss_weight),
        "amp": bool(train_args.amp),
        "amp_dtype": str(train_args.amp_dtype),
        "seed": int(cli_args.seed),
        "skipped_train_batches": int(train_args._skipped_train_batches),
        "memory_gate_start_step": (
            None if saved_training_step is None else int(saved_training_step)
        ),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Run a cheap response-only rescue fine-tune.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--train",
        action="append",
        default=None,
        help="JSONL rescue file. Can be passed multiple times. Default: tools/rescue_alpaca_seed.jsonl",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "salvaged_kortexHOS"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=sorted(TRAINABLE_PRESETS), default="head")
    parser.add_argument("--include", action="append", default=[], help="Additional parameter-name prefix to train.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Accumulate by supervised token mass across this many minibatches.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum verified optimizer updates; 0 means no explicit limit.",
    )
    parser.add_argument(
        "--max-skipped-train-batches",
        type=int,
        default=0,
        help=(
            "Explicit loss/gradient/OOM skip budget. The fail-closed default is "
            "zero."
        ),
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CUDA AMP. Defaults on for CUDA and off for CPU.",
    )
    parser.add_argument("--train-prompt-tokens", action="store_true", help="Also train prompt tokens during rescue instead of response-only masking.")
    parser.add_argument("--prompt-loss-weight", type=float, default=0.15)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-boundary-loss-weight", type=float, default=5.0)
    parser.add_argument("--response-boundary-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Explicitly allow the tokenizer repository to execute custom Python "
            "code. Disabled by default; use only for a repository you trust."
        ),
    )
    args = parser.parse_args()

    validate_cli_args(args)
    if args.probe_only and args.no_probe:
        raise ValueError("--probe-only and --no-probe cannot be used together.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if not args.probe_only:
        validate_output_destination(args.model_path, Path(args.out_dir))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token; coherent "
                "response rescue cannot infer either one."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model, config = load_full_model_with_config(args.model_path, device)
    validate_training_geometry(config)
    tokenizer_verified = validate_loaded_tokenizer(model, config, tokenizer)
    if tokenizer_verified:
        print("INFO: Training tokenizer content fingerprint verified.")
    else:
        print(
            "WARNING: Legacy checkpoint has no tokenizer fingerprint; only "
            "vocabulary-size compatibility could be verified."
        )
    clear_ltm_working_memory(model)
    model.suppress_hebbian = True

    if not args.no_probe:
        print("\n=== Before Rescue Probe ===")
        probe_model(model, tokenizer, device, DEFAULT_PROBES)

    if args.probe_only:
        return 0

    train_paths = [Path(p) for p in (args.train or [str(ROOT / "tools" / "rescue_alpaca_seed.jsonl")])]
    rows = []
    for train_path in train_paths:
        rows.extend(load_jsonl(train_path))
    encoded = [
        encode_row(
            tokenizer,
            row,
            args.max_length,
            train_prompt_tokens=args.train_prompt_tokens,
            prompt_loss_weight=args.prompt_loss_weight,
            response_loss_weight=args.response_loss_weight,
            response_boundary_loss_weight=args.response_boundary_loss_weight,
            response_boundary_tokens=args.response_boundary_tokens,
        )
        for row in rows
    ]
    total, trainable = select_trainable(model, args.preset, args.include)
    print(
        f"\nTrainable preset: {args.preset} | "
        f"{trainable:,}/{total:,} parameters ({100.0 * trainable / max(1, total):.2f}%)"
    )
    print(f"Rows: {len(encoded)} | epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr:g}")
    print(
        "Loss weights: "
        f"prompt={args.prompt_loss_weight:g} ({'trained' if args.train_prompt_tokens else 'masked'}), "
        f"response={args.response_loss_weight:g}, "
        f"boundary={args.response_boundary_loss_weight:g}x first {args.response_boundary_tokens} token(s)"
    )
    rescue_metadata = run_rescue_training(
        model,
        config,
        tokenizer,
        device,
        encoded,
        args,
    )
    rescue_metadata["trainable_parameters"] = int(trainable)
    rescue_metadata["total_parameters"] = int(total)

    clear_ltm_working_memory(model)
    save_inference_dir(
        model,
        config,
        tokenizer,
        Path(args.out_dir),
        rescue_metadata=rescue_metadata,
    )
    print(f"\nSaved salvaged inference model to: {Path(args.out_dir).resolve()}")

    if not args.no_probe:
        clear_ltm_working_memory(model)
        print("\n=== After Rescue Probe ===")
        probe_model(model, tokenizer, device, DEFAULT_PROBES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
