"""Named architecture revisions and their serialized behavioral contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


LEGACY_REVISION = "legacy-v8"
COHERENT_REVISION = "coherent-v9"
ARCHITECTURE_CONTRACT_SCHEMA_VERSION = 2


# These fields alter tensor geometry, the learned function, recurrent/memory
# state semantics, or the objective that produced the learned weights. They
# belong in checkpoint/run identity, not in a loose collection of runtime flags.
# Pure execution choices (compile backend, worker count, progress logging, loss
# chunk row count) are deliberately excluded.
ARCHITECTURE_CONTRACT_FIELDS = (
    "architecture_revision",
    # Tensor geometry.
    "vocab_size",
    "context_dim",
    "persistent_dim",
    "ltm_slots",
    "ltm_key_dim",
    "ltm_val_dim",
    "ltm_topk",
    "h_hidden",
    "l_hidden",
    "rwkv_head_size",
    "h_rwkv_head_size",
    "l_rwkv_head_size",
    "rwkv_n_layer_hint",
    "h_stride",
    "max_h_steps",
    "max_l_steps",
    # Recurrent and adaptive-compute semantics.
    "core_recurrence_version",
    "drift_recurrence_mode",
    "rwkv_state_readout_mode",
    "manager_state_commit_mode",
    "manager_compute_mode",
    "min_h_steps",
    "h_halt_thresh",
    "act_depth_temperature",
    "l_conv_atol",
    "commitment_cost_mode",
    "commitment_threshold",
    "drift_delta_scale",
    "detach_every_n_steps",
    "full_sample_bptt",
    "inference_logit_parity",
    "inference_recurrence_mode",
    # Forward numerical policy. These clamps are part of the function.
    "halt_logit_clamp",
    "recurrent_state_clamp",
    "context_state_clamp",
    "drift_state_clamp",
    "drift_norm_clamp",
    "activation_clamp",
    "rwkv_channel_mix_key_clamp",
    "rwkv_channel_mix_deepembed_clamp",
    "inference_logit_clamp",
    # Token/memory representation and state policy.
    "use_deepembed",
    "deepembed_mode",
    "use_rosa",
    "rosa_embedding_mode",
    "token_adapter_rank",
    "memory_token_routers",
    "rosa_max_context",
    "enforce_rosa_max_context",
    "rosa_zero_no_prediction",
    "isolate_batch_ltm",
    "ltm_training_mode",
    "ltm_lr",
    "ltm_momentum",
    "ltm_weight_decay",
    "ltm_forget_rate",
    "ltm_score_grad_scale",
    "reference_chunk_len",
    "training_chunk_size",
    "allow_untrained_hebbian_writer",
    "memory_gate_warmup_steps",
    "memory_gate_warmup_floor",
    # Auxiliary objectives that shape the learned function.
    "adaptive_ponder",
    "ponder_objective",
    "ponder_target_scale",
    "ponder_huber_beta",
    "ponder_loss_weight",
    "encourage_thinking",
    "commitment_loss_weight",
    "max_commitment_cost_for_backward",
    "max_ponder_cost_for_backward",
    "z_loss_weight",
)


def _get(config, name: str, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _set(config, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _contains(config, name: str) -> bool:
    if isinstance(config, Mapping):
        return name in config
    return hasattr(config, name)


def _setdefault(config, name: str, value) -> None:
    current = _get(config, name, None)
    if current in (None, "", "auto"):
        _set(config, name, value)


def normalize_architecture_revision(value) -> str:
    normalized = str(value or LEGACY_REVISION).strip().lower().replace("_", "-")
    aliases = {
        "legacy": LEGACY_REVISION,
        "v8": LEGACY_REVISION,
        "legacy-v1": LEGACY_REVISION,
        "coherent": COHERENT_REVISION,
        "v9": COHERENT_REVISION,
        "v9-coherent": COHERENT_REVISION,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {LEGACY_REVISION, COHERENT_REVISION}:
        raise ValueError(
            "architecture_revision must be 'legacy-v8' or 'coherent-v9', "
            f"got {value!r}"
        )
    return normalized


def apply_architecture_revision_defaults(config) -> str:
    """Fill, then serialize, every revision-controlled default.

    Explicit per-feature settings remain available for ablations.  The named
    revision supplies defaults only; checkpoint identity records the resolved
    values so an ablation cannot masquerade as a stock revision.
    """

    revision = normalize_architecture_revision(
        _get(config, "architecture_revision", None)
    )
    _set(config, "architecture_revision", revision)
    # The public CLI uses 0/negative values to mean "never detach", while the
    # executable recurrent contract uses None. Canonicalize before hashing so
    # training checkpoints, inference exports, and JSON configs cannot serialize
    # two different representations of the same learned function.
    if _contains(config, "detach_every_n_steps"):
        raw_detach = _get(config, "detach_every_n_steps", None)
        if raw_detach is not None:
            try:
                normalized_detach = int(raw_detach)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "detach_every_n_steps must be an integer or None, "
                    f"got {raw_detach!r}"
                ) from exc
            _set(
                config,
                "detach_every_n_steps",
                normalized_detach if normalized_detach > 0 else None,
            )
    if revision == COHERENT_REVISION:
        defaults = {
            "core_recurrence_version": 2,
            "manager_compute_mode": "hard-masked",
            "manager_state_commit_mode": "hard-selected",
            "commitment_cost_mode": "mean-square",
            "deepembed_mode": "shared-factorized",
            "rosa_embedding_mode": "shared-factorized",
            "enforce_rosa_max_context": True,
            "rosa_zero_no_prediction": True,
            "adaptive_ponder": True,
            "ponder_objective": "symmetric-huber",
            "inference_logit_clamp": 0.0,
            "inference_logit_parity": True,
            "l_conv_atol": 1e-4,
            "commitment_threshold": 0.05,
        }
    else:
        defaults = {
            "core_recurrence_version": 1,
            "manager_compute_mode": "soft-act",
            "manager_state_commit_mode": "legacy-real-step",
            "commitment_cost_mode": "sum-square",
            "deepembed_mode": "legacy-table",
            "rosa_embedding_mode": "legacy-table",
            "enforce_rosa_max_context": False,
            "rosa_zero_no_prediction": False,
            "adaptive_ponder": False,
            "ponder_objective": "auto",
            "inference_logit_clamp": 30.0,
            "inference_logit_parity": False,
            "l_conv_atol": 0.01,
            "commitment_threshold": 0.1,
        }
    for name, value in defaults.items():
        _setdefault(config, name, value)

    # Canonicalize defaults that were historically scattered across the core,
    # WorkerLoop, LTM, trainer, and CLI. A serialized contract must contain the
    # concrete function, not ``None`` values whose meaning changes when a code
    # default changes.
    common_defaults = {
        "persistent_dim": 128,
        "ltm_slots": 1024,
        "ltm_key_dim": 128,
        "ltm_val_dim": 128,
        "ltm_topk": 4,
        "rwkv_n_layer_hint": 2,
        "h_stride": 4,
        "max_h_steps": 5,
        "max_l_steps": 5,
        "min_h_steps": 1,
        "h_halt_thresh": 0.9,
        "act_depth_temperature": 0.05,
        "drift_delta_scale": 1.0,
        "detach_every_n_steps": 32,
        "full_sample_bptt": False,
        "inference_logit_parity": False,
        "halt_logit_clamp": 30.0,
        "recurrent_state_clamp": 50.0,
        "context_state_clamp": 50.0,
        "drift_state_clamp": 5.0,
        "drift_norm_clamp": 0.0,
        "activation_clamp": 100.0,
        "rwkv_channel_mix_key_clamp": 12.0,
        "rwkv_channel_mix_deepembed_clamp": 4.0,
        "use_deepembed": True,
        "use_rosa": True,
        "memory_token_routers": True,
        "rosa_max_context": 512,
        "isolate_batch_ltm": True,
        "ltm_training_mode": "inner-update",
        "ltm_lr": 1e-3,
        "ltm_momentum": 0.9,
        "ltm_weight_decay": 1e-4,
        "ltm_forget_rate": 0.01,
        "ltm_score_grad_scale": 1.0,
        "training_chunk_size": 128,
        "allow_untrained_hebbian_writer": False,
        "memory_gate_warmup_steps": 0,
        "memory_gate_warmup_floor": 0.0,
        "ponder_target_scale": 0.5,
        "ponder_huber_beta": 0.5,
        "ponder_loss_weight": 0.01,
        "encourage_thinking": False,
        "commitment_loss_weight": 0.5,
        "max_commitment_cost_for_backward": 2.0,
        "max_ponder_cost_for_backward": 0.0,
        "z_loss_weight": 1e-4,
    }
    for name, value in common_defaults.items():
        # ``None`` is the normalized, executable representation of the public
        # detach-every-n-steps=0 sentinel. Do not turn it back into 32.
        if name == "detach_every_n_steps" and _contains(config, name):
            continue
        _setdefault(config, name, value)

    context_dim = _get(config, "context_dim", None)
    if context_dim is not None:
        _setdefault(config, "h_hidden", context_dim)
        _setdefault(config, "l_hidden", context_dim)
        _setdefault(config, "token_adapter_rank", min(64, int(context_dim)))
    _setdefault(
        config,
        "reference_chunk_len",
        _get(config, "training_chunk_size", 128),
    )
    _setdefault(
        config,
        "inference_recurrence_mode",
        (
            "full-sample"
            if bool(_get(config, "full_sample_bptt", False))
            else "tbptt"
        ),
    )
    return revision


def architecture_contract(config) -> dict[str, Any]:
    """Return a stable JSON-compatible snapshot of learned-function settings."""

    apply_architecture_revision_defaults(config)
    contract = {
        "architecture_contract_schema_version": ARCHITECTURE_CONTRACT_SCHEMA_VERSION,
    }
    for field in ARCHITECTURE_CONTRACT_FIELDS:
        value = _get(config, field, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            contract[field] = value
        else:
            contract[field] = str(value)
    return contract


def architecture_contract_hash(config) -> str:
    encoded = json.dumps(
        architecture_contract(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_architecture_contract(
    config,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    expected_hash: str | None = None,
    source: str = "checkpoint",
) -> tuple[dict[str, Any], str]:
    """Fail closed when serialized learned-function metadata was altered.

    Legacy checkpoints may call this with no expected metadata; in that case it
    simply returns the resolved contract. New checkpoints persist both the
    contract and its digest, so changes to a recurrent, memory, geometry, clamp,
    or objective setting are detected before model construction.
    """

    resolved = architecture_contract(config)
    resolved_hash = hashlib.sha256(json.dumps(
        resolved,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()

    if expected_contract is not None:
        if not isinstance(expected_contract, Mapping):
            raise ValueError(
                f"Invalid architecture contract metadata in {source}: expected a mapping."
            )
        serialized = dict(expected_contract)
        if serialized != resolved:
            mismatches = []
            for key in sorted(set(serialized) | set(resolved)):
                old = serialized.get(key)
                new = resolved.get(key)
                if old != new:
                    mismatches.append(f"{key}: saved={old!r}, resolved={new!r}")
            details = "; ".join(mismatches[:12])
            if len(mismatches) > 12:
                details += f"; ... {len(mismatches) - 12} more"
            raise ValueError(
                f"Architecture contract mismatch in {source}: {details}. "
                "Refusing to construct a model with altered learned-function settings."
            )

    if expected_hash is not None:
        normalized_hash = str(expected_hash).strip().lower()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in normalized_hash
        ):
            raise ValueError(
                f"Invalid architecture contract SHA-256 in {source}: "
                f"{expected_hash!r}"
            )
        if normalized_hash != resolved_hash:
            raise ValueError(
                f"Architecture contract SHA-256 mismatch in {source}: "
                f"saved={normalized_hash}, resolved={resolved_hash}. "
                "Refusing to construct a model with altered learned-function settings."
            )

    return resolved, resolved_hash
