"""Named architecture revisions and their serialized behavioral contracts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


LEGACY_REVISION = "legacy-v8"
COHERENT_REVISION = "coherent-v9"
ARCHITECTURE_CONTRACT_SCHEMA_VERSION = 3
LEGACY_TRAINING_CHUNK_SIZE = 128
COHERENT_TRAINING_CHUNK_SIZE = 256
LEGACY_COMMITMENT_SUM_SQUARE_BUDGET = 0.1
COHERENT_COMMITMENT_REFERENCE_WIDTH = 448


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
    "ltm_time_feature_mode",
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
    "ltm_value_alignment_weight",
    "ltm_value_alignment_stride",
    "ltm_value_alignment_min_updates",
    "ltm_value_alignment_ready_threshold",
    "ltm_value_alignment_ema_decay",
    "ltm_value_writer_max_norm",
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


def _canonical_contract_int(
    config,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    """Validate and persist one integer-valued architecture setting."""

    if not _contains(config, name):
        return None
    raw_value = _get(config, name, None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be an integer, got {raw_value!r}")
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be a finite integer, got {raw_value!r}")
    value = int(numeric_value)
    if minimum is not None and value < minimum:
        if minimum == 1:
            raise ValueError(
                f"{name} must be a positive integer, got {raw_value!r}"
            )
        raise ValueError(f"{name} must be >= {minimum}, got {raw_value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {raw_value!r}")
    _set(config, name, value)
    return value


def _canonical_contract_float(
    config,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float | None:
    """Validate and persist one finite floating-point architecture setting."""

    if not _contains(config, name):
        return None
    raw_value = _get(config, name, None)
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be a finite number, got {raw_value!r}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number, got {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw_value!r}")
    if minimum is not None:
        below_minimum = value < minimum if minimum_inclusive else value <= minimum
        if below_minimum:
            comparator = ">=" if minimum_inclusive else ">"
            raise ValueError(
                f"{name} must be {comparator} {minimum}, got {raw_value!r}"
            )
    if maximum is not None:
        above_maximum = value > maximum if maximum_inclusive else value >= maximum
        if above_maximum:
            comparator = "<=" if maximum_inclusive else "<"
            raise ValueError(
                f"{name} must be {comparator} {maximum}, got {raw_value!r}"
            )
    _set(config, name, value)
    return value


def validate_architecture_numeric_contract(config) -> None:
    """Canonicalize numeric learned-function settings or fail closed.

    Architecture hashes must describe the function that the forward actually
    executes.  Runtime helpers historically replaced invalid/non-positive raw
    values with private defaults, which allowed (for example) a contract that
    hashed ``recurrent_state_clamp=0`` while the cell executed a clamp of 50.
    Validate at the shared revision boundary so checkpoint hashing, model
    construction, expansion, training, and inference all see one concrete
    numeric contract.

    Missing geometry is allowed because callers may resolve a partial contract
    before tokenizer-dependent dimensions are known.  Every present value is
    canonicalized, and dependent constraints are checked when both operands are
    available.
    """

    positive_int_fields = (
        "vocab_size",
        "context_dim",
        "ltm_slots",
        "ltm_key_dim",
        "ltm_val_dim",
        "ltm_topk",
        "h_hidden",
        "l_hidden",
        "h_stride",
        "max_h_steps",
        "max_l_steps",
        "min_h_steps",
        "training_chunk_size",
        "reference_chunk_len",
        "ltm_value_alignment_stride",
        "ltm_value_alignment_min_updates",
    )
    for name in positive_int_fields:
        _canonical_contract_int(config, name, minimum=1)

    _canonical_contract_int(
        config,
        "core_recurrence_version",
        minimum=1,
        maximum=2,
    )
    _canonical_contract_int(config, "persistent_dim", minimum=0)
    _canonical_contract_int(config, "rwkv_n_layer_hint", minimum=2)
    _canonical_contract_int(config, "memory_gate_warmup_steps", minimum=0)

    # Zero/None/"auto" remain supported head-size sentinels until the concrete
    # cell widths resolve them during model construction.
    for name in (
        "rwkv_head_size",
        "h_rwkv_head_size",
        "l_rwkv_head_size",
    ):
        raw_value = _get(config, name, None)
        if raw_value in (None, "", "auto", 0):
            continue
        _canonical_contract_int(config, name, minimum=1)

    raw_adapter_rank = _get(config, "token_adapter_rank", None)
    if raw_adapter_rank not in (None, "", "auto", 0):
        _canonical_contract_int(config, "token_adapter_rank", minimum=1)

    rosa_max_context = _canonical_contract_int(
        config,
        "rosa_max_context",
        minimum=0,
    )
    if (
        bool(_get(config, "use_rosa", True))
        and bool(_get(config, "enforce_rosa_max_context", False))
        and (rosa_max_context is None or rosa_max_context <= 0)
    ):
        raise ValueError(
            "rosa_max_context must be positive when bounded ROSA is enabled"
        )

    strictly_positive_float_fields = (
        "halt_logit_clamp",
        "recurrent_state_clamp",
        "context_state_clamp",
        "drift_state_clamp",
        "activation_clamp",
        "l_conv_atol",
        "act_depth_temperature",
        "ponder_huber_beta",
    )
    for name in strictly_positive_float_fields:
        _canonical_contract_float(
            config,
            name,
            minimum=0.0,
            minimum_inclusive=False,
        )

    nonnegative_float_fields = (
        # Explicit zero disables these clamps/scales/objectives.
        "drift_norm_clamp",
        "drift_delta_scale",
        "rwkv_channel_mix_key_clamp",
        "rwkv_channel_mix_deepembed_clamp",
        "inference_logit_clamp",
        "commitment_threshold",
        "ltm_lr",
        "ltm_weight_decay",
        "ltm_score_grad_scale",
        "ponder_target_scale",
        "ponder_loss_weight",
        "commitment_loss_weight",
        "max_commitment_cost_for_backward",
        "max_ponder_cost_for_backward",
        "z_loss_weight",
        "ltm_value_alignment_weight",
        "ltm_value_writer_max_norm",
    )
    for name in nonnegative_float_fields:
        _canonical_contract_float(config, name, minimum=0.0)

    _canonical_contract_float(
        config,
        "h_halt_thresh",
        minimum=0.0,
        maximum=1.0,
    )
    _canonical_contract_float(
        config,
        "ltm_momentum",
        minimum=0.0,
        maximum=1.0,
    )
    _canonical_contract_float(
        config,
        "ltm_forget_rate",
        minimum=0.0,
        maximum=1.0,
    )
    _canonical_contract_float(
        config,
        "memory_gate_warmup_floor",
        minimum=0.0,
        maximum=0.95,
    )
    _canonical_contract_float(
        config,
        "ltm_value_alignment_ready_threshold",
        minimum=0.0,
    )
    _canonical_contract_float(
        config,
        "ltm_value_alignment_ema_decay",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )

    raw_time_feature_mode = str(
        _get(config, "ltm_time_feature_mode", "absolute-sinusoidal")
        or "absolute-sinusoidal"
    ).strip().lower().replace("_", "-")
    time_feature_aliases = {
        "sinusoidal": "absolute-sinusoidal",
        "absolute": "absolute-sinusoidal",
        "none": "metadata-only",
        "off": "metadata-only",
        "metadata": "metadata-only",
    }
    time_feature_mode = time_feature_aliases.get(
        raw_time_feature_mode,
        raw_time_feature_mode,
    )
    if time_feature_mode not in {"absolute-sinusoidal", "metadata-only"}:
        raise ValueError(
            "ltm_time_feature_mode must be 'absolute-sinusoidal' or "
            f"'metadata-only', got {raw_time_feature_mode!r}"
        )
    _set(config, "ltm_time_feature_mode", time_feature_mode)

    if _contains(config, "ltm_training_mode"):
        _set(
            config,
            "ltm_training_mode",
            normalize_ltm_training_mode(_get(config, "ltm_training_mode")),
        )

    ltm_slots = _get(config, "ltm_slots", None)
    ltm_topk = _get(config, "ltm_topk", None)
    if ltm_slots is not None and ltm_topk is not None and ltm_topk > ltm_slots:
        raise ValueError(
            "ltm_topk cannot exceed ltm_slots; larger values allocate dead "
            f"readout blocks (got ltm_topk={ltm_topk}, ltm_slots={ltm_slots})"
        )

    min_h_steps = _get(config, "min_h_steps", None)
    max_h_steps = _get(config, "max_h_steps", None)
    if (
        min_h_steps is not None
        and max_h_steps is not None
        and min_h_steps > max_h_steps
    ):
        raise ValueError(
            "min_h_steps cannot exceed max_h_steps; "
            f"got min_h_steps={min_h_steps}, max_h_steps={max_h_steps}"
        )

    shared_head_size = _get(config, "rwkv_head_size", None)
    for width_name, head_name in (
        ("h_hidden", "h_rwkv_head_size"),
        ("l_hidden", "l_rwkv_head_size"),
    ):
        width = _get(config, width_name, None)
        # An explicitly supplied per-cell sentinel disables the shared request,
        # matching the exact fallback semantics in HierarchosCore.
        head_size = (
            _get(config, head_name, None)
            if _contains(config, head_name)
            else shared_head_size
        )
        if width is not None and head_size not in (None, "", "auto", 0):
            if width % head_size != 0:
                raise ValueError(
                    f"{head_name}={head_size} does not divide "
                    f"{width_name}={width}"
                )


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


def normalize_ltm_training_mode(value) -> str:
    """Return the canonical LTM training mode and reject unknown behavior.

    Historically the trainer silently mapped typos to the expensive,
    label-gradient ``inner-update`` path while the core treated the same typo
    as read-only.  A learned-function contract must fail closed instead.
    """

    normalized = str(value or "inner-update").strip().lower().replace("_", "-")
    aliases = {
        "inner": "inner-update",
        "inner-updates": "inner-update",
        "gradient": "inner-update",
        "grad": "inner-update",
        "supervised": "inner-update",
        "supervised-inner-update": "inner-update",
        "titans": "inner-update",
        "titans-inner-update": "inner-update",
        "readonly": "read-only",
        "inference": "read-only",
        "inference-like": "read-only",
        "inference-like-ltm": "read-only",
        "off": "read-only",
        "none": "read-only",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"inner-update", "read-only"}:
        raise ValueError(
            "ltm_training_mode must be 'inner-update' or 'read-only', "
            f"got {value!r}"
        )
    return normalized


def architecture_default_commitment_threshold(config_or_revision=None) -> float:
    """Return the revision-correct commitment hinge threshold.

    Legacy-v8 measures total drift energy with ``sum(drift**2)`` and used a
    free-energy budget of 0.1. Coherent-v9 measures width-invariant mean-square
    drift, so retaining the same total L2 budget requires dividing by the drift
    width. Bare revision-only configs use the reference training width so their
    serialized contract remains deterministic until geometry is supplied.
    """

    if isinstance(config_or_revision, str) or config_or_revision is None:
        revision_value = config_or_revision
        context_dim = None
    else:
        revision_value = _get(
            config_or_revision,
            "architecture_revision",
            None,
        )
        context_dim = _get(config_or_revision, "context_dim", None)

    revision = normalize_architecture_revision(revision_value)
    if revision != COHERENT_REVISION:
        return LEGACY_COMMITMENT_SUM_SQUARE_BUDGET

    if context_dim in (None, "", "auto"):
        width = COHERENT_COMMITMENT_REFERENCE_WIDTH
    else:
        if isinstance(context_dim, bool):
            raise ValueError(
                f"context_dim must be a positive integer, got {context_dim!r}"
            )
        try:
            numeric_width = float(context_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"context_dim must be a positive integer, got {context_dim!r}"
            ) from exc
        if not math.isfinite(numeric_width) or not numeric_width.is_integer():
            raise ValueError(
                f"context_dim must be a positive integer, got {context_dim!r}"
            )
        width = int(numeric_width)
        if width <= 0:
            raise ValueError(
                f"context_dim must be a positive integer, got {context_dim!r}"
            )

    return LEGACY_COMMITMENT_SUM_SQUARE_BUDGET / float(width)


def architecture_default_training_chunk_size(config_or_revision=None) -> int:
    """Return the compatibility-safe TBPTT/LTM geometry for a revision.

    Historical configs used 128 when the field was absent.  The supported v9
    CLI and GUI both expose 256, so a bare coherent-v9 config must resolve to
    the same learned-function contract instead of quietly hashing a different
    LTM reference geometry.
    """

    if isinstance(config_or_revision, str) or config_or_revision is None:
        revision_value = config_or_revision
    else:
        revision_value = _get(
            config_or_revision,
            "architecture_revision",
            None,
        )
    revision = normalize_architecture_revision(revision_value)
    if revision == COHERENT_REVISION:
        return COHERENT_TRAINING_CHUNK_SIZE
    return LEGACY_TRAINING_CHUNK_SIZE


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
            normalized_detach = _canonical_contract_int(
                config,
                "detach_every_n_steps",
            )
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
            # Gradient-written fast memory is unavailable during ordinary
            # autoregressive generation (there is no target-label loss then).
            # Keep new coherent runs train/chat aligned unless the experiment
            # explicitly opts into the legacy Titans inner-update ablation.
            "ltm_training_mode": "read-only",
            # Token/wall clocks remain available for filtering and provenance.
            # Adding an untrained, full-amplitude absolute sinusoid after a write
            # made tiny fast-memory updates produce learning-rate-independent
            # logit jumps, so coherent-v9 does not expose it to the model.
            "ltm_time_feature_mode": "metadata-only",
            # Train the already-present cheap validation writer without changing
            # model geometry.  A deterministic stride keeps the auxiliary well
            # below the cost of the recurrent language path.
            "ltm_value_alignment_weight": 0.01,
            "ltm_value_alignment_stride": 8,
            "ltm_value_alignment_min_updates": 100,
            "ltm_value_alignment_ready_threshold": 0.95,
            "ltm_value_alignment_ema_decay": 0.95,
            "ltm_value_writer_max_norm": 64.0,
            "adaptive_ponder": True,
            "ponder_objective": "symmetric-huber",
            "inference_logit_clamp": 0.0,
            "inference_logit_parity": True,
            "l_conv_atol": 1e-4,
            "training_chunk_size": COHERENT_TRAINING_CHUNK_SIZE,
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
            "ltm_training_mode": "inner-update",
            "ltm_time_feature_mode": "absolute-sinusoidal",
            "ltm_value_alignment_weight": 0.0,
            "ltm_value_alignment_stride": 1,
            "ltm_value_alignment_min_updates": 100,
            "ltm_value_alignment_ready_threshold": 0.95,
            "ltm_value_alignment_ema_decay": 0.95,
            "ltm_value_writer_max_norm": 64.0,
            "adaptive_ponder": False,
            "ponder_objective": "auto",
            "inference_logit_clamp": 30.0,
            "inference_logit_parity": False,
            "l_conv_atol": 0.01,
            "training_chunk_size": LEGACY_TRAINING_CHUNK_SIZE,
        }
    for name, value in defaults.items():
        _setdefault(config, name, value)
    _setdefault(
        config,
        "commitment_threshold",
        architecture_default_commitment_threshold(config),
    )

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
        if _get(config, "token_adapter_rank", None) in (None, "", "auto", 0):
            _set(config, "token_adapter_rank", min(64, int(context_dim)))
    _setdefault(
        config,
        "reference_chunk_len",
        _get(
            config,
            "training_chunk_size",
            architecture_default_training_chunk_size(config),
        ),
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
    validate_architecture_numeric_contract(config)
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
