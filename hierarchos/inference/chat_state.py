"""Helpers for Hierarchos chat runtime state files.

This module is intentionally inference-only. It knows how to describe and
normalize recurrent chat state tensors, including the RWKV v8 packed matrix
state layout, without touching model weights or training state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from ..models.revisions import (
    architecture_contract,
    architecture_contract_hash,
    validate_architecture_contract,
)
from ..utils.rosa import ROSAState


CHAT_STATE_KIND = "hierarchos_chat_runtime_state"
CHAT_STATE_VERSION = 4
LEGACY_CHAT_STATE_VERSIONS = frozenset({1, 2, 3})
CHAT_CONTINUITY_METADATA_VERSION = 1
MAX_CHAT_HISTORY_TURNS = 256
MAX_CHAT_HISTORY_CHARS = 1_000_000
RWKV_V8_LAYOUT = "rwkv_v8_matrix_packed"
LEGACY_SCALAR_LAYOUT = "legacy_scalar_wkv"

_SIGNATURE_KEYS = (
    "context_dim",
    "h_hidden",
    "l_hidden",
    "h_stride",
    "max_h_steps",
    "max_l_steps",
    "vocab_size",
)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _maybe_int(value: Any) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def tensor_to_cpu(value: Any) -> Optional[torch.Tensor]:
    return value.detach().cpu().clone() if torch.is_tensor(value) else None


def clear_ltm_working_memory(model: Any) -> bool:
    """Clear transient LTM working memory for a fresh inference session."""
    ltm = getattr(model, "ltm", None)
    if ltm is None:
        return False

    if hasattr(ltm, "reset_working_memory"):
        ltm.reset_working_memory()
        return True

    cleared = False
    with torch.no_grad():
        for attr in ("fast_vals", "_mom_vals", "timestamps", "wallclock_timestamps"):
            value = getattr(ltm, attr, None)
            if torch.is_tensor(value):
                value.zero_()
                cleared = True

        sources = getattr(ltm, "sources", None)
        if torch.is_tensor(sources):
            sources.fill_(int(getattr(ltm, "SRC_UNKNOWN", 0)))
            cleared = True

    return cleared


def chat_state_config_signature(config: Any, model: Any = None) -> Dict[str, Any]:
    """Small architecture fingerprint for model-neutral chat state files."""
    signature: Dict[str, Any] = {}
    for key in _SIGNATURE_KEYS:
        value = _config_value(config, key, None)
        if value is not None:
            signature[key] = _maybe_int(value)

    rwkv_head_size = _config_value(config, "rwkv_head_size", None)
    if rwkv_head_size is None and model is not None:
        h_rnn = getattr(model, "h_rnn", None)
        rwkv_head_size = getattr(h_rnn, "head_size", None)
    if rwkv_head_size is not None:
        signature["rwkv_head_size"] = _maybe_int(rwkv_head_size)

    return signature


def chat_state_architecture_metadata(config: Any) -> Dict[str, Any]:
    """Return the exact learned-function contract for a new state file."""
    return {
        "architecture_contract": architecture_contract(config),
        "architecture_contract_sha256": architecture_contract_hash(config),
    }


def _cell_state_spec(cell: Any, fallback_hidden: Any = None) -> Dict[str, Any]:
    hidden = getattr(cell, "n_embd", fallback_hidden)
    state_size = getattr(cell, "state_size", None)
    head_size = getattr(cell, "head_size", None)
    n_head = getattr(cell, "n_head", None)

    if state_size is None and hidden is not None:
        state_size = 5

    layout = RWKV_V8_LAYOUT if state_size is not None and head_size is not None else LEGACY_SCALAR_LAYOUT
    spec: Dict[str, Any] = {"layout": layout}

    for key, value in (
        ("hidden", hidden),
        ("state_size", state_size),
        ("head_size", head_size),
        ("n_head", n_head),
        ("matrix_offset", getattr(cell, "matrix_offset", None)),
        ("state_readout_mode", getattr(cell, "state_readout_mode", None)),
        ("layer_id", getattr(cell, "layer_id", None)),
        ("n_layer", getattr(cell, "n_layer", None)),
    ):
        if value is not None:
            spec[key] = _maybe_int(value)

    return spec


def recurrent_state_layout(model: Any = None, config: Any = None) -> Dict[str, Dict[str, Any]]:
    """Return expected recurrent state layout for the loaded inference model."""
    layout: Dict[str, Dict[str, Any]] = {}
    for label, attr, hidden_key in (
        ("h", "h_rnn", "h_hidden"),
        ("l", "l_rnn", "l_hidden"),
    ):
        cell = getattr(model, attr, None) if model is not None else None
        fallback_hidden = _config_value(config, hidden_key, _config_value(config, "context_dim", None))
        if cell is None and fallback_hidden is None:
            continue
        layout[label] = _cell_state_spec(cell, fallback_hidden)
    return layout


def _shape(value: Any) -> Optional[list[int]]:
    if torch.is_tensor(value):
        return [int(dim) for dim in value.shape]
    return None


def _cell_exact_shape(cell: Any, batch_size: int) -> list[int]:
    hidden = getattr(cell, "n_embd", None)
    state_size = getattr(cell, "state_size", None)
    if hidden is None:
        raise RuntimeError("Recurrent cell does not expose n_embd geometry.")
    if state_size is None:
        # The supported legacy quantized RWKV cell has an exact five-slot
        # scalar state but predates the explicit ``state_size`` attribute.
        state_size = 5
    return [int(batch_size), int(hidden), int(state_size)]


def recurrent_state_metadata(
    *,
    model: Any = None,
    config: Any = None,
    h_state: Any = None,
    l_state: Any = None,
) -> Dict[str, Any]:
    return {
        "recurrent_state_layout": recurrent_state_layout(model, config),
        "recurrent_state_shapes": {
            "h_state": _shape(h_state),
            "l_state": _shape(l_state),
        },
    }


def _payload_version(payload: Dict[str, Any]) -> int:
    version = payload.get("version")
    if isinstance(version, bool):
        raise RuntimeError("Chat state has an invalid boolean version.")
    try:
        version = int(version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Chat state is missing a valid version.") from exc
    if version > CHAT_STATE_VERSION:
        raise RuntimeError(
            f"Chat state version {version} is newer than supported version "
            f"{CHAT_STATE_VERSION}."
        )
    if version != CHAT_STATE_VERSION and version not in LEGACY_CHAT_STATE_VERSIONS:
        raise RuntimeError(f"Unsupported chat state version: {version}.")
    return version


def _validate_strict_recurrent_shapes(
    payload: Dict[str, Any],
    model: Any,
) -> Optional[int]:
    saved_shapes = payload.get("recurrent_state_shapes")
    if not isinstance(saved_shapes, dict):
        raise RuntimeError(
            "Version-4 chat state is missing recurrent_state_shapes metadata."
        )

    batch_size: Optional[int] = None
    present_states = []
    for state_name, module_name in (
        ("h_state", "h_rnn"),
        ("l_state", "l_rnn"),
    ):
        state = payload.get(state_name)
        actual_shape = _shape(state)
        declared_shape = saved_shapes.get(state_name)
        if declared_shape != actual_shape:
            raise RuntimeError(
                f"Chat state {state_name} shape metadata mismatch: "
                f"declared={declared_shape}, actual={actual_shape}."
            )
        if state is None:
            continue
        present_states.append(state_name)
        if state.dim() != 3 or int(state.shape[0]) <= 0:
            raise RuntimeError(
                f"Chat state {state_name} must have exact [B, C, S] shape; "
                f"got {actual_shape}."
            )
        cell = getattr(model, module_name, None) if model is not None else None
        if cell is None:
            raise RuntimeError(
                f"Cannot verify strict {state_name} geometry without model.{module_name}."
            )
        expected_shape = _cell_exact_shape(cell, int(state.shape[0]))
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Chat state recurrent tensor shape mismatch for {state_name}: "
                f"saved={actual_shape}, current={expected_shape}."
            )
        if state.is_floating_point() and not bool(torch.isfinite(state).all().item()):
            raise RuntimeError(f"Chat state {state_name} contains non-finite values.")
        if batch_size is None:
            batch_size = int(state.shape[0])
        elif int(state.shape[0]) != batch_size:
            raise RuntimeError(
                "Chat state recurrent tensors disagree on batch size: "
                f"expected {batch_size}, got {int(state.shape[0])} for {state_name}."
            )

    if len(present_states) == 1:
        raise RuntimeError(
            "Version-4 chat state must contain both h_state and l_state, or neither."
        )
    return batch_size


def _validate_strict_context_states(
    payload: Dict[str, Any],
    config: Any,
    batch_size: Optional[int],
) -> Optional[int]:
    """Validate every non-recurrent tensor carrier in a current chat state."""
    context_dim = int(_config_value(config, "context_dim", 0) or 0)
    if context_dim <= 0:
        raise RuntimeError("Current model has no valid context_dim for chat-state validation.")

    context_names = ("prev_context", "target_context", "drift_state")
    present = [name for name in context_names if payload.get(name) is not None]
    if batch_size is not None and len(present) != len(context_names):
        missing = sorted(set(context_names) - set(present))
        raise RuntimeError(
            "Version-4 chat state is missing context carrier(s): "
            + ", ".join(missing)
        )
    if batch_size is None and present:
        raise RuntimeError(
            "Version-4 chat state cannot contain context carriers without recurrent state."
        )

    for name in present:
        value = payload.get(name)
        if not torch.is_tensor(value):
            raise RuntimeError(f"Chat state {name} must be a tensor.")
        expected = (int(batch_size), context_dim)
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"Chat state {name} shape mismatch: saved={tuple(value.shape)}, "
                f"current={expected}."
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Chat state {name} contains non-finite values.")
    return batch_size


def _validated_total_tokens(payload: Dict[str, Any]) -> int:
    value = payload.get("total_tokens_generated", 0)
    if isinstance(value, bool):
        raise RuntimeError("Chat state total_tokens_generated cannot be boolean.")
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            "Chat state total_tokens_generated must be a nonnegative integer."
        ) from exc
    if value < 0:
        raise RuntimeError(
            "Chat state total_tokens_generated must be a nonnegative integer."
        )
    return value


def _validated_bounded_int(
    value: Any,
    *,
    name: str,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"Chat state {name} cannot be boolean.")
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Chat state {name} must be a nonnegative integer."
        ) from exc
    if value < 0:
        raise RuntimeError(f"Chat state {name} must be a nonnegative integer.")
    if maximum is not None and value > maximum:
        raise RuntimeError(
            f"Chat state {name} exceeds the supported maximum of {maximum}."
        )
    return value


def validate_chat_continuity_metadata(
    payload: Dict[str, Any],
    *,
    expected_prefill_chunk_size: Any = None,
) -> Optional[Dict[str, Any]]:
    """Validate the optional CLI chat-continuity envelope.

    Version-4 state files created before this envelope remain loadable. New CLI
    state files include it so the absolute TBPTT phase and the bounded textual
    turn history cannot silently change across a resumed session.
    """
    metadata = payload.get("chat_continuity")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise RuntimeError("Chat state has malformed chat_continuity metadata.")

    version = _validated_bounded_int(
        metadata.get("version"),
        name="chat_continuity version",
    )
    if version != CHAT_CONTINUITY_METADATA_VERSION:
        raise RuntimeError(
            "Unsupported chat continuity metadata version: "
            f"{version}."
        )

    prefill_chunk_size = _validated_bounded_int(
        metadata.get("prefill_chunk_size"),
        name="prefill_chunk_size",
    )
    history_max_turns = _validated_bounded_int(
        metadata.get("history_max_turns"),
        name="history_max_turns",
        maximum=MAX_CHAT_HISTORY_TURNS,
    )
    history_max_chars = _validated_bounded_int(
        metadata.get("history_max_chars"),
        name="history_max_chars",
        maximum=MAX_CHAT_HISTORY_CHARS,
    )
    carry_chat_state = metadata.get("carry_chat_state")
    if not isinstance(carry_chat_state, bool):
        raise RuntimeError("Chat state carry_chat_state must be boolean.")

    turn_history = metadata.get("turn_history")
    if not isinstance(turn_history, list):
        raise RuntimeError("Chat state turn_history must be a list of strings.")
    if len(turn_history) > history_max_turns:
        raise RuntimeError(
            "Chat state turn_history exceeds its persisted turn bound."
        )
    if any(not isinstance(turn, str) or not turn.strip() for turn in turn_history):
        raise RuntimeError(
            "Chat state turn_history must contain only nonempty strings."
        )
    history_text = "\n\n".join(turn_history)
    if len(history_text) > history_max_chars:
        raise RuntimeError(
            "Chat state turn_history exceeds its persisted character bound."
        )
    if (history_max_turns == 0 or history_max_chars == 0) and turn_history:
        raise RuntimeError(
            "Chat state turn_history must be empty when a history bound is zero."
        )

    total_tokens = _validated_total_tokens(payload)
    expected_phase = total_tokens % prefill_chunk_size if prefill_chunk_size > 0 else 0
    saved_phase = _validated_bounded_int(
        metadata.get("absolute_chunk_phase"),
        name="absolute_chunk_phase",
    )
    if saved_phase != expected_phase:
        raise RuntimeError(
            "Chat state prefill chunk phase is inconsistent with its token offset: "
            f"saved={saved_phase}, expected={expected_phase}."
        )

    if expected_prefill_chunk_size is not None:
        requested = _validated_bounded_int(
            expected_prefill_chunk_size,
            name="requested prefill_chunk_size",
        )
        if requested != prefill_chunk_size:
            raise RuntimeError(
                "Chat state prefill chunk geometry mismatch: "
                f"saved={prefill_chunk_size}, requested={requested}. "
                "Resume without an override or use the saved chunk size."
            )

    return {
        "prefill_chunk_size": prefill_chunk_size,
        "absolute_chunk_phase": saved_phase,
        "history_max_turns": history_max_turns,
        "history_max_chars": history_max_chars,
        "carry_chat_state": carry_chat_state,
        "turn_history": list(turn_history),
    }


def _validate_rosa_state_structure(state: ROSAState, row_tokens: list[int]) -> None:
    """Reject malformed current ROSA state instead of silently rebuilding it."""
    if state.tokens != row_tokens:
        raise RuntimeError(
            "Version-4 chat ROSA automaton tokens do not match rosa_past_tokens."
        )
    try:
        num_states = int(state.num_states)
        last_state = int(state.last_state)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Chat ROSA state has invalid scalar metadata.") from exc
    if (
        num_states <= 0
        or num_states > 2 * len(row_tokens) + 1
        or last_state < 0
        or last_state >= num_states
    ):
        raise RuntimeError("Chat ROSA state has invalid state-count metadata.")
    if not isinstance(state.transitions, dict):
        raise RuntimeError("Chat ROSA state transitions must be a dictionary.")
    for name in ("suffix_links", "lengths", "endpos"):
        value = getattr(state, name, None)
        if not isinstance(value, list) or len(value) != num_states:
            raise RuntimeError(
                f"Chat ROSA state {name} does not exactly cover its declared states."
            )
    if (
        state.suffix_links[0] != -1
        or state.lengths[0] != 0
        or set(state.transitions) != set(range(num_states))
    ):
        raise RuntimeError("Chat ROSA state has an invalid root/state table.")

    token_count = len(row_tokens)
    for index in range(num_states):
        suffix_link = state.suffix_links[index]
        length = state.lengths[index]
        end_position = state.endpos[index]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (suffix_link, length, end_position)
        ):
            raise RuntimeError("Chat ROSA state arrays must contain integers.")
        if length < 0 or length > token_count:
            raise RuntimeError("Chat ROSA state contains an invalid match length.")
        if end_position < -1 or end_position >= token_count:
            raise RuntimeError("Chat ROSA state contains an invalid end position.")
        if index == 0:
            if suffix_link != -1:
                raise RuntimeError("Chat ROSA root must have suffix link -1.")
        elif (
            suffix_link < 0
            or suffix_link >= num_states
            or state.lengths[suffix_link] >= length
        ):
            # Strictly decreasing lengths make every suffix walk finite and
            # rule out cycles in untrusted persisted automata.
            raise RuntimeError("Chat ROSA state contains an invalid suffix link.")
    if state.lengths[last_state] != token_count:
        raise RuntimeError("Chat ROSA final state does not cover its token history.")

    history_symbols = set(row_tokens)
    for source, transitions in state.transitions.items():
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or source < 0
            or source >= num_states
            or not isinstance(transitions, dict)
        ):
            raise RuntimeError("Chat ROSA state contains an invalid transition source.")
        for token, target in transitions.items():
            if (
                isinstance(token, bool)
                or not isinstance(token, int)
                or token not in history_symbols
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target < 0
                or target >= num_states
                or state.lengths[target] <= state.lengths[source]
            ):
                raise RuntimeError("Chat ROSA state contains an invalid transition.")


def _validate_strict_rosa_state(
    payload: Dict[str, Any],
    config: Any,
    batch_size: Optional[int],
    total_tokens: int,
) -> None:
    past_tokens = payload.get("rosa_past_tokens")
    rosa_states = payload.get("rosa_states")
    if past_tokens is None:
        if rosa_states is not None:
            raise RuntimeError(
                "Version-4 chat state cannot contain ROSA automata without token history."
            )
        return
    if not torch.is_tensor(past_tokens):
        raise RuntimeError("Chat state rosa_past_tokens must be a tensor.")
    if past_tokens.dtype != torch.long or past_tokens.dim() != 2:
        raise RuntimeError(
            "Chat state rosa_past_tokens must have int64 [B, T] layout."
        )
    rosa_batch = int(past_tokens.shape[0])
    if rosa_batch <= 0:
        raise RuntimeError("Chat state ROSA history must contain at least one batch row.")
    if batch_size is not None and rosa_batch != batch_size:
        raise RuntimeError(
            f"Chat state ROSA batch {rosa_batch} does not match recurrent batch "
            f"{batch_size}."
        )
    vocab_size = int(_config_value(config, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        raise RuntimeError("Current model has no valid vocabulary for ROSA validation.")
    if past_tokens.numel() > 0:
        valid_ids = (past_tokens >= 0) & (past_tokens < vocab_size)
        if not bool(valid_ids.all().item()):
            raise RuntimeError(
                "Chat state ROSA history contains token IDs outside the model vocabulary."
            )
    if total_tokens < int(past_tokens.shape[1]):
        raise RuntimeError(
            "Chat state token offset is shorter than its retained ROSA history."
        )
    if rosa_states is None:
        # A checked rebuild from token history is an explicitly supported state.
        return
    if not isinstance(rosa_states, (list, tuple)) or len(rosa_states) != rosa_batch:
        raise RuntimeError(
            "Chat state rosa_states must contain exactly one automaton per batch row."
        )
    for row, state in enumerate(rosa_states):
        if state is None:
            continue
        if not isinstance(state, ROSAState):
            raise RuntimeError("Chat state contains an unsupported ROSA automaton type.")
        _validate_rosa_state_structure(state, past_tokens[row].tolist())


def validate_chat_state_payload_compatible(
    payload: Dict[str, Any],
    config: Any,
    model: Any = None,
) -> bool:
    """Raise on incompatibility and return whether legacy migration is allowed."""
    version = _payload_version(payload)
    allow_legacy_migration = version in LEGACY_CHAT_STATE_VERSIONS
    saved = payload.get("config_signature") or {}
    current = chat_state_config_signature(config, model)
    signature_keys = (
        ("context_dim", "h_hidden", "l_hidden", "vocab_size")
        if allow_legacy_migration
        else _SIGNATURE_KEYS
    )
    for key in signature_keys:
        if key in saved and key in current and saved[key] != current[key]:
            raise RuntimeError(
                f"Chat state was saved for {key}={saved[key]}, "
                f"but the loaded model has {key}={current[key]}."
            )

    saved_layout = payload.get("recurrent_state_layout") or {}
    current_layout = recurrent_state_layout(model, config)
    for label in ("h", "l"):
        saved_spec = saved_layout.get(label) or {}
        current_spec = current_layout.get(label) or {}
        if not saved_spec or not current_spec:
            continue

        # Only explicitly versioned legacy files may migrate. A current-format
        # file must describe the exact active recurrent representation.
        if (
            allow_legacy_migration
            and saved_spec.get("layout") == LEGACY_SCALAR_LAYOUT
        ):
            continue

        layout_keys = (
            "layout",
            "hidden",
            "state_size",
            "head_size",
            "n_head",
            "matrix_offset",
            "state_readout_mode",
        )
        for key in layout_keys:
            if key in saved_spec and key in current_spec and saved_spec[key] != current_spec[key]:
                raise RuntimeError(
                    f"Chat state recurrent layout mismatch for {label}_state: "
                    f"saved {key}={saved_spec[key]}, current {key}={current_spec[key]}."
                )

    if allow_legacy_migration:
        return True

    if not isinstance(payload.get("recurrent_state_layout"), dict):
        raise RuntimeError(
            "Version-4 chat state is missing recurrent_state_layout metadata."
        )
    batch_size = _validate_strict_recurrent_shapes(payload, model)
    _validate_strict_context_states(payload, config, batch_size)
    total_tokens = _validated_total_tokens(payload)
    _validate_strict_rosa_state(
        payload,
        config,
        batch_size,
        total_tokens,
    )

    expected_contract = payload.get("architecture_contract")
    expected_hash = payload.get("architecture_contract_sha256")
    if not isinstance(expected_contract, dict) or expected_hash is None:
        raise RuntimeError(
            "Version-4 chat state is missing its architecture contract/hash."
        )
    try:
        validate_architecture_contract(
            config,
            expected_contract=expected_contract,
            expected_hash=expected_hash,
            source="chat runtime state",
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    return False


def _legacy_initial_state(batch_size: int, hidden: int, device: Any = None) -> torch.Tensor:
    state = torch.zeros(int(batch_size), int(hidden), 5, device=device, dtype=torch.float32)
    state[:, :, 3] = -1e30
    return state


def normalize_recurrent_state_for_model(
    state: Any,
    model: Any,
    module_name: str,
    *,
    device: Any = None,
    batch_size: Optional[int] = None,
    allow_legacy_migration: bool = False,
) -> Optional[torch.Tensor]:
    """Convert a loaded recurrent state tensor to the active model layout.

    Version-1 chat files saved legacy 5-slot RWKV state. RWKV v8 uses
    [B, C, 3 + head_size], so this helper performs the same conservative
    migration the cell uses at runtime and returns a tensor ready to pass into
    inference.
    """
    if not torch.is_tensor(state):
        return None

    cell = getattr(model, module_name, None) if model is not None else None
    target_device = device if device is not None else state.device
    if not allow_legacy_migration:
        if cell is None:
            raise RuntimeError(
                f"Cannot verify strict recurrent state without model.{module_name}."
            )
        expected = tuple(
            _cell_exact_shape(
                cell,
                int(state.shape[0]) if state.dim() >= 1 else 0,
            )
        )
        if state.dim() != 3 or tuple(state.shape) != expected:
            raise RuntimeError(
                f"Chat state recurrent tensor shape mismatch for {module_name}: "
                f"saved={tuple(state.shape)}, current={expected}."
            )
        return state.to(device=target_device, dtype=torch.float32).detach()

    squeezed = state.squeeze(0) if state.dim() == 4 and state.shape[0] == 1 else state
    inferred_batch = int(batch_size or (squeezed.shape[0] if squeezed.dim() >= 1 else 1))

    if cell is not None and hasattr(cell, "_prepare_state") and hasattr(cell, "n_embd"):
        dummy = torch.zeros(
            inferred_batch,
            int(cell.n_embd),
            device=target_device,
            dtype=torch.float32,
        )
        old_migration_flag = getattr(cell, "allow_legacy_state_migration", None)
        if old_migration_flag is not None:
            cell.allow_legacy_state_migration = True
        try:
            prepared = cell._prepare_state(state, dummy)
        finally:
            if old_migration_flag is not None:
                cell.allow_legacy_state_migration = old_migration_flag
        return prepared.detach()

    if cell is None:
        return state.to(device=target_device, dtype=torch.float32).detach()

    hidden = int(getattr(cell, "n_embd", squeezed.shape[1] if squeezed.dim() >= 2 else 0))
    if squeezed.dim() == 3 and squeezed.shape == (inferred_batch, hidden, 5):
        return squeezed.to(device=target_device, dtype=torch.float32).detach()

    migrated = _legacy_initial_state(inferred_batch, hidden, device=target_device)
    source = squeezed.to(device=target_device, dtype=torch.float32)
    if source.dim() == 3 and hidden > 0:
        common_b = min(inferred_batch, source.shape[0])
        common_c = min(hidden, source.shape[1])
        if source.shape[-1] > 0:
            migrated[:common_b, :common_c, 0] = source[:common_b, :common_c, 0]
        if source.shape[-1] > 1:
            migrated[:common_b, :common_c, 4] = source[:common_b, :common_c, 1]
    return migrated.detach()
