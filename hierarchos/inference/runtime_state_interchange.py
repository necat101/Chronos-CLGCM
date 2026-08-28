"""Backend-neutral Hierarchos inference-state interchange.

The JSON schema in this module is shared with the pure-Rust runtime.  It keeps
the recurrent state in the coherent-v9 PyTorch/Vulkan packed layout rather than
serializing backend-specific objects, so a live sequence can cross an inference
backend boundary without replaying its complete prefix.

Schema v1 intentionally covers read-only inference memory state.  The Rust
runtime does not perform the optional PyTorch Hebbian fast-memory write path, so
LTM momentum/timestamp/source metadata are reconstructed from model defaults on
import and are not serialized here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from hierarchos.models.revisions import architecture_contract_hash
from hierarchos.utils.rosa import ROSAState, rosa_single


RUNTIME_STATE_INTERCHANGE_KIND = "hierarchos_runtime_state_interchange"
RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION = 1
RWKV_V8_MATRIX_PACKED_LAYOUT = "rwkv_v8_matrix_packed"
RWKV_EXPLICIT_OUTPUT_MODE = "explicit-output"


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _device_of(model: Any, device: Any = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _finite_flat_tensor(
    name: str,
    value: Any,
    *,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, expected {expected_shape}"
        )
    detached = value.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return detached


def _cell_geometry(cell: Any) -> tuple[int, int, int]:
    hidden = int(getattr(cell, "n_embd"))
    head_size = int(getattr(cell, "head_size"))
    state_size = int(getattr(cell, "state_size"))
    if getattr(cell, "state_readout_mode", None) != RWKV_EXPLICIT_OUTPUT_MODE:
        raise ValueError(
            "runtime-state interchange requires coherent-v9 explicit-output RWKV state"
        )
    if state_size != 4 + head_size:
        raise ValueError(
            f"explicit-output RWKV state_size={state_size} does not equal 4 + head_size={head_size}"
        )
    return hidden, head_size, state_size


def _snapshot_rwkv_state(name: str, value: Any, cell: Any) -> dict[str, Any]:
    hidden, head_size, state_size = _cell_geometry(cell)
    tensor = _finite_flat_tensor(
        name,
        value,
        expected_shape=(1, hidden, state_size),
    )
    return {
        "layout": RWKV_V8_MATRIX_PACKED_LAYOUT,
        "state_readout_mode": RWKV_EXPLICIT_OUTPUT_MODE,
        "hidden": hidden,
        "head_size": head_size,
        "state_size": state_size,
        "shape": [1, hidden, state_size],
        "values": tensor.reshape(-1).tolist(),
    }


def _restore_rwkv_state(
    name: str,
    payload: Any,
    cell: Any,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(payload, Mapping):
        raise ValueError(f"runtime-state {name} must be an object")
    hidden, head_size, state_size = _cell_geometry(cell)
    expected_shape = [1, hidden, state_size]
    geometry = (
        payload.get("layout"),
        payload.get("state_readout_mode"),
        payload.get("hidden"),
        payload.get("head_size"),
        payload.get("state_size"),
        payload.get("shape"),
    )
    expected = (
        RWKV_V8_MATRIX_PACKED_LAYOUT,
        RWKV_EXPLICIT_OUTPUT_MODE,
        hidden,
        head_size,
        state_size,
        expected_shape,
    )
    if geometry != expected:
        raise ValueError(
            f"runtime-state {name} geometry/layout does not match this model: "
            f"saved={geometry!r}, expected={expected!r}"
        )
    values = payload.get("values")
    if not isinstance(values, list) or len(values) != hidden * state_size:
        raise ValueError(
            f"runtime-state {name} packed value count does not match {expected_shape}"
        )
    tensor = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"runtime-state {name} contains non-finite values")
    return tensor.reshape(expected_shape).to(device=device)


def _snapshot_rosa_state(state: ROSAState) -> dict[str, Any]:
    transitions: list[list[dict[str, int]]] = []
    for source in range(int(state.num_states)):
        row = state.transitions.get(source, {})
        transitions.append(
            [
                {"symbol": int(symbol), "target": int(target)}
                for symbol, target in sorted(row.items())
            ]
        )
    return {
        "transitions": transitions,
        "suffix_links": [int(value) for value in state.suffix_links],
        "lengths": [int(value) for value in state.lengths],
        "endpos": [int(value) for value in state.endpos],
        "last_state": int(state.last_state),
        "tokens": [int(token) for token in state.tokens],
    }


def _restore_rosa_state(payload: Any, *, vocab_size: int) -> ROSAState:
    if not isinstance(payload, Mapping):
        raise ValueError("runtime-state rosa must be an object")
    transitions = payload.get("transitions")
    suffix_links = payload.get("suffix_links")
    lengths = payload.get("lengths")
    endpos = payload.get("endpos")
    last_state = payload.get("last_state")
    tokens = payload.get("tokens")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("runtime-state ROSA must contain at least the root state")
    states = len(transitions)
    for name, values in (
        ("suffix_links", suffix_links),
        ("lengths", lengths),
        ("endpos", endpos),
    ):
        if not isinstance(values, list) or len(values) != states:
            raise ValueError(
                f"runtime-state ROSA {name} must cover all {states} automaton states"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"runtime-state ROSA {name} must contain integers")
    if isinstance(last_state, bool) or not isinstance(last_state, int):
        raise ValueError("runtime-state ROSA last_state must be an integer")
    if not isinstance(tokens, list) or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= vocab_size
        for token in tokens
    ):
        raise ValueError("runtime-state ROSA tokens are invalid for this vocabulary")
    if last_state < 0 or last_state >= states:
        raise ValueError("runtime-state ROSA last_state is outside its state table")
    if suffix_links[0] != -1 or lengths[0] != 0:
        raise ValueError("runtime-state ROSA root state is invalid")
    if lengths[last_state] != len(tokens):
        raise ValueError("runtime-state ROSA final state does not cover its token history")

    restored_transitions: dict[int, dict[int, int]] = {}
    for source, row in enumerate(transitions):
        if not isinstance(row, list):
            raise ValueError("runtime-state ROSA transition row must be a list")
        restored_row: dict[int, int] = {}
        for transition in row:
            if not isinstance(transition, Mapping):
                raise ValueError("runtime-state ROSA transition must be an object")
            symbol = transition.get("symbol")
            target = transition.get("target")
            if (
                isinstance(symbol, bool)
                or not isinstance(symbol, int)
                or symbol < 0
                or symbol >= vocab_size
                or isinstance(target, bool)
                or not isinstance(target, int)
                or target < 0
                or target >= states
                or lengths[target] <= lengths[source]
                or symbol in restored_row
            ):
                raise ValueError("runtime-state ROSA contains an invalid transition")
            restored_row[symbol] = target
        restored_transitions[source] = restored_row

    token_count = len(tokens)
    for state_index in range(states):
        suffix = suffix_links[state_index]
        length = lengths[state_index]
        end = endpos[state_index]
        if length < 0 or length > token_count or end < -1 or end >= token_count:
            raise ValueError("runtime-state ROSA state metadata is out of range")
        if state_index == 0:
            if suffix != -1:
                raise ValueError("runtime-state ROSA root suffix link must be -1")
        elif (
            suffix < 0
            or suffix >= states
            or lengths[suffix] >= length
        ):
            raise ValueError("runtime-state ROSA contains an invalid suffix link")

    return ROSAState(
        transitions=restored_transitions,
        suffix_links=list(suffix_links),
        lengths=list(lengths),
        endpos=list(endpos),
        last_state=last_state,
        num_states=states,
        tokens=list(tokens),
    )


def _single_memory_values(
    name: str,
    value: Any,
    *,
    slots: int,
    value_dim: int,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if tuple(value.shape) == (slots, value_dim):
        tensor = value
    elif tuple(value.shape) == (1, slots, value_dim):
        tensor = value[0]
    else:
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}, expected "
            f"{(slots, value_dim)} or {(1, slots, value_dim)}"
        )
    detached = tensor.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return detached


@dataclass
class PortableRuntimeState:
    """PyTorch carrier restored from a backend-neutral runtime snapshot."""

    position: int
    history: list[int]
    h_state: torch.Tensor
    l_state: torch.Tensor
    prev_context: torch.Tensor
    target_context: torch.Tensor
    drift_state: torch.Tensor
    ltm_memory_state: tuple[Any, ...]

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "h_state": self.h_state,
            "l_state": self.l_state,
            "prev_context": self.prev_context,
            "target_context": self.target_context,
            "drift_state": self.drift_state,
            "ltm_memory_state": self.ltm_memory_state,
            "global_pos_offset": self.position,
        }


def snapshot_runtime_state_interchange(
    model: Any,
    *,
    h_state: torch.Tensor,
    l_state: torch.Tensor,
    prev_context: torch.Tensor,
    target_context: torch.Tensor,
    drift_state: torch.Tensor,
    ltm_memory_state: Sequence[Any],
    position: int,
    history: Sequence[int],
) -> dict[str, Any]:
    """Create the exact JSON object understood by the Rust runtime."""

    config = model.config
    vocab_size = int(_config_value(config, "vocab_size"))
    context_dim = int(_config_value(config, "context_dim"))
    slots = int(_config_value(config, "ltm_slots"))
    value_dim = int(_config_value(config, "ltm_val_dim"))
    position = int(position)
    history_list = [int(token) for token in history]
    if position != len(history_list):
        raise ValueError(
            f"runtime-state position/history mismatch: position={position} history={len(history_list)}"
        )
    if any(token < 0 or token >= vocab_size for token in history_list):
        raise ValueError("runtime-state history contains a token outside the model vocabulary")
    if not isinstance(ltm_memory_state, (tuple, list)) or len(ltm_memory_state) < 2:
        raise ValueError("ltm_memory_state must contain at least fast and momentum values")

    fast_vals = _single_memory_values(
        "ltm_memory_state.fast_vals",
        ltm_memory_state[0],
        slots=slots,
        value_dim=value_dim,
    )

    rosa_state = None
    if len(ltm_memory_state) >= 4 and ltm_memory_state[3] is not None:
        states = ltm_memory_state[3]
        if not isinstance(states, (tuple, list)) or len(states) != 1:
            raise ValueError(
                "runtime-state interchange currently supports exactly one ROSA batch row"
            )
        rosa_state = states[0]
        if rosa_state is not None and not isinstance(rosa_state, ROSAState):
            raise ValueError("ltm_memory_state contains an unsupported ROSA state object")
    if rosa_state is None:
        max_context = (
            int(_config_value(config, "rosa_max_context", 0) or 0)
            if bool(_config_value(config, "enforce_rosa_max_context", False))
            else 0
        )
        _, rosa_state = rosa_single(history_list, max_context=max_context)
    if history_list[-len(rosa_state.tokens) :] != rosa_state.tokens if rosa_state.tokens else False:
        raise ValueError("ROSA token history is not a suffix of the canonical runtime history")

    prev = _finite_flat_tensor(
        "prev_context", prev_context, expected_shape=(1, context_dim)
    )
    target = _finite_flat_tensor(
        "target_context", target_context, expected_shape=(1, context_dim)
    )
    drift = _finite_flat_tensor(
        "drift_state", drift_state, expected_shape=(1, context_dim)
    )

    return {
        "kind": RUNTIME_STATE_INTERCHANGE_KIND,
        "schema_version": RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION,
        "architecture_revision": str(_config_value(config, "architecture_revision")),
        "architecture_contract_sha256": architecture_contract_hash(config),
        "position": position,
        "history": history_list,
        "h_state": _snapshot_rwkv_state("h_state", h_state, model.h_rnn),
        "l_state": _snapshot_rwkv_state("l_state", l_state, model.l_rnn),
        "prev_context": prev.reshape(-1).tolist(),
        "target_context": target.reshape(-1).tolist(),
        "final_drift": drift.reshape(-1).tolist(),
        "fast_vals": fast_vals.reshape(-1).tolist(),
        "rosa": _snapshot_rosa_state(rosa_state),
    }


def save_runtime_state_interchange(
    model: Any,
    path: str | Path,
    **state: Any,
) -> Path:
    payload = snapshot_runtime_state_interchange(model, **state)
    output = Path(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_runtime_state_interchange(
    model: Any,
    source: str | Path | Mapping[str, Any],
    *,
    device: Any = None,
) -> PortableRuntimeState:
    """Restore a Rust/PyTorch/Vulkan runtime snapshot for PyTorch inference."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime-state interchange root must be a JSON object")
    if (
        payload.get("kind") != RUNTIME_STATE_INTERCHANGE_KIND
        or payload.get("schema_version") != RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported runtime-state interchange kind/schema "
            f"{payload.get('kind')!r}/{payload.get('schema_version')!r}"
        )

    config = model.config
    revision = str(_config_value(config, "architecture_revision"))
    contract_hash = architecture_contract_hash(config)
    if (
        payload.get("architecture_revision") != revision
        or payload.get("architecture_contract_sha256") != contract_hash
    ):
        raise ValueError(
            "runtime-state learned-function identity does not match this PyTorch model"
        )

    vocab_size = int(_config_value(config, "vocab_size"))
    context_dim = int(_config_value(config, "context_dim"))
    slots = int(_config_value(config, "ltm_slots"))
    value_dim = int(_config_value(config, "ltm_val_dim"))
    position = payload.get("position")
    history = payload.get("history")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError("runtime-state position must be a non-negative integer")
    if not isinstance(history, list) or len(history) != position or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= vocab_size
        for token in history
    ):
        raise ValueError("runtime-state history is invalid for the saved position/vocabulary")

    target_device = _device_of(model, device)
    h_state = _restore_rwkv_state(
        "h_state", payload.get("h_state"), model.h_rnn, device=target_device
    )
    l_state = _restore_rwkv_state(
        "l_state", payload.get("l_state"), model.l_rnn, device=target_device
    )

    def restore_vector(name: str, width: int) -> torch.Tensor:
        values = payload.get(name)
        if not isinstance(values, list) or len(values) != width:
            raise ValueError(
                f"runtime-state {name} has invalid width; expected {width} values"
            )
        tensor = torch.tensor(values, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"runtime-state {name} contains non-finite values")
        return tensor.reshape(1, width).to(device=target_device)

    prev_context = restore_vector("prev_context", context_dim)
    target_context = restore_vector("target_context", context_dim)
    drift_state = restore_vector("final_drift", context_dim)
    fast_values = payload.get("fast_vals")
    expected_fast = slots * value_dim
    if not isinstance(fast_values, list) or len(fast_values) != expected_fast:
        raise ValueError(
            f"runtime-state fast_vals has invalid size; expected {expected_fast} values"
        )
    fast_vals = torch.tensor(fast_values, dtype=torch.float32)
    if not bool(torch.isfinite(fast_vals).all().item()):
        raise ValueError("runtime-state fast_vals contains non-finite values")
    fast_vals = fast_vals.reshape(slots, value_dim).to(device=target_device)

    rosa_state = _restore_rosa_state(payload.get("rosa"), vocab_size=vocab_size)
    if len(rosa_state.tokens) > len(history) or (
        rosa_state.tokens and history[-len(rosa_state.tokens) :] != rosa_state.tokens
    ):
        raise ValueError("runtime-state ROSA history is not a suffix of canonical history")

    base_momentum = getattr(model.ltm, "_mom_vals", None)
    if not torch.is_tensor(base_momentum) or tuple(base_momentum.shape) != (slots, value_dim):
        raise ValueError("PyTorch model has incompatible LTM momentum geometry")
    momentum = base_momentum.detach().float().to(device=target_device).clone()
    past_tokens = torch.tensor([rosa_state.tokens], dtype=torch.long)
    ltm_memory_state = (fast_vals, momentum, past_tokens, [rosa_state])

    return PortableRuntimeState(
        position=position,
        history=list(history),
        h_state=h_state,
        l_state=l_state,
        prev_context=prev_context,
        target_context=target_context,
        drift_state=drift_state,
        ltm_memory_state=ltm_memory_state,
    )


__all__ = [
    "PortableRuntimeState",
    "RUNTIME_STATE_INTERCHANGE_KIND",
    "RUNTIME_STATE_INTERCHANGE_SCHEMA_VERSION",
    "RWKV_V8_MATRIX_PACKED_LAYOUT",
    "load_runtime_state_interchange",
    "save_runtime_state_interchange",
    "snapshot_runtime_state_interchange",
]
