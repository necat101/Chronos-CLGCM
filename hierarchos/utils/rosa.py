"""
ROSA — Rapid Online Suffix Automaton (RWKV-v8)
Persistent incremental implementation with an asynchronous GPU pipeline.

Key optimizations:
  1. Single-pass persistent automata across TBPTT chunks
  2. Bounded parallel batch processing via ThreadPoolExecutor
  3. Request-owned pinned buffers for race-free asynchronous DMA transfers
  4. CUDA stream-based async pipeline for CPU/GPU overlap
  5. Suffix automaton state persistence across TBPTT chunks (O(T_chunk) not O(T_total))
  6. Proper tie-breaking per v8 spec (largest j on equal match length)
"""

import os
import threading
import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

ROSA_BOUNDED_CONTEXT_MODE = "bounded-segment-v1"
ROSA_UNBOUNDED_CONTEXT_MODE = "legacy-unbounded-v1"


def rosa_context_mode(enforce_max_context: bool) -> str:
    return (
        ROSA_BOUNDED_CONTEXT_MODE
        if bool(enforce_max_context)
        else ROSA_UNBOUNDED_CONTEXT_MODE
    )

# ─────────────────────────────────────────────────────────────
# Numba JIT Detection
# ─────────────────────────────────────────────────────────────
# The active persistent/incremental ROSA path does not call the historical
# stateless Numba kernel. Importing Numba in every trainer/DataLoader process
# therefore adds startup time and resident memory without accelerating a model
# forward. Keep the debug kernel available only through an explicit opt-in.
_USE_NUMBA = os.environ.get("ROSA_USE_NUMBA", "0").lower() not in ("0", "false")
_NUMBA_OK = False

if _USE_NUMBA:
    try:
        import numba as _nb
        _NUMBA_OK = True
    except ImportError:
        _NUMBA_OK = False

# ─────────────────────────────────────────────────────────────
# Thread Pool (process-safe, lazy init)
# ─────────────────────────────────────────────────────────────
_ROSA_POOL = None
_ROSA_POOL_PID = None
_ROSA_PIPELINE_POOL = None
_ROSA_PIPELINE_POOL_PID = None
_ROSA_AUTO_WORKER_LIMIT = 32
_ROSA_POOL_INIT_LOCK = threading.Lock()

def _get_pool(max_workers: int = 0) -> Optional[ThreadPoolExecutor]:
    """Returns a per-process thread pool for parallel ROSA batch computation."""
    global _ROSA_POOL, _ROSA_POOL_PID
    pid = os.getpid()
    if _ROSA_POOL is None or _ROSA_POOL_PID != pid:
        with _ROSA_POOL_INIT_LOCK:
            if _ROSA_POOL is not None and _ROSA_POOL_PID == pid:
                return _ROSA_POOL
            if _ROSA_POOL is not None:
                try:
                    _ROSA_POOL.shutdown(wait=False)
                except Exception:
                    pass
            if max_workers <= 0:
                cpu_count = os.cpu_count() or 1
                gpu_count = 1
                try:
                    gpu_count = max(1, torch.cuda.device_count())
                except Exception:
                    pass
                max_workers = min(
                    _ROSA_AUTO_WORKER_LIMIT,
                    max(1, cpu_count // gpu_count),
                )
            _ROSA_POOL = ThreadPoolExecutor(max_workers=max_workers)
            _ROSA_POOL_PID = pid
    return _ROSA_POOL


def _get_pipeline_pool() -> ThreadPoolExecutor:
    """Separate executor prevents nested ROSA batch work from deadlocking."""
    global _ROSA_PIPELINE_POOL, _ROSA_PIPELINE_POOL_PID
    pid = os.getpid()
    if _ROSA_PIPELINE_POOL is None or _ROSA_PIPELINE_POOL_PID != pid:
        with _ROSA_POOL_INIT_LOCK:
            if (
                _ROSA_PIPELINE_POOL is not None
                and _ROSA_PIPELINE_POOL_PID == pid
            ):
                return _ROSA_PIPELINE_POOL
            if _ROSA_PIPELINE_POOL is not None:
                try:
                    _ROSA_PIPELINE_POOL.shutdown(wait=False)
                except Exception:
                    pass
            cpu_count = os.cpu_count() or 1
            _ROSA_PIPELINE_POOL = ThreadPoolExecutor(
                max_workers=min(4, max(1, cpu_count))
            )
            _ROSA_PIPELINE_POOL_PID = pid
    return _ROSA_PIPELINE_POOL


# ─────────────────────────────────────────────────────────────
# Pinned Memory Buffer Pool
# ─────────────────────────────────────────────────────────────
# A process-global tensor pool is deliberately not used. A same-shaped request
# can begin before an earlier asynchronous DMA completes, so each pipeline call
# owns its host buffers for the full transfer lifetime.

# ─────────────────────────────────────────────────────────────
# Cached CUDA Stream for Async Pipeline
# ─────────────────────────────────────────────────────────────
_ROSA_STREAM = None
_ROSA_STREAM_DEVICE = None

def _get_rosa_stream(device):
    """Returns a cached CUDA stream for ROSA async transfers."""
    global _ROSA_STREAM, _ROSA_STREAM_DEVICE
    if _ROSA_STREAM is None or _ROSA_STREAM_DEVICE != device:
        _ROSA_STREAM = torch.cuda.Stream(device=device)
        _ROSA_STREAM_DEVICE = device
    return _ROSA_STREAM


# ─────────────────────────────────────────────────────────────
# ROSAState — Persistent Automaton State for TBPTT
# ─────────────────────────────────────────────────────────────
@dataclass
class ROSAState:
    """
    Encapsulates the suffix automaton state for one batch element.
    Persisted across TBPTT chunks to enable incremental extension
    instead of full rebuilds.
    """
    # Automaton arrays (dense numpy for Numba compat)
    transitions: dict = field(default_factory=dict)  # state -> {symbol: state}
    suffix_links: list = field(default_factory=list)  # suffix link per state
    lengths: list = field(default_factory=list)        # max length per state
    endpos: list = field(default_factory=list)         # rightmost endpos per state
    last_state: int = 0                                # current "last" state
    num_states: int = 1                                # total allocated states
    # Token history for prediction lookback
    tokens: list = field(default_factory=list)         # full token history

    @staticmethod
    def new():
        """Create a fresh automaton state with initial node."""
        s = ROSAState()
        s.transitions = {0: {}}
        s.suffix_links = [-1]
        s.lengths = [0]
        s.endpos = [-1]
        s.last_state = 0
        s.num_states = 1
        s.tokens = []
        return s


# ─────────────────────────────────────────────────────────────
# Core ROSA Algorithm — Pure Python (Fallback)
# ─────────────────────────────────────────────────────────────
def ROSA(x):
    """
    Rapid Online Suffix Automaton (ROSA) — Original reference implementation.
    Given token sequence x, returns y where y[i] = predicted next token after
    the longest matching suffix of x[0..i] in x[0..i-1].

    space = O(n), time = adaptive, typical O(n), worst-case O(n^2)
    Proper tie-breaking: on equal match length, largest j (most recent) wins.
    """
    n = len(x)
    y = [-1] * n
    s = 2 * n + 1
    b = [None] * s
    c = [-1] * s
    d = [0] * s
    e = [-1] * s
    b[0] = {}
    g = 0
    z = 1

    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u

        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]
                break
            v = c[v]

        y[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]

    return y


def _reset_rosa_state_in_place(state: ROSAState) -> None:
    """Reset an automaton without replacing the caller-owned state object."""
    state.transitions = {0: {}}
    state.suffix_links = [-1]
    state.lengths = [0]
    state.endpos = [-1]
    state.last_state = 0
    state.num_states = 1
    state.tokens = []


def _rosa_incremental(
    state: ROSAState,
    new_tokens: List[int],
    max_context: int = 0,
) -> List[int]:
    """
    Incrementally extend an existing suffix automaton with new tokens.
    Returns predictions for the new tokens only.
    """
    b = state.transitions
    c = state.suffix_links
    d = state.lengths
    e = state.endpos
    g = state.last_state
    z = state.num_states
    all_tokens = state.tokens
    max_context = max(0, int(max_context or 0))

    predictions = []

    for t in new_tokens:
        # A suffix automaton does not support inexpensive deletion. Versioned
        # bounded mode therefore uses deterministic fixed-size segments: once a
        # segment reaches the configured cap, begin a fresh automaton before the
        # next token. This keeps O(max_context) state and remains invariant to
        # caller chunk boundaries.
        if max_context > 0 and len(all_tokens) >= max_context:
            _reset_rosa_state_in_place(state)
            b = state.transitions
            c = state.suffix_links
            d = state.lengths
            e = state.endpos
            g = state.last_state
            z = state.num_states
            all_tokens = state.tokens

        i = len(all_tokens)
        all_tokens.append(t)

        # Extend automaton
        r = z
        z += 1
        if r not in b:
            b[r] = {}
        else:
            b[r] = {}
        if r >= len(c):
            c.extend([-1] * (r - len(c) + 1))
            d.extend([0] * (r - len(d) + 1))
            e.extend([-1] * (r - len(e) + 1))

        d[r] = d[g] + 1
        p = g

        while p != -1 and t not in b.get(p, {}):
            if p not in b:
                b[p] = {}
            b[p][t] = r
            p = c[p]

        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                if u >= len(c):
                    c.extend([-1] * (u - len(c) + 1))
                    d.extend([0] * (u - len(d) + 1))
                    e.extend([-1] * (u - len(e) + 1))
                b[u] = dict(b.get(q, {}))
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b.get(p, {}).get(t) == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u

        # Prediction: walk suffix links for longest match with valid endpos
        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                next_pos = e[v] + 1
                if next_pos < len(all_tokens):
                    a = all_tokens[next_pos]
                break
            v = c[v]

        predictions.append(a)

        # Update endpos: rightmost tracking (v8 tie-breaking)
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]

    state.last_state = g
    state.num_states = z
    return predictions


# ─────────────────────────────────────────────────────────────
# Numba-Accelerated ROSA
# ─────────────────────────────────────────────────────────────
if _NUMBA_OK:
    @_nb.njit(cache=True, nogil=True)
    def _rosa_numba(x_arr, n):
        """Numba JIT-compiled ROSA. Dense-array automaton for maximum throughput."""
        y = np.full(n, -1, dtype=np.int64)
        s = 2 * n + 2
        # Transition table: (state, symbol) -> next_state
        # Use a flat array with hash-like open addressing for transitions
        # For simplicity and cache-friendliness, use suffix links + lengths + endpos
        c = np.full(s, -1, dtype=np.int64)   # suffix links
        d = np.zeros(s, dtype=np.int64)       # lengths
        e = np.full(s, -1, dtype=np.int64)    # endpos (rightmost)

        # Transitions stored as an adjacency list mapped to contiguous flat arrays
        # head[state] points to the first transition index.
        # next_tr[tr_idx] points to the next transition index for the same parent.
        MAX_TRANS = 4 * n + 4
        head = np.full(s, -1, dtype=np.int64)
        next_tr = np.full(MAX_TRANS, -1, dtype=np.int64)
        tr_symbol = np.full(MAX_TRANS, -1, dtype=np.int64)
        tr_child = np.full(MAX_TRANS, -1, dtype=np.int64)
        tr_count = 0

        def _find_trans(parent, symbol):
            """Scan only the outgoing transitions of parent state using adjacency list."""
            idx = head[parent]
            while idx != -1:
                if tr_symbol[idx] == symbol:
                    return tr_child[idx], idx
                idx = next_tr[idx]
            return -1, -1

        def _set_trans(parent, symbol, child):
            nonlocal tr_count
            # Try to update existing transition
            idx = head[parent]
            while idx != -1:
                if tr_symbol[idx] == symbol:
                    tr_child[idx] = child
                    return
                idx = next_tr[idx]
            # Create new transition
            if tr_count < MAX_TRANS:
                tr_symbol[tr_count] = symbol
                tr_child[tr_count] = child
                next_tr[tr_count] = head[parent]
                head[parent] = tr_count
                tr_count += 1
            else:
                raise RuntimeError("ROSA transition buffer exhausted")

        def _copy_trans(src_state, dst_state):
            """Copy all transitions of src_state to dst_state."""
            idx = head[src_state]
            while idx != -1:
                _set_trans(dst_state, tr_symbol[idx], tr_child[idx])
                idx = next_tr[idx]

        g = 0  # last state
        z = 1  # next free state

        for i in range(n):
            t = x_arr[i]
            r = z
            z += 1
            d[r] = d[g] + 1
            p = g

            while p != -1:
                child, _ = _find_trans(p, t)
                if child != -1:
                    break
                _set_trans(p, t, r)
                p = c[p]

            if p == -1:
                c[r] = 0
            else:
                q_child, _ = _find_trans(p, t)
                q = q_child
                if d[p] + 1 == d[q]:
                    c[r] = q
                else:
                    u = z
                    z += 1
                    _copy_trans(q, u)
                    d[u] = d[p] + 1
                    c[u] = c[q]
                    e[u] = e[q]
                    while p != -1:
                        child_p, _ = _find_trans(p, t)
                        if child_p != q:
                            break
                        _set_trans(p, t, u)
                        p = c[p]
                    c[q] = c[r] = u

            v = r
            g = r
            a = np.int64(-1)
            while v != -1:
                if d[v] > 0 and e[v] >= 0:
                    next_pos = e[v] + 1
                    if next_pos < n and next_pos <= i:
                        a = x_arr[next_pos]
                    break
                v = c[v]

            y[i] = a

            v = g
            while v != -1 and e[v] < i:
                e[v] = i
                v = c[v]

        return y

    def _rosa_jit(x_list: List[int]) -> List[int]:
        """Wrapper around Numba ROSA for List[int] input."""
        arr = np.array(x_list, dtype=np.int64)
        result = _rosa_numba(arr, len(arr))
        return result.tolist()

else:
    _rosa_jit = None


# ─────────────────────────────────────────────────────────────
# Batch Processing API
# ─────────────────────────────────────────────────────────────
def rosa_single(
    x: List[int],
    state: Optional[ROSAState] = None,
    max_context: int = 0,
) -> Tuple[List[int], ROSAState]:
    """
    Process a single sequence through ROSA.
    If state is provided, incrementally extends the existing automaton.
    Returns (predictions_for_new_tokens, updated_state).
    """
    if state is not None:
        preds = _rosa_incremental(state, x, max_context=max_context)
        return preds, state

    # Building the persistent automaton already computes the predictions. The
    # old path first ran the stateless JIT kernel and then replayed every token
    # to construct this state, duplicating first-chunk work.
    new_state = ROSAState.new()
    preds = _rosa_incremental(new_state, x, max_context=max_context)
    return preds, new_state


def rosa_batch_parallel(
    batch: List[List[int]],
    states: Optional[List[Optional[ROSAState]]] = None,
    max_context: int = 0,
) -> Tuple[List[List[int]], List[ROSAState]]:
    """
    Process a batch of sequences through ROSA in parallel.

    Args:
        batch: List of B token sequences
        states: Optional list of B ROSAState objects for incremental processing

    Returns:
        (predictions, updated_states) — each is a list of length B
    """
    B = len(batch)
    if B == 0:
        return [], []

    if states is None:
        states = [None] * B
    elif len(states) != B:
        raise ValueError(
            f"ROSA states length {len(states)} does not match batch size {B}"
        )

    # For single-element batches, skip thread pool overhead
    if B == 1:
        preds, new_state = rosa_single(batch[0], states[0], max_context=max_context)
        return [preds], [new_state]

    pool = _get_pool()
    futures = []
    for b in range(B):
        futures.append(
            pool.submit(
                rosa_single,
                batch[b],
                states[b],
                max_context,
            )
        )

    all_preds = []
    all_states = []
    for fut in futures:
        preds, st = fut.result()
        all_preds.append(preds)
        all_states.append(st)

    return all_preds, all_states


# ─────────────────────────────────────────────────────────────
# Async GPU Pipeline
# ─────────────────────────────────────────────────────────────
def precompute_rosa_ids_for_chunks(
    tokens: List[int],
    vocab_size: int,
    chunk_size: int,
    rosa_max_ctx: int = 512,
    enforce_max_context: bool = False,
) -> List[int]:
    """
    Precompute the exact ROSA ids produced by the training forward path.

    This mirrors ``rosa_async_pipeline`` chunk-by-chunk with persistent
    automaton state. Legacy mode carries full history; bounded-segment mode
    resets at the configured boundary and is intentionally versioned.
    """
    tokens = [int(token) for token in tokens]
    total = len(tokens)
    if total == 0:
        return []

    no_prediction = int(vocab_size)
    chunk_size = max(1, int(chunk_size or total))
    cached = [no_prediction] * total
    state = None
    effective_max_context = (
        max(1, int(rosa_max_ctx))
        if enforce_max_context and int(rosa_max_ctx or 0) > 0
        else 0
    )

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        rosa_chunk, state = rosa_single(
            tokens[start:end],
            state,
            max_context=effective_max_context,
        )
        cached[start:end] = [
            no_prediction if int(pred) == -1 else int(pred)
            for pred in rosa_chunk
        ]

    return cached


class _ImmediateFuture:
    """A future-like object that wraps an already-computed value."""
    __slots__ = ("_value",)
    def __init__(self, value):
        self._value = value
    def result(self, timeout=None):
        return self._value


def rosa_async_pipeline(
    input_ids: torch.Tensor,
    past_tokens: Optional[torch.Tensor],
    rosa_states: Optional[List[Optional[ROSAState]]],
    vocab_size: int,
    device: torch.device,
    rosa_max_ctx: int = 512,
    enforce_max_context: bool = False,
):
    """
    Launches ROSA computation asynchronously so CPU work overlaps with GPU.

    Returns a callable (future) that, when called, returns:
        (rosa_batch_tensor, new_past_tokens, new_rosa_states)

    The tensor is ready for embedding lookup on `device`.
    """
    B, T = input_ids.shape
    is_cuda = (device.type == 'cuda')
    no_prediction = vocab_size
    effective_max_context = (
        max(1, int(rosa_max_ctx))
        if enforce_max_context and int(rosa_max_ctx or 0) > 0
        else 0
    )

    # ``ROSAState.tokens`` is the authoritative live history. ``past_tokens`` is
    # retained only as a compatibility/rebuild carrier for states loaded without
    # an automaton. Do not clone a valid CPU history on every autoregressive
    # token; legacy unbounded chat otherwise turns a linear incremental suffix
    # update into quadratic host-copy traffic.
    if past_tokens is not None:
        past_cpu = past_tokens.detach()
        if past_cpu.device.type != "cpu" or past_cpu.dtype != torch.int64:
            past_cpu = past_cpu.to(device="cpu", dtype=torch.int64)
        if past_cpu.dim() == 1:
            past_cpu = past_cpu.unsqueeze(0)
        if past_cpu.shape[0] == 1 and B > 1:
            past_cpu = past_cpu.expand(B, -1)
        elif past_cpu.shape[0] != B:
            raise ValueError(
                f"ROSA past_tokens batch {past_cpu.shape[0]} does not match input batch {B}"
            )
    else:
        past_cpu = None

    # Transfer only the current model tokens to CPU (async if CUDA). The
    # previous history already lives on CPU between chunks in core.py.
    if is_cuda:
        # A fresh per-call buffer prevents concurrent inference requests from
        # overwriting one another's in-flight D2H/H2D transfers.
        host_buf = torch.empty(
            (B, T),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )

        copy_stream = _get_rosa_stream(device)
        copy_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(copy_stream):
            host_buf.copy_(input_ids.to(torch.int64), non_blocking=True)
            ev_copy = torch.cuda.Event()
            ev_copy.record(copy_stream)
    else:
        host_buf = input_ids.to(torch.int64)
        ev_copy = None

    # Build the states list
    if rosa_states is None:
        rosa_states = [None] * B
    elif len(rosa_states) != B:
        raise ValueError(
            f"ROSA state count {len(rosa_states)} does not match batch size {B}"
        )

    # Launch CPU computation on thread pool (overlaps with GPU forward)
    pool = _get_pipeline_pool()

    def _cpu_work():
        # Wait for D2H copy to complete
        if ev_copy is not None:
            ev_copy.synchronize()

        current_tokens_cpu = host_buf.detach().cpu()
        current_input_lists = current_tokens_cpu.tolist()

        state_inputs = []
        state_seeds = []
        rebuild_flags = []

        for row, (current_row, state) in enumerate(
            zip(current_input_lists, rosa_states)
        ):
            if state is not None:
                if not isinstance(state, ROSAState):
                    raise ValueError(
                        f"ROSA state row {row} has unsupported type "
                        f"{type(state).__name__}"
                    )
                if (
                    past_cpu is not None
                    and len(state.tokens) != int(past_cpu.shape[1])
                ):
                    raise ValueError(
                        "ROSA automaton/history length mismatch: "
                        f"state row {row} has {len(state.tokens)} tokens while "
                        f"past_tokens has {int(past_cpu.shape[1])}. A validated "
                        "automaton is authoritative; discard it explicitly to "
                        "request a history rebuild."
                    )
                # Incremental state already represents the complete history.
                # Avoid converting/comparing that history on every token.
                state_inputs.append(current_row)
                state_seeds.append(state)
                rebuild_flags.append(False)
            else:
                # Rebuild from the complete history when state is missing,
                # including legacy chat states that saved token history only.
                past_row = (
                    past_cpu[row].tolist()
                    if past_cpu is not None
                    else []
                )
                state_inputs.append(past_row + current_row)
                state_seeds.append(None)
                rebuild_flags.append(True)

        # Parallel ROSA batch computation
        rosa_preds, new_states = rosa_batch_parallel(
            state_inputs,
            state_seeds,
            max_context=effective_max_context,
        )

        # Return exactly one ROSA token per model token.
        rosa_raw = []
        for preds, rebuilt in zip(rosa_preds, rebuild_flags):
            current_preds = preds[-T:] if rebuilt else preds
            if len(current_preds) != T:
                current_preds = current_preds[-T:] if T > 0 else []
                current_preds = [-1] * (T - len(current_preds)) + current_preds
            rosa_raw.append(current_preds)

        # Vectorized post-processing via numpy
        rosa_np = np.array(rosa_raw, dtype=np.int64)
        rosa_np[rosa_np == -1] = no_prediction  # sentinel for "no prediction"

        # The automata already own their token histories. Carry no duplicate
        # tensor during live execution; chat-state serialization materializes it
        # once on save, and cached-ID training retains its separate tensor fallback
        # in core.py.
        next_past_tokens = None

        # Fresh per-call result storage makes parallel forwards race-free.
        result_buf = torch.empty(
            (B, T),
            dtype=torch.int64,
            device="cpu",
            pin_memory=is_cuda,
        )
        np.copyto(np.asarray(result_buf), rosa_np, casting="unsafe")

        return result_buf, next_past_tokens, new_states

    # CUDA still benefits from overlap for the overwhelmingly common B=1 chat
    # path: run the D2H wait and suffix update on the pipeline worker while the
    # caller performs token embedding. CPU B=1 remains inline to avoid thread
    # scheduling overhead where there is no device work to overlap.
    if pool is not None and (B > 1 or is_cuda):
        future = pool.submit(_cpu_work)
    else:
        future = _ImmediateFuture(_cpu_work())

    def _finalize():
        result_buf, new_past, new_states = future.result() if hasattr(future, 'result') else future._value

        # Async H2D transfer
        rosa_tensor = result_buf.to(device, non_blocking=is_cuda)

        return rosa_tensor, new_past, new_states

    return _finalize


# ─────────────────────────────────────────────────────────────
# Warmup (pre-compile Numba on import)
# ─────────────────────────────────────────────────────────────
def warmup():
    """Pre-compiles the Numba JIT kernel to avoid first-call latency."""
    if _NUMBA_OK and _rosa_jit is not None:
        try:
            _rosa_jit([1, 2, 3, 1, 2])
        except Exception:
            pass

# The persistent path no longer invokes the stateless JIT kernel, so importing
# this module should not launch a background compilation thread. ``warmup`` is
# retained as an explicit compatibility/debugging helper.
