"""Backend-neutral stochastic training primitives for Hierarchos.

This module is intentionally a correctness/reference path. The same Philox
word mapping is implemented by ``hierarchos-vulkan`` without depending on a
PyTorch, CUDA, or Vulkan-native RNG state. Fast fused CUDA kernels can replace
the Python word materialization later without changing the checkpoint ABI.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn


CANONICAL_COUNTER_RNG_ALGORITHM = "philox4x32-10-word-v1"

_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85


def _u32(value: int) -> int:
    return int(value) & _U32_MASK


def canonical_philox_word(seed: int, word_offset: int) -> int:
    """Return one Random123-compatible Philox4x32-10 output word.

    ABI mapping: ``word_offset // 4`` occupies counter lanes 0/1, counter
    lanes 2/3 are zero, and the 64-bit seed occupies key lanes 0/1.
    """

    seed = int(seed)
    word_offset = int(word_offset)
    if not 0 <= seed <= _U64_MASK or not 0 <= word_offset <= _U64_MASK:
        raise ValueError("canonical RNG seed/word offset must fit unsigned 64-bit range")
    block = word_offset >> 2
    lane = word_offset & 3
    counter = [block & _U32_MASK, (block >> 32) & _U32_MASK, 0, 0]
    key = [seed & _U32_MASK, (seed >> 32) & _U32_MASK]
    for _ in range(10):
        p0 = _PHILOX_M0 * counter[0]
        p1 = _PHILOX_M1 * counter[2]
        counter = [
            _u32((p1 >> 32) ^ counter[1] ^ key[0]),
            _u32(p1),
            _u32((p0 >> 32) ^ counter[3] ^ key[1]),
            _u32(p0),
        ]
        key[0] = _u32(key[0] + _PHILOX_W0)
        key[1] = _u32(key[1] + _PHILOX_W1)
    return counter[lane]


@dataclass(frozen=True)
class CanonicalRngReservation:
    seed: int
    start_word: int
    word_count: int

    def word(self, offset: int) -> int:
        offset = int(offset)
        if not 0 <= offset < self.word_count:
            raise IndexError("canonical RNG reservation word offset is out of range")
        absolute = self.start_word + offset
        if absolute > _U64_MASK:
            raise OverflowError("canonical RNG reservation word offset overflow")
        return canonical_philox_word(self.seed, absolute)


@dataclass
class CanonicalRngReservationTape:
    """Immutable-reservation sequence captured by one checkpointed forward."""

    reservations: list[CanonicalRngReservation] = field(default_factory=list)


@dataclass
class CanonicalRngState:
    seed: int
    next_word: int = 0
    algorithm: str = CANONICAL_COUNTER_RNG_ALGORITHM
    _recording_tape: CanonicalRngReservationTape | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _replay_tape: CanonicalRngReservationTape | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _replay_index: int = field(default=0, init=False, repr=False, compare=False)

    def validate(self) -> None:
        if self.algorithm != CANONICAL_COUNTER_RNG_ALGORITHM:
            raise ValueError(f"unsupported canonical RNG algorithm {self.algorithm!r}")
        if not 0 <= int(self.seed) <= _U64_MASK:
            raise ValueError("canonical RNG seed must fit unsigned 64-bit range")
        if not 0 <= int(self.next_word) <= _U64_MASK:
            raise ValueError("canonical RNG next_word must fit unsigned 64-bit range")

    def reserve_words(self, word_count: int) -> CanonicalRngReservation:
        self.validate()
        word_count = int(word_count)
        if word_count < 0:
            raise ValueError("canonical RNG reservation length must be non-negative")
        if self._replay_tape is not None:
            if self._replay_index >= len(self._replay_tape.reservations):
                raise RuntimeError(
                    "canonical RNG checkpoint replay requested more reservations "
                    "than the original forward"
                )
            reservation = self._replay_tape.reservations[self._replay_index]
            self._replay_index += 1
            if reservation.seed != int(self.seed):
                raise RuntimeError(
                    "canonical RNG checkpoint replay seed changed after the original forward"
                )
            if reservation.word_count != word_count:
                raise RuntimeError(
                    "canonical RNG checkpoint replay changed stochastic tensor geometry: "
                    f"recorded {reservation.word_count} words, requested {word_count}"
                )
            # Replay consumes the immutable reservation by value. It must not
            # advance the model-level cursor a second time.
            return reservation
        end = int(self.next_word) + word_count
        if end > _U64_MASK:
            raise OverflowError("canonical RNG word cursor overflow")
        reservation = CanonicalRngReservation(
            seed=int(self.seed),
            start_word=int(self.next_word),
            word_count=word_count,
        )
        self.next_word = end
        if self._recording_tape is not None:
            self._recording_tape.reservations.append(reservation)
        return reservation

    @contextmanager
    def record_reservations(self, tape: CanonicalRngReservationTape):
        """Capture reservations made by one original checkpointed forward."""

        if self._recording_tape is not None or self._replay_tape is not None:
            raise RuntimeError("canonical RNG reservation contexts cannot be nested")
        self._recording_tape = tape
        try:
            yield
        finally:
            self._recording_tape = None

    @contextmanager
    def replay_reservations(self, tape: CanonicalRngReservationTape):
        """Replay an original forward's reservations without moving ``next_word``.

        Non-reentrant PyTorch checkpointing is allowed to stop rematerialization
        once every requested activation has been rebuilt, so consuming only a
        prefix of the tape is valid. Asking for a reservation beyond the tape or
        changing a reservation's element count fails closed in ``reserve_words``.
        """

        if self._recording_tape is not None or self._replay_tape is not None:
            raise RuntimeError("canonical RNG reservation contexts cannot be nested")
        self._replay_tape = tape
        self._replay_index = 0
        try:
            yield
        finally:
            self._replay_tape = None
            self._replay_index = 0

    def execution_policy_state(self) -> dict:
        self.validate()
        return {
            "mode": "canonical-counter",
            "state_required": False,
            "canonical_counter": {
                "algorithm": self.algorithm,
                "seed": int(self.seed),
                "next_word": int(self.next_word),
            },
        }


def dropout_threshold(probability: float) -> int:
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout probability must be finite and in [0, 1)")
    return int(math.floor(probability * (1 << 32)))


def canonical_dropout_from_reservation(
    input_tensor: torch.Tensor,
    reservation: CanonicalRngReservation,
    probability: float,
) -> torch.Tensor:
    """Apply the canonical Hierarchos dropout mask on CPU or CUDA.

    CUDA tensors use the optional fused Hierarchos kernel when the local CUDA
    extension can be loaded. The fused forward and backward both regenerate
    the mask directly from the immutable reservation, so the optimization adds
    no backend RNG state and does not change the checkpoint ABI. Every other
    backend retains the portable host-word reference path.
    """

    probability = float(probability)
    threshold = dropout_threshold(probability)
    element_count = int(input_tensor.numel())
    if reservation.word_count < element_count:
        raise ValueError(
            f"canonical dropout reservation has {reservation.word_count} words "
            f"but input needs {element_count}"
        )
    if element_count == 0:
        return input_tensor.clone()
    if int(reservation.start_word) > _U64_MASK - (element_count - 1):
        raise OverflowError("canonical RNG reservation word offset overflow")

    if input_tensor.is_cuda:
        # Keep the CUDA accelerator an implementation detail of this canonical
        # operation. Import lazily so CPU/Vulkan-only installations never pull
        # in PyTorch's C++/CUDA extension machinery during module import.
        from .cuda_stochastic import canonical_cuda_dropout

        cuda_output = canonical_cuda_dropout(
            input_tensor,
            seed=int(reservation.seed),
            start_word=int(reservation.start_word),
            word_count=int(reservation.word_count),
            threshold=threshold,
            scale=1.0 / (1.0 - probability),
        )
        if cuda_output is not None:
            return cuda_output

    words = torch.tensor(
        [reservation.word(index) for index in range(element_count)],
        dtype=torch.int64,
        device=input_tensor.device,
    ).reshape(input_tensor.shape)
    keep = words >= threshold
    scale = 1.0 / (1.0 - probability)
    return torch.where(keep, input_tensor * scale, torch.zeros_like(input_tensor))


def canonical_dropout(
    input_tensor: torch.Tensor,
    rng: CanonicalRngState,
    probability: float,
) -> tuple[torch.Tensor, CanonicalRngReservation]:
    """Reserve exactly one word per tensor element, then apply canonical dropout."""

    reservation = rng.reserve_words(int(input_tensor.numel()))
    return (
        canonical_dropout_from_reservation(input_tensor, reservation, probability),
        reservation,
    )


class CanonicalDropout(nn.Module):
    """PEFT-compatible dropout whose only stochastic state is Hierarchos Philox."""

    def __init__(self, probability: float, rng: CanonicalRngState):
        super().__init__()
        self.p = float(probability)
        dropout_threshold(self.p)
        self.rng = rng

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input_tensor
        output, _reservation = canonical_dropout(input_tensor, self.rng, self.p)
        return output

    def extra_repr(self) -> str:
        return f"p={self.p}, algorithm={self.rng.algorithm}"


def canonical_rng_checkpoint_context_fn(rng: CanonicalRngState):
    """Return a PyTorch non-reentrant checkpoint context factory for ``rng``.

    Every checkpoint invocation gets a fresh reservation tape. The original
    forward owns cursor advancement; backward rematerialization reuses exactly
    the recorded immutable reservations and therefore never touches a CPU/CUDA
    backend RNG state.
    """

    def context_fn():
        tape = CanonicalRngReservationTape()
        return rng.record_reservations(tape), rng.replay_reservations(tape)

    return context_fn
