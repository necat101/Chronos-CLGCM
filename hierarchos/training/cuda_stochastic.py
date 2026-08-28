"""Optional fused CUDA implementations of Hierarchos stochastic primitives.

This module is deliberately outside the stochastic-state contract. It consumes
the exact same immutable Philox reservation as the Python and Vulkan paths and
may disappear entirely without changing model/checkpoint semantics.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

import torch


_EXTENSION = None
_EXTENSION_ERROR: BaseException | None = None
_WARNED_EXTENSION_ERROR = False

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _extension_disabled() -> bool:
    value = os.environ.get("HIERARCHOS_DISABLE_CUDA_STOCHASTIC", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or _extension_disabled():
        return None
    if torch.version.cuda is None or not torch.cuda.is_available():
        return None

    try:
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).with_name("csrc")
        _EXTENSION = load(
            name="hierarchos_cuda_stochastic_v1",
            sources=[
                str(source_dir / "canonical_dropout_cuda.cpp"),
                str(source_dir / "canonical_dropout_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=os.environ.get("HIERARCHOS_CUDA_STOCHASTIC_VERBOSE") == "1",
        )
    except BaseException as error:  # pragma: no cover - host toolchain dependent
        _EXTENSION_ERROR = error
        return None
    return _EXTENSION


def cuda_stochastic_extension_error() -> BaseException | None:
    """Return the cached JIT-build/load failure, if a CUDA attempt has occurred."""

    return _EXTENSION_ERROR


def _warn_fallback_once() -> None:
    global _WARNED_EXTENSION_ERROR
    if _WARNED_EXTENSION_ERROR or _EXTENSION_ERROR is None:
        return
    _WARNED_EXTENSION_ERROR = True
    warnings.warn(
        "Hierarchos fused canonical CUDA dropout is unavailable; falling back "
        f"to the portable Philox reference path ({_EXTENSION_ERROR})",
        RuntimeWarning,
        stacklevel=3,
    )


def _launch(
    input_tensor: torch.Tensor,
    seed: int,
    start_word: int,
    word_count: int,
    threshold: int,
    scale: float,
) -> torch.Tensor | None:
    extension = _load_extension()
    if extension is None:
        _warn_fallback_once()
        return None
    return extension.canonical_dropout(
        input_tensor.contiguous(),
        seed,
        start_word,
        word_count,
        threshold,
        scale,
    ).reshape(input_tensor.shape)


class _CanonicalCudaDropout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        seed: int,
        start_word: int,
        word_count: int,
        threshold: int,
        scale: float,
    ) -> torch.Tensor:
        output = _launch(
            input_tensor,
            seed,
            start_word,
            word_count,
            threshold,
            scale,
        )
        if output is None:
            raise RuntimeError("canonical CUDA dropout extension disappeared after dispatch")
        ctx.seed = seed
        ctx.start_word = start_word
        ctx.word_count = word_count
        ctx.threshold = threshold
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _launch(
            grad_output,
            ctx.seed,
            ctx.start_word,
            ctx.word_count,
            ctx.threshold,
            ctx.scale,
        )
        if grad_input is None:
            raise RuntimeError(
                "canonical CUDA dropout extension became unavailable during backward"
            )
        return grad_input, None, None, None, None, None


def canonical_cuda_dropout(
    input_tensor: torch.Tensor,
    *,
    seed: int,
    start_word: int,
    word_count: int,
    threshold: int,
    scale: float,
) -> torch.Tensor | None:
    """Apply canonical dropout in one CUDA kernel, or return ``None`` for fallback."""

    if not input_tensor.is_cuda or input_tensor.dtype not in _SUPPORTED_DTYPES:
        return None
    extension = _load_extension()
    if extension is None:
        _warn_fallback_once()
        return None
    return _CanonicalCudaDropout.apply(
        input_tensor,
        int(seed),
        int(start_word),
        int(word_count),
        int(threshold),
        float(scale),
    )
