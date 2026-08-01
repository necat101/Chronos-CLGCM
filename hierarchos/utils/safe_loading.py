"""Fail-closed helpers for loading PyTorch tensor artifacts.

``torch.load`` used unrestricted pickle by default on older PyTorch releases.
Every active Hierarchos artifact loader goes through this module so an older
runtime can never silently turn a requested safe load into arbitrary code
execution.
"""

from contextlib import nullcontext
import inspect

import torch


_SAFE_LOAD_REQUIREMENT = (
    "Safe .pt artifact loading requires PyTorch >= 1.13 with explicit "
    "torch.load(..., weights_only=True) support"
)
_SAFE_GLOBALS_REQUIREMENT = (
    "allowlisted Hierarchos checkpoints additionally require PyTorch >= 2.4 "
    "with torch.serialization.safe_globals"
)


def _safe_load_unavailable(*, needs_safe_globals: bool, detail: str = ""):
    requirement = _SAFE_LOAD_REQUIREMENT
    if needs_safe_globals:
        requirement += "; " + _SAFE_GLOBALS_REQUIREMENT
    suffix = f" ({detail})" if detail else ""
    return RuntimeError(
        f"{requirement}. Refusing to fall back to unrestricted pickle loading; "
        f"upgrade PyTorch before loading this artifact{suffix}."
    )


def _torch_load_supports_weights_only() -> bool:
    """Return whether this runtime explicitly implements restricted loading."""
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        # An opaque/builtin loader cannot prove that it honors the security
        # contract, so treat it as unsupported instead of probing with a file.
        return False


def load_tensor_payload_safely(
    path,
    *,
    map_location="cpu",
    allowed_globals=None,
):
    """Load a tensor payload with the restricted PyTorch unpickler.

    ``allowed_globals`` is reserved for narrow, project-owned checkpoint
    classes. Passing it requires PyTorch's scoped ``safe_globals`` context.
    There is intentionally no legacy ``torch.load`` fallback.
    """
    needs_safe_globals = allowed_globals is not None
    if not _torch_load_supports_weights_only():
        raise _safe_load_unavailable(
            needs_safe_globals=needs_safe_globals,
            detail="weights_only is unavailable",
        )

    safe_context = nullcontext()
    if needs_safe_globals:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if not callable(safe_globals):
            raise _safe_load_unavailable(
                needs_safe_globals=True,
                detail="safe_globals is unavailable",
            )
        safe_context = safe_globals(list(allowed_globals))

    try:
        with safe_context:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )
    except TypeError as exc:
        # Capability wrappers and vendor forks can expose a misleading
        # signature. A single restricted attempt is safe; retrying without the
        # keyword would not be.
        raise _safe_load_unavailable(
            needs_safe_globals=needs_safe_globals,
            detail="the runtime rejected weights_only=True",
        ) from exc
