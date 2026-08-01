"""Parameter-efficient token-conditioned adapters used by coherent architectures.

The legacy V8 architecture allocated three additional vocabulary-sized tables:
two 4x-width DeepEmbed tables and one ROSA table.  Those tables dominate the
parameter count and duplicate token identity already represented by the tied
token embedding / language-model head.

New architecture revisions can instead condition small low-rank adapters on the
shared token embedding.  The adapters are deliberately initialized to reproduce
the neutral behavior of the old tables:

* DeepEmbed starts at an all-ones multiplicative modulation.
* ROSA starts at an all-zeros additive modulation.

Legacy checkpoint loading remains in ``core.py`` / ``checkpoint.py``; this
module contains no migration guesses and only implements the new learned path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .revisions import COHERENT_REVISION, apply_architecture_revision_defaults


VALID_DEEPEMBED_MODES = frozenset({"off", "legacy-table", "shared-factorized"})
VALID_ROSA_EMBEDDING_MODES = frozenset({"off", "legacy-table", "shared-factorized"})


def _config_value(config, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _set_config_value(config, name: str, value) -> None:
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def _normalize_mode(raw_value, *, name: str, valid_modes) -> str:
    value = str(raw_value).strip().lower().replace("_", "-")
    aliases = {
        "none": "off",
        "false": "off",
        "legacy": "legacy-table",
        "table": "legacy-table",
        "shared": "shared-factorized",
        "factorized": "shared-factorized",
    }
    value = aliases.get(value, value)
    if value not in valid_modes:
        choices = ", ".join(sorted(valid_modes))
        raise ValueError(f"{name} must be one of {{{choices}}}, got {raw_value!r}")
    return value


def resolve_token_adapter_modes(config) -> tuple[str, str]:
    """Resolve and persist DeepEmbed/ROSA representation modes.

    Absence of an architecture revision is intentionally treated as legacy.
    This makes programmatic configs and old checkpoints retain their learned
    function.  The CLI marks newly-created models as ``coherent-v9``.
    """

    revision = apply_architecture_revision_defaults(config)
    coherent_revision = revision == COHERENT_REVISION

    use_deepembed = bool(_config_value(config, "use_deepembed", True))
    raw_deepembed_mode = _config_value(config, "deepembed_mode", None)
    if not use_deepembed:
        deepembed_mode = "off"
    elif raw_deepembed_mode in (None, "", "auto"):
        deepembed_mode = "shared-factorized" if coherent_revision else "legacy-table"
    else:
        deepembed_mode = _normalize_mode(
            raw_deepembed_mode,
            name="deepembed_mode",
            valid_modes=VALID_DEEPEMBED_MODES,
        )

    use_rosa = bool(_config_value(config, "use_rosa", True))
    raw_rosa_mode = _config_value(config, "rosa_embedding_mode", None)
    if not use_rosa:
        rosa_mode = "off"
    elif raw_rosa_mode in (None, "", "auto"):
        rosa_mode = "shared-factorized" if coherent_revision else "legacy-table"
    else:
        rosa_mode = _normalize_mode(
            raw_rosa_mode,
            name="rosa_embedding_mode",
            valid_modes=VALID_ROSA_EMBEDDING_MODES,
        )

    # Keep the historical feature booleans authoritative and serialize the
    # concrete representation so a checkpoint never depends on future defaults.
    _set_config_value(config, "architecture_revision", revision)
    _set_config_value(config, "use_deepembed", deepembed_mode != "off")
    _set_config_value(config, "deepembed_mode", deepembed_mode)
    _set_config_value(config, "use_rosa", rosa_mode != "off")
    _set_config_value(config, "rosa_embedding_mode", rosa_mode)
    return deepembed_mode, rosa_mode


def resolve_adapter_rank(config, *, input_dim: int) -> int:
    raw_rank = _config_value(config, "token_adapter_rank", None)
    if raw_rank in (None, "", "auto", 0):
        rank = min(64, int(input_dim))
    else:
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token_adapter_rank must be a positive integer, got {raw_rank!r}") from exc
        if rank <= 0:
            raise ValueError(f"token_adapter_rank must be a positive integer, got {raw_rank!r}")
    _set_config_value(config, "token_adapter_rank", rank)
    return rank


class SharedTokenAdapter(nn.Module):
    """Low-rank projection over the tied token embedding.

    ``output_bias`` defines the exact neutral initialization.  The up projection
    starts at zero, so constructing a coherent-v9 model does not inject a random
    vocabulary-conditioned signal before the adapter has learned one.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        *,
        output_bias: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or rank <= 0:
            raise ValueError(
                "SharedTokenAdapter dimensions must be positive; "
                f"got input_dim={input_dim}, output_dim={output_dim}, rank={rank}"
            )
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.norm = nn.LayerNorm(self.input_dim, elementwise_affine=False)
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)
        self.bias = nn.Parameter(torch.full((self.output_dim,), float(output_bias)))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        if token_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected token feature width {self.input_dim}, "
                f"got {token_features.shape[-1]}"
            )
        return self.forward_normalized(self.norm(token_features))

    def forward_normalized(
        self,
        normalized_token_features: torch.Tensor,
    ) -> torch.Tensor:
        """Project features already normalized by an equivalent shared norm."""
        if normalized_token_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected normalized token feature width {self.input_dim}, "
                f"got {normalized_token_features.shape[-1]}"
            )
        hidden = torch.nn.functional.silu(
            self.down(normalized_token_features)
        )
        return self.bias + self.up(hidden)


def shared_token_lookup(
    token_ids: torch.Tensor,
    shared_weight: torch.Tensor,
    *,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up valid token IDs and return an explicit validity mask.

    ROSA uses ``vocab_size`` as its no-prediction sentinel.  Clamping that value
    to a real token would silently inject token 0, so callers must multiply the
    adapter result by the returned mask.
    """

    if token_ids.dtype != torch.long:
        token_ids = token_ids.to(dtype=torch.long)
    valid = (token_ids >= 0) & (token_ids < int(vocab_size))
    safe_ids = token_ids.clamp(min=0, max=max(0, int(vocab_size) - 1))
    features = torch.nn.functional.embedding(safe_ids, shared_weight)
    return features, valid.unsqueeze(-1).to(dtype=features.dtype)
