"""Adaptive-computation primitives shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ACTWeights:
    weights: torch.Tensor
    remainder: torch.Tensor
    expected_steps: torch.Tensor


@dataclass(frozen=True)
class HardACTSelection:
    output: torch.Tensor
    state: torch.Tensor
    executed_steps: torch.Tensor
    selected_index: torch.Tensor


def hard_act_depth_straight_through(
    halt_probabilities: torch.Tensor,
    executed_steps: torch.Tensor,
    *,
    threshold: float,
    min_steps: int = 1,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Return actual hard depth with a differentiable quantile-depth gradient.

    Hard inference executes another step while the cumulative halt CDF remains
    below ``threshold``. The forward value is therefore the exact selected
    runtime depth; the backward surrogate smoothly approximates those same
    continuation decisions. This prevents the ponder objective from optimizing
    a distribution mean while runtime uses a different quantile.
    """

    if halt_probabilities.ndim != 2 or halt_probabilities.shape[0] <= 0:
        raise ValueError(
            "halt_probabilities must have shape [positive_steps, batch], "
            f"got {tuple(halt_probabilities.shape)}"
        )
    steps, batch = halt_probabilities.shape
    if executed_steps.shape != (batch,):
        raise ValueError(
            f"executed_steps must have shape [{batch}], "
            f"got {tuple(executed_steps.shape)}"
        )
    min_steps = int(min_steps)
    if min_steps <= 0 or min_steps > steps:
        raise ValueError(f"min_steps must be in [1, {steps}], got {min_steps}")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    temperature = float(temperature)
    if not temperature > 0.0:
        raise ValueError(
            f"temperature must be positive, got {temperature}"
        )

    probabilities = halt_probabilities.float().clamp(1e-6, 1.0 - 1e-6)
    cumulative_halt = 1.0 - torch.cumprod(
        1.0 - probabilities,
        dim=0,
    )
    if min_steps >= steps:
        soft_depth = cumulative_halt.new_full((batch,), float(steps))
    else:
        # After completing step i+1, continue iff CDF_i < threshold.
        continuation_cdf = cumulative_halt[min_steps - 1 : steps - 1]
        soft_depth = (
            cumulative_halt.new_full((batch,), float(min_steps))
            + torch.sigmoid(
                (threshold - continuation_cdf) / temperature
            ).sum(dim=0)
        )
    hard_depth = executed_steps.to(dtype=soft_depth.dtype)
    return soft_depth + (hard_depth - soft_depth).detach()


def normalized_act_weights(halt_probabilities: torch.Tensor) -> ACTWeights:
    """Compute stable Graves-style ACT weights in float32.

    Input shape is ``[steps, batch]``.  The remainder is assigned to the last
    computed state, preserving total probability mass exactly.
    """

    if halt_probabilities.ndim != 2 or halt_probabilities.shape[0] <= 0:
        raise ValueError(
            "halt_probabilities must have shape [positive_steps, batch], "
            f"got {tuple(halt_probabilities.shape)}"
        )
    probs = torch.nan_to_num(
        halt_probabilities.float(),
        nan=0.5,
        posinf=1.0 - 1e-6,
        neginf=1e-6,
    ).clamp(1e-6, 1.0 - 1e-6)
    remain = 1.0 - probs
    shifted = torch.cat((torch.ones_like(remain[:1]), remain[:-1]), dim=0)
    survival = torch.cumprod(shifted, dim=0)
    weights = probs * survival
    remainder = survival[-1] * remain[-1]
    total = weights.sum(dim=0) + remainder
    safe_total = total.clamp_min(1e-8)
    weights = weights / safe_total.unsqueeze(0)
    remainder = remainder / safe_total
    expected_steps = survival.sum(dim=0)
    return ACTWeights(
        weights=weights,
        remainder=remainder,
        expected_steps=expected_steps,
    )


def hard_act_selection(
    outputs: torch.Tensor,
    states: torch.Tensor,
    halt_probabilities: torch.Tensor,
    *,
    threshold: float,
    min_steps: int = 1,
) -> HardACTSelection:
    """Select the first per-row halt without batch-coupled control flow.

    ``outputs`` has shape ``[steps, batch, hidden]`` and ``states`` has shape
    ``[steps, batch, ...]``.  A row that never crosses the threshold uses the
    final step.  This full-stack formulation is compile friendly; an eager
    caller may stop producing steps once every row has crossed because the
    selected values can no longer change.
    """

    if outputs.ndim != 3:
        raise ValueError(f"outputs must have shape [steps, batch, hidden], got {tuple(outputs.shape)}")
    if states.ndim < 3:
        raise ValueError(f"states must have shape [steps, batch, ...], got {tuple(states.shape)}")
    if halt_probabilities.ndim != 2:
        raise ValueError(
            "halt_probabilities must have shape [steps, batch], "
            f"got {tuple(halt_probabilities.shape)}"
        )
    if outputs.shape[:2] != states.shape[:2] or outputs.shape[:2] != halt_probabilities.shape:
        raise ValueError(
            "ACT outputs, states, and halt probabilities must share [steps, batch]"
        )
    steps, batch = halt_probabilities.shape
    if steps <= 0:
        raise ValueError("ACT requires at least one step")
    min_steps = int(min_steps)
    if min_steps <= 0 or min_steps > steps:
        raise ValueError(f"min_steps must be in [1, {steps}], got {min_steps}")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    probabilities = torch.nan_to_num(
        halt_probabilities.float(),
        nan=0.5,
        posinf=1.0 - 1e-6,
        neginf=1e-6,
    ).clamp(1e-6, 1.0 - 1e-6)
    # The network emits conditional halt hazards. Hard execution must threshold
    # their cumulative CDF, not an individual hazard, or its semantics diverge
    # from the expected-step objective.
    cumulative_halt = 1.0 - torch.cumprod(1.0 - probabilities, dim=0)
    eligible = torch.arange(steps, device=halt_probabilities.device).unsqueeze(1) >= (min_steps - 1)
    crossed = (cumulative_halt >= threshold) & eligible
    # Force a decision at the final available step.
    crossed = crossed.clone()
    crossed[-1] = True
    selected_index = crossed.to(dtype=torch.int64).argmax(dim=0)
    batch_index = torch.arange(batch, device=selected_index.device)
    selected_output = outputs[selected_index, batch_index]
    selected_state = states[selected_index, batch_index]
    return HardACTSelection(
        output=selected_output,
        state=selected_state,
        executed_steps=selected_index.to(dtype=torch.float32) + 1.0,
        selected_index=selected_index,
    )
