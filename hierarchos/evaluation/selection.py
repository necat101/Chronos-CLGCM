"""Deterministic checkpoint-selection helpers for lm-eval results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def parse_metric_selector(selector: str) -> tuple[str, str]:
    """Parse ``TASK:METRIC`` while allowing commas inside lm-eval metrics."""

    text = str(selector or "").strip()
    if ":" not in text:
        raise ValueError(
            "best-checkpoint metric must use TASK:METRIC syntax, "
            "for example 'hellaswag:acc_norm,none'"
        )
    task, metric = (part.strip() for part in text.split(":", 1))
    if not task or not metric:
        raise ValueError(
            "best-checkpoint metric must include both task and metric names"
        )
    return task, metric


def extract_selection_metric(results: Mapping[str, Any], selector: str) -> float:
    """Extract one finite scalar from an lm-eval result payload."""

    task, metric = parse_metric_selector(selector)
    task_results = results.get("results") if isinstance(results, Mapping) else None
    if not isinstance(task_results, Mapping):
        raise ValueError("Evaluation payload has no mapping-valued 'results' field")
    metrics = task_results.get(task)
    if not isinstance(metrics, Mapping):
        available = ", ".join(sorted(str(key) for key in task_results)) or "<none>"
        raise ValueError(
            f"Evaluation task {task!r} is absent; available tasks: {available}"
        )
    if metric not in metrics:
        available = ", ".join(sorted(str(key) for key in metrics)) or "<none>"
        raise ValueError(
            f"Metric {metric!r} is absent for task {task!r}; available metrics: {available}"
        )
    raw_value = metrics[metric]
    if isinstance(raw_value, bool):
        raise ValueError(f"Selection metric {selector!r} is boolean, not numeric")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Selection metric {selector!r} is not numeric: {raw_value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Selection metric {selector!r} is non-finite: {raw_value!r}"
        )
    return value


def normalize_selection_mode(mode: str) -> str:
    normalized = str(mode or "max").strip().lower()
    aliases = {"maximize": "max", "higher": "max", "minimize": "min", "lower": "min"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"max", "min"}:
        raise ValueError(f"best-checkpoint mode must be 'max' or 'min', got {mode!r}")
    return normalized


@dataclass
class BestMetric:
    """Serializable comparison state for one immutable selection metric."""

    selector: str
    mode: str = "max"
    value: float | None = None
    epoch: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        parse_metric_selector(self.selector)
        self.mode = normalize_selection_mode(self.mode)
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError(f"Existing best metric must be finite, got {self.value!r}")

    def would_improve(self, candidate: float) -> bool:
        candidate = float(candidate)
        if not math.isfinite(candidate):
            raise ValueError(f"Candidate selection metric must be finite, got {candidate!r}")
        if self.value is None:
            return True
        if self.mode == "max":
            return candidate > float(self.value)
        return candidate < float(self.value)

    def update(self, candidate: float, *, epoch: int, step: int | None = None) -> bool:
        if not self.would_improve(candidate):
            return False
        self.value = float(candidate)
        self.epoch = int(epoch)
        self.step = None if step is None else int(step)
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "mode": self.mode,
            "value": self.value,
            "epoch": self.epoch,
            "step": self.step,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BestMetric":
        return cls(
            selector=str(state["selector"]),
            mode=str(state.get("mode", "max")),
            value=state.get("value"),
            epoch=state.get("epoch"),
            step=state.get("step"),
        )
