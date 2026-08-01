import pytest
from types import SimpleNamespace

from hierarchos.evaluation.selection import (
    BestMetric,
    extract_selection_metric,
    parse_metric_selector,
)
from hierarchos.training.trainer import initialize_best_metric_tracker


RESULTS = {
    "results": {
        "hellaswag": {
            "acc,none": 0.51,
            "acc_norm,none": 0.57,
            "acc_stderr,none": 0.01,
        }
    }
}


def test_selector_preserves_metric_commas():
    assert parse_metric_selector("hellaswag:acc_norm,none") == (
        "hellaswag",
        "acc_norm,none",
    )
    assert extract_selection_metric(RESULTS, "hellaswag:acc_norm,none") == 0.57


def test_metric_extraction_fails_closed_for_wrong_task_or_metric():
    with pytest.raises(ValueError, match="absent"):
        extract_selection_metric(RESULTS, "arc_easy:acc,none")
    with pytest.raises(ValueError, match="absent"):
        extract_selection_metric(RESULTS, "hellaswag:f1,none")


def test_best_metric_max_mode_and_resume_state():
    tracker = BestMetric("hellaswag:acc_norm,none", mode="max")

    assert tracker.update(0.50, epoch=1)
    assert not tracker.update(0.49, epoch=2)
    assert tracker.update(0.58, epoch=3, step=100)

    restored = BestMetric.from_state_dict(tracker.state_dict())
    assert restored.value == 0.58
    assert restored.epoch == 3
    assert restored.step == 100
    assert not restored.would_improve(0.58)


def test_best_metric_min_mode():
    tracker = BestMetric("validation:loss", mode="min", value=2.0)

    assert tracker.update(1.9, epoch=2)
    assert not tracker.update(2.1, epoch=3)


def test_trainer_best_metric_state_resumes_exactly():
    args = SimpleNamespace(
        best_checkpoint_metric="hellaswag:acc_norm,none",
        best_checkpoint_mode="max",
    )
    tracker = initialize_best_metric_tracker(
        args,
        {
            "best_metric_state": {
                "selector": "hellaswag:acc_norm,none",
                "mode": "max",
                "value": 0.57,
                "epoch": 2,
                "step": 100,
            }
        },
    )

    assert tracker.value == 0.57
    assert args._best_metric_state == tracker.state_dict()


def test_trainer_rejects_changed_selection_rule_on_resume():
    args = SimpleNamespace(
        best_checkpoint_metric="hellaswag:acc_norm,none",
        best_checkpoint_mode="min",
    )
    with pytest.raises(RuntimeError, match="selector changed"):
        initialize_best_metric_tracker(
            args,
            {
                "best_metric_state": {
                    "selector": "hellaswag:acc_norm,none",
                    "mode": "max",
                    "value": 0.57,
                }
            },
        )


@pytest.mark.parametrize("selector", ["", "hellaswag", ":acc,none", "hellaswag:"])
def test_invalid_selector_rejected(selector):
    with pytest.raises(ValueError):
        BestMetric(selector)
