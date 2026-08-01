import pytest
import torch
from types import SimpleNamespace

from hierarchos.training.objectives import (
    adaptive_ponder_objective,
    resolve_ponder_objective,
)
from hierarchos.training.trainer import _adaptive_ponder_loss


def test_auto_objective_is_versioned():
    assert resolve_ponder_objective("auto", architecture_revision=None) == "legacy-one-sided"
    assert (
        resolve_ponder_objective("auto", architecture_revision="coherent-v9")
        == "symmetric-huber"
    )


def test_symmetric_objective_penalizes_over_and_under_computation():
    difficulty = torch.tensor(4.0)
    at_target = adaptive_ponder_objective(
        torch.tensor(2.0),
        difficulty,
        max_steps=5,
        target_scale=0.5,
    )
    under = adaptive_ponder_objective(
        torch.tensor(1.0),
        difficulty,
        max_steps=5,
        target_scale=0.5,
    )
    over = adaptive_ponder_objective(
        torch.tensor(3.0),
        difficulty,
        max_steps=5,
        target_scale=0.5,
    )

    torch.testing.assert_close(at_target.target_steps, torch.tensor(2.0))
    torch.testing.assert_close(at_target.loss, torch.tensor(0.0))
    assert under.loss.item() > 0.0
    assert over.loss.item() > 0.0


def test_legacy_objective_retains_one_sided_behavior():
    result = adaptive_ponder_objective(
        torch.tensor(4.0),
        torch.tensor(2.0),
        max_steps=5,
        target_scale=0.5,
        mode="legacy-one-sided",
    )
    torch.testing.assert_close(result.loss, torch.tensor(0.0))


def test_trainer_auto_mode_uses_checkpoint_architecture_contract():
    args = SimpleNamespace(
        ponder_objective="auto",
        max_h_steps=5,
        min_h_steps=1,
        ponder_target_scale=0.5,
        ponder_huber_beta=0.5,
    )
    expected = torch.tensor(4.0)
    difficulty = torch.tensor(2.0)

    coherent = _adaptive_ponder_loss(
        args,
        SimpleNamespace(config=SimpleNamespace(architecture_revision="coherent-v9")),
        expected,
        difficulty,
    )
    legacy = _adaptive_ponder_loss(
        args,
        SimpleNamespace(config=SimpleNamespace(architecture_revision="legacy-v8")),
        expected,
        difficulty,
    )

    assert coherent.item() > 0.0
    torch.testing.assert_close(legacy, torch.tensor(0.0))


def test_difficulty_is_detached_but_halt_policy_gets_gradient():
    expected = torch.tensor(3.0, requires_grad=True)
    difficulty = torch.tensor(2.0, requires_grad=True)

    result = adaptive_ponder_objective(
        expected,
        difficulty,
        max_steps=5,
        target_scale=0.5,
    )
    result.loss.backward()

    assert expected.grad is not None
    assert expected.grad.item() != 0.0
    assert difficulty.grad is None


def test_token_weights_ignore_padding_and_nonfinite_items():
    expected = torch.tensor([[1.0, 4.0, float("nan")]], requires_grad=True)
    difficulty = torch.tensor([[2.0, 2.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0, 1.0]])

    result = adaptive_ponder_objective(
        expected,
        difficulty,
        max_steps=5,
        target_scale=1.0,
        weight=weight,
    )

    assert torch.isfinite(result.loss)
    # Only item zero contributes: smooth-L1(|1 - 2|, beta=.5) = .75.
    torch.testing.assert_close(result.loss, torch.tensor(0.75))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_steps": 0, "target_scale": 1.0}, "max_steps"),
        ({"max_steps": 5, "target_scale": -1.0}, "target_scale"),
        ({"max_steps": 5, "target_scale": 1.0, "min_steps": 0.0}, "min_steps"),
        ({"max_steps": 5, "target_scale": 1.0, "huber_beta": 0.0}, "huber_beta"),
    ],
)
def test_invalid_objective_configuration_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        adaptive_ponder_objective(torch.tensor(1.0), torch.tensor(1.0), **kwargs)
