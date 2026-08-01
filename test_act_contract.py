import pytest
import torch

from hierarchos.models.act import (
    hard_act_depth_straight_through,
    hard_act_selection,
    normalized_act_weights,
)


def test_soft_act_probability_mass_and_expected_steps():
    probabilities = torch.tensor(
        [
            [0.5, 0.1],
            [0.5, 0.2],
            [0.5, 0.9],
        ]
    )

    result = normalized_act_weights(probabilities)

    torch.testing.assert_close(
        result.weights.sum(dim=0) + result.remainder,
        torch.ones(2),
    )
    torch.testing.assert_close(
        result.expected_steps,
        torch.tensor([1.75, 2.62]),
    )


def test_hard_act_selects_each_row_independently():
    # Value encodes step*10 + row, making the chosen step unambiguous.
    outputs = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
            [[30.0], [31.0], [32.0]],
        ]
    )
    states = outputs.unsqueeze(-1).expand(-1, -1, 1, 2).clone()
    probabilities = torch.tensor(
        [
            [0.95, 0.10, 0.10],
            [0.99, 0.91, 0.10],
            [0.99, 0.99, 0.20],
            [0.99, 0.99, 0.30],
        ]
    )

    result = hard_act_selection(
        outputs,
        states,
        probabilities,
        threshold=0.9,
    )

    assert result.selected_index.tolist() == [0, 1, 3]
    assert result.executed_steps.tolist() == [1.0, 2.0, 4.0]
    torch.testing.assert_close(result.output[:, 0], torch.tensor([0.0, 11.0, 32.0]))
    torch.testing.assert_close(result.state[:, 0, 0], torch.tensor([0.0, 11.0, 32.0]))


def test_hard_act_respects_minimum_steps():
    outputs = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    states = outputs.unsqueeze(-1)
    probabilities = torch.full((3, 1), 0.99)

    result = hard_act_selection(
        outputs,
        states,
        probabilities,
        threshold=0.9,
        min_steps=2,
    )

    assert result.selected_index.item() == 1


def test_hard_act_uses_the_same_cumulative_hazard_as_soft_act():
    outputs = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
    states = outputs.unsqueeze(-1)
    probabilities = torch.full((5, 1), 0.60)

    soft = normalized_act_weights(probabilities)
    hard = hard_act_selection(
        outputs,
        states,
        probabilities,
        threshold=0.90,
    )

    # CDF: 0.60, 0.84, 0.936, so the hard path must stop at step three.
    assert hard.selected_index.item() == 2
    assert hard.executed_steps.item() == 3.0
    assert soft.expected_steps.item() == pytest.approx(1.6496, rel=1e-5)


def test_straight_through_ponder_value_matches_actual_quantile_depth():
    probabilities = torch.full((5, 1), 0.50, requires_grad=True)
    outputs = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
    selection = hard_act_selection(
        outputs,
        outputs.unsqueeze(-1),
        probabilities,
        threshold=0.90,
    )

    depth = hard_act_depth_straight_through(
        probabilities,
        selection.executed_steps,
        threshold=0.90,
        temperature=0.05,
    )

    # CDF: .5, .75, .875, .9375 => the 90th-percentile policy runs 4,
    # even though the distribution mean is 1.9375.
    assert depth.item() == 4.0
    assert normalized_act_weights(probabilities).expected_steps.item() == pytest.approx(
        1.9375
    )
    depth.sum().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad) > 0


def test_hard_act_selection_is_batch_composition_invariant():
    torch.manual_seed(5)
    outputs = torch.randn(5, 3, 7)
    states = torch.randn(5, 3, 4, 6)
    probabilities = torch.rand(5, 3)

    batched = hard_act_selection(
        outputs,
        states,
        probabilities,
        threshold=0.6,
    )
    for row in range(3):
        single = hard_act_selection(
            outputs[:, row : row + 1],
            states[:, row : row + 1],
            probabilities[:, row : row + 1],
            threshold=0.6,
        )
        torch.testing.assert_close(batched.output[row], single.output[0])
        torch.testing.assert_close(batched.state[row], single.state[0])
        torch.testing.assert_close(
            batched.executed_steps[row],
            single.executed_steps[0],
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_hard_act_rejects_invalid_threshold(threshold):
    outputs = torch.zeros(1, 1, 1)
    states = torch.zeros(1, 1, 1)
    probabilities = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="threshold"):
        hard_act_selection(
            outputs,
            states,
            probabilities,
            threshold=threshold,
        )
