from types import SimpleNamespace

import pytest
import torch

from hierarchos.models.shared_adapters import (
    SharedTokenAdapter,
    resolve_adapter_rank,
    resolve_token_adapter_modes,
    shared_token_lookup,
)


def test_configs_without_revision_remain_legacy_compatible():
    config = SimpleNamespace(use_deepembed=True, use_rosa=True)

    deepembed_mode, rosa_mode = resolve_token_adapter_modes(config)

    assert deepembed_mode == "legacy-table"
    assert rosa_mode == "legacy-table"
    assert config.architecture_revision == "legacy-v8"


def test_coherent_revision_selects_factorized_shared_adapters():
    config = {
        "architecture_revision": "coherent-v9",
        "use_deepembed": True,
        "use_rosa": True,
    }

    deepembed_mode, rosa_mode = resolve_token_adapter_modes(config)

    assert deepembed_mode == "shared-factorized"
    assert rosa_mode == "shared-factorized"
    assert config["deepembed_mode"] == "shared-factorized"
    assert config["rosa_embedding_mode"] == "shared-factorized"


def test_explicit_disable_wins_over_representation_mode():
    config = SimpleNamespace(
        architecture_revision="coherent-v9",
        use_deepembed=False,
        deepembed_mode="shared-factorized",
        use_rosa=False,
        rosa_embedding_mode="legacy-table",
    )

    assert resolve_token_adapter_modes(config) == ("off", "off")


def test_invalid_adapter_mode_fails_before_model_allocation():
    config = SimpleNamespace(use_deepembed=True, deepembed_mode="mystery", use_rosa=False)

    with pytest.raises(ValueError, match="deepembed_mode"):
        resolve_token_adapter_modes(config)


def test_adapter_neutral_initialization_and_gradient_flow():
    torch.manual_seed(3)
    features = torch.randn(4, 8, requires_grad=True)
    adapter = SharedTokenAdapter(8, 16, 3, output_bias=1.0)

    initial = adapter(features)
    torch.testing.assert_close(initial, torch.ones_like(initial))

    # The zero-initialized up projection receives a gradient immediately.  Once
    # it moves, gradients also reach the down projection and shared features.
    initial.square().mean().backward()
    assert adapter.up.weight.grad is not None
    assert torch.count_nonzero(adapter.up.weight.grad) > 0

    with torch.no_grad():
        adapter.up.weight.add_(0.01)
    adapter.zero_grad(set_to_none=True)
    features.grad = None
    adapter(features).square().mean().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad) > 0
    assert adapter.down.weight.grad is not None
    assert torch.count_nonzero(adapter.down.weight.grad) > 0


def test_no_prediction_sentinel_has_zero_shared_lookup_mask():
    weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    ids = torch.tensor([[0, 4, 5, -1]])

    features, mask = shared_token_lookup(ids, weight, vocab_size=5)

    torch.testing.assert_close(features[0, 0], weight[0])
    torch.testing.assert_close(features[0, 1], weight[4])
    assert mask.tolist() == [[[1.0], [1.0], [0.0], [0.0]]]


def test_adapter_rank_is_bounded_by_default_and_serialized():
    config = {}

    assert resolve_adapter_rank(config, input_dim=32) == 32
    assert config["token_adapter_rank"] == 32
