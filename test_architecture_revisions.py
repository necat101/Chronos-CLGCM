from types import SimpleNamespace

import pytest

from hierarchos.models.revisions import (
    ARCHITECTURE_CONTRACT_SCHEMA_VERSION,
    COHERENT_TRAINING_CHUNK_SIZE,
    COHERENT_REVISION,
    LEGACY_TRAINING_CHUNK_SIZE,
    LEGACY_REVISION,
    apply_architecture_revision_defaults,
    architecture_default_training_chunk_size,
    architecture_contract,
    architecture_contract_hash,
)


def test_absent_revision_is_strictly_legacy():
    config = {}

    assert apply_architecture_revision_defaults(config) == LEGACY_REVISION
    assert config["core_recurrence_version"] == 1
    assert config["manager_compute_mode"] == "soft-act"
    assert config["commitment_cost_mode"] == "sum-square"
    assert config["deepembed_mode"] == "legacy-table"
    assert config["rosa_embedding_mode"] == "legacy-table"
    assert config["enforce_rosa_max_context"] is False
    assert config["rosa_zero_no_prediction"] is False
    assert config["ltm_training_mode"] == "inner-update"
    assert config["adaptive_ponder"] is False
    assert config["training_chunk_size"] == LEGACY_TRAINING_CHUNK_SIZE
    assert config["reference_chunk_len"] == LEGACY_TRAINING_CHUNK_SIZE


def test_coherent_revision_resolves_corrected_contract():
    config = SimpleNamespace(architecture_revision="v9")

    assert apply_architecture_revision_defaults(config) == COHERENT_REVISION
    assert config.core_recurrence_version == 2
    assert config.manager_compute_mode == "hard-masked"
    assert config.manager_state_commit_mode == "hard-selected"
    assert config.commitment_cost_mode == "mean-square"
    assert config.deepembed_mode == "shared-factorized"
    assert config.rosa_embedding_mode == "shared-factorized"
    assert config.enforce_rosa_max_context is True
    assert config.rosa_zero_no_prediction is True
    assert config.ltm_training_mode == "read-only"
    assert config.adaptive_ponder is True
    assert config.ponder_objective == "symmetric-huber"
    assert config.training_chunk_size == COHERENT_TRAINING_CHUNK_SIZE
    assert config.reference_chunk_len == COHERENT_TRAINING_CHUNK_SIZE


def test_revisioned_chunk_defaults_preserve_explicit_checkpoint_geometry():
    coherent = {
        "architecture_revision": "coherent-v9",
        "training_chunk_size": 128,
    }

    apply_architecture_revision_defaults(coherent)

    assert architecture_default_training_chunk_size("coherent-v9") == 256
    assert architecture_default_training_chunk_size("legacy-v8") == 128
    assert coherent["training_chunk_size"] == 128
    assert coherent["reference_chunk_len"] == 128


def test_explicit_ablation_is_preserved_and_changes_contract_hash():
    base = {"architecture_revision": "coherent-v9"}
    ablation = {
        "architecture_revision": "coherent-v9",
        "deepembed_mode": "off",
    }

    assert architecture_contract(ablation)["deepembed_mode"] == "off"
    assert architecture_contract_hash(base) != architecture_contract_hash(ablation)
    assert len(architecture_contract_hash(base)) == 64


def test_contract_covers_geometry_recurrence_memory_and_objective_settings():
    base = {
        "architecture_revision": "coherent-v9",
        "vocab_size": 100,
        "context_dim": 32,
        "max_h_steps": 5,
        "recurrent_state_clamp": 50.0,
        "rosa_max_context": 512,
        "ponder_loss_weight": 0.01,
    }
    contract = architecture_contract(dict(base))

    assert (
        contract["architecture_contract_schema_version"]
        == ARCHITECTURE_CONTRACT_SCHEMA_VERSION
    )
    for changed_field, changed_value in (
        ("vocab_size", 101),
        ("max_h_steps", 6),
        ("recurrent_state_clamp", 40.0),
        ("rosa_max_context", 256),
        ("ponder_loss_weight", 0.02),
    ):
        changed = dict(base)
        changed[changed_field] = changed_value
        assert architecture_contract_hash(changed) != architecture_contract_hash(base)


def test_contract_hash_is_mapping_order_independent():
    left = {"architecture_revision": "coherent-v9", "token_adapter_rank": 32}
    right = {"token_adapter_rank": 32, "architecture_revision": "coherent-v9"}

    assert architecture_contract_hash(left) == architecture_contract_hash(right)


def test_detach_zero_and_none_share_one_canonical_contract():
    zero = {
        "architecture_revision": "coherent-v9",
        "detach_every_n_steps": 0,
    }
    disabled = {
        "architecture_revision": "coherent-v9",
        "detach_every_n_steps": None,
    }

    assert architecture_contract(zero)["detach_every_n_steps"] is None
    assert architecture_contract_hash(zero) == architecture_contract_hash(disabled)


def test_mutable_writer_capability_is_not_part_of_immutable_architecture_hash():
    before = {
        "architecture_revision": "coherent-v9",
        "val_proj_trained": False,
    }
    after = {
        "architecture_revision": "coherent-v9",
        "val_proj_trained": True,
    }

    assert architecture_contract_hash(before) == architecture_contract_hash(after)


def test_unknown_revision_fails_closed():
    with pytest.raises(ValueError, match="architecture_revision"):
        apply_architecture_revision_defaults({"architecture_revision": "future-maybe"})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("core_recurrence_version", 3),
        ("rwkv_n_layer_hint", 1),
        ("ltm_slots", 3.5),
        ("recurrent_state_clamp", 0),
        ("drift_norm_clamp", -0.1),
        ("drift_delta_scale", -0.1),
        ("memory_gate_warmup_floor", 0.951),
        ("ltm_forget_rate", 1.1),
        ("ltm_momentum", float("nan")),
        ("rosa_max_context", 0),
    ),
)
def test_invalid_numeric_contract_values_fail_before_model_allocation(field, value):
    with pytest.raises(ValueError, match=field):
        apply_architecture_revision_defaults(
            {
                "architecture_revision": "coherent-v9",
                field: value,
            }
        )


def test_dependent_numeric_contract_values_fail_closed():
    with pytest.raises(ValueError, match="ltm_topk"):
        apply_architecture_revision_defaults(
            {
                "architecture_revision": "coherent-v9",
                "ltm_slots": 4,
                "ltm_topk": 5,
            }
        )

    with pytest.raises(ValueError, match="rwkv_head_size"):
        apply_architecture_revision_defaults(
            {
                "architecture_revision": "coherent-v9",
                "context_dim": 24,
                "rwkv_head_size": 7,
            }
        )


@pytest.mark.parametrize("value", (2.5, float("nan"), True))
def test_detach_frequency_rejects_non_integer_values(value):
    with pytest.raises(ValueError, match="detach_every_n_steps"):
        apply_architecture_revision_defaults(
            {
                "architecture_revision": "coherent-v9",
                "detach_every_n_steps": value,
            }
        )


def test_zero_disable_values_remain_explicit_zeroes():
    config = {
        "architecture_revision": "coherent-v9",
        "drift_norm_clamp": 0,
        "drift_delta_scale": 0,
        "rwkv_channel_mix_key_clamp": 0,
        "rwkv_channel_mix_deepembed_clamp": 0,
        "inference_logit_clamp": 0,
        "ltm_lr": 0,
        "ltm_weight_decay": 0,
        "ltm_score_grad_scale": 0,
    }

    apply_architecture_revision_defaults(config)

    for name in (
        "drift_norm_clamp",
        "drift_delta_scale",
        "rwkv_channel_mix_key_clamp",
        "rwkv_channel_mix_deepembed_clamp",
        "inference_logit_clamp",
        "ltm_lr",
        "ltm_weight_decay",
        "ltm_score_grad_scale",
    ):
        assert config[name] == 0.0


def test_adapter_rank_zero_and_auto_resolve_to_one_contract():
    automatic = {
        "architecture_revision": "coherent-v9",
        "context_dim": 32,
    }
    zero_sentinel = {
        "architecture_revision": "coherent-v9",
        "context_dim": 32,
        "token_adapter_rank": 0,
    }

    assert architecture_contract(automatic)["token_adapter_rank"] == 32
    assert architecture_contract(zero_sentinel)["token_adapter_rank"] == 32
    assert architecture_contract_hash(automatic) == architecture_contract_hash(zero_sentinel)
