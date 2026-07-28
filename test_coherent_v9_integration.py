import pytest
import torch

from hierarchos import AttrDict, HierarchosCore
from hierarchos.inference.chat import build_chat_ltm_checkpoint_payload
from hierarchos.models.revisions import architecture_contract, architecture_contract_hash
from hierarchos.utils.checkpoint import load_full_model_with_config


def _config(*, revision, vocab_size=101, use_deepembed=True, use_rosa=True):
    return AttrDict(
        architecture_revision=revision,
        vocab_size=vocab_size,
        context_dim=8,
        h_hidden=8,
        l_hidden=8,
        persistent_dim=4,
        ltm_slots=8,
        ltm_key_dim=4,
        ltm_val_dim=4,
        ltm_topk=2,
        max_h_steps=3,
        min_h_steps=1,
        max_l_steps=2,
        h_stride=2,
        h_halt_thresh=0.9,
        l_conv_atol=1e-4,
        commitment_threshold=0.01,
        use_deepembed=use_deepembed,
        use_rosa=use_rosa,
        memory_token_routers=False,
        compile=False,
        gradient_checkpointing=False,
        inference_logit_parity=True,
        detach_every_n_steps=0,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        activation_clamp=100.0,
        z_loss_weight=0.0,
        rosa_max_context=16,
    )


def test_coherent_revision_builds_shared_factorized_paths():
    model = HierarchosCore(_config(revision="coherent-v9"))

    assert model.deepembed_mode == "shared-factorized"
    assert model.rosa_embedding_mode == "shared-factorized"
    assert hasattr(model, "h_deepembed_adapter")
    assert hasattr(model, "l_deepembed_adapter")
    assert hasattr(model, "rosa_adapter")
    assert not hasattr(model, "h_deepemb")
    assert not hasattr(model, "rosa_emb")
    assert model.config.core_recurrence_version == 2
    assert model.config.manager_compute_mode == "hard-masked"
    assert model.config.manager_state_commit_mode == "hard-selected"


def test_legacy_revision_keeps_checkpoint_tensor_layout():
    model = HierarchosCore(_config(revision="legacy-v8"))

    assert model.deepembed_mode == "legacy-table"
    assert model.rosa_embedding_mode == "legacy-table"
    assert hasattr(model, "h_deepemb")
    assert hasattr(model, "l_deepemb")
    assert hasattr(model, "rosa_emb")
    assert not hasattr(model, "h_deepembed_adapter")
    assert model.config.core_recurrence_version == 1


def test_shared_paths_are_neutral_at_initialization():
    torch.manual_seed(21)
    model = HierarchosCore(_config(revision="coherent-v9"))
    token_features = model.tok_emb(torch.tensor([1, 2, 3]))

    torch.testing.assert_close(
        model.h_deepembed_adapter(token_features),
        torch.ones(3, model.config.h_hidden * 4),
    )
    torch.testing.assert_close(
        model.l_deepembed_adapter(token_features),
        torch.ones(3, model.config.l_hidden * 4),
    )
    torch.testing.assert_close(
        model.rosa_adapter(token_features),
        torch.zeros(3, model.config.context_dim),
    )


def test_factorization_removes_vocabulary_scaled_auxiliary_parameters():
    legacy = HierarchosCore(_config(revision="legacy-v8", vocab_size=1000))
    coherent = HierarchosCore(_config(revision="coherent-v9", vocab_size=1000))

    legacy_count = sum(parameter.numel() for parameter in legacy.parameters())
    coherent_count = sum(parameter.numel() for parameter in coherent.parameters())

    assert coherent_count < legacy_count
    # At this tiny width the fixed core dominates, but the removed vocabulary
    # tables still save more than 50k parameters at vocab=1000.
    assert legacy_count - coherent_count > 50_000


def test_hard_manager_semantics_match_between_train_and_eager_inference():
    torch.manual_seed(22)
    model = HierarchosCore(
        _config(
            revision="coherent-v9",
            use_deepembed=False,
            use_rosa=False,
        )
    )
    with torch.no_grad():
        model.h_halt_proj.weight.zero_()
        model.h_halt_proj.bias.fill_(10.0)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)

    model.train()
    train_output = model(ids, attention_mask=mask)
    model.eval()
    with torch.no_grad():
        eval_output = model(ids, attention_mask=mask)

    torch.testing.assert_close(
        train_output["logits"],
        eval_output["logits"],
        rtol=1e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        train_output["h_state"],
        eval_output["h_state"],
        rtol=1e-5,
        atol=2e-6,
    )
    plan_steps = eval_output["step_telemetry"]["h_effective_steps"][0, ::2]
    assert plan_steps.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("training_step", [0, 10])
def test_memory_gate_schedule_has_train_eval_logit_parity(training_step):
    torch.manual_seed(221)
    config = _config(
        revision="coherent-v9",
        use_deepembed=False,
        use_rosa=False,
    )
    config.memory_gate_warmup_steps = 10
    config.memory_gate_warmup_floor = 0.5
    model = HierarchosCore(config)
    model.set_training_step(training_step)
    ids = torch.tensor([[1, 2, 3]])

    model.train()
    train_logits = model(ids)["logits"]
    model.eval()
    with torch.no_grad():
        eval_logits = model(ids)["logits"]

    assert model.memory_gate_warmup_step.item() == training_step
    torch.testing.assert_close(train_logits, eval_logits, rtol=1e-5, atol=2e-6)


def test_coherent_checkpoint_roundtrip_preserves_raw_logits(tmp_path):
    torch.manual_seed(23)
    model = HierarchosCore(
        _config(
            revision="coherent-v9",
            use_deepembed=True,
            use_rosa=False,
        )
    ).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model(ids)["logits"]
    checkpoint_path = tmp_path / "coherent.pt"
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    saved_config["architecture_contract_sha256"] = contract_hash
    torch.save(
        {
            "checkpoint_version": 4,
            "model_state_dict": model.state_dict(),
            "config": saved_config,
            "architecture_contract": contract,
            "architecture_contract_sha256": contract_hash,
            "training_complete": True,
        },
        checkpoint_path,
    )

    restored, restored_config = load_full_model_with_config(
        str(checkpoint_path),
        torch.device("cpu"),
    )
    with torch.no_grad():
        actual = restored(ids)["logits"]

    assert restored_config.architecture_revision == "coherent-v9"
    assert restored.deepembed_mode == "shared-factorized"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_loader_canonicalizes_public_detach_zero_sentinel(tmp_path):
    model = HierarchosCore(
        _config(revision="coherent-v9", use_deepembed=False, use_rosa=False)
    ).eval()
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    # Final user-facing configs may spell the disabled-detach sentinel as 0;
    # the executable architecture contract stores the canonical None value.
    saved_config["detach_every_n_steps"] = 0
    saved_config["architecture_contract_sha256"] = contract_hash
    checkpoint_path = tmp_path / "detach-zero.pt"
    torch.save(
        {
            "checkpoint_version": 4,
            "model_state_dict": model.state_dict(),
            "config": saved_config,
            "architecture_contract": contract,
            "architecture_contract_sha256": contract_hash,
            "training_complete": True,
        },
        checkpoint_path,
    )

    restored, restored_config = load_full_model_with_config(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert restored_config.detach_every_n_steps is None
    assert restored.config.detach_every_n_steps is None


def test_chat_ltm_reexport_preserves_coherent_checkpoint_identity(tmp_path):
    model = HierarchosCore(
        _config(revision="coherent-v9", use_deepembed=False, use_rosa=False)
    ).eval()
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    saved_config["architecture_contract_sha256"] = contract_hash
    run_identity = {
        "version": 1,
        "architecture_contract": contract,
        "architecture_contract_sha256": contract_hash,
        "sha256": "identity-fixture",
    }
    original_path = tmp_path / "original.pt"
    torch.save(
        {
            "checkpoint_version": 4,
            "checkpoint_kind": "inference",
            "model_state_dict": model.state_dict(),
            "config": saved_config,
            "architecture_contract": contract,
            "architecture_contract_sha256": contract_hash,
            "completed_epoch": 15,
            "run_identity": run_identity,
            "effective_training_config": {"training_chunk_size": 256},
            "optimizer_grouping_version": 2,
            "training_complete": True,
        },
        original_path,
    )

    loaded, _ = load_full_model_with_config(original_path, torch.device("cpu"))
    reexport = build_chat_ltm_checkpoint_payload(loaded)
    assert reexport["checkpoint_version"] == 4
    assert reexport["checkpoint_kind"] == "inference-ltm-consolidated"
    assert reexport["derived_from_checkpoint_kind"] == "inference"
    assert reexport["architecture_contract"] == contract
    assert reexport["architecture_contract_sha256"] == contract_hash
    assert reexport["run_identity"] == run_identity
    assert reexport["effective_training_config"] == {
        "training_chunk_size": 256
    }
    assert reexport["completed_epoch"] == 15

    reexport_path = tmp_path / "reexport.pt"
    torch.save(reexport, reexport_path)
    restored, restored_config = load_full_model_with_config(
        reexport_path,
        torch.device("cpu"),
    )
    assert restored_config.architecture_revision == "coherent-v9"
    assert restored.config.manager_compute_mode == "hard-masked"


def test_coherent_checkpoint_rejects_architecture_config_tampering(tmp_path):
    model = HierarchosCore(
        _config(revision="coherent-v9", use_deepembed=False, use_rosa=False)
    )
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    tampered_config = dict(model.config)
    tampered_config["max_h_steps"] = int(tampered_config["max_h_steps"]) + 1
    checkpoint_path = tmp_path / "tampered.pt"
    torch.save(
        {
            "checkpoint_version": 4,
            "model_state_dict": model.state_dict(),
            "config": tampered_config,
            "architecture_contract": contract,
            "architecture_contract_sha256": contract_hash,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="Architecture contract"):
        load_full_model_with_config(str(checkpoint_path), torch.device("cpu"))


def test_coherent_checkpoint_requires_persisted_gate_schedule_state(tmp_path):
    model = HierarchosCore(
        _config(revision="coherent-v9", use_deepembed=False, use_rosa=False)
    )
    state = dict(model.state_dict())
    state.pop("memory_gate_warmup_step")
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    saved_config = dict(model.config)
    saved_config["architecture_contract_sha256"] = contract_hash
    checkpoint_path = tmp_path / "missing-gate-step.pt"
    torch.save(
        {
            "checkpoint_version": 4,
            "model_state_dict": state,
            "config": saved_config,
            "architecture_contract": contract,
            "architecture_contract_sha256": contract_hash,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="memory_gate_warmup_step"):
        load_full_model_with_config(str(checkpoint_path), torch.device("cpu"))
