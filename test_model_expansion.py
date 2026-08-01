import argparse
import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from expand_model import (
    ARCH_UPDATE_KEYS,
    _copy_named_tensor_,
    _infer_missing_config,
    _load_source_artifact,
    _publish_directory_atomically,
    _resolve_output_paths,
    _sha256_json,
    _validate_output_location,
    build_expanded_config,
    transplant_weights,
)
from hierarchos import AttrDict, HierarchosCore
from hierarchos.models.revisions import architecture_contract, architecture_contract_hash
from hierarchos.utils.checkpoint import (
    _validate_expansion_provenance,
    load_full_model_with_config,
    sanitize_model_state_dict,
    save_checkpoint_safely,
)
from hierarchos.utils.tokenizer import (
    tokenizer_identity,
    validate_inference_tokenizer_identity,
)


def _base_config(vocab_size: int) -> dict:
    config = {
        "architecture_revision": "coherent-v9",
        "vocab_size": int(vocab_size),
        "context_dim": 8,
        "persistent_dim": 4,
        "ltm_slots": 8,
        "ltm_key_dim": 4,
        "ltm_val_dim": 4,
        "ltm_topk": 2,
        "h_hidden": 8,
        "l_hidden": 8,
        "h_stride": 2,
        "max_h_steps": 2,
        "max_l_steps": 2,
        "rwkv_head_size": 4,
        "max_length": 16,
        "detach_every_n_steps": None,
        "compile": False,
    }
    architecture_contract(config)
    config["architecture_contract_sha256"] = architecture_contract_hash(config)
    return config


def _expansion_args(source: Path) -> argparse.Namespace:
    values = {key: None for key in ARCH_UPDATE_KEYS}
    values.update(
        {
            "old_model_path": str(source),
            "context_dim": 12,
            "persistent_dim": 6,
            "ltm_slots": 10,
            "ltm_key_dim": 6,
            "ltm_val_dim": 6,
            "ltm_topk": 3,
            "rwkv_head_size": 4,
            "auto_max_length": False,
            "dataset_for_length": None,
            "new_max_length": None,
            "kayla": False,
            "alpaca": False,
            "trust_remote_code": False,
        }
    )
    return argparse.Namespace(**values)


def test_segment_aware_projection_copy_preserves_feature_boundaries():
    source_config = {
        "context_dim": 2,
        "persistent_dim": 1,
        "ltm_topk": 2,
        "ltm_val_dim": 2,
    }
    target_config = {
        "context_dim": 3,
        "persistent_dim": 2,
        "ltm_topk": 3,
        "ltm_val_dim": 3,
    }

    q_source = torch.tensor([[10.0, 11.0, 20.0, 21.0]])
    q_target = torch.full((1, 6), -1.0)
    result = _copy_named_tensor_(
        "qproj.weight",
        q_target,
        q_source,
        source_config,
        target_config,
    )
    assert result[0] == "resized"
    assert q_target.tolist() == [[10.0, 11.0, -1.0, 20.0, 21.0, -1.0]]

    in_source = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    in_target = torch.full((1, 14), -1.0)
    result = _copy_named_tensor_(
        "in_proj.weight",
        in_target,
        in_source,
        source_config,
        target_config,
    )
    assert result[0] == "resized"
    assert in_target.tolist() == [[
        0.0,
        1.0,
        -1.0,
        2.0,
        -1.0,
        3.0,
        4.0,
        -1.0,
        5.0,
        6.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]]


def test_infer_missing_config_preserves_explicit_detach_none():
    state_dict = {
        "tok_emb.weight": torch.zeros(4, 2),
        "lm_head.weight": torch.zeros(4, 2),
    }
    inferred = _infer_missing_config(
        {"detach_every_n_steps": None},
        state_dict,
    )
    assert inferred["detach_every_n_steps"] is None


def test_atomic_directory_publication_requires_explicit_overwrite(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _publish_directory_atomically(staging, output, overwrite=False)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert staging.exists()

    _publish_directory_atomically(staging, output, overwrite=True)
    assert not (output / "old.txt").exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()


def test_legacy_file_output_never_resolves_to_its_parent_directory(tmp_path):
    output_dir, weights_path = _resolve_output_paths(
        str(tmp_path / "hierarchos.pt")
    )
    assert output_dir == tmp_path / "hierarchos"
    assert weights_path == output_dir / "hierarchos.pt"

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep", encoding="utf-8")
    source_checkpoint = tmp_path / "source.pt"
    source_checkpoint.write_bytes(b"source")
    source_artifact = {
        "checkpoint_path": source_checkpoint,
        "model_root": tmp_path,
        "source_is_directory": False,
    }
    with pytest.raises(ValueError, match="non-empty directory"):
        _validate_output_location(
            source_artifact,
            unrelated,
            overwrite=True,
        )


def test_expansion_provenance_rejects_tampering():
    provenance = {
        "version": 1,
        "mapping_version": "segment-aware-v1",
        "source": {"checkpoint_sha256": "a" * 64},
        "expanded": {
            "architecture_contract_sha256": "b" * 64,
            "tokenizer_identity": {"vocab_size": 4, "sha256": "c" * 64},
        },
    }
    provenance["sha256"] = _sha256_json(provenance)
    checkpoint = {
        "architecture_contract_sha256": "b" * 64,
        "tokenizer_identity": copy.deepcopy(
            provenance["expanded"]["tokenizer_identity"]
        ),
        "expansion_provenance": provenance,
    }
    assert _validate_expansion_provenance(checkpoint, "test")

    tampered = copy.deepcopy(checkpoint)
    tampered["expansion_provenance"]["source"]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="provenance SHA-256"):
        _validate_expansion_provenance(tampered, "test")


def test_expansion_publishes_reloadable_authenticated_package(tmp_path):
    tokenizer_source = Path(__file__).resolve().parent / "cuda_test_model"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        local_files_only=True,
    )
    config = _base_config(len(tokenizer))
    model = HierarchosCore(AttrDict(copy.deepcopy(config)))
    contract = architecture_contract(model.config)
    contract_hash = architecture_contract_hash(model.config)
    config = dict(model.config)
    config["architecture_contract_sha256"] = contract_hash

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    tokenizer.save_pretrained(source_dir)
    identity = tokenizer_identity(tokenizer)
    source_checkpoint = {
        "checkpoint_version": 4,
        "checkpoint_kind": "inference",
        "model_state_dict": sanitize_model_state_dict(model),
        "config": config,
        "architecture_contract": contract,
        "architecture_contract_sha256": contract_hash,
        "tokenizer_identity": identity,
        "training_complete": True,
    }
    save_checkpoint_safely(
        source_checkpoint,
        str(source_dir / "hierarchos.pt"),
    )

    source_artifact = _load_source_artifact(str(source_dir), "cpu")
    args = _expansion_args(source_dir)
    expanded_config = build_expanded_config(
        args,
        "cpu",
        source_artifact=source_artifact,
    )
    assert expanded_config["detach_every_n_steps"] is None

    output_dir = tmp_path / "expanded"
    transplant_weights(
        str(source_dir),
        expanded_config,
        str(output_dir),
        "cpu",
        source_artifact=source_artifact,
    )

    expected_files = {
        "hierarchos.pt",
        "hierarchos.pt.sha256",
        "hierarchos_config.json",
        "expansion_provenance.json",
        "tokenizer.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    reloaded_model, reloaded_config = load_full_model_with_config(
        str(output_dir),
        torch.device("cpu"),
    )
    reloaded_tokenizer = AutoTokenizer.from_pretrained(
        str(output_dir),
        local_files_only=True,
    )
    metadata = reloaded_model._hierarchos_checkpoint_metadata
    assert metadata["checkpoint_kind"] == "inference-expanded"
    assert validate_inference_tokenizer_identity(reloaded_tokenizer, metadata)
    assert reloaded_config.context_dim == 12
    assert reloaded_config.persistent_dim == 6
    assert reloaded_config.ltm_slots == 10
    assert reloaded_config.ltm_topk == 3

    provenance = json.loads(
        (output_dir / "expansion_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance == metadata["expansion_provenance"]
    assert provenance["mapping_version"] == "segment-aware-v1"

    input_ids = torch.tensor(
        [[reloaded_tokenizer.eos_token_id, reloaded_tokenizer.eos_token_id]],
        dtype=torch.long,
    )
    with torch.no_grad():
        output = reloaded_model(
            input_ids,
            return_last_logit_only=True,
            suppress_hebbian_update=True,
        )
    assert output["logits"].shape == (1, 1, len(reloaded_tokenizer))
    assert torch.isfinite(output["logits"]).all()
