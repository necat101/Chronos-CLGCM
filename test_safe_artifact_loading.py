import functools
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch

import hierarchos_cli
import hierarchos.inference.chat as chat_module
import hierarchos.training.datasets as datasets_module
from hierarchos.training.datasets import TokenizedBinaryDataset
from hierarchos.utils import checkpoint as checkpoint_utils
from hierarchos.utils.checkpoint import (
    load_checkpoint_payload_compatible,
    save_checkpoint_safely,
)
from hierarchos.utils import safe_loading


ROOT = Path(__file__).resolve().parent


def _install_legacy_loader(monkeypatch):
    calls = []

    def legacy_torch_load(path, map_location=None):
        calls.append((path, map_location))
        return {"unsafe": True}

    monkeypatch.setattr(safe_loading.torch, "load", legacy_torch_load)
    return calls


def test_safe_loader_rejects_legacy_torch_before_opening_artifact(monkeypatch):
    calls = _install_legacy_loader(monkeypatch)

    with pytest.raises(RuntimeError, match=r"PyTorch >= 1\.13"):
        safe_loading.load_tensor_payload_safely("must-not-be-opened.pt")

    assert calls == []


def test_safe_loader_never_retries_after_weights_only_rejection(monkeypatch):
    calls = []

    def rejecting_torch_load(path, map_location=None, *, weights_only=None):
        calls.append(weights_only)
        if weights_only is True:
            raise TypeError("simulated vendor runtime without restricted loading")
        return {"unsafe": True}

    monkeypatch.setattr(safe_loading.torch, "load", rejecting_torch_load)

    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        safe_loading.load_tensor_payload_safely("must-not-retry.pt")

    assert calls == [True]


def test_checkpoint_loader_requires_scoped_safe_globals_before_torch_load(
    monkeypatch,
):
    real_load = safe_loading.torch.load
    calls = []

    @functools.wraps(real_load)
    def forbidden_load(*args, **kwargs):
        calls.append(dict(kwargs))
        raise AssertionError("torch.load must not run without safe_globals")

    monkeypatch.setattr(safe_loading.torch, "load", forbidden_load)
    monkeypatch.delattr(
        safe_loading.torch.serialization,
        "safe_globals",
        raising=False,
    )

    with pytest.raises(RuntimeError, match=r"PyTorch >= 2\.4"):
        load_checkpoint_payload_compatible("must-not-be-opened.pt")

    assert calls == []


def test_checkpoint_checksum_load_uses_the_verified_open_file(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([7])}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (tmp_path / "checkpoint.pt.sha256").write_text(digest, encoding="utf-8")
    real_safe_load = checkpoint_utils.load_tensor_payload_safely
    observed_sources = []

    def tracking_safe_load(source, **kwargs):
        observed_sources.append(source)
        assert not isinstance(source, (str, bytes, Path))
        return real_safe_load(source, **kwargs)

    monkeypatch.setattr(
        checkpoint_utils,
        "load_tensor_payload_safely",
        tracking_safe_load,
    )
    loaded = load_checkpoint_payload_compatible(str(checkpoint_path))

    assert torch.equal(loaded["value"], torch.tensor([7]))
    assert len(observed_sources) == 1
    assert observed_sources[0].closed


@pytest.mark.parametrize("sidecar", ["", "not-a-digest"])
def test_checkpoint_loader_rejects_malformed_checksum_sidecar(tmp_path, sidecar):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"value": torch.tensor([7])}, checkpoint_path)
    (tmp_path / "checkpoint.pt.sha256").write_text(sidecar, encoding="utf-8")

    with pytest.raises(RuntimeError, match="sidecar is (empty|malformed)"):
        load_checkpoint_payload_compatible(str(checkpoint_path))


def test_atomic_checkpoint_save_restores_previous_pair_if_checksum_publish_fails(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint_safely(
        {"generation": torch.tensor([1])},
        str(checkpoint_path),
    )
    real_replace = checkpoint_utils.os.replace

    def fail_new_checksum_publish(source, destination):
        if str(source).endswith(".tmp.sha256"):
            raise OSError("simulated checksum publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_utils.os, "replace", fail_new_checksum_publish)
    with pytest.raises(OSError, match="checksum publication failure"):
        save_checkpoint_safely(
            {"generation": torch.tensor([2])},
            str(checkpoint_path),
        )

    restored = load_checkpoint_payload_compatible(str(checkpoint_path))
    assert torch.equal(restored["generation"], torch.tensor([1]))
    assert checkpoint_path.exists()
    assert (tmp_path / "checkpoint.pt.sha256").exists()


def test_dataset_and_cli_artifact_ingress_fail_closed_on_legacy_torch(
    tmp_path,
    monkeypatch,
    capsys,
):
    index_path = tmp_path / "index.pt"
    data_path = tmp_path / "tokens.bin"
    success_path = tmp_path / "_SUCCESS"
    index_path.write_bytes(b"not read")
    data_path.write_bytes(b"")
    success_path.write_text(json.dumps({}), encoding="utf-8")
    calls = _install_legacy_loader(monkeypatch)

    with pytest.raises(RuntimeError, match=r"PyTorch >= 1\.13"):
        datasets_module._load_tensor_artifact_weights_only(str(index_path))
    with pytest.raises(RuntimeError, match=r"PyTorch >= 1\.13"):
        TokenizedBinaryDataset(str(tmp_path))
    with pytest.raises(RuntimeError, match=r"PyTorch >= 1\.13"):
        hierarchos_cli._read_token_cache_identity(str(tmp_path))

    args = SimpleNamespace()
    assert (
        hierarchos_cli.auto_tune_length_bucket_size_from_token_cache(
            args,
            str(tmp_path),
        )
        is None
    )
    assert "Refusing to fall back" in capsys.readouterr().out
    assert calls == []


def test_chat_ltm_overlay_ingress_uses_fail_closed_loader(
    monkeypatch,
):
    calls = _install_legacy_loader(monkeypatch)
    model = SimpleNamespace(ltm=object())

    with pytest.raises(RuntimeError, match=r"PyTorch >= 1\.13"):
        chat_module.load_ltm_delta_overlay(model, "must-not-be-opened.pt")

    assert calls == []


def test_supported_executables_have_one_restricted_torch_load_boundary():
    excluded = {
        ROOT / "hierarchos.py",  # guarded unsupported historical monolith
        ROOT / "hierarchos_stable_snippet.py",  # non-executable historical fragment
        ROOT / "hierarchos" / "utils" / "safe_loading.py",  # central boundary
    }
    candidates = list((ROOT / "hierarchos").rglob("*.py"))
    candidates.extend(
        path
        for path in ROOT.glob("*.py")
        if not path.name.startswith("test_")
    )
    candidates.extend((ROOT / "tools").glob("*.py"))

    direct_loaders = []
    remote_code_defaults = []
    for path in candidates:
        if path in excluded:
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"\btorch\.load\s*\(", source):
            direct_loaders.append(str(path.relative_to(ROOT)))
        if re.search(r"\btrust_remote_code\s*=\s*True\b", source):
            remote_code_defaults.append(str(path.relative_to(ROOT)))

    assert direct_loaders == []
    assert remote_code_defaults == []
