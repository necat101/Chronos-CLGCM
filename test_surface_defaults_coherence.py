import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import hierarchos_cli
from hierarchos.inference.chat import sample_next_token
from hierarchos.models.revisions import (
    apply_architecture_revision_defaults,
    architecture_default_commitment_threshold,
)
from hierarchos.training.trainer import _resolved_training_chunk_size


ROOT = Path(__file__).resolve().parent


def test_revision_cli_gui_and_trainer_share_chunk_geometry_defaults():
    coherent = {"architecture_revision": "coherent-v9"}
    legacy = {"architecture_revision": "legacy-v8"}
    apply_architecture_revision_defaults(coherent)
    apply_architecture_revision_defaults(legacy)

    assert coherent["training_chunk_size"] == 256
    assert coherent["reference_chunk_len"] == 256
    assert legacy["training_chunk_size"] == 128
    assert legacy["reference_chunk_len"] == 128
    assert hierarchos_cli.DEFAULT_TRAINING_CHUNK_SIZE == 256
    assert _resolved_training_chunk_size(
        SimpleNamespace(architecture_revision="coherent-v9")
    ) == 256
    assert _resolved_training_chunk_size(
        SimpleNamespace(architecture_revision="legacy-v8")
    ) == 128

    rust_bridge = (ROOT / "hierarchos-gui" / "src" / "bridge.rs").read_text(
        encoding="utf-8"
    )
    assert "fn default_training_chunk_size() -> u32 {\n    256\n}" in rust_bridge
    assert "training_chunk_size: default_training_chunk_size()" in rust_bridge

    quantized_source = (ROOT / "hierarchos" / "models" / "quantized.py").read_text(
        encoding="utf-8"
    )
    assert "architecture_default_training_chunk_size(self.config)" in quantized_source
    assert "getattr(self.config, 'training_chunk_size', 128)" not in quantized_source

    salvage_source = (ROOT / "tools" / "salvage_response_finetune.py").read_text(
        encoding="utf-8"
    )
    assert "architecture_default_training_chunk_size(config)" in salvage_source


def test_coherent_commitment_default_is_width_calibrated_across_surfaces():
    coherent = {
        "architecture_revision": "coherent-v9",
        "context_dim": 448,
    }
    apply_architecture_revision_defaults(coherent)

    expected = 0.1 / 448
    assert coherent["commitment_threshold"] == expected
    assert architecture_default_commitment_threshold(coherent) == expected

    cli_source = (ROOT / "hierarchos_cli.py").read_text(encoding="utf-8")
    bridge_source = (ROOT / "hierarchos_bridge_server.py").read_text(
        encoding="utf-8"
    )
    assert '"--commitment-threshold",\n        type=float,\n        default=None' in cli_source
    assert "architecture_default_commitment_threshold(_config)" in bridge_source


def test_all_user_facing_sampling_surfaces_default_to_temperature_point_seven():
    assert hierarchos_cli.DEFAULT_CHAT_TEMPERATURE == 0.7
    assert inspect.signature(sample_next_token).parameters["temperature"].default == 0.7

    bridge_source = (ROOT / "hierarchos_bridge_server.py").read_text(
        encoding="utf-8"
    )
    rust_bridge = (ROOT / "hierarchos-gui" / "src" / "bridge.rs").read_text(
        encoding="utf-8"
    )
    assert 'sampling.get("temperature", 0.7)' in bridge_source
    assert "temperature: 0.7" in rust_bridge


def test_removed_quantized_online_learning_flags_fail_before_model_loading():
    result = subprocess.run(
        [
            sys.executable,
            "hierarchos_cli.py",
            "chat",
            "--enable-quantized-learning",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "quantized online learning is unsupported" in result.stderr
