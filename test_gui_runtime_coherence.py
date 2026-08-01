import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def test_embedded_gui_runtime_contains_every_hierarchos_module():
    source = (ROOT / "hierarchos-gui" / "src" / "embedded.rs").read_text(
        encoding="utf-8"
    )
    embedded = set(
        re.findall(r'include_str!\("\.\./\.\./(hierarchos/[^"\n]+\.py)"\)', source)
    )
    package = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "hierarchos").rglob("*.py")
    }
    assert embedded == package


def test_embedded_runtime_is_complete_and_fresh_before_activation():
    source = (ROOT / "hierarchos-gui" / "src" / "embedded.rs").read_text(
        encoding="utf-8"
    )
    assert "extraction_complete(&base_dir)" in source
    assert "python.staging." in source
    assert "fs::rename(&staging_dir, &base_dir)" in source
    assert "python.previous." in source


def test_gui_bridge_tracks_training_terminal_events_and_auto_fallback():
    source = (ROOT / "hierarchos-gui" / "src" / "bridge.rs").read_text(
        encoding="utf-8"
    )
    assert '"training_complete" =>' in source
    assert "training.store(false, Ordering::SeqCst)" in source
    assert 'requested.eq_ignore_ascii_case("auto")' in source
    assert '"full_sample_checkpoint_segment_size"' in source

    training_panel = (ROOT / "hierarchos-gui" / "src" / "panels" / "training.rs").read_text(
        encoding="utf-8"
    )
    assert "state.config.full_sample_activation_checkpointing = true;" in training_panel


def test_gui_passive_online_learning_is_opt_in():
    settings = (
        ROOT / "hierarchos-gui" / "src" / "panels" / "settings.rs"
    ).read_text(encoding="utf-8")
    assert "passive_learning: false," in settings
    assert "Generated responses are never self-trained." in settings
    assert "Surprise Threshold" not in settings

    bridge = (ROOT / "hierarchos-gui" / "src" / "bridge.rs").read_text(
        encoding="utf-8"
    )
    assert '"passive_learning": passive_learning' in bridge
    assert '"passive_lr": passive_lr' in bridge
    assert '"learning_rate": learning_rate' in bridge

    chat = (ROOT / "hierarchos-gui" / "src" / "panels" / "chat.rs").read_text(
        encoding="utf-8"
    )
    assert "settings.passive_learning" in chat
    assert "settings.passive_lr" in chat
    assert "settings.ltm_lr" in chat
