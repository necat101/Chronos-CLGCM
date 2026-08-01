from pathlib import Path

from setuptools import find_namespace_packages


ROOT = Path(__file__).resolve().parent


def test_setup_discovers_the_complete_modular_runtime():
    expected_packages = {
        path.parent.relative_to(ROOT).as_posix().replace("/", ".")
        for path in (ROOT / "hierarchos").rglob("*.py")
    }
    discovered = set(
        find_namespace_packages(
            where=str(ROOT),
            include=["hierarchos", "hierarchos.*"],
        )
    )
    assert expected_packages <= discovered

    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "packages=find_namespace_packages" in setup_source
    assert 'include=["hierarchos", "hierarchos.*"]' in setup_source
    assert "str(os.cpu_count() or 1)" in setup_source
