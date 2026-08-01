from pathlib import Path
import os
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dist",
    "cuda_test_model",
    "rog_ally_model",
    "rog_ally_model_v8_preflight",
    "rog_ally_model_test",
    "target",
}

SKIP_FILES = {
    "hierarchos.py",
    "hierarchos_old.py.bak",
    "log.txt",
    "log_verify.txt",
    "log_verify_2.txt",
}

TEXT_SUFFIXES = {
    ".bat",
    ".cmd",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".txt",
}

LEGACY_COMMAND_PATTERNS = [
    re.compile(r"\bpython(?:3)?\s+hierarchos[.]py\b"),
    re.compile(r"\bpy(?:\s+-\d+(?:[.]\d+)?)?\s+hierarchos[.]py\b"),
    re.compile(r"%PYTHON_EXE%\"\s+hierarchos[.]py\b"),
]


def _candidate_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def test_docs_and_launchers_do_not_call_legacy_monolith():
    offenders = []
    for path in _candidate_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LEGACY_COMMAND_PATTERNS:
            match = pattern.search(text)
            if match:
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")

    assert not offenders, (
        "Command-style references to the legacy monolith must use "
        "hierarchos_cli.py instead:\n" + "\n".join(offenders)
    )


def test_core_package_import_does_not_load_optional_benchmark_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import hierarchos; "
                "assert 'lm_eval' not in sys.modules; "
                "assert 'soundfile' not in sys.modules; "
                "assert 'transformers.pipelines' not in sys.modules; "
                "assert 'numba' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "ROSA_USE_NUMBA": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
