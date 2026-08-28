# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parent

hiddenimports = collect_submodules("hierarchos")
hiddenimports += [
    "hierarchos_cli",
    "huggingface_hub",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "torch._C",
    "torch.backends.cuda",
    "torch.cuda",
    "torch.nn",
    "torch.nn.functional",
    "tqdm",
    "transformers",
    "transformers.models.auto.tokenization_auto",
]

# HierarchosCLI.exe owns the public CLI surface, but Python-only compatibility
# modes execute inside this bundled runtime.  Collect their lazily-imported
# packages explicitly so PEFT LoRA, HF datasets, and lm-eval remain available
# without requiring a separately installed Python environment.
for package in ["datasets", "peft", "accelerate", "lm_eval"]:
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

datas = [
    (str(ROOT / "hierarchos"), "hierarchos"),
]

binaries = []

try:
    binaries += collect_dynamic_libs("torch")
except Exception:
    pass

try:
    datas += collect_data_files("transformers", include_py_files=False)
except Exception:
    pass

for package in [
    "torch",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "numpy",
    "tqdm",
    "datasets",
    "peft",
    "accelerate",
    "lm_eval",
    "pyarrow",
    "pandas",
    "sklearn",
    "scipy",
]:
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

for package in ["datasets", "peft", "accelerate", "lm_eval"]:
    try:
        datas += collect_data_files(package, include_py_files=False)
    except Exception:
        pass

excludes = [
    "bitsandbytes",
    "IPython",
    "lxml",
    "mcp",
    "nltk",
    "optuna",
    "PIL",
    "PIL.ImageQt",
    "redis",
    "soundfile",
    "sqlalchemy",
    "tiktoken",
    "tkinter",
    "torchaudio",
    "torchtext",
    "torchvision",
    "boto3",
    "botocore",
    "cv2",
    "django",
    "fastapi",
    "jupyter",
    "matplotlib",
    "notebook",
    "openai",
    "pygame",
    "pytest",
    "tensorflow",
    "uvicorn",
]

a = Analysis(
    [str(ROOT / "hierarchos_bridge_server.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hierarchos-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="hierarchos-backend",
)
