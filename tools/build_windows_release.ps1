param(
    [string]$Python = "py",
    [string[]]$PythonArgs = @(),
    [switch]$InstallDeps,
    [switch]$SkipBackend,
    [switch]$SkipRust
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GuiDir = Join-Path $Root "hierarchos-gui"
$VulkanDir = Join-Path $Root "hierarchos-vulkan"
$ReleaseDir = Join-Path $Root "dist\Hierarchos-Windows"
$BackendDist = Join-Path $Root "dist\hierarchos-backend"
$BackendTarget = Join-Path $ReleaseDir "backend"
$VulkanTarget = Join-Path $ReleaseDir "vulkan"
$CompatTarget = Join-Path $ReleaseDir "compat"

function Invoke-HierarchosPython {
    param([Parameter(Position = 0)][string[]]$Arguments)
    & $Python @PythonArgs @Arguments
    $script:LastPythonExitCode = $LASTEXITCODE
}

if ($InstallDeps) {
    Invoke-HierarchosPython @("-m", "pip", "install", "-r", (Join-Path $Root "tools\windows_backend_requirements.txt"))
    if ($script:LastPythonExitCode -ne 0) { throw "Python dependency installation failed." }

    $verifyTorch = "import torch; print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda); raise SystemExit(0 if torch.version.cuda else 2)"
    Invoke-HierarchosPython @("-c", $verifyTorch)
    if ($script:LastPythonExitCode -ne 0) {
        throw "Installed PyTorch is CPU-only. The Windows release must use the CUDA wheel so the same package supports CUDA and CPU fallback."
    }
}

if (-not $SkipBackend) {
    $BackendBuild = Join-Path $Root "build\hierarchos_backend"
    if (Test-Path $BackendDist) { Remove-Item -LiteralPath $BackendDist -Recurse -Force }
    if (Test-Path $BackendBuild) { Remove-Item -LiteralPath $BackendBuild -Recurse -Force }
    Invoke-HierarchosPython @("-m", "PyInstaller", "--clean", "--noconfirm", (Join-Path $Root "tools\hierarchos_backend.spec"))
    if ($script:LastPythonExitCode -ne 0) { throw "PyInstaller backend build failed." }
}

if (-not $SkipRust) {
    Push-Location $GuiDir
    try {
        cargo build --release --bin hierarchos-gui --bin hierarchos-native --bin hierarchos-native-cli
        if ($LASTEXITCODE -ne 0) { throw "Rust GUI build failed." }
    }
    finally {
        Pop-Location
    }

    cargo build --release --manifest-path (Join-Path $VulkanDir "Cargo.toml") --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices
    if ($LASTEXITCODE -ne 0) { throw "Rust Vulkan training runtime build failed." }
}

if (Test-Path $ReleaseDir) {
    Remove-Item -LiteralPath $ReleaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseDir | Out-Null

$GuiExe = Join-Path $GuiDir "target\release\hierarchos-gui.exe"
if (-not (Test-Path $GuiExe)) {
    throw "GUI executable not found: $GuiExe"
}
Copy-Item -LiteralPath $GuiExe -Destination (Join-Path $ReleaseDir "Hierarchos.exe")

$NativeGuiExe = Join-Path $GuiDir "target\release\hierarchos-native.exe"
if (-not (Test-Path $NativeGuiExe)) {
    throw "Native Rust GUI executable not found: $NativeGuiExe"
}
Copy-Item -LiteralPath $NativeGuiExe -Destination (Join-Path $ReleaseDir "HierarchosNativeFP32.exe")

$NativeCliExe = Join-Path $GuiDir "target\release\hierarchos-native-cli.exe"
if (-not (Test-Path $NativeCliExe)) {
    throw "Native Rust CLI executable not found: $NativeCliExe"
}
Copy-Item -LiteralPath $NativeCliExe -Destination (Join-Path $ReleaseDir "HierarchosCLI.exe")

$VulkanTrainerExe = Join-Path $VulkanDir "target\release\hierarchos-vulkan-train.exe"
$VulkanDevicesExe = Join-Path $VulkanDir "target\release\hierarchos-vulkan-devices.exe"
if ((Test-Path $VulkanTrainerExe) -and (Test-Path $VulkanDevicesExe)) {
    New-Item -ItemType Directory -Path $VulkanTarget | Out-Null
    Copy-Item -LiteralPath $VulkanTrainerExe -Destination $VulkanTarget
    Copy-Item -LiteralPath $VulkanDevicesExe -Destination $VulkanTarget
}
elseif (-not $SkipRust) {
    throw "Vulkan training runtime binaries were not found after the Rust build."
}

if (Test-Path $BackendDist) {
    New-Item -ItemType Directory -Path $BackendTarget | Out-Null
    Get-ChildItem -LiteralPath $BackendDist | Copy-Item -Destination $BackendTarget -Recurse -Force
}
elseif (-not $SkipBackend) {
    throw "Backend dist not found: $BackendDist"
}

# Keep a source-Python fallback for development/custom environments. Packaged
# releases normally execute compatibility-only CLI modes through the bundled
# backend, so end users do not need a separate Python installation.
New-Item -ItemType Directory -Path $CompatTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "hierarchos_cli.py") -Destination $CompatTarget
Copy-Item -LiteralPath (Join-Path $Root "hierarchos") -Destination (Join-Path $CompatTarget "hierarchos") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Root "requirements_kernel.txt") -Destination (Join-Path $CompatTarget "requirements_kernel.txt")

foreach ($name in @("LICENSE.md", "README.md")) {
    $path = Join-Path $Root $name
    if (Test-Path $path) {
        Copy-Item -LiteralPath $path -Destination $ReleaseDir
    }
}

$readme = @"
Hierarchos Windows Release

Run Hierarchos.exe.

HierarchosNativeFP32.exe is the Python-free frontend. It runs the native Rust
FP32 inference engine and can launch the bundled Vulkan trainer directly from
its Vulkan Training tab, including live loss/LR/throughput events, multi-adapter
selection, periodic checkpoints, persisted recurrent state, and exact resume.

HierarchosCLI.exe is the native command-line frontend. Native `train`, `chat`,
`finetune`, `benchmark` (local throughput), `devices`, `pull`, SafeTensors
`merge-lora`, and SafeTensors `ckpt-2-inf` paths do not require PyTorch. `train`
accepts raw local JSONL, tokenized JSONL, or the same schema-v6 token cache as
the reference loader and launches the bundled all-Vulkan trainer. From-scratch
model bootstrap from tokenizer assets and supported Hugging Face model/tokenizer
or JSONL/NDJSON dataset downloads are also implemented in Rust. Framework-only
features such as external lm-eval/ARC catalogs, arbitrary Python dataset-builder
execution, legacy framework-object `.pt` loading, arbitrary new PEFT geometry
injection, and train-time framework eval hooks remain compatibility-stack
features and are not silently invoked by the native commands.

Those compatibility-only commands normally run through
backend\hierarchos-backend.exe, which bundles the canonical Python CLI together
with PEFT, Hugging Face datasets, Accelerate, and lm-eval. No system Python is
required for the packaged compatibility surface. For development overrides,
HIERARCHOS_COMPAT_BACKEND selects another backend; setting HIERARCHOS_PYTHON
forces the source-Python fallback in compat\hierarchos_cli.py (or the path named
by HIERARCHOS_CLI/HIERARCHOS_ROOT). Native train/chat/devices/local-benchmark
paths do not use either compatibility route.

The GUI first looks for backend\hierarchos-backend.exe. That backend bundles
the Hierarchos Python package plus the PyTorch/Transformers runtime, so users
do not need to clone this repository or install Python for normal inference.

The backend is built with the CUDA-enabled PyTorch wheel. That single runtime
also includes CPU execution: Auto uses NVIDIA CUDA when PyTorch can see a CUDA
GPU, and otherwise falls back to CPU for non-NVIDIA systems and handheld PCs.
Selecting CUDA explicitly will report a clear error if the NVIDIA driver/GPU is
not available instead of silently pretending to run on GPU.

The Training panel also includes a native Vulkan backend. It launches the
bundled vulkan\hierarchos-vulkan-train.exe directly, with no PyTorch or CUDA
runtime in the training process. Native Vulkan training consumes a local
Hierarchos SafeTensors package plus tokenized JSONL and writes the same
SafeTensors parameter ABI used by PyTorch CPU/CUDA and native Rust inference.
Comma-separated Vulkan adapter indices (for example 0,1) enable the trainer's
synchronous multi-device data-parallel path.

Vulkan checkpoint packages preserve ordinary model-package sidecars such as
tokenizer.json. The native frontend can therefore load the final trained output
directly for Rust inference, while PyTorch can consume the same FP32-master
model.safetensors on CPU or NVIDIA CUDA.

Model sources accepted by the GUI:
- A Hugging Face repo id, for example author/model-name
- A local model directory containing hierarchos.pt or model.pt plus tokenizer files
- A direct .pt inference checkpoint with config embedded or a neighboring
  hierarchos_config.json

Tokenizer path selection is not required. Hierarchos loads the tokenizer from
the model directory automatically.

When closing with a model loaded, the app asks whether to save runtime LTM
updates. Saving writes hierarchos_ltm_updates.pt next to the loaded model and
reloads that sidecar automatically on future loads. Discard closes without
writing new LTM updates.

Downloaded Hugging Face models are cached under the user's local Hierarchos
app data directory. If the bundled backend is missing, Settings can fall back
to a system Python by changing the backend field from bundled to python.
"@
Set-Content -LiteralPath (Join-Path $ReleaseDir "README_RELEASE.txt") -Value $readme -Encoding UTF8

Write-Host "Release bundle created:"
Write-Host "  $ReleaseDir"
