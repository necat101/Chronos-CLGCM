param(
    [switch]$SkipTests,
    [switch]$SkipDeviceProbe
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $Root "dist"
$ReleaseDir = Join-Path $DistRoot "Hierarchos-Native"
$StagingDir = Join-Path $DistRoot (".Hierarchos-Native.staging-{0}" -f $PID)

$InferenceManifest = Join-Path $Root "hierarchos-inference\Cargo.toml"
$VulkanManifest = Join-Path $Root "hierarchos-vulkan\Cargo.toml"
$CliManifest = Join-Path $Root "hierarchos-native-cli\Cargo.toml"
$GuiManifest = Join-Path $Root "hierarchos-gui\Cargo.toml"

function Invoke-Cargo {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    Write-Host ("cargo " + ($Arguments -join " "))
    & cargo @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Cargo command failed with exit code $LASTEXITCODE."
    }
}

function Assert-NativeDependencyTree {
    param(
        [Parameter(Mandatory = $true)][string]$Manifest,
        [Parameter(Mandatory = $true)][string]$Label
    )

    Write-Host "Auditing $Label dependency tree for framework bindings..."
    # Windows PowerShell 5.1 can promote native stderr records to terminating
    # PowerShell errors when ErrorActionPreference is Stop. `cargo tree` writes
    # normal registry/download progress to stderr, so capture both streams while
    # deciding success exclusively from Cargo's process exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $tree = (& cargo tree --manifest-path $Manifest --locked 2>&1 | Out-String)
        $cargoTreeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($cargoTreeExitCode -ne 0) {
        throw "Could not inspect the $Label dependency tree."
    }
    if ($tree -match '(?im)^.*\b(pyo3|pyo3-ffi|tch|torch-sys|libtorch)\b.*$') {
        throw "Native dependency audit rejected $Label because a Python/libtorch binding was found:`n$($Matches[0])"
    }
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "cargo was not found on PATH. Install a Rust toolchain before building Hierarchos Native."
}

New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null
if (Test-Path -LiteralPath $StagingDir) {
    Remove-Item -LiteralPath $StagingDir -Recurse -Force
}
New-Item -ItemType Directory -Path $StagingDir | Out-Null

try {
    Assert-NativeDependencyTree -Manifest $InferenceManifest -Label "hierarchos-inference"
    Assert-NativeDependencyTree -Manifest $VulkanManifest -Label "hierarchos-vulkan"
    Assert-NativeDependencyTree -Manifest $CliManifest -Label "hierarchos-native-cli"
    Assert-NativeDependencyTree -Manifest $GuiManifest -Label "hierarchos-native GUI"

    if (-not $SkipTests) {
        Invoke-Cargo @("test", "--manifest-path", $InferenceManifest, "--locked")
        Invoke-Cargo @("test", "--manifest-path", $VulkanManifest, "--lib", "--locked")
        Invoke-Cargo @("test", "--manifest-path", $CliManifest, "--locked")
        Invoke-Cargo @("test", "--manifest-path", $GuiManifest, "--bin", "hierarchos-native", "--locked")
    }

    Invoke-Cargo @(
        "build", "--release", "--manifest-path", $VulkanManifest,
        "--bin", "hierarchos-vulkan-train",
        "--bin", "hierarchos-vulkan-devices",
        "--locked"
    )
    Invoke-Cargo @("build", "--release", "--manifest-path", $CliManifest, "--locked")
    Invoke-Cargo @("build", "--release", "--manifest-path", $GuiManifest, "--bin", "hierarchos-native", "--locked")

    $NativeGuiExe = Join-Path $Root "hierarchos-gui\target\release\hierarchos-native.exe"
    $NativeCliExe = Join-Path $Root "hierarchos-native-cli\target\release\hierarchos-native-cli.exe"
    $VulkanTrainerExe = Join-Path $Root "hierarchos-vulkan\target\release\hierarchos-vulkan-train.exe"
    $VulkanDevicesExe = Join-Path $Root "hierarchos-vulkan\target\release\hierarchos-vulkan-devices.exe"

    foreach ($required in @($NativeGuiExe, $NativeCliExe, $VulkanTrainerExe, $VulkanDevicesExe)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "Expected native release binary was not produced: $required"
        }
    }

    $VulkanTarget = Join-Path $StagingDir "vulkan"
    New-Item -ItemType Directory -Path $VulkanTarget | Out-Null
    Copy-Item -LiteralPath $NativeGuiExe -Destination (Join-Path $StagingDir "HierarchosNative.exe")
    Copy-Item -LiteralPath $NativeCliExe -Destination (Join-Path $StagingDir "HierarchosCLI.exe")
    Copy-Item -LiteralPath $VulkanTrainerExe -Destination $VulkanTarget
    Copy-Item -LiteralPath $VulkanDevicesExe -Destination $VulkanTarget

    $NativeGuide = Join-Path $Root "NATIVE_BACKEND.md"
    if (-not (Test-Path -LiteralPath $NativeGuide -PathType Leaf)) {
        throw "Native release guide is missing: $NativeGuide"
    }
    # Keep the standalone distribution documentation native-only. The source
    # repository README intentionally also documents the historical framework
    # path, which would make the isolated Rust/Vulkan bundle look less separate.
    Copy-Item -LiteralPath $NativeGuide -Destination (Join-Path $StagingDir "README.md")
    Copy-Item -LiteralPath $NativeGuide -Destination (Join-Path $StagingDir "NATIVE_BACKEND.md")

    $License = Join-Path $Root "LICENSE.md"
    if (Test-Path -LiteralPath $License -PathType Leaf) {
        Copy-Item -LiteralPath $License -Destination $StagingDir
    }

    $forbidden = Get-ChildItem -LiteralPath $StagingDir -Recurse -File | Where-Object {
        $_.Extension -in @(".py", ".pyc", ".pyd") -or $_.Name -match '(?i)^python.*\.dll$'
    }
    if ($forbidden) {
        $names = ($forbidden | ForEach-Object FullName) -join "`n"
        throw "Native release staging unexpectedly contains Python runtime artifacts:`n$names"
    }

    if (-not $SkipDeviceProbe) {
        Write-Host "Running bundled Vulkan device probe..."
        & (Join-Path $VulkanTarget "hierarchos-vulkan-devices.exe")
        if ($LASTEXITCODE -ne 0) {
            throw "The bundled Vulkan device probe failed with exit code $LASTEXITCODE. Use -SkipDeviceProbe only for a headless build host."
        }
    }

    $hashFiles = Get-ChildItem -LiteralPath $StagingDir -Recurse -File |
        Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
        Sort-Object FullName
    $hashLines = foreach ($file in $hashFiles) {
        # Windows PowerShell 5.1 runs on .NET Framework, which predates
        # System.IO.Path.GetRelativePath. Every staged file is rooted under
        # $StagingDir, so a validated prefix trim is portable across both 5.1
        # and modern PowerShell/.NET.
        if (-not $file.FullName.StartsWith($StagingDir, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Checksum input escaped the native staging directory: $($file.FullName)"
        }
        $relative = $file.FullName.Substring($StagingDir.Length).TrimStart([char[]]"\/").Replace('\', '/')
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
    Set-Content -LiteralPath (Join-Path $StagingDir "SHA256SUMS.txt") -Value $hashLines -Encoding ASCII

    if (Test-Path -LiteralPath $ReleaseDir) {
        Remove-Item -LiteralPath $ReleaseDir -Recurse -Force
    }
    Move-Item -LiteralPath $StagingDir -Destination $ReleaseDir
}
catch {
    if (Test-Path -LiteralPath $StagingDir) {
        Remove-Item -LiteralPath $StagingDir -Recurse -Force
    }
    throw
}

Write-Host ""
Write-Host "Hierarchos Native release created:"
Write-Host "  $ReleaseDir"
Write-Host ""
Write-Host "Entrypoints:"
Write-Host "  HierarchosNative.exe  - pure-Rust GUI + Vulkan training"
Write-Host "  HierarchosCLI.exe     - pure-Rust CLI + Vulkan training"
Write-Host "  vulkan\hierarchos-vulkan-train.exe"
Write-Host "  vulkan\hierarchos-vulkan-devices.exe"
