# Hierarchos Native: Rust + Vulkan

Hierarchos Native is the framework-free execution path for the coherent-v9 / RWKV-v9 model used by Hierarchos Alpha v0.30. In this backend, model loading, tokenization, inference, CLI orchestration, fresh-model initialization, checkpoint handling, and the user-facing native GUI are Rust. Full-model training and optimization execute through Vulkan compute.

Python, PyTorch, libtorch, `tch`, and `pyo3` are not runtime dependencies of `hierarchos-inference`, `hierarchos-vulkan`, or `hierarchos-native-cli`. References to “PyTorch parity” in the native source mean numerical/semantic parity and a shared checkpoint layout, not a PyTorch integration layer.

## Interchange contract

The portable boundary is a canonical SafeTensors package. Model parameters retain the same tensor names, shapes, and row-major layouts expected by Hierarchos' reference PyTorch/CUDA implementation. Native Vulkan training keeps FP32 master parameters and writes `model.safetensors` plus JSON/backend-neutral resume state.

That means a model trained through Vulkan can be loaded by the pure-Rust inference runtime and can also be consumed by an external CUDA implementation that understands the same Hierarchos tensor contract. The native backend itself still uses Vulkan on NVIDIA hardware; “CUDA compatible” here means checkpoint/model interchange, not that CUDA is linked into the native executable.

## Build the native-only Windows release

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_native_release.ps1
```

The builder runs the native Rust test gates, audits the dependency trees for Python/libtorch bindings, builds optimized release binaries, optionally probes the local Vulkan installation, and creates:

```text
dist\Hierarchos-Native\
  HierarchosNative.exe
  HierarchosCLI.exe
  vulkan\
    hierarchos-vulkan-train.exe
    hierarchos-vulkan-devices.exe
  README.md
  NATIVE_BACKEND.md
  LICENSE.md
  SHA256SUMS.txt
```

Inside this standalone distribution, `README.md` contains this same native-only guide. The repository-wide README is intentionally not copied into the bundle because it also documents the separate historical Python/PyTorch implementation.

For a CI/headless host with no Vulkan-capable display device, add `-SkipDeviceProbe`. `-SkipTests` is available for a rebuild only after the same source revision has already passed the native gates.

### Verified native acceptance status (August 28, 2026)

The isolated release builder currently passes with `12/12` `hierarchos-inference` tests, `197/197` runnable `hierarchos-vulkan` library tests (`8` additional GPU microprofiles intentionally ignored), `16/16` `hierarchos-native-cli` tests, and `6/6` tests for the dedicated `hierarchos-native` GUI. The GUI gate covers exact-resume policy rehydration, Vulkan device parsing/selection, and native training-event parsing in addition to compiling the executable. The dependency audit reports no `pyo3`, `tch`, `torch-sys`, or libtorch bindings in the native release graph, and the staged `dist\\Hierarchos-Native` bundle contains no `.py`, `.pyc`, `.pyd`, or Python DLL runtime artifacts.

The rebuilt bundled device probe enumerated the local `AMD Radeon Graphics` Vulkan adapter. The public bundled `HierarchosCLI.exe` then completed a four-row, two-microbatch FP32 training smoke with gradient accumulation, wrote an optimizer-boundary checkpoint, resumed that checkpoint into epoch 2 with both optimizer and portable training-session state restored, and produced a model package that loaded directly through `hierarchos-inference`. This is a correctness/readiness smoke on one AMD device, not a cross-vendor performance claim; NVIDIA/CUDA interoperability remains a checkpoint-ABI property until the same package is exercised on an NVIDIA host.

On other platforms, build the same crates directly with Cargo and place the trainer/device binaries beside the CLI or under a `vulkan/` subdirectory:

```bash
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices --locked
cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml --locked
cargo build --release --manifest-path hierarchos-gui/Cargo.toml --bin hierarchos-native --locked
```

## Inspect Vulkan devices

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe devices
```

The native CLI automatically discovers the bundled trainer and device probe in its `vulkan` directory. `HIERARCHOS_VULKAN_BIN_DIR` can point at a different native Vulkan binary directory for development.

## Train an existing model package

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe train `
  --model-path .\hierarchos_model `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_model `
  --epochs 3 `
  --batch_size 4 `
  --accumulation-steps 4 `
  --starting-lr 1e-4 `
  --min-lr 1e-6 `
  --warmup-ratio 0.03 `
  --training-chunk-size 256 `
  --precision fp16-storage-parity `
  --device-index 0
```

The native frontend understands the common root-CLI spellings such as `--batch_size`, `--accumulation-steps`, `--starting-lr`, `--rwkv-weight-decay`, `--adamw-eps`, and `--training-chunk-size`, then launches the Vulkan trainer directly.

Training data may be tokenized JSONL, ordinary local text/instruction JSONL that can be tokenized with the package tokenizer, or a Hierarchos schema-v6 token-cache directory. Schema-v6 is the strict cross-runtime data-objective interchange path because IDs, labels, masks, loss weights, and content identity are already materialized.

## Train from scratch without Python

An existing model package is not required. Give the CLI local tokenizer assets and it will construct a coherent-v9 parameter package in Rust before entering the Vulkan trainer:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe train `
  --tokenizer-path .\tokenizer_assets `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_fresh `
  --epochs 3 `
  --batch_size 4 `
  --training-chunk-size 256 `
  --precision fp32 `
  --device-index 0
```

The native CLI can also retrieve supported canonical model/tokenizer assets and JSONL/NDJSON datasets from Hugging Face over Rust HTTPS. This does not launch `huggingface_hub`, Python, or PyTorch:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe pull `
  --repo OWNER/MODEL `
  --out-dir .\model

.\dist\Hierarchos-Native\HierarchosCLI.exe train `
  --hf-tokenizer openai-community/gpt2 `
  --hf-dataset OWNER/DATASET `
  --out-dir .\fresh_model `
  --epochs 1 `
  --device-index 0
```

Dataset repositories that require arbitrary Python dataset-builder code intentionally fail closed; the native Hub path supports repository metadata plus JSONL/NDJSON files.

## Native fine-tuning

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe finetune `
  --model-path .\hierarchos_vulkan_model `
  --train .\domain_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_finetuned `
  --epochs 1 `
  --batch_size 4 `
  --starting-lr 1e-5 `
  --precision fp32 `
  --device-index 0
```

By default native `finetune` freezes the optimizer to coherent-v9's existing low-rank/shared recurrent factors, DeepEmbed/ROSA factors and routers, and slow-LTM tensors. Repeat `--trainable-prefix PREFIX` to explicitly choose canonical parameter subtrees. Because these factors are already part of the architecture, the output is a complete model package rather than a framework-owned adapter object.

Bound PEFT-LoRA SafeTensors adapters can also be merged into a canonical package entirely in Rust:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe merge-lora `
  --model-path .\base_model `
  --lora-adapter-path .\adapter `
  --out-dir .\merged_model
```

Arbitrary new PEFT geometry injection at runtime is not silently emulated and remains outside the native contract.

## Exact resume versus weights-only continuation

Use `--resume-from-ckpt` for an exact continuation. The trainer validates and restores model weights, AdamW slots and clocks, pending accumulated gradients, LR scheduler, dynamic loss scaler when active, data cursor/shuffle state, and the portable recurrent/LTM/ROSA replay carrier.

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe train `
  --resume-from-ckpt .\hierarchos_vulkan_model\checkpoint-epoch-1-step-100 `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_resumed `
  --epochs 4 `
  --batch_size 4 `
  --accumulation-steps 4 `
  --starting-lr 1e-4 `
  --min-lr 1e-6 `
  --training-chunk-size 256 `
  --precision fp16-storage-parity `
  --device-index 0
```

Use `--model-path` instead when you intentionally want a new optimizer/session initialized from existing model weights.

## Native inference and chat

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe chat `
  --model-path .\hierarchos_vulkan_model `
  --prompt "Explain hierarchical recurrent reasoning." `
  --temperature 0.7 `
  --top-k 40 `
  --top-p 0.9
```

The native chat path supports portable recurrent/ROSA/LTM state save/resume and the opt-in raw-logit uncertainty stop guards exposed by the root CLI.

For local native throughput measurement:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe benchmark `
  --model-path .\hierarchos_vulkan_model `
  --benchmark-iterations 20
```

External `lm-eval`/ARC catalogs are not launched from the native executable.

## Native GUI

`HierarchosNative.exe` is the dedicated Rust GUI. It loads canonical SafeTensors packages with `hierarchos-inference` and launches the Vulkan trainer directly. It is deliberately separate from the repository's compatibility GUI/backend so a native release does not acquire a Python/PyTorch dependency through packaging.

## Precision policies

The Vulkan trainer exposes explicit precision policies rather than relying on framework autocast:

- `fp32`: canonical FP32 execution.
- `fp16-storage-fp32-compute`: FP16 parameter mirrors where qualified, with FP32 compute/masters.
- `fp16-storage-parity`: qualified mixed-storage parity path with native dynamic loss scaling.
- `fp16-storage-fp16-lm-backward`: native Float16 LM-head backward products with FP32 softmax, accumulation, master parameters, and optimizer state.

Low-precision policies are capability checked and do not silently downgrade to a different numerical contract.

## Current scope and fail-closed boundaries

The native path covers coherent-v9 full-model training, native parameter-efficient fine-tuning, Rust inference/chat, native Hugging Face transport for supported file-based repositories, exact native resume, device enumeration, local throughput benchmarking, SafeTensors package conversion/validation, and bound SafeTensors LoRA merge.

Legacy framework-object `.pt` checkpoint deserialization, arbitrary Python dataset builders, external Python benchmark registries, arbitrary new PEFT-LoRA geometry injection, and the old scalar-RWKV quantized format are not native features. Those requests fail closed instead of launching Python behind the user's back.

For production claims, distinguish three things explicitly:

1. Rust/Vulkan execution: what these native binaries actually run.
2. PyTorch parity: the numerical/reference behavior the implementation is tested against.
3. CUDA interchange: external CUDA consumers can load the same canonical tensor package; this does not make the native trainer a CUDA program.
