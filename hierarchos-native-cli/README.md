# Hierarchos Native CLI

`hierarchos-native-cli` is the framework-free command-line frontend for the
Hierarchos native stack. It links the pure-Rust inference engine and launches
only Hierarchos' Rust/Vulkan executables for GPU training and device discovery.
There is no Python or PyTorch compatibility dispatcher in this crate. Network
model/tokenizer/data acquisition is also implemented in Rust; Hugging Face Hub
downloads do not invoke `python`, `huggingface_hub`, Git LFS, or a framework
loader.

The canonical interchange boundary is a Hierarchos model package containing
`model.safetensors`, `hierarchos_rust_config.json`, `hierarchos_config.json`, and
local tokenizer assets. FP32 master tensors keep the same names and shapes across
the native trainer and external consumers, while native exact-resume state uses
backend-neutral sidecars owned by `hierarchos-vulkan`.

Build:

```powershell
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices
cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml
```

Typical commands:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe devices

# Download a published canonical Hierarchos package directly from the Hub.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe pull `
  --repo YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --revision main `
  --out-dir .\hierarchos_from_hf

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\dataset.jsonl `
  --out-dir .\trained `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --training-chunk-size 256 --precision fp32

# Native assistant-SFT recovery profile. Explicit values still override the preset.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\alpaca.jsonl `
  --out-dir .\assistant_recovery `
  --assistant-recovery --precision fp32

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe finetune `
  --model-path .\trained `
  --train .\domain_dataset.jsonl `
  --out-dir .\finetuned `
  --epochs 1 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-5 --training-chunk-size 256 --precision fp32

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\finetuned `
  --carry-chat-state --chat-state-file .\chat-state.json
```

The Hub can also be used directly at the training boundary. `--hf-model` pulls
a canonical Hierarchos package, `--hf-tokenizer` pulls standard tokenizer
assets for Rust-only fresh initialization, and `--hf-dataset` discovers
JSONL/NDJSON training files from Hub repository metadata. The root CLI spelling
`--tokenizer-path OWNER/REPO` is also recognized automatically when that value
is not an existing local path:

```powershell
# Warm-start a native Vulkan run from a canonical model and dataset on HF.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-model YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --hf-model-revision main `
  --hf-dataset YOUR_ORG/YOUR_DATASET `
  --hf_dataset_split train `
  --hf-dataset-file data/train.jsonl `
  --hf-dataset-revision main `
  --out-dir .\trained `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --training-chunk-size 256 --precision fp32

# Or initialize coherent-v9 from an ordinary Hub tokenizer with no model input.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-tokenizer openai-community/gpt2 `
  --train .\dataset.jsonl `
  --out-dir .\fresh_native `
  --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64
```

`--hf-cache-dir DIR` relocates the native cache; otherwise it uses
`.hierarchos-hf-cache` beside the repository. `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` is attached as a bearer token for private/gated repos.
Revisions can be pinned independently with `--hf-model-revision`,
`--hf-tokenizer-revision`, and `--hf-dataset-revision`.

The release path has been exercised end-to-end with an ordinary Hub tokenizer
and with a Hub-hosted JSONL dataset: `openai-community/gpt2` can seed fresh
coherent-v9 initialization, while `polinaeterna/jsonl_test` with
`--hf-dataset-file data/train.jsonl` is downloaded, tokenized, and trained without
leaving the native Rust/Vulkan stack.

The native Hub path is deliberately file-oriented and fail-closed. Model pulls
must expose the canonical Hierarchos `model.safetensors`,
`hierarchos_rust_config.json`, `hierarchos_config.json`, and `tokenizer.json`.
Dataset discovery honors `--hf_dataset_config` and `--hf_dataset_split` as
selection hints. If a split consists of multiple JSONL/NDJSON shards they are
downloaded in lexical shard order and combined into one cached line stream. If
discovery is ambiguous, `--hf-dataset-file PATH.jsonl` selects the exact file.
The native CLI does not execute remote dataset builder scripts, convert
Parquet/CSV implicitly, or execute arbitrary remote code.

Raw JSONL preprocessing is native as well. The frontend recognizes `text`/`content`,
`instruction`/`output`, `prompt`/`completion`, and `question`/`answer` schemas,
appends the tokenizer EOS token, drops blank completions by default, preserves the
prompt suffix plus the beginning of the answer when an example must be truncated,
and supports `--min-response-tokens`, `--allow-empty-completions`, response-boundary
weights, and `--assistant-recovery`. The assistant-recovery preset applies the
supported root-CLI SFT defaults (Alpaca formatting, four epochs, `6e-5` LR,
`0.03` warmup ratio, `0.10/1.0` prompt/response weights, `2x` the first 32
response tokens, 16 reserved answer tokens, `0.003` ponder weight, and a
5000-step fresh-model memory-gate warmup). Independent framework-side LTM
optimizer/value-alignment knobs are not fabricated by the native frontend; use a
schema-v6 token cache for the exact already-tokenized cross-runtime data objective.

Set `HIERARCHOS_VULKAN_BIN_DIR` when the Vulkan trainer/device binaries are
installed outside the repository or are not placed next to the CLI executable.

Native-only behavior is fail-closed. Framework-object `.pt` checkpoints,
framework-style Hugging Face dataset builders, external Python benchmark
registries, and PEFT LoRA geometry injection are not silently delegated. Native
`finetune` trains
coherent-v9's existing recurrent low-rank factors, DeepEmbed/ROSA adapter factors
and routers, and slow-LTM tensors through the same Vulkan forward/backward graph;
repeat `--trainable-prefix` to replace that default selection. `merge-lora` is
implemented in Rust for a bound Hierarchos PEFT SafeTensors adapter package and
emits a standalone canonical model package without importing PEFT, Python, or
PyTorch. Creating a brand-new arbitrary PEFT-LoRA geometry at runtime remains
intentionally unsupported because that would change the canonical architecture.

The high-level native `train` frontend preserves the root CLI's ordinary
training defaults (`epochs=3`, `batch_size=64`, `seed=1337`, `min_lr=1e-6`, and
`ponder_loss_weight=0.01`) while still letting explicit arguments win. Legacy
`--amp` maps to the qualified `fp16-storage-parity` Vulkan policy and `--no-amp`
maps to `fp32`; the native binary never dispatches either option to a framework.
