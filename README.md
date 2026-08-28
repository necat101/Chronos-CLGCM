-----

# Hierarchos Alpha v0.30 — Native/Vulkan Major Update

> [!IMPORTANT]
> **Hierarchos Native is now the primary execution path and the center of active
> backend development.** Alpha v0.30 can initialize, train, fine-tune, resume,
> load, chat with, and benchmark coherent-v9 / RWKV-v9 model packages without a
> Python or PyTorch runtime. Rust owns model/package orchestration, tokenization,
> inference, the CLI, and the native GUI; full-model training and optimization
> run through Vulkan compute.

The native stack is intentionally framework-free at runtime:

- **`hierarchos-vulkan/`** — Vulkan compute training, backward passes, gradient
  accumulation, AdamW, mixed-storage precision policies, checkpointing, device
  discovery, and GPU profiling.
- **`hierarchos-inference/`** — pure-Rust coherent-v9 model loading and inference.
- **`hierarchos-native-cli/`** — native `train`, `finetune`, `chat`, `benchmark`,
  `devices`, Hugging Face transport, package conversion/validation, and LoRA
  merge workflows.
- **`hierarchos-gui/`** — the dedicated `HierarchosNative.exe` Rust GUI, sharing
  the same native model and Vulkan training path rather than wrapping Python.
- **Canonical SafeTensors packages** are the interchange boundary. Native Vulkan
  training keeps FP32 master parameters and writes portable model plus
  backend-neutral resume state that can be consumed by the Rust runtime or by
  external implementations that understand the Hierarchos tensor contract.

Python/PyTorch remains in the repository as a reference, compatibility, and
research implementation. It is no longer the recommended starting point for a
normal Hierarchos v0.30 build or local training workflow. Framework-only
features remain explicit and fail closed from the native CLI instead of causing
an invisible fallback to Python.

## Native-first architecture

```text
                         HIERARCHOS ALPHA v0.30
                              native path

     +----------------------+        +-----------------------+
     | HierarchosNative.exe |        |   HierarchosCLI.exe   |
     |      Rust GUI        |        |       Rust CLI        |
     +----------+-----------+        +-----------+-----------+
                |                                |
                +---------------+----------------+
                                |
                                v
                 +------------------------------+
                 | Rust package/model frontend  |
                 | tokenizer + config + HF I/O  |
                 +---------------+--------------+
                                 |
                  +--------------+--------------+
                  |                             |
                  v                             v
     +---------------------------+   +---------------------------+
     | hierarchos-inference      |   | hierarchos-vulkan         |
     | coherent-v9 Rust runtime  |   | forward/backward + AdamW  |
     | chat + recurrent state    |   | full-model Vulkan training|
     +-------------+-------------+   +-------------+-------------+
                   |                               |
                   +---------------+---------------+
                                   |
                                   v
                    +-----------------------------+
                    | Canonical SafeTensors model |
                    | + portable resume sidecars  |
                    +-----------------------------+
```

This split keeps the hot path native while preserving a stable model-package
contract across implementations. "CUDA compatible" in this project means an
external CUDA implementation can consume the same canonical package; the native
trainer itself remains Vulkan on NVIDIA, AMD, and other conformant Vulkan
devices.

## Build and run the native release

On Windows, the recommended entry point is the isolated native release builder:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_native_release.ps1
```

It audits the Rust dependency graph for Python/libtorch bindings, runs the native
test gates, builds optimized binaries, probes Vulkan when available, and stages:

```text
dist\Hierarchos-Native\
  HierarchosNative.exe
  HierarchosCLI.exe
  vulkan\
    hierarchos-vulkan-train.exe
    hierarchos-vulkan-devices.exe
  README.md
  NATIVE_BACKEND.md
  SHA256SUMS.txt
```

Then inspect the available Vulkan adapters:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe devices
```

Start a completely fresh coherent-v9 run without Python:

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

Or train an existing canonical package with native mixed-storage execution:

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

For native chat:

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe chat `
  --model-path .\hierarchos_vulkan_model `
  --prompt "Explain hierarchical recurrent reasoning."
```

See [NATIVE_BACKEND.md](NATIVE_BACKEND.md) for exact-resume semantics, native
fine-tuning, Hugging Face transport, LoRA merge, precision policies, manual
cross-platform Cargo builds, and the explicit fail-closed boundary.

### Native acceptance status — August 28, 2026

The isolated release gate currently records `12/12` `hierarchos-inference`
tests, `197/197` runnable `hierarchos-vulkan` library tests (`8` additional GPU
microprofiles intentionally ignored), `16/16` `hierarchos-native-cli` tests,
and `6/6` native GUI tests. The release dependency audit reports no `pyo3`,
`tch`, `torch-sys`, or libtorch bindings, and the staged native bundle is
checked to contain no Python runtime artifacts.

The bundled smoke path has also exercised Vulkan device discovery on an AMD
Radeon adapter, FP32 training with gradient accumulation, optimizer-boundary
checkpoint creation, exact resume into the next epoch, and direct loading of the
trained package through `hierarchos-inference`. These are readiness/correctness
gates, not cross-vendor performance claims.

## Release and model contract

**Current release: Hierarchos Alpha v0.30.** New models use the RWKV-v9 core with
corrected manager/worker recurrence, hard per-row ACT, bounded ROSA, shared
factorized token adapters, persisted memory-gate schedule state, a causally
trained transactional fast-memory writer, and fail-closed checkpoint/resume
boundaries. A reference `448/448/448` Alpha v0.30 model with the GPT-2 vocabulary
has `30,227,653` unique parameters and `102` state-dict entries. The historical
RWKV-v8 design has `232,516,229` parameters and `95` state-dict entries.

`v0.30` is the Hierarchos product release; RWKV-v8 and RWKV-v9 are core
architecture generations. The literal `legacy-v8` and `coherent-v9` values stay
in CLI/config/checkpoint metadata as compatibility-stable internal identifiers.
v2/v3 checkpoints remain on `legacy-v8` and are never silently reinterpreted as
RWKV-v9. The old scalar-RWKV quantized `.npz` path cannot reproduce the active
matrix-state contract and remains intentionally unsupported. Read the
[RWKV-v9 migration notes](COHERENT_V9_MIGRATION.md) before moving an expensive
legacy run.

## Model architecture: RWKV core v8 to v9

The diagram below is the model-level architecture transition. The right-hand
side is the coherent-v9 core used by **Hierarchos Alpha v0.30** in both the
native implementation and the framework reference path.

```text
                RWKV CORE v8                                  RWKV CORE v9
       (historical Hierarchos core)                    (Hierarchos Alpha v0.30 core)

     +-------------------------+                       +-------------------------+
     | Input token IDs         |                       | Input token IDs         |
     +------------+------------+                       +------------+------------+
                  |                                                     |
                  v                                                     v
     +-------------------------+                       +-------------------------+
     | Tied token embedding    |                       | Tied token embedding    |
     | and language-model head |                       | and language-model head |
     +------------+------------+                       +------------+------------+
                  |                                                     |
                  v                                                     v
     +-------------------------+                       +-------------------------+
     | Three extra full-vocab  |                       | Shared low-rank token   |
     | feature tables:         |                       | adapters for H, L, and  |
     | H DeepEmbed,            |                       | ROSA features           |
     | L DeepEmbed, and ROSA   |                       | No duplicate vocab      |
     +------------+------------+                       | tables                  |
                  |                                    +------------+------------+
                  v                                                     |
     +-------------------------+                                        v
     | Legacy ROSA history and |                       +-------------------------+
     | no-prediction sentinel  |                       | Bounded deterministic   |
     | behavior                |                       | ROSA; zero-valued       |
     +------------+------------+                       | no-prediction sentinel  |
                  |                                    +------------+------------+
                  v                                                     |
     +-------------------------+                                        v
     | Legacy absolute-time    |                       +-------------------------+
     | LTM feature injection   |                       | Valid-slot-masked LTM   |
     | and clock filtering;    |                       | metadata-only token and |
     | target-gradient inner   |                       | wall timestamps;        |
     | updates by default      |                       | trained value writer;   |
     |                         |                       | transactional fast      |
     +------------+------------+                       | memory policies         |
                  |                                    +------------+------------+
                  v                                                     |
     +-------------------------+                                        v
     | Manager soft ACT and    |                       +-------------------------+
     | legacy state commit     |                       | Hard per-row cumulative |
     +------------+------------+                       | ACT and selected-state  |
                  |                                    | commit                  |
                  v                                    +------------+------------+
     +-------------------------+                                        |
     | Worker recurrence v1;   |                                        v
     | legacy refinement and   |                       +-------------------------+
     | sum-square commitment   |                       | Corrected matrix-state  |
     +------------+------------+                       | recurrence, row-local   |
                  |                                    | refinement, mean-square |
                  v                                    | commitment              |
     +-------------------------+                       +------------+------------+
     | Tied LM head -> logits  |                                    |
     +------------+------------+                                    v
                  |                                    +-------------------------+
                  v                                    | Tied LM head -> logits  |
       232,516,229 parameters                          +------------+------------+
       95 state-dict entries                                        |
                                                                    v
                                                         30,227,653 parameters
                                                         102 state-dict entries
                                                         86.9998% fewer params
```

The parameter reduction comes primarily from replacing three vocabulary-sized
feature tables with shared low-rank adapters. The output embedding remains tied,
so Alpha v0.30 keeps the normal full-vocabulary language-model objective without
adding a second output matrix.

## Safe online adaptation in Alpha v0.30

Online adaptation remains an intended part of Hierarchos. The safe RWKV-v9
contract separates *learning a writer* from *allowing that writer to mutate a
live conversation*:

- `--ltm-training-mode read-only` disables supervised target-gradient writes to
  fast LTM slots during training. It does not disable the learned LTM reader or
  the causal `val_proj` writer auxiliary, and it does not remove the runtime
  transaction path.
- `--no-persist-state` resets H/L/context/LTM carriers between unrelated,
  shuffled dataset rows. State still flows forward across every 256-token TBPTT
  chunk within one sample. This flag does not disable post-training chat
  adaptation.
- Fresh RWKV-v9 runs train the value writer with an energy-normalized causal
  alignment objective at weight `0.01`, sampled every eighth token. A checkpoint
  needs at least `100` successful writer updates and must pass finite-loss,
  alignment-EMA, and writer-norm readiness gates before any `val_proj`/Hebbian
  writer path is permitted. The default explicit feedback transaction is
  gradient-derived and remains separately bounded by replay and norm checks.
- Normal chat prefill and token generation are read-only. At an allowed action
  boundary, Hierarchos clones the candidate fast state, applies a bounded update
  with learning-rate backoff, checks finite and norm budgets, replays the local
  objective, and commits only a non-worsening candidate. A rejection leaves the
  live state unchanged.

The chat policy is explicit:

| `--online-adaptation-policy` | Allowed writes |
| --- | --- |
| `off` | No writes, including explicit actions. Use for static evaluation and logit parity. |
| `validated` (default) | Only explicit `/learn`, `/reject`, and `/correct <target>` actions. This is the safe interactive default. |
| `prompt` | `validated` plus conservative passive writes from observed user prompts; generated answers remain read-only. Legacy `--passive-learning` maps here. |
| `prompt+response` | `prompt` plus quality/surprise-gated writes from model responses. This is the highest-risk, opt-in policy. Legacy `--passive-response-learning` maps here and requires prompt-passive learning. |

`/learn` reinforces the last completed response, `/reject` applies an
unlikelihood update to it, and `/correct <target>` rejects that response before
learning the supplied correction. Natural-language feedback interpretation is
off by default; enable `--natural-feedback-detection` only when that heuristic
behavior is desired. Exact slash actions are the auditable path.

For framework-reference CUDA reproduction, the measured Colab profile below is
the literal one-line Alpha v0.30 PyTorch fresh-run command (batch `64`,
accumulation `1`, no legacy checkpoint resume). New users should start with the
native Vulkan workflow above unless they specifically need a framework-only
feature:

```bash
!cd ./Hierarchos && python hierarchos_cli.py train --architecture-revision coherent-v9 --hf_dataset "netcat420/Experiment_0.1" --hf_dataset_split "train" --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" --out-dir "./chatHRM_alpha_v030" --tokenizer-path "openai-community/gpt2" --epochs 15 --batch_size 64 --accumulation-steps 1 --accumulation-normalization weighted-token --max_length 8880 --training-chunk-size 256 --no-full-sample-bptt --no-full-sample-activation-checkpointing --detach-every-n-steps 0 --no-persist-state --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64 --max_h_steps 5 --max_l_steps 5 --alpaca --assistant-recovery --ltm-training-mode read-only --ltm-value-alignment-weight 0.01 --ltm-value-alignment-stride 8 --ltm-value-alignment-min-updates 100 --train-prompt-tokens --prompt-loss-weight 0.10 --response-loss-weight 1.0 --response-boundary-loss-weight 2.0 --response-boundary-tokens 64 --min-response-tokens 1 --starting-lr 2e-5 --min-lr 1e-7 --warmup-ratio 0.01 --ltm-lr 3e-4 --min-ltm-lr 1e-5 --adaptive-ponder --ponder-target-scale 0.65 --ponder-loss-weight 0.003 --commitment-loss-weight 0.5 --max-commitment-cost-for-backward 4.0 --max-ce-loss-for-backward 0 --max-ponder-cost-for-backward 0 --startup-weight-max-abs 0 --halt-logit-clamp 30.0 --recurrent-state-clamp 50.0 --context-state-clamp 50.0 --activation-clamp 100.0 --drift-state-clamp 2.0 --drift-norm-clamp 4.0 --rwkv-channel-mix-key-clamp 12.0 --rwkv-channel-mix-deepembed-clamp 4.0 --drift-delta-scale 0.35 --memory-gate-warmup-steps 5000 --memory-gate-warmup-floor 0.10 --grad-clip 0.75 --device cuda --amp --force-compile --compile-mode max-autotune-no-cudagraphs --compile-static-worker-loop --hf-token-cache --hf-token-cache-dir "/content/hierarchos_token_cache/experiment_0_1_alpha_v030" --token-cache-build-batch-size 256 --length-bucket-auto-sample-size 1000000 --cuda-prefetch --cuda-loss-chunk-rows 0 --num_workers -1 --padding-metric-steps 0 --save-steps 600
```

The Markdown renderer may visually wrap the command, but it is one physical
line for Colab/autoclicker use. The nonzero LTM LR range and enabled schedule
belong to the Alpha v0.30 memory-training contract; runtime feedback uses the
separate `--online-ltm-lr` transaction setting. The pinned Experiment_0.1
revision contains 777,420 responses shorter than 16 GPT-2 tokens (about 14.8%
of its 5,251,582 rows), so this dataset-specific recipe explicitly overrides
the assistant-recovery preset with `--min-response-tokens 1` rather than
silently discarding that material part of the corpus.

**Historical Alpha v0.21 / RWKV-v8 focus**: the v0.21 material immediately below
describes the checkpoint-compatible RWKV-v8 full-sample-BPTT and input
pipeline work. Its `232,516,229` parameter, `95` tensor, and historical test
figures are not claims about Hierarchos Alpha v0.30.

**Exact per-sample gradient connectivity**: `--full-sample-bptt` removes recurrent gradient detachment across every active token in a dataset sample. Non-reentrant activation checkpointing rematerializes bounded temporal segments during one global backward pass; segment boundaries never truncate gradients or externally reseed drift. This gives full BPTT within each row up to `max_length`, but deliberately does not connect unrelated dataset rows or guarantee semantic coherence by itself.

**RTX PRO 6000 Blackwell profile**: keep `--batch_size 64 --training-chunk-size 256` and explicitly use `--full-sample-checkpoint-segment-size 224` after a worst-case memory preflight. An 8,880-token row is compile-padded to 8,960 tokens, exactly `40` segments of `224`. This setting should run closer to `256`-segment throughput while retaining more OOM headroom; `128` remains the safe general default and `64` is the first OOM fallback. Segment size changes speed and VRAM only, not the full-sample gradient horizon.

**Input-pipeline savings**: Hugging Face revisions are pinned to immutable SHAs, fast tokenizers process native batches, Arrow rows use batched fetches, compact schema-v6 caches are mmap-backed, and length buckets are deterministically auto-tuned. The weighted GPT-2+ROSA profile with prompt-token training uses about 4 token-data bytes/token, approximately 20 GB decimal for 5B tokens before index/RLE metadata.

**Precision, clipping, and parity**: Blackwell training uses BF16 autocast while parameters, AdamW state, sensitive recurrence, LTM math, and loss accumulation retain FP32 precision. Finite loss/gradient rejection, unscale-before-clip ordering, global gradient clipping, state/context/drift bounds, and channel-mix clamps remain active. Exact checkpoints use the aligned full-precision chat/evaluation recurrence. Quantized inference is unavailable for the Alpha v0.30 RWKV-v9 core and fails closed rather than claiming control-flow or logit parity.

**Historical verification**: this legacy-v8 release recorded `283 passed, 4 skipped`; `python tools/check_architecture_integrity.py` reported `65 passed`, `1` documented legacy-quantization warning, and `0 failures`. Direct-versus-checkpointed exact-BPTT tests preserved all gradients with maximum observed parameter-gradient delta `1.49e-08`; correct segmented recurrence matched monolithic logits within `2.38e-07` in the boundary control.

## Previous v0.20.7 Notes: Epoch-13 Compatibility + Safe LTM Writers

**v0.20.7 focus**: validate the coherent epoch-13 release weights as a real control and close the remaining untrained Hebbian-writer path without changing a learned parameter name or shape. The reference model remains `232,516,229` parameters and `95` state tensors. See [EPOCH13_CHECKPOINT_AUDIT.md](EPOCH13_CHECKPOINT_AUDIT.md) and the machine-readable [epoch-13 control report](checkpoint_audits/epoch13-control.json).

**Real checkpoint compatibility**: all `95` tensors in the re-downloaded epoch-13 checkpoint strict-load finitely, tied embeddings remain tied, and the exact local tokenizer matches the `50,257`-token vocabulary. Chunked epoch-13 recurrence and one-token chat streaming match across a real `256`-token TBPTT boundary with maximum logit delta `1.144409e-05` over `270` tokens. A single nested download directory now resolves safely for both weights and tokenizer assets.

**Safe online memory**: the historical RWKV-v8 language objective never trained `val_proj`, the projection used for Hebbian validation writes. Legacy checkpoints block that random writer at the model boundary while retaining gradient-derived transactional feedback learning. RWKV-v8 runs can opt into `--ltm-value-alignment-weight`; its default remains `0.0` to preserve the legacy objective. Fresh Alpha v0.30 RWKV-v9 runs instead train the same existing writer causally at weight `0.01` and stride `8`, exclude it from AdamW decay, and require at least `100` successful updates plus readiness checks before a `val_proj`/Hebbian write can run.

**Epoch-13 weight findings**: the checkpoint is coherent and usable as a budget-saving v2 warm start, but it is specialized rather than optimal. ROSA routing is strong, LTM routing is nearly closed, and the two DeepEmbed tables average about `0.355` after historical weight decay from their `1.0` identity initialization. Do not reset these learned compensations; rehabilitate them gradually with diverse data and the corrected no-decay optimizer.

**Verification**: `python -m pytest -q` reports `230 passed, 3 skipped`; `python tools/check_architecture_integrity.py` reports `65 passed, 1 documented warning, 0 failures`. The warning remains the legacy scalar-RWKV quantized path; coherent v8 checkpoints should use full precision.

## Previous v0.20.6 Notes: Checkpoint-Compatible Architecture Hardening

**v0.20.6 focus**: harden the coherent full-precision v8 path without changing the learned checkpoint layout. The reference `448/448/448` model remains `232,516,229` parameters, the state dictionary remains `95` tensors, and existing coherent checkpoints require no weight conversion. Strict synthetic checkpoint round-trip testing preserves every state key, tied embedding storage, and logits exactly.

**Runtime parity fixes**: chat now applies drift only at absolute TBPTT boundaries, including when a carried conversation begins partway through a chunk, and consumes the final emitted token exactly once before state carry/save. The architecture control reports `2.384e-07` maximum logit drift across an in-chunk turn split.

**Checkpoint and resume safety**: learned tensors must load completely and finitely; unsupported shape mismatches, duplicate compiled keys, and conflicting tied weights fail loudly. Saving never repairs or mutates live learned weights, gradients, optimizer moments, or finite LTM state. Checkpoint installation is atomic with rollback, exact resume requires finite optimizer/scheduler/scaler state, inference export preserves the tokenizer source recorded by training, and transient LTM working state is reset for clean static inference.

**Training and memory hardening**: finite model gradients remain norm-clipped while NaN/Inf gradients skip the optimizer step. Filtered LTM retrieval is now per-query and differentiable, rows without valid slots do not decay because another row matched, DeepEmbed stays out of AdamW decay, and full-sequence LoRA finetuning no longer leaks post-backward fast-memory writes between unrelated batches.

**Efficiency and evaluation fixes**: persistent ROSA builds its first automaton in one pass instead of computing the first chunk twice, automatic ROSA workers are bounded, token caches are integrity-checked before use, and lm-eval uses joint context/continuation BPE with correct causal scoring and real batching.

**Verification**: `python -m pytest -q` reports `226 passed, 3 skipped`; `python tools/check_architecture_integrity.py` reports `57 passed, 1 documented warning, 0 failures`. The warning is that CPU/Vulkan quantized inference still targets the older scalar-RWKV format; current coherent v8 checkpoints should use the full-precision path.

## Previous v0.20.5 Notes: KortexHOS Release Profile + Local Benchmark Parity

**v0.20.5 focus**: the current KortexHOS/Hierarchos assistant release path is documented around the settings that produced coherent full-precision chat and stable local benchmark checks. Chat, lm-eval benchmarking, and TBPTT recurrence use aligned chunk/drift behavior, so local CPU benchmark runs can be used as a cheap sanity check before spending more cloud compute.

**Recommended release chat profile**: for Alpaca-trained assistant checkpoints, start with full precision, static chat state, no passive LTM writes, no previous-turn input history, and conservative sampling: `--temperature 0.4 --top-k 40 --top-p 0.9 --repetition-penalty 1.15 --max-new-tokens 256 --no-passive-learning --chat-input-history-turns 0`.

**ROG Ally / local benchmark preset**: `--benchmark-preset rog-ally` applies bounded local defaults: CPU, batch size `1`, sequential tasks, `max_new_tokens=64`, `eval_limit=25` unless overridden, output under `benchmark_results/rog_ally`, and the light suite `arc_easy`, `hellaswag`, `truthfulqa_mc1`.

**Current release smoke result**: on the bounded ROG Ally preset with `--eval-limit 100`, the current checkpoint reported `arc_easy acc=0.3600`, `hellaswag acc=0.3400 / acc_norm=0.3700`, and `truthfulqa_mc1 acc=0.2200`. These are local sanity metrics, not leaderboard claims, but they show the model is not collapsed and that HellaSwag/ARC signal is measurable on consumer hardware.

**hierarchos/evaluation/lm_eval_wrapper.py** and **hierarchos_cli.py**: benchmark mode clears transient LTM working memory, suppresses Hebbian/passive LTM writes, and evaluates with checkpoint-sized TBPTT chunks so benchmark logits stay aligned with the static chat path.

**Research report**: see [HIERARCHOS_FINDINGS_PAPER.md](HIERARCHOS_FINDINGS_PAPER.md) for the preliminary technical paper covering the 232M release, train/chat parity fixes, stability findings, local benchmark results, and proposed scaling plan.

## Previous v0.20.4 Notes: Inference-Like LTM Training

**v0.20.4 focus**: assistant SFT and rescue runs can now train with inference-like LTM dynamics. `--ltm-training-mode read-only` carries ROSA/history state across TBPTT chunks but disables supervised gradient fast-memory writes, preventing the model from learning to depend on an LTM inner-update signal that normal chat generation does not receive. `--assistant-recovery` now defaults to this read-only mode unless explicitly overridden.

**hierarchos/models/core.py** and **hierarchos/training/trainer.py**: the model forward path can skip retained LTM retrieval tensors when training in read-only mode. The trainer still preserves cross-chunk recurrent state and ROSA token history, but it no longer writes supervised fast-memory values that would be absent during chat.

**hierarchos/training/trainer.py**: LoRA finetune now follows the same LTM training-mode contract as full training. Read-only mode disables retained retrieval tensors, supervised LTM inner updates, and LTM LR schedule advancement there too, while still applying runtime stability clamp overrides to the loaded base model.

**hierarchos/training/trainer.py**: DeepEmbed weights are excluded from AdamW weight decay. DeepEmbed starts as a multiplicative identity gate at `1.0`; decaying it toward zero over many epochs can quietly weaken the RWKV channel-mix path.

**hierarchos_cli.py**: adds `--ltm-training-mode {inner-update,read-only}` and the shortcut `--inference-like-ltm-training`. Saved configs and checkpoint hydration preserve the mode, while explicit CLI values still win during resume.

**hierarchos/inference/chat.py**: chat now prints the checkpoint's saved LTM training mode at startup and in `/status`. Read-only checkpoints are identified as aligned with normal prefill/generation; older inner-update checkpoints emit a warning because normal chat cannot reproduce supervised label-gradient LTM writes.

## Previous v0.20.3 Notes: DeepEmbed Channel-Mix Clamp

**v0.20.3 focus**: long assistant SFT and rescue runs now also clamp DeepEmbed's multiplicative RWKV channel-mix modulation. `--rwkv-channel-mix-deepembed-clamp` caps the token-specific DeepEmbed multiplier before `value_cm`, preventing rare learned modulation spikes from re-amplifying the already-clamped ReLU-squared channel-mix FFN.

**hierarchos/models/rwkv_cell.py**: after the v0.20.2 `key_cm` clamp, DeepEmbed modulation is clamped before it can amplify the channel-mix FFN input to `value_cm`. Defaults are `12.0` for `key_cm` and `4.0` for DeepEmbed, so ordinary LayerNorm-fed activations pass through while pathological spikes are bounded.

**hierarchos/models/core.py** and **hierarchos/utils/checkpoint.py**: old checkpoints are backfilled with `rwkv_channel_mix_key_clamp=12.0` and `rwkv_channel_mix_deepembed_clamp=4.0`, and runtime refresh updates existing H/L RWKV cells during resume. This is shape-compatible and does not add or remove learned tensors.

**hierarchos_cli.py** and **hierarchos/training/trainer.py**: the new channel-mix clamp flags are recorded in run configs, printed in stability logs, and remain explicit runtime safety overrides.

## Previous v0.20.2 Notes: Channel-Mix Key Clamp

**v0.20.2 focus**: long assistant SFT and rescue runs gained an explicit RWKV channel-mix preactivation clamp. `--rwkv-channel-mix-key-clamp` caps the `key_cm` activation before the ReLU-squared channel-mix FFN, preventing rare large activations from becoming backward-pass NaNs while preserving normal channel-mix behavior.

**hierarchos/models/rwkv_cell.py**: `key_cm` output is clamped before `relu().square()`. The default clamp is `12.0`, so ordinary LayerNorm-fed activations pass through while pathological spikes are bounded.

## Previous v0.20.1 Notes: Drift Clamp Rescue

**v0.20.1 focus**: large assistant SFT recovery runs now have explicit drift/commit stabilization controls. This patch adds L2 drift-norm clamping, drift-delta scaling, and straight-through commitment-loss capping so runaway worker drift can still receive corrective gradient while the auxiliary loss value remains bounded.

**hierarchos/models/core.py**: `--drift-norm-clamp` caps the worker drift vector's total L2 norm, while `--drift-delta-scale` scales each accumulated worker drift update. These are resume-safe runtime controls and do not require resetting learned `h_module` / `l_module` weights.

**hierarchos/training/trainer.py**: commitment-cost capping now preserves gradient for the commitment penalty. This keeps `--max-commitment-cost-for-backward` numerically safe without making high raw commit drift gradient-dead above the cap.

## Previous v0.20 Notes: Assistant SFT Safety Guardrails

**v0.20 focus**: large assistant SFT runs fail safer and resume cleaner. This release documents the response-preserving dataset path, weighted assistant-loss recipe, disabled CE-backward cap for from-scratch language training, and HF token-cache/resume guardrails.

**hierarchos/training/trainer.py**: `--max-ce-loss-for-backward` now defaults to `0.0` (disabled). This prevents a finite-loss cap below random 50k-vocab CE from silently zeroing language-model gradients during early from-scratch training.

**hierarchos/training/datasets.py**: prompt/completion examples now preserve assistant response tokens when overlong rows are truncated. Blank completions are dropped by default; `--allow-empty-completions` keeps them only when EOS-only answers are intentional labels.

**hierarchos_cli.py**: assistant recovery applies the large-assistant SFT recipe: `prompt_loss_weight=0.10`, `response_loss_weight=1.0`, `response_boundary_loss_weight=2.0`, `response_boundary_tokens=32`, `min_response_tokens=16`, `warmup_ratio=0.03`, `starting_lr=6e-5`, `min_lr=1e-6`, `ltm_lr=3e-4`, `min_ltm_lr=1e-5`, `ponder_loss_weight=0.003`, and `memory_gate_warmup_steps=5000`.

**HF token cache safety**: cache keys include formatting, prompt/response weights, `min_response_tokens`, empty-completion policy, ROSA settings, and chunk size. Resume hydration no longer persists one-shot rebuild flags like `--refresh-hf-token-cache`, nor stale CE-cap settings from older checkpoints.

## Previous v0.19.3 Notes: Progress Sync Throttling

**hierarchos/models/ltm.py** (line 122): CUDA LTM updates now aggregate only touched slots with torch.unique plus indexed writes, avoiding full dense [slots, val_dim] and [batch, slots, val_dim] scatter buffers.

**hierarchos/models/core.py** (line 628): added _compute_cuda_chunked_lm_loss, matching dense CE plus z-loss behavior while chunking lm_head rows for large-vocab CUDA memory pressure.

**hierarchos/training/trainer.py**: progress-bar scalar logging is now throttled with `--progress-log-steps`, avoiding a CUDA sync from `.item()` every single batch on fast GPUs.

**hierarchos/training/datasets.py**: Alpaca formatting is standardized around the explicit `### Instruction:`, optional `### Input:`, and `### Response:` prompt string while preserving the same supervised completion labels.

**The "Progress Sync Throttling" update** — Hierarchos now avoids unnecessary CUDA-to-CPU metric syncs during training progress display. The training objective, batch data, labels, and model architecture are unchanged.

**The "DataLoader Throughput Tuning" update** — Hierarchos now keeps CUDA input queues bounded by tying auto prefetch to worker count, uses conservative CUDA worker defaults for pre-tokenized datasets, and keeps pinned memory specific to CUDA training.

**The "Optimization and GUI Update"** — Hierarchos keeps its CPU-friendly math paths intact while automatically switching hot LTM memory operations to GPU-friendly gather/scatter math on CUDA. This release also highlights the Windows GUI bundle workflow for easier local inference and experimentation.

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) and RWKV linear attention to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### New in v0.20.5: KortexHOS Release Profile + ROG Ally Benchmarking

#### Coherent Full-Precision Assistant Profile
- **Best-Known Chat Parameters**: use `--temperature 0.4 --top-k 40 --top-p 0.9 --repetition-penalty 1.15 --max-new-tokens 256 --no-passive-learning --chat-input-history-turns 0` for first-pass release testing of Alpaca-trained checkpoints.
- **Static Eval First**: passive learning and chat history are useful experiments, but the release-quality baseline should be static. This keeps chat aligned with the benchmark path and avoids prompt-history/LTM pollution while judging core weights.
- **TBPTT Drift Parity**: chat generation uses recurrent state continuity without per-token drift reseeding. Benchmark evaluation uses the same checkpoint-sized chunking contract so local benchmark numbers reflect the model path used in clean chat.
- **ROG Ally Preset**: `--benchmark-preset rog-ally` runs a bounded consumer-hardware sanity suite on CPU with batch size `1` and sequential tasks. Use `--eval-limit 100` when you want a less noisy local read.
- **Example Local Scores**: current release smoke test at `--eval-limit 100`: ARC Easy `0.3600`, HellaSwag `0.3400` raw / `0.3700` normalized, TruthfulQA MC1 `0.2200`.

### Previous v0.20.4: Inference-Like LTM Training

#### Training/Chat Dynamics Alignment
- **Read-Only LTM Training Mode**: `--ltm-training-mode read-only` carries recurrent state and ROSA token history across TBPTT chunks but skips supervised LTM fast-memory writes. This better matches normal chat inference, where generation does not receive label-gradient inner updates.
- **Assistant Recovery Default**: `--assistant-recovery` now selects `ltm_training_mode=read-only` unless you explicitly pass `--ltm-training-mode inner-update`.
- **Chat Visibility**: chat reports the saved LTM training mode at startup and in `/status`, making read-only/inference-like checkpoints distinguishable from older supervised-inner-update runs.
- **Legacy Titans Path Preserved**: `--ltm-training-mode inner-update` keeps the previous gradient-based fast-memory behavior for experiments that intentionally train with Titans-style inner updates.
- **DeepEmbed No-Decay**: DeepEmbed embeddings are excluded from AdamW decay so their identity initialization is not slowly pulled toward zero during long runs.

### Previous v0.20.3: DeepEmbed Channel-Mix Clamp

#### RWKV Channel-Mix Stabilization
- **DeepEmbed Modulation Clamp**: `--rwkv-channel-mix-deepembed-clamp N` caps DeepEmbed's multiplicative channel-mix modulation before `value_cm`. Default `4.0`; set `0` only for ablations.
- **Shape-Compatible Resume Safety**: this is a runtime numeric guard, not an architecture-shape change. Existing checkpoints load with the default clamp backfilled.
- **Compile-Compatible Hot Path**: the clamp lives inside the RWKV forward path and remains compatible with `torch.compile`, including `max-autotune-no-cudagraphs`.
- **Late-Run NaN Prevention**: this specifically targets rare `key_cm -> relu -> square -> DeepEmbed multiply` spikes that can produce non-finite gradients in `value_cm`, `key_cm`, and upstream projection layers.

### Previous v0.20.2: Channel-Mix Key Clamp

#### RWKV Key Preactivation Stabilization
- **Channel-Mix Key Clamp**: `--rwkv-channel-mix-key-clamp N` caps the RWKV `key_cm` preactivation before ReLU-squared channel mixing. Default `12.0`; set `0` only for ablations.
- **Shape-Compatible Resume Safety**: this is a runtime numeric guard, not an architecture-shape change. Existing checkpoints load with the default clamp backfilled.
- **Compile-Compatible Hot Path**: the clamp lives inside the RWKV forward path and remains compatible with `torch.compile`, including `max-autotune-no-cudagraphs`.

### Previous v0.20.1: Drift Clamp Rescue

#### Commit/Drift Stabilization
- **L2 Drift Norm Clamp**: `--drift-norm-clamp N` caps the worker drift state's total L2 norm. This is more targeted than per-element clamping when many small dimensions collectively create high commitment cost.
- **Drift Delta Scaling**: `--drift-delta-scale F` scales each worker drift update before accumulation. Values such as `0.35-0.75` slow runaway drift while preserving the learned hierarchy.
- **Straight-Through Commit Cap**: commitment auxiliary capping remains bounded in the forward loss, but high raw commit now still receives corrective gradient. This prevents over-cap commit drift from becoming invisible to the penalty.
- **Recovery Guidance**: for commit/loss spikes, resume from a pre-spike checkpoint with lower `--starting-lr`, lower `--ltm-lr`, `--drift-norm-clamp 3.5-4.0`, `--drift-delta-scale 0.35-0.60`, and `--grad-clip 0.75-1.0` rather than resetting `h_module` weights.

### New in v0.20: Assistant SFT Safety Guardrails

#### Large Assistant Training Safety
- **CE Backward Cap Disabled by Default**: `--max-ce-loss-for-backward` now defaults to `0.0`. Leave it disabled for from-scratch language-model training; a finite cap below random-token CE can clamp the loss and silently remove the CE gradient early in training.
- **Response-Preserving Truncation**: Alpaca prompt/completion rows now reserve assistant response tokens when an overlong row is truncated. `--min-response-tokens` defaults to `1`, and `--assistant-recovery` raises it to `16`.
- **Blank Completion Guardrail**: Empty completions are dropped by default so EOS-only labels do not pollute assistant SFT. Use `--allow-empty-completions` only when blank answers are intentional supervised labels.
- **Assistant Loss Weights**: The recovery preset trains prompt tokens at `0.10x`, response tokens at `1.0x`, and the first `32` response tokens at `2.0x`.
- **Resume/Cache Hygiene**: HF token cache keys include loss-weight, formatting, chunking, and response-guardrail settings. Resume hydration no longer persists one-shot refresh flags or stale CE caps from older checkpoints.

#### v0.20 Recovery Preset Values

`--assistant-recovery` applies:

```text
epochs=4
starting_lr=6e-5
min_lr=1e-6
warmup_ratio=0.03
ltm_training_mode=read-only
ltm_lr=3e-4
min_ltm_lr=1e-5
train_prompt_tokens=True
prompt_loss_weight=0.10
response_loss_weight=1.0
response_boundary_loss_weight=2.0
response_boundary_tokens=32
min_response_tokens=16
drop_empty_completions=True
ponder_loss_weight=0.003
memory_gate_warmup_steps=5000
max_ce_loss_for_backward=0.0
```

### 🚀 **New in v0.19.3: Progress Sync Throttling**

#### CUDA Training Loop
- **Throttled Progress Metrics**: `train_step` only returns display metrics on scheduled progress updates, so CUDA scalar `.item()` calls no longer synchronize every batch by default.
- **One Host Contract Audit per Batch**: Right-padding, labels, and loss weights are validated before accelerator transfer, then the checked padding geometry is reused across TBPTT chunks instead of forcing a CUDA-to-host synchronization for every chunk.
- **Configurable Logging Interval**: Use `--progress-log-steps N` to update tqdm scalar metrics every N steps. Default is `25`; use `1` for the old every-step behavior.
- **Non-Destructive Runtime Patch**: This changes only metric reporting cadence. Model architecture, dataset formatting, labels, losses, optimizer steps, and scheduler behavior are unchanged.
- **Alpaca Prompt String Documented**: `--alpaca` uses `### Instruction:`, optional `### Input:`, and `### Response:` formatting before the supervised output text.

### 🚀 **New in v0.19.2: DataLoader Throughput Tuning**

#### CUDA/CPU Input Pipeline
- **Conservative CUDA Auto Workers**: `--num_workers -1` now auto-selects a smaller CUDA worker pool, targeting the common pre-tokenized sweet spot instead of oversubscribing CPU and pinned-memory bandwidth.
- **Worker-Tied Prefetch**: When `--prefetch-factor` is omitted, total queued batches stay bounded relative to worker count. More workers lower per-worker prefetch instead of silently multiplying queued batches.
- **CUDA-Only Pinned Memory**: DataLoader pinning is enabled for CUDA training and avoided for CPU/DirectML paths where it adds overhead without async H2D benefit.
- **Override Still Available**: Use explicit `--num_workers` and `--prefetch-factor` when slow storage, Hugging Face tokenization, or JSON parsing makes the input pipeline the bottleneck.

### 🚀 **New in v0.19: The "Optimization and GUI Update"**

#### Optimization and GUI
- **Internal CUDA Math Switch**: LTM retrieval and memory updates automatically use CUDA-friendly gather/scatter paths on NVIDIA GPUs while preserving the existing CPU-friendly dense math on CPU.
- **No User Flag Required**: The architecture selects the math path internally based on tensor device placement, keeping CLI and GUI configuration simple.
- **ROSA Preserved**: ROSA remains CPU-side and VRAM-light by design.
- **Windows GUI Release Flow**: The README documents the portable GUI bundle workflow for shipping `Hierarchos.exe` with the bundled backend.

#### 🧠 Architecture
- **RWKV v8 Backbone**: Replaced GRU cells with full RWKV v8 (Receptance Weighted Key Value) cells featuring linear attention, Time Mixing with WKV recurrence, and ReLU-squared Channel Mixing.
- **DeepEmbed (4x Scale)**: New learnable token embeddings at 4× hidden dimension that gate the RWKV channel mixing FFN, providing richer per-token modulation.
- **ROSA (Rapid Online Suffix Automaton)**: A neurosymbolic inner monologue — a CPU-side Suffix Automaton predicts likely next tokens, which are embedded and added to the input representation. Gives the model a "heads up" about upcoming patterns.
- **V7 Backward Compatibility**: Set `use_deepembed=False, use_rosa=False` in config to run in pure V7 mode. All V7 checkpoints load cleanly.

#### ⚡ CUDA Datacenter Optimizations (Zero Config)
- **Auto-AMP**: Mixed precision auto-enables on CUDA — no `--amp` flag needed.
- **bfloat16 on Ampere+**: SM ≥ 8.0 GPUs automatically use bfloat16 (better dynamic range, no GradScaler overhead).
- **TF32 Matmul**: 3-8× faster linear layers on Ampere+ GPUs, enabled automatically.
- **cuDNN Benchmark**: Auto-tunes convolution kernels for hardware.
- **torch.compile Auto-Enable**: Worker loop compiled on CUDA (no Windows CPU hang issue).
- **Non-blocking Transfers**: Host-to-device copies overlap with GPU computation via `non_blocking=True`.
- **Pinned Memory**: DataLoader uses `pin_memory=True` for CUDA training.
- **Bounded DataLoader Prefetch**: Auto prefetch is tied to worker count to avoid over-queuing pinned batches.
- **`--no-amp` Flag**: Explicitly disable AMP if needed.

#### 🧪 Test Suite
- **226 Tests Pass**: Validation covers gradient flow, recurrent state continuity, strict checkpoint loading, train/chat logit parity, LTM update and filtering behavior, sampling, token-cache integrity, evaluator scoring, and forward/backward execution. Three environment-specific tests are skipped on this Windows CPU test host.
- **Self-Contained Tests**: All tests create models in-memory — no hardcoded checkpoint paths.


## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

Hierarchos combines recurrent sequence modeling, hierarchical iterative
reasoning, and structured long-term memory. With Alpha v0.30, the project also
makes a backend-level commitment: the primary implementation is now the
framework-free **Hierarchos Native** stack, with Rust for orchestration and
inference and Vulkan compute for full-model training. The Python/PyTorch codebase
remains valuable as a reference implementation and compatibility surface, but it
is no longer the architectural center of the project.

## Core Concepts

Hierarchos v0.30 is organized around four layers:

🔄 **RWKV-v9 Backbone (The Neural Engine)**

The coherent-v9 core uses corrected matrix-state recurrence, ReLU-squared
channel mixing, shared factorized token adapters, bounded ROSA features, and
row-local recurrent refinement. It preserves constant-size recurrent generation
state rather than growing a Transformer-style KV cache with sequence length.

🧠 **Titans-Inspired Memory (The Cognitive Substrate)**

A structured long-term-memory workspace provides learned retrieval plus bounded,
transactional adaptation. Memory slots carry explicit metadata and the v0.30
contract separates training a writer from allowing that writer to mutate live
conversation state.

⚙️ **Hierarchical Reasoning (The Cognitive Process)**

A high-level Manager and low-level Worker add adaptive iterative computation.
Hard per-row ACT selects committed states, while bounded convergence and
commitment controls keep the recurrent process explicit and testable.

🦀 **Rust + Vulkan (The Execution Layer)**

The primary backend is not a Python launcher. Rust owns the native CLI, GUI,
tokenization, package lifecycle, and inference runtime; Vulkan compute owns the
training hot path. Canonical SafeTensors keep the model portable across the
native runtime and compatible external implementations.

## Architecture Diagram

```text
┌────────────────────────────────────────────────────────────────────┐
│ Rust tokenizer / canonical model package                           │
│                    ↓                                               │
│ tied token embedding + shared H/L/ROSA adapters                    │
│                    ↓                                               │
│ metadata-safe LTM retrieval + learned value path                   │
│                    ↓                                               │
│ Manager H recurrence (RWKV-v9 matrix state + hard per-row ACT)     │
│                    ↓                                               │
│ Worker L recurrence (RWKV-v9 row-local iterative refinement)       │
│                    ↓                                               │
│ out_norm → tied lm_head → logits                                   │
│                    │                                               │
│          ┌─────────┴─────────┐                                     │
│          ↓                   ↓                                     │
│  Rust inference/chat   Vulkan training graph                       │
│  recurrent/LTM state   CE + auxiliaries → backward → AdamW         │
│          │                   │                                     │
│          └─────────┬─────────┘                                     │
│                    ↓                                               │
│      canonical SafeTensors + portable resume state                 │
└────────────────────────────────────────────────────────────────────┘
```

## Features ✨

### Primary native capabilities

  * 🦀 **Framework-Free Rust Runtime**: Native package handling, tokenization, inference, chat, CLI orchestration, and the dedicated GUI run without Python, PyTorch, `tch`, `pyo3`, or libtorch.
  * 🌋 **Full-Model Vulkan Training**: Coherent-v9 forward/backward execution, recurrent replay, gradient accumulation, AdamW, mixed-storage precision policies, checkpointing, and device selection run through Vulkan compute.
  * 🪟 **Native Windows Release**: `tools/build_native_release.ps1` produces `HierarchosNative.exe`, `HierarchosCLI.exe`, and the Vulkan trainer/device tools as an isolated distribution with dependency and runtime-artifact audits.
  * 💾 **Exact Native Resume**: Restores weights, AdamW slots/clocks, scheduler and loss-scaler state, pending accumulated gradients, data cursor/shuffle state, and portable recurrent/LTM/ROSA replay state.
  * 🌐 **Rust Hugging Face Transport**: Supported canonical model/tokenizer assets and JSONL/NDJSON datasets can be fetched directly over Rust HTTPS without `huggingface_hub` or Python.
  * 📦 **Canonical SafeTensors ABI**: Vulkan training retains FP32 master parameters and writes packages that load directly in `hierarchos-inference` and remain consumable by compatible external implementations.
  * 🎛️ **Explicit Precision Policies**: Native execution exposes capability-checked `fp32`, mixed FP16-storage/FP32-compute, parity, and qualified FP16 LM-backward modes rather than framework autocast.
  * 🔄 **RWKV-v9 Coherent Core**: Corrected matrix-state recurrence, hard per-row ACT, bounded ROSA, shared factorized token adapters, and row-local refinement are the model contract for new Alpha v0.30 runs.

### Model and framework-reference capabilities
  * 📊 **Integrated Benchmarking**: Optional support for `lm-evaluation-harness`. Track model accuracy on standard benchmarks (HellaSwag, ARC, etc.) during or after training with `--eval-tasks`, or use `--benchmark-preset rog-ally` for a bounded local CPU smoke test.
  * 🎮 **AMD GPU Support (DirectML/ZLUDA)**: Train on AMD Radeon GPUs using DirectML backend on Windows. Opt-in via `--device dml` with automatic compatibility handling and optimized fallbacks.
  * 🎓 **Proper Temporal Learning**: Configurable truncated BPTT (`--detach-every-n-steps`) enables learning across multiple timesteps while managing memory. Default 32-step gradients flow allows the model to **learn temporal dependencies** effectively.
  * 🔗 **Exact Per-Sample Gradient Connectivity**: `--full-sample-bptt` connects Manager, Worker, routed memory, and recurrent state across every retained token in a sample. It removes deliberate TBPTT truncation but does not guarantee semantic coherence or eliminate numerical failures.
  * 🎯 **Train/Test Consistency**: Fixes train/test mismatch from unconditional state detachment, improving model coherence and stability.
  * 🌐 **Hugging Face `datasets` Integration**: Load datasets directly from the HF Hub or local paths in various formats (CSV, Parquet, JSON, etc.) using `--hf_dataset`.
  * 💾 **Optimized Consolidated Chunk Loading**: Dramatically reduces RAM usage and speeds up training startup for large datasets using pre-processed, consolidated `.pt` tensor files and a manifest (`--pre_pt_dataset`). Includes file caching for efficiency.
  * 📜 **Iterable Dataset Support**: Option to load pre-chunked JSONL datasets line-by-line (`--pre_chunked_dataset`) for minimal memory overhead during training.
  * ✂️ **Dataset Consolidation Script (`dataset_chunk_create.py`)**: Enhanced tool to prepare large datasets, chunking them into **consolidated `.pt` files** and creating a `manifest.jsonl` for efficient loading.
  * 📉 **Gradient Checkpointing**: Significantly reduces VRAM usage during training/fine-tuning (`--gradient-checkpointing`), enabling larger models or batches on memory-constrained hardware.
  * 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  * 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  * 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  * 🦀 **Native Rust/Vulkan Runtime**: `hierarchos-native-cli`,
    `hierarchos-inference`, and `hierarchos-vulkan` form an independent
    Python-free path. Inference/chat and package handling are Rust; training and
    optimization execute through Vulkan compute. Canonical FP32-master
    SafeTensors are the interchange boundary with other runtimes.
  * 🚀 **PyTorch 2.4+ Framework Runtime Contract**: The separate
    Python/framework path requires PyTorch 2.4 or newer for restricted artifact
    loading; CUDA builds can use `torch.compile` with `--compile` /
    `--force-compile`.
  * 🛡️ **Training Guardrails**: Finite-gradient rejection, gradient clipping (`--grad-clip`), Z-loss regularization, and state/activation clamps reduce instability risk. Clamps intentionally alter values when triggered and cannot guarantee convergence.
  * 📦 **Self-Contained & Portable Models**: Models are saved as HuggingFace-style directories containing weights, tokenizer, and architecture config for easy sharing and deployment.
  * 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones.
  * 🛡️ **Fail-Closed Deployment Contract**: The Python/framework loader accepts
    framework checkpoints only through PyTorch's restricted `weights_only` path
    with narrowly allowlisted project types. The native loader accepts canonical
    SafeTensors packages and does not deserialize framework-object `.pt` files.
    Legacy scalar-RWKV `.npz`, automatic re-quantization, and the old scalar
    Vulkan inference path remain unsupported for the Alpha v0.30 RWKV-v9 core.
  * 🐍 **Framework Runtime Versions**: Python 3.10+ and PyTorch 2.4+ are required
    only for the Python/framework path. Python 3.13 is usable only with a
    compatible PyTorch build; DirectML currently requires Python 3.10–3.12.
    The native Rust/Vulkan binaries do not require Python or PyTorch at runtime.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

Choose the runtime you intend to use:

  * **Native Rust/Vulkan:** a stable Rust toolchain plus a working Vulkan loader
    and driver. Build the isolated package with
    `powershell -ExecutionPolicy Bypass -File .\tools\build_native_release.ps1`.
    Python, PyTorch, CUDA, libtorch, and `pyo3` are not runtime dependencies of
    the native CLI/trainer/inference crates. See [NATIVE_BACKEND.md](NATIVE_BACKEND.md).
  * **Python/framework:** Python 3.10+ and PyTorch 2.4+ are required by the
    fail-closed framework checkpoint/cache loaders.
  * **For Hugging Face Datasets on the framework path:** `pip install datasets`.
  * **For AMD framework training (Windows):** Install DirectML via
    `pip install torch-directml` and follow [README_ZLUDA.md](README_ZLUDA.md).
    Its current Python support is 3.10–3.12; do not use a Python 3.13 DirectML
    environment.
  * **Optional framework CUDA training:** NVIDIA GPU with CUDA support (Compute
    Capability 7.0+ recommended) and a PyTorch build with CUDA enabled.

### Native installation (recommended)

1. **Clone the repository:**

   ```powershell
   git clone https://github.com/your-username/Hierarchos.git
   cd Hierarchos
   ```

2. **Build the isolated Rust/Vulkan release:**

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\tools\build_native_release.ps1
   ```

3. **Verify Vulkan device discovery:**

   ```powershell
   .\dist\Hierarchos-Native\HierarchosCLI.exe devices
   ```

4. **Launch the native GUI or use the native CLI:**

   ```powershell
   .\dist\Hierarchos-Native\HierarchosNative.exe
   .\dist\Hierarchos-Native\HierarchosCLI.exe --help
   ```

No Python environment is required for this path. For a CI/headless build host,
the release builder supports `-SkipDeviceProbe`; use `-SkipTests` only when the
same source revision has already passed the native gates.

### Framework/Python installation (reference and compatibility workflows)

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Hierarchos.git
    cd Hierarchos
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\Activate
    # On Linux/macOS
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**

      * **Core (required for supported full-precision training/chat):**
        ```bash
        pip install -r core_requirements.txt
        ```
      * **Full (includes datasets, LoRA, and development dependencies):**
        ```bash
        pip install -r requirements_kernel.txt
        ```
      * **DirectML (AMD GPU on Windows):**
        ```bash
        pip install -r requirements_dml.txt
        ```

    *(Note: `requirements_kernel.txt` includes `datasets`)*

4.  **Compile Historical C++ Kernel (Developer Reference Only):**
    This step is not required or supported for Alpha v0.30 inference:

    ```bash
    # Ensure you have CMake, a C++ compiler, and installed dependencies from requirements_kernel.txt
    # On Windows
    setup.bat
    # On Linux/macOS
    bash setup.sh
    ```

    This creates `Hierarchos_matmul.*` in your project root for legacy
    experiments. It does not enable Alpha v0.30 RWKV-v9 quantization: current `.npz`
    export, quantized chat, and Vulkan model inference are intentionally
    unsupported because the old kernel path does not reproduce the active
    matrix-state recurrence.

-----

## 📚 User Guide: Comprehensive Workflows

This guide covers common scenarios from data preparation to inference.

### Choosing Your Entry Point

> [!TIP]
> Start with `HierarchosCLI.exe` / `hierarchos-native-cli` for Alpha v0.30. The
> Python CLI is retained for framework-specific research and compatibility work.
> If you do use the Python stack, `hierarchos_cli.py` is its supported modular
> entry point; the original `hierarchos.py` is legacy and unmaintained.

| Entry Point | Status | Description |
|-------------|--------|-------------|
| `HierarchosCLI.exe` / `hierarchos-native-cli` | ✅ **Primary / Recommended** | Pure-Rust CLI for coherent-v9 package management, inference/chat, and full-model Vulkan training; no Python/PyTorch runtime dispatcher. |
| `HierarchosNative.exe` | ✅ **Primary GUI** | Dedicated Rust GUI using `hierarchos-inference` and the Vulkan trainer directly. |
| `hierarchos_cli.py` | ✅ **Framework Reference** | Supported modular Python/PyTorch CLI for framework-only research and compatibility workflows. |
| `hierarchos.py` | ⚠️ **Legacy** | Unmaintained monolith (5,600 lines). Kept only as reference for agentic AI workflows. | <-- DO NOT USE THIS! ITS 16 VERSIONS OUT OF DATE!!

> **Product version boundary:** Hierarchos Alpha v0.30 is the current release.
> Fresh Alpha v0.30 configs use the internal `coherent-v9` identifier for the
> RWKV-v9 core. Use the [RWKV-v9 migration notes](COHERENT_V9_MIGRATION.md) for
> its preflight and release-gate command. Examples or performance figures below
> that explicitly say `232M`, `232.5M`, or `95` state tensors describe the
> historical RWKV-v8 core (`legacy-v8`) unless they say otherwise.

**Native example:**

```powershell
.\dist\Hierarchos-Native\HierarchosCLI.exe train `
    --tokenizer-path .\tokenizer_assets `
    --train .\instruct_dataset.jsonl `
    --out-dir .\hierarchos_vulkan_model `
    --epochs 3 `
    --batch_size 4 `
    --training-chunk-size 256 `
    --precision fp32 `
    --device-index 0
```

**Framework-reference example:**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
    --out-dir "./my_model" \
    --epochs 3 \
    --force-compile
```


### Workflow 1: Training a New Model

Choose **one** data source option:

**(A) Local JSON/JSONL File (Fits in RAM):**

```bash
python hierarchos_cli.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" `# Or your preferred tokenizer` \
    --out-dir "./my_Hierarchos_model" \
    --epochs 3 \
    --batch_size 64 \
    --accumulation-steps 1 \
    --auto-max-length `# Automatically determines max sequence length` \
    --context_dim 448 `# ~30.2M Alpha v0.30/RWKV-v9 params; RWKV-v8 was ~232.5M` \
    --h_hidden 448 \
    --l_hidden 448 \
    --rwkv-head-size 64 \
    --max_h_steps 5 \
    --max_l_steps 5 \
    --amp `# Auto-enabled on CUDA; explicit here for clarity`
```

**(B) Hugging Face Dataset (Text Completion):**

```bash
python hierarchos_cli.py train \
    --hf_dataset "wikitext" \
    --hf_dataset_config "wikitext-2-raw-v1" \
    --hf_dataset_split "train" \
    --text_column "text" `# Column containing the text` \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_wikitext_model" \
    --epochs 1 \
    --batch_size 2 \
    --accumulation-steps 4 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(C) Hugging Face Dataset (Instruction/Alpaca/Kayla Format):**

```bash
python hierarchos_cli.py train \
    --hf_dataset "databricks/databricks-dolly-15k" \
    --prompt_column "Instruction" \
    --completion_column "output" \
    # --alpaca # Add for instruction/input/output datasets; defaults columns to instruction/output \
    # --kayla # Add if your HF data structure matches Kayla format (instruction, output, thought-process, feelings) \
    # --text_column "context" # Example: Map 'context' field if needed for your format \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_dolly_model" \
    --epochs 2 \
    --batch_size 1 \
    --accumulation-steps 8 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(D) Pre-Chunked Local Dataset (Very Large Dataset):**

  * **Step 1: Create Chunks**
    ```bash
    python dataset_chunk_create.py \
        --dataset "path/to/very_large_data.jsonl" \
        --tokenizer-path "openai-community/gpt2" \
        --output-dir "./very_large_data_chunked" \
        --overlap 512 \
        --chunks-per-file 1000
    # Note the MAX_SEQ_LENGTH printed by the script (e.g., 3153)
    ```
  * **Step 2: Train using Chunks**
    ```bash
    python hierarchos_cli.py train \
        --pre_pt_dataset `# Enable loading via manifest` \
        --train "./very_large_data_chunked" `# Directory with .pt files & manifest` \
        --max_length 3153 `# MUST match chunker output` \
        --tokenizer-path "openai-community/gpt2" `# Still needed for model init` \
        --out-dir "./my_large_model" \
        --epochs 1 \
        --batch_size 1 \
        --accumulation-steps 8 \
        --amp \
        --gradient-checkpointing # Add this if VRAM is limited
    ```

**(E) Training on AMD GPU (DirectML/Windows):**

```bash
python hierarchos_cli.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_amd_model" \
    --device dml `# Explicitly enable DirectML` \
    --epochs 3 \
    --batch_size 2 \
    --accumulation-steps 4 \
    --auto-max-length \
    --gradient-checkpointing # Recommended for AMD GPUs
```

-----

💡 **CUDA Auto-Optimization:** On NVIDIA GPUs, AMP, TF32, cuDNN benchmark, and torch.compile are **auto-enabled**. Long CUDA runs default to `--compile-mode max-autotune-no-cudagraphs`, static RWKV worker-loop capture, and H-RNN cell compilation. This keeps autotuned CUDA kernels while avoiding CUDA graph fast-path warnings from TBPTT submodule calls. Use `--compile-mode max-autotune --compile-cudagraphs` only when benchmarking CUDA graphs explicitly.
💾 **Training on Low Memory:** For TBPTT/finetuning, `--gradient-checkpointing` reduces VRAM at the cost of recomputation. Exact full-sample training instead uses `--full-sample-activation-checkpointing`; lower `--full-sample-checkpoint-segment-size` when exact-mode VRAM is tight.
🎮 **AMD GPU Training:** Use `--device dml` to train on AMD Radeon GPUs via DirectML. AMP is automatically disabled for stability.
🚀 **Datacenter Training:** Alpha v0.30's RWKV-v9 `448/448/448` reference is
30.2M parameters, not the 232M RWKV-v8 reference. Start from the Alpha v0.30
command in the [RWKV-v9 migration notes](COHERENT_V9_MIGRATION.md), then size batch,
activation-checkpoint segments, and compilation settings from a measured
maximum-length preflight on the actual GPU. The old 96GB Blackwell `232M`
profile is a historical planning reference, not an Alpha v0.30 capacity guarantee.

## ⚠️ **HRM Convergence & Training Speed:** Higher `--max_h_steps` and `--max_l_steps` allow deeper reasoning but **significantly increase training time** per batch due to the iterative HRM process. Adjust based on your task and compute resources.

The HF cache builder pins a mutable Hub branch/tag to its resolved commit SHA, parallelizes Arrow preparation, fetches Arrow rows and fast-tokenizes them in batches, and lets completed worker batches arrive out of FIFO order because every row is materialized before training. The runtime cache is mmap-backed, reads each sampled batch in file-offset order, retains the final incomplete batch, and overlaps pinned nonblocking transfers on a dedicated CUDA stream. These change storage and scheduling only; token IDs, sample boundaries, loss weights, and model equations are preserved.

GPT-2-sized vocabularies use compact uint16 input/ROSA streams and exact float32-palette RLE loss weights. With `--train-prompt-tokens`, the writer also verifies that every real label equals its input ID and stores labels as a checked alias instead of a duplicate stream. For the weighted+ROSA profile this is about 4 data bytes/token, so 5B tokens require about 20 GB decimal (18.6 GiB) for `tokens.bin`, plus the run/index metadata and Hugging Face source cache. Without label aliasing it is about 30 GB; the previous int32+inline-fp16 representation was about 70 GB. Put the cache root on instance-local NVMe when the rental exposes it.

### Exact Per-Sample Gradient Connectivity (Full BPTT)

Use `--full-sample-bptt` when every supervised token in a sample must remain connected to one autograd graph. The trainer disables recurrent detachment and cross-sample state persistence. By default it partitions that graph into attached 128-token activation-checkpoint segments, carries differentiable H/L/context state across every boundary, composes the global sample loss once, and performs one backward pass after the last segment. The safe default remains `128`; the explicit v0.21 profile for batch `64`, `max_length=8880`, and a 96GB RTX PRO 6000 recommends `224` only after the rental passes a maximum-length memory preflight. This follows PyTorch's recommended explicit `use_reentrant=False` checkpoint API and the standard recomputation-for-memory method described by Chen et al.: [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html), [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).

The current forward contains no dropout or random sampling, so exact checkpoint segments disable RNG-state save/restore overhead. Direct-versus-segmented tests compare the complete recurrent state, objective, and every parameter gradient; a future stochastic forward must re-enable RNG preservation before using this optimization.

`--no-persist-state` refers only to model H/L/context/LTM values crossing from one shuffled training batch into unrelated samples. It does not disable hierarchical drift, and it is unrelated to PyTorch DataLoader `persistent_workers`; CUDA loader workers remain persistent and prefetched for throughput. Drift is still computed, trained, and clamped at every worker step. At an activation-only segment boundary, the attached L-state derives the next drift exactly as an uninterrupted forward would, so the previous segment's terminal drift is not injected a second time.

When an exact checkpoint is resumed with `--no-full-sample-bptt`, the saved inference contract records TBPTT recurrence geometry separately from refinement parity. Chat and evaluation therefore restore `training_chunk_size` prefill boundaries and the matching boundary-drift behavior, while retaining the fixed Manager/Worker refinement policy selected by `inference_logit_parity`. This prevents a budget-driven exact-to-TBPTT continuation from silently training with chunk boundaries but serving with a different monolithic recurrence.

The configured `--training-chunk-size` is deliberately left unchanged. It remains cache/ROSA/LTM-decay/compile geometry, so enabling full-sample BPTT does **not** invalidate an existing multi-billion-token token cache. `--full-sample-checkpoint-segment-size` controls only recomputation boundaries; it never detaches state or changes the forward objective. There is still one attached gradient graph across the complete active sample.

Segment length is therefore a speed/VRAM control, not a gradient-horizon control. At the worst-case compiled length of `8,960`, segment `128` creates `70` attached checkpoints, `224` creates exactly `40`, and `256` creates `35`. Relative to `256`, `224` reduces the activation-dominated segment length by `12.5%` while adding only a small amount of retained boundary state. Relative to `128`, it removes `30` checkpoint invocations but rematerializes a `1.75x` larger segment. The exact optimum remains GPU/runtime dependent; judge steady-state throughput only after compilation and test peak memory on a genuinely maximum-length batch.

```bash
python hierarchos_cli.py train \
    --model-path "./pre_chat_model/hierarchos_epoch_4.pt" \
    --tokenizer-path "./pre_chat_model" \
    --hf_dataset "your_org/larger_dataset" \
    --out-dir "./full_bptt_continuation" \
    --max_length 8880 \
    --training-chunk-size 256 \
    --full-sample-bptt \
    --full-sample-checkpoint-segment-size 224 \
    --batch_size 64 \
    --accumulation-steps 1
```

`--model-path` is a weights-only continuation: it loads the older tensors unchanged, starts a fresh optimizer/schedule for the larger dataset, and leaves the source checkpoint untouched. Keep the original tokenizer/vocabulary. The attached 224-token checkpoint segments target a practical speed/headroom balance without retaining 8,880 tokens of activations at once. The audited epoch-13 checkpoint used a real 256-token TBPTT boundary; on the same hardware and tokens, budget approximately `1.25-1.45x` its wall time (`70-80%` throughput) for checkpointed full BPTT at `224`, while treating those figures as planning estimates until the rental is measured. If the rental OOMs at batch 64, lower only the checkpoint segment to `128`, then `64`; this preserves the mathematical objective and full gradient horizon, although floating-point roundoff can differ slightly. If `224` leaves ample measured headroom, `256` is the next speed trial. An OOM never silently falls back to truncated gradients.

Adding more dataset rows does not itself consume more VRAM. Shorter conversations reduce typical padding and memory, but the longest batch still determines the peak; a dataset with no row above the previous `max_length` can still produce a full `64 x 8,960` compiled batch. Loader prefetch and compact input tensors are small compared with recurrent activations, so do not use average short-sample memory to certify the longest bucket.

Per-token ROSA/LTM routing remains active. The isolated one-forward mode skips only the terminal post-backward LTM fast-memory write: that write cannot affect the sample that produced it and would be reset before the next sample. Ordinary routed parameter gradients and optional LTM value-alignment gradients are unchanged.

Validate the rented Blackwell runtime before starting the paid dataset pass:

```bash
python benchmark_full_sample_bptt.py \
    --device cuda \
    --amp \
    --require-blackwell \
    --batch-size 64 \
    --sequence-length 1024 \
    --checkpoint-segment-size 224 \
    --repeats 5
```

The benchmark uses synchronized device timing and `torch.cuda.max_memory_allocated()` peak tracking. `--require-blackwell` rejects a runtime unless the visible GPU reports compute capability 12.0 and the PyTorch build advertises `sm_120`, preventing an incompatible wheel from consuming rental time. It uses a deliberately small parity model, so it validates CUDA compatibility and checkpoint behavior but does **not** certify production `448/448/448`, batch-64, 8,880-token VRAM. Measure that peak with the actual profile before committing the full rental budget.

### Blackwell Precision Contract

Keep AMP mixed precision enabled. On supported Blackwell CUDA builds, Hierarchos selects BF16 autocast for eligible Tensor Core work while keeping FP32 parameters, gradients, AdamW moments, recurrent matrix state, LTM updates, ACT weighting, and language-loss accumulation. PyTorch explicitly recommends leaving the model in default precision under autocast rather than calling `model.half()` or `model.bfloat16()`: [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html).

For the historical 232.5M legacy-v8 profile, pure FP16/BF16 model and optimizer state would save only about `1.73 GiB` and would not remove checkpoint recomputation. At the v0.21 continuation LR (`1e-5` down to `1e-7`), low-precision parameter storage can round updates away; pure FP16 also risks second-moment underflow. Quantization-aware training is intended to adapt a model for later quantized serving, not to accelerate this exact pretraining loop: [TorchAO QAT documentation](https://docs.pytorch.org/ao/stable/workflows/qat.html). Hierarchos v0.21 implements neither a QAT training flag nor a validated FP8 recipe; FP8 remains a separate architecture-specific experiment for eligible linear layers: [TorchAO quantized-training documentation](https://docs.pytorch.org/ao/stable/workflows/training.html).

### Historical Legacy-v8 Assistant SFT Profile for 232M / ~1B Tokens

For a 232M `448/448/448` assistant run on a large Alpaca-style instruction dataset, start with the assistant recovery preset. Explicit CLI values override the preset, so a full-budget 10-epoch run can keep the v0.20 safety settings while setting `--epochs 10`:

```bash
python hierarchos_cli.py train \
    --hf_dataset "your_org/assistant_dataset" \
    --hf_dataset_split "train" \
    --alpaca \
    --assistant-recovery \
    --epochs 10 \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./hierarchos_232m_assistant" \
    --context_dim 448 \
    --h_hidden 448 \
    --l_hidden 448 \
    --rwkv-head-size 64 \
    --max_length 8880 \
    --training-chunk-size 256 \
    --full-sample-bptt \
    --full-sample-activation-checkpointing \
    --full-sample-checkpoint-segment-size 224 \
    --detach-every-n-steps 0 \
    --no-persist-state \
    --batch_size 64 \
    --accumulation-steps 1 \
    --hf-token-cache \
    --max-ce-loss-for-backward 0 \
    --save-steps 600 \
    --padding-metric-steps 50
```

`--assistant-recovery` keeps prompt tokens in the language-model objective but downweights them to `0.10`, weights assistant response tokens at `1.0`, boosts the first `32` response tokens by `2.0x`, uses warmup+cosine LR, lowers the ponder penalty to `0.003`, lengthens memory-gate warmup to `5000` steps, and reserves at least `16` answer tokens when overlong prompts are truncated. The added v0.21 flags make this an exact per-sample gradient-connectivity profile; segment `224` targets a 96GB rental but must pass a maximum-length preflight there and should be reduced on smaller cards. Blank completions are dropped by default; pass `--allow-empty-completions` only if empty answers are intentional labels. Keep `--max-ce-loss-for-backward 0` for from-scratch LM training. The default 4-epoch preset gives a 232.5M model about 17.2 training tokens per parameter on a 1B-token dataset; `--epochs 10` is about 43 training tokens per parameter.

#### v0.20.4 Stability Rescue Resume

If a long assistant run develops the pattern `loss up + ponder up + commit up`, shows rare non-finite gradient skips in the RWKV channel-mix path, or produces poor chat coherence despite sane loss, do not reset learned hierarchy weights first. Resume from a pre-spike or latest clean checkpoint with a cooler schedule, bounded drift, channel-mix clamps, and inference-like LTM training:

```bash
python hierarchos_cli.py train \
    --resume-from-ckpt "./chatHRM/hierarchos_epoch_4_step_6600.pt" \
    --epochs 8 \
    --rebuild-lr-schedule \
    --starting-lr 4e-5 \
    --min-lr 1e-9 \
    --warmup-ratio 0.0 \
    --ltm-lr 2e-5 \
    --min-ltm-lr 1e-9 \
    --ltm-training-mode read-only \
    --adaptive-ponder \
    --ponder-target-scale 0.65 \
    --ponder-loss-weight 0.003 \
    --commitment-loss-weight 0.5 \
    --max-commitment-cost-for-backward 4.0 \
    --drift-state-clamp 2.0 \
    --drift-norm-clamp 4.0 \
    --drift-delta-scale 0.50 \
    --rwkv-channel-mix-key-clamp 12.0 \
    --rwkv-channel-mix-deepembed-clamp 4.0 \
    --grad-clip 0.75 \
    --save-steps 600
```

`--rebuild-lr-schedule` preserves AdamW/scaler moments while intentionally
changing only the remaining LR schedule. Keep the same dataset, tokenizer,
architecture dimensions, Alpaca/assistant-recovery flags, response-loss
weights, HF token cache, and compile flags from the original run. If the first
rescue attempt still spends multiple checkpoint intervals above `loss=3.5`
with commit above `25`, fall back to an earlier checkpoint and reduce
`--starting-lr`, `--ltm-lr`, and `--drift-delta-scale` another step.

### Workflow 2: Fine-Tuning with LoRA

Adapt a pre-trained model using new data (any supported format).

```bash
python hierarchos_cli.py finetune \
    --model-path "./my_Hierarchos_model" `# Path to your trained base model` \
    --hf_dataset "squad" `# Example: Use SQuAD for QA fine-tuning` \
    --prompt_column "question" \
    --completion_column "answers" `# Might need custom processing depending on format` \
    --text_column "context" `# Use context as part of the prompt` \
    --out-dir "./my_squad_lora" \
    --epochs 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --amp \
    --gradient-checkpointing `# Use if fine-tuning large models on limited VRAM`
```

Fine-tuning writes a local PEFT LoRA adapter in `safetensors` format together
with a Hierarchos adapter manifest. The manifest binds the adapter to the
source architecture contract, tokenizer fingerprint, LoRA geometry, and file
hashes. Treat the adapter as a delta for that exact base model, not as a
standalone chat model or an exact-resume training checkpoint. Pickle-based
adapter weights and adapters without the bound manifest are rejected.

### Workflow 3: Merging LoRA Adapter

Combine the base model and the LoRA adapter into a new, standalone model.

```bash
python hierarchos_cli.py merge-lora \
    --model-path "./my_Hierarchos_model" \
    --lora-adapter-path "./my_squad_lora" \
    --out-dir "./my_model_merged_squad"
```

`merge-lora` verifies the adapter manifest, adapter hashes, architecture
contract, and tokenizer identity before asking PEFT for its checked
`safe_merge=True` path. It validates the merged full-precision state and
publishes the standalone package atomically. An existing output directory is
refused unless `--overwrite-merge-output` is explicit. A merged package is for
inference or a fresh weights-only continuation; it does not contain the
optimizer, scheduler, sampler cursor, or gradient state required for exact
training resume.

### Workflow 4: Quantization *(Currently Unsupported)*

The former `.npz` exporter and CPU/Vulkan loader implement an older
scalar-state RWKV model and omit parts of the active manager/worker,
DeepEmbed/ROSA, ACT, and state contract. The modular CLI therefore rejects
quantized export and loading. Use a full-precision `.pt` checkpoint until a
matrix-state RWKV-v9 quantized implementation has its own complete
recurrent-state and logit-parity validation. It is not a supported release
path today.

### Workflow 5: Running Chat Inference

Interact with your trained or fine-tuned model.

**Full Precision:**

```bash
python hierarchos_cli.py chat --model-path "./my_model_merged_squad"
```

> ⚠️ **Important for Alpaca-Trained Models:** If you trained on instruction datasets like Alpaca, your model expects **instruction-formatted prompts**, not casual conversation. See "Using Your Trained Model" section below.

**Recommended KortexHOS / Alpaca Assistant Release Profile:**

```bash
python hierarchos_cli.py chat \
    --model-path "./chatHRM" \
    --temperature 0.4 \
    --top-k 40 \
    --top-p 0.9 \
    --repetition-penalty 1.15 \
    --max-new-tokens 256 \
    --no-passive-learning \
    --chat-input-history-turns 0
```

Use this profile as the static coherence baseline before enabling chat history or passive learning. It keeps inference close to the training/benchmark path: no passive LTM writes and no previous-turn text injected into the Alpaca `### Input:` field. Exact v0.21 checkpoints default to monolithic prompt prefill, fixed refinement policy, and no external drift seed at artificial boundaries; legacy TBPTT checkpoints retain their saved chunk-boundary behavior. Leave recurrent chat-state carry disabled unless you are intentionally testing multi-turn stateful behavior. Saved recurrent chat states are model/tokenizer-bound continuation artifacts, not portable model-neutral conversation files. Full-precision parity means the same checkpoint, tokenizer, runtime policy, dtype/backend, and state boundaries within floating tolerance—not bitwise equality across different hardware, AMP versus FP32, sampling, or quantization.

If a model directory contains both a stale `.npz` file and a coherent
full-precision checkpoint, chat ignores the unsupported archive and loads the
full-precision checkpoint. A directory containing only `.npz` artifacts is
rejected.

Tokenizer repositories do not execute remote Python by default. Use
`--trust-remote-code` only for a tokenizer repository whose code you have
reviewed and explicitly trust.

### Workflow 6: Resuming Interrupted Training

Continue a `train` run from a saved checkpoint (`.pt` file).

```bash
python hierarchos_cli.py train \
    # Dataset args might be loaded from checkpoint, specify only if needed \
    --out-dir "./my_large_model" \
    --resume-from-ckpt "./my_large_model/Hierarchos_epoch_1.pt" \
    --epochs 3 `# Total desired epochs` \
    --amp \
    --gradient-checkpointing # Ensure flag is consistent with the resumed run if needed
```

  * A normal exact resume restores optimizer, scaler, scheduler, pending
    gradients, RNG, and sampler state. It restores the saved LR curve rather
    than starting warmup again.
  * Schedule-defining settings (including start/min LR, warmup, and LTM
    schedule settings) are part of the exact-run identity. A conflicting CLI
    value is rejected instead of silently changing the next update.
  * Use `--rebuild-lr-schedule` only when intentionally changing the remaining
    schedule while keeping optimizer/scaler moments. `--reset-optimizer-state`
    is a separate fresh-moments experiment.
  * Exact mid-epoch replay requires a proven immutable random-access source and
    sampler cursor. Mutable or non-replayable iterable sources fail closed
    rather than skip or reorder batches.
  * `--override-scheduling` is the legacy combined reset and also discards
    optimizer/scaler state; do not use it accidentally.

### Workflow 7: Expanding a Model *(Requires `expand_model.py`)*

Create a larger model within the same authenticated architecture revision and
initialize its compatible subspaces from a smaller trained model.

```bash
python expand_model.py \
    --old-model-path "./my_Hierarchos_model" \
    --output-dir "./expanded_model" \
    --context-dim 1024 \
    --persistent-dim 256 \
    --ltm-slots 4096 \
    --ltm-key-dim 256 \
    --ltm-val-dim 256 \
    --rwkv-head-size 64
```

Expansion now produces one atomic model package containing `hierarchos.pt`, its
SHA-256 sidecar, the resolved config, the exact tokenizer, and hashed expansion
provenance. Concatenated projections are transplanted by semantic block, so
token, context, persistent, and individual LTM-slot coordinates cannot slide
into one another when dimensions change. The source checkpoint is loaded once,
and optimizer/scaler/scheduler state is released before the larger model is
allocated.

The tokenizer must be present in the source package and its IDs must match the
source embedding rows. `vocab_size` cannot be changed by this tool because that
requires a separately audited tokenizer-ID migration. Expansion also cannot
turn internal `legacy-v8` weights into internal `coherent-v9` RWKV-v9 weights;
it preserves the authenticated
source revision. Use `--overwrite-output` only when intentionally replacing an
existing output package atomically.

### Workflow 8: Continuing Training (After Expanding or from Inference Checkpoint)

Start a *new* training session using only the *weights* from an existing model directory (not resuming optimizer/scheduler state).

```bash
python hierarchos_cli.py train \
    --hf_dataset "new_dataset_for_larger_model" \
    --text_column "text" \
    --model-path "./expanded_model" `# Load weights from expanded/previous model directory` \
    --tokenizer-path "./expanded_model" `# Use its tokenizer (assuming it was copied)` \
    --out-dir "./expanded_model_trained" \
    --epochs 2 \
    --starting-lr 5e-5 `# Start with a potentially smaller LR` \
    --amp \
    --gradient-checkpointing # Add if VRAM is limited
```

### Workflow 9: Converting Checkpoints to Inference Models

Convert a training checkpoint to a clean, inference-ready model directory.

```bash
python hierarchos_cli.py ckpt-2-inf \
    --ckpt-input "./my_model/hierarchos_epoch_60.pt" \
    --inf-output "./my_inference_model" \
    --ckpt-tok-path "openai-community/gpt2"  # Tokenizer used during training
```

This creates a HuggingFace-style directory:
```
my_inference_model/
├── hierarchos.pt         # Clean canonical model weights (~66% smaller than checkpoint)
├── hierarchos_config.json # Model configuration
├── tokenizer.json         # Tokenizer files
├── vocab.json
└── merges.txt
```

### Windows GUI Release Bundle

Build a portable Windows GUI bundle with a bundled PyTorch/Transformers backend:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\tools\build_windows_release.ps1
```

The release is written to `dist\Hierarchos-Windows\` and can be zipped for
distribution. Users run `Hierarchos.exe`; the GUI launches
`backend\hierarchos-backend.exe`, so they do not need to clone this repo or
install Python for normal inference. The GUI accepts a Hugging Face repo id,
a local model directory containing `hierarchos.pt` or `model.pt`, or a direct
`.pt` checkpoint with embedded config or a neighboring `hierarchos_config.json`.

### Workflow 10: Benchmark Evaluation (lm-eval)

Run standardized LLM benchmarks on your model. Requires `pip install lm-eval` (automatically installed through the setup script if you used it). Hierarchos now has named benchmark suites for post-training reporting, including common frontier-model benchmarks such as MMLU-Pro, GPQA Diamond, AIME 2025, BBH, IFEval, coding tasks, and ARC-AGI.

**List supported suites and benchmarks:**
```bash
python hierarchos_cli.py benchmark --list-benchmarks
```

**ROG Ally / local release sanity check:**
```bash
python hierarchos_cli.py benchmark \
    --model-path "./chatHRM" \
    --benchmark-preset rog-ally \
    --eval-limit 100
```

`--benchmark-preset rog-ally` is the recommended consumer-hardware smoke test before renting more cloud time. It uses CPU, batch size `1`, sequential execution, `max_new_tokens=64`, the light `arc_easy` / `hellaswag` / `truthfulqa_mc1` suite, and writes artifacts under `benchmark_results/rog_ally`. Without an explicit `--eval-limit`, the preset uses `25` samples per task for a quick run; `100` is a better local confidence check on a ROG Ally.

Current KortexHOS release smoke result at `--eval-limit 100`:

```text
arc_easy:     acc=0.3600, acc_norm=0.3200
hellaswag:    acc=0.3400, acc_norm=0.3700
truthfulqa:   acc=0.2200
```

**Post-training frontier text suite:**
```bash
python hierarchos_cli.py benchmark \
    --model-path ./hierarchos_model \
    --benchmark-suite frontier-text \
    --eval-batch-size 1 \
    --eval-limit 100
```

Results are written to `benchmark_results/<run>/results.json`, with a reproducibility manifest and Markdown summary beside it.

**Chain every registered benchmark sequentially:**
```bash
python hierarchos_cli.py benchmark \
    --model-path ./hierarchos_model \
    --benchmark-all \
    --eval-batch-size 1
```

`--benchmark-all` runs all runnable `lm-eval` benchmarks one after another and prints one combined scoreboard in the terminal at the end. External/official-path benchmarks such as ARC-AGI-3, SWE-bench Verified, Terminal-Bench, and MMMU are included in the manifest and skipped section unless their required external runner or local path is provided.

**ARC-AGI local JSON run:**
```bash
python hierarchos_cli.py benchmark \
    --model-path ./hierarchos_model \
    --benchmark arc-agi \
    --arc-agi-path ./ARC-AGI/data/evaluation \
    --arc-agi-max-tasks 20
```

`--arc-agi-path` accepts a single ARC-style JSON file or a directory tree of JSON tasks with `train` and `test` pairs. This is a local public-data runner; use the official ARC Prize path for private leaderboard-comparable numbers.

**Official ARC Prize path:**

- ARC-AGI-1/2 technical guide and data format: https://arcprize.org/guide/1
- ARC-AGI-1 public data source: https://github.com/fchollet/ARC-AGI
- ARC-AGI-2 public data source: https://github.com/arcprize/ARC-AGI-2
- ARC-AGI-2 official Kaggle/private evaluation route: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2
- ARC-AGI-3 docs and API/toolkit route: https://docs.arcprize.org/
- ARC-AGI-3 official Kaggle route: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- ARC Prize community leaderboard submission repo: https://github.com/arcprize/ARC-AGI-Community-Leaderboard

**During Training (End of Epoch):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
    --eval-tasks smoke \
    --eval-every-epoch 1 \
    --eval-limit 100 # Optional: test on only 100 samples for speed
```

**Step-Based Evaluation (Frequent tracking):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
    --eval-tasks arc_easy gpqa-diamond \
    --eval-steps 500 # Runs every 500 steps
    --eval-limit 10
```

-----

## 🎯 Using Your Trained Model

### Instruction-Trained Models (Alpaca, Dolly, etc.)

If you trained on **instruction-following datasets** like Alpaca, your model expects prompts formatted as instructions, not casual conversation.

**❌ This won't work well:**
```
>>> hello!
hierarchos: Journey.  (incoherent)
```

**✅ Use instruction-style prompts:**
```
>>> Explain what machine learning is in simple terms.
hierarchos: Machine learning is a type of artificial intelligence that uses 
algorithms to learn from data and improve performance...
```

For models trained with `--alpaca`, the training formatter uses this prompt string before the supervised output:
```text
### Instruction:
<instruction>

### Input:
<optional input>

### Response:
```

**Good prompt examples:**
```
>>> Write a short poem about learning.
>>> List 3 benefits of exercise.
>>> What is the capital of France?
>>> Explain photosynthesis to a 5-year-old.
```

### Sampling Parameters

Adjust generation quality with:
```bash
python hierarchos_cli.py chat --model-path "./my_model" --temperature 0.4 --top-k 40 --top-p 0.9 --repetition-penalty 1.15 --online-adaptation-policy off --no-passive-learning --chat-input-history-turns 0
```

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `--temperature` | Lower = more focused, higher = more creative | 0.4 for release checks, 0.5-0.7 for more variety |
| `--top-k` | Limit vocab to top K tokens | 40 |
| `--top-p` | Nucleus sampling threshold | 0.9 |
| `--repetition-penalty` | Penalize repeated tokens (1.0=off, >1.0=stronger) | 1.15 for KortexHOS release checks; 1.2 if repetition appears |
| `--online-adaptation-policy` | Selects `off`, explicit-only `validated`, passive-user `prompt`, or passive-user-and-response `prompt+response` writes | `validated` for chat; `off` for static parity |
| `--no-passive-learning` | Compatibility switch that disables passive prompt writes; explicit actions remain governed by the policy | Keep for legacy scripts; use the policy directly in new scripts |
| `--chat-input-history-turns 0` | Disables previous-turn injection into Alpaca `### Input:` | Recommended for first-turn and release-quality checks |

-----

## ⚙️ Command-Line Reference

### `hierarchos_cli.py` Arguments

| Argument                     | Mode(s)                             | Description                                                                                                                              | Default                 |
| :----------------------------- | :---------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :---------------------- |
| **Paths & Data** |                                     |                                                                                                                                          |                         |
| `--train`                      | `train`, `finetune`                 | Path to **local** data: JSON/JSONL file, or directory for `--pre_pt_dataset`. Use flag without path if using `--hf_dataset`. Mutually Exclusive with `--hf_dataset` path. | `None`                  |
| `--hf_dataset`                 | `train`, `finetune`                 | Name or path to a Hugging Face dataset (e.g., 'wikitext', 'c4', 'path/to/my\_csv/'). Mutually Exclusive with `--train` path.         | `None`                  |
| `--hf_dataset_config`          | `train`, `finetune`                 | Optional configuration name for the HF dataset (e.g., 'wikitext-103-raw-v1').                                                            | `None`                  |
| `--hf_dataset_split`           | `train`, `finetune`                 | Dataset split to use (e.g., 'train', 'validation', 'train[:10%]').                                                                       | `train`                 |
| `--text_column`                | `train`, `finetune`                 | Column name for text completion data in HF dataset (mutually exclusive with prompt/completion). Defaults to 'text' if available.           | `None`                  |
| `--prompt_column`              | `train`, `finetune`                 | Column name for prompt/instruction in HF dataset. Use with `--completion_column`.                                                        | `None`                  |
| `--completion_column`          | `train`, `finetune`                 | Column name for completion/response in HF dataset. Use with `--prompt_column`.                                                           | `None`                  |
| `--pre_chunked_dataset`        | `train`, `finetune`                 | Load pre-chunked **JSONL** dataset iteratively (requires `--max_length`). Mutually Exclusive with `--pre_pt_dataset` & `--hf_dataset`.     | `False`                 |
| `--pre_pt_dataset`             | `train`, `finetune`                 | Load pre-chunked **consolidated `.pt` tensor** dataset from directory specified in `--train` (requires `--max_length`). Mutually Exclusive with `--pre_chunked_dataset` & `--hf_dataset`. | `False`                 |
| `--model-path`                 | `train`, `finetune`, `merge`, `chat` | Path to model directory. **[Train]**: Loads old weights unchanged as a clean base with a fresh optimizer/schedule; use a new output directory. **[Other]**: Loads for the specified mode. | `None`                  |
| `--out-dir`                    | `train`, `finetune`, `merge` | Directory to save new models, checkpoints, or adapters.                                                                                | `./hierarchos_model`       |
| `--tokenizer-path`             | `train`, `finetune`, `merge` | Path or HF name of tokenizer (if not loading from model-path).                                                                           | `openai-community/gpt2` |
| `--resume-from-ckpt`           | `train`                             | Path to a training `.pt` checkpoint for exact same-run continuation, including optimizer/scheduler/scaler/data state. Do not use it to warm-start a different dataset/schedule. | `None`                  |
| `--shadow-model-path`          | reserved                            | Legacy compatibility argument; quantized online learning is disabled.                                                                   | `None`                  |
| `--lora-adapter-path`          | `merge`, `finetune`                 | Path to the trained LoRA adapter directory.                                                                                            | `None`                  |
| **Training/Fine-Tuning** |                                     |                                                                                                                                          |                         |
| `--epochs`                     | `train`, `finetune`                 | Number of training epochs.                                                                                                               | `3`                     |
| `--batch_size`                 | `train`, `finetune`                 | Number of samples per forward pass. The throughput-oriented default targets the documented 96GB Blackwell profile; reduce it on smaller devices. | `64`                    |
| `--accumulation-steps`         | `train`, `finetune`                 | Number of steps to accumulate gradients over (simulates larger batch size).                                                              | `1`                     |
| `--gradient-checkpointing`     | `train`, `finetune`                 | **Enable gradient checkpointing to save VRAM (trades compute for memory).** | `False`                 |
| `--full-sample-bptt`           | `train`                             | Use one attached graph over each complete active sample; forces `detach_every_n_steps=0`, disables unrelated cross-sample recurrent carry, and preserves cache metadata. | `False` |
| `--full-sample-activation-checkpointing` | `train`                    | Recompute attached temporal segments during one full-sample backward, bounding activations without truncating state gradients. Defaults on with full-sample BPTT. | `Auto` |
| `--full-sample-checkpoint-segment-size` | `train`                    | Activation-only segment length for exact checkpointed full BPTT; does not alter token-cache/ROSA/LTM geometry or detach recurrent state. Safe default `128`; v0.21 RTX PRO 6000 96GB B64/T8880 profile `224` after longest-batch preflight. | `128` |
| `--no-persist-state`           | `train`                             | Prevent unrelated DataLoader batches from sharing recurrent values. Within-sample recurrence and learned drift remain active; exact full BPTT forces this mode. | `False`                  |
| `--grad-clip`                  | `train`, `finetune`                 | Bound the global norm of finite gradients after FP16 unscale when applicable (`0` disables). Reduces explosion risk but cannot guarantee stability. | `1.0`                   |
| `--max-ce-loss-for-backward`   | `train`, `finetune`                 | Optional finite cap on CE used for backward only. `0` disables the cap and is recommended for from-scratch LM/assistant SFT training.     | `0.0`                   |
| `--ponder-loss-weight`         | `train`, `finetune`                 | Weight for the Ponder Cost auxiliary loss.                                                                                               | `0.01`                  |
| `--encourage-thinking`         | `train`                             | **Invert ponder loss to REWARD thinking.** Useful for ACT recovery training.                                                              | `False`                 |
| `--adaptive-ponder`            | `train`                             | **Scale ponder target with CE loss.** Harder content triggers more thinking.                                                              | `True` for Alpha v0.30 |
| `--ponder-target-scale`        | `train`                             | Scaling factor for adaptive ponder target (target = loss × scale).                                                                        | `0.5`                   |
| `--reset-halt-bias`            | `train`                             | **SURGICAL FIX:** Reset `h_halt_proj.bias` to this value on checkpoint load (e.g., `-2.0` for ~12% halt prob).                            | `None`                  |
| `--commitment-loss-weight`     | `train`, `finetune`                 | Weight for the commitment auxiliary loss to prevent posterior collapse.                                                                  | `0.5`                   |
| `--commitment-threshold`       | `train`, `finetune`                 | Hinge threshold for drift penalty. `coherent-v9` uses mean-square drift and defaults to the legacy-equivalent total-energy budget `0.1/context_dim` (`2.232e-4` at width 448); `legacy-v8` uses sum-square drift with `0.1`. | Revision-calibrated     |
| `--drift-state-clamp`          | `train`, `finetune`                 | Per-element clamp for worker drift states. Useful as a hard finite-value guard during unstable resumes.                                  | `5.0`                   |
| `--drift-norm-clamp`           | `train`, `finetune`                 | Optional L2 norm clamp for worker drift states. `0` disables it; use `3.5-4.0` for commit/drift rescue resumes.                         | `0.0`                   |
| `--drift-delta-scale`          | `train`, `finetune`                 | Scale applied to each worker drift update before accumulation. Values below `1.0` slow runaway drift growth.                            | `1.0`                   |
| `--recurrent-state-clamp`      | `train`, `finetune`                 | Per-element finite clamp for H/L recurrent states; changes values only when the bound is reached.                                       | `50.0`                  |
| `--context-state-clamp`        | `train`, `finetune`                 | Per-element finite clamp for manager context state.                                                                                      | `50.0`                  |
| `--activation-clamp`           | `train`, `finetune`                 | Per-element finite clamp for internal manager/worker activations.                                                                        | `100.0`                 |
| `--halt-logit-clamp`           | `train`, `finetune`                 | Per-element finite clamp for ACT halt logits.                                                                                            | `30.0`                  |
| `--rwkv-channel-mix-key-clamp` | `train`, `finetune`                 | Clamp RWKV `key_cm` preactivation before ReLU-squared channel mixing. `12.0` is the recommended stability default; `0` disables it.      | `12.0`                  |
| `--rwkv-channel-mix-deepembed-clamp` | `train`, `finetune`          | Clamp DeepEmbed's multiplicative RWKV channel-mix modulation before `value_cm`. `4.0` is the recommended stability default; `0` disables it. | `4.0`                |
| `--override-scheduling`        | `train`                             | Legacy combined reset: aliases both `--reset-optimizer-state` and `--rebuild-lr-schedule`. Do not use for a normal exact resume.          | `False`                 |
| `--rebuild-lr-schedule`        | `train`                             | On resume, preserve optimizer/scaler moments but intentionally build a new LR schedule over the remaining work.                           | `False`                 |
| `--reset-optimizer-state`      | `train`                             | On resume, intentionally discard optimizer/scaler moments without implicitly rebuilding the saved LR schedule.                           | `False`                 |
| `--starting-lr`                | `train`, `finetune`                 | Max Learning Rate for the schedule, or fixed LR if schedule disabled.                                                                    | `1e-4`                  |
| `--min-lr`                     | `train`, `finetune`                 | Minimum Learning Rate for cosine annealing schedule.                                                                                     | `1e-6`                  |
| `--warmup-steps`               | `train`, `finetune`                 | Optimizer update steps spent linearly warming from `--min-lr` to `--starting-lr`; overrides `--warmup-ratio` when set.                  | `0`                     |
| `--warmup-ratio`               | `train`, `finetune`                 | Fraction of optimizer updates used for LR warmup before cosine decay.                                                                    | `0.0`                   |
| `--disable-lr-schedule`        | `train`, `finetune`                 | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing.                                                                 | `False`                 |
| `--ltm-lr`                     | `train`, `finetune`                 | Maximum LTM inner-update LR used by the saved training schedule. This is distinct from transactional chat feedback.                     | `1e-3`                  |
| `--ltm-training-mode`          | `train`                             | `read-only` disables supervised target-gradient fast-slot writes while retaining the learned LTM reader and leakage-free writer auxiliary. It does not disable transactional runtime adaptation. `inner-update` is a legacy/Titans ablation. | `read-only` for Alpha v0.30 |
| `--inference-like-ltm-training` | `train`                            | Shortcut for `--ltm-training-mode read-only`; recommended for assistant rescue/coherence runs.                                         | `False`                 |
| `--ltm-value-alignment-weight` | `train`, `finetune`                 | Energy-normalized causal auxiliary for training the existing `val_proj` fast-memory writer. `0` preserves the historical RWKV-v8 objective. | `0.01` for fresh RWKV-v9; `0.0` for RWKV-v8 |
| `--ltm-value-alignment-stride` | `train`, `finetune`                 | Computes the writer auxiliary on one causal token in every N to limit training cost without adding parameters.                          | `8` for fresh RWKV-v9; `1` for RWKV-v8 |
| `--ltm-value-alignment-min-updates` | `train`, `finetune`             | Successful writer-training optimizer updates required before a checkpoint can enable transactional fast-memory writes.                 | `100`                   |
| `--ltm-value-alignment-ready-threshold` | `train`, `finetune`          | Maximum finite alignment-loss EMA accepted by the writer-readiness gate.                                                                | `0.95` for fresh RWKV-v9 |
| `--ltm-value-alignment-ema-decay` | `train`, `finetune`              | Decay used to track the writer alignment-loss EMA saved with the checkpoint.                                                            | `0.95`                  |
| `--ltm-value-writer-max-norm` | `train`, `finetune`                  | Maximum finite `val_proj` weight norm accepted by the writer-readiness gate.                                                            | `64.0`                  |
| `--assistant-recovery`         | `train`                             | Apply the v0.20.4 large-assistant SFT preset: Alpaca formatting, 4 epochs unless overridden, warmup+cosine LR, inference-like LTM training, prompt/response weights `0.10/1.0`, `2.0x` first `32` response tokens, `16` reserved response tokens, `0.003` ponder weight, and `5000` memory-gate warmup steps. | `False`                 |
| `--mask-prompt-tokens`         | `train`, `finetune`                 | Legacy SFT behavior: exclude prompt/instruction/input tokens from CE. By default, prompt tokens are trained and can be downweighted.      | `False`                 |
| `--allow-masked-active-labels` | `train`, `finetune`                 | Disable the fail-fast audit that rejects masked labels on real prompt/completion tokens when prompt-token training is active.             | `False`                 |
| `--prompt-loss-weight`         | `train`, `finetune`                 | Per-token CE weight for prompt/instruction/input tokens when prompt tokens are trained.                                                  | `1.0`                   |
| `--response-loss-weight`       | `train`, `finetune`                 | Per-token CE weight for completion/assistant response tokens. Response tokens are always the primary supervised target.                  | `1.0`                   |
| `--response-boundary-loss-weight` | `train`, `finetune`              | Multiplier for the first `--response-boundary-tokens` non-EOS response tokens.                                                           | `1.0`                   |
| `--response-boundary-tokens`   | `train`, `finetune`                 | Number of initial assistant response tokens multiplied by `--response-boundary-loss-weight`.                                             | `0`                     |
| `--min-response-tokens`        | `train`, `finetune`                 | Minimum non-EOS assistant response tokens kept when truncating prompt-completion rows. `--assistant-recovery` sets this to `16`.         | `1`                     |
| `--allow-empty-completions`    | `train`, `finetune`                 | Keep blank completion rows instead of dropping them. Use only when EOS-only answers are intended.                                        | `False`                 |
| `--no-memory-token-routers`    | `train`, `finetune`                 | Disable lightweight per-token routers for ROSA/LTM gates and use scalar memory gates only.                                              | `False`                 |
| `--memory-gate-warmup-steps`   | `train`, `finetune`                 | Training batches over which the ROSA/LTM gate floor decays to zero.                                                                     | `2000`                  |
| `--memory-gate-warmup-floor`   | `train`, `finetune`                 | Initial soft minimum for ROSA/LTM gates during warmup.                                                                                  | `0.10`                  |
| `--compile`                    | `train`, `finetune`                 | **Enable torch.compile (auto-enabled on CUDA).**                                                                              | `False`                 |
| `--force-compile`              | `train`, `finetune`                 | Force torch.compile even on Windows CPU (overrides safety check).                                                                         | `False`                 |
| `--compile-mode`               | `train`, `finetune`                 | torch.compile mode for the RWKV hot path. `max-autotune-no-cudagraphs` has longer startup than eager/reduce-overhead but keeps autotuned kernels without CUDA graph recapture warnings. | `max-autotune-no-cudagraphs` |
| `--compile-cudagraphs`         | `train`, `finetune`                 | Enable CUDA graph capture inside torch.compile for explicit benchmarking.                                                                 | `False`                 |
| `--no-compile-pad-to-chunk-size` | `train`, `finetune`               | Disable CUDA compile padding to `training_chunk_size` multiples; leaving it enabled reduces shape recompiles.                            | `False`                 |
| `--no-compile-static-worker-loop` | `train`, `finetune`              | Disable the compile-friendly fixed WorkerLoop used by CUDA compile.                                                                       | `False`                 |
| `--amp`                        | `train`, `finetune`, `chat`         | **Enable Automatic Mixed Precision (auto-enabled on CUDA).**                                                                                     | `False`                 |
| `--no-amp`                     | `train`, `finetune`                 | **Explicitly disable AMP** (overrides auto-detection on CUDA).                                                                                   | N/A                     |
| `--num_workers`                | `train`, `finetune`                 | Number of CPU workers for data loading (`-1` = auto; CUDA uses up to 8 for batch>=64, CPU/DML use 0).                                   | `-1`                    |
| `--prefetch-factor`            | `train`, `finetune`                 | Batches prefetched per DataLoader worker. Omit it to keep total queued batches tied to worker count.                                    | `None`                  |
| `--progress-log-steps`         | `train`                             | Update tqdm scalar metrics every N steps to reduce CUDA sync overhead from progress logging (`1` = every step).                         | `25`                    |
| `--hf-token-cache`             | `train`, `finetune`                 | Build/reuse a random-access binary token cache for HF datasets. Enabled by default; use `--no-hf-token-cache` to disable.               | `True`                  |
| `--hf-token-cache-dir`         | `train`, `finetune`                 | Directory for random-access HF token caches. Cache keys include formatter, weights, response guardrails, chunking, and architecture settings. | `None`              |
| `--hf-dataset-revision`        | `train`, `finetune`                 | Optional Hub commit/tag/branch; online cache builds resolve it to an immutable commit SHA before cache lookup.                         | `None`                  |
| `--refresh-hf-token-cache`     | `train`, `finetune`                 | Rebuild the random-access HF token cache once. This one-shot flag is not persisted by resume hydration.                                  | `False`                 |
| `--token-cache-build-batch-size` | `train`, `finetune`               | Batch size for Arrow fetch, fast tokenization, ROSA precompute, and compact cache encoding (`0` = auto, up to 512).                    | `0`                     |
| `--token-cache-only`           | `train`, `finetune`                 | Build/reuse the cache and exit before model allocation, suitable for a cheaper CPU preprocessing machine.                              | `False`                 |
| `--length-bucket-auto-sample-size` | `train`, `finetune`              | Maximum cached lengths sampled for one-time bucket tuning; the chosen window is persisted (`0` = all).                                | `1000000`               |
| `--cuda-loss-chunk-rows`       | `train`                             | Rows per CUDA vocabulary-loss chunk (`0` = choose once from live free VRAM). Lower values reduce loss-workspace peak at additional launch cost. | `0`                     |
| `--pt-cache-size`              | `train`, `finetune`                 | Number of `.pt` chunk files to keep hot per worker when using `--pre_pt_dataset`.                                                       | `2`                     |
| `--lora_r`                     | `finetune`                          | LoRA rank 'r'.                                                                                                                           | `8`                     |
| `--lora_alpha`                 | `finetune`                          | LoRA alpha scaling factor.                                                                                                               | `16`                    |
| `--finetune-unlock-percent`    | `finetune`                          | Target % of params to train (approx.). Overrides `--lora_r` if set.                                                                     | `None`                  |
| `--kayla`                      | `train`, `finetune`                 | Enable Kayla-style instruction tuning format (with thought-process). **Ignored if using pre-chunked formats or --text\_column.** | `False`                 |
| `--alpaca`                     | `train`, `finetune`                 | Enable Alpaca `instruction`/`input`/`output` formatting. Defaults prompt/completion columns to `instruction`/`output` and includes `input` as a `### Input:` context block. | `False`                 |
| **Inference** |                                     |                                                                                                                                          |                         |
| `quantize` mode                | unsupported                         | Exits with code 2. Legacy `.npz` export/loading cannot reproduce the Alpha v0.30 RWKV-v9 core and is deliberately fail-closed.           | N/A                     |
| `--device`                     | `chat`, `train`                     | Device for inference/training (`cpu`, `cuda`, or `dml`). **Note:** `dml` requires `torch-directml` and Windows. DirectML requires explicit opt-in. | `auto`                   |
| `--h-halt-thresh`              | `chat`                              | Probability threshold for early exiting the HRM reasoning loop during inference.                                                         | `0.9`                   |
| `--max-new-tokens`             | `chat`                              | Maximum number of tokens to generate in chat mode.                                                                                       | `512`                   |
| `--online-adaptation-policy`   | `chat`                              | Controls fast-memory transactions: `off`, explicit-action-only `validated`, passive-user `prompt`, or passive-user-and-response `prompt+response`. Ordinary prefill/generation remains read-only in every mode. | `validated`             |
| `--natural-feedback-detection` | `chat`                             | Opt into heuristic natural-language praise/rejection/correction detection. Exact slash actions do not require this heuristic.            | `False`                 |
| `--passive-learning`           | `chat`                              | Legacy compatibility switch that maps to at least the `prompt` policy and permits conservative observed-prompt writes.                  | `False`                 |
| `--passive-response-learning`  | `chat`                              | Legacy compatibility switch that maps to `prompt+response`; requires prompt-passive learning and enables quality/surprise-gated response writes. | `False`           |
| `--online-ltm-lr`              | `chat`                              | Explicit-feedback transactional fast-memory LR, separate from the training inner-update LR.                                             | `1e-3`                  |
| `--ltm-lora-path`              | `chat`                              | Optional: Path to save/load LTM updates as a separate delta file in chat mode.                                                           | `None`                  |
| `--static-ltm-lr`              | `chat`                              | Disable cosine annealing for chat LTM updates, use fixed `--ltm_lr`; pass `--dynamic-ltm-lr` to opt into cosine annealing.               | `True`                  |
| `--ltm-schedule-steps`         | `chat`                              | Number of chat updates per LTM LR cosine cycle.                                                                                          | `100`                   |
| `--ltm-schedule-min-lr`        | `chat`                              | Minimum LR for chat LTM cosine schedule.                                                                                                 | `1e-5`                  |
| **Benchmarking** |                                     |                                                                                                                                          |                         |
| `--benchmark-preset`           | `benchmark`                         | Bounded local benchmark profile. `rog-ally`/`ally`/`handheld`/`local` applies CPU, batch `1`, sequential execution, `eval_limit=25`, `max_new_tokens=64`, and the `rog-ally` suite unless tasks are explicit. | `None`                  |
| `--benchmark-suite`            | `benchmark`                         | Named benchmark suite to run, such as `rog-ally`, `smoke`, or `frontier-text`.                                                            | `None`                  |
| `--benchmark`                  | `benchmark`                         | Explicit benchmark task key(s) to run instead of a suite.                                                                                | `None`                  |
| `--benchmark-all`              | `benchmark`                         | Run every registered runnable benchmark sequentially.                                                                                    | `False`                 |
| `--eval-limit`                 | `train`, `benchmark`                | Limit eval samples per task. Use `100` for the recommended ROG Ally confidence check, or omit with `--benchmark-preset rog-ally` for a faster 25-sample smoke run. | `None`                  |
| `--eval-batch-size`            | `train`, `benchmark`                | Evaluation batch size. Use `1` on consumer/handheld hardware.                                                                            | `1`                     |
| **Architecture (Train)** |                                     | *(Used only if starting train from scratch)* |                         |
| `--context_dim`                | `train`                             | Core embedding dimension. The default 448 keeps 64-wide RWKV matrix-state heads; the complete Alpha v0.30 RWKV-v9 model is about 30.2M parameters with the GPT-2 vocabulary (RWKV-v8 was about 232.5M). | `448`                   |
| `--persistent_dim`             | `train`                             | Dimension of the fixed Persistent Memory.                                                                                                | `128`                   |
| `--ltm_slots`                  | `train`                             | Number of slots in the Long-Term Memory.                                                                                                 | `1024`                  |
| `--ltm_key_dim`                | `train`                             | Dimension of LTM keys.                                                                                                                   | `128`                   |
| `--ltm_val_dim`                | `train`                             | Dimension of LTM values.                                                                                                                 | `128`                   |
| `--h_hidden`                   | `train`                             | Hidden size of the High-Level (CEO) RNN. Defaults to `context_dim`.                                                                       | `448`                   |
| `--l_hidden`                   | `train`                             | Hidden size of the Low-Level (Worker) RNN. Defaults to `context_dim`.                                                                     | `448`                   |
| `--max_h_steps`                | `train`                             | **Maximum** number of reasoning steps H-module can take. **Impacts training speed.** | `5`                     |
| `--max_l_steps`                | `train`                             | **Maximum** number of iterations for L-module convergence per H-step. **Impacts training speed.** | `5`                     |
| `--l_conv_atol`                | `train`                             | Absolute tolerance for checking L-module state convergence.                                                                              | `1e-4`                  |
| `--ltm_topk`                   | `train`                             | Number of LTM slots to retrieve per token.                                                                                               | `4`                     |
| `--detach-every-n-steps`       | `train`                             | **Truncated BPTT:** detach RNN state gradients every N timesteps (`0` disables). Exact full-sample BPTT forces `0`.                     | `32`                    |
| `--training-chunk-size`        | `train`, `finetune`                 | TBPTT length and cache/ROSA/LTM/compile geometry. In exact mode it remains `256` while the separate checkpoint-segment flag controls activation replay and never the gradient horizon. | `256`                   |
| `--max_length`                 | `train`, `finetune`                 | Maximum sequence length. **Required if using pre-chunked formats.** Set via scan (`--auto-max-length`), manually, or loaded from config. | `1024`                  |
| `--auto-max-length`            | `train`, `finetune`                 | Automatically scan dataset (`--train` or `--hf_dataset`) to set `max_length`. **Ignored if using pre-chunked formats.** | `False`                 |
| **Other** |                                     |                                                                                                                                          |                         |
| `--threads`                    | `All`                               | Number of CPU threads for PyTorch/OpenMP.                                                                                                | `CPU_Count/2`           |

### `dataset_chunk_create.py` Arguments ✂️

| Argument            | Description                                                                                       | Required | Default                         |
| :------------------ | :------------------------------------------------------------------------------------------------ | :------- | :------------------------------ |
| `--dataset`         | Path to the input **JSONL** dataset file (Kayla format recommended).                              | Yes      |                                 |
| `--tokenizer-path`  | Path or Hugging Face name of the tokenizer to use for chunking.                                   | No       | `openai-community/gpt2`         |
| `--output-dir`      | Directory to save the output **consolidated** `.pt` chunk files and `manifest.jsonl`.             | No       | `train_Hierarchos_chunked_tensors` |
| `--overlap`         | Number of tokens to overlap between consecutive chunks.                                           | No       | `1024`                          |
| `--chunks-per-file` | Number of individual chunks to **consolidate** into a single `.pt` file.                          | No       | `1000`                          |

### `expand_model.py` Arguments 🌱

| Argument | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `--old-model-path` | Source model directory or direct `.pt` checkpoint. A local tokenizer must be available beside the resolved checkpoint. | Yes | |
| `--output-dir` / `--output-path` | Output model-package directory. A legacy `.pt` value is converted to a package-directory name. | Yes | |
| `--context-dim` | New context width. `h_hidden` follows it unless explicitly supplied; `l_hidden` follows only when context width changes. | No | Source value |
| `--persistent-dim` | New persistent-vector width. | No | Source value |
| `--ltm-slots`, `--ltm-key-dim`, `--ltm-val-dim`, `--ltm-topk` | Resize associative-memory geometry using slot-aware transplantation. | No | Source values |
| `--h-hidden`, `--l-hidden`, `--rwkv-head-size` | Override recurrent widths/head geometry. A shared head-size override updates both recurrent cells. | No | Source values |
| `--token-adapter-rank` | Override the shared Alpha v0.30 RWKV-v9 token-adapter rank. An inherited rank is capped when shrinking below it. | No | Source value |
| `--new-max-length` / `--auto-max-length` | Update sequence-length metadata directly or by scanning a dataset. | No | Source value |
| `--overwrite-output` | Atomically replace an existing package after the staged package passes checkpoint and tokenizer verification. | No | `False` |
| `--trust-remote-code` | Permit custom tokenizer code. Disabled by default and unnecessary for normal local packages. | No | `False` |

-----

## Native Rust/Vulkan backend — detailed developer reference

The coherent-v9 full-model native training backend now lives in
`hierarchos-vulkan/`. It uses raw Vulkan compute through Rust/`ash` and standard
FP32 SafeTensors, with the same canonical tensor names, row-major layouts, and
shapes consumed by external PyTorch/CUDA tooling and the `hierarchos-inference`
native runtime. The current graph carries raw-token coherent-v9 training through
the Hierarchos memory/control frontend, recurrent H/L reverse passes, exact
sparse state replay, full-model gradient accumulation, cross-entropy, and
canonical AdamW updates while preserving that exported SafeTensors contract.

The production native backend does not import or launch Python/PyTorch. Optional
developer-only cross-runtime parity checks live outside the native binaries and
can be run with:

Here, "PyTorch-compatible" describes the portable tensor/checkpoint ABI and the
numerical parity target only. PyTorch is not embedded, linked, imported, or
invoked by `hierarchos-native-cli`, `hierarchos-vulkan`, or
`hierarchos-inference`; training and inference stay on the Rust/Vulkan path.

```powershell
python tools/verify_vulkan_training_parity.py
python tools/verify_vulkan_model_interchange.py
```

The second check exports a coherent-v9 package, performs a Vulkan training step,
rewrites `model.safetensors`, and verifies that both PyTorch and the pure-Rust
inference engine consume the trained package. On hosts with CUDA available it
also executes the PyTorch CUDA inference check.

The native Rust trainer is also a standalone CLI. It consumes either the same
schema-v6 token cache used by the PyTorch data pipeline or legacy tokenized
JSONL, trains directly through Vulkan, and writes portable FP32-master
SafeTensors plus backend-neutral optimizer/resume sidecars:

```powershell
$env:HIERARCHOS_VULKAN_TRAINING_PRECISION="fp16-storage-parity"
cargo run --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train -- `
  --model .\hierarchos_model `
  --dataset .\token_cache `
  --output .\hierarchos_vulkan_model `
  --epochs 3 --batch-size 4 --gradient-accumulation-steps 4 `
  --lr 1e-4 --min-lr 1e-6 --warmup-ratio 0.03 `
  --tbptt-chunk-size 256 --device-index 0 --save-steps 100
```

There is now a higher-level native command-line frontend in its own Rust crate.
The native GUI keeps a tiny wrapper entrypoint around the same implementation,
so CLI behavior has one source of truth and the command-line runtime does not
pull in the GUI dependency graph. The preferred Windows build is the isolated
native-only release builder; it audits the Rust dependency graph, runs the native
test gates, builds the Rust GUI/CLI plus Vulkan runtime, and refuses Python
runtime artifacts in the staged bundle:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_native_release.ps1
```

That produces `dist\Hierarchos-Native\HierarchosNative.exe`,
`HierarchosCLI.exe`, and the Vulkan trainer/device binaries under `vulkan\`.
See [NATIVE_BACKEND.md](NATIVE_BACKEND.md) for the complete native workflow.
To build only the lower-level standalone binaries manually:

```powershell
cargo build --release --manifest-path hierarchos-vulkan/Cargo.toml --bin hierarchos-vulkan-train --bin hierarchos-vulkan-devices
cargo build --release --manifest-path hierarchos-native-cli/Cargo.toml
```

The resulting `hierarchos-native-cli` binary exposes the root CLI's major modes
while keeping the model/training hot paths native. In particular, `train`
translates the familiar Python option aliases into the Vulkan trainer, accepts
the canonical schema-v6 token cache directly, and can also tokenize ordinary
local JSONL with the package's `tokenizer.json` before launching Vulkan. For a
fresh run, `--model-path` is optional: pass local tokenizer assets and the CLI
constructs the coherent-v9 parameter package in Rust, then hands that package to
the same Vulkan trainer:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe devices

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --model-path .\hierarchos_model `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_model `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --min-lr 1e-6 --warmup-ratio 0.03 `
  --training-chunk-size 256 --precision fp16-storage-parity --device-index 0

# Exact continuation restores optimizer, scheduler/scaler, data cursor,
# pending accumulation state, and portable recurrent/LTM/ROSA replay state.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --resume-from-ckpt .\hierarchos_vulkan_model `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_resumed `
  --epochs 4 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --min-lr 1e-6 --warmup-ratio 0.03 `
  --training-chunk-size 256 --precision fp16-storage-parity --device-index 0

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe finetune `
  --model-path .\hierarchos_vulkan_model `
  --train .\domain_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_finetuned `
  --epochs 1 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-5 --training-chunk-size 256 `
  --precision fp32 --device-index 0

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\hierarchos_vulkan_finetuned `
  --prompt "Explain hierarchical recurrent reasoning." `
  --temperature 0.7 --top-k 40 --top-p 0.9 `
  --entropy-stop-threshold 0 --eos-stop-prob 0
```

The native chat frontend also implements the root CLI's opt-in raw-logit
uncertainty guards: `--entropy-stop-threshold`, `--entropy-stop-min-tokens`,
`--entropy-stop-top-prob`, and `--eos-stop-prob`. They use a stable Rust
softmax over the unmodified model logits before sampling, preserve the root
defaults (`0`, `3`, `0.05`, `0`), and stop fail-closed if an active guard sees
non-finite logits. No Python tensor/runtime is involved.

Fresh initialization is available directly from the same executable:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --tokenizer-path .\tokenizer_assets `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_fresh `
  --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64 `
  --persistent_dim 128 --ltm_slots 1024 --ltm_key_dim 128 --ltm_val_dim 128 --ltm_topk 4 `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --min-lr 1e-6 --warmup-ratio 0.03 `
  --training-chunk-size 256 --precision fp16-storage-parity --device-index 0
```

This fresh path constructs `model.safetensors`, both native/config interchange
files, and tokenizer-bound package assets in Rust before the first Vulkan
training submission. It does not import, embed, or launch Python/PyTorch.

Hugging Face acquisition is also native. The same executable now performs HTTPS
Hub downloads in Rust, writes through a local cache, and then enters the exact
same package/tokenization/training path. It does not shell out to Python,
`huggingface_hub`, Git LFS, or a framework loader:

```powershell
# Pull a complete published Hierarchos package for native chat/training.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe pull `
  --repo YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --revision main `
  --out-dir .\hierarchos_from_hf

# Warm-start Vulkan training from a canonical HF model package and a JSONL split
# discovered from an HF dataset repository.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --hf-model YOUR_ORG/YOUR_HIERARCHOS_REPO `
  --hf-model-revision main `
  --hf-dataset YOUR_ORG/YOUR_DATASET `
  --hf_dataset_split train `
  --hf-dataset-revision main `
  --out-dir .\hierarchos_vulkan_hf `
  --epochs 3 --batch_size 4 --accumulation-steps 4 `
  --starting-lr 1e-4 --training-chunk-size 256 --precision fp32

# A standard HF tokenizer can seed a completely fresh coherent-v9 package.
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe train `
  --tokenizer-path openai-community/gpt2 `
  --train .\instruct_dataset.jsonl `
  --out-dir .\hierarchos_vulkan_fresh_hf_tokenizer `
  --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64
```

`--hf-cache-dir DIR` relocates the native cache; otherwise
`.hierarchos-hf-cache` is used. `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` enables
private/gated repositories. `--hf-model-revision`, `--hf-tokenizer-revision`,
and `--hf-dataset-revision` pin the three sources independently. Model pulls are
intentionally strict: a Hub repo must contain the canonical Hierarchos
`model.safetensors`, both config files, and `tokenizer.json`. For tokenizer
compatibility, a non-existent `--tokenizer-path OWNER/REPO` is treated as the
root CLI's Hugging Face tokenizer form. Dataset discovery reads only repository
metadata, honors `--hf_dataset_config`/`--hf_dataset_split`, selects
JSONL/NDJSON files, and combines matching shards in lexical order. Ambiguous
repositories can be pinned to one exact file with `--hf-dataset-file`. The
native binary never executes remote dataset builder code, implicitly converts
Parquet/CSV, or enables `trust_remote_code`.

The compatibility target is workflow/mode parity, not emulation of every
framework-specific Python flag. The native command intentionally has no hidden
fallback path:

| Root CLI mode | Native status |
| --- | --- |
| `train` | Full coherent-v9 model training in Vulkan, including fresh Rust-only model initialization from local tokenizer assets, warm-start from canonical SafeTensors, gradient accumulation, AdamW, LR scheduling, periodic exact-resume checkpoints, raw local JSONL tokenization, schema-v6 token-cache input, and single-/multi-device Vulkan selection. |
| `finetune` | Vulkan training with a frozen optimizer selection. By default it trains existing coherent-v9 recurrent low-rank factors, DeepEmbed/ROSA factors and routers, plus slow-LTM tensors; repeat `--trainable-prefix` for an explicit canonical selection. |
| `chat` | Pure-Rust full-precision inference through `hierarchos-inference`, including portable recurrent/ROSA/LTM chat-state save/resume and the root CLI's opt-in raw-logit entropy/EOS uncertainty stop guards. |
| `benchmark` | Pure-Rust local inference/throughput benchmarking. Python `lm-eval`, ARC catalogs, and other external benchmark registries are deliberately not launched by the native binary. |
| `pull` | Pure-Rust HTTPS download of a canonical Hierarchos SafeTensors/config/tokenizer package from Hugging Face, with revision pinning, caching, optional bearer-token authentication, and staged validation before publication. |
| `merge-lora` | Pure-Rust merge of a bound Hierarchos PEFT-LoRA SafeTensors adapter into a standalone canonical model package. |
| `ckpt-2-inf` | Native SafeTensors-package export/validation. Framework-object `.pt` checkpoints are rejected rather than deserialized through PyTorch. |
| `quantize` | Intentionally unavailable for coherent-v9 until a matrix-state quantized format can preserve the current learned function. The command fails closed. |

Exact native continuation is deliberately stricter than a weights-only restart.
`--resume-from-ckpt` restores and validates the optimizer, scheduler/scaler,
data cursor, pending accumulation state, and portable recurrent/LTM/ROSA replay
state. To start a new optimizer/schedule from existing weights, pass the package
with `--model-path` instead. Framework-only resume conveniences such as arbitrary
Python optimizer-object mutation are not treated as native parity features.

The GUI crate also keeps a tiny wrapper entrypoint around this same native CLI
implementation, but the standalone binary above has no GUI dependency graph.
The chat command can save and resume the backend-neutral recurrent/ROSA/LTM
runtime state without a framework-specific object format:

```powershell
.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\hierarchos_vulkan_model `
  --carry-chat-state --chat-state-file .\chat-state.json

.\hierarchos-native-cli\target\release\hierarchos-native-cli.exe chat `
  --model-path .\hierarchos_vulkan_model `
  --resume-chat-from-state-file .\chat-state.json
```

For exact PyTorch/Vulkan data-objective parity, the schema-v6 token cache remains
the canonical interchange path because it preserves the already-tokenized IDs,
labels, masks, loss weights, and content identity. Native raw-JSONL tokenization
is a convenience path for local datasets and intentionally keeps the package
tokenizer authoritative. That raw path now mirrors the root data contract for
`text`/`content`, `instruction`/`output`, `prompt`/`completion`, and
`question`/`answer` rows, resolves and appends EOS natively, drops blank
completions by default, preserves the prompt suffix plus the start of the answer
during truncation, and supports `--min-response-tokens`,
`--allow-empty-completions`, response-boundary weighting, and the supported
native portion of `--assistant-recovery`. Schema-v6 remains the exact interchange
path when a run must preserve a precomputed PyTorch-side token/loss object
bit-for-bit.

`chat`, the local throughput `benchmark`, Vulkan `devices`, native `train`,
native `finetune`, SafeTensors-package `ckpt-2-inf`, and bound SafeTensors
`merge-lora` are pure Rust/Vulkan-native paths. The standalone native binary has
no Python/PyTorch dispatcher and never silently crosses runtimes. Native
`hierarchos-vulkan`, `hierarchos-native-cli`, and `hierarchos-inference` also do
not link `pyo3`, `tch`, or libtorch; Python tooling elsewhere in the repository is
outside this native backend and is not invoked by it. Native
`finetune` freezes the canonical optimizer by default to coherent-v9's existing
low-rank recurrent factors, DeepEmbed/ROSA adapter factors and routers, and slow
LTM tensors; `--trainable-prefix` can replace that selection with an explicit
canonical tensor-prefix set. Because those factors are already part of the model
architecture, finetuning emits a complete `model.safetensors` package directly
and requires no framework adapter object or post-training merge. `merge-lora`
separately validates an externally supplied bound SafeTensors adapter/base and
architecture contract in Rust, applies the standard LoRA `B @ A` delta, restores
any saved slow-LTM tensors, and emits another canonical model package.

A request for a genuinely framework-only workflow still fails closed with a
native error instead of silently launching Python. That currently includes
Hugging Face dataset-builder execution beyond the native repository-metadata +
JSONL/NDJSON path, framework-object `.pt` conversion, external lm-eval/ARC
catalogs, and injecting a new arbitrary PEFT-LoRA geometry at runtime. Legacy
`--lora_r`, `--lora_alpha`, `--lora_dropout`, and
`--finetune-unlock-percent` options are accepted by native `finetune` only as
compatibility geometry hints; they do not mutate coherent-v9's architecture.
`quantize` remains intentionally disabled for coherent-v9 because the old
scalar-RWKV format cannot represent the current learned function.

The native training objective supports both the root defaults
(`--max-ce-loss-for-backward=0` and `--max-ponder-cost-for-backward=0`) and the
optional nonzero sequence-scalar CE/ponder backward caps. When either cap is
active, the Vulkan trainer performs a forward-only scalar preflight for each
historical PyTorch-TBPTT chunk, derives the exact `torch.minimum(value, cap)`
backward gate (`1` below the cap, `0` above it, and `0.5` at an exact tie), and
then applies that gate to the native reverse-mode sources. The cap policy is
carried through both dense and sparse replay plus gradient-accumulation windows;
the standalone `hierarchos-native-cli train` frontend forwards the same flags
without invoking Python/PyTorch. The commitment cap remains gradient-transparent,
matching the root trainer's straight-through `preserve_gradient=True` behavior.

The higher-level native `train` frontend also carries the root CLI's ordinary
defaults rather than inheriting the low-level trainer's deliberately tiny smoke
defaults: 3 epochs, batch size 64, seed 1337, minimum LR `1e-6`, and ponder-loss
weight `0.01`. Explicit launch arguments remain last-write-wins. `--amp` maps to
the qualified Vulkan `fp16-storage-parity` policy and `--no-amp` maps to `fp32`,
so existing launch scripts can select mixed precision without introducing a
framework dependency.

### Native backend acceptance status

On the local AMD Radeon Graphics Vulkan target on August 27, 2026, the release
native CLI completed a four-step FP32 training run, emitted optimizer-boundary
periodic checkpoints, and then resumed the exported package for a second epoch
through optimizer step 8. The resumed run reported that model, optimizer,
training-session, scheduler, and data-stream state were restored rather than
reinitialized. A fresh native `finetune` smoke selected 27 canonical tensors and
froze 68, completed four Vulkan optimizer steps, and produced a package that the
standalone pure-Rust inference binary immediately reloaded and executed. A
separate from-scratch acceptance run started with only local GPT-2 tokenizer
assets plus tokenized JSONL, constructed a 50,257-vocabulary coherent-v9 model
entirely in Rust, completed two FP32 Vulkan optimizer steps on the same AMD GPU
(recorded loss `10.8399` then `10.8068`), emitted canonical `model.safetensors`
plus optimizer/resume state, and was immediately reloaded for generation by the
pure-Rust `chat` path. The
current Rust test gate also reports `16 passed` for `hierarchos-native-cli`,
`197 passed, 8 ignored` for the runnable `hierarchos-vulkan` library tests, and
`6 passed` for the dedicated `hierarchos-native` GUI; the
ignored cases are explicit GPU microprofiles rather than failed correctness
tests. `hierarchos-inference` additionally reports `12 passed`, including native
bootstrap/package loading and recurrent/ROSA runtime-state coverage. The native
GUI tests cover exact-resume policy rehydration, Vulkan device parsing/selection,
and native training-event parsing rather than relying on a compile-only gate.

The optimized release binaries were rebuilt after the raw-JSONL parity changes.
The standalone native CLI's `devices` command then enumerated the local
`AMD Radeon Graphics` Vulkan adapter from that release executable. A Cargo
dependency-tree audit of `hierarchos-native-cli` and its native stack found no
`pyo3`, `tch`, `torch-sys`, or libtorch dependency; those binaries therefore do
not gain a hidden Python/PyTorch runtime through their Rust dependency graph.

A final release-path revalidation on August 27, 2026 used the public
`hierarchos-native-cli` executable and the rebuilt Vulkan trainer/device binaries.
The CLI discovered the AMD adapter, completed an FP32 optimizer step with an
optimizer-boundary checkpoint, resumed that package exactly into epoch 2 with
both `resumed_optimizer=true` and `resumed_training_session=true`, reached
optimizer step 2, and then loaded the resumed package through the pure-Rust
`benchmark` inference path. This revalidation did not invoke the repository's
Python tooling.

After the native chat uncertainty guards were added, the release binaries were
validated again through the same public frontend. A four-row masked tokenized
JSONL fixture completed one FP32 Vulkan optimizer step on `AMD Radeon Graphics`
at batch size 4 (`mean_recorded_loss=10.738201`), emitted an optimizer-boundary
checkpoint, and published a canonical SafeTensors package. That newly trained
package immediately reloaded through native `chat` with all four entropy/EOS
guard switches enabled and through the pure-Rust local `benchmark` path. The
post-change CLI gate reports `16 passed, 0 failed`; the Vulkan library gate
reports `197 passed, 0 failed, 8 ignored` and `hierarchos-inference` remains
`12 passed, 0 failed`.

On August 28, 2026, the release public `hierarchos-native-cli train` entrypoint
was rebuilt and exercised on the local `AMD Radeon Graphics` adapter with
`--max-ce-loss-for-backward 4.0`, `--max-ponder-cost-for-backward 1.0`, and a
two-microbatch gradient-accumulation window. The four-row native fixture
completed both batches, performed one optimizer step, wrote an optimizer-boundary
checkpoint, reported `mean_recorded_loss=3.743555`, and exited successfully.
Exact resume from that checkpoint restored both optimizer and training-session
state, continued at epoch 2, reached optimizer step 2, and reported
`mean_recorded_loss=3.740881` with the same cap policy. These runs used the Rust
CLI and Vulkan trainer directly; no Python/PyTorch runtime participated in the
training path.

Also on August 28, 2026, `tools/build_native_release.ps1` was exercised as the
isolated native release gate. It audited `hierarchos-inference`,
`hierarchos-vulkan`, `hierarchos-native-cli`, and the dedicated
`hierarchos-native` GUI dependency trees for Python/libtorch Rust bindings, ran
the native correctness gates (`12` inference tests, `197` Vulkan tests with `8`
explicit GPU microprofiles ignored, `16` native-CLI tests, and `6` dedicated
native-GUI tests), compiled the native GUI, and produced `dist/Hierarchos-Native`
without `.py`, `.pyc`, `.pyd`,
or Python DLL artifacts. The bundled device probe enumerated `AMD Radeon
Graphics`. The bundled `HierarchosCLI.exe` then completed a one-step FP32 Vulkan
training run (`mean_recorded_loss=11.027534`), wrote an optimizer-boundary
checkpoint, resumed it into epoch 2 with both `resumed_optimizer=true` and
`resumed_training_session=true` (`mean_recorded_loss=10.988756`), and the
resulting package loaded successfully through the bundled pure-Rust local
`benchmark` inference path. These tiny-fixture losses are acceptance values, not
quality or performance benchmarks.

The Hub transport itself was re-exercised on August 27, 2026 through the release
native binary. `openai-community/gpt2` supplied `tokenizer.json` and its available
sidecars through the Rust HTTPS/cache layer; those Hub tokenizer assets were then
used to initialize a coherent-v9 package from scratch and complete a Vulkan FP32
optimizer step without a local model input. The same binary also fetched
`polinaeterna/jsonl_test` `data/train.jsonl` through `--hf-dataset` /
`--hf-dataset-file`, tokenized its three `text` rows natively, and completed three
Vulkan optimizer steps. The current native CLI unit gate reports `16 passed`, including
traversal rejection, URL/path encoding, dataset split/shard-selection tests for
the Hub boundary, response-preserving truncation, EOS-safe response-boundary
weighting, and root-style raw-text schema detection. A canonical Hierarchos model
pull remains package-validation-bound: publication occurs only after the required
native SafeTensors/config/tokenizer files are present and the tokenizer vocabulary
matches the model contract.

That AMD run is evidence for the Vulkan backend, not an NVIDIA/CUDA execution
claim. CUDA interoperability is defined at the canonical SafeTensors tensor
names/shapes/layout boundary so an external CUDA implementation can consume the
same trained weights. The native backend itself remains Vulkan on NVIDIA as
well; qualifying a particular NVIDIA driver/GPU still requires running the
repository's cross-runtime qualification checks on that hardware.

The native training contract is deliberately explicit. `train` accepts
either (a) an existing canonical SafeTensors model package, (b) an exact native
resume package, or (c) no model at all when local tokenizer assets are supplied
for fresh coherent-v9 initialization. `--hf-model` can materialize case (a) from
a canonical Hub repository, and `--hf-tokenizer` can supply case (c) without a
local tokenizer copy. `finetune` remains model-bound. Training data can be local
JSONL/tokenized JSONL, a schema-v6 token cache, or an explicit JSONL-compatible
file pulled from a Hub dataset repository with `--hf-dataset`; config/split
selection follows the root-compatible `--hf_dataset_config` and
`--hf_dataset_split` hints, while `--hf-dataset-file` remains the exact-file
override. The package keeps
ordinary tensor names/layouts and `hierarchos_config.json` alongside the stricter
`hierarchos_rust_config.json`; this is the interoperability boundary for external
CUDA consumers while the actual native optimizer/training loop stays entirely in
Rust and Vulkan. Training/checkpoint export carries tokenizer assets into the
resulting package, so fresh and warm-started outputs remain directly usable by
the native `chat` command. Minimal token-ID parity fixtures may intentionally omit
tokenizer assets; those can still be consumed by the lower-level pure-Rust
inference engine with explicit token IDs.

The main `hierarchos-gui` Training dashboard can launch this same executable by
selecting **Vulkan (native)**. Its Vulkan parity controls map directly to the
Rust trainer's precision, AdamW, warmup/cosine schedule, objective weights,
backward safety caps, TBPTT, recurrent persistence, seed, and shuffle policy.
Exact resume is fail-closed: the GUI reloads trajectory-defining values and the
saved precision policy from `training_state.json` before launch, while leaving
only runtime choices such as output directory, target epoch count, save cadence,
and Vulkan device topology editable. This prevents an FP16/GradScaler checkpoint
from being accidentally resumed as a nominally similar FP32/default-optimizer
run.

End-to-end training-submission profiling is plan-aware rather than limited to
individual kernel islands. The benchmark can force a memory-safe sequence
microbatch/checkpoint plan, and the collector persists rank/device/architecture
geometry plus timing, queue-submission, memory-headroom, and rejected-plan
records in JSONL:

```powershell
python tools/benchmark_vulkan_training_submission.py --tokens 8 --sequences 2 --microbatch-size 1 --checkpoint-stride 2
python tools/benchmark_vulkan_training_submission.py --tokens 8 --sequences 2 --microbatch-size 1 --checkpoint-stride 2 --precision fp16-storage-fp32-compute
python tools/benchmark_vulkan_training_submission.py --tokens 8 --sequences 2 --microbatch-size 1 --checkpoint-stride 2 --precision fp16-storage-fp16-lm-backward
python tools/collect_vulkan_training_submission_profiles.py --tokens 4,8,16 --sequences 1,2,4 --microbatches auto,1,2,4 --checkpoint-strides 1,2,4 --numerics strict,fast-recurrent-tree,fast-recurrent-tiled --precisions fp32,fp16-storage-fp32-compute,fp16-storage-fp16-lm-backward
# Exercise a 64-wide RWKV head with a two-lane WG128 recurrent reduction.
python tools/collect_vulkan_training_submission_profiles.py --fixture-width 64 --tokens 4 --sequences 1 --microbatches 1 --checkpoint-strides 1 --kernel-geometries 128 --numerics strict,fast-recurrent-tree,fast-recurrent-tiled
```

The opt-in FP16-storage arm now covers the six recurrent low-rank matrices in
each H/L tower plus dense `lm_head.weight` forward and hidden-adjoint reads. The
canonical masters, gradients, AdamW moments, tied embedding reads, and exported
SafeTensors remain FP32, so a Vulkan-trained package uses the same model-file ABI
as PyTorch CPU/CUDA and `hierarchos-inference`. FP16 mirror writes use explicit
IEEE round-to-nearest-ties-to-even rather than driver-dependent half packing, and
the parity harness checks the post-AdamW LM mirror bit-exactly against the Vulkan
FP32 master. Benchmark/profile records expose an
`lm_head_fp16_parameter_storage_active` bit alongside the H/L low-rank bits and
reject a requested FP16 run if any of those consumers silently falls back.

The first true FP16-compute backward tranche is available as
`fp16-storage-fp16-lm-backward`. It retains the FP32 softmax/log-sum-exp,
gradient accumulators, master parameters, AdamW moments, and checkpoint ABI, but
rounds the final source-scaled CE adjoint to FP16 and executes the LM-head
`W^T`/`dW` products as native Float16 multiplies before widening each product
into FP32 accumulation. This precision policy requires Vulkan Float16 + 16-bit
storage support and never silently degrades to the storage-only arm. The
Vulkan↔PyTorch/CUDA trajectory gate can pin the same policy on every Vulkan leg
with `python tools/verify_vulkan_cuda_vulkan_trajectory.py --precision fp16-storage-fp16-lm-backward --require-cuda`.

The dense LM-head FP16 plan now has vocabulary-major CE-tape candidates in
addition to the row-major baseline. The projection writes compact per-vocabulary
tile `(max, scaled-exp-sum, target-logit)` partials alongside the reusable logit
tape, so CE row statistics no longer reread every FP32 logit. A rows16 variant
keeps packed half2 weights in shared memory and unpacks at FP32 accumulation
time, reducing the width-448 shared footprint from roughly 30.5 KiB to 16.2 KiB
while halving vocabulary workgroups versus rows8. On the local 32 KiB,
subgroup-64 AMD Radeon at vocab 50,257, the width-448 LM backward microprofile
measured rows16 at about `22.24 ms` versus rows8 at `24.07 ms` for two rows, and
`38.86 ms` versus `46.13 ms` for eight rows. The LM autotuner selected rows16 in
both cases; this remains a device/geometry-specific race rather than a global
default, and it does not alter the FP32 SafeTensors/PyTorch/CUDA/native-inference
ABI.

This expansion is intentionally not the default yet. On the current AMD Radeon
Graphics width-32 scheduler fixture, an isolated automatic-plan A/B selected the
same WG64/microbatch-2 plan for both precision arms, but measured `367.76`
batch-tokens/s for FP32 versus `346.49` for FP16 storage. Representative
large-vocabulary geometry must show a stable throughput win before the scheduler
evidence gate is considered strong enough for default promotion.

The Rust tape scheduler now consumes that JSONL database automatically when it
is present at `benchmark_results/vulkan_training_submission_profiles.v1.jsonl`.
For an installed/runtime-specific database, set
`HIERARCHOS_VULKAN_TAPE_PROFILE_DB=PATH`. Matching is exact across Vulkan device,
subgroup width, coherent-v9 model geometry, batch, sequence count, and token
span, compiled H/L backward geometry, RWKV reduction-numerics policy, and
training precision policy. Legacy profile records mean FP32 precision.
Matching controlled explicit profiling records are aggregated with recency and
uncertainty-aware throughput statistics rather than treating one fast sample as
a universal winner. Automatic-plan
records remain useful diagnostics but are not re-ingested as training data, so
the scheduler cannot reinforce its own previous choices. A candidate is never
selectable unless the live Vulkan memory budget and conservative tape footprint
model say it fits. Set
`HIERARCHOS_VULKAN_DISABLE_TAPE_PROFILES=1` to restore the memory-only heuristic,
or `HIERARCHOS_VULKAN_TAPE_PROFILE_LOG=1` to print profile hits and the safety
headroom that admitted them.

For a real coherent-v9 PyTorch checkpoint, pass `--source-model MODEL_OR_DIR` to
the collector. It exports the standard Rust/SafeTensors package into a temporary
directory before profiling, so the measured execution path is the same package
contract used for native inference and PyTorch/CUDA interchange. Legacy v8 and
scalar-RWKV checkpoints remain intentionally fail-closed.

## Roadmap

  * [x] Promote the Rust/Vulkan stack to the primary Hierarchos Alpha v0.30 execution path and ship it as an isolated native release.
  * [x] Ship a dedicated pure-Rust GUI wrapper for native inference and direct Vulkan training (`hierarchos-native`).
  * [x] Implement the coherent-v9 full-model training loop in Vulkan with portable FP32-master SafeTensors, AdamW/resume state, TBPTT, mixed-precision policies, and native multi-adapter execution.
  * [x] Add native parameter-efficient coherent-v9 finetuning over built-in low-rank/shared factors plus Rust SafeTensors LoRA adapter merge parity.
  * [ ] Expand and benchmark native multi-device Vulkan scaling beyond the current adapter-selection/execution path, including controlled cross-vendor NVIDIA/AMD validation.
  * [ ] Add optional on-the-fly arbitrary PEFT-LoRA geometry injection/training without changing the canonical native runtime contract.
  * [ ] Extend the architecture to support multi-modal inputs (images, audio).
  * [ ] Optimize LTM retrieval with approximate nearest neighbor search for larger memory capacities.
  * [ ] Continue Vulkan shader fusion, autotuning, and low-precision kernel work for higher training throughput while preserving the canonical SafeTensors contract.

## License

The source code of Hierarchos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Hierarchos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  * This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  * **RWKV** architecture by BlinkDL — linear attention with RNN efficiency.
  * The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  * **pybind11** for seamless C++/Python integration.
  * **Hugging Face `datasets`** library for broad data compatibility.
  * **PyTorch Team** for gradient checkpointing functionality.
  * **DirectML/ZLUDA communities** for enabling AMD GPU acceleration on Windows.

## Changelog

### v0.30 (alpha)

  * **Native/Vulkan Major Update**: Hierarchos Native is promoted to the primary Alpha v0.30 execution path. The standalone stack now covers Rust-side model/package handling, tokenization, fresh coherent-v9 initialization, CLI/GUI orchestration and inference, plus full-model Vulkan training/optimization, native fine-tuning, exact resume, supported Hugging Face transport, local benchmarking, and canonical SafeTensors interchange without a Python/PyTorch runtime dependency.
  * **Native Release Gate**: `tools/build_native_release.ps1` audits the Rust dependency graph for Python/libtorch bindings, runs the native crate/GUI test gates, builds `HierarchosNative.exe`, `HierarchosCLI.exe`, and the Vulkan trainer/device binaries, probes Vulkan when available, and rejects Python runtime artifacts from the staged distribution.
  * **Product and Core Naming**: The public release is **Hierarchos Alpha v0.30**. RWKV-v8 and RWKV-v9 identify core architecture generations only; `legacy-v8` and `coherent-v9` remain compatibility-stable internal checkpoint/config identifiers.
  * **RWKV-v9 Core**: Introduces the corrected matrix-state recurrence, per-row hard ACT, bounded deterministic ROSA, isolated metadata-safe LTM, and shared factorized token adapters.
  * **Affordable Reference Architecture**: The `448/448/448` reference has `30,227,653` parameters and `102` state-dict entries, versus `232,516,229` parameters and `95` entries for the historical RWKV-v8 reference—an `86.9998%` parameter reduction.
  * **Leakage-Free Writer Training**: Fresh RWKV-v9 runs train the existing `val_proj` writer causally at weight `0.01` and stride `8`, with a 100-update minimum plus finite alignment-EMA and writer-norm readiness gates.
  * **Transactional Online Adaptation**: Restores intended fast-memory learning behind `off`, `validated`, `prompt`, and `prompt+response` policies. Explicit feedback is the safe default; passive prompt and self-response writes remain opt-in, bounded, replay-validated transactions.
  * **End-to-End Coherence**: Hardens checkpoint/resume, cache/tokenizer, CLI/GUI, packaging, and inference boundaries, with an ASCII RWKV-v8-to-v9 architecture flowchart near the top of this README.

### v0.21 (alpha)

  * **Exact Full-Sample BPTT**: `--full-sample-bptt` forces recurrent detachment off and prevents unrelated batches from sharing recurrent values. Attached non-reentrant activation checkpoints preserve one mathematical objective and gradient horizon across every retained token in each sample.
  * **96GB Blackwell Reference Profile**: Documents BF16 AMP, batch `64`, `max_length=8880`, training/cache geometry `256`, and activation segment `224`, subject to a required longest-batch preflight on the rental GPU. The padded 8,960-token maximum divides into 40 equal segments; `128` remains the safe default and `64` the first OOM fallback.
  * **Drift and Inference Parity**: Drift remains a learned, clipped part of the hierarchy. Exact activation boundaries derive drift from attached L-state instead of injecting terminal drift twice; full-precision chat, lm-eval, ARC-AGI, and the GUI bridge honor the saved exact recurrence/refinement policy.
  * **Compact Multi-Billion-Token Pipeline**: Pins mutable Hub revisions to immutable SHAs, batches Arrow fetch/tokenization, adds mmap schema-v6 uint16/RLE caches with checked label aliasing, auto-tunes length buckets, keeps persistent workers, and overlaps pinned transfers on a dedicated CUDA stream.
  * **Precision and Numeric Guardrails**: Keeps FP32 parameters/AdamW/sensitive state with BF16 autocast on supported Blackwell CUDA. Finite loss/gradient rejection, unscale-before-clip ordering, global gradient clipping, state/context/drift/activation/halt bounds, and channel-mix clamps remain active. QAT/FP8 are not release training modes, and quantized inference is not numerically logit-identical.
  * **Checkpoint-Compatible Promotion**: No learned tensor name or shape changed. The reference model remains `232,516,229` unique parameters and `95` state-dict entries; older coherent checkpoints remain valid weights-only bases through `--model-path` with their original tokenizer.
  * **Expanded Verification**: `283 passed, 4 skipped`; architecture integrity `65 passed`, `1` documented legacy-quantization warning, `0 failures`. Direct-versus-checkpointed full-BPTT gradient and inference-boundary parity controls pass.

### v0.20.7 (alpha)

  * **Epoch-13 Control Validated**: The real `232,516,229`-parameter checkpoint strict-loads all `95` tensors, contains no non-finite learned state, and matches one-token chat streaming across a `256`-token boundary within `1.144409e-05` logits.
  * **Legacy Hebbian Guard**: Checkpoints whose `val_proj` writer was never trained cannot inject random projected values into fast LTM memory. Gradient-derived chat feedback remains available.
  * **Opt-In Writer Training**: Adds an energy-normalized `--ltm-value-alignment-weight` auxiliary, writer no-decay grouping, and a configurable readiness-update threshold without changing checkpoint tensor layout.
  * **Nested Download Resolution**: A single unambiguous nested model package resolves its weights and exact tokenizer automatically.
  * **Reproducible Weight Audit**: Adds `tools/audit_checkpoint_health.py`, a saved epoch-13 JSON control, and a detailed v2 warm-start assessment in `EPOCH13_CHECKPOINT_AUDIT.md`.
  * **Expanded Verification**: `230 passed, 3 skipped`; architecture audit `65 passed, 1 documented legacy-quantization warning, 0 failures`.

### v0.20.6 (alpha)

  * **Checkpoint Compatibility Preserved**: No learned parameter names or shapes changed. The reference model remains `232,516,229` parameters with `95` state tensors.
  * **Strict and Atomic Checkpoints**: Coherent loading rejects missing, unexpected, conflicting, shape-incompatible, or non-finite learned tensors. Saves reject rather than rewrite corrupt learned/optimizer/gradient state, do not mutate live finite LTM memory, and retain the previous checkpoint if atomic installation fails.
  * **Carried-Chat TBPTT Parity**: Absolute chunk boundaries prevent a new conversation turn from accidentally reseeding drift in the middle of a training chunk; the final generated token is flushed once into carried state.
  * **LTM Correctness**: Filtered retrieval handles no-match rows independently without severing addressing gradients, mixed batches avoid unintended decay, and non-finite inner-update gradients are rejected.
  * **Trainer and Export Guardrails**: Exact resume restores optimizer/scheduler/scaler state, DeepEmbed is excluded from weight decay, full-sequence LoRA uses read-only fast memory, and inference export honors the tokenizer path stored by training.
  * **ROSA and Evaluation Efficiency**: Persistent ROSA avoids duplicate first-chunk work and bounds automatic workers; token caches receive structural validation; lm-eval uses correct joint BPE continuation scoring and actual batches.
  * **Expanded Verification**: `226 passed, 3 skipped`; architecture audit `57 passed, 1 documented legacy-quantization warning, 0 failures`.

### v0.20.5 (alpha)

  * **KortexHOS Release Profile**: Documents the best-known static full-precision chat settings for Alpaca assistant checkpoints: `temperature=0.4`, `top_k=40`, `top_p=0.9`, `repetition_penalty=1.15`, no passive learning, and zero injected previous-turn history.
  * **ROG Ally Benchmark Preset**: Adds `--benchmark-preset rog-ally` documentation for bounded local CPU evaluation with ARC Easy, HellaSwag, and TruthfulQA MC1.
  * **Benchmark/Chat Parity**: Notes that benchmark mode clears transient LTM state, suppresses Hebbian writes, and uses checkpoint-sized TBPTT chunking so local benchmark logits stay aligned with clean chat inference.
  * **Release Smoke Scores**: Records the current local `--eval-limit 100` sanity result: ARC Easy `0.3600`, HellaSwag `0.3400` raw / `0.3700` normalized, TruthfulQA MC1 `0.2200`.

### v0.20.4 (alpha)

  * **Inference-Like LTM Training**: Adds `--ltm-training-mode read-only` and `--inference-like-ltm-training` so assistant SFT can carry ROSA/history state without supervised fast-memory writes that chat generation cannot reproduce.
  * **Assistant Recovery Update**: `--assistant-recovery` now defaults to read-only LTM training unless explicitly overridden.
  * **DeepEmbed No-Decay**: DeepEmbed identity gates are excluded from AdamW weight decay to avoid quietly weakening RWKV channel mixing over long runs.

### v0.20.3 (alpha)

  * **DeepEmbed Channel-Mix Clamp**: Adds `--rwkv-channel-mix-deepembed-clamp` to cap DeepEmbed's multiplicative channel-mix modulation before `value_cm`.
  * **Resume/Override Guardrails**: Checkpoint hydration and explicit CLI override tests now cover both RWKV channel-mix clamps, matching the safety pattern used for ROSA-era config backfills.
  * **Post-DeepEmbed FFN Bound**: The channel-mix FFN input to `value_cm` is deterministically bounded when both key and DeepEmbed clamps are active.

### v0.20.2 (alpha)

  * **Channel-Mix Key Clamp**: Adds `--rwkv-channel-mix-key-clamp` to cap RWKV `key_cm` preactivation before the ReLU-squared channel-mix FFN.
  * **Resume-Safe Numeric Guard**: Older checkpoints backfill `rwkv_channel_mix_key_clamp=12.0`; the clamp does not change tensor shapes or checkpoint state layout.
  * **Compile-Compatible Stabilization**: The key clamp remains inside the RWKV hot path and is covered by the RWKV compile/integrity tests.

### v0.20.1 (alpha)

  * **Drift Clamp Rescue**: Adds `--drift-norm-clamp` and `--drift-delta-scale` so commit/drift-heavy runs can be resumed with bounded worker drift instead of resetting learned hierarchy weights.
  * **Straight-Through Commitment Cap**: Commitment-cost capping now keeps the forward auxiliary value bounded while preserving corrective gradient for over-cap raw commit drift.
  * **Recovery Recipe**: Documents a cooler checkpoint-resume recipe for `loss up + ponder up + commit up` instability, including lower model/LTM LR, tighter drift bounds, and adaptive ponder scaling.

### v0.20 (alpha)

  * **Assistant SFT Safety Guardrails**: Adds response-preserving truncation for prompt/completion rows, drops blank completions by default, and documents `--min-response-tokens` / `--allow-empty-completions`.
  * **Loss-Cap Safety**: `--max-ce-loss-for-backward` now defaults to `0.0` so cold-start LM training is not clamped below random-token CE.
  * **Weighted Assistant Recipe**: `--assistant-recovery` now documents the v0.20 recipe: prompt `0.10x`, response `1.0x`, first `32` response tokens `2.0x`, `16` reserved response tokens, `0.003` ponder loss, and `5000` memory-gate warmup steps.
  * **HF Token Cache Hygiene**: Cache keys include formatting, loss weights, response guardrails, chunking, and architecture settings. Resume hydration no longer persists one-shot cache refresh flags or stale CE-cap values from older checkpoints.

### v0.19.3 (alpha)

  * **Progress Sync Throttling**: Training progress metrics are now collected only on scheduled tqdm updates, reducing CUDA synchronization from per-step `.item()` logging.
  * **`--progress-log-steps`**: Train flag controls metric update cadence; default `25`, with `1` restoring every-step logging.
  * **Alpaca Prompt String**: The `--alpaca` formatter is documented as `### Instruction:`, optional `### Input:`, and `### Response:` before the supervised output string.

### v0.19.2 (alpha)

  * **DataLoader Throughput Tuning**: CUDA auto worker selection now defaults to a conservative worker pool for pre-tokenized training, avoiding unnecessary CPU contention and pinned-memory churn.
  * **Worker-Tied Prefetch**: Auto `prefetch_factor` now keeps total queued batches bounded relative to `num_workers`; explicit `--prefetch-factor` still overrides this when the input pipeline is truly the bottleneck.
  * **CUDA-Only Pinning**: DataLoader pinned memory is selected from the requested device, so CPU and DirectML training avoid pinning overhead while CUDA keeps async host-to-device transfers.
  * **Chunked Dataset Efficiency**: Pre-tokenized JSONL workers shard by file byte range, and `.pt` chunk loading keeps a small hot cache per worker.

### v0.19 (alpha)

  * **Optimization and GUI Update**: Release focus for CUDA/CPU math selection and the Windows GUI workflow.
  * **Adaptive LTM Math Paths**: LTM retrieval and memory updates keep the existing CPU-friendly dense one-hot/matmul path on CPU, while CUDA tensors automatically use gather/scatter-based math for better GPU utilization.
  * **Zero-Config Device Selection**: The architecture chooses the GPU-friendly LTM path internally when running on CUDA; no new CLI or GUI flag is required.
  * **ROSA Remains CPU-Side**: ROSA is intentionally unchanged so it stays fast, VRAM-light, and CPU-friendly.
  * **GUI Release Documentation**: Windows GUI bundle instructions are called out for portable `Hierarchos.exe` distribution with the bundled backend.

### v0.18 (alpha)

  * **🧠 RWKV v8 Backbone**: Complete replacement of GRU cells with RWKV v8 cells featuring:
      * **Time Mixing**: WKV (Weighted Key Value) recurrence with exponential decay and `time_first` / `time_decay` learnable parameters.
      * **Channel Mixing**: ReLU-squared feed-forward network with 4× expansion.
      * **5-Slot State**: `(sx, aa, bb, pp, sx_cm)` replaces the old 3-slot GRU state for richer temporal representation.
      * **Float32 WKV**: Critical exponential calculations run in float32 for numerical stability, even under AMP.
  * **🎨 DeepEmbed (4× Scale)**: New `h_deepemb` and `l_deepemb` embeddings at `hidden_dim × 4` that gate the RWKV channel mixing FFN, providing per-token modulation of the feed-forward pathway.
  * **🔮 ROSA (Rapid Online Suffix Automaton)**: A neurosymbolic inner monologue module:
      * CPU-side Suffix Automaton predicts likely next tokens from input history.
      * Predictions are embedded via `rosa_emb` and added to the input representation.
      * Gives the model a "heads up" about upcoming patterns (O(n) precomputation).
      * `past_tokens` state maintained across inference turns for continuity.
  * **⚡ CUDA Datacenter Auto-Optimization** (zero config):
      * **AMP auto-enable**: Mixed precision activates on CUDA without `--amp` flag.
      * **bfloat16 on Ampere+**: SM ≥ 8.0 GPUs use bf16 (no GradScaler overhead).
      * **TF32 matmul**: 3-8× faster linear layers on Ampere+.
      * **cuDNN benchmark**: Auto-tunes kernel selection for hardware.
      * **torch.compile auto-enable**: Worker loop compiled on CUDA.
      * **Non-blocking transfers**: `to(device, non_blocking=True)` for async H2D.
      * **pin_memory on CUDA**: Enables async H2D when CUDA training is active.
      * **bounded DataLoader prefetch**: Auto prefetch stays tied to worker count.
      * **drop_last on CUDA**: Prevents irregular batch OOM.
  * **🧪 Test Suite Modernized**: 11/11 tests pass. Rewrote 3 stale tests (`test_forward.py`, `test_inference.py`, `verify_parity_deep.py`) to be self-contained — create models in-memory instead of loading hardcoded checkpoints.
  * **🛡️ Stability Hardening**:
      * `ltm_state` detach handles both 2-tuple and 3-tuple formats (forward compat).
      * `verify_ltm_decay.py` and `verify_momentum_inference.py` fixed for correct tuple unpacking.
  * **🔙 V7 Backward Compatibility**: Setting `use_deepembed=False, use_rosa=False` produces a valid V7 model. All V7 checkpoints load cleanly.
  * **📦 HuggingFace Directory Output Restored**: Training exports `hierarchos.pt` + full tokenizer suite + `hierarchos_config.json` in a self-contained directory.
  * **🆕 CLI Additions**: `--no-amp` flag, improved help text for `--amp`, `--compile`, `--num_workers`.
  * **📊 GPU Diagnostics**: Training startup prints GPU name, VRAM, SM version, and all auto-enabled optimizations.

### v0.17 (alpha)

  * **LM-Evaluation-Harness Integration**: Added optional benchmarking during/after training.
  * **HierarchosLM Wrapper**: Custom implementation of `loglikelihood`, `loglikelihood_rolling`, and `generate_until` for full compatibility with `lm-eval`.
  * **Periodic Step-Based Eval**: Added `--eval-steps` to trigger evaluation every N steps for high-granularity progress tracking.
  * **Configurable Eval**: Added `--eval-every-epoch`, `--eval-batch-size`, and `--eval-limit` control flags.
  * **Startup Confirmation**: Training now confirms if evaluation is enabled at launch.

### v0.16.2.1 (alpha)

  * **⚠️ CRITICAL: LTM Threshold Bugfix**:
      * Fixed bug where passive learning updated LTM on *every* turn, regardless of threshold
      * Could corrupt model weights over time — **restore from backup if you used v0.16.1-v0.16.2**
      * Added `compute_only` parameter to separate loss computation from actual updates
  * **Repetition Penalty**: `--repetition-penalty` (default 1.2) prevents output loops
  * **Passive Learning**: LTM learns from conversations automatically (threshold-gated)
  * **Checkpoint Converter**: `ckpt-2-inf` mode for HuggingFace-style directories
  * **First Coherent Release**: 25M model trained on Alpaca produces coherent output

*(Older changelog entries have been archived for brevity. See git history for versions prior to v0.16.)*

-----

© 2026 Makhi Burroughs
