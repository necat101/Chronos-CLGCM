# Hierarchos Architecture Audit

> **2026-07-28 coherent-v9 remediation:** The issues identified after the
> epoch-15 coherence investigation have been implemented as a new
> learned-function revision. Corrected recurrence/ACT, bounded ROSA, shared
> token adapters, memory and warmup-state parity, deterministic data/resume
> identity, fail-closed non-finite handling, checkpoint contracts, and immutable
> best-metric selection are documented in
> [COHERENT_V9_MIGRATION.md](COHERENT_V9_MIGRATION.md). Existing v2/v3 weights
> retain `legacy-v8` behavior; they cannot be relabeled as corrected v9 weights.
> The supported path is full-precision `hierarchos_cli.py`. Quantized `.npz`
> paths and direct execution of the legacy monolith now fail closed.

> **2026-08-01 release-gate status:** This May audit is retained as historical
> architectural context. The supported coherent-v9 tree currently passes all
> `89/89` strict executable architecture checks. Run both release-gate commands
> in [COHERENT_V9_MIGRATION.md](COHERENT_V9_MIGRATION.md) on the candidate tree;
> the broad pytest total is intentionally not hardcoded because the regression
> suite grows with each remediation. The older epoch-13 control remains
> documented in [EPOCH13_CHECKPOINT_AUDIT.md](EPOCH13_CHECKPOINT_AUDIT.md).

Audit date: 2026-05-23

Scope: local repository state under `C:\Users\User\Downloads\Hierarchos-main GUI experiment\Hierarchos-main`. This audit describes the implementation that exists now, not the stronger aspirational claims made in the README.

## Executive Summary

Hierarchos is best described as an experimental memory-augmented recurrent language model with hierarchical iterative control.

The active architecture is not a Transformer and should not be presented as evidence of AGI. It is a PyTorch language-model prototype that combines:

1. RWKV-style recurrent sequence processing.
2. A two-level manager/worker control loop inspired by hierarchical reasoning.
3. A differentiable slot memory with slow learned memory and fast working-memory state.
4. An optional suffix-automaton feature path called ROSA.
5. Training and inference utilities, including a Rust GUI that talks to a Python backend.

The strongest accurate scientific framing is:

> Hierarchos is a recurrent, memory-augmented language-model architecture that combines RWKV-like matrix-state recurrence, hierarchical manager-worker iterative refinement, and a differentiable slot-based associative memory. It explores whether explicit recurrent state, retrieved memory slots, and bounded internal refinement loops can provide useful alternatives to pure Transformer scaling for local language-model experimentation.

## What The Active Model Is

### Core model

The active model class is `HierarchosCore` in `hierarchos/models/core.py`. The modular package is the active path used by `hierarchos_cli.py` and the GUI bridge, while the older top-level `hierarchos.py` remains as a legacy monolith.

The forward pass has this practical structure:

1. Token ids are embedded with a tied input/output embedding matrix.
2. If enabled, ROSA predicts pattern-continuation token ids from suffix-automaton logic, embeds those ids, and injects them through a learned gate/router.
3. The model queries long-term memory using the current token representation and previous high-level context.
4. Top-k memory values are retrieved, timestamp-encoded, gated, and concatenated with the token embedding and a persistent learned vector.
5. A high-level RWKV recurrent module updates continuously and periodically performs ACT-style refinement to generate a strided context plan.
6. A low-level RWKV recurrent module iteratively refines a local drift vector against that context plan.
7. The final per-token hidden state is normalized and projected through the tied language-model head.
8. Training uses cross-entropy, optional z-loss, ponder cost, and commitment/drift cost.

Relevant implementation anchors:

- `hierarchos/models/core.py:211`: active `HierarchosCore`.
- `hierarchos/models/core.py:246`: token embedding.
- `hierarchos/models/core.py:249`: optional DeepEmbed path.
- `hierarchos/models/core.py:259`: optional ROSA path.
- `hierarchos/models/core.py:282`: LTM module construction.
- `hierarchos/models/core.py:345`: tied input/output token weights.
- `hierarchos/models/core.py:431`: model forward method.
- `hierarchos/models/core.py:602`: main token loop.
- `hierarchos/models/core.py:663`: strided manager planning step.
- `hierarchos/models/core.py:731`: worker-loop invocation.
- `hierarchos/models/core.py:855`: chunked language-model loss path.

### RWKV recurrent cell

The active recurrent block is not the older scalar-state RWKV variant left in the monolith. It is a newer matrix-state RWKV/Heron-style cell in `hierarchos/models/rwkv_cell.py`.

It maintains a packed recurrent state of shape `[batch, channels, 3 + head_size]`, where the final segment is reshaped into a per-head matrix state. The cell uses time mixing, learned decay/adaptation terms, a matrix-state recurrence, group normalization over the WKV output, and a ReLU-squared channel-mixing branch. DeepEmbed, when enabled, multiplicatively modulates the channel-mixing feed-forward activation.

Relevant anchors:

- `hierarchos/models/rwkv_cell.py:52`: active `RWKVCell`.
- `hierarchos/models/rwkv_cell.py:59`: packed state layout.
- `hierarchos/models/rwkv_cell.py:73`: state size is `3 + head_size`.
- `hierarchos/models/rwkv_cell.py:241`: matrix-state view.
- `hierarchos/models/rwkv_cell.py:282`: matrix-state update.
- `hierarchos/models/rwkv_cell.py:305`: DeepEmbed modulation.
- `hierarchos/models/rwkv_cell.py:309`: new recurrent state construction.

### Long-term memory

The LTM module is a slot-based associative memory. It has learned slow keys and values, plus a fast working-memory value buffer and momentum buffer. Retrieval computes key-query similarity, selects top-k slots, and returns the corresponding memory values and timestamps. Training performs a gradient-derived fast-memory update after backward using gradients retained on retrieved memory values. Evaluation/chat can optionally perform a Hebbian-style update, but plain eval generation is read-only by default.

This is best called "Titans-style" or "inspired by neural memory systems." It should not be described as a faithful reproduction of Google's Titans architecture unless that equivalence is separately demonstrated.

Relevant anchors:

- `hierarchos/models/ltm.py:6`: LTM module.
- `hierarchos/models/ltm.py:23`: learned slow-memory keys.
- `hierarchos/models/ltm.py:31`: fast working-memory buffer.
- `hierarchos/models/ltm.py:46`: timestamp metadata.
- `hierarchos/models/ltm.py:291`: top-k retrieval.
- `hierarchos/models/ltm.py:419`: gradient-based inner update.
- `hierarchos/models/ltm.py:624`: Hebbian update wrapper.

### ROSA

ROSA is a deterministic suffix-automaton preprocessing path. Given token histories, it predicts likely continuation token ids based on repeated suffix patterns. The model then embeds those ids and learns how much to use the signal. This is not a symbolic reasoner; it is an auxiliary exact-pattern feature generator.

Relevant anchors:

- `hierarchos/utils/rosa.py:102`: persistent ROSA state object.
- `hierarchos/utils/rosa.py:135`: reference suffix-automaton implementation.
- `hierarchos/utils/rosa.py:485`: precomputed ROSA ids for chunks.
- `hierarchos/utils/rosa.py:539`: async ROSA pipeline.

### Training system

The trainer uses truncated backpropagation through time over chunks, carries recurrent and memory states within a sequence, and resets cross-batch state by default unless `persist_state` is enabled. It retains gradients on retrieved memory values, then applies a Titans-style memory update to the fast memory state. CUDA-specific options include AMP, bfloat16 selection on supported hardware, TF32, `torch.compile`, bounded DataLoader prefetch, and chunked loss to avoid full large-vocabulary logits during training.

Relevant anchors:

- `hierarchos/training/trainer.py:270`: `train_step`.
- `hierarchos/training/trainer.py:339`: default cross-batch state reset.
- `hierarchos/training/trainer.py:414`: model call per TBPTT chunk.
- `hierarchos/training/trainer.py:485`: gradient-derived LTM update.
- `hierarchos/training/trainer.py:621`: training entry point.

## What It Is Not

Hierarchos should not currently be presented as:

- Proof of AGI or a decisive step toward AGI.
- A validated replacement for Transformer scaling.
- A complete reproduction of Titans, HRM, or RWKV-v8 papers.
- A benchmark-proven architecture with demonstrated superiority.
- A mature production inference stack.

Those claims are not supported by the repository evidence. The repo contains promising engineering tests and small smoke validations, but not scientific benchmark tables, ablation studies, scaling curves, peer-reviewed comparisons, or reproducible external evaluation results.

## Current Implementation Split

There are two major code paths:

1. Active modular package: `hierarchos/`, used by `hierarchos_cli.py` and the Rust GUI bridge.
2. Legacy monolith: top-level `hierarchos.py`, which still contains older architecture code, older CLI defaults, and quantization/export logic.

This matters because README claims and some legacy code refer to features that are not fully integrated in the active modular CLI.

Notable mismatch:

- The modular `QuantizedRWKVCell` expects older names such as `time_decay`, `time_first`, and `receptance_cm`.
- The active modular `RWKVCell` uses matrix-state parameters such as `x_r`, `w0/w1/w2`, `a0/a1/a2`, `r_k`, `key_cm`, and `value_cm`.
- The modular CLI accepts `quantize` as a mode, but the execution branch currently falls through to "not yet fully integrated."

Therefore, current quantized CPU/Vulkan inference is disabled and fails closed.
The older implementation remains only as historical reference until a
matrix-state exporter and loader are verified end to end.

## Historical May Verification

The following list records the environment-limited May audit, not the current
release gate. Current candidates must use the two commands linked in the status
note above.

Full `pytest` execution could not be used in that historical environment because `pytest` was not installed in its available Python runtime. The Python 3.12.13 executable visible through the Python launcher also lacked `torch`, so it could not run Hierarchos. The torch-capable runtime available during that audit was the PowerShell-resolved Python 3.13 WindowsApps executable.

Direct checks run successfully:

- ROSA correctness: 8 unittest checks passed.
- RWKV-v8 integrity: 14 unittest checks passed.
- Forward pass smoke test: passed.
- Inference loop smoke test: passed.
- Inference memory gating test: passed.
- Context drift suite: 7 checks passed.
- Architecture flag inference: 3 unittest checks passed.
- CUDA/default tuning logic: 5 unittest checks passed.
- LTM, sampling, ACT, z-loss, and coherence direct runner: 40 checks passed.
- Dataset streaming/direct runner: 21 checks passed.
- Gradient-flow validation script: 4 checks passed.

Important observed caveat:

- `test_hierarchos.py` skips compile testing on Windows CPU.
- `test_gradient_flow.py` reported `val_proj.weight` missing gradients in one end-to-end gradient check. That may be expected because `val_proj` is used for explicit Hebbian/inference updates rather than the default supervised loss path, but it should be documented as a role-specific path rather than universal gradient flow.

## Scientific Description For External Communication

### Recommended short description

Hierarchos is an experimental recurrent memory-augmented language model. It combines RWKV-style matrix-state recurrence, a hierarchical manager-worker refinement loop, and a differentiable slot-based long-term memory. It is designed to investigate whether local recurrent state, explicit memory retrieval/update, and bounded internal iterative computation can improve small-model reasoning and continuity without relying solely on Transformer attention scaling.

### Recommended abstract

We present Hierarchos, an experimental memory-augmented recurrent language-model architecture. The model replaces Transformer self-attention with RWKV-style recurrent cells that maintain compact per-token state. A high-level manager recurrent module produces a strided context plan using adaptive refinement, while a low-level worker recurrent module performs bounded iterative updates against that plan through a learned drift state. The token stream is augmented by a differentiable slot memory that retrieves top-k learned memory values using context-aware queries and updates fast memory state from retrieved-value gradients during training. An optional deterministic suffix-automaton path, ROSA, supplies exact repeated-pattern continuation hints through a learned embedding gate. The system is implemented as a PyTorch research prototype with training, chat, checkpoint, dataset streaming, and GUI tooling. Current evidence supports functional gradient flow, memory isolation, recurrent-state continuity, and several implementation-level invariants on small tests; it does not yet establish benchmark superiority, scaling behavior, or claims about general intelligence.

### Recommended methods-language description

For each token, Hierarchos embeds the token id and optionally adds a learned gated embedding of a ROSA suffix-automaton prediction. It constructs a memory query from the token embedding and prior high-level context, retrieves top-k memory slots from a learned key-value memory, adds timestamp encodings to retrieved values, and gates the retrieved memory before concatenating it with a persistent learned vector. The resulting representation is processed by a high-level RWKV manager. At configurable strides, the manager performs adaptive refinement over a shadow recurrent state to produce a future context target, and intermediate tokens interpolate between previous and target contexts. A low-level RWKV worker then iteratively refines the token representation against this interpolated context through a drift vector. The final hidden representation is normalized and projected through a tied language-model head. Training optimizes cross-entropy with optional z-loss, ponder cost, and drift commitment regularization, while a separate fast-memory update is computed from gradients on retrieved memory values.

## Claims Table

| Claim | Accurate status |
| --- | --- |
| "Uses RWKV-style recurrence" | Supported. Active cells implement matrix-state recurrent RWKV-like time mixing and channel mixing. |
| "Has a hierarchical manager/worker loop" | Supported. The active forward pass has manager planning and worker iterative refinement. |
| "Has long-term memory" | Supported if framed as learned slot memory plus fast working memory. |
| "Implements Titans" | Partially supported only as inspiration. It is a Titans-style fast-memory update, not a demonstrated faithful reproduction. |
| "Implements HRM" | Partially supported as HRM-inspired hierarchical iterative control, not a validated reproduction. |
| "ROSA is neurosymbolic reasoning" | Overstated. It is deterministic suffix-automaton pattern prediction injected as a learned feature path. |
| "Outperforms Transformers" | Not established by this repo. |
| "AGI path/decisive step" | Not scientifically supported. |
| "Quantized Vulkan inference is current" | False. The legacy path is disabled; coherent-v9 is full-precision only. |
| "GUI is part of the architecture" | No. The GUI is tooling around the Python backend, not a model component. |

## Open Scientific Questions

1. Does the memory system improve perplexity, long-context behavior, or sample efficiency compared with matched-parameter RWKV and Transformer baselines?
2. Which components matter? Required ablations include no-LTM, no-ROSA, no-DeepEmbed, single-level recurrent control, no-worker-iteration, and no-gradient memory update.
3. Does the fast-memory update remain stable at scale and under real instruction tuning?
4. Does the manager's ACT-style ponder loop learn useful dynamic compute allocation, or is it mostly regularization/overhead?
5. Does ROSA provide measurable benefit beyond n-gram-like repetition priors?
6. Can quantized inference be brought into parity with the active matrix-state RWKV implementation?
7. How does training throughput and memory use compare against simpler RWKV baselines at the same parameter count?

## Recommended Public Framing

Use:

- "experimental recurrent memory-augmented language model"
- "RWKV-like matrix-state recurrent backbone"
- "hierarchical manager-worker iterative refinement"
- "Titans-style differentiable fast memory"
- "suffix-automaton auxiliary pattern feature"
- "research prototype"

Avoid:

- "AGI"
- "human-like cognition"
- "revolutionary"
- "decisive step beyond scale"
- "proven Transformer replacement"
- "full Titans plus HRM plus RWKV implementation" unless backed by formal parity tests and benchmark results.

## Bottom Line

Hierarchos is a serious and technically interesting research prototype. Its real contribution, as implemented now, is not an AGI claim; it is an attempt to integrate recurrent sequence modeling, explicit differentiable memory, deterministic pattern hints, and hierarchical iterative computation into one trainable local language-model system.

The scientifically honest description is that it is a memory-augmented recurrent hierarchical LM architecture under active experimentation. Its novelty should be argued through controlled ablations and benchmarks, not through aspirational terminology.
