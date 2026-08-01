# Hierarchos Alpha v0.30: RWKV-v9 core migration guide

**Hierarchos Alpha v0.30** is the product release. Its supported fresh-run
RWKV-v9 core uses the compatibility-stable internal identifier `coherent-v9` in
CLI/config/checkpoint metadata; that identifier is not the product name. The
core is a new learned-function revision, not a metadata rename for v2/v3
weights. Existing checkpoints without an architecture revision are loaded as
`legacy-v8` so their historical function is preserved.

## What was corrected

- Manager ACT now makes a hard, per-row cumulative-halt decision in both
  training and inference. Its auxiliary loss reports the depth actually
  executed at inference and uses a straight-through smooth gradient.
- Manager and worker recurrence use the corrected state/readout/commit
  contract. Padding rows freeze every carried state, and TBPTT detachment covers
  every recurrent carrier.
- DeepEmbed and ROSA token conditioning use low-rank adapters over the tied
  token embedding instead of three additional vocabulary-sized tables.
- ROSA has a versioned, deterministic bounded-context mode. The no-prediction
  sentinel contributes exactly zero rather than being clamped to a real token.
- LTM masks invalid retrievals, keeps token and wall timestamps as
  metadata-only filters/provenance instead of injecting full-amplitude absolute
  time features, preserves update-source metadata, trains the value writer
  causally, and keeps batch-local working memory isolated.
- Memory-gate warmup position is checkpoint state and is applied identically in
  train, evaluation, prefill, and generation.
- Non-finite recurrent states, logits, losses, gradients, optimizer state, and
  learned checkpoint tensors fail closed. NaN/Inf values are not silently
  rewritten into plausible finite outputs.
- Token-cache format v6 records ordered dataset content, tokenizer identity,
  formatting, truncation/rejection statistics, and ROSA semantics. Exact resume
  checks those identities before training.
- Cached ROSA training consumes the precomputed prediction stream directly. It
  no longer copies every accelerator chunk back to the host merely to rebuild a
  duplicate token-history carrier that cannot affect cached predictions.
- The canonical trainer validates right-padding, labels, and loss weights once
  on the host and reuses the checked padding geometry across TBPTT chunks. Long
  CUDA samples no longer incur one device synchronization per chunk merely to
  repeat an unchanged batch-contract audit.
- Gradient accumulation defaults to supervised-token-weighted normalization.
  New optimizers use corrected parameter grouping; legacy exact resumes retain
  their saved grouping.
- Optimizer, scaler, main scheduler, LTM scheduler, pending gradients, RNG, and
  dataloader state are restored independently. Resuming no longer rebuilds a
  warmup schedule unless explicitly requested.
- Checkpoint format v4 stores a hashed architecture contract. Current
  checkpoints reject contract, tensor-geometry, tokenizer, or exact-run drift.
- Independent-sample runs omit terminal recurrent/ROSA carriers from periodic
  checkpoints because the next batch is guaranteed to reset them. Explicitly
  contiguous streams still preserve the complete state for exact resume.
- Chat-state v4 records that same contract plus exact recurrent layouts and is
  bound to the originating model weights and tokenizer; it is not a portable,
  model-neutral conversation file.
- Optional best-checkpoint selection accepts one immutable, finite lm-eval
  metric and saves an exact-resume `hierarchos_best.pt` only on improvement.

## Online adaptation remains part of Alpha v0.30

The RWKV-v9 design keeps a stable learned model and a small writable LTM tier.
`fast_vals` is that transient tier: retrieved fast values let one chat adapt to
validated feedback without rewriting the base language-model weights. It is not
dead state. Normal prefill and token generation read it but do not mutate it;
only a discrete, policy-authorized transaction may produce a new fast state.
Conversation state and optional LTM overlays can preserve the transient tier,
while a clean base-model export does not silently bake chat-specific writes into
the model.

Training and runtime controls intentionally describe different boundaries:

- `--ltm-training-mode read-only` disables target-gradient inner writes to fast
  slots during SFT. The normal CE objective still trains LTM keys, slow values,
  query/gate/readout paths, and the causal value-writer auxiliary. The flag does
  not remove runtime adaptation.
- `--no-persist-state` prevents unrelated shuffled dataset rows from sharing
  H/L/context/LTM carriers. Within one sample, forward state still crosses all
  TBPTT chunks; only the backward graph is truncated at a chunk boundary. The
  flag says nothing about post-training chat transactions.
- Fresh RWKV-v9 training uses value-alignment weight `0.01`, stride `8`, and a
  minimum of `100` successful writer-training optimizer updates. Writer
  readiness also requires a finite alignment-loss EMA no greater than `0.95`
  and a finite `val_proj` weight norm no greater than `64.0`.
- `inner-update` remains available only as a legacy/Titans ablation. It exposes
  target-gradient fast writes that ordinary autoregressive generation cannot
  reproduce, so it is not the Alpha v0.30 default.

Every online write is transactional. Hierarchos clones the live fast state,
computes a bounded candidate update, clips and checks finite/norm budgets,
backs off the LR when needed, and replays the local objective. It commits only
when the candidate is finite and non-worsening; otherwise the original fast
state remains byte-for-byte authoritative. Ordinary prefill/generation remains
read-only under every policy, including the passive policies: any permitted
write happens only after the observed prompt or completed response reaches its
transaction boundary.

| `--online-adaptation-policy` | Runtime contract |
| --- | --- |
| `off` | No fast-memory writes, including explicit actions. Use for static evaluation and raw-logit parity. |
| `validated` (default) | Explicit `/learn`, `/reject`, and `/correct <target>` transactions only. No passive or heuristic feedback writes. |
| `prompt` | `validated` plus conservative passive writes from observed user prompts. Generated model answers remain write-ineligible. Legacy `--passive-learning` maps here. |
| `prompt+response` | `prompt` plus quality/surprise-gated writes after completed model responses. This is the highest-risk, explicit opt-in. Legacy `--passive-response-learning` maps here and requires prompt-passive learning. |

`/learn` reinforces the last completed response. `/reject` applies an
unlikelihood objective to it. `/correct <target>` rejects the previous answer
and then learns the supplied replacement. Heuristic natural-language praise,
rejection, and correction detection is off by default; enable
`--natural-feedback-detection` only when that ambiguity is intentional. GUI
positive/negative feedback actions use the same transactional boundary rather
than a separate unsafe writer path.

The training `--ltm-lr`/`--min-ltm-lr` schedule is separate from runtime
feedback, whose explicit transaction LR is controlled by `--online-ltm-lr`
(default `1e-3`). A ready writer does not imply that training or chat will write
automatically; the training mode and runtime policy still govern mutation.

## Checkpoint compatibility

| Operation | Supported behavior |
| --- | --- |
| Resume an Alpha v0.30 RWKV-v9 (`coherent-v9`) training checkpoint | Exact resume; identity and all available optimizer/scheduler/data state are verified. |
| Chat with an Alpha v0.30 RWKV-v9 (`coherent-v9`) inference checkpoint | Supported full-precision path; architecture contract is verified before construction. |
| Resume or chat with v2/v3 | Loaded as `legacy-v8`; its historical learned function is preserved. |
| Turn v2/v3 into RWKV-v9 by changing a flag | Unsupported. The adapter geometry and recurrent/objective semantics differ; train Alpha v0.30's RWKV-v9 core from a fresh initialization. |
| Expand a model's dimensions | Supported only within the authenticated source revision. `expand_model.py` binds the exact tokenizer IDs, uses semantic-block projection mapping, releases resume-only optimizer state before allocating the larger model, and atomically publishes a hashed package. Vocabulary-ID or legacy-v8-to-v9 migration is intentionally refused. |
| Continue only the weights with a fresh optimizer | Use `--model-path`, understanding that this begins a new run rather than an exact resume. |
| Quantized `.npz` chat/export | Intentionally unsupported. The old scalar-RWKV implementation cannot reproduce the active matrix-state architecture. |
| Run the top-level `hierarchos.py` monolith | Unsupported. Use `hierarchos_cli.py`; the monolith is historical reference code. |

No software change can retroactively make old v2/v3 weights learn the corrected
RWKV-v9 function. They remain usable controls, but a scientifically clean
RWKV-v9 result requires new training.

## Data preflight before an expensive run

Build and audit the cache before allocating the model:

```bash
python hierarchos_cli.py train \
  --architecture-revision coherent-v9 \
  --hf_dataset "netcat420/Experiment_0.1" \
  --hf_dataset_split train \
  --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" \
  --tokenizer-path "openai-community/gpt2" \
  --alpaca \
  --assistant-recovery \
  --max_length 8880 \
  --training-chunk-size 256 \
  --train-prompt-tokens \
  --prompt-loss-weight 0.10 \
  --response-loss-weight 1.0 \
  --response-boundary-loss-weight 2.0 \
  --response-boundary-tokens 64 \
  --min-response-tokens 16 \
  --hf-token-cache \
  --hf-token-cache-dir "/content/hierarchos_token_cache/experiment_0_1_alpha_v030" \
  --token-cache-build-batch-size 256 \
  --num_workers -1 \
  --token-cache-only
```

Inspect `cache_audit.json`. The default rejected-row budget is zero and the
default truncated-row budget is 5%, so schema mistakes and unexpected data loss
stop the run. If a larger threshold is intentional, set an explicit
`--max-cache-rejected-samples` or `--max-cache-rejected-fraction` only after
reviewing the reasons. In particular, `--min-response-tokens 128` rejects every
shorter answer; use it only if the audit proves that this is the desired corpus.

The preflight repeats every cache-affecting data/objective option from the full
run below. In particular, the response-boundary length must match; otherwise a
different cache key is correct and training will rebuild the cache.

## Starting a fresh Alpha v0.30 run with the RWKV-v9 core

`coherent-v9` is the new-run default, but spelling the internal architecture
identifier out makes experiment records self-documenting. The following is a
literal one-line Colab/autoclicker command; it is one physical line even if the
Markdown renderer wraps it visually:

```bash
!cd ./Hierarchos && python hierarchos_cli.py train --architecture-revision coherent-v9 --hf_dataset "netcat420/Experiment_0.1" --hf_dataset_split "train" --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" --out-dir "./chatHRM_alpha_v030" --tokenizer-path "openai-community/gpt2" --epochs 15 --batch_size 64 --accumulation-steps 1 --accumulation-normalization weighted-token --max_length 8880 --training-chunk-size 256 --no-full-sample-bptt --no-full-sample-activation-checkpointing --detach-every-n-steps 0 --no-persist-state --context_dim 448 --h_hidden 448 --l_hidden 448 --rwkv-head-size 64 --max_h_steps 5 --max_l_steps 5 --alpaca --assistant-recovery --ltm-training-mode read-only --ltm-value-alignment-weight 0.01 --ltm-value-alignment-stride 8 --ltm-value-alignment-min-updates 100 --train-prompt-tokens --prompt-loss-weight 0.10 --response-loss-weight 1.0 --response-boundary-loss-weight 2.0 --response-boundary-tokens 64 --min-response-tokens 16 --starting-lr 2e-5 --min-lr 1e-7 --warmup-ratio 0.01 --ltm-lr 3e-4 --min-ltm-lr 1e-5 --adaptive-ponder --ponder-target-scale 0.65 --ponder-loss-weight 0.003 --commitment-loss-weight 0.5 --max-commitment-cost-for-backward 4.0 --max-ce-loss-for-backward 0 --max-ponder-cost-for-backward 0 --startup-weight-max-abs 0 --halt-logit-clamp 30.0 --recurrent-state-clamp 50.0 --context-state-clamp 50.0 --activation-clamp 100.0 --drift-state-clamp 2.0 --drift-norm-clamp 4.0 --rwkv-channel-mix-key-clamp 12.0 --rwkv-channel-mix-deepembed-clamp 4.0 --drift-delta-scale 0.35 --memory-gate-warmup-steps 5000 --memory-gate-warmup-floor 0.10 --grad-clip 0.75 --device cuda --amp --force-compile --compile-mode max-autotune-no-cudagraphs --compile-static-worker-loop --hf-token-cache --hf-token-cache-dir "/content/hierarchos_token_cache/experiment_0_1_alpha_v030" --token-cache-build-batch-size 256 --length-bucket-auto-sample-size 1000000 --cuda-prefetch --cuda-loss-chunk-rows 0 --num_workers -1 --padding-metric-steps 0 --save-steps 600
```

This retains the broad direction of the previous run while changing the
response-length floor from 128 to a safer 16. Keep 128 only if the cache audit
justifies it. The writer auxiliary adds no parameters and samples only one in
eight causal token positions to keep its compute bounded. `read-only` still
prevents target-gradient fast-slot writes; readiness training does not mutate
the batch-local memory. The command records a non-inert `3e-4` to `1e-5` LTM LR
range and does not disable its saved schedule; the training mode remains the
authority over actual slot mutation.

This is a fresh learned-function run: never append a `legacy-v8` checkpoint or
the earlier `--resume-from-ckpt` path to this command. This project uses the
measured Colab profile of batch `64`, accumulation `1`, one optimizer step per
batch, and no WorkerLoop activation checkpointing. Re-run a longest-row memory
preflight if the Colab GPU type or runtime image changes.

## Exact mid-epoch resume

Use the same immutable dataset and output directory and add the checkpoint:

```bash
python hierarchos_cli.py train \
  --hf_dataset "netcat420/Experiment_0.1" \
  --hf_dataset_split train \
  --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" \
  --out-dir "./chatHRM_alpha_v030" \
  --epochs 15 \
  --resume-from-ckpt "./chatHRM_alpha_v030/hierarchos_epoch_14_step_61800.pt"
```

The checkpoint hydrates the remaining training configuration and exact-resume
identity. Do **not** add `--override-scheduling` to a normal resume. That legacy
flag deliberately means both `--reset-optimizer-state` and
`--rebuild-lr-schedule`.

Use the narrower flags only for an intentional recovery experiment:

- `--rebuild-lr-schedule` keeps optimizer/scaler moments but constructs a new
  schedule over the remaining work.
- `--reset-optimizer-state` discards optimizer/scaler moments but does not, by
  itself, request a new LR schedule.

## Full-precision inference parity

The raw-logit reference command removes sampling transformations and online
state mutation:

```powershell
$env:PYTHONUTF8="1"; python hierarchos_cli.py chat `
  --model-path "C:\path\to\chatHRM_alpha_v030" `
  --alpaca `
  --device cpu `
  --threads 16 `
  --no-amp `
  --chat-prefill-chunk-size 256 `
  --chat-input-history-turns 0 `
  --online-adaptation-policy off `
  --no-passive-learning `
  --temperature 0 `
  --top-k 0 `
  --top-p 1 `
  --repetition-penalty 1 `
  --max-new-tokens 192 `
  --entropy-stop-threshold 0 `
  --eos-stop-prob 0
```

Parity means the same checkpoint, tokenizer, architecture contract, recurrent
boundary policy, carried state, dtype, and backend within numerical tolerance.
It does not mean sampled text must match under temperature/top-k/top-p changes,
or that CPU FP32 must be bit-identical to CUDA BF16.

## Multi-turn continuity on optimized TBPTT checkpoints

`--no-persist-state` resets recurrent values between unrelated training rows;
it does not reset them between the 256-token chunks of one row. H/L state,
manager context, drift, LTM/ROSA state, and absolute position continue across
every forward chunk. TBPTT truncates the backward graph at a chunk boundary,
not the forward conversational state. The checkpoint records that geometry, and
chat prefill reuses it by default.

The safest CLI multi-turn mode re-encodes a bounded number of earlier turns in
the same Alpaca `### Previous Context` field used by the training pipeline:

```bash
python hierarchos_cli.py chat \
  --model-path "./chatHRM_alpha_v030" \
  --alpaca \
  --chat-input-history-turns 4 \
  --chat-input-history-chars 3000 \
  --online-adaptation-policy validated \
  --no-passive-learning
```

For an explicitly continuous hidden-state stream, use recurrent carry instead:

```bash
python hierarchos_cli.py chat \
  --model-path "./chatHRM_alpha_v030" \
  --alpaca \
  --carry-chat-state \
  --chat-state-file auto \
  --chat-input-history-turns 0 \
  --online-adaptation-policy validated \
  --no-passive-learning
```

Choose one continuity representation for a baseline run. Carried recurrent
state suppresses textual history replay so prior turns are not processed twice.
The `validated` policy preserves exact-action online learning without passive
prompt or generated-response writes; use `off` instead when the baseline must
forbid even `/learn`, `/reject`, and `/correct` transactions.
Current CLI chat-state files persist the effective TBPTT prefill geometry,
absolute boundary phase, and bounded turn history. Resuming a chat-state file
automatically enables recurrent carry and rejects an explicit conflicting chunk
size. Older state files remain loadable. The GUI bridge already uses the
continuous hidden-state form and keeps absolute TBPTT offsets until the user
resets the conversation.

The runtime can preserve continuity, but useful multi-turn behavior still
depends on training examples containing relevant prior-turn text. In Alpaca
mode, the dataset `input` field is formatted as `### Previous Context`.

## Release gate

Before committing a multi-billion-token budget:

```bash
python -m pytest -q
python tools/check_architecture_integrity.py --strict
```

Then run a small Alpha v0.30 RWKV-v9 overfit and exact-resume experiment with
the real tokenizer and cache. Verify:

1. loss falls on the tiny set;
2. an interrupted/resumed run matches the uninterrupted LR and next update;
3. chat and evaluation raw logits match the direct model path;
4. `cache_audit.json` has the intended schema and response-length distribution;
5. no skipped training batch is tolerated unless an explicit nonzero budget was
   reviewed in advance.

The strict executable architecture audit currently validates all `89/89`
checks. The broad pytest count is intentionally not frozen in this document as
the regression suite grows; a release candidate is ready only when both
commands above finish without failures. A separate CLI smoke run verified fresh checkpoint v4
training, safe `ROSAState` serialization, exact mid-epoch optimizer/scheduler
resume, matching subsequent losses, final inference export, architecture-hash
verification, finite reloaded logits, and authenticated atomic expansion with
tokenizer/provenance verification.
