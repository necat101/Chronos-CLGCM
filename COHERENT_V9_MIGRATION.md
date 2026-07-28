# Coherent-v9 training and inference contract

Coherent-v9 is the supported architecture for a new Hierarchos training run. It
is a new learned-function revision, not a metadata rename for v2/v3 weights.
Existing checkpoints without an architecture revision are loaded as
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
- LTM masks invalid retrievals after all timestamp features, separates token
  clocks from wall-clock recency, preserves update-source metadata, and keeps
  batch-local working memory isolated.
- Memory-gate warmup position is checkpoint state and is applied identically in
  train, evaluation, prefill, and generation.
- Non-finite recurrent states, logits, losses, gradients, optimizer state, and
  learned checkpoint tensors fail closed. NaN/Inf values are not silently
  rewritten into plausible finite outputs.
- Token-cache format v6 records ordered dataset content, tokenizer identity,
  formatting, truncation/rejection statistics, and ROSA semantics. Exact resume
  checks those identities before training.
- Gradient accumulation defaults to supervised-token-weighted normalization.
  New optimizers use corrected parameter grouping; legacy exact resumes retain
  their saved grouping.
- Optimizer, scaler, main scheduler, LTM scheduler, pending gradients, RNG, and
  dataloader state are restored independently. Resuming no longer rebuilds a
  warmup schedule unless explicitly requested.
- Checkpoint format v4 stores a hashed architecture contract. Current
  checkpoints reject contract, tensor-geometry, tokenizer, or exact-run drift.
- Chat-state v4 records that same contract plus exact recurrent layouts.
- Optional best-checkpoint selection accepts one immutable, finite lm-eval
  metric and saves an exact-resume `hierarchos_best.pt` only on improvement.

## Checkpoint compatibility

| Operation | Supported behavior |
| --- | --- |
| Resume a coherent-v9 training checkpoint | Exact resume; identity and all available optimizer/scheduler/data state are verified. |
| Chat with a coherent-v9 inference checkpoint | Supported full-precision path; architecture contract is verified before construction. |
| Resume or chat with v2/v3 | Loaded as `legacy-v8`; its historical learned function is preserved. |
| Turn v2/v3 into coherent-v9 by changing a flag | Unsupported. The adapter geometry and recurrent/objective semantics differ; train coherent-v9 from a fresh initialization. |
| Continue only the weights with a fresh optimizer | Use `--model-path`, understanding that this begins a new run rather than an exact resume. |
| Quantized `.npz` chat/export | Intentionally unsupported. The old scalar-RWKV implementation cannot reproduce the active matrix-state architecture. |
| Run the top-level `hierarchos.py` monolith | Unsupported. Use `hierarchos_cli.py`; the monolith is historical reference code. |

No software change can retroactively make old v2/v3 weights learn the corrected
coherent-v9 function. They remain usable controls, but a scientifically clean
v9 result requires new training.

## Data preflight before an expensive run

Build and audit the cache before allocating the model:

```bash
python hierarchos_cli.py train \
  --hf_dataset "netcat420/Experiment_0.1" \
  --hf_dataset_split train \
  --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" \
  --tokenizer-path "openai-community/gpt2" \
  --alpaca \
  --assistant-recovery \
  --max_length 8880 \
  --training-chunk-size 256 \
  --min-response-tokens 16 \
  --hf-token-cache \
  --hf-token-cache-dir "/content/hierarchos_token_cache/experiment_0_1_v9" \
  --token-cache-build-batch-size 256 \
  --token-cache-only
```

Inspect `cache_audit.json`. The default rejected-row budget is zero and the
default truncated-row budget is 5%, so schema mistakes and unexpected data loss
stop the run. If a larger threshold is intentional, set an explicit
`--max-cache-rejected-samples` or `--max-cache-rejected-fraction` only after
reviewing the reasons. In particular, `--min-response-tokens 128` rejects every
shorter answer; use it only if the audit proves that this is the desired corpus.

## Starting a fresh coherent-v9 run

`coherent-v9` is the new-run default, but spelling it out makes experiment
records self-documenting:

```bash
python hierarchos_cli.py train \
  --architecture-revision coherent-v9 \
  --hf_dataset "netcat420/Experiment_0.1" \
  --hf_dataset_split train \
  --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" \
  --out-dir "./chatHRM_coherent_v9" \
  --tokenizer-path "openai-community/gpt2" \
  --epochs 15 \
  --batch_size 64 \
  --accumulation-steps 1 \
  --accumulation-normalization weighted-token \
  --max_length 8880 \
  --training-chunk-size 256 \
  --no-full-sample-bptt \
  --detach-every-n-steps 0 \
  --no-persist-state \
  --context_dim 448 \
  --h_hidden 448 \
  --l_hidden 448 \
  --rwkv-head-size 64 \
  --max_h_steps 5 \
  --max_l_steps 5 \
  --alpaca \
  --assistant-recovery \
  --ltm-training-mode read-only \
  --train-prompt-tokens \
  --prompt-loss-weight 0.10 \
  --response-loss-weight 1.0 \
  --response-boundary-loss-weight 2.0 \
  --response-boundary-tokens 64 \
  --min-response-tokens 16 \
  --starting-lr 2e-5 \
  --min-lr 1e-7 \
  --warmup-ratio 0.01 \
  --ltm-lr 1e-8 \
  --min-ltm-lr 1e-8 \
  --disable-ltm-lr-schedule \
  --adaptive-ponder \
  --ponder-target-scale 0.65 \
  --ponder-loss-weight 0.003 \
  --commitment-loss-weight 0.5 \
  --max-commitment-cost-for-backward 4.0 \
  --max-ce-loss-for-backward 0 \
  --max-ponder-cost-for-backward 0 \
  --memory-gate-warmup-steps 5000 \
  --memory-gate-warmup-floor 0.10 \
  --grad-clip 0.75 \
  --device cuda \
  --amp \
  --force-compile \
  --compile-mode max-autotune-no-cudagraphs \
  --compile-static-worker-loop \
  --hf-token-cache \
  --hf-token-cache-dir "/content/hierarchos_token_cache/experiment_0_1_v9" \
  --token-cache-build-batch-size 256 \
  --length-bucket-auto-sample-size 1000000 \
  --cuda-prefetch \
  --save-steps 600
```

This retains the broad direction of the previous run while changing the
response-length floor from 128 to a safer 16. Keep 128 only if the cache audit
justifies it.

## Exact mid-epoch resume

Use the same immutable dataset and output directory and add the checkpoint:

```bash
python hierarchos_cli.py train \
  --hf_dataset "netcat420/Experiment_0.1" \
  --hf_dataset_split train \
  --hf-dataset-revision "4ef25be0ca46e7da7c70121b0b6d8e99cc232a51" \
  --out-dir "./chatHRM_coherent_v9" \
  --epochs 15 \
  --resume-from-ckpt "./chatHRM_coherent_v9/hierarchos_epoch_14_step_61800.pt"
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
  --model-path "C:\path\to\chatHRM_coherent_v9" `
  --alpaca `
  --device cpu `
  --threads 16 `
  --no-amp `
  --chat-prefill-chunk-size 256 `
  --chat-input-history-turns 0 `
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

## Release gate

Before committing a multi-billion-token budget:

```bash
python -m pytest -q
python tools/check_architecture_integrity.py --strict
```

Then run a small coherent-v9 overfit and exact-resume experiment with the real
tokenizer and cache. Verify:

1. loss falls on the tiny set;
2. an interrupted/resumed run matches the uninterrupted LR and next update;
3. chat and evaluation raw logits match the direct model path;
4. `cache_audit.json` has the intended schema and response-length distribution;
5. no skipped training batch is tolerated unless an explicit nonzero budget was
   reviewed in advance.

The remediation tree was validated with `389 passed, 4 skipped, 4 subtests
passed` and `73 passed, 0 warnings, 0 failures` from the strict executable
architecture audit. A separate CLI smoke run verified fresh checkpoint v4
training, safe `ROSAState` serialization, exact mid-epoch optimizer/scheduler
resume, matching subsequent losses, final inference export, architecture-hash
verification, and finite reloaded logits.
