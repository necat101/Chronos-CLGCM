---
language:
- en
license: other
library_name: pytorch
pipeline_tag: text-generation
tags:
- text-generation
- custom-architecture
- recurrent
- rwkv
- memory-augmented
- hierarchical-reasoning
- alpaca
- pytorch
- hierarchos
datasets:
- netcat420/Experiment_0.1
model-index:
- name: Hierarchos 232M
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: ai2_arc
      name: ARC Easy
    metrics:
    - type: accuracy
      value: 0.3600
      name: acc
    - type: accuracy
      value: 0.3200
      name: acc_norm
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: hellaswag
      name: HellaSwag
    metrics:
    - type: accuracy
      value: 0.3400
      name: acc
    - type: accuracy
      value: 0.3700
      name: acc_norm
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      type: truthfulqa
      name: TruthfulQA MC1
    metrics:
    - type: accuracy
      value: 0.2200
      name: acc
---

# Hierarchos 232M

Hierarchos 232M is the first coherent public checkpoint from the Hierarchos / KortexHOS research project. It is a small experimental assistant model using a custom recurrent memory-augmented architecture rather than a standard Transformer.

The model combines:

- RWKV-style recurrent sequence modeling
- a hierarchical manager/worker refinement loop
- differentiable slot-based long-term memory
- DeepEmbed token-conditioned channel-mix modulation
- ROSA, a suffix-automaton auxiliary pattern feature

This is an early research release. It shows usable short-form assistant behavior and measurable benchmark signal at 232M parameters, but it is not a GPT-3.5-class model and should not be treated as a replacement for larger general-purpose LLMs.

## Architecture and Inference Code

This checkpoint requires the Hierarchos architecture code for inference:

**GitHub repository:** [necat101/Hierarchos](https://github.com/necat101/Hierarchos)

The model is released in full precision. Quantized inference is not currently recommended for this checkpoint because our experiments found that Hierarchos' hierarchical drift/state dynamics were sensitive to accumulated quantization error.

## Model Details

| Field | Value |
| --- | --- |
| Model type | Custom recurrent memory-augmented language model |
| Parameters | Approximately 232M |
| Architecture | Hierarchos / KortexHOS |
| Tokenizer | GPT-2 tokenizer |
| Training format | Alpaca-style instruction/input/output |
| Precision | Full precision |
| Release status | Experimental research checkpoint |
| Inference repo | [necat101/Hierarchos](https://github.com/necat101/Hierarchos) |
| Training dataset | [netcat420/Experiment_0.1](https://huggingface.co/datasets/netcat420/Experiment_0.1) |

## Training Data

The model was trained on an in-house Alpaca-style dataset:

[netcat420/Experiment_0.1](https://huggingface.co/datasets/netcat420/Experiment_0.1)

The dataset uses instruction, optional input/context, and output/assistant-response fields. In Hierarchos chat mode, prompts are formatted internally using:

```text
### Instruction:
<user instruction>

### Input:
<optional previous context>

### Response:
```

## Training Run

This release was self-funded and trained for 13 epochs on an RTX 6000 Blackwell-generation 96GB GPU. The project went through several discarded or superseded runs while the architecture and training dynamics were refined.

Important stability and parity fixes made during development include:

- chat/train drift-state parity for streamed generation
- inference-like read-only LTM training mode
- drift norm and drift delta clamps
- RWKV channel-mix key clamp
- DeepEmbed channel-mix clamp
- DeepEmbed exclusion from AdamW weight decay
- static benchmark mode with passive memory writes suppressed

## Intended Use

This model is intended for:

- research on recurrent and memory-augmented language models
- local experimentation with the Hierarchos architecture
- short-form instruction following
- small-model assistant behavior testing
- architecture scaling and ablation studies

It is not intended for high-stakes use, factual authority, medical/legal/financial advice, or deployment where mistakes could cause harm.

## How to Run

Clone the architecture repository:

```bash
git clone https://github.com/necat101/Hierarchos
cd Hierarchos
```

Install the project dependencies according to the repository instructions, then run chat mode with the downloaded model directory:

```bash
python hierarchos_cli.py chat \
    --model-path "/path/to/this/model" \
    --temperature 0.4 \
    --top-k 40 \
    --top-p 0.9 \
    --repetition-penalty 1.15 \
    --max-new-tokens 256 \
    --no-passive-learning \
    --chat-input-history-turns 0
```

These are the recommended baseline inference settings for this release. They keep the model in a static evaluation-like mode, with passive LTM writes disabled and no previous-turn history injected into the Alpaca input field.

## Prompting

The model was trained as an instruction-following assistant. Good prompts are direct instructions or questions:

```text
Explain what machine learning is in simple terms.
```

```text
Write a short story about a kid finding a strange machine in the woods.
```

```text
List three reasons exercise can improve mood.
```

For best results, keep prompts concise and use the recommended static chat parameters above.

## Evaluation

We evaluated the release checkpoint with the local ROG Ally benchmark preset in the Hierarchos repository:

```bash
python hierarchos_cli.py benchmark \
    --model-path "/path/to/this/model" \
    --benchmark-preset rog-ally \
    --eval-limit 100
```

Results:

| Benchmark | Metric | Score | Std. Err. |
| --- | ---: | ---: | ---: |
| ARC Easy | acc | 0.3600 | 0.0482 |
| ARC Easy | acc_norm | 0.3200 | 0.0469 |
| HellaSwag | acc | 0.3400 | 0.0476 |
| HellaSwag | acc_norm | 0.3700 | 0.0485 |
| TruthfulQA MC1 | acc | 0.2200 | 0.0416 |

These are local smoke-test metrics, not leaderboard-comparable claims. They indicate that the model is not collapsed and has learned measurable commonsense and question-answering signal, especially for a small experimental checkpoint.

## Strengths

- Coherent short-form responses under the recommended inference settings
- Non-Transformer architecture with recurrent and memory-augmented components
- Measurable HellaSwag and ARC Easy signal at 232M parameters
- Small enough for local experimentation in full precision
- Useful as a research checkpoint for architecture development and ablation studies

## Limitations

- Not GPT-3 or GPT-3.5 class
- General language ability is still brittle
- Weak arithmetic and long-form consistency
- Limited broad world knowledge due to dataset scope and model scale
- No matched same-size Transformer baseline has been published yet
- Full precision is recommended; quantized inference is not currently the release path
- May hallucinate, repeat, or produce incorrect information
- Safety behavior has not been extensively validated

## Recommended Framing

The most accurate way to describe this checkpoint is:

> Hierarchos 232M is an experimental recurrent memory-augmented assistant model. It shows coherent short-form instruction behavior and measurable benchmark signal at small scale, but remains brittle and requires broader pretraining, baselines, and ablations before stronger capability claims can be made.

## Research Report

A preliminary technical writeup of the architecture findings, training/inference parity fixes, stability lessons, benchmark results, and scaling plan is available in the GitHub repository:

[necat101/Hierarchos](https://github.com/necat101/Hierarchos)

## Future Work

Planned next steps include:

- broader foundation pretraining on multiple licensed datasets
- matched 232M Transformer and RWKV-only baselines
- ablations for LTM, ROSA, DeepEmbed, and the hierarchical worker loop
- improved arithmetic and code-focused midtraining
- preference or distillation polish
- v8-compatible quantized inference once parity is verified

## Acknowledgements

This project was self-funded. A huge thanks to Lost Time for donating the lion's share of the funds needed for the training run and making this release possible.

Project contacts:

- Lost Time Discord: `losttime10`
- netcat Discord: `netcat7`

If you would like to support future scaling runs:

- Patreon: [Makhi Burroughs](https://patreon.com/MakhiBurroughs?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)
- Buy Me a Coffee: [netcat420](https://buymeacoffee.com/netcat420)

## Citation

If you use this model or architecture in research, please cite the GitHub repository and this model page:

```bibtex
@misc{hierarchos232m2026,
  title = {Hierarchos 232M: A Recurrent Memory-Augmented Assistant Model},
  author = {Burroughs, Makhi and Hierarchos contributors},
  year = {2026},
  howpublished = {\url{https://github.com/necat101/Hierarchos}}
}
```

## Disclaimer

This is an experimental research model. Outputs may be incorrect, unsafe, biased, repetitive, or hallucinated. Do not rely on this model for high-stakes decisions.
