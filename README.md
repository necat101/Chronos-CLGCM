-----

# Chronos v0.3 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 📢 **Major Update in v0.3: Self-Contained & Portable Models\!**

> To improve reliability and ease of use, Chronos now saves and loads models as **self-contained directories**. Each directory includes the model weights (`.pt` or `.npz`), the necessary tokenizer files, and **the model's architecture configuration**.
>
> This is a critical change with two major benefits:
>
> 1.  It permanently fixes tokenizer mismatch errors.
> 2.  You **no longer need to specify architecture flags** (`--context_dim`, etc.) when quantizing or running inference\! The model knows its own configuration.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Chronos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Chronos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Chronos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," allowing it to consolidate new knowledge at inference time without catastrophic forgetting.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth, enabling it to solve complex, multi-step algorithmic problems where massive LLMs fail.

## Features

  - 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config.
  - 🧠 **Hybrid Memory-Reasoning**: Deeply integrates a lifelong memory system with a hierarchical reasoning engine.
  - 📚 **Continual "Online" Learning**: Learns and updates its Long-Term Memory (LTM) at inference time based on new experiences.
  - 💾 **Automatic Re-quantization**: After a learning session, Chronos can automatically re-quantize a model to persist the new knowledge.
  - ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`).
  - 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via Vulkan for broad hardware compatibility.
  - 🔧 **Comprehensive Tooling**: Includes a single, powerful script for training, LoRA fine-tuning, merging, quantization, and interactive chat.

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  - Python 3.8+
  - A C++ compiler (e.g., MSVC on Windows, GCC on Linux)
  - CMake (must be available in your system's `PATH`)
  - **For GPU Inference**: A Vulkan-compatible GPU and installed drivers.
  - **For Developers (Re-compiling the Kernel)**: The official [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) must be installed to compile the kernel with Vulkan support.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/necat101/Chronos.git
    cd chronos
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv

    # On Windows
    .\.venv\Scripts\Activate

    # On Linux/macOS
    source .venv/bin/activate
    ```

3.  **Run the setup script:** This will install Python dependencies and compile the C++ inference kernel.

    ```bash
    # On Windows
    setup.bat

    # On Linux/macOS
    bash setup.sh
    ```

    This will create a `chronos_matmul` library file in your project root.

## 📚 User Guide: A Step-by-Step Workflow

The `chronos.py` script is the main entry point for all operations. All models are now handled as directories.

### 1\. Training

Train a new model from scratch. This creates a new, self-contained model directory at `--out-dir`.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "microsoft/phi-2" \
    --out-dir "./my_chronos_model" \
    --epochs 5 \
    --batch_size 2 \
    --context_dim 512
```

> **Note:** You can use any compatible tokenizer from the Hugging Face Hub by changing `--tokenizer-path`.

#### Kayla Mode Training (Chain-of-Thought)

Enable Kayla Mode by adding the `--kayla` flag. The dataset should have `Instruction`, `thought-process`, `output`, and optionally `feelings` fields.

```bash
python chronos.py train \
    --train "path/to/kayla_data.jsonl" \
    --kayla \
    --out-dir "./my_kayla_model"
```

#### Resuming Training

If your training is interrupted, resume by pointing to the specific checkpoint file you want to load using `--resume-from-ckpt`.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --resume-from-ckpt "./my_chronos_model/chronos_epoch_3.pt" \
    --out-dir "./my_chronos_model"
```

### 2\. Fine-Tuning (LoRA)

Adapt a pre-trained model using LoRA. This loads a full model directory and saves the adapter to a separate output directory.

```bash
python chronos.py finetune \
    --model-path "./my_chronos_model" \
    --train "path/to/new_data.jsonl" \
    --out-dir "./my_lora_adapter" \
    --epochs 3
```

> **Tip**: Control the adapter size with `--finetune-unlock-percent 1.5` (for 1.5% trainable parameters).

### 3\. Merging a LoRA Adapter

Merge the adapter back into the base model to create a new, standalone model directory.

```bash
python chronos.py merge-lora \
    --model-path "./my_chronos_model" \
    --lora-adapter-path "./my_lora_adapter" \
    --out-dir "./my_model_merged"
```

### 4\. Quantization

Convert a full-precision model directory into a quantized model directory. **No architecture flags are needed\!** The script reads them from the model file.

```bash
python chronos.py quantize \
    --model-path "./my_model_merged" \
    --out-dir "./my_model_merged-INT4" \
    --qtype INT4
```

> The output directory will contain the quantized `.npz` file and a copy of the tokenizer.
> **Available `qtype`:** `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.

### 5\. Inference (Chat Mode)

Run an interactive chat session by pointing to any model directory (quantized or full-precision). **No architecture flags are needed\!**

#### Running a Quantized Model (Recommended)

**On CPU:**

```bash
python chronos.py chat --model-path "./my_model_merged-INT4"
```

**On GPU (Vulkan):**

```bash
python chronos.py chat --model-path "./my_model_merged-INT4" --device vulkan
```

#### Enabling Online Learning in Chat

Allow the model to learn from your conversation. This requires both the quantized model (`--model-path`) and the original full-precision model (`--shadow-model-path`) to calculate the updates.

**Method 1: Modify & Re-quantize (Recommended)**
This updates the model in memory and asks if you want to save the changes by re-quantizing upon exit. This makes your model permanently smarter.

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged"
```

**Method 2: Save Updates Separately (LoRA-style)**
To save only the LTM updates without modifying the base model, use `--ltm-lora-path`. The memory updates will be saved to the specified file when you quit.

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged" \
    --ltm-lora-path "./my_ltm_updates.pt"
```

-----

## ⚙️ Command-Line Reference

### Main Modes

| Mode       | Description                                         |
| :--------- | :-------------------------------------------------- |
| `train`    | Train a new model from scratch.                     |
| `finetune` | Apply LoRA fine-tuning to an existing model.        |
| `merge-lora`| Merge a LoRA adapter into a base model.             |
| `quantize` | Convert a model directory to a quantized version.   |
| `chat`     | Run an interactive chat session.                    |

### Key Arguments

| Argument                  | Description                                                                     | Default             |
| :------------------------ | :------------------------------------------------------------------------------ | :------------------ |
| **Paths** |                                                                                 |                     |
| `--model-path`            | Path to the model directory (for `finetune`, `merge`, `quantize`, `chat`).        | `None`              |
| `--train`                 | Path to the training `.json` or `.jsonl` file.                                  | `None`              |
| `--out-dir`               | Directory to save new models, checkpoints, or adapters.                         | `./chronos_model`   |
| `--tokenizer-path`        | `[Train]` Path or HF name of the tokenizer for a new model.                     | `microsoft/phi-2`   |
| `--resume-from-ckpt`      | `[Train]` Path to a specific `.pt` checkpoint file to resume training.          | `None`              |
| `--shadow-model-path`     | `[Chat]` Path to the original full-precision model for online learning.         | `None`              |
| **Training & Fine-Tuning** |                                                                                 |                     |
| `--epochs`                | Number of training epochs.                                                      | `3`                 |
| `--batch_size`            | Number of samples per batch.                                                    | `4`                 |
| `--accumulation-steps`    | Accumulate gradients to simulate a larger batch size.                           | `1`                 |
| `--starting-lr`           | The maximum learning rate for the scheduler.                                    | `1e-4`              |
| `--kayla`                 | `[Train]` Enable Chain-of-Thought style training.                               | `False`             |
| `--finetune-unlock-percent`| `[Finetune]` Target percentage of trainable parameters for LoRA.                 | `None`              |
| **Quantization & Inference** |                                                                                 |                     |
| `--qtype`                 | Quantization format. Options: `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.                   | `INT4`              |
| `--device`                | `[Chat]` Device for quantized inference. Options: `cpu`, `vulkan`.              | `cpu`               |
| `--enable-quantized-learning` | `[Chat]` Enable LTM updates for quantized models.                           | `False`             |
| **Model Architecture (`train` only)** |                                                                                 |                     |
| `--max_length`            | Maximum sequence length.                                                        | `1024`              |
| `--context_dim`           | The main embedding dimension of the model.                                      | `512`               |

-----

## Roadmap

  - [ ] Develop a user-friendly GUI wrapper for easier interaction.
  - [ ] Extend the architecture to support multi-modal inputs (images, audio).
  - [ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.

## License

The source code of Chronos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Chronos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  - This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  - The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  - **pybind11** for seamless C++/Python integration.

<br>

## v0.3 (alpha) Changelog

  - **Models are now self-contained**, storing their own architecture configuration. You no longer need to re-specify hyperparameters for inference or quantization.
  - Refactored to a **directory-based model system** to bundle weights, config, and tokenizer files together, fixing all tokenizer mismatch errors.
  - Simplified the command-line interface by unifying model loading under the `--model-path` argument.
  - Fixed the training resume process, which now correctly uses the `--resume-from-ckpt` flag to load a specific checkpoint file.
  - Deprecated old file-based flags (`--ckpt`, `--load-quantized`, `--resume-from-model-path`) in favor of the new, more robust system.

-----

© 2025 Makhi Burroughs
