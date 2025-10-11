-----

# Chronos v0.4 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 📢 **Major Update in v0.4: Dynamic Online Learning Rate & Stability**

> To make the model's lifelong learning more robust and human-like, Chronos now uses a **Cosine Annealing schedule for its Long-Term Memory (LTM) updates by default**.
>
> This means the model learns more aggressively from new information at the start of a session and then gradually reduces its learning rate to refine its knowledge, preventing instability. This dynamic, scheduled approach replaces the previous static learning rate.
>
> You can still use a fixed learning rate by adding the `--static-ltm-lr` flag during chat.

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

  - 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a new **Cosine Annealing LR schedule** by default for more stable knowledge consolidation.
  - 🛡️ **Stable Training**: Built-in gradient clipping to prevent model instability and ensure smoother convergence during training and fine-tuning.
  - 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config.
  - 💾 **Automatic Re-quantization**: After a learning session, Chronos can automatically re-quantize a model to persist the new knowledge.
  - ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`).
  - 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via Vulkan for broad hardware compatibility.
  - 🔧 **Comprehensive Tooling**: Includes a single, powerful script for training, LoRA fine-tuning, merging, quantization, and interactive chat.

-----

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

    If this does not compile for you, try manually installing the kernel by navigating to the chronos root directory and running `pip install .`

    for CUDA support, please install the cuda accelerated version of torch by running the following command: `pip install torch torchvision torau dio --index-url https://download.pytorch.org/whl/cu121`

-----

## 📚 User Guide: A Step-by-Step Workflow

The `chronos.py` script is the main entry point for all operations. All models are now handled as directories.

### 1\. Training

Train a new model from scratch. This creates a new, self-contained model directory at `--out-dir`. Using `--grad-clip 1.0` is recommended for stability.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "microsoft/phi-2" \
    --out-dir "./my_chronos_model" \
    --epochs 5 \
    --batch_size 2 \
    --context_dim 512 \
    --grad-clip 1.0
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

### 3\. Merging a LoRA Adapter

Merge the adapter back into the base model to create a new, standalone model directory.

```bash
python chronos.py merge-lora \
    --model-path "./my_chronos_model" \
    --lora-adapter-path "./my_lora_adapter" \
    --out-dir "./my_model_merged"
```

### 4\. Quantization

Convert a full-precision model directory into a quantized model directory. **No architecture flags are needed\!**

```bash
python chronos.py quantize \
    --model-path "./my_model_merged" \
    --out-dir "./my_model_merged-INT4" \
    --qtype INT4
```

> **Available `qtype`:** `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.

### 5\. Inference (Chat Mode)

Run an interactive chat session by pointing to any model directory (quantized or full-precision).

#### Enabling Online Learning in Chat

Allow the model to learn from your conversation. This requires both the quantized model (`--model-path`) and the original full-precision model (`--shadow-model-path`) to calculate the updates.

**Method 1: Dynamic LR & Re-quantize (Recommended)**
This is the default behavior. It uses a dynamic learning rate and asks to save changes by re-quantizing upon exit.

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged"
```

**Method 2: Static LR & Re-quantize**
To use a fixed learning rate instead of the dynamic schedule, add the `--static-ltm-lr` flag.

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged" \
    --static-ltm-lr \
    --ltm_lr 0.005
```

### Advanced Model Surgery: Using `expand_model.py`

The `expand_model.py` script is a powerful utility for increasing a model's capacity without retraining from scratch. It allows you to "transplant" the learned weights from a smaller, trained model into a new, larger architecture. The new, larger parts of the model are randomly initialized, while the existing parts retain their knowledge, providing an excellent starting point for further fine-tuning.

**Step 1: Expand the Model Checkpoint**

Run the script, pointing to an existing training checkpoint (`.pt` file) and defining the new, larger dimensions.

```bash
# Example: Expanding a model with context_dim=128 to context_dim=512
python expand_model.py \
    --old-model-path "./my_chronos_model/chronos_epoch_5.pt" \
    --output-path "./my_expanded_model/expanded_checkpoint.pt" \
    --context_dim 512 \
    --h_hidden 512 \
    --l_hidden 512
```

**Step 2: Fine-Tune the Expanded Model**

This command creates a new, full **training checkpoint** at the specified output path. You can then use this checkpoint with `--resume-from-ckpt` to fine-tune your new, larger model on your dataset, allowing it to learn how to use its expanded capacity.

```bash
# Now, resume training to fine-tune the expanded model
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --out-dir "./my_expanded_model_finetuned" \
    --resume-from-ckpt "./my_expanded_model/expanded_checkpoint.pt" \
    --epochs 10
```

-----

## ⚙️ Command-Line Reference

### Main Modes

| Mode | Description |
| :--- | :--- |
| `train` | Train a new model from scratch. |
| `finetune` | Apply LoRA fine-tuning to an existing model. |
| `merge-lora` | Merge a LoRA adapter into a base model. |
| `quantize` | Convert a model directory to a quantized version. |
| `chat` | Run an interactive chat session. |

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| **Paths** | | |
| `--model-path` | Path to the model directory (for `finetune`, `merge`, `quantize`, `chat`). | `None` |
| `--train` | Path to the training `.json` or `.jsonl` file. | `None` |
| `--out-dir` | Directory to save new models, checkpoints, or adapters. | `./chronos_model` |
| `--tokenizer-path` | `[Train]` Path or HF name of the tokenizer for a new model. | `microsoft/phi-2` |
| `--shadow-model-path` | `[Chat]` Path to the original full-precision model for online learning. | `None` |
| **Training & Fine-Tuning** | | |
| `--epochs` | Number of training epochs. | `3` |
| `--starting-lr` | The maximum learning rate for the main model scheduler. | `1e-4` |
| `--grad-clip` | `[Train/Finetune]` Prevents gradient explosion for stable training. 0 to disable. | `1.0` |
| `--kayla` | `[Train]` Enable Chain-of-Thought style training. | `False` |
| **Quantization & Inference**| | |
| `--qtype` | Quantization format. Options: `INT4`, `Q4_0`, `Q8_0`, `Q2_K`. | `INT4` |
| `--device` | `[Chat]` Device for quantized inference. Options: `cpu`, `vulkan`. | `cpu` |
| `--enable-quantized-learning`| `[Chat]` Enable LTM updates for quantized models. | `False` |
| `--ltm_lr` | `[Chat]` Max LR for LTM schedule, or the fixed rate if `--static-ltm-lr` is used.| `0.01` |
| `--static-ltm-lr` | `[Chat]` Disable the LTM cosine schedule and use a fixed learning rate. | `False` |

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

## v0.4 (alpha) Changelog

  - **Implemented Dynamic LTM Learning Rate**: Online learning now defaults to a `CosineAnnealingLR` schedule. This improves learning stability by starting with a high learning rate for new information and gradually decaying it.
  - **Added Static LR Fallback**: The `--static-ltm-lr` flag can be used in chat mode to revert to the old fixed learning rate behavior for the LTM.
  - **Added Gradient Clipping**: The `--grad-clip` argument is now available for `train` and `finetune` modes to prevent gradient explosion and improve training stability.

-----

© 2025 Makhi Burroughs
