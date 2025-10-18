-----

# Chronos v0.5.1 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 📢 **Major Update in v0.5: Enhanced Reasoning & Structured Memory**

> This version introduces two major architectural upgrades that work in tandem to create a more powerful and efficient cognitive process.
>
> 1.  **Adaptive Reasoning Depth ("Ponder Time"):** Chronos can now **"think longer" for harder problems**. The fixed-step reasoning loop has been replaced with a dynamic mechanism where the model learns to decide when it's confident enough to stop reasoning. This makes the model both more powerful and more efficient, better emulating the variable effort of human thought.
>
> 2.  **Structured & Queryable Long-Term Memory:** The LTM is no longer just an associative memory. Each memory slot is now **augmented with metadata, including a timestamp and a source identifier**. This allows the model—and the user—to perform sophisticated, context-aware queries, such as retrieving only memories learned from user interaction within the last hour. This brings the "Chronos" name to life.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Chronos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Chronos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Chronos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are now structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth, enabling it to solve complex, multi-step algorithmic problems where massive LLMs fail.

## Features

  - 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  - 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  - 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  - 🛡️ **Stable Training**: Built-in gradient clipping to prevent model instability and ensure smoother convergence.
  - 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config for easy sharing and use.
  - 💾 **Automatic Re-quantization**: After a learning session, Chronos can automatically re-quantize a model to persist the new knowledge.
  - ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`).
  - 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX) or on GPUs via Vulkan for broad hardware compatibility.
  - 🔧 **Comprehensive Tooling**: Includes a single, powerful script for training, LoRA fine-tuning, merging, quantization, and interactive chat, plus a script for model expansion.

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
    git clone https://github.com/your-username/Chronos.git
    cd Chronos
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

    This will create a `chronos_matmul` library file in your project root. If this fails, you can try running `pip install .` from the project root.

-----

## 📚 User Guide: Step-by-Step Workflows

The `chronos.py` script is the main entry point for most operations. All models are handled as directories.

### Basic Workflow

1.  **Training:** Train a new model from scratch. This creates a new, self-contained model directory at `--out-dir`. The `--max_h_steps` defines the upper limit for the model's reasoning depth.

    ```bash
    python chronos.py train \
        --train "path/to/your_data.jsonl" \
        --tokenizer-path "microsoft/phi-2" \
        --out-dir "./my_chronos_model" \
        --epochs 5 \
        --batch_size 2 \
        --context_dim 512 \
        --max_h_steps 10 \
        --ponder-loss-weight 0.01 \
        --grad-clip 1.0
    ```

2.  **Fine-Tuning (LoRA):** Adapt a pre-trained model using LoRA. This loads a full model directory and saves the adapter to a separate output directory.

    ```bash
    python chronos.py finetune \
        --model-path "./my_chronos_model" \
        --train "path/to/new_data.jsonl" \
        --out-dir "./my_lora_adapter" \
        --epochs 3
    ```

3.  **Merging a LoRA Adapter:** Merge the adapter back into the base model to create a new, standalone model directory.

    ```bash
    python chronos.py merge-lora \
        --model-path "./my_chronos_model" \
        --lora-adapter-path "./my_lora_adapter" \
        --out-dir "./my_model_merged"
    ```

4.  **Quantization:** Convert a full-precision model directory into a quantized model directory. **No architecture flags are needed\!**

    ```bash
    python chronos.py quantize \
        --model-path "./my_model_merged" \
        --out-dir "./my_model_merged-INT4" \
        --qtype INT4
    ```

    > **Available `qtype`:** `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.

5.  **Inference (Chat Mode):** Run an interactive chat session by pointing to any model directory.

    ```bash
    python chronos.py chat \
        --model-path "./my_model_merged-INT4"
    ```

### Resuming Training with Modified Learning Rate

If you need to resume training from a checkpoint but want to **change the learning rate schedule** (e.g., start with a different maximum LR or use a different minimum LR), use the `--override-scheduling` flag along with your new `--starting-lr` and `--min-lr` values.

```bash
python chronos.py train \
    --train "path/to/your_data.jsonl" \
    --out-dir "./my_chronos_model" \
    --epochs 10 \
    --resume-from-ckpt "./my_chronos_model/chronos_epoch_5.pt" \
    --starting-lr 1e-5 \
    --min-lr 1e-7 \
    --override-scheduling
```

> **Warning:** Without `--override-scheduling`, any new LR flags will be **ignored** when resuming, and the schedule from the checkpoint will be used.

### Expanding a Trained Model (Weight Transplantation) 🌱

You can efficiently train a larger model by **transplanting the weights** from a smaller, already-trained Chronos model. This technique, sometimes called "Net2Net" or model surgery, initializes the larger model with the knowledge learned by the smaller one, often leading to faster convergence during fine-tuning.

Use the `expand_model.py` script for this:

1.  **Train a smaller base model** (e.g., with `--context_dim 512`).

2.  **Use `expand_model.py` to create a larger model**, specifying the path to the trained smaller model (`--old-model-path`), the desired output path (`--output-path`), and the **new, larger architecture dimensions**. The script will copy matching weights and initialize new parameters randomly.

    ```bash
    python expand_model.py \
        --old-model-path "./small_model/chronos.pt" \
        --output-path "./large_model/chronos.pt" \
        --context_dim 1024 \
        --h_hidden 1024 \
        --l_hidden 1024
    ```

3.  **Fine-tune the expanded model** on your target dataset using `chronos.py finetune` or continue training using `chronos.py train --resume-from-ckpt ./large_model/chronos.pt`.

> **Note:** Remember to also copy the `tokenizer.json` and other tokenizer files from the old model directory to the new directory (`./large_model/` in the example) after expanding.

### Chat Mode Features

#### Querying Structured Memory

During a chat session, you can use the `/filter` command to constrain the model's memory retrieval for its next response.

  - **Filter by time:**

    ```
    >>> /filter time=-3600
    [INFO: Memory filtered to events after Sat Oct 18 01:44:00 2025]
    ```

    This command tells Chronos to only use memories it has learned in the last hour (3600 seconds).

  - **Filter by source:**

    ```
    >>> /filter source=1
    [INFO: Memory filtered to source ID: 1]
    ```

    This tells Chronos to only use memories learned from `user interaction` (Source ID 1). (Source ID 2 is `training data`).

  - **Reset filters:**

    ```
    >>> /filter reset
    [INFO: Memory filters have been reset.]
    ```

#### Enabling Online Learning in Chat

This requires both the quantized model (`--model-path`) and the original full-precision model (`--shadow-model-path`) to calculate the updates.

```bash
python chronos.py chat \
    --model-path "./my_model_merged-INT4" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged"
```

-----

## ⚙️ Command-Line Reference

### Main Modes

| Mode       | Description                                  |
| :--------- | :------------------------------------------- |
| `train`    | Train a new model from scratch.              |
| `finetune` | Apply LoRA fine-tuning to an existing model. |
| `merge-lora`| Merge a LoRA adapter into a base model.      |
| `quantize` | Convert a model directory to a quantized version. |
| `chat`     | Run an interactive chat session.             |

### Key Arguments

| Argument                  | Description                                                                     | Default           |
| :------------------------ | :------------------------------------------------------------------------------ | :---------------- |
| **Paths** |                                                                                 |                   |
| `--model-path`            | Path to the model directory (for `finetune`, `merge`, `quantize`, `chat`).        | `None`            |
| `--train`                 | Path to the training `.json` or `.jsonl` file.                                  | `None`            |
| `--out-dir`               | Directory to save new models, checkpoints, or adapters.                         | `./chronos_model` |
| `--tokenizer-path`        | `[Train]` Path or HF name of the tokenizer for a new model.                     | `microsoft/phi-2` |
| `--resume-from-ckpt`      | `[Train]` Path to a specific training checkpoint `.pt` file to resume from.     | `None`            |
| **Training & Fine-Tuning**|                                                                                 |                   |
| `--epochs`                | Number of training epochs.                                                      | `3`               |
| `--grad-clip`             | `[Train/Finetune]` Prevents gradient explosion for stable training. 0 to disable.| `1.0`             |
| `--ponder-loss-weight`    | `[Train/Finetune]` Weight for the Ponder Cost auxiliary loss.                     | `0.01`            |
| `--override-scheduling`   | `[Train]` If resuming, **ignore** checkpoint's schedule state and use new LR args.| `False`           |
| `--starting-lr`           | `[Train/Finetune]` Max Learning Rate for the schedule.                            | `1e-4`            |
| `--min-lr`                | `[Train/Finetune]` Minimum Learning Rate for cosine annealing.                    | `1e-6`            |
| **Quantization & Inference**|                                                                                 |                   |
| `--qtype`                 | Quantization format. Options: `INT4`, `Q4_0`, `Q8_0`, `Q2_K`.                     | `INT4`            |
| `--device`                | `[Chat]` Device for quantized inference. Options: `cpu`, `vulkan`.              | `cpu`             |
| `--h-halt-thresh`         | `[Chat]` Probability threshold for early exiting the reasoning loop.              | `0.9`             |
| `--enable-quantized-learning`| `[Chat]` Enable LTM updates for quantized models (requires `--shadow-model-path`).| `False`           |
| `--ltm_lr`                | `[Chat]` Max LR for LTM schedule, or the fixed rate if `--static-ltm-lr` is used.| `0.01`            |
| **Model Architecture** |                                                                                 |                   |
| `--max_h_steps`           | `[Train]` Maximum number of reasoning steps the H-module can take.              | `10`              |

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

## Changelog

### v0.5.1 (alpha)

  - **Added `--override-scheduling` flag**: Allows users to force new learning rate schedule settings (`--starting-lr`, `--min-lr`) when resuming training from a checkpoint.
  - **Documentation**: Added usage instructions for `--override-scheduling` and `expand_model.py`.

### v0.5 (alpha)

  - **Implemented Structured Long-Term Memory**: Memory slots are now augmented with timestamps and source metadata, enabling temporal and source-based filtering during chat.
  - **Implemented Adaptive Reasoning Depth (Ponder Time)**: The HRM's reasoning depth is now dynamic. The model learns to "think longer" for complex tokens and halt early on simple ones.
  - **Added Ponder Cost**: A new auxiliary loss (`--ponder-loss-weight`) trains the model to be computationally efficient.
  - **Added Halting Threshold**: A new inference flag (`--h-halt-thresh`) allows users to control the trade-off between speed and reasoning depth during chat.

### v0.4 (alpha)

  - **Implemented Dynamic LTM Learning Rate**: Online learning now defaults to a `CosineAnnealingLR` schedule for more stable knowledge consolidation.
  - **Added Static LR Fallback**: The `--static-ltm-lr` flag can be used in chat mode to revert to a fixed LTM learning rate.
  - **Added Gradient Clipping**: The `--grad-clip` argument was added to `train` and `finetune` modes to improve training stability.

-----

© 2025 Makhi Burroughs
