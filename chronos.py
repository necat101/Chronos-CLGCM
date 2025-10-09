import os
import sys
import json
import argparse
import time
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Optional Imports ---
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    _HAS_PEFT = True
except ImportError:
    print("Warning: 'peft' library not found. LoRA fine-tuning and merging will be unavailable.")
    _HAS_PEFT = False

try:
    import keyboard
    _HAS_KEYBOARD = True
except ImportError:
    print("Warning: 'keyboard' library not found. The Ctrl+X interrupt feature will be disabled in chat mode.")
    _HAS_KEYBOARD = False

# --- MODIFIED FOR CPU TRAINING ---
print("Warning: bitsandbytes is disabled for CPU training. Falling back to standard AdamW optimizer.")
ADAM_OPTIMIZER = torch.optim.AdamW
_HAS_BNB = False
# --- END MODIFICATION ---

# --- C++ Kernel Import ---
try:
    import chronos_matmul
    _HAS_KERNEL = True
    print("Successfully imported C++ quantization kernel.")
    if hasattr(chronos_matmul, "VULKAN_SUPPORT") and chronos_matmul.VULKAN_SUPPORT:
        print("INFO: Vulkan support is enabled in the compiled kernel.")
        _HAS_VULKAN = True
    else:
        print("INFO: Vulkan support is disabled in the compiled kernel.")
        _HAS_VULKAN = False
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! ERROR: The compiled C++ kernel 'chronos_matmul' was not found.           !!!")
    print("!!!                                                                         !!!")
    print("!!! To fix this, please run the appropriate setup script:                   !!!")
    print("!!!  - On Windows:   Run setup.bat                                          !!!")
    print("!!!  - On Linux/macOS: Run bash setup.sh                                    !!!")
    print("!!!                                                                         !!!")
    print("!!! If you have already run the setup, you may need to activate the         !!!")
    print("!!! virtual environment first:                                              !!!")
    print("!!!  - On Windows:   .\\.venv\\Scripts\\Activate                              !!!")
    print("!!!  - On Linux/macOS: source .venv/bin/activate                            !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    sys.exit(1) # Exit the program because the kernel is essential


# --- CONSTANTS ---
MODEL_WEIGHTS_NAME = "chronos.pt"
QUANTIZED_MODEL_WEIGHTS_NAME_TPL = "chronos-{qtype}.npz" # Template for the name


# --- HELPER CLASS: AttrDict ---
class AttrDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# --- Device Utilities ---
def pick_device():
    """Picks the best available device for PyTorch training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_threads(n: int):
    try:
        torch.set_num_threads(n)
        os.environ['OMP_NUM_THREADS'] = str(n)
    except Exception as e:
        print(f"Warning: Could not set thread count. {e}")


# --- Dataset & Dataloader ---
class JSONLDataset(Dataset):
    """
    Handles both .jsonl (one JSON object per line) and .json (a list of objects) files.
    Also supports standard and "Kayla" instruction formats.
    """
    def __init__(self, path: str, tokenizer, max_length: int, kayla_mode: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.kayla_mode = kayla_mode
        self._load(path)

    def _process_object(self, obj):
        """Processes a single JSON object into tokenized input_ids and labels."""
        try:
            if self.kayla_mode:
                # --- Context part (masked out in labels) ---
                instruction_text = f"### Instruction:\n{obj.get('Instruction', '')}\n\n"
                
                feelings_text = ""
                # Check for the feelings key and ensure it has content to include it
                if obj.get('feelings'):
                    feelings_text = f"### Feelings:\n{obj.get('feelings')}\n\n"

                # The full prompt context is instruction + feelings
                prompt_context_text = instruction_text + feelings_text

                # --- Generation part (predicted by the model) ---
                thought_text = f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                output_text = f"### Response:\n{obj.get('output', '')}"
                
                # Tokenize the different parts
                prompt_context_tokens = self.tokenizer.encode(prompt_context_text, add_special_tokens=True)
                thought_tokens = self.tokenizer.encode(thought_text, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)

                # Combine into final input_ids and labels
                input_ids = prompt_context_tokens + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_context_tokens)) + thought_tokens + output_tokens + [self.tokenizer.eos_token_id]
            else:
                prompt = f"### Instruction:\n{obj.get('instruction', '')}\n\n### Response:\n"
                completion = obj.get('output', '') or obj.get('response', '')

                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)

                input_ids = prompt_tokens + completion_tokens + [self.tokenizer.eos_token_id]
                labels = ([-100] * len(prompt_tokens)) + completion_tokens + [self.tokenizer.eos_token_id]

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
        except (KeyError, AttributeError, TypeError):
            return None

    def _load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        print(f"Loading and tokenizing dataset from {path}...")
        if self.kayla_mode:
            print("INFO: Kayla-style instruction tuning is ENABLED.")

        with open(path, "r", encoding="utf-8") as f:
            try:
                # Handle .json file (a list of objects)
                data = json.load(f)
                if isinstance(data, list):
                    print("Detected JSON file (list of objects). Processing...")
                    for obj in tqdm(data, desc="Tokenizing samples"):
                        processed = self._process_object(obj)
                        if processed: self.samples.append(processed)
                    return
            except json.JSONDecodeError:
                # Handle .jsonl file (one object per line)
                print("Detected JSONL file (one object per line). Processing...")
                f.seek(0)
                for line in tqdm(f, desc="Tokenizing samples"):
                    try:
                        obj = json.loads(line)
                        processed = self._process_object(obj)
                        if processed: self.samples.append(processed)
                    except (json.JSONDecodeError):
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_dataloader(path, tokenizer, max_length, batch_size, pad_token_id, kayla_mode=False):
    """Creates a DataLoader for training or fine-tuning."""
    dataset = JSONLDataset(path, tokenizer, max_length, kayla_mode=kayla_mode)

    def collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)

        input_ids_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask_batch = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids_batch[i, :seq_len] = item['input_ids']
            labels_batch[i, :seq_len] = item['labels']
            attention_mask_batch[i, :seq_len] = 1

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


# --- Quantization & Model Serialization ---
def get_q_block_size(qtype: str) -> int:
    """Returns the block size for a given quantization type."""
    if qtype in ["INT4", "Q4_0", "Q8_0"]:
        return 32
    elif qtype == "Q2_K":
        return 256
    else:
        raise ValueError(f"Unknown quantization type: {qtype}")

def export_and_quantize_model(output_dir: str, model: nn.Module, tokenizer, qtype: str):
    """Quantizes and exports the model to a directory containing the .npz and tokenizer."""
    if not _HAS_KERNEL:
        print("ERROR: C++ kernel is required for quantization. Aborting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, QUANTIZED_MODEL_WEIGHTS_NAME_TPL.format(qtype=qtype))

    print(f"Quantizing and exporting model to {output_path} with {qtype} weights...")
    state_dict = model.state_dict()
    quantized_tensors = {}

    config_to_save = dict(model.config)
    quantized_tensors['_config'] = np.array(config_to_save)

    Q_BLOCK_SIZE = get_q_block_size(qtype)

    for name, tensor in tqdm(state_dict.items(), desc="Quantizing Tensors"):
        if tensor.ndim == 2 and "emb" not in name and "ltm" not in name:
            float_tensor_np = tensor.cpu().float().numpy()
            M, K = float_tensor_np.shape

            pad_cols = 0
            if K % Q_BLOCK_SIZE != 0:
                pad_cols = Q_BLOCK_SIZE - (K % Q_BLOCK_SIZE)
                float_tensor_np = np.pad(float_tensor_np, ((0, 0), (0, pad_cols)), 'constant')

            quantized_data = chronos_matmul.quantize(float_tensor_np, qtype)
            quantized_tensors[name] = {
                "quantized": quantized_data,
                "qtype": qtype,
                "original_shape": [M, K],
            }
        else:
            quantized_tensors[name] = {"raw": tensor.cpu().numpy()}

    np.savez_compressed(output_path, **quantized_tensors)
    print(f"Model weights successfully exported to {output_path}")

    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer files saved to {output_dir}")

class QuantizedLinear:
    """A wrapper for a quantized linear layer that uses the C++ kernel for inference."""
    def __init__(self, name: str, q_data: dict):
        self.name = name
        weight_data = q_data[f'{name}.weight'].item()
        self.quantized_w = weight_data['quantized']
        self.qtype = str(weight_data['qtype'])
        self.original_shape = weight_data['original_shape']
        self.M, self.K = self.original_shape

        bias_obj = q_data.get(f'{name}.bias')
        if bias_obj is not None:
            self.bias = bias_obj.item()['raw']
        else:
            self.bias = None

    def __call__(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        if not _HAS_KERNEL: raise ImportError("C++ kernel required for quantized matmul")

        x_np = x.cpu().float().numpy()
        y_np = chronos_matmul.matmul_quantized(x_np, self.quantized_w, self.M, self.qtype, device)

        if y_np.shape[-1] != self.M:
            y_np = y_np[..., :self.M]

        if self.bias is not None: y_np += self.bias
        return torch.from_numpy(y_np)

class QuantizedGRUCell:
    """A quantized GRU cell implementation using QuantizedLinear layers."""
    def __init__(self, input_size, hidden_size, name_prefix, q_data):
        self.input_size, self.hidden_size = input_size, hidden_size
        self.W_ir = QuantizedLinear(f'{name_prefix}.W_ir', q_data)
        self.W_hr = QuantizedLinear(f'{name_prefix}.W_hr', q_data)
        self.W_iz = QuantizedLinear(f'{name_prefix}.W_iz', q_data)
        self.W_hz = QuantizedLinear(f'{name_prefix}.W_hz', q_data)
        self.W_in = QuantizedLinear(f'{name_prefix}.W_in', q_data)
        self.W_hn = QuantizedLinear(f'{name_prefix}.W_hn', q_data)

    def __call__(self, x: torch.Tensor, h: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x, device) + self.W_hr(h, device))
        z = torch.sigmoid(self.W_iz(x, device) + self.W_hz(h, device))
        n = torch.tanh(self.W_in(x, device) + r * self.W_hn(h, device))
        return (1 - z) * n + z * h

class GRUCell(nn.Module):
    """A custom GRU cell implementation using individual Linear layers."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        return (1 - z) * n + z * h

class LTMModule(nn.Module):
    """Titans Long-Term Memory Module. Capable of test-time updates."""
    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(n_slots, val_dim) * 0.02)
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd
        self.register_buffer("ltm_deltas", torch.zeros_like(self.vals.data))
        self.accumulate_deltas = False

    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Retrieves the top-k most similar values from memory."""
        sim = queries @ self.keys.t()
        _, idx = torch.topk(sim, k=topk, dim=-1)
        return self.vals[idx], idx

    def inner_update(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor, current_lr: float):
        """
        Performs a meta-learning update on the LTM value slots based on the "surprise" gradient.
        Now accepts a dynamic learning rate.
        """
        with torch.no_grad():
            if grads_tensor is None: return
            device = self.vals.device

            slot_grads = torch.zeros_like(self.vals.data)
            idx_flat = topk_idx.view(-1)
            grads_flat = grads_tensor.view(-1, self.vals.size(1))
            slot_grads.index_add_(0, idx_flat.to(device), grads_flat.to(device))

            counts = torch.zeros(self.vals.size(0), device=device)
            counts.index_add_(0, idx_flat.to(device), torch.ones_like(idx_flat, dtype=torch.float, device=device))
            nonzero_mask = counts > 0
            if nonzero_mask.any():
                slot_grads[nonzero_mask] /= counts[nonzero_mask].unsqueeze(-1)

            self._mom_vals.data.mul_(self.momentum).add_(slot_grads)
            update_delta = (self._mom_vals.data + self.weight_decay * self.vals.data)
            update_delta.mul_(-current_lr)

            final_update = torch.zeros_like(self.vals.data)
            final_update[nonzero_mask] = update_delta[nonzero_mask]

            if self.accumulate_deltas:
                self.ltm_deltas.data.add_(final_update)
                self.vals.data.add_(final_update)
            else:
                self.vals.data.add_(final_update)

class ChronosCore(nn.Module):
    """The full, trainable Chronos model, integrating HRM as the core processor."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = AttrDict(config)

        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.context_dim)
        self.pos_emb = nn.Embedding(self.config.max_length, self.config.context_dim)

        self.persistent = nn.Parameter(torch.randn(self.config.persistent_dim) * 0.02)
        self.ltm = LTMModule(
            n_slots=self.config.ltm_slots,
            key_dim=self.config.ltm_key_dim,
            val_dim=self.config.ltm_val_dim,
            lr=self.config.ltm_lr
        )
        self.qproj = nn.Linear(self.config.context_dim, self.config.ltm_key_dim, bias=False)

        in_dim = self.config.context_dim + self.config.persistent_dim + (self.config.ltm_val_dim * self.config.ltm_topk)
        self.in_proj = nn.Linear(in_dim, self.config.context_dim)

        self.h_rnn = GRUCell(self.config.context_dim, self.config.h_hidden)
        self.h_to_context = nn.Linear(self.config.h_hidden, self.config.context_dim)
        self.l_rnn = GRUCell(self.config.context_dim * 2, self.config.l_hidden)
        self.l_to_out = nn.Linear(self.config.l_hidden, self.config.context_dim)

        self.out_norm = nn.LayerNorm(self.config.context_dim)
        self.lm_head = nn.Linear(self.config.context_dim, self.config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _get_prompt_embedding(self, prompt_embedding):
        return prompt_embedding

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None, labels: Optional[torch.LongTensor] = None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device

        tok_embs = self.tok_emb(input_ids)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = tok_embs + self.pos_emb(pos)

        if attention_mask is None: attention_mask = torch.ones_like(input_ids)

        all_topk_vals = []
        all_topk_idx = []
        final_token_embeddings = []

        h_state = torch.zeros(B, self.config.h_hidden, device=device)
        l_state = torch.zeros(B, self.config.l_hidden, device=device)

        for t in range(T):
            token_emb = x[:, t, :]

            p_read = self.persistent.unsqueeze(0).expand(B, -1)
            query = self.qproj(token_emb)
            topk_vals, topk_idx = self.ltm.retrieve_topk(query, topk=self.config.ltm_topk)

            if self.training or torch.is_grad_enabled():
                topk_vals.retain_grad()

            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)

            ltm_summary_flat = topk_vals.view(B, -1)

            mac_input = torch.cat([token_emb, p_read, ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input))
            
            for _ in range(self.config.h_steps):
                h_state = self.h_rnn(enc, h_state)
                context = self.h_to_context(h_state)
                l_input = torch.cat([enc, context], dim=-1)
                
                l_state_prev = torch.zeros_like(l_state)
                for _ in range(self.config.max_l_steps):
                    l_state_prev = l_state.clone()
                    l_state = self.l_rnn(l_input, l_state)
                    if torch.allclose(l_state, l_state_prev, atol=self.config.l_conv_atol):
                        break
                
                enc = enc + self.l_to_out(l_state)

            final_token_embeddings.append(enc)

        final_embeddings = torch.stack(final_token_embeddings, dim=1)
        final_embeddings = self.out_norm(final_embeddings)
        logits = self.lm_head(final_embeddings)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        seq_topk_vals = torch.stack(all_topk_vals, dim=1)
        seq_topk_idx = torch.stack(all_topk_idx, dim=1)

        return {"loss": loss, "logits": logits, "topk_vals": seq_topk_vals, "topk_idx": seq_topk_idx}


class QuantizedChronos:
    """The quantized Chronos model for CPU/Vulkan inference."""
    def __init__(self, config: dict, q_data: dict):
        self.config = AttrDict(config)
        first_quantized_layer_meta = q_data['qproj.weight'].item()
        self.qtype = first_quantized_layer_meta['qtype']
        
        self.tok_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['tok_emb.weight'].item()['raw']))
        self.pos_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['pos_emb.weight'].item()['raw']))
        self.persistent = torch.from_numpy(q_data['persistent'].item()['raw'])
        self.out_norm = nn.LayerNorm(self.config.context_dim)
        self.out_norm.load_state_dict({
            'weight': torch.from_numpy(q_data['out_norm.weight'].item()['raw']),
            'bias': torch.from_numpy(q_data['out_norm.bias'].item()['raw'])
        })

        self.ltm = LTMModule(n_slots=self.config.ltm_slots, key_dim=self.config.ltm_key_dim, val_dim=self.config.ltm_val_dim)
        self.ltm.load_state_dict({
            'keys': torch.from_numpy(q_data['ltm.keys'].item()['raw']),
            'vals': torch.from_numpy(q_data['ltm.vals'].item()['raw'])
        }, strict=False)

        self.qproj = QuantizedLinear('qproj', q_data)
        self.in_proj = QuantizedLinear('in_proj', q_data)
        self.h_rnn = QuantizedGRUCell(self.config.context_dim, self.config.h_hidden, 'h_rnn', q_data)
        self.h_to_context = QuantizedLinear('h_to_context', q_data)
        self.l_rnn = QuantizedGRUCell(self.config.context_dim * 2, self.config.l_hidden, 'l_rnn', q_data)
        self.l_to_out = QuantizedLinear('l_to_out', q_data)
        self.lm_head = QuantizedLinear('lm_head', q_data)
        print("Initialized QuantizedChronos model from config.")

    def __call__(self, input_ids: torch.LongTensor, h_state: torch.Tensor, l_state: torch.Tensor, device: str = "cpu"):
        B, T = input_ids.shape
        current_pos_start = T - 1 if T > 1 else 0
        
        for t in range(current_pos_start, T):
            pos_ids = torch.tensor([t], dtype=torch.long)
            token_emb = self.tok_emb(input_ids[:, t]) + self.pos_emb(pos_ids)
            
            p_read = self.persistent.unsqueeze(0).expand(B, -1)
            query = self.qproj(token_emb, device=device)
            topk_vals, _ = self.ltm.retrieve_topk(query, topk=self.config.ltm_topk)
            ltm_summary_flat = topk_vals.view(B, -1)

            mac_input = torch.cat([token_emb, p_read, ltm_summary_flat], dim=-1)
            enc = F.gelu(self.in_proj(mac_input, device=device))

            for _ in range(self.config.h_steps):
                h_state = self.h_rnn(enc, h_state, device=device)
                context = self.h_to_context(h_state, device=device)
                for _ in range(self.config.max_l_steps):
                    l_input = torch.cat([enc, context], dim=-1)
                    l_state = self.l_rnn(l_input, l_state, device=device)
                enc = enc + self.l_to_out(l_state, device=device)
        
        final_embedding = self.out_norm(enc)
        logits = self.lm_head(final_embedding, device=device)
        return {"logits": logits.unsqueeze(1), "h_state": h_state, "l_state": l_state}

def load_quantized(model_path: str):
    """Loads a quantized model directory, automatically finding the .npz and tokenizer."""
    print(f"Loading quantized model from directory: {model_path}")
    
    npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"No quantized model .npz file found in {model_path}")
    if len(npz_files) > 1:
        print(f"Warning: Multiple .npz files found. Using the first one: {npz_files[0]}")
    
    weights_path = os.path.join(model_path, npz_files[0])
    q_data = np.load(weights_path, allow_pickle=True)

    if '_config' not in q_data:
        raise ValueError("Quantized model file is missing '_config' data. Please re-quantize the model.")

    config = q_data['_config'].item()
    return QuantizedChronos(config, q_data), AttrDict(config)

def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model and its config from a directory."""
    weights_path = os.path.join(model_path, MODEL_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights file '{MODEL_WEIGHTS_NAME}' not found in '{model_path}'")
        
    checkpoint = torch.load(weights_path, map_location=device)

    if 'config' not in checkpoint:
        raise ValueError("Model config not found in checkpoint. The model file is likely corrupted or from an old version.")
    config = AttrDict(checkpoint['config'])
    
    if 'model_type' not in config:
        config['model_type'] = 'chronos'

    model = ChronosCore(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, config


def train(args, device, tokenizer):
    print("Running in TRAIN mode...")
    config = vars(args)
    config['model_type'] = 'chronos' 
    
    model = ChronosCore(config).to(device)
    optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
    dataloader = create_dataloader(args.train, tokenizer, args.max_length, args.batch_size, tokenizer.pad_token_id, kayla_mode=args.kayla)

    scheduler = None
    if not args.disable_lr_schedule:
        num_update_steps = (len(dataloader) // args.accumulation_steps) * args.epochs
        print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
        scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

    start_epoch = 0
    if args.resume_from_ckpt:
        if not os.path.exists(args.resume_from_ckpt):
            raise FileNotFoundError(f"Checkpoint to resume from not found at {args.resume_from_ckpt}")

        print(f"Resuming training from checkpoint: {args.resume_from_ckpt}")
        checkpoint = torch.load(args.resume_from_ckpt, map_location=device)
        
        if 'optimizer_state_dict' not in checkpoint:
            raise ValueError("The specified checkpoint is a final inference model, not a training checkpoint. Cannot resume.")

        if 'config' in checkpoint:
            model_config = AttrDict(checkpoint['config'])
            model = ChronosCore(model_config).to(device)
        else:
            print("Warning: Config not found in checkpoint. Using current CLI args for model architecture.")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('completed_epoch', 0)

        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            print("Resuming learning rate scheduler state.")
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Successfully loaded model and optimizer. Resuming from epoch {start_epoch + 1}.")

    model.train()
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    os.makedirs(args.out_dir, exist_ok=True)
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        total_loss = 0.0

        for i, batch in enumerate(pbar):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            if loss is not None:
                loss = loss / args.accumulation_steps
                loss.backward()
                total_loss += loss.item() * args.accumulation_steps

                if (i + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
                pbar.set_postfix({"loss": f"{total_loss / (i + 1):.4f}", "lr": f"{current_lr:.2e}"})

        # <<< FIX IS HERE: THIS BLOCK IS NOW INDENTED TO BE INSIDE THE LOOP >>>
        ckpt_path = os.path.join(args.out_dir, f"chronos_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} complete. Saving training checkpoint to {ckpt_path}")
        torch.save({
            'completed_epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': dict(model.config),
        }, ckpt_path)

    final_save_path = os.path.join(args.out_dir, MODEL_WEIGHTS_NAME)
    print(f"\nTraining finished. Saving final inference model to {final_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(model.config)
    }, final_save_path)
    
    tokenizer.save_pretrained(args.out_dir)
    print(f"Tokenizer files saved to {args.out_dir}")

    if args.quantize_on_complete:
        print("\n--- Training Complete: Starting On-the-Fly Quantization ---")
        quantize_out_dir = args.out_dir.rstrip('/\\') + f"-{args.qtype}"
        quantize(args, device, model, tokenizer, quantize_out_dir)


def finetune(args, device, tokenizer):
    if not _HAS_PEFT: raise ImportError("Please install 'peft' for fine-tuning.")
    print("Running in FINETUNE mode with LoRA...")

    model, model_config = load_full_model_with_config(args.model_path, device)

    lora_r = args.lora_r
    if args.finetune_unlock_percent:
        if args.lora_r != 8:
            print(f"Warning: Both --lora_r ({args.lora_r}) and --finetune-unlock-percent were specified. Prioritizing --lora_r.")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            target_modules = ["qproj", "in_proj", "h_to_context", "l_to_out", "lm_head"]
            lora_param_sum_per_r = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(tm in name for tm in target_modules):
                    lora_param_sum_per_r += module.in_features + module.out_features

            target_trainable_count = total_params * (args.finetune_unlock_percent / 100.0)
            if lora_param_sum_per_r > 0:
                estimated_r = target_trainable_count / lora_param_sum_per_r
                lora_r = max(1, int(round(estimated_r)))
                print(f"Targeting ~{args.finetune_unlock_percent}% trainable parameters. Estimated LoRA rank 'r' = {lora_r}")
            else:
                print("Warning: Could not find target modules for LoRA. Using default r=8.")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qproj", "in_proj", "h_to_context", "l_to_out"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataloader = create_dataloader(args.train, tokenizer, model_config.max_length, args.batch_size, tokenizer.pad_token_id, kayla_mode=args.kayla)
    optimizer = ADAM_OPTIMIZER(model.parameters(), lr=args.starting_lr)
    os.makedirs(args.out_dir, exist_ok=True)

    scheduler = None
    if not args.disable_lr_schedule:
        num_update_steps = (len(dataloader) // args.accumulation_steps) * args.epochs
        print(f"INFO: Step-based Cosine Annealing LR scheduler ENABLED for finetuning. Total update steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
        scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)

    optimizer.zero_grad()
    for epoch in range(args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}")
        total_loss = 0.0
        for i, batch in enumerate(pbar):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            if loss is not None:
                loss = loss / args.accumulation_steps
                loss.backward()
                total_loss += loss.item() * args.accumulation_steps

                if (i + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()

                current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
                pbar.set_postfix({"loss": f"{total_loss / (i+1):.4f}", "lr": f"{current_lr:.2e}"})
    
    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)


def merge_lora(args, device, tokenizer):
    if not _HAS_PEFT: raise ImportError("Please install 'peft' for merging.")
    print("Running in MERGE-LORA mode...")
    
    print(f"Loading base model from {args.model_path}...")
    base_model, _ = load_full_model_with_config(args.model_path, device)

    print(f"Loading LoRA adapter from {args.lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)

    print("Merging adapter into the base model...")
    model = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, MODEL_WEIGHTS_NAME)
    print(f"Saving merged model to {output_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(model.config)
    }, output_path)
    
    tokenizer.save_pretrained(args.out_dir)
    print(f"Tokenizer files saved to {args.out_dir}")
    print("Merge complete.")


def quantize(args, device, model=None, tokenizer=None, out_dir=None):
    print(f"Running in QUANTIZE mode with {args.qtype} precision...")
    
    if model is None or tokenizer is None:
        if not args.model_path:
            raise ValueError("--model-path is required for quantize mode.")
        print(f"Loading full-precision model from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model, _ = load_full_model_with_config(args.model_path, device)

    if out_dir is None:
        if not args.out_dir:
            out_dir = args.model_path.rstrip('/\\') + f"-{args.qtype}"
        else:
            out_dir = args.out_dir

    export_and_quantize_model(out_dir, model, tokenizer, qtype=args.qtype)


def chat(args, device, tokenizer):
    print("Running in CHAT mode...")

    model = None
    shadow_model = None
    config = None
    is_quantized = False
    inference_device = "cpu"
    ltm_has_been_updated = False

    npz_files = [f for f in os.listdir(args.model_path) if f.endswith('.npz')]
    if npz_files:
        if not _HAS_KERNEL:
            print("ERROR: Cannot run quantized chat without the C++ kernel.")
            return
        model, config = load_quantized(args.model_path)
        is_quantized = True
        print(f"Loaded quantized model with {model.qtype} weights.")

        if args.device == 'vulkan':
            if not _HAS_VULKAN:
                print("WARNING: Vulkan support not found in kernel. Falling back to CPU.")
            else:
                inference_device = "vulkan"
                print("INFO: Using Vulkan for inference.")
        
        if args.enable_quantized_learning:
            if not args.shadow_model_path:
                raise ValueError("To enable learning on a quantized model, you must provide the original full-precision model directory via --shadow-model-path.")
            print("Loading full-precision 'shadow' model for online learning...")
            shadow_model, _ = load_full_model_with_config(args.shadow_model_path, device)
            
            shadow_model.ltm.load_state_dict(model.ltm.state_dict())
            shadow_model.eval()

    else:
        model, config = load_full_model_with_config(args.model_path, device)
        inference_device = device

    if args.ltm_lora_path:
        print(f"LTM online learning is ACTIVE. Updates will be stored separately at: {args.ltm_lora_path}")
        updatable_model = shadow_model if is_quantized else model
        updatable_model.ltm.accumulate_deltas = True
        if os.path.exists(args.ltm_lora_path):
            print("Loading existing LTM deltas...")
            deltas = torch.load(args.ltm_lora_path)
            updatable_model.ltm.vals.data.add_(deltas.to(updatable_model.ltm.vals.device))
            updatable_model.ltm.ltm_deltas.data = deltas.to(updatable_model.ltm.ltm_deltas.device)
            if is_quantized:
                model.ltm.load_state_dict(updatable_model.ltm.state_dict())
    else:
        print("LTM online learning is ACTIVE. Updates will modify base model weights directly in memory.")

    if not is_quantized:
        model.eval()
        
    ltm_scheduler = None
    if not args.static_ltm_lr and (not is_quantized or args.enable_quantized_learning):
        print("INFO: Using Cosine Annealing schedule for LTM updates.")
        print(f"      - Max LR: {args.ltm_lr:.2e}, Min LR: {args.ltm_schedule_min_lr:.2e}, Cycle Steps: {args.ltm_schedule_steps}")
        dummy_param = nn.Parameter(torch.tensor(0.0))
        ltm_optimizer = torch.optim.SGD([dummy_param], lr=args.ltm_lr)
        ltm_scheduler = CosineAnnealingLR(
            ltm_optimizer,
            T_max=args.ltm_schedule_steps,
            eta_min=args.ltm_schedule_min_lr
        )

    print("\nWelcome to Chronos Chat. Type 'exit' or 'quit' to end.")
    if _HAS_KEYBOARD:
        print("Press Ctrl+X to stop generation at any time.")
    print("="*50)

    try:
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ["exit", "quit"]:
                break

            prompt_format = f"### Instruction:\n{prompt}\n\n### Response:\n"
            prompt_ids = tokenizer.encode(prompt_format, return_tensors="pt").to(device)

            print("\nChronos: ", end="", flush=True)
            response_ids = []

            h_state = torch.zeros(1, config.h_hidden, device=device if not is_quantized else "cpu")
            l_state = torch.zeros(1, config.l_hidden, device=device if not is_quantized else "cpu")

            current_ids = prompt_ids
            with torch.no_grad():
                for i in range(args.max_new_tokens):
                    if _HAS_KEYBOARD and keyboard.is_pressed('ctrl+x'):
                        print("\n[Generation interrupted by user.]", end="", flush=True)
                        break

                    model_input_ids = current_ids.to("cpu") if is_quantized else current_ids.to(device)

                    if is_quantized:
                        outputs = model(model_input_ids, h_state, l_state, device=inference_device)
                        h_state, l_state = outputs['h_state'], outputs['l_state']
                    else:
                        outputs = model(model_input_ids)

                    logits = outputs["logits"].to(device)
                    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

                    response_ids.append(next_token_id.item())
                    decoded_token = tokenizer.decode([next_token_id.item()])
                    if "###" in decoded_token:
                        break
                    print(decoded_token, end="", flush=True)

                    current_ids = torch.cat([current_ids, next_token_id], dim=1)

            if len(response_ids) > 0 and (not is_quantized or args.enable_quantized_learning):
                update_model = shadow_model if is_quantized else model
                target_device = device

                if is_quantized:
                    print("\n[Updating LTM via shadow model...]", end="", flush=True)
                
                update_model.train()
                with torch.enable_grad():
                    full_sequence = torch.cat([prompt_ids[0], torch.tensor(response_ids, device=target_device)], dim=0).unsqueeze(0)
                    labels = torch.cat([torch.full_like(prompt_ids[0], -100), torch.tensor(response_ids, device=target_device)], dim=0).unsqueeze(0)

                    update_model.zero_grad()
                    outputs = update_model(input_ids=full_sequence, labels=labels)
                    loss = outputs["loss"]

                    if loss is not None and not torch.isnan(loss):
                        loss.backward()
                        ltm_grads = outputs["topk_vals"].grad

                        if ltm_scheduler:
                            current_ltm_lr = ltm_scheduler.get_last_lr()[0]
                            print(f"[LTM LR: {current_ltm_lr:.2e}]", end="", flush=True)
                            ltm_scheduler.step()
                        else:
                            current_ltm_lr = update_model.ltm.lr
                        
                        update_model.ltm.inner_update(outputs["topk_idx"], ltm_grads, current_lr=current_ltm_lr)
                        ltm_has_been_updated = True

                        if is_quantized:
                            model.ltm.load_state_dict(update_model.ltm.state_dict())
                
                update_model.eval()
                if is_quantized:
                    print("[Done]", end="", flush=True)

            print("\n\n" + "="*50)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected. Exiting chat.]")
    
    finally:
        updatable_model = shadow_model if is_quantized else model
        if args.ltm_lora_path and updatable_model.ltm.accumulate_deltas:
            if torch.any(updatable_model.ltm.ltm_deltas != 0):
                print(f"\nSaving LTM memory deltas to {args.ltm_lora_path}...")
                torch.save(updatable_model.ltm.ltm_deltas.cpu(), args.ltm_lora_path)
                print("Done.")
            else:
                print("\nNo new LTM updates to save as LoRA.")
        
        elif not is_quantized and not args.ltm_lora_path and ltm_has_been_updated:
                while True:
                    response = input(f"Do you want to save the learned LTM updates back to '{args.model_path}'? (y/n): ").lower()
                    if response in ["y", "yes"]:
                        print(f"\nSaving updated model to {args.model_path}...")
                        output_weights_path = os.path.join(args.model_path, MODEL_WEIGHTS_NAME)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': dict(model.config)
                        }, output_weights_path)
                        print("✅ Save complete.")
                        break
                    elif response in ["n", "no"]:
                        print("Changes will be discarded. Exiting.")
                        break
        
        elif is_quantized and args.enable_quantized_learning and ltm_has_been_updated:
            print("\n--- LTM has been updated during this session ---")
            
            output_dir = args.model_path
            
            while True:
                response = input(f"Do you want to save these changes by re-quantizing the model to '{output_dir}'? (y/n): ").lower()
                if response in ["y", "yes"]:
                    print(f"\nRe-quantizing model with updated LTM to {output_dir}...")
                    export_and_quantize_model(output_dir, shadow_model, tokenizer, model.qtype)
                    print("✅ Save complete.")
                    break
                elif response in ["n", "no"]:
                    print("Changes will be discarded. Exiting.")
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")


def main():
    parser = argparse.ArgumentParser(description="Chronos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora"], help="Operation mode.")

    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, default=None, help="[Train/Finetune] Path to training JSON or JSONL file.")
    path_group.add_argument("--model-path", type=str, default=None, help="[All except Train] Path to the model directory for loading.")
    path_group.add_argument("--out-dir", type=str, default="./chronos_model", help="[Train/Finetune/Merge/Quantize] Directory to save the new model/adapter.")
    path_group.add_argument("--lora-adapter-path", type=str, default=None, help="[Merge] Path to the trained LoRA adapter directory.")
    path_group.add_argument("--tokenizer-path", type=str, default="microsoft/phi-2", help="[Train] Path or HF name of the tokenizer to use for a new model.")
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="[Train] Path to a specific training checkpoint .pt file to resume from.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="[Chat] Path to the original full-precision model dir, required for online learning with a quantized model.")

    arch_group = parser.add_argument_group('Architecture (for --mode train)')
    arch_group.add_argument("--context_dim", type=int, default=512)
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=2048)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    arch_group.add_argument("--h_hidden", type=int, default=512)
    arch_group.add_argument("--l_hidden", type=int, default=512)
    arch_group.add_argument("--h_steps", type=int, default=1, help="Number of high-level refinement steps in HRM.")
    arch_group.add_argument("--max_l_steps", type=int, default=10, help="[HRM] Maximum number of low-level iterations before forcing completion.")
    arch_group.add_argument("--l_conv_atol", type=float, default=1e-5, help="[HRM] Absolute tolerance for checking L-module state convergence.")
    arch_group.add_argument("--ltm_topk", type=int, default=4, help="Number of LTM slots to retrieve per token.")
    arch_group.add_argument("--max_length", type=int, default=1024)
    arch_group.add_argument("--auto-max-length", action="store_true", help="Automatically scan the dataset to find the longest sequence and set it as max_length.")

    train_group = parser.add_argument_group('Training and Finetuning')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", type=int, default=1, help="Simulates a larger batch size.")
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6, help="Min LR for cosine annealing.")
    train_group.add_argument("--disable-lr-schedule", action="store_true", help="Use a fixed LR instead of cosine annealing.")
    train_group.add_argument("--ltm_lr", type=float, default=1e-2, help="[Static] LR for LTM updates, or [Scheduled] MAX LR for the LTM cosine schedule.")
    train_group.add_argument("--kayla", action="store_true", help="Enable Kayla-style instruction tuning (with thought-process).")
    train_group.add_argument("--lora_r", type=int, default=8, help="[Finetune] LoRA rank.")
    train_group.add_argument("--lora_alpha", type=int, default=16, help="[Finetune] LoRA alpha.")
    train_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="[Finetune] Target percentage of params to train (e.g., 1.5 for 1.5%%). Overrides --lora_r.")
    train_group.add_argument("--quantize-on-complete", action="store_true", help="[Train] Automatically quantize after training.")

    infer_group = parser.add_argument_group('Inference (Chat)')
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--enable-quantized-learning", action="store_true", help="[Chat] Enable LTM updates for quantized models. Requires --shadow-model-path.")
    infer_group.add_argument("--ltm-lora-path", type=str, default=None, help="[Chat] Optional: Path to save/load LTM updates as a separate delta file.")
    infer_group.add_argument("--device", type=str, default="cpu", choices=["cpu", "vulkan"], help="[Chat] Device for quantized inference.")
    infer_group.add_argument("--static-ltm-lr", action="store_true", help="[Chat] Disable the cosine annealing schedule for LTM updates and use a fixed LR instead.")
    infer_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="[Chat] The number of updates in one cosine annealing cycle for LTM learning.")
    infer_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="[Chat] The minimum learning rate for the LTM cosine annealing schedule.")

    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument("--qtype", type=str, default="INT4", choices=["INT4", "Q4_0", "Q8_0", "Q2_K"], help="Quantization type/format.")
    other_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))

    args = parser.parse_args()

    set_threads(args.threads)
    pt_device = pick_device()
    print(f"Using PyTorch device: {pt_device}")

    tokenizer = None
    if args.mode == "train":
        print(f"Loading tokenizer '{args.tokenizer_path}' for new model training...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    elif args.model_path:
        print(f"Loading tokenizer from model directory: {args.model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer from '{args.model_path}': {e}")
            sys.exit(1)
    elif args.mode not in ['train', 'quantize']:
        parser.error(f"--model-path is required for mode '{args.mode}'.")


    if tokenizer and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "train":
        args.vocab_size = len(tokenizer)

    if args.auto_max_length:
        if not args.train:
            parser.error("--auto-max-length requires a --train file to be specified.")

        print("INFO: --auto-max-length enabled. Scanning dataset to find the maximum sequence length...")
        max_found_length = 0

        with open(args.train, 'r', encoding='utf-8') as f:

            def get_text_from_obj(obj, kayla_mode):
                try:
                    if kayla_mode:
                        feelings_part = f"### Feelings:\n{obj.get('feelings')}\n\n" if obj.get('feelings') else ""
                        return (f"### Instruction:\n{obj.get('Instruction', '')}\n\n"
                                f"{feelings_part}"
                                f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                                f"### Response:\n{obj.get('output', '')}")
                    else:
                        return f"### Instruction:\n{obj.get('instruction', '')}\n\n### Response:\n{obj.get('output', '') or obj.get('response', '')}"
                except:
                    return ""

            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in tqdm(data, desc="Scanning JSON"):
                        text = get_text_from_obj(obj, args.kayla)
                        length = len(tokenizer.encode(text))
                        if length > max_found_length: max_found_length = length
            except json.JSONDecodeError:
                f.seek(0)
                for line in tqdm(f, desc="Scanning JSONL"):
                    try:
                        obj = json.loads(line)
                        text = get_text_from_obj(obj, args.kayla)
                        length = len(tokenizer.encode(text))
                        if length > max_found_length: max_found_length = length
                    except:
                        continue

        if max_found_length > 0:
            max_found_length = (max_found_length + 16 + 7) & -8 
            print(f"✅ Auto-scan complete. Setting max_length to {max_found_length}.")
            args.max_length = max_found_length
            tokenizer.model_max_length = max_found_length
        else:
            print("⚠️ WARNING: Auto-scan did not find any valid entries. Using default max_length.")

    if args.mode == "train":
        if not args.train: parser.error("`--train` is required for train mode.")
        train(args, pt_device, tokenizer)
    elif args.mode == "finetune":
        if not args.train: parser.error("`--train` is required for finetune mode.")
        if not args.model_path: parser.error("`--model-path` is required for finetune mode.")
        finetune(args, pt_device, tokenizer)
    elif args.mode == "merge-lora":
        if not args.model_path: parser.error("`--model-path` is required for merge-lora mode.")
        if not args.lora_adapter_path: parser.error("`--lora-adapter-path` is required for merge-lora mode.")
        merge_lora(args, pt_device, tokenizer)
    elif args.mode == "quantize":
        quantize(args, pt_device)
    elif args.mode == "chat":
        chat(args, pt_device, tokenizer)


if __name__ == "__main__":
    main()
