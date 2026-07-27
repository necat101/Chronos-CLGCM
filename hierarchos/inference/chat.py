"""
Hierarchos Chat Module - Ported from monolith for feature parity.

Features:
- Online learning with positive/negative feedback
- LTM updates with gradient-based memory system
- Proper top-k/top-p sampling
- Slash commands (/reset, /reset_ltm, /status, /settings, /temp, /topk, /topp, /filter, /learn)
- State persistence (drift_state, RWKV states, global_pos_offset)
- Save on exit logic
- Interrupt handling (Ctrl+C)
- Quantized model support with shadow model for learning
"""

import os
import sys
import signal
import traceback
import math
import time
import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils.device import is_directml_device
from ..utils.checkpoint import (
    load_full_model_with_config,
    sanitize_model_state_dict,
    save_checkpoint_safely,
)
from .chat_state import (
    CHAT_STATE_KIND,
    CHAT_STATE_VERSION,
    chat_state_config_signature,
    clear_ltm_working_memory,
    normalize_recurrent_state_for_model,
    recurrent_state_metadata,
    tensor_to_cpu,
    validate_chat_state_payload_compatible,
)

# --- Signal Handling for Interrupt ---
_interrupt_flag = False
_original_sigint_handler = None

def _handle_interrupt(sig, frame):
    """Sets the interrupt flag when SIGINT (Ctrl+C) is received."""
    global _interrupt_flag
    if not _interrupt_flag:
        print("\n[Interrupt received. Finishing current generation... Press Ctrl+C again to force exit.]", flush=True)
        _interrupt_flag = True
    else:
        print("\n[Forcing exit...]", flush=True)
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)
        sys.exit(1)


# --- Feedback Detection ---
def is_positive_feedback(text: str) -> bool:
    """Checks if the user input looks like positive validation."""
    text = text.lower().strip()
    positive_triggers = {
        "good", "great", "correct", "yes", "nice", "cool", "perfect",
        "thanks", "thx", "+", "right", "accurate"
    }
    if text in positive_triggers:
        return True
    first_word = text.split(' ')[0] if ' ' in text else text
    first_word = ''.join(c for c in first_word if c.isalnum())
    return first_word in positive_triggers or text.startswith("/learn")


def is_correction_or_instruction(text: str) -> bool:
    """Checks if user input looks like a correction or new instruction."""
    text_lower = text.lower().strip()
    correction_triggers = ["no", "wrong", "incorrect", "actually", "false", "not true"]
    for trigger in correction_triggers:
        if text_lower.startswith(trigger):
            return len(text.split()) > 3
    word_count = len(text.split())
    if word_count > 5 and not is_positive_feedback(text):
        return True
    return False


def extract_correction_text(text: str):
    """Extracts corrected target text from terse feedback like 'its 4'."""
    clean = text.strip()
    lower = clean.lower()
    prefixes = (
        "it's ",
        "its ",
        "it is ",
        "actually ",
        "actually, ",
        "no, ",
        "wrong, ",
        "incorrect, ",
        "the answer is ",
        "answer is ",
        "it should be ",
        "should be ",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            correction = clean[len(prefix):].strip()
            return correction if correction else None
    return None


# --- Training-Parity Format Wrapper & Parser ---
def wrap_for_hierarchos(raw_text, system_prompt=None, alpaca_mode=False, input_context=None):
    """Wraps user input into the exact format the model saw during training.
    
    The training pipeline supports both Alpaca and User/Assistant prompt
    formats. This wrapper must produce the matching prefix so the model sees
    in-distribution text at inference time.
    
    An optional system_prompt is prepended inside the current instruction to
    guide behavior without introducing OOD formatting.
    """
    clean_text = raw_text.strip()
    if system_prompt:
        clean_text = f"[{system_prompt}]\n{clean_text}"
    if alpaca_mode:
        prompt = ""
        input_context = str(input_context or "").strip()
        if input_context:
            prompt += f"### Previous Context:\n{input_context}\n\n"
        prompt += f"### Instruction:\n{clean_text}\n\n"
        return prompt + "### Response:\n"
    return f"User: {clean_text}\n\nAssistant: "


def build_chat_input_context(turn_history, max_turns=4, max_chars=3000):
    """Build Alpaca Input text from previous chat turns."""
    try:
        max_turns = int(max_turns)
    except (TypeError, ValueError):
        max_turns = 0
    try:
        max_chars = int(max_chars)
    except (TypeError, ValueError):
        max_chars = 0
    if max_turns <= 0 or max_chars <= 0:
        return ""
    selected = list(turn_history[-max_turns:])
    context = "\n\n".join(t.strip() for t in selected if str(t).strip()).strip()
    if len(context) > max_chars:
        context = context[-max_chars:].lstrip()
    return context


def clean_hierarchos_output(raw_generation):
    """Cleans any trailing artifacts from the model's generation."""
    # Model generates plain text followed by EOS; strip whitespace only
    return raw_generation.strip()


def passive_response_quality(token_ids):
    """Conservative quality gate for self-generated passive LTM writes."""
    ids = [int(t) for t in token_ids]
    if len(ids) < 8:
        return False, "too short"

    unique_ratio = len(set(ids)) / max(1, len(ids))
    if len(ids) >= 20 and unique_ratio < 0.35:
        return False, "low token diversity"

    for n in (3, 4):
        if len(ids) < n * 2:
            continue
        ngrams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        ngram_unique_ratio = len(set(ngrams)) / max(1, len(ngrams))
        if ngram_unique_ratio < 0.75:
            return False, f"repeated {n}-grams"

    return True, "ok"


def parse_temperature_setting(raw_value: str) -> float:
    """Parses a runtime temperature value constrained to 0.00..1.00 in 0.05 steps."""
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        raise ValueError("temperature must be between 0 and 1")

    scaled = round(value * 20)
    stepped = scaled / 20.0
    if abs(value - stepped) > 1e-8:
        raise ValueError("temperature must use 0.05 increments")
    return stepped


def _chat_state_config_signature(config, model=None):
    return chat_state_config_signature(config, model)


def _tensor_to_cpu(value):
    return tensor_to_cpu(value)


def _default_chat_state_path(model_path: str) -> str:
    safe_model_name = os.path.basename(os.path.abspath(model_path).rstrip(os.sep)) or "model"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    state_dir = os.path.abspath(os.path.join(os.getcwd(), "hierarchos_chat_states"))
    os.makedirs(state_dir, exist_ok=True)
    return os.path.join(state_dir, f"{safe_model_name}-{timestamp}.pt")


def _normalize_chat_state_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser((path or "").strip().strip('"')))


def save_hierarchical_chat_state(
    path,
    *,
    config,
    model=None,
    model_path,
    h_state,
    l_state,
    prev_context,
    target_context,
    drift_state,
    ltm_state=None,
    total_tokens_generated,
):
    """Save only tiny chat continuation tensors; never LTM/model weights."""
    path = _normalize_chat_state_path(path)
    if not path:
        raise ValueError("No chat state path provided.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rosa_past_tokens = _rosa_past_tokens_from_ltm_state(ltm_state)
    rosa_states = _rosa_states_from_ltm_state(ltm_state)
    payload = {
        "version": CHAT_STATE_VERSION,
        "kind": CHAT_STATE_KIND,
        "saved_at": time.time(),
        "model_path": os.path.abspath(model_path) if model_path else None,
        "config_signature": _chat_state_config_signature(config, model=model),
        "total_tokens_generated": int(total_tokens_generated or 0),
        "h_state": _tensor_to_cpu(h_state),
        "l_state": _tensor_to_cpu(l_state),
        "prev_context": _tensor_to_cpu(prev_context),
        "target_context": _tensor_to_cpu(target_context),
        "drift_state": _tensor_to_cpu(drift_state),
        "rosa_past_tokens": rosa_past_tokens,
        "rosa_states": rosa_states,
    }
    payload.update(
        recurrent_state_metadata(
            model=model,
            config=config,
            h_state=h_state,
            l_state=l_state,
        )
    )
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_hierarchical_chat_state(path, *, config, device, model=None):
    """Load a tiny chat state file without restoring LTM working memory."""
    path = _normalize_chat_state_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chat state file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != CHAT_STATE_KIND:
        raise RuntimeError(f"Not a Hierarchos chat runtime state file: {path}")

    validate_chat_state_payload_compatible(payload, config, model=model)

    def tensor(name):
        value = payload.get(name)
        if torch.is_tensor(value):
            return value.to(device)
        return None
    
    h_state = tensor("h_state")
    l_state = tensor("l_state")
    if model is not None:
        if h_state is not None:
            h_state = normalize_recurrent_state_for_model(
                h_state,
                model,
                "h_rnn",
                device=device,
            )
        if l_state is not None:
            l_state = normalize_recurrent_state_for_model(
                l_state,
                model,
                "l_rnn",
                device=device,
            )

    return {
        "h_state": h_state,
        "l_state": l_state,
        "prev_context": tensor("prev_context"),
        "target_context": tensor("target_context"),
        "drift_state": tensor("drift_state"),
        "rosa_past_tokens": payload.get("rosa_past_tokens").cpu()
        if torch.is_tensor(payload.get("rosa_past_tokens"))
        else None,
        "rosa_states": copy.deepcopy(payload.get("rosa_states"))
        if payload.get("rosa_states") is not None
        else None,
        "total_tokens_generated": int(payload.get("total_tokens_generated") or 0),
    }


def _rosa_past_tokens_from_ltm_state(ltm_state):
    """Extract full ROSA token history from an LTM state tuple."""
    if ltm_state is None or not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 3:
        return None
    return _tensor_to_cpu(ltm_state[2])


def _rosa_states_from_ltm_state(ltm_state):
    """Extract V8 ROSA automaton state from an LTM state tuple."""
    if ltm_state is None or not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 4:
        return None
    rosa_states = ltm_state[3]
    if rosa_states is None:
        return None
    try:
        return copy.deepcopy(rosa_states)
    except Exception:
        return rosa_states


def _chat_ltm_state_from_rosa_context(model, rosa_past_tokens, rosa_states=None):
    """Rehydrate V8 ROSA continuation without restoring per-chat LTM fast buffers."""
    if not torch.is_tensor(rosa_past_tokens) or not hasattr(model, "ltm"):
        return None

    ltm = model.ltm
    fast_vals = getattr(ltm, "fast_vals", None)
    mom_vals = getattr(ltm, "_mom_vals", None)
    if not torch.is_tensor(fast_vals) or not torch.is_tensor(mom_vals):
        return None

    timestamps = getattr(ltm, "timestamps", None)
    sources = getattr(ltm, "sources", None)
    return (
        fast_vals,
        torch.zeros_like(mom_vals),
        rosa_past_tokens.detach().cpu().clone(),
        copy.deepcopy(rosa_states) if rosa_states is not None else None,
        timestamps,
        sources,
    )


def _chat_ltm_state_from_rosa_past(model, rosa_past_tokens):
    return _chat_ltm_state_from_rosa_context(model, rosa_past_tokens, None)


def _setting_value(settings, name, default):
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _normalize_ltm_training_mode(value) -> str:
    mode = str(value or "inner-update").strip().lower().replace("_", "-")
    aliases = {
        "inner": "inner-update",
        "inner-update": "inner-update",
        "inner-updates": "inner-update",
        "supervised": "inner-update",
        "supervised-inner-update": "inner-update",
        "readonly": "read-only",
        "read-only": "read-only",
        "read_only": "read-only",
        "inference": "read-only",
        "inference-like": "read-only",
        "inference-like-ltm": "read-only",
    }
    return aliases.get(mode, "inner-update")


def _checkpoint_ltm_training_mode(config) -> str:
    return _normalize_ltm_training_mode(getattr(config, "ltm_training_mode", "inner-update"))


def _checkpoint_has_trained_hebbian_writer(config) -> bool:
    return bool(_setting_value(config, "val_proj_trained", False))


def _print_ltm_chat_alignment(config, learning_enabled: bool):
    mode = _checkpoint_ltm_training_mode(config)
    print(f"Checkpoint LTM training mode: {mode}")
    if mode == "read-only":
        print("Chat LTM alignment: using read-only recurrent/LTM state during prefill and generation.")
        if learning_enabled:
            print("Explicit feedback/validation can still write LTM memory; passive response learning remains opt-in.")
    else:
        print("WARNING: checkpoint was trained with supervised LTM inner updates.")
        print("Normal chat generation has no label-gradient inner update, so coherence may lag until retrained/rescued with --ltm-training-mode read-only.")
    if learning_enabled and not _checkpoint_has_trained_hebbian_writer(config):
        print(
            "Legacy checkpoint note: val_proj was not trained, so unsafe Hebbian "
            "validation writes are disabled; gradient-derived feedback remains available."
        )
    return mode


def should_stop_generation_from_uncertainty(logits, response_ids, tokenizer=None, settings=None):
    """Stop when the model has clearly fallen into high-entropy tail sampling."""
    if logits is None:
        return False
    try:
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        logits = logits.float()
        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return True

        generated_count = len(response_ids or [])
        entropy_threshold = float(_setting_value(settings, "entropy_stop_threshold", 0.0) or 0.0)
        min_tokens = int(_setting_value(settings, "entropy_stop_min_tokens", 3) or 0)
        top_prob_ceiling = float(_setting_value(settings, "entropy_stop_top_prob", 0.05) or 0.0)
        eos_prob_threshold = float(_setting_value(settings, "eos_stop_prob", 0.0) or 0.0)

        top_prob = float(probs.max().item())
        eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        if (
            eos_id is not None
            and 0 <= int(eos_id) < probs.shape[-1]
            and eos_prob_threshold > 0
            and generated_count >= 1
        ):
            eos_prob = float(probs[0, int(eos_id)].item())
            if eos_prob >= eos_prob_threshold:
                return True

        if entropy_threshold <= 0 or generated_count < min_tokens:
            return False

        entropy = -((probs * torch.log(probs + 1e-10)).sum(-1)).item()
        return entropy >= entropy_threshold and (
            top_prob_ceiling <= 0 or top_prob <= top_prob_ceiling
        )
    except Exception:
        return False


def sample_next_token(
    logits,
    *,
    temperature=0.7,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    previous_tokens=None,
):
    """Sample one token without mutating caller-owned logits."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"Expected [batch, vocab] logits, got shape {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("Refusing to sample from non-finite logits.")

    scores = logits.float().clone()
    repetition_penalty = float(repetition_penalty)
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be greater than zero")
    if previous_tokens is not None and len(previous_tokens) > 0 and repetition_penalty != 1.0:
        token_ids = sorted({int(token) for token in previous_tokens if 0 <= int(token) < scores.shape[-1]})
        if token_ids:
            token_index = torch.tensor(token_ids, device=scores.device, dtype=torch.long)
            selected = scores.index_select(1, token_index)
            selected = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
            scores.index_copy_(1, token_index, selected)

    temperature = float(temperature)
    if temperature <= 0:
        return scores.argmax(dim=-1, keepdim=True)
    scores.div_(max(temperature, 1e-6))

    top_k = int(top_k or 0)
    if top_k > 0 and top_k < scores.shape[-1]:
        threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)

    top_p = float(top_p)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in the interval (0, 1]")
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        scores.masked_fill_(remove, -torch.inf)

    probs = F.softmax(scores, dim=-1)
    if not bool(torch.isfinite(probs).all().item()) or bool((probs.sum(dim=-1) <= 0).any().item()):
        raise RuntimeError("Sampling filters produced an invalid probability distribution.")
    return torch.multinomial(probs, num_samples=1)


def tbptt_chunk_ranges(length, chunk_size, global_offset=0):
    """Split a sequence on absolute TBPTT boundaries, including carried chat state."""
    length = max(0, int(length or 0))
    chunk_size = int(chunk_size or 0)
    global_offset = max(0, int(global_offset or 0))
    if length == 0:
        return []
    if chunk_size <= 0:
        return [(0, length)]

    ranges = []
    start = 0
    while start < length:
        absolute_start = global_offset + start
        remainder = absolute_start % chunk_size
        width = chunk_size - remainder if remainder else chunk_size
        end = min(length, start + width)
        ranges.append((start, end))
        start = end
    return ranges


def uses_full_sample_inference_recurrence(config) -> bool:
    """Return whether inference should use uninterrupted sample recurrence.

    New checkpoints persist ``inference_recurrence_mode`` so recurrence geometry
    is independent from ``inference_logit_parity``.  The latter still controls
    training-equivalent Manager/Worker refinement during evaluation, but a model
    resumed under TBPTT must retain TBPTT chunk-boundary drift semantics.

    Older checkpoints predate the explicit geometry field.  An explicitly
    persisted ``full_sample_bptt`` value is still authoritative: in particular,
    a TBPTT checkpoint may independently enable inference_logit_parity so eval
    uses the training Manager/Worker refinement policy.  Only checkpoints that
    lack both geometry fields fall back to the historical
    ``inference_logit_parity`` behavior.
    """
    if isinstance(config, dict):
        recurrence_mode = config.get("inference_recurrence_mode")
        has_full_sample_bptt = "full_sample_bptt" in config
        full_sample_bptt = config.get("full_sample_bptt", False)
        inference_logit_parity = config.get("inference_logit_parity", False)
    else:
        recurrence_mode = getattr(config, "inference_recurrence_mode", None)
        has_full_sample_bptt = hasattr(config, "full_sample_bptt")
        full_sample_bptt = getattr(config, "full_sample_bptt", False)
        inference_logit_parity = getattr(config, "inference_logit_parity", False)

    if recurrence_mode is not None:
        normalized = str(recurrence_mode).strip().lower().replace("_", "-")
        if normalized == "full-sample":
            return True
        if normalized == "tbptt":
            return False

    if has_full_sample_bptt:
        return bool(full_sample_bptt)
    return bool(inference_logit_parity)


def resolve_inference_prefill_chunk_size(config, requested=None):
    """Choose the prefill geometry that matches the checkpoint's training graph."""
    if requested is not None:
        return max(0, int(requested or 0))
    if uses_full_sample_inference_recurrence(config):
        # training_chunk_size is retained by exact-BPTT training for token-cache,
        # ROSA, LTM-decay, and compile metadata.  It is not a forward boundary.
        return 0
    if isinstance(config, dict):
        training_chunk_size = config.get("training_chunk_size", 0)
    else:
        training_chunk_size = getattr(config, "training_chunk_size", 0)
    return max(0, int(training_chunk_size or 0))


def boundary_drift_seed(
    drift_state,
    global_offset,
    chunk_size,
    *,
    exact_full_sample=False,
):
    """Return the drift seed used by the checkpoint's training recurrence.

    Exact full-sample training has no semantic boundary at an activation-only
    segment edge. Its next token derives drift from the attached worker state,
    exactly as an uninterrupted forward does. Legacy TBPTT intentionally
    carried final drift into aligned training-chunk boundaries, so retain that
    behavior for older checkpoints.
    """
    if exact_full_sample:
        return None
    chunk_size = int(chunk_size or 0)
    global_offset = int(global_offset or 0)
    if chunk_size > 0 and global_offset > 0 and global_offset % chunk_size == 0:
        return drift_state
    return None


def advance_chat_model_state(
    model,
    token_ids,
    *,
    device,
    h_state,
    l_state,
    prev_context,
    target_context,
    drift_state,
    drift_seed,
    ltm_state,
    global_pos_offset,
    min_timestamp=0.0,
    source_filter=None,
    is_quantized=False,
    inference_device=None,
):
    """Consume tokens once and return the model outputs plus updated runtime state."""
    model_input_ids = token_ids.cpu() if is_quantized else token_ids.to(device)

    def _quantized_state(value):
        return value.cpu() if torch.is_tensor(value) else value

    call_kwargs = {
        "input_ids": model_input_ids,
        "h_state": _quantized_state(h_state) if is_quantized else h_state,
        "l_state": _quantized_state(l_state) if is_quantized else l_state,
        "prev_context": _quantized_state(prev_context) if is_quantized else prev_context,
        "target_context": _quantized_state(target_context) if is_quantized else target_context,
        "drift_state": _quantized_state(drift_seed) if is_quantized else drift_seed,
        "ltm_memory_state": ltm_state,
        "global_pos_offset": int(global_pos_offset or 0),
        "min_timestamp": min_timestamp,
        "source_filter": source_filter,
    }
    if is_quantized:
        call_kwargs["device"] = inference_device

    outputs = model(**call_kwargs)
    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise RuntimeError("Chat model forward must return a dict containing logits")

    def _updated(name, previous):
        value = outputs.get(name)
        return previous if value is None else value

    state = (
        _updated("h_state", h_state),
        _updated("l_state", l_state),
        _updated("prev_context", prev_context),
        _updated("target_context", target_context),
        _updated("drift_state", drift_state),
        _updated("ltm_memory_state", ltm_state),
    )
    return outputs, state


def reset_active_ltm_state(model, ltm_state, *, preserve_rosa=True):
    """Clear both module buffers and the state tuple consumed by the next forward."""
    clear_ltm_working_memory(model)
    if not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 2:
        return None

    state = list(ltm_state)
    if torch.is_tensor(state[0]):
        state[0] = torch.zeros_like(state[0])
    if torch.is_tensor(state[1]):
        state[1] = torch.zeros_like(state[1])
    if len(state) >= 3 and not preserve_rosa:
        state[2] = None
    if len(state) >= 4 and not preserve_rosa:
        state[3] = None
    if len(state) >= 5 and torch.is_tensor(state[4]):
        state[4] = torch.zeros_like(state[4])
    if len(state) >= 6 and torch.is_tensor(state[5]):
        unknown_source = int(getattr(getattr(model, "ltm", None), "SRC_UNKNOWN", 0))
        state[5] = torch.full_like(state[5], unknown_source)
    return tuple(state)


def zero_ltm_momentum_state(model, ltm_state):
    """Reset optimizer-like LTM momentum in both active and module-owned state."""
    module_momentum = getattr(getattr(model, "ltm", None), "_mom_vals", None)
    if torch.is_tensor(module_momentum):
        with torch.no_grad():
            module_momentum.zero_()
    if not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 2:
        return ltm_state
    state = list(ltm_state)
    if torch.is_tensor(state[1]):
        state[1] = torch.zeros_like(state[1])
    return tuple(state)


def prepare_online_ltm_gradients(grads, max_norm):
    """Reject non-finite memory gradients, then value- and norm-clip finite ones."""
    if not torch.is_tensor(grads):
        return None
    prepared = grads.detach().float().clone()
    if not bool(torch.isfinite(prepared).all().item()):
        return None
    max_norm = float(max_norm or 0.0)
    if max_norm > 0:
        prepared.clamp_(min=-max_norm, max=max_norm)
        grad_norm = prepared.norm()
        if not bool(torch.isfinite(grad_norm).item()):
            return None
        prepared.mul_(torch.clamp(prepared.new_tensor(max_norm) / (grad_norm + 1e-8), max=1.0))
    return prepared


def ltm_replay_seed_state(ltm_state):
    """Reuse fast memory while replaying the supervised sequence from fresh ROSA history."""
    if not isinstance(ltm_state, (tuple, list)) or len(ltm_state) < 2:
        return ltm_state
    state = list(ltm_state)
    if len(state) >= 3:
        state[2] = None
    if len(state) >= 4:
        state[3] = None
    return tuple(state)


def consolidate_ltm_state_for_save(model, ltm_state) -> bool:
    """Fold fast memory into slow LTM values so a fresh chat load retains it."""
    ltm = getattr(model, "ltm", None)
    slow_vals = getattr(ltm, "vals", None)
    if ltm is None or not torch.is_tensor(slow_vals):
        return False
    fast_vals = ltm_state[0] if isinstance(ltm_state, (tuple, list)) and ltm_state else getattr(ltm, "fast_vals", None)
    if not torch.is_tensor(fast_vals):
        return False
    fast_vals = fast_vals.detach()
    if fast_vals.dim() == slow_vals.dim() + 1 and fast_vals.shape[0] == 1:
        fast_vals = fast_vals.squeeze(0)
    if fast_vals.shape != slow_vals.shape:
        raise ValueError(
            f"Cannot consolidate LTM state with shape {tuple(fast_vals.shape)} into {tuple(slow_vals.shape)}"
        )
    if not bool(torch.isfinite(fast_vals).all().item()):
        raise ValueError("Cannot save non-finite LTM fast memory")
    with torch.no_grad():
        consolidated = slow_vals.float() + fast_vals.to(device=slow_vals.device, dtype=torch.float32)
        if not bool(torch.isfinite(consolidated).all().item()):
            raise ValueError("LTM consolidation produced non-finite slow values")
        slow_vals.copy_(consolidated.to(dtype=slow_vals.dtype))
    clear_ltm_working_memory(model)
    return True


# --- Simple Generation Helper ---
def generate_sample(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """Simple generation for testing/comparison."""
    model.eval()
    previous_suppress_hebbian = getattr(model, "suppress_hebbian", False)
    model.suppress_hebbian = True
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    h_state, l_state, p_ctx, t_ctx = None, None, None, None
    drift_state = None
    ltm_state = None
    model_config = getattr(model, "config", None)
    exact_full_sample = uses_full_sample_inference_recurrence(model_config)
    prefill_chunk_size = resolve_inference_prefill_chunk_size(model_config)
    total_tokens_generated = 0

    try:
        generated = tokens
        with torch.no_grad():
            prefill_len = int(tokens.shape[1])
            prefill_step = prefill_chunk_size if prefill_chunk_size > 0 else prefill_len
            prefill_step = max(1, int(prefill_step))
            logits = None
            chunk_drift_state = None
            for start in range(0, prefill_len, prefill_step):
                end = min(start + prefill_step, prefill_len)
                outputs = model(
                    input_ids=tokens[:, start:end],
                    h_state=h_state,
                    l_state=l_state,
                    prev_context=p_ctx,
                    target_context=t_ctx,
                    drift_state=chunk_drift_state,
                    ltm_memory_state=ltm_state,
                    global_pos_offset=start,
                    suppress_hebbian=True,
                )
                h_state, l_state = outputs['h_state'], outputs['l_state']
                p_ctx, t_ctx = outputs['prev_context'], outputs['target_context']
                drift_state = outputs.get('drift_state', drift_state)
                ltm_state = outputs.get('ltm_memory_state')
                chunk_drift_state = boundary_drift_seed(
                    drift_state,
                    end,
                    prefill_chunk_size,
                    exact_full_sample=exact_full_sample,
                )
                logits = outputs['logits'][:, -1, :]
            total_tokens_generated = prefill_len

            for _ in range(max_new_tokens):
                next_token = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

                outputs = model(
                    input_ids=next_token,
                    h_state=h_state,
                    l_state=l_state,
                    prev_context=p_ctx,
                    target_context=t_ctx,
                    drift_state=boundary_drift_seed(
                        drift_state,
                        total_tokens_generated,
                        prefill_chunk_size,
                        exact_full_sample=exact_full_sample,
                    ),
                    ltm_memory_state=ltm_state,
                    global_pos_offset=total_tokens_generated,
                    suppress_hebbian=True,
                )
                total_tokens_generated += 1
                h_state, l_state = outputs['h_state'], outputs['l_state']
                p_ctx, t_ctx = outputs['prev_context'], outputs['target_context']
                drift_state = outputs.get('drift_state', drift_state)
                ltm_state = outputs.get('ltm_memory_state')
                logits = outputs['logits'][:, -1, :]
    finally:
        model.suppress_hebbian = previous_suppress_hebbian

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# --- Main Chat Function ---
def chat(args, device, tokenizer):
    """
    Interactive chat mode with online learning support.
    
    Ported from hierarchos.py monolith for full feature parity.
    """
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    print("Running in CHAT mode...")
    
    # Import here to avoid circular imports
    from ..models.core import HierarchosCore
    from ..models.ltm import LTMModule
    
    # Try importing quantized model support
    try:
        from ..models.quantized import QuantizedHierarchos, load_quantized
        _HAS_QUANTIZED = True
    except ImportError:
        _HAS_QUANTIZED = False
    
    # =================================================================
    # 1. SETUP & SIGNAL HANDLING
    # =================================================================
    global _interrupt_flag, _original_sigint_handler
    _interrupt_flag = False
    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _handle_interrupt)
    except ValueError as e:
        print(f"Warning: Could not set SIGINT handler: {e}")
        _original_sigint_handler = None

    model = None
    shadow_model = None
    config = None
    is_quantized = False
    inference_device = device
    ltm_has_been_updated = False
    pending_training_data = None

    # =================================================================
    # 2. MODEL LOADING
    # =================================================================
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"Error: Model path not found at {args.model_path}")
        sys.exit(1)

    npz_files = (
        [f for f in os.listdir(args.model_path) if f.endswith('.npz')]
        if os.path.isdir(args.model_path)
        else []
    )

    if npz_files and _HAS_QUANTIZED:
        try:
            model, config = load_quantized(args.model_path, device=device)
            clear_ltm_working_memory(model)
            if isinstance(model, QuantizedHierarchos):
                is_quantized = True
                print(f"Loaded quantized model with {model.qtype} weights.")
                inference_device = "cpu"  # Quantized models run on CPU by default
            else:
                print("INFO: Loaded full-precision model (fallback active).")
                is_quantized = False
        except Exception as e:
            print(f"Error loading quantized model: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            model, config = load_full_model_with_config(args.model_path, device)
            clear_ltm_working_memory(model)
            print("Loaded full-precision model.")
        except Exception as e:
            print(f"Error loading model from {args.model_path}: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Handle shadow model for quantized learning
    enable_quantized_learning = getattr(args, 'enable_quantized_learning', False)
    shadow_model_path = getattr(args, 'shadow_model_path', None)
    
    if enable_quantized_learning and is_quantized:
        if not shadow_model_path:
            print("Error: --enable-quantized-learning requires --shadow-model-path")
            sys.exit(1)
        print("Loading full-precision 'shadow' model for online learning...")
        try:
            shadow_model, _ = load_full_model_with_config(shadow_model_path, device)
            shadow_model.ltm.load_state_dict(model.ltm.state_dict())
            clear_ltm_working_memory(shadow_model)
            shadow_model.eval()
        except Exception as e:
            print(f"Error loading shadow model: {e}")
            traceback.print_exc()
            sys.exit(1)

    tokenizer_vocab = len(tokenizer)
    model_vocab = getattr(config, "vocab_size", None)
    if model_vocab is not None and int(model_vocab) != int(tokenizer_vocab):
        print(
            f"Error: tokenizer vocabulary ({tokenizer_vocab}) does not match "
            f"checkpoint vocabulary ({int(model_vocab)})."
        )
        print("Use the exact tokenizer from training; generation with mismatched token IDs is invalid.")
        sys.exit(1)

    # =================================================================
    # 3. LTM & OPTIMIZER SETUP
    # =================================================================
    ltm_lora_path = getattr(args, 'ltm_lora_path', None)
    learning_enabled = not is_quantized or enable_quantized_learning
    
    if ltm_lora_path and learning_enabled:
        print(f"LTM online learning is ACTIVE. Updates will be stored at: {ltm_lora_path}")
        updatable_model = shadow_model if is_quantized else model
        if updatable_model and hasattr(updatable_model.ltm, 'accumulate_deltas'):
            updatable_model.ltm.accumulate_deltas = True
            if os.path.exists(ltm_lora_path):
                print("Loading existing LTM deltas...")
                try:
                    deltas = torch.load(ltm_lora_path, weights_only=True)
                    updatable_model.ltm.vals.data.add_(deltas.to(updatable_model.ltm.vals.device))
                except Exception as e:
                    print(f"Warning: Failed to load LTM deltas: {e}")
    elif learning_enabled:
        print("LTM online learning is ACTIVE for passive prompt memory and explicit feedback/validation.")
        print("Passive response learning is OFF by default; use --passive-response-learning to opt in.")
    trained_ltm_mode = _print_ltm_chat_alignment(config, learning_enabled)

    if not is_quantized:
        model.eval()
    if model is not None:
        model.suppress_hebbian = True
    if shadow_model is not None:
        shadow_model.suppress_hebbian = True

    # LTM Scheduler setup
    ltm_scheduler = None
    static_ltm_lr = getattr(args, 'static_ltm_lr', True)
    ltm_lr = getattr(args, 'ltm_lr', 0.001)
    
    if not static_ltm_lr and learning_enabled:
        print("INFO: Using Cosine Annealing schedule for LTM updates.")
        ltm_schedule_steps = getattr(args, 'ltm_schedule_steps', 100)
        ltm_schedule_min_lr = getattr(args, 'ltm_schedule_min_lr', 1e-5)
        dummy_param = nn.Parameter(torch.tensor(0.0))
        ltm_optimizer = torch.optim.SGD([dummy_param], lr=ltm_lr)
        ltm_scheduler = CosineAnnealingLR(ltm_optimizer, T_max=ltm_schedule_steps, eta_min=ltm_schedule_min_lr)

    # AMP Setup
    use_amp = getattr(args, 'amp', False) and learning_enabled and device.type == 'cuda'
    # BFloat16 does NOT use GradScaler — only float16 needs it (same as trainer.py)
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(config, 'amp_dtype', 'float16')
    scaler = GradScaler() if (use_amp and amp_dtype_str == 'float16') else None
    dummy_optimizer = None
    if use_amp:
        dummy_param_amp = nn.Parameter(torch.tensor(0.0)).to(device)
        dummy_optimizer = torch.optim.SGD([dummy_param_amp], lr=1.0)
        print("INFO: AMP ENABLED for online learning.")

    def _set_online_update_warmup_complete(update_model):
        """Match exported late-training checkpoints when computing LTM gradients."""
        setter = getattr(update_model, "set_training_step", None)
        if callable(setter):
            setter(int(getattr(config, "memory_gate_warmup_steps", 0) or 0))

    def _merge_ltm_state_into_model_state(update_model, state) -> bool:
        """Copy active chat fast-memory state into model.ltm before saving weights."""
        if (
            update_model is None
            or state is None
            or not hasattr(update_model, "ltm")
            or not isinstance(state, (tuple, list))
            or len(state) < 2
        ):
            return False
        ltm = update_model.ltm

        def _copy_tensor_attr(attr_name, value):
            target = getattr(ltm, attr_name, None)
            if not torch.is_tensor(target) or not torch.is_tensor(value):
                return False
            src = value.detach()
            if src.dim() == target.dim() + 1 and src.shape[0] == 1:
                src = src.squeeze(0)
            if tuple(src.shape) != tuple(target.shape):
                return False
            with torch.no_grad():
                target.copy_(src.to(device=target.device, dtype=target.dtype))
            return True

        copied = _copy_tensor_attr("fast_vals", state[0])
        copied = _copy_tensor_attr("_mom_vals", state[1]) or copied
        if len(state) >= 5:
            copied = _copy_tensor_attr("timestamps", state[4]) or copied
        if len(state) >= 6:
            copied = _copy_tensor_attr("sources", state[5]) or copied
        return copied

    # =================================================================
    # 4. LOCAL HELPER FOR LTM UPDATE
    # =================================================================
    def perform_ltm_update(input_ids_tensor, label_ids_tensor, source_id, penalty=False, lr_override=None, silent=False, compute_only=False, learn_input_tokens=False):
        """Performs LTM update. Returns loss value if successful, else None.
        If compute_only=True, only computes loss without updating LTM."""
        nonlocal ltm_has_been_updated, ltm_state
        
        update_model = shadow_model if is_quantized else model
        if update_model is None:
            if not silent:
                print(" (No updatable model available)")
            return None
            
        target_device = device

        # Online feedback should observe the same adaptive inference path used
        # to produce the answer. eval() still permits gradients.
        update_model.eval()
        _set_online_update_warmup_complete(update_model)
        with torch.enable_grad():
            if label_ids_tensor is None:
                full_sequence = input_ids_tensor.unsqueeze(0)
                labels = full_sequence.clone() if learn_input_tokens else torch.full_like(full_sequence, -100)
            else:
                full_sequence = torch.cat([input_ids_tensor, label_ids_tensor], dim=0).unsqueeze(0)
                if learn_input_tokens:
                    labels = full_sequence.clone()
                else:
                    labels = torch.cat([torch.full_like(input_ids_tensor, -100), label_ids_tensor], dim=0).unsqueeze(0)
            
            max_length = getattr(config, 'max_length', 1024)
            if full_sequence.shape[1] > max_length:
                full_sequence = full_sequence[:, -max_length:]
                labels = labels[:, -max_length:]

            if use_amp and dummy_optimizer:
                dummy_optimizer.zero_grad(set_to_none=True)
            update_model.zero_grad(set_to_none=True)

            autocast_device = 'cpu' if is_directml_device(target_device) else target_device.type
            with autocast(device_type=autocast_device, enabled=use_amp):
                outputs = update_model(
                    input_ids=full_sequence,
                    labels=None,
                    # Replay this sample once. Reusing the active ROSA history
                    # here would append the same prompt/answer a second time.
                    ltm_memory_state=ltm_replay_seed_state(ltm_state),
                    suppress_hebbian=True,
                )
                logits = outputs["logits"]
                
                # LTM feedback learning must retain gradients on the exact
                # retrieval tensors consumed by the forward graph. retrieve_topk
                # already runs in float32, matching trainer.py.
                if outputs.get("raw_topk_vals") is not None:
                    for t in outputs["raw_topk_vals"]:
                        if t.requires_grad:
                            t.retain_grad()

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, config.vocab_size)
                flat_labels = shift_labels.view(-1)
                valid_mask = flat_labels != -100
                active_logits = flat_logits[valid_mask]
                active_labels = flat_labels[valid_mask]

                if active_labels.numel() == 0:
                    update_model.zero_grad(set_to_none=True)
                    update_model.eval()
                    if not silent:
                        print(" (No supervised tokens for LTM update)")
                    return None

                if penalty:
                    probs = F.softmax(active_logits, dim=-1)
                    target_probs = torch.gather(probs, 1, active_labels.unsqueeze(1)).squeeze(1)
                    target_probs = torch.clamp(target_probs, min=1e-7, max=1.0 - 1e-7)
                    loss = -torch.log(1.0 - target_probs).mean()
                else:
                    loss = F.cross_entropy(active_logits, active_labels)

            if not bool(torch.isfinite(loss.detach()).all().item()):
                update_model.zero_grad(set_to_none=True)
                if not silent:
                    print(" (Rejected non-finite online-learning loss)")
                return None

            # Surprise/quality probes need only the scalar loss. Avoid a full
            # backward pass when no memory write was requested.
            if compute_only:
                loss_value = float(loss.detach().item())
                update_model.zero_grad(set_to_none=True)
                return loss_value

            if use_amp and scaler:
                scaler.scale(loss + dummy_param_amp * 0.0).backward()
            else:
                loss.backward()

            # Extract and apply LTM gradients
            ltm_grads = None
            if outputs.get("raw_topk_vals") is not None:
                grads_list = [t.grad for t in outputs["raw_topk_vals"]]
                if any(g is not None for g in grads_list):
                    cleaned_grads = []
                    for i, g in enumerate(grads_list):
                        if g is None:
                            cleaned_grads.append(torch.zeros_like(outputs["raw_topk_vals"][i]))
                        else:
                            cleaned_grads.append(g)
                    ltm_grads = torch.stack(cleaned_grads, dim=1)

            if ltm_grads is not None:
                ltm_grads_copy = ltm_grads.detach().clone()
                
                current_ltm_lr = lr_override if lr_override is not None else ltm_lr
                if ltm_scheduler and lr_override is None:
                    current_ltm_lr = ltm_scheduler.get_last_lr()[0]
                    ltm_scheduler.step()

                if use_amp and scaler:
                    current_scale = scaler.get_scale()
                    if current_scale != 1.0:
                        ltm_grads_copy = ltm_grads_copy / current_scale

                ltm_grads_copy = prepare_online_ltm_gradients(
                    ltm_grads_copy,
                    getattr(args, "online_ltm_grad_clip", 0.75),
                )
                if ltm_grads_copy is None:
                    update_model.zero_grad(set_to_none=True)
                    if not silent:
                        print(" (Rejected non-finite online LTM gradient; memory unchanged)")
                    return None

                curr_ltm = outputs.get("ltm_memory_state")
                curr_fast = curr_ltm[0] if curr_ltm is not None else None
                curr_mom = curr_ltm[1] if curr_ltm is not None else None
                old_past_tokens = ltm_state[2] if ltm_state is not None and len(ltm_state) >= 3 else None
                old_rosa_states = ltm_state[3] if ltm_state is not None and len(ltm_state) >= 4 else None
                curr_timestamps = curr_ltm[4] if curr_ltm is not None and len(curr_ltm) >= 5 else None
                curr_sources = curr_ltm[5] if curr_ltm is not None and len(curr_ltm) >= 6 else None

                new_fast, new_mom = update_model.ltm.inner_update(
                    outputs["topk_idx"],
                    ltm_grads_copy,
                    current_lr=current_ltm_lr,
                    timestamp=0.0,
                    source=source_id,
                    tokens_covered=full_sequence.shape[1],
                    fast_vals=curr_fast,
                    mom_vals=curr_mom,
                    timestamps=curr_timestamps,
                    sources=curr_sources,
                    inplace=True
                )
                if curr_ltm is not None:
                    if is_quantized:
                        new_fast = new_fast.detach().cpu()
                        new_mom = new_mom.detach().cpu()
                        if isinstance(curr_timestamps, torch.Tensor):
                            curr_timestamps = curr_timestamps.detach().cpu()
                        if isinstance(curr_sources, torch.Tensor):
                            curr_sources = curr_sources.detach().cpu()
                    ltm_state = (
                        new_fast.detach(),
                        new_mom.detach(),
                        old_past_tokens.detach() if isinstance(old_past_tokens, torch.Tensor) else old_past_tokens,
                        old_rosa_states,
                        curr_timestamps.detach() if isinstance(curr_timestamps, torch.Tensor) else curr_timestamps,
                        curr_sources.detach() if isinstance(curr_sources, torch.Tensor) else curr_sources,
                    )
                ltm_has_been_updated = True

                if use_amp and scaler and dummy_optimizer:
                    scaler.unscale_(dummy_optimizer)
                    scaler.step(dummy_optimizer)
                    scaler.update()

                update_model.zero_grad(set_to_none=True)

                if is_quantized and shadow_model:
                    model.ltm.load_state_dict(update_model.ltm.state_dict())

                update_model.eval()
                if penalty:
                    if not silent:
                        print(f" Done. (Unlikelihood | Loss: {loss.item():.3f})")
                else:
                    if not silent:
                        print(f" Done. (Reinforced | Loss: {loss.item():.3f})")
                return loss.item()
            else:
                update_model.eval()
                if not silent:
                    print(" (No LTM gradients generated)")
                return None

        update_model.eval()
        return None

    def perform_validation_hebbian_update(input_ids_tensor, label_ids_tensor, source_id, lr_override=None, silent=False):
        """Stores a validated exchange in fast LTM using Hebbian writes only after praise/validation."""
        nonlocal ltm_has_been_updated, ltm_state

        if not _checkpoint_has_trained_hebbian_writer(config):
            if not silent:
                print(" (Hebbian validation skipped: checkpoint value writer is untrained)", end="", flush=True)
            return None

        update_model = model
        if update_model is None:
            if not silent:
                print(" (No model available for validation memory)")
            return None

        full_sequence = torch.cat([input_ids_tensor, label_ids_tensor], dim=0).unsqueeze(0)
        max_length = getattr(config, 'max_length', 1024)
        if full_sequence.shape[1] > max_length:
            full_sequence = full_sequence[:, -max_length:]

        model_input = full_sequence.cpu() if is_quantized else full_sequence.to(device)
        rnn_device_local = "cpu" if is_quantized else device
        if is_quantized or not hasattr(model, "h_rnn"):
            local_h = torch.zeros(1, getattr(config, 'h_hidden', config.context_dim), 5, device=rnn_device_local)
            local_l = torch.zeros(1, getattr(config, 'l_hidden', config.context_dim), 5, device=rnn_device_local)
            local_h[:, :, 3] = -1e30
            local_l[:, :, 3] = -1e30
        else:
            local_h = model.h_rnn.initial_state(1, device=rnn_device_local)
            local_l = model.l_rnn.initial_state(1, device=rnn_device_local)
        local_prev = torch.zeros(1, config.context_dim, device=rnn_device_local)
        local_target = torch.zeros(1, config.context_dim, device=rnn_device_local)

        previous_suppress = getattr(update_model, "suppress_hebbian", True)
        previous_lr = getattr(update_model.config, "ltm_lr", None)
        current_ltm_lr = lr_override if lr_override is not None else ltm_lr
        update_model.suppress_hebbian = False
        update_model.config.ltm_lr = current_ltm_lr

        try:
            if hasattr(update_model, "eval"):
                update_model.eval()
            with torch.no_grad():
                outputs = update_model(
                    input_ids=model_input,
                    h_state=local_h if is_quantized else None,
                    l_state=local_l if is_quantized else None,
                    prev_context=local_prev if is_quantized else None,
                    target_context=local_target if is_quantized else None,
                    ltm_memory_state=ltm_state,
                    global_pos_offset=0,
                    min_timestamp=min_ts_filter,
                    source_filter=source_id_filter,
                    allow_hebbian_update=True,
                )

            updated_ltm = outputs.get("ltm_memory_state")
            if updated_ltm is not None:
                old_past_tokens = ltm_state[2] if ltm_state is not None and len(ltm_state) >= 3 else None
                old_rosa_states = ltm_state[3] if ltm_state is not None and len(ltm_state) >= 4 else None
                updated_mom = torch.zeros_like(updated_ltm[1]) if isinstance(updated_ltm[1], torch.Tensor) else updated_ltm[1]
                ltm_state = (
                    updated_ltm[0],
                    updated_mom,
                    old_past_tokens,
                    old_rosa_states,
                    updated_ltm[4] if len(updated_ltm) >= 5 else None,
                    updated_ltm[5] if len(updated_ltm) >= 6 else None,
                )
                ltm_has_been_updated = True
                if not silent:
                    fv_norm = ltm_state[0].float().norm().item()
                    print(f" [Hebbian validation memory | fast_vals norm: {fv_norm:.6e}]", end="", flush=True)
                return ltm_state[0].float().norm().item() if updated_ltm[0] is not None else 0.0
            if not silent:
                print(" (No LTM state returned for validation memory)", end="", flush=True)
            return None
        finally:
            update_model.suppress_hebbian = previous_suppress
            if previous_lr is not None:
                update_model.config.ltm_lr = previous_lr

    # =================================================================
    # 5. PRINT WELCOME MESSAGE
    # =================================================================
    print("\nWelcome to Hierarchos Chat. Type 'exit' or 'quit' to end.")
    print("Commands:")
    print("  /filter time=-<seconds> | /filter source=<id>  : Constrain memory retrieval")
    print("  /settings [temperature <float>] | /temp <float> : View/Change temperature")
    print("  /topk <int> | /topp <float>                    : Change sampling filters")
    print("  /system <prompt> | /system clear               : Set/clear system prompt")
    print("  /reset                                         : Clear RNN & Hierarchical states")
    print("  /reset_ltm                                     : Clear LTM memory (fast_vals)")
    print("  /status                                        : Show model state info")
    print("Press Ctrl+C to stop generation at any time.")
    print("=" * 50)

    resume_chat_state_path = getattr(args, "resume_chat_from_state_file", None)
    requested_chat_state_path = resume_chat_state_path or getattr(args, "chat_state_file", None)
    if requested_chat_state_path == "auto":
        chat_state_path = _default_chat_state_path(args.model_path)
    elif requested_chat_state_path:
        chat_state_path = _normalize_chat_state_path(requested_chat_state_path)
    else:
        chat_state_path = None

    if chat_state_path:
        print(f"Chat hierarchical state file: {chat_state_path}")
    else:
        print("Chat hierarchical state autosave: disabled (you can save on exit)")

    h_state = None
    l_state = None
    prev_context = None
    target_context = None
    drift_state = None
    total_tokens_generated = 0

    def save_chat_state_to(path, silent=True):
        if not path:
            return
        if h_state is None or l_state is None:
            return
        try:
            save_hierarchical_chat_state(
                path,
                config=config,
                model=model,
                model_path=args.model_path,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                ltm_state=ltm_state,
                total_tokens_generated=total_tokens_generated,
            )
            if not silent:
                print(f"Saved hierarchical chat state to {path}")
        except Exception as exc:
            print(f"WARNING: Could not save hierarchical chat state: {exc}")

    def autosave_chat_state(silent=True):
        save_chat_state_to(chat_state_path, silent=silent)

    def prompt_save_chat_state_on_exit():
        if h_state is None or l_state is None:
            return

        if chat_state_path:
            save_chat_state_to(chat_state_path, silent=False)
            return

        suggested_path = _default_chat_state_path(args.model_path)
        while True:
            try:
                response = input(
                    f"Do you want to save the hierarchical chat state to '{suggested_path}'? (y/n): "
                ).lower()
                if response in ["y", "yes"]:
                    save_chat_state_to(suggested_path, silent=False)
                    break
                if response in ["n", "no"]:
                    print("Hierarchical chat state discarded. Exiting.")
                    break
                print("Invalid input.")
            except EOFError:
                print("\nEOF detected. Assuming 'no' for hierarchical state saving.")
                break
            except KeyboardInterrupt:
                print("\nInterrupted. Hierarchical chat state will be discarded.")
                break

    try:
        min_ts_filter = 0.0
        source_id_filter = None
        system_prompt = None  # Optional system prompt prepended inside the active training format
        alpaca_chat_format = bool(getattr(args, "alpaca", False) or getattr(config, "alpaca", False))
        chat_input_history_turns = int(getattr(args, "chat_input_history_turns", 4) or 0)
        chat_input_history_chars = int(getattr(args, "chat_input_history_chars", 3000) or 0)
        carry_chat_state = bool(getattr(args, "carry_chat_state", False))
        requested_prefill_chunk = getattr(args, "chat_prefill_chunk_size", None)
        chat_prefill_chunk_size = resolve_inference_prefill_chunk_size(
            config,
            requested_prefill_chunk,
        )
        exact_inference_recurrence = uses_full_sample_inference_recurrence(config)
        chat_turn_history = []
        if alpaca_chat_format:
            if chat_input_history_turns > 0 and chat_input_history_chars > 0:
                print(
                    "Chat prompt format: Alpaca ### Previous Context/Instruction/Response "
                    f"(Previous Context uses last {chat_input_history_turns} turn(s), capped at {chat_input_history_chars} chars)"
                )
            else:
                print("Chat prompt format: Alpaca ### Instruction/Response")
        else:
            print("Chat prompt format: User/Assistant")
        if carry_chat_state:
            print("Chat state carry: ON (experimental; recurrent state persists across user turns).")
        else:
            print("Chat state carry: OFF (train-parity mode; use Previous Context text for turn history).")
        if chat_prefill_chunk_size > 0:
            if exact_inference_recurrence:
                print(
                    f"Chat prefill chunking: {chat_prefill_chunk_size} tokens "
                    "(exact full-sample recurrence; activation boundary only)."
                )
            else:
                print(f"Chat prefill chunking: {chat_prefill_chunk_size} tokens (TBPTT train-parity mode).")
        else:
            print("Chat prefill chunking: OFF (single full prompt forward).")
        print(
            "Chat decoding: "
            f"temperature={float(getattr(args, 'temperature', 1.0)):.2f}, "
            f"top-k={int(getattr(args, 'top_k', 0) or 0)}, "
            f"top-p={float(getattr(args, 'top_p', 1.0)):.2f}, "
            f"repetition-penalty={float(getattr(args, 'repetition_penalty', 1.0)):.2f}"
        )
        if (
            float(getattr(args, "temperature", 1.0)) > 0.0
            or float(getattr(args, "repetition_penalty", 1.0)) != 1.0
        ):
            print(
                "Chat decoding transforms/samples the raw model logits. For deterministic "
                "raw-logit decoding use --temperature 0 --top-k 0 --top-p 1 "
                "--repetition-penalty 1."
            )

        # =================================================================
        # 6. STATE INITIALIZATION
        # =================================================================
        rnn_device = "cpu" if is_quantized else device
        h_hidden = getattr(config, 'h_hidden', config.context_dim)
        l_hidden = getattr(config, 'l_hidden', config.context_dim)
        context_dim = config.context_dim

        def _new_h_state():
            if not is_quantized and hasattr(model, "h_rnn"):
                return model.h_rnn.initial_state(1, device=rnn_device)
            state = torch.zeros(1, h_hidden, 5, device=rnn_device)
            state[:, :, 3] = -1e30
            return state

        def _new_l_state():
            if not is_quantized and hasattr(model, "l_rnn"):
                return model.l_rnn.initial_state(1, device=rnn_device)
            state = torch.zeros(1, l_hidden, 5, device=rnn_device)
            state[:, :, 3] = -1e30
            return state

        h_state = _new_h_state()
        l_state = _new_l_state()
        
        prev_context = torch.zeros(1, context_dim, device=rnn_device)
        target_context = torch.zeros(1, context_dim, device=rnn_device)
        drift_state = torch.zeros(1, context_dim, device=rnn_device)
        ltm_state = None  # Will hold (fast_vals, mom_vals, past_tokens, rosa_states)
        
        total_tokens_generated = 0

        if resume_chat_state_path:
            try:
                restored = load_hierarchical_chat_state(
                    chat_state_path,
                    config=config,
                    device=rnn_device,
                    model=model,
                )
                if restored["h_state"] is not None:
                    h_state = restored["h_state"]
                if restored["l_state"] is not None:
                    l_state = restored["l_state"]
                if restored["prev_context"] is not None:
                    prev_context = restored["prev_context"]
                if restored["target_context"] is not None:
                    target_context = restored["target_context"]
                if restored["drift_state"] is not None:
                    drift_state = restored["drift_state"]
                total_tokens_generated = restored["total_tokens_generated"]
                ltm_state = _chat_ltm_state_from_rosa_context(
                    model,
                    restored.get("rosa_past_tokens"),
                    restored.get("rosa_states"),
                )
                print(f"Resumed hierarchical chat state from {chat_state_path}")
            except FileNotFoundError:
                print(f"No state file found yet; a new one will be created at {chat_state_path}")
            except Exception as exc:
                print(f"WARNING: Could not resume chat state: {exc}")

        # --- Startup Diagnostic: Verify model produces reasonable predictions ---
        print("INFO: Running inference diagnostic...")
        try:
            _diag_prompt = wrap_for_hierarchos(
                "Hello",
                system_prompt=None,
                alpaca_mode=alpaca_chat_format,
            )
            _diag_ids = tokenizer.encode(_diag_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _diag_out = model(
                    _diag_ids if not is_quantized else _diag_ids.cpu(),
                    h_state=None, l_state=None,
                    prev_context=None, target_context=None,
                )
                _diag_logits = _diag_out["logits"][:, -1, :]
                _diag_probs = torch.softmax(_diag_logits.float(), dim=-1)
                _diag_topk = torch.topk(_diag_probs, 5)
                print(f"  Prompt: {repr(_diag_prompt)}")
                print(f"  Top-5 next tokens:")
                for i in range(5):
                    _tok = tokenizer.decode([_diag_topk.indices[0, i].item()])
                    _prob = _diag_topk.values[0, i].item()
                    print(f"    {i+1}. {repr(_tok):>15s}  ({_prob:.4f})")
                _entropy = -((_diag_probs * torch.log(_diag_probs + 1e-10)).sum(-1)).item()
                print(f"  Logit entropy: {_entropy:.2f} (random={math.log(config.vocab_size):.2f})")
                if _entropy > math.log(config.vocab_size) * 0.8:
                    print("  WARNING: High entropy; model weights or runtime configuration may be invalid.")
                else:
                    print(
                        "  OK: Load smoke test passed. This fixed easy prompt does not "
                        "measure response quality or calibration."
                    )
        except Exception as e:
            print(f"  Diagnostic failed: {e}")

        # =================================================================
        # 7. MAIN CHAT LOOP
        # =================================================================
        while True:
            _interrupt_flag = False
            try:
                prompt = input(">>> ")
            except EOFError:
                print("\n[EOF detected. Exiting chat.]")
                break

            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit"]:
                break

            # --- Slash Commands ---
            if prompt.startswith('/filter'):
                parts = prompt.split()
                if len(parts) >= 2:
                    for part in parts[1:]:
                        if part.startswith("time="):
                            try:
                                min_ts_filter = float(part.split("=")[1])
                                print(f"Set time filter to {min_ts_filter}")
                            except:
                                print("Usage: /filter time=<float>")
                        elif part.startswith("source="):
                            try:
                                source_id_filter = int(part.split("=")[1])
                                print(f"Set source filter to {source_id_filter}")
                            except:
                                print("Usage: /filter source=<int>")
                continue

            if prompt.startswith('/temp') or prompt.startswith('/temperature'):
                try:
                    val = parse_temperature_setting(prompt.split()[1])
                    args.temperature = val
                    print(f"Set temperature to {args.temperature:.2f}")
                except (IndexError, ValueError):
                    print("Usage: /temp <0.00-1.00 in 0.05 increments>")
                continue

            if prompt.startswith('/topk'):
                try:
                    val = int(prompt.split()[1])
                    args.top_k = max(0, val)
                    print(f"Set top_k to {args.top_k}")
                except (IndexError, ValueError):
                    print("Usage: /topk <int>")
                continue

            if prompt.startswith('/topp'):
                try:
                    val = float(prompt.split()[1])
                    args.top_p = max(0.0, min(1.0, val))
                    print(f"Set top_p to {args.top_p}")
                except (IndexError, ValueError):
                    print("Usage: /topp <float>")
                continue

            if prompt.startswith('/settings'):
                parts = prompt.split()
                if len(parts) >= 3 and parts[1].lower() in ("temperature", "temp"):
                    try:
                        val = parse_temperature_setting(parts[2])
                        args.temperature = val
                        print(f"Set temperature to {args.temperature:.2f}")
                    except ValueError:
                        print("Usage: /settings temperature <0.00-1.00 in 0.05 increments>")
                    continue
                if len(parts) > 1:
                    print("Usage: /settings [temperature <0.00-1.00 in 0.05 increments>]")
                    continue
                print(
                    f"Current Settings:\n  Temperature: {args.temperature:.2f}\n  Top-K: {args.top_k}\n"
                    f"  Top-P: {args.top_p}\n  Carry Chat State: {carry_chat_state}\n"
                    f"  Prefill Chunk Size: {chat_prefill_chunk_size or 'off'}"
                )
                if float(getattr(args, "entropy_stop_threshold", 0.0) or 0.0) > 0:
                    print(
                        f"  Entropy Stop: >= {args.entropy_stop_threshold:.2f} "
                        f"after {getattr(args, 'entropy_stop_min_tokens', 3)} tokens"
                    )
                print(f"  System Prompt: {repr(system_prompt) if system_prompt else '(none)'}")
                continue

            if prompt.startswith('/system'):
                rest = prompt[len('/system'):].strip()
                if not rest or rest.lower() == 'clear':
                    system_prompt = None
                    print("System prompt cleared.")
                else:
                    system_prompt = rest
                    print(f"System prompt set to: {repr(system_prompt)}")
                continue

            if prompt.startswith('/reset_ltm'):
                print("Resetting LTM memory...")
                ltm_state = reset_active_ltm_state(model, ltm_state, preserve_rosa=True)
                if is_quantized and shadow_model:
                    reset_active_ltm_state(shadow_model, None, preserve_rosa=True)
                ltm_has_been_updated = True
                print("LTM Reset complete.")
                continue

            if prompt.startswith('/reset'):
                print("Resetting all RNN and Hierarchical states...")
                h_state = _new_h_state()
                l_state = _new_l_state()
                prev_context.zero_()
                target_context.zero_()
                drift_state.zero_()
                ltm_state = None  # Reset ROSA automaton states too
                total_tokens_generated = 0
                chat_turn_history.clear()
                autosave_chat_state()
                print("Hierarchical state reset complete. LTM memory was left unchanged.")
                continue

            if prompt.startswith('/status'):
                print(f"Model Status:")
                print(f"  Total Tokens Generated: {total_tokens_generated}")
                print(f"  Device: {device}")
                print(f"  Quantized: {is_quantized}")
                print(f"  LTM Learning: {'ACTIVE' if learning_enabled else 'OFF'}")
                print(f"  Checkpoint LTM Training Mode: {trained_ltm_mode}")
                print(f"  Chat LTM Runtime: read-only during normal prefill/generation")
                print(f"  Chat Prefill Chunk Size: {chat_prefill_chunk_size or 'off'}")
                print(f"  Chat State File: {chat_state_path or '(disabled)'}")
                continue

            # =================================================================
            # A. CHECK FOR FEEDBACK & PERFORM UPDATES
            # =================================================================
            if learning_enabled:
                correction_text = extract_correction_text(prompt) if pending_training_data is not None else None
                if correction_text:
                    print("[Correction received. Updating previous answer memory...]", end="", flush=True)
                    perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_CORRECTION,
                        penalty=True
                    )
                    correction_ids = tokenizer.encode(correction_text, add_special_tokens=False)
                    if tokenizer.eos_token_id is not None:
                        correction_ids = correction_ids + [tokenizer.eos_token_id]
                    correction_tensor = torch.tensor(correction_ids, device=device)
                    perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        correction_tensor,
                        LTMModule.SRC_CORRECTION,
                        penalty=False
                    )
                    pending_training_data = None
                    print("")
                    continue

                if is_positive_feedback(prompt) and pending_training_data is not None:
                    print("[Positive feedback. Reinforcing previous memory...]", end="", flush=True)
                    perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_USER_INTERACTION,
                        penalty=False
                    )
                    perform_validation_hebbian_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_USER_INTERACTION,
                    )
                    pending_training_data = None
                    print("")
                    continue

                elif prompt.strip().lower() in ["no", "n", "bad", "wrong", "bad bot"]:
                    if pending_training_data is not None:
                        print("[Negative feedback. Minimizing probability of previous output...]", end="", flush=True)
                        perform_ltm_update(
                            pending_training_data['prompt_ids'][0],
                            pending_training_data['response_ids'],
                            LTMModule.SRC_USER_INTERACTION,
                            penalty=True
                        )

            if prompt.strip() == "/learn" and pending_training_data:
                print("[Manual learn command. Reinforcing previous...]", end="", flush=True)
                perform_ltm_update(
                    pending_training_data['prompt_ids'][0],
                    pending_training_data['response_ids'],
                    LTMModule.SRC_USER_INTERACTION,
                    penalty=False
                )
                perform_validation_hebbian_update(
                    pending_training_data['prompt_ids'][0],
                    pending_training_data['response_ids'],
                    LTMModule.SRC_USER_INTERACTION,
                )
                print("")
                continue
            elif prompt.strip() == "/learn":
                print("[Nothing pending to learn]")
                continue

            # =================================================================
            # B. GENERATION LOGIC
            # =================================================================
            if not carry_chat_state:
                # Alpaca SFT samples are independent. For chat/train parity, each
                # user turn should prefill from fresh recurrent/context/ROSA state;
                # conversation memory enters through the formatted Previous Context
                # text instead of hidden state carried from the previous answer.
                h_state = _new_h_state()
                l_state = _new_l_state()
                prev_context = torch.zeros(1, context_dim, device=rnn_device)
                target_context = torch.zeros(1, context_dim, device=rnn_device)
                drift_state = torch.zeros(1, context_dim, device=rnn_device)
                ltm_state = None
                total_tokens_generated = 0

            prompt_format = wrap_for_hierarchos(
                prompt,
                system_prompt=system_prompt,
                alpaca_mode=alpaca_chat_format,
                input_context=build_chat_input_context(
                    chat_turn_history,
                    max_turns=chat_input_history_turns,
                    max_chars=chat_input_history_chars,
                ) if alpaca_chat_format else None,
            )
            prompt_ids = tokenizer.encode(prompt_format, return_tensors="pt").to(device)
            passive_learning = getattr(args, 'passive_learning', False)
            passive_response_learning = getattr(args, 'passive_response_learning', False)
            passive_lr = getattr(args, 'passive_lr', 1e-5)
            surprise_threshold = getattr(args, 'surprise_threshold', 0.5)

            print("\nhierarchos: ", end="", flush=True)
            response_ids = []
            _display_buffer = ""  # Buffer last 2 chars to catch trailing JSON close

            # 1. PREFILL PASS
            # Keep unsupervised Hebbian writes off by default. The model trained
            # around gradient-derived LTM updates from raw_topk_vals, so normal
            # chat learning should happen through explicit feedback/validation.
            model.suppress_hebbian = True
            with torch.no_grad():
                prefill_len = int(prompt_ids.shape[1])
                outputs = None
                prefill_ranges = tbptt_chunk_ranges(
                    prefill_len,
                    chat_prefill_chunk_size,
                    total_tokens_generated,
                )
                for prefill_start, prefill_end in prefill_ranges:
                    absolute_start = total_tokens_generated + prefill_start
                    outputs, runtime_state = advance_chat_model_state(
                        model,
                        prompt_ids[:, prefill_start:prefill_end],
                        device=device,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        drift_seed=boundary_drift_seed(
                            drift_state,
                            absolute_start,
                            chat_prefill_chunk_size,
                            exact_full_sample=exact_inference_recurrence,
                        ),
                        ltm_state=ltm_state,
                        global_pos_offset=absolute_start,
                        min_timestamp=min_ts_filter,
                        source_filter=source_id_filter,
                        is_quantized=is_quantized,
                        inference_device=inference_device,
                    )
                    (
                        h_state,
                        l_state,
                        prev_context,
                        target_context,
                        drift_state,
                        ltm_state,
                    ) = runtime_state

                logits = outputs["logits"].to(device)
                next_token_logits = logits[:, -1, :]
                total_tokens_generated += prompt_ids.shape[1]

                max_new_tokens = max(0, int(getattr(args, 'max_new_tokens', 512) or 0))
                if max_new_tokens > 0:
                    next_token_id = sample_next_token(
                        next_token_logits,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=getattr(args, 'repetition_penalty', 1.2),
                        previous_tokens=response_ids,
                    )
                else:
                    next_token_id = None

                pending_state_token = False
                if next_token_id is not None and next_token_id.item() != tokenizer.eos_token_id:
                    response_ids.append(next_token_id.item())
                    decoded_token = tokenizer.decode([next_token_id.item()])
                    # Buffer output to catch JSON closing syntax
                    _display_buffer += decoded_token
                    if len(_display_buffer) > 2:
                        _flush = _display_buffer[:-2]
                        print(_flush, end="", flush=True)
                        _display_buffer = _display_buffer[-2:]
                    current_ids = next_token_id
                    pending_state_token = True
                else:
                    current_ids = None

            # 2. INCREMENTAL GENERATION LOOP
            # CRITICAL: Suppress Hebbian LTM updates during autoregressive
            # generation. Each generated token was triggering momentum-amplified
            # memory updates that compound exponentially, causing the latent
            # space to bleed (gibberish output after ~10-15 tokens).
            model.suppress_hebbian = True
            if current_ids is not None:
                with torch.no_grad():
                    for i in range(max_new_tokens - 1):
                        if _interrupt_flag:
                            _interrupt_flag = False
                            print("\n[Generation interrupted by user.]", end="", flush=True)
                            break

                        outputs, runtime_state = advance_chat_model_state(
                            model,
                            current_ids,
                            device=device,
                            h_state=h_state,
                            l_state=l_state,
                            prev_context=prev_context,
                            target_context=target_context,
                            drift_state=drift_state,
                            drift_seed=boundary_drift_seed(
                                drift_state,
                                total_tokens_generated,
                                chat_prefill_chunk_size,
                                exact_full_sample=exact_inference_recurrence,
                            ),
                            ltm_state=ltm_state,
                            global_pos_offset=total_tokens_generated,
                            min_timestamp=min_ts_filter,
                            source_filter=source_id_filter,
                            is_quantized=is_quantized,
                            inference_device=inference_device,
                        )
                        (
                            h_state,
                            l_state,
                            prev_context,
                            target_context,
                            drift_state,
                            ltm_state,
                        ) = runtime_state
                        pending_state_token = False

                        logits = outputs["logits"].to(device)
                        total_tokens_generated += current_ids.shape[-1]
                        next_token_logits = logits[:, -1, :]

                        if should_stop_generation_from_uncertainty(next_token_logits, response_ids, tokenizer, args):
                            break

                        next_token_id = sample_next_token(
                            next_token_logits,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            repetition_penalty=getattr(args, 'repetition_penalty', 1.2),
                            previous_tokens=response_ids,
                        )

                        if next_token_id.item() == tokenizer.eos_token_id:
                            break

                        response_ids.append(next_token_id.item())
                        try:
                            decoded_token = tokenizer.decode([next_token_id.item()])
                        except Exception as e:
                            decoded_token = ""

                        if "###" in decoded_token and len(decoded_token) <= 5:
                            current_ids = None
                            break

                        # Buffer output to catch JSON closing syntax
                        _display_buffer += decoded_token
                        if len(_display_buffer) > 2:
                            _flush = _display_buffer[:-2]
                            print(_flush, end="", flush=True)
                            _display_buffer = _display_buffer[-2:]
                        current_ids = next_token_id
                        pending_state_token = True

            # The sampled token that reaches max_new_tokens has not yet been
            # consumed by the recurrent model. Flush it once so carried/saved
            # state and ROSA history describe every token already shown.
            if pending_state_token and current_ids is not None:
                with torch.no_grad():
                    _, runtime_state = advance_chat_model_state(
                        model,
                        current_ids,
                        device=device,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        drift_seed=boundary_drift_seed(
                            drift_state,
                            total_tokens_generated,
                            chat_prefill_chunk_size,
                            exact_full_sample=exact_inference_recurrence,
                        ),
                        ltm_state=ltm_state,
                        global_pos_offset=total_tokens_generated,
                        min_timestamp=min_ts_filter,
                        source_filter=source_id_filter,
                        is_quantized=is_quantized,
                        inference_device=inference_device,
                    )
                    (
                        h_state,
                        l_state,
                        prev_context,
                        target_context,
                        drift_state,
                        ltm_state,
                    ) = runtime_state
                    total_tokens_generated += current_ids.shape[-1]
                    pending_state_token = False

            # Leave Hebbian writes suppressed between turns. They are opened
            # only inside the praise/validation feedback path above.
            model.suppress_hebbian = True

            # Flush display buffer, stripping JSON closing syntax if present
            if _display_buffer:
                if _display_buffer.endswith('"}'):
                    _display_buffer = _display_buffer[:-2]
                elif _display_buffer.endswith('"'):
                    _display_buffer = _display_buffer[:-1]
                if _display_buffer:
                    print(_display_buffer, end="", flush=True)
            print("\n")

            # =================================================================
            # C. BUFFER DATA FOR NEXT TURN
            # =================================================================
            if passive_learning and learning_enabled:
                perform_ltm_update(
                    prompt_ids[0],
                    None,
                    LTMModule.SRC_USER_INTERACTION,
                    lr_override=passive_lr,
                    silent=True,
                    learn_input_tokens=True,
                )

            if len(response_ids) > 0:
                response_text = clean_hierarchos_output(
                    tokenizer.decode(response_ids, skip_special_tokens=True)
                )
                if response_text:
                    chat_turn_history.append(f"User: {prompt.strip()}\nAssistant: {response_text}")
                    max_keep_turns = max(chat_input_history_turns, 1)
                    if len(chat_turn_history) > max_keep_turns:
                        del chat_turn_history[:-max_keep_turns]
                pending_training_data = {
                    'prompt_ids': prompt_ids,
                    'response_ids': torch.tensor(response_ids, device=device)
                }
                
                # =================================================================
                # D. PASSIVE LEARNING (if enabled)
                # =================================================================
                if passive_response_learning and learning_enabled:
                    # First compute loss WITHOUT updating to check surprise threshold
                    loss_val = perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_TRAINING_DATA,
                        penalty=False,
                        lr_override=passive_lr,
                        silent=True,
                        compute_only=True  # Only compute, don't update yet
                    )
                    
                    quality_ok, quality_reason = passive_response_quality(response_ids)

                    # Only learn self-generated text when it is low-surprise and non-degenerate.
                    if loss_val is not None and loss_val <= surprise_threshold and quality_ok:
                        perform_ltm_update(
                            pending_training_data['prompt_ids'][0],
                            pending_training_data['response_ids'],
                            LTMModule.SRC_TRAINING_DATA,
                            penalty=False,
                            lr_override=passive_lr,
                            silent=True,
                            compute_only=False  # Actually update
                        )
                        print(f"[Passive response LTM update | Loss: {loss_val:.3f} <= {surprise_threshold:.2f}]")
                    elif loss_val is not None:
                        if loss_val > surprise_threshold:
                            print(f"[Passive response LTM skipped | Loss: {loss_val:.3f} > {surprise_threshold:.2f}]")
                        else:
                            print(f"[Passive response LTM skipped | {quality_reason} | Loss: {loss_val:.3f}]")

            # --- LTM State Health: Reset momentum and log norms ---
            # Momentum is an optimizer-like transient. Do not carry it across
            # conversation turns, even when fast_vals are intentionally retained.
            if ltm_state is not None and len(ltm_state) >= 2:
                ltm_state = zero_ltm_momentum_state(model, ltm_state)
                if shadow_model is not None:
                    zero_ltm_momentum_state(shadow_model, None)
                fv_norm = ltm_state[0].float().norm().item()
                print(f"[LTM State | fast_vals norm: {fv_norm:.6e} | momentum reset]")

            autosave_chat_state()

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected. Exiting chat.]")

    finally:
        # Restore original signal handler
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)

        # =================================================================
        # 8. SAVE ON EXIT LOGIC
        # =================================================================
        MODEL_WEIGHTS_NAME = "hierarchos.pt"
        updatable_model = shadow_model if is_quantized and enable_quantized_learning else model
        can_update = updatable_model is not None and learning_enabled

        if can_update and ltm_lora_path and hasattr(updatable_model.ltm, 'accumulate_deltas') and updatable_model.ltm.accumulate_deltas:
            if hasattr(updatable_model.ltm, 'ltm_deltas') and torch.any(updatable_model.ltm.ltm_deltas != 0):
                print(f"\nSaving LTM memory deltas to {ltm_lora_path}...")
                try:
                    torch.save(updatable_model.ltm.ltm_deltas.cpu(), ltm_lora_path)
                    print("Deltas saved.")
                except Exception as e:
                    print(f"Error saving LTM deltas: {e}")
            else:
                print("\nNo new LTM updates to save as LoRA.")

        elif can_update and not ltm_lora_path and ltm_has_been_updated:
            if not is_quantized:
                while True:
                    try:
                        response = input(f"Do you want to save the learned LTM updates back to '{args.model_path}'? (y/n): ").lower()
                        if response in ["y", "yes"]:
                            print(f"\nSaving updated model to {args.model_path}...")
                            output_weights_path = (
                                os.path.join(args.model_path, MODEL_WEIGHTS_NAME)
                                if os.path.isdir(args.model_path)
                                else args.model_path
                            )
                            try:
                                if _merge_ltm_state_into_model_state(model, ltm_state):
                                    print("Merged active chat LTM state into model weights.")
                                if consolidate_ltm_state_for_save(model, ltm_state):
                                    print("Consolidated fast LTM memory into persistent slow values.")
                                save_checkpoint_safely({
                                    'model_state_dict': sanitize_model_state_dict(model),
                                    'config': dict(model.config),
                                    'training_complete': True,
                                }, output_weights_path)
                                print("Save complete.")
                            except Exception as e:
                                print(f"Error saving model: {e}")
                            break
                        elif response in ["n", "no"]:
                            print("LTM changes will be discarded.")
                            break
                        else:
                            print("Invalid input.")
                    except EOFError:
                        print("\nEOF detected. Assuming 'no' for saving.")
                        break
                    except KeyboardInterrupt:
                        print("\nInterrupted. Changes will be discarded. Exiting.")
                        break

        elif ltm_has_been_updated:
            print("\n[Warning] LTM was updated, but no valid save configuration was found. Changes lost.")

        prompt_save_chat_state_on_exit()
