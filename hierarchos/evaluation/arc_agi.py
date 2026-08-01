"""
Local ARC-AGI evaluator for ARC-style JSON task files.

This is a practical public-data runner, not a replacement for the official
private ARC Prize evaluation service.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..inference.chat import (
    boundary_drift_seed,
    resolve_inference_prefill_chunk_size,
    uses_full_sample_inference_recurrence,
    wrap_for_hierarchos,
)
from ..utils.tokenizer import validate_inference_tokenizer_identity


Grid = List[List[int]]


@dataclass
class ArcAgiExample:
    input: Grid
    output: Grid


@dataclass
class ArcAgiTask:
    task_id: str
    train: List[ArcAgiExample]
    test: List[ArcAgiExample]


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    width = None
    for row in value:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        if len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False
    return True


def _parse_examples(items: Sequence[Dict[str, Any]]) -> List[ArcAgiExample]:
    examples = []
    for item in items:
        inp = item.get("input")
        out = item.get("output")
        if _is_grid(inp) and _is_grid(out):
            examples.append(ArcAgiExample(input=inp, output=out))
    return examples


def load_arc_agi_tasks(path: str, max_tasks: Optional[int] = None) -> List[ArcAgiTask]:
    """Load ARC-AGI JSON files from a file or directory tree."""
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"ARC-AGI path not found: {path}")

    files: List[str] = []
    if os.path.isfile(resolved):
        files = [resolved]
    else:
        for root, _, names in os.walk(resolved):
            for name in names:
                if name.lower().endswith(".json"):
                    files.append(os.path.join(root, name))
        files.sort()

    tasks: List[ArcAgiTask] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        train = _parse_examples(data.get("train", []))
        test = _parse_examples(data.get("test", []))
        if not train or not test:
            continue
        task_id = os.path.splitext(os.path.basename(file_path))[0]
        tasks.append(ArcAgiTask(task_id=task_id, train=train, test=test))
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    return tasks


def _grid_json(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def build_arc_agi_prompt(task: ArcAgiTask, test_input: Grid) -> str:
    lines = [
        "You are solving an ARC-AGI grid transformation task.",
        "Each grid is a JSON list of rows. Each cell is an integer 0 through 9.",
        "Infer the transformation from the training pairs.",
        "Return only the predicted output grid as valid compact JSON.",
        "",
        "Training pairs:",
    ]
    for idx, example in enumerate(task.train, 1):
        lines.append(f"Example {idx} input: {_grid_json(example.input)}")
        lines.append(f"Example {idx} output: {_grid_json(example.output)}")
    lines.extend(["", f"Test input: {_grid_json(test_input)}", "Output grid:"])
    return "\n".join(lines)


def extract_json_grid(text: str) -> Optional[Grid]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if _is_grid(value):
            return value
    return None


def _model_max_length(model, tokenizer, max_new_tokens: int) -> int:
    model_config = getattr(model, "config", {})
    max_length = getattr(model_config, "max_length", None)
    if max_length is None and isinstance(model_config, dict):
        max_length = model_config.get("max_length")
    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", 1024)
    if not isinstance(max_length, int) or max_length <= 0 or max_length > 100000:
        max_length = 1024
    return max(max_new_tokens + 1, max_length)


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    context = tokenizer.encode(prompt, add_special_tokens=False)
    max_length = _model_max_length(model, tokenizer, max_new_tokens)
    if len(context) > max_length - max_new_tokens:
        context = context[-(max_length - max_new_tokens):]
    if not context:
        eos = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "pad_token_id", None) or 0
        context = [int(eos)]

    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    generated: List[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    h_state = None
    l_state = None
    prev_context = None
    target_context = None
    drift_state = None
    ltm_state = None
    model_config = getattr(model, "config", None)
    prefill_chunk_size = resolve_inference_prefill_chunk_size(model_config)
    exact_full_sample = uses_full_sample_inference_recurrence(model_config)
    total_tokens_seen = 0

    model.eval()
    with torch.inference_mode():
        prefill_step = prefill_chunk_size if prefill_chunk_size > 0 else input_ids.shape[1]
        prefill_step = max(1, int(prefill_step))
        chunk_drift_state = None
        outputs = None
        for start in range(0, input_ids.shape[1], prefill_step):
            end = min(start + prefill_step, input_ids.shape[1])
            outputs = model(
                input_ids=input_ids[:, start:end],
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=chunk_drift_state,
                ltm_memory_state=ltm_state,
                global_pos_offset=start,
                suppress_hebbian=True,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
                return_last_logit_only=True,
            )
            h_state = outputs.get("h_state")
            l_state = outputs.get("l_state")
            prev_context = outputs.get("prev_context")
            target_context = outputs.get("target_context")
            drift_state = outputs.get("drift_state")
            ltm_state = outputs.get("ltm_memory_state")
            chunk_drift_state = boundary_drift_seed(
                drift_state,
                end,
                prefill_chunk_size,
                exact_full_sample=exact_full_sample,
            )
        total_tokens_seen = len(context)
        logits = outputs["logits"][0, -1, :]

        for step in range(max_new_tokens):
            if temperature <= 0 or temperature < 1e-4:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            token_id = int(next_token.item())
            generated.append(token_id)
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            # ARC answers are self-delimiting JSON. Once a complete valid grid
            # exists, later autoregressive tokens cannot improve the parsed
            # answer and only spend benchmark compute.
            if len(generated) >= 3:
                partial_text = tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                )
                if extract_json_grid(partial_text) is not None:
                    break

            next_input = next_token.reshape(1, 1).to(device)
            generation_drift_state = boundary_drift_seed(
                drift_state,
                total_tokens_seen,
                prefill_chunk_size,
                exact_full_sample=exact_full_sample,
            )
            outputs = model(
                input_ids=next_input,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                # Epoch-13 TBPTT parity: drift is fed only at chunk boundaries.
                drift_state=generation_drift_state,
                ltm_memory_state=ltm_state,
                global_pos_offset=total_tokens_seen,
                suppress_hebbian=True,
                return_topk_values=False,
                return_raw_topk_values=False,
                return_topk_indices=False,
                return_step_telemetry=False,
                return_numerics=False,
                return_last_logit_only=True,
            )
            total_tokens_seen += 1
            h_state = outputs.get("h_state")
            l_state = outputs.get("l_state")
            prev_context = outputs.get("prev_context")
            target_context = outputs.get("target_context")
            drift_state = outputs.get("drift_state")
            ltm_state = outputs.get("ltm_memory_state")
            logits = outputs["logits"][0, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True)


def _iter_test_items(
    tasks: Sequence[ArcAgiTask],
    max_test_items: Optional[int] = None,
) -> Iterable[Tuple[ArcAgiTask, int, ArcAgiExample]]:
    count = 0
    for task in tasks:
        for idx, example in enumerate(task.test):
            yield task, idx, example
            count += 1
            if max_test_items is not None and count >= max_test_items:
                return


def run_arc_agi_eval(
    model,
    tokenizer,
    device,
    dataset_path: str,
    max_tasks: Optional[int] = None,
    max_test_items: Optional[int] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    keep_samples: bool = False,
    result_key: str = "arc_agi",
) -> Dict[str, Any]:
    validate_inference_tokenizer_identity(
        tokenizer,
        getattr(model, "_hierarchos_checkpoint_metadata", {}),
    )
    tasks = load_arc_agi_tasks(dataset_path, max_tasks=max_tasks)
    total = 0
    correct = 0
    parse_failures = 0
    samples = []
    model_config = getattr(model, "config", {})
    alpaca_mode = bool(
        model_config.get("alpaca", False)
        if isinstance(model_config, dict)
        else getattr(model_config, "alpaca", False)
    )

    for task, test_idx, example in _iter_test_items(tasks, max_test_items=max_test_items):
        prompt = wrap_for_hierarchos(
            build_arc_agi_prompt(task, example.input),
            alpaca_mode=alpaca_mode,
        )
        raw = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        predicted = extract_json_grid(raw)
        is_correct = predicted == example.output
        total += 1
        correct += int(is_correct)
        if predicted is None:
            parse_failures += 1
        if keep_samples:
            samples.append(
                {
                    "task_id": task.task_id,
                    "test_index": test_idx,
                    "prediction": predicted,
                    "target": example.output,
                    "correct": is_correct,
                    "raw_generation": raw,
                }
            )

    accuracy = correct / total if total else 0.0
    result: Dict[str, Any] = {
        "results": {
            result_key: {
                "exact_match,none": accuracy,
                "correct": correct,
                "total": total,
                "task_count": len(tasks),
                "parse_failures": parse_failures,
            }
        },
        "versions": {result_key: "local_json"},
        "config": {
            "dataset_path": os.path.abspath(os.path.expanduser(dataset_path)),
            "max_tasks": max_tasks,
            "max_test_items": max_test_items,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "prompt_format": (
                "alpaca-instruction-response"
                if alpaca_mode
                else "user-assistant"
            ),
        },
    }
    if keep_samples:
        result["samples"] = {result_key: samples}
    return result
