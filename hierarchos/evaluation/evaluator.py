"""
Evaluation runner for Hierarchos models using lm-evaluation-harness.

This module provides a simple interface to run standardized benchmarks
on Hierarchos models during or after training.
"""
import json
from typing import Dict, List, Optional, Any

_LM_EVAL_IMPORT_ERROR = None


def _load_simple_evaluate():
    """Import the optional benchmark stack only in the process using it.

    Importing lm-eval at module initialization made every spawned Windows
    DataLoader worker load transformers' audio/image dependencies. Besides
    wasting substantial RAM/startup time, an optional libsndfile failure could
    kill an otherwise unrelated training worker.
    """

    global _LM_EVAL_IMPORT_ERROR
    try:
        from lm_eval import simple_evaluate
    except Exception as exc:
        _LM_EVAL_IMPORT_ERROR = exc
        return None
    _LM_EVAL_IMPORT_ERROR = None
    return simple_evaluate


def is_lm_eval_available() -> bool:
    """Check that lm-eval and its runtime dependencies import successfully."""
    return _load_simple_evaluate() is not None


def run_eval(
    model,
    tokenizer,
    device,
    tasks: List[str] = ["hellaswag"],
    batch_size: int = 1,
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    verbosity: str = "INFO"
) -> Optional[Dict[str, Any]]:
    """
    Run lm-evaluation-harness benchmarks on a Hierarchos model.
    
    This function wraps the model in HierarchosLM and runs evaluations
    using lm-eval's simple_evaluate interface.
    
    Args:
        model: HierarchosCore model instance
        tokenizer: Tokenizer for the model
        device: torch.device to run inference on
        tasks: List of benchmark task names (e.g., ["hellaswag", "arc_easy"])
               See: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
        batch_size: Batch size for evaluation (default: 1)
        limit: Optional sample limit per task (for fast testing)
        num_fewshot: Number of few-shot examples (None = task default)
        verbosity: Logging verbosity ("DEBUG", "INFO", "WARNING", "ERROR")
    
    Returns:
        dict: Results dictionary with metrics per task, or None if lm-eval not installed
        
    Example:
        >>> results = run_eval(model, tokenizer, device, tasks=["hellaswag"])
        >>> print(results["results"]["hellaswag"]["acc,none"])
        0.45
    """
    simple_evaluate = _load_simple_evaluate()
    if simple_evaluate is None:
        print("WARNING: lm-evaluation-harness is unavailable.")
        print("         Install with: pip install lm-eval>=0.4.0")
        if _LM_EVAL_IMPORT_ERROR is not None:
            print(f"         Import error: {_LM_EVAL_IMPORT_ERROR}")
        print("         Skipping evaluation.")
        return None
    
    # Import here to avoid issues when lm-eval not installed
    from .lm_eval_wrapper import HierarchosLM
    
    # Wrap model
    lm = HierarchosLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size
    )
    
    # Set model to eval mode
    model.eval()
    
    # Build evaluate kwargs
    eval_kwargs = {
        "model": lm,
        "tasks": tasks,
        "batch_size": batch_size,
        "verbosity": verbosity,
    }
    
    if limit is not None:
        eval_kwargs["limit"] = limit
    
    if num_fewshot is not None:
        eval_kwargs["num_fewshot"] = num_fewshot
    
    # Run evaluation
    print(f"Running lm-eval on tasks: {tasks}")
    if limit:
        print(f"  (Limited to {limit} samples per task)")
    
    try:
        results = simple_evaluate(**eval_kwargs)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return None
    
    return results


def format_results(results: Dict[str, Any], tasks: Optional[List[str]] = None) -> str:
    """
    Format evaluation results as a human-readable string.
    
    Args:
        results: Results dict from run_eval
        tasks: Optional list of tasks to include (None = all)
    
    Returns:
        Formatted string with key metrics per task
    """
    if results is None:
        return "No results available."
    
    task_results = results.get("results", {})
    
    lines = ["=" * 50, "Evaluation Results", "=" * 50]
    
    for task_name, metrics in task_results.items():
        if tasks is not None and task_name not in tasks:
            continue
        
        lines.append(f"\n{task_name}:")
        
        # Common metric keys to display
        display_metrics = ["acc,none", "acc_norm,none", "exact_match,none", "f1,none"]
        
        shown = False
        for key, value in metrics.items():
            if any(dm in key for dm in display_metrics) or "acc" in key.lower():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
                shown = True
        
        # If no specific metrics, show all
        if not shown:
            for key, value in metrics.items():
                if key.startswith("alias"):
                    continue
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Results dict from run_eval
        filepath: Path to save JSON file
    """
    if results is None:
        print("No results to save.")
        return
    
    # Make results JSON serializable
    serializable = {}
    for key, value in results.items():
        if key == "samples":
            # Skip samples (too large)
            continue
        serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"Results saved to: {filepath}")
