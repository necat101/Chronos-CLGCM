"""
Hierarchos Evaluation Module

Optional integration with lm-evaluation-harness for standardized benchmarking.

Usage:
    from hierarchos.evaluation import run_eval, is_lm_eval_available
    
    if is_lm_eval_available():
        results = run_eval(model, tokenizer, device, tasks=["hellaswag"])
"""

from .evaluator import run_eval, is_lm_eval_available, format_results, save_results
from .benchmarks import (
    BENCHMARKS,
    SUITES,
    BenchmarkSpec,
    format_benchmark_catalog,
    get_benchmark,
    list_benchmarks,
    list_suites,
    resolve_task_names,
)
from .post_training import (
    run_post_training_benchmarks,
    write_benchmark_artifacts,
)

def __getattr__(name):
    """Lazily expose the optional lm-eval wrapper.

    The evaluation package is initialized when the trainer imports the
    lightweight checkpoint-selection helpers. Do not pull lm-eval/transformers
    into spawned data workers merely because this package was initialized.
    """

    if name == "HierarchosLM":
        from .lm_eval_wrapper import HierarchosLM

        return HierarchosLM
    raise AttributeError(name)

__all__ = [
    "run_eval",
    "is_lm_eval_available", 
    "format_results",
    "save_results",
    "HierarchosLM",
    "BENCHMARKS",
    "SUITES",
    "BenchmarkSpec",
    "format_benchmark_catalog",
    "get_benchmark",
    "list_benchmarks",
    "list_suites",
    "resolve_task_names",
    "run_post_training_benchmarks",
    "write_benchmark_artifacts",
]
