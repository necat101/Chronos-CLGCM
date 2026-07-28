from .models.core import HierarchosCore
from .models.ltm import LTMModule
from .models.quantized import QuantizedHierarchos, load_quantized
from .training.trainer import train, finetune
from .training.datasets import (
    IterableChunkedJSONLDataset, 
    PTChunkedDataset, 
    TokenizedBinaryDataset,
    StreamingJSONLDataset,
    OriginalJSONLDataset, 
    HuggingFaceStreamingDataset,
    HuggingFaceMapStyleDataset,
    process_text_sample,
    process_text_samples_batch,
    process_tokenized_sample,
    create_dataloader_for_jsonl,
    create_dataloader_for_hf_streaming,
    create_dataloader_for_chunked,
    create_dataloader_pt_chunked,
    create_dataloader_for_tokenized_cache,
    create_map_style_dataloader
)
from .utils.device import (
    configure_torch_runtime,
    cuda_diagnostics,
    describe_device,
    pick_device,
    set_threads,
    is_directml_device,
    setup_msvc_environment,
)
from .utils.checkpoint import AttrDict, load_full_model_with_config, save_checkpoint_safely
from .inference.chat import chat

# Lightweight evaluation orchestration is always importable. The optional
# lm-eval/transformers stack is loaded only when a benchmark is actually
# requested, which keeps Windows DataLoader spawn workers lean and reliable.
from .evaluation import (
    run_eval,
    is_lm_eval_available,
    format_results,
    format_benchmark_catalog,
    resolve_task_names,
    run_post_training_benchmarks,
    write_benchmark_artifacts,
)


def __getattr__(name):
    if name == "HierarchosLM":
        from .evaluation.lm_eval_wrapper import HierarchosLM

        return HierarchosLM
    raise AttributeError(name)
