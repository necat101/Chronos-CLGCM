// embedded.rs — Embeds the Hierarchos Python package into the binary
//
// At build time, all Python source files are embedded via include_str!().
// At runtime, extract_embedded_python() writes them to the app data directory
// so the bridge server can import them.

use std::fs;
use std::path::{Path, PathBuf};

/// Embedded Python source files.
/// Each entry: (relative_path, file_contents)
const EMBEDDED_FILES: &[(&str, &str)] = &[
    // Bridge server
    (
        "hierarchos_bridge_server.py",
        include_str!("../../hierarchos_bridge_server.py"),
    ),
    // Package root
    (
        "hierarchos/__init__.py",
        include_str!("../../hierarchos/__init__.py"),
    ),
    // Models
    (
        "hierarchos/models/act.py",
        include_str!("../../hierarchos/models/act.py"),
    ),
    (
        "hierarchos/models/core.py",
        include_str!("../../hierarchos/models/core.py"),
    ),
    (
        "hierarchos/models/ltm.py",
        include_str!("../../hierarchos/models/ltm.py"),
    ),
    (
        "hierarchos/models/rwkv_cell.py",
        include_str!("../../hierarchos/models/rwkv_cell.py"),
    ),
    (
        "hierarchos/models/quantized.py",
        include_str!("../../hierarchos/models/quantized.py"),
    ),
    (
        "hierarchos/models/revisions.py",
        include_str!("../../hierarchos/models/revisions.py"),
    ),
    (
        "hierarchos/models/shared_adapters.py",
        include_str!("../../hierarchos/models/shared_adapters.py"),
    ),
    // Training
    (
        "hierarchos/training/trainer.py",
        include_str!("../../hierarchos/training/trainer.py"),
    ),
    (
        "hierarchos/training/datasets.py",
        include_str!("../../hierarchos/training/datasets.py"),
    ),
    (
        "hierarchos/training/optimizers.py",
        include_str!("../../hierarchos/training/optimizers.py"),
    ),
    (
        "hierarchos/training/objectives.py",
        include_str!("../../hierarchos/training/objectives.py"),
    ),
    // Inference
    (
        "hierarchos/inference/chat.py",
        include_str!("../../hierarchos/inference/chat.py"),
    ),
    (
        "hierarchos/inference/chat_state.py",
        include_str!("../../hierarchos/inference/chat_state.py"),
    ),
    // Utils
    (
        "hierarchos/utils/device.py",
        include_str!("../../hierarchos/utils/device.py"),
    ),
    (
        "hierarchos/utils/checkpoint.py",
        include_str!("../../hierarchos/utils/checkpoint.py"),
    ),
    (
        "hierarchos/utils/rosa.py",
        include_str!("../../hierarchos/utils/rosa.py"),
    ),
    (
        "hierarchos/utils/lora_merge.py",
        include_str!("../../hierarchos/utils/lora_merge.py"),
    ),
    (
        "hierarchos/utils/safe_loading.py",
        include_str!("../../hierarchos/utils/safe_loading.py"),
    ),
    (
        "hierarchos/utils/tokenizer.py",
        include_str!("../../hierarchos/utils/tokenizer.py"),
    ),
    // Evaluation
    (
        "hierarchos/evaluation/__init__.py",
        include_str!("../../hierarchos/evaluation/__init__.py"),
    ),
    (
        "hierarchos/evaluation/evaluator.py",
        include_str!("../../hierarchos/evaluation/evaluator.py"),
    ),
    (
        "hierarchos/evaluation/lm_eval_wrapper.py",
        include_str!("../../hierarchos/evaluation/lm_eval_wrapper.py"),
    ),
    (
        "hierarchos/evaluation/arc_agi.py",
        include_str!("../../hierarchos/evaluation/arc_agi.py"),
    ),
    (
        "hierarchos/evaluation/benchmarks.py",
        include_str!("../../hierarchos/evaluation/benchmarks.py"),
    ),
    (
        "hierarchos/evaluation/post_training.py",
        include_str!("../../hierarchos/evaluation/post_training.py"),
    ),
    (
        "hierarchos/evaluation/selection.py",
        include_str!("../../hierarchos/evaluation/selection.py"),
    ),
];

/// A simple version hash based on total content length.
/// Changes when any embedded file is modified.
fn content_hash() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (path, content) in EMBEDDED_FILES {
        hasher.update(path.as_bytes());
        hasher.update(content.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn extraction_complete(base_dir: &Path) -> bool {
    EMBEDDED_FILES
        .iter()
        .all(|(rel_path, _)| base_dir.join(rel_path).is_file())
}

fn write_embedded_files(base_dir: &Path, current_hash: &str) -> Result<(), String> {
    for (rel_path, content) in EMBEDDED_FILES {
        let full_path = base_dir.join(rel_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;

            // Namespace-package directories are made regular packages in the
            // extracted runtime, keeping imports stable across Python builds.
            let init = parent.join("__init__.py");
            if !init.exists() && parent != base_dir {
                fs::write(&init, "").map_err(|e| format!("Failed to write {:?}: {}", init, e))?;
            }
        }

        fs::write(&full_path, content)
            .map_err(|e| format!("Failed to write {:?}: {}", full_path, e))?;
    }

    let hash_file = base_dir.join(".version_hash");
    fs::write(&hash_file, current_hash)
        .map_err(|e| format!("Failed to write {:?}: {}", hash_file, e))?;

    if !extraction_complete(base_dir) {
        return Err("Embedded Python extraction is incomplete.".to_string());
    }
    Ok(())
}

/// Get the app data directory for Hierarchos.
pub fn get_app_data_dir() -> PathBuf {
    if let Some(proj_dirs) = directories::ProjectDirs::from("com", "hierarchos", "hierarchos-gui") {
        proj_dirs.data_local_dir().to_path_buf()
    } else {
        // Fallback to a directory next to the executable
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hierarchos_data")
    }
}

/// Extract all embedded Python files to the given directory.
/// Returns the path to `hierarchos_bridge_server.py`.
///
/// Files are only re-extracted if the version hash has changed,
/// making subsequent launches instant.
pub fn extract_embedded_python() -> Result<PathBuf, String> {
    let app_data_dir = get_app_data_dir();
    let base_dir = app_data_dir.join("python");
    let hash_file = base_dir.join(".version_hash");
    let current_hash = content_hash();

    // Check if already extracted with same version
    if hash_file.exists() {
        if let Ok(existing) = fs::read_to_string(&hash_file) {
            if existing.trim() == current_hash && extraction_complete(&base_dir) {
                return Ok(base_dir.join("hierarchos_bridge_server.py"));
            }
        }
    }

    // Build a complete fresh tree first. Replacing the directory removes
    // orphaned modules from older GUI versions and never exposes a partially
    // written package if extraction fails midway.
    fs::create_dir_all(&app_data_dir)
        .map_err(|e| format!("Failed to create {:?}: {}", app_data_dir, e))?;
    let suffix = std::process::id();
    let staging_dir = app_data_dir.join(format!("python.staging.{}", suffix));
    let backup_dir = app_data_dir.join(format!("python.previous.{}", suffix));
    if staging_dir.exists() {
        fs::remove_dir_all(&staging_dir)
            .map_err(|e| format!("Failed to clear {:?}: {}", staging_dir, e))?;
    }
    write_embedded_files(&staging_dir, &current_hash)?;

    if backup_dir.exists() {
        fs::remove_dir_all(&backup_dir)
            .map_err(|e| format!("Failed to clear {:?}: {}", backup_dir, e))?;
    }
    if base_dir.exists() {
        fs::rename(&base_dir, &backup_dir)
            .map_err(|e| format!("Failed to stage existing runtime {:?}: {}", base_dir, e))?;
    }

    if let Err(e) = fs::rename(&staging_dir, &base_dir) {
        if backup_dir.exists() {
            let _ = fs::rename(&backup_dir, &base_dir);
        }
        let _ = fs::remove_dir_all(&staging_dir);
        return Err(format!("Failed to activate embedded Python runtime: {}", e));
    }
    if backup_dir.exists() {
        let _ = fs::remove_dir_all(&backup_dir);
    }

    Ok(base_dir.join("hierarchos_bridge_server.py"))
}

/// Get the extraction directory (for PYTHONPATH).
pub fn get_python_base_dir() -> PathBuf {
    get_app_data_dir().join("python")
}

/// Get the default model download directory.
pub fn get_models_dir() -> PathBuf {
    get_app_data_dir().join("models")
}
