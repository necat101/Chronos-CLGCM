// bridge.rs — Python ↔ Rust communication via subprocess + JSON-RPC
//
// Launches a Python subprocess running the Hierarchos bridge server,
// communicates via stdin/stdout JSON messages, and streams results
// back to the UI thread through tokio channels.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const NATIVE_VULKAN_EVENT_PREFIX: &str = "HIERARCHOS_EVENT ";

/// Events flowing from the Python backend to the GUI.
#[derive(Debug, Clone)]
pub enum BridgeEvent {
    /// A single token from streaming generation.
    Token(String),
    /// Generation has completed.
    GenerationComplete,
    /// Training reached a terminal state.
    TrainingComplete { status: String },
    /// Training metrics for a step.
    TrainingMetrics {
        epoch: u32,
        step: u32,
        total_steps: Option<u32>,
        loss: f64,
        lr: f64,
        ponder_cost: Option<f64>,
        commitment_cost: Option<f64>,
        tokens_per_sec: Option<f64>,
    },
    /// Model successfully loaded.
    ModelLoaded(ModelConfig),
    /// Backend/model loading progress for user-facing feedback.
    LoadProgress(LoadProgress),
    /// Current model was unloaded by the backend.
    ModelUnloaded,
    /// LTM memory snapshot for visualization.
    LtmSnapshot {
        fast_vals: Vec<Vec<f32>>,
        slow_vals: Vec<Vec<f32>>,
        timestamps: Vec<f32>,
        sources: Vec<i32>,
    },
    /// Status update from the backend.
    Status(String),
    /// An error occurred.
    Error(String),
    /// Model info for the inspector.
    ModelInfo(ModelInspection),
    /// Runtime LTM updates were saved by the backend.
    LtmSaved(String),
    /// Connection to the Python backend was lost.
    ConnectionLost(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadProgress {
    pub progress: f32,
    pub label: String,
}

/// Model configuration returned after loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub context_dim: u32,
    pub h_hidden: u32,
    pub l_hidden: u32,
    pub ltm_slots: u32,
    pub ltm_key_dim: u32,
    pub ltm_val_dim: u32,
    pub ltm_topk: u32,
    pub vocab_size: u32,
    pub max_length: u32,
    pub h_stride: u32,
    pub max_h_steps: u32,
    pub max_l_steps: u32,
    pub persistent_dim: u32,
    #[serde(default = "default_training_chunk_size")]
    pub training_chunk_size: u32,
    #[serde(default)]
    pub full_sample_bptt: bool,
    #[serde(default = "default_activation_checkpointing")]
    pub full_sample_activation_checkpointing: bool,
    #[serde(default = "default_checkpoint_segment_size")]
    pub full_sample_checkpoint_segment_size: u32,
    pub is_quantized: bool,
    pub device: String,
    #[serde(default)]
    pub device_label: Option<String>,
    #[serde(default)]
    pub torch_version: Option<String>,
    #[serde(default)]
    pub cuda_built: bool,
    #[serde(default)]
    pub cuda_available: bool,
    #[serde(default)]
    pub cuda_version: Option<String>,
    #[serde(default)]
    pub cuda_device_name: Option<String>,
    #[serde(default)]
    pub vram_total_mb: Option<u64>,
    pub total_params: u64,
}

/// Model inspection data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInspection {
    pub layers: Vec<LayerInfo>,
    pub total_params: u64,
    pub trainable_params: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub name: String,
    pub param_count: u64,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

/// Sampling parameters sent to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_new_tokens: u32,
    pub cpu_threads: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.2,
            max_new_tokens: 512,
            cpu_threads: default_cpu_threads(),
        }
    }
}

fn default_cpu_threads() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
        .saturating_div(2)
        .max(1)
}

fn default_training_chunk_size() -> u32 {
    256
}

fn default_activation_checkpointing() -> bool {
    true
}

fn default_checkpoint_segment_size() -> u32 {
    128
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingBackend {
    Pytorch,
    Vulkan,
}

impl Default for TrainingBackend {
    fn default() -> Self {
        Self::Pytorch
    }
}

impl TrainingBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pytorch => "PyTorch",
            Self::Vulkan => "Vulkan (native)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VulkanTrainingPrecision {
    Fp32,
    Fp16StorageFp32Compute,
    Fp16Parity,
    Fp16LmBackward,
}

impl Default for VulkanTrainingPrecision {
    fn default() -> Self {
        Self::Fp32
    }
}

impl VulkanTrainingPrecision {
    pub const ALL: [Self; 4] = [
        Self::Fp32,
        Self::Fp16StorageFp32Compute,
        Self::Fp16Parity,
        Self::Fp16LmBackward,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32 parity",
            Self::Fp16StorageFp32Compute => "FP16 storage / FP32 compute",
            Self::Fp16Parity => "FP16 parity + GradScaler",
            Self::Fp16LmBackward => "FP16 LM backward + GradScaler",
        }
    }

    fn env_value(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16StorageFp32Compute => "fp16-storage-fp32-compute",
            Self::Fp16Parity => "fp16-storage-parity",
            Self::Fp16LmBackward => "fp16-storage-fp16-lm-backward",
        }
    }

    fn from_manifest_label(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fp32" => Ok(Self::Fp32),
            "fp16" | "fp16-storage-fp32-compute" => Ok(Self::Fp16StorageFp32Compute),
            "fp16-parity" | "fp16-storage-parity" => Ok(Self::Fp16Parity),
            "fp16-lm-backward" | "fp16-storage-fp16-lm-backward" => Ok(Self::Fp16LmBackward),
            other => Err(format!(
                "Checkpoint requests unsupported Vulkan training precision {other:?}."
            )),
        }
    }
}

/// Training configuration sent to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub backend: TrainingBackend,
    pub vulkan_device_indices: String,
    pub vulkan_exact_resume: bool,
    pub vulkan_precision: VulkanTrainingPrecision,
    pub vulkan_tbptt_enabled: bool,
    pub data_path: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub min_lr: f64,
    pub warmup_steps: u64,
    pub warmup_ratio: f64,
    pub disable_lr_schedule: bool,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub z_loss_weight: f32,
    pub ponder_loss_weight: f32,
    pub commitment_loss_weight: f32,
    pub max_ce_loss_for_backward: f32,
    pub max_ponder_cost_for_backward: f32,
    pub max_commitment_cost_for_backward: f32,
    pub max_skipped_train_batches: u64,
    pub seed: u64,
    pub shuffle: bool,
    pub training_chunk_size: u32,
    pub full_sample_bptt: bool,
    pub full_sample_activation_checkpointing: bool,
    pub full_sample_checkpoint_segment_size: u32,
    pub accumulation_steps: u32,
    pub grad_clip: f32,
    pub persist_state: bool,
    pub amp: bool,
    pub save_steps: u32,
    pub out_dir: String,
    pub context_dim: u32,
    pub h_hidden: u32,
    pub l_hidden: u32,
    pub persistent_dim: u32,
    pub ltm_slots: u32,
    pub ltm_key_dim: u32,
    pub ltm_val_dim: u32,
    pub ltm_topk: u32,
    pub h_stride: u32,
    pub max_h_steps: u32,
    pub max_l_steps: u32,
    pub max_length: u32,
    pub auto_max_length: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            backend: TrainingBackend::Pytorch,
            vulkan_device_indices: "0".to_string(),
            vulkan_exact_resume: false,
            vulkan_precision: VulkanTrainingPrecision::Fp32,
            vulkan_tbptt_enabled: true,
            data_path: String::new(),
            epochs: 3,
            batch_size: 64,
            learning_rate: 1e-4,
            min_lr: 1e-6,
            warmup_steps: 0,
            warmup_ratio: 0.0,
            disable_lr_schedule: false,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            weight_decay: 0.1,
            z_loss_weight: 1.0e-4,
            ponder_loss_weight: 0.003,
            commitment_loss_weight: 0.5,
            max_ce_loss_for_backward: 0.0,
            max_ponder_cost_for_backward: 0.0,
            max_commitment_cost_for_backward: 2.0,
            max_skipped_train_batches: 0,
            seed: 0,
            shuffle: true,
            training_chunk_size: default_training_chunk_size(),
            full_sample_bptt: false,
            full_sample_activation_checkpointing: true,
            full_sample_checkpoint_segment_size: 128,
            accumulation_steps: 1,
            grad_clip: 1.0,
            persist_state: false,
            amp: true,
            save_steps: 0,
            out_dir: "./hierarchos_model".to_string(),
            context_dim: 448,
            h_hidden: 448,
            l_hidden: 448,
            persistent_dim: 128,
            ltm_slots: 1024,
            ltm_key_dim: 128,
            ltm_val_dim: 128,
            ltm_topk: 4,
            h_stride: 4,
            max_h_steps: 5,
            max_l_steps: 5,
            max_length: 1024,
            auto_max_length: false,
        }
    }
}

impl TrainingConfig {
    pub fn sync_architecture_from_model(&mut self, model: &ModelConfig) {
        self.context_dim = model.context_dim;
        self.h_hidden = model.h_hidden;
        self.l_hidden = model.l_hidden;
        self.persistent_dim = model.persistent_dim;
        self.ltm_slots = model.ltm_slots;
        self.ltm_key_dim = model.ltm_key_dim;
        self.ltm_val_dim = model.ltm_val_dim;
        self.ltm_topk = model.ltm_topk;
        self.h_stride = model.h_stride;
        self.max_h_steps = model.max_h_steps;
        self.max_l_steps = model.max_l_steps;
        self.max_length = model.max_length;
        self.training_chunk_size = model.training_chunk_size;
        self.full_sample_bptt = model.full_sample_bptt;
        self.full_sample_activation_checkpointing = model.full_sample_activation_checkpointing;
        self.full_sample_checkpoint_segment_size = model.full_sample_checkpoint_segment_size;
    }
}

/// JSON-RPC request format.
#[derive(Serialize)]
struct RpcRequest {
    method: String,
    params: serde_json::Value,
}

/// The main Python bridge.
pub struct PythonBridge {
    event_tx: mpsc::UnboundedSender<BridgeEvent>,
    event_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<BridgeEvent>>>,
    runtime: Arc<tokio::runtime::Runtime>,
    model_loaded: Arc<AtomicBool>,
    generating: Arc<AtomicBool>,
    training: Arc<AtomicBool>,
    connecting: Arc<AtomicBool>,
    loading: Arc<AtomicBool>,
    connected: Arc<AtomicBool>,
    /// Handle for writing to the child process stdin.
    child_stdin: Arc<tokio::sync::Mutex<Option<tokio::process::ChildStdin>>>,
    /// Handle to the child process for cleanup.
    child_handle: Arc<tokio::sync::Mutex<Option<Child>>>,
    /// Direct native Vulkan trainer process, separate from the Python inference bridge.
    native_training_handle: Arc<tokio::sync::Mutex<Option<Child>>>,
    native_training_active: Arc<AtomicBool>,
    native_training_stop_requested: Arc<AtomicBool>,
}

enum BackendLaunch {
    Bundled {
        exe: PathBuf,
        working_dir: PathBuf,
    },
    Python {
        python: String,
        script: PathBuf,
        pythonpath: PathBuf,
    },
}

fn find_bundled_backend() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))?;

    let candidates = [
        exe_dir.join("hierarchos-backend.exe"),
        exe_dir.join("backend").join("hierarchos-backend.exe"),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn platform_executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn find_vulkan_trainer() -> Option<PathBuf> {
    let name = platform_executable("hierarchos-vulkan-train");
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("vulkan").join(&name));
            candidates.push(exe_dir.join(&name));
        }
    }
    let gui_manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = gui_manifest_dir.parent() {
        candidates.push(
            repo_root
                .join("hierarchos-vulkan")
                .join("target")
                .join("release")
                .join(&name),
        );
        candidates.push(
            repo_root
                .join("hierarchos-vulkan")
                .join("target")
                .join("debug")
                .join(&name),
        );
    }
    candidates.into_iter().find(|path| path.is_file())
}

fn resolve_vulkan_model_dir(model_path: &str) -> Result<PathBuf, String> {
    let requested = PathBuf::from(model_path.trim());
    if model_path.trim().is_empty() {
        return Err(
            "Load a local Hierarchos SafeTensors model package before Vulkan training.".to_string(),
        );
    }
    let model_dir = if requested.is_file() {
        let is_safetensors = requested
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("model.safetensors"));
        if !is_safetensors {
            return Err(format!(
                "Native Vulkan training requires a local package directory containing model.safetensors; the loaded model is {}.",
                requested.display()
            ));
        }
        requested.parent().map(Path::to_path_buf).ok_or_else(|| {
            "Could not resolve the loaded SafeTensors package directory.".to_string()
        })?
    } else {
        requested
    };
    if !model_dir.is_dir() || !model_dir.join("model.safetensors").is_file() {
        return Err(format!(
            "Native Vulkan training requires a local Hierarchos package with model.safetensors; not found under {}.",
            model_dir.display()
        ));
    }
    Ok(model_dir)
}

fn config_json_f64(config: &serde_json::Value, key: &str) -> Result<Option<f64>, String> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("Checkpoint training field {key:?} must be numeric.")),
    }
}

fn config_json_u64(config: &serde_json::Value, key: &str) -> Result<Option<u64>, String> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            format!("Checkpoint training field {key:?} must be an unsigned integer.")
        }),
    }
}

fn config_json_bool(config: &serde_json::Value, key: &str) -> Result<Option<bool>, String> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| format!("Checkpoint training field {key:?} must be boolean.")),
    }
}

fn resolve_vulkan_exact_resume_config_from_manifest(
    manifest: &serde_json::Value,
    requested: &TrainingConfig,
) -> Result<TrainingConfig, String> {
    let session = manifest
        .get("training_session")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            "Exact resume requires backend-neutral training_session metadata in training_state.json."
                .to_string()
        })?;
    let config = session
        .get("effective_training_config")
        .filter(|value| value.is_object())
        .ok_or_else(|| {
            "Exact resume checkpoint is missing effective_training_config metadata.".to_string()
        })?;
    let precision_label = manifest
        .get("training_precision_policy")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            "Exact resume checkpoint is missing training_precision_policy.".to_string()
        })?;

    // Runtime-only choices stay under GUI control: dataset, output directory,
    // target epoch count, save cadence, and device topology. Every field that
    // participates in the persisted numerical/data trajectory is reconstructed
    // from the checkpoint so the GUI cannot accidentally turn exact resume into
    // a different training run.
    let mut resolved = requested.clone();
    resolved.vulkan_precision = VulkanTrainingPrecision::from_manifest_label(precision_label)?;

    if let Some(value) = config_json_u64(config, "batch_size")? {
        resolved.batch_size = u32::try_from(value)
            .map_err(|_| "Checkpoint batch_size exceeds the GUI range.".to_string())?;
    }
    if let Some(value) = config_json_u64(config, "gradient_accumulation_steps")? {
        resolved.accumulation_steps = u32::try_from(value).map_err(|_| {
            "Checkpoint gradient_accumulation_steps exceeds the GUI range.".to_string()
        })?;
    }
    if let Some(value) = config_json_f64(config, "starting_lr")? {
        resolved.learning_rate = value;
    }
    if let Some(value) = config_json_f64(config, "min_lr")? {
        resolved.min_lr = value;
    }
    if let Some(value) = config_json_u64(config, "warmup_steps")? {
        resolved.warmup_steps = value;
    }
    if let Some(value) = config_json_f64(config, "warmup_ratio")? {
        resolved.warmup_ratio = value;
    }
    if let Some(value) = config_json_bool(config, "disable_lr_schedule")? {
        resolved.disable_lr_schedule = value;
    }
    if let Some(value) = config_json_f64(config, "beta1")? {
        resolved.beta1 = value as f32;
    }
    if let Some(value) = config_json_f64(config, "beta2")? {
        resolved.beta2 = value as f32;
    }
    if let Some(value) = config_json_f64(config, "eps")? {
        resolved.eps = value as f32;
    }
    if let Some(value) = config_json_f64(config, "weight_decay")? {
        resolved.weight_decay = value as f32;
    }
    if let Some(value) = config_json_f64(config, "z_loss_weight")? {
        resolved.z_loss_weight = value as f32;
    }
    if let Some(value) = config_json_f64(config, "ponder_loss_weight")? {
        resolved.ponder_loss_weight = value as f32;
    }
    if let Some(value) = config_json_f64(config, "commitment_loss_weight")? {
        resolved.commitment_loss_weight = value as f32;
    }
    if let Some(value) = config_json_f64(config, "max_ce_loss_for_backward")? {
        resolved.max_ce_loss_for_backward = value as f32;
    }
    if let Some(value) = config_json_f64(config, "max_ponder_cost_for_backward")? {
        resolved.max_ponder_cost_for_backward = value as f32;
    }
    if let Some(value) = config_json_f64(config, "max_commitment_cost_for_backward")? {
        resolved.max_commitment_cost_for_backward = value as f32;
    }
    if let Some(value) = config_json_u64(config, "max_skipped_train_batches")? {
        resolved.max_skipped_train_batches = value;
    }
    if let Some(value) = config_json_u64(config, "seed")? {
        resolved.seed = value;
    }
    if let Some(value) = config_json_bool(config, "shuffle")? {
        resolved.shuffle = value;
    }
    if let Some(value) = config_json_f64(config, "grad_clip")? {
        resolved.grad_clip = value as f32;
    }
    match config.get("tbptt_chunk_size") {
        None => {}
        Some(serde_json::Value::Null) => resolved.vulkan_tbptt_enabled = false,
        Some(value) => {
            let value = value.as_u64().ok_or_else(|| {
                "Checkpoint training field \"tbptt_chunk_size\" must be an unsigned integer or null."
                    .to_string()
            })?;
            resolved.training_chunk_size = u32::try_from(value)
                .map_err(|_| "Checkpoint tbptt_chunk_size exceeds the GUI range.".to_string())?;
            resolved.vulkan_tbptt_enabled = true;
        }
    }
    if let Some(value) = config_json_bool(config, "persist_state")? {
        resolved.persist_state = value;
    }

    Ok(resolved)
}

fn resolve_vulkan_launch_config(
    model_dir: &Path,
    requested: &TrainingConfig,
) -> Result<TrainingConfig, String> {
    if !requested.vulkan_exact_resume {
        return Ok(requested.clone());
    }
    let manifest_path = model_dir.join("training_state.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|error| format!("Could not read {}: {error}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Could not decode {}: {error}", manifest_path.display()))?;
    resolve_vulkan_exact_resume_config_from_manifest(&manifest, requested)
}

fn parse_vulkan_device_indices(raw: &str) -> Result<Vec<usize>, String> {
    let mut indices = Vec::new();
    for value in raw.split(',') {
        let value = value.trim();
        if value.is_empty() {
            return Err(
                "Vulkan device indices must be a comma-separated list such as 0 or 0,1."
                    .to_string(),
            );
        }
        let index = value
            .parse::<usize>()
            .map_err(|_| format!("Invalid Vulkan device index {value:?}."))?;
        if indices.contains(&index) {
            return Err(format!("Vulkan device index {index} is duplicated."));
        }
        indices.push(index);
    }
    if indices.is_empty() {
        return Err("Select at least one Vulkan device index.".to_string());
    }
    Ok(indices)
}

fn parse_native_training_event(line: &str) -> Option<BridgeEvent> {
    let payload = line.strip_prefix(NATIVE_VULKAN_EVENT_PREFIX)?;
    let msg: serde_json::Value = serde_json::from_str(payload).ok()?;
    match msg.get("event").and_then(|value| value.as_str())? {
        "training_started" => {
            let device = msg
                .get("device")
                .and_then(|value| value.as_str())
                .unwrap_or("Vulkan");
            let total_steps = msg
                .get("total_steps")
                .and_then(|value| value.as_u64())
                .unwrap_or(0);
            Some(BridgeEvent::Status(format!(
                "Native Vulkan training started on {device} ({total_steps} batches scheduled)."
            )))
        }
        "training_metrics" => Some(BridgeEvent::TrainingMetrics {
            epoch: msg
                .get("epoch")
                .and_then(|value| value.as_u64())
                .unwrap_or(0) as u32,
            step: msg
                .get("step")
                .and_then(|value| value.as_u64())
                .unwrap_or(0) as u32,
            total_steps: msg
                .get("total_steps")
                .and_then(|value| value.as_u64())
                .map(|value| value.min(u64::from(u32::MAX)) as u32),
            loss: msg
                .get("loss")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            lr: msg
                .get("lr")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            ponder_cost: None,
            commitment_cost: None,
            tokens_per_sec: msg.get("tokens_per_sec").and_then(|value| value.as_f64()),
        }),
        _ => None,
    }
}

fn resolve_backend_launch(python_path: &str) -> Result<BackendLaunch, String> {
    let requested = python_path.trim();
    let prefer_bundled = requested.is_empty()
        || requested.eq_ignore_ascii_case("auto")
        || requested.eq_ignore_ascii_case("bundled");

    if prefer_bundled {
        if let Some(exe) = find_bundled_backend() {
            let working_dir = exe
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            return Ok(BackendLaunch::Bundled { exe, working_dir });
        }
    }

    let script = crate::embedded::extract_embedded_python()?;
    let pythonpath = crate::embedded::get_python_base_dir();
    let python = if requested.is_empty()
        || requested.eq_ignore_ascii_case("auto")
        || requested.eq_ignore_ascii_case("bundled")
    {
        "python".to_string()
    } else {
        requested.to_string()
    };

    Ok(BackendLaunch::Python {
        python,
        script,
        pythonpath,
    })
}

fn hide_backend_window(command: &mut Command) {
    #[cfg(windows)]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

impl PythonBridge {
    pub fn new() -> Self {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime"),
        );

        Self {
            event_tx,
            event_rx: Arc::new(tokio::sync::Mutex::new(event_rx)),
            runtime,
            model_loaded: Arc::new(AtomicBool::new(false)),
            generating: Arc::new(AtomicBool::new(false)),
            training: Arc::new(AtomicBool::new(false)),
            connecting: Arc::new(AtomicBool::new(false)),
            loading: Arc::new(AtomicBool::new(false)),
            connected: Arc::new(AtomicBool::new(false)),
            child_stdin: Arc::new(tokio::sync::Mutex::new(None)),
            child_handle: Arc::new(tokio::sync::Mutex::new(None)),
            native_training_handle: Arc::new(tokio::sync::Mutex::new(None)),
            native_training_active: Arc::new(AtomicBool::new(false)),
            native_training_stop_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connect to the Python bridge server subprocess.
    pub fn connect(&self, python_path: &str) {
        let tx = self.event_tx.clone();
        if self.connected.load(Ordering::SeqCst) {
            tx.send(BridgeEvent::Status(
                "Backend already connected.".to_string(),
            ))
            .ok();
            return;
        }
        if self.connecting.swap(true, Ordering::SeqCst) {
            tx.send(BridgeEvent::Status(
                "Backend connection already in progress.".to_string(),
            ))
            .ok();
            return;
        }
        tx.send(BridgeEvent::LoadProgress(LoadProgress {
            progress: 0.03,
            label: "Starting backend".to_string(),
        }))
        .ok();

        let connected = self.connected.clone();
        let connecting = self.connecting.clone();
        let model_loaded = self.model_loaded.clone();
        let generating = self.generating.clone();
        let training = self.training.clone();
        let loading = self.loading.clone();
        let stdin_holder = self.child_stdin.clone();
        let handle_holder = self.child_handle.clone();
        let launch = match resolve_backend_launch(python_path) {
            Ok(launch) => launch,
            Err(e) => {
                self.connecting.store(false, Ordering::SeqCst);
                tx.send(BridgeEvent::Error(format!(
                    "Failed to prepare backend: {}",
                    e
                )))
                .ok();
                return;
            }
        };

        self.runtime.spawn(async move {
            let child_result = match &launch {
                BackendLaunch::Bundled { exe, working_dir } => {
                    tx.send(BridgeEvent::LoadProgress(LoadProgress {
                        progress: 0.06,
                        label: "Launching bundled runtime".to_string(),
                    })).ok();
                    tx.send(BridgeEvent::Status(format!(
                        "Connecting to bundled backend: {}",
                        exe.display()
                    ))).ok();
                    let mut command = Command::new(exe);
                    command
                        .current_dir(working_dir)
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .kill_on_drop(true);
                    hide_backend_window(&mut command);
                    command.spawn()
                }
                BackendLaunch::Python { python, script, pythonpath } => {
                    tx.send(BridgeEvent::LoadProgress(LoadProgress {
                        progress: 0.06,
                        label: "Launching Python runtime".to_string(),
                    })).ok();
                    tx.send(BridgeEvent::Status(format!(
                        "Connecting to Python backend: {} {}",
                        python,
                        script.display()
                    ))).ok();
                    let mut command = Command::new(python);
                    command
                        .arg(script)
                        .env("PYTHONPATH", pythonpath)
                        .current_dir(pythonpath)
                        .stdin(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .kill_on_drop(true);
                    hide_backend_window(&mut command);
                    command.spawn()
                }
            };

            let mut child = match child_result {
                Ok(c) => c,
                Err(e) => {
                    connecting.store(false, Ordering::SeqCst);
                    tx.send(BridgeEvent::Error(format!(
                        "Failed to start backend: {}. Use bundled backend or set a Python path in Settings.",
                        e
                    ))).ok();
                    return;
                }
            };

            let stdout = child.stdout.take().expect("Failed to capture stdout");
            let stderr = child.stderr.take();
            let child_stdin_handle = child.stdin.take().expect("Failed to capture stdin");

            if let Some(stderr) = stderr {
                let tx_stderr = tx.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stderr).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            tx_stderr.send(BridgeEvent::Status(format!(
                                "Backend: {}",
                                trimmed
                            ))).ok();
                        }
                    }
                });
            }

            // Store stdin handle for sending commands
            {
                let mut holder = stdin_holder.lock().await;
                *holder = Some(child_stdin_handle);
            }
            {
                let mut holder = handle_holder.lock().await;
                *holder = Some(child);
            }

            connected.store(true, Ordering::SeqCst);
            connecting.store(false, Ordering::SeqCst);
            tx.send(BridgeEvent::LoadProgress(LoadProgress {
                progress: 0.12,
                label: "Backend process connected".to_string(),
            })).ok();
            tx.send(BridgeEvent::Status("Backend connected.".to_string())).ok();

            // Read stdout line by line and dispatch events
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(&line);
                match parsed {
                    Ok(msg) => {
                        let event_type = msg.get("event").and_then(|v| v.as_str()).unwrap_or("");
                        match event_type {
                            "token" => {
                                if let Some(text) = msg.get("text").and_then(|v| v.as_str()) {
                                    tx.send(BridgeEvent::Token(text.to_string())).ok();
                                }
                            }
                            "generation_complete" => {
                                generating.store(false, Ordering::SeqCst);
                                tx.send(BridgeEvent::GenerationComplete).ok();
                            }
                            "training_complete" => {
                                training.store(false, Ordering::SeqCst);
                                let status = msg
                                    .get("status")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("completed")
                                    .to_string();
                                tx.send(BridgeEvent::TrainingComplete { status }).ok();
                            }
                            "training_metrics" => {
                                tx.send(BridgeEvent::TrainingMetrics {
                                    epoch: msg.get("epoch").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    step: msg.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    total_steps: msg
                                        .get("total_steps")
                                        .and_then(|v| v.as_u64())
                                        .map(|value| value.min(u64::from(u32::MAX)) as u32),
                                    loss: msg.get("loss").and_then(|v| v.as_f64()).unwrap_or(0.0),
                                    lr: msg.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.0),
                                    ponder_cost: msg.get("ponder_cost").and_then(|v| v.as_f64()),
                                    commitment_cost: msg.get("commitment_cost").and_then(|v| v.as_f64()),
                                    tokens_per_sec: msg.get("tokens_per_sec").and_then(|v| v.as_f64()),
                                }).ok();
                            }
                            "model_loaded" => {
                                if let Some(config_val) = msg.get("config") {
                                    match serde_json::from_value::<ModelConfig>(config_val.clone()) {
                                        Ok(config) => {
                                            model_loaded.store(true, Ordering::SeqCst);
                                            loading.store(false, Ordering::SeqCst);
                                            tx.send(BridgeEvent::LoadProgress(LoadProgress {
                                                progress: 1.0,
                                                label: "Model ready".to_string(),
                                            })).ok();
                                            tx.send(BridgeEvent::ModelLoaded(config)).ok();
                                        }
                                        Err(e) => {
                                            loading.store(false, Ordering::SeqCst);
                                            tx.send(BridgeEvent::Error(
                                                format!("Failed to parse model config: {}", e)
                                            )).ok();
                                        }
                                    }
                                }
                            }
                            "model_unloaded" => {
                                model_loaded.store(false, Ordering::SeqCst);
                                tx.send(BridgeEvent::ModelUnloaded).ok();
                            }
                            "ltm_snapshot" => {
                                let fast_vals = parse_nested_f32_vec(msg.get("fast_vals"));
                                let slow_vals = parse_nested_f32_vec(msg.get("slow_vals"));
                                let timestamps = parse_f32_vec(msg.get("timestamps"));
                                let sources = parse_i32_vec(msg.get("sources"));
                                tx.send(BridgeEvent::LtmSnapshot {
                                    fast_vals, slow_vals, timestamps, sources,
                                }).ok();
                            }
                            "model_info" => {
                                if let Ok(info) = serde_json::from_value::<ModelInspection>(
                                    serde_json::json!({
                                        "layers": msg.get("layers").cloned().unwrap_or(serde_json::json!([])),
                                        "total_params": msg.get("total_params").and_then(|v| v.as_u64()).unwrap_or(0),
                                        "trainable_params": msg.get("trainable_params").and_then(|v| v.as_u64()).unwrap_or(0),
                                    })
                                ) {
                                    tx.send(BridgeEvent::ModelInfo(info)).ok();
                                }
                            }
                            "ltm_saved" => {
                                let path = msg
                                    .get("path")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                tx.send(BridgeEvent::LtmSaved(path)).ok();
                            }
                            "load_progress" => {
                                let progress = msg
                                    .get("progress")
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0)
                                    .clamp(0.0, 1.0) as f32;
                                let label = msg
                                    .get("label")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("Loading")
                                    .to_string();
                                tx.send(BridgeEvent::LoadProgress(LoadProgress {
                                    progress,
                                    label,
                                })).ok();
                            }
                            "status" => {
                                if let Some(message) = msg.get("message").and_then(|v| v.as_str()) {
                                    tx.send(BridgeEvent::Status(message.to_string())).ok();
                                }
                            }
                            "error" => {
                                if let Some(message) = msg.get("message").and_then(|v| v.as_str()) {
                                    loading.store(false, Ordering::SeqCst);
                                    tx.send(BridgeEvent::Error(message.to_string())).ok();
                                }
                            }
                            "pong" => {
                                // Heartbeat acknowledged
                            }
                            _ => {
                                // Unknown event type — log for debugging
                            }
                        }
                    }
                    Err(_) => {
                        // Non-JSON output from Python (e.g., print statements) — ignore
                    }
                }
            }

            // If we reach here, the subprocess has exited
            connected.store(false, Ordering::SeqCst);
            connecting.store(false, Ordering::SeqCst);
            model_loaded.store(false, Ordering::SeqCst);
            generating.store(false, Ordering::SeqCst);
            training.store(false, Ordering::SeqCst);
            loading.store(false, Ordering::SeqCst);
            tx.send(BridgeEvent::ConnectionLost(
                "Python bridge process exited.".to_string()
            )).ok();
        });
    }

    /// Disconnect from the Python bridge server.
    pub fn disconnect(&self) {
        let handle_holder = self.child_handle.clone();
        let stdin_holder = self.child_stdin.clone();
        let connected = self.connected.clone();
        let connecting = self.connecting.clone();
        let model_loaded = self.model_loaded.clone();
        let generating = self.generating.clone();
        let training = self.training.clone();
        let loading = self.loading.clone();
        let native_training_handle = self.native_training_handle.clone();
        let native_training_active = self.native_training_active.clone();
        let native_training_stop_requested = self.native_training_stop_requested.clone();

        self.runtime.spawn(async move {
            // Drop stdin to signal EOF
            {
                let mut holder = stdin_holder.lock().await;
                *holder = None;
            }
            // Kill the child process
            {
                let mut holder = handle_holder.lock().await;
                if let Some(mut child) = holder.take() {
                    let _ = child.kill().await;
                }
            }
            {
                let mut holder = native_training_handle.lock().await;
                if let Some(child) = holder.as_mut() {
                    native_training_stop_requested.store(true, Ordering::SeqCst);
                    let _ = child.kill().await;
                }
                *holder = None;
            }
            connected.store(false, Ordering::SeqCst);
            connecting.store(false, Ordering::SeqCst);
            model_loaded.store(false, Ordering::SeqCst);
            generating.store(false, Ordering::SeqCst);
            training.store(false, Ordering::SeqCst);
            native_training_active.store(false, Ordering::SeqCst);
            loading.store(false, Ordering::SeqCst);
        });
    }

    /// Send an RPC request to the Python subprocess.
    fn send_rpc(&self, method: &str, params: serde_json::Value) {
        let stdin_holder = self.child_stdin.clone();
        let tx = self.event_tx.clone();
        let method = method.to_string();

        self.runtime.spawn(async move {
            let mut holder = stdin_holder.lock().await;
            if let Some(ref mut stdin) = *holder {
                let request = RpcRequest {
                    method: method.clone(),
                    params,
                };
                let mut line = match serde_json::to_string(&request) {
                    Ok(s) => s,
                    Err(e) => {
                        tx.send(BridgeEvent::Error(format!("JSON serialize error: {}", e)))
                            .ok();
                        return;
                    }
                };
                line.push('\n');
                if let Err(e) = stdin.write_all(line.as_bytes()).await {
                    tx.send(BridgeEvent::Error(format!(
                        "Failed to send to backend: {}. Is the bridge connected?",
                        e
                    )))
                    .ok();
                }
                let _ = stdin.flush().await;
            } else {
                tx.send(BridgeEvent::Error(
                    "Not connected to backend. Connect via Settings first.".to_string(),
                ))
                .ok();
            }
        });
    }

    /// Try to receive pending events (non-blocking).
    pub fn poll_events(&self) -> Vec<BridgeEvent> {
        let mut events = Vec::new();
        if let Ok(mut rx) = self.event_rx.try_lock() {
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }
        }
        events
    }

    /// Load a model from the given directory path.
    pub fn load_model(&self, model_path: String, device: String) {
        if !self.connected.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Connect to the backend before loading a model.".to_string(),
                ))
                .ok();
            return;
        }
        if self.generating.load(Ordering::SeqCst) || self.training.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Stop generation or training before loading another model.".to_string(),
                ))
                .ok();
            return;
        }
        if self.loading.swap(true, Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Status(
                    "Model load already in progress.".to_string(),
                ))
                .ok();
            return;
        }
        self.event_tx
            .send(BridgeEvent::LoadProgress(LoadProgress {
                progress: 0.18,
                label: "Sending model load request".to_string(),
            }))
            .ok();

        self.send_rpc(
            "load_model",
            serde_json::json!({
                "model_path": model_path,
                "device": device,
                "cache_dir": crate::embedded::get_models_dir().to_string_lossy().to_string(),
            }),
        );
    }

    /// Persist current runtime LTM updates next to the loaded model.
    pub fn save_ltm_updates(&self) {
        self.send_rpc("save_ltm_updates", serde_json::json!({}));
    }

    /// Persist the active chat's tiny hierarchical runtime state.
    pub fn save_chat_runtime_state(&self, path: String) {
        self.send_rpc(
            "save_chat_runtime_state",
            serde_json::json!({
                "path": path,
            }),
        );
    }

    /// Restore a previously saved chat runtime state.
    pub fn load_chat_runtime_state(&self, path: String) {
        self.send_rpc(
            "load_chat_runtime_state",
            serde_json::json!({
                "path": path,
            }),
        );
    }

    /// Reset backend runtime state and save an empty snapshot for this chat.
    pub fn reset_chat_runtime_state(&self, path: String) {
        self.send_rpc(
            "reset_chat_runtime_state",
            serde_json::json!({
                "path": path,
            }),
        );
    }

    /// Send a chat message and stream tokens back.
    pub fn send_message(
        &self,
        message: String,
        params: SamplingParams,
        passive_learning: bool,
        passive_lr: f64,
    ) {
        if !self.connected.load(Ordering::SeqCst) || !self.model_loaded.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Connect and load a model before generating.".to_string(),
                ))
                .ok();
            self.event_tx.send(BridgeEvent::GenerationComplete).ok();
            return;
        }
        if self.loading.load(Ordering::SeqCst) || self.training.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Generation is unavailable while loading or training.".to_string(),
                ))
                .ok();
            self.event_tx.send(BridgeEvent::GenerationComplete).ok();
            return;
        }
        if self.generating.swap(true, Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Status(
                    "Generation is already in progress.".to_string(),
                ))
                .ok();
            return;
        }
        self.send_rpc(
            "generate",
            serde_json::json!({
                "message": message,
                "sampling": {
                    "temperature": params.temperature,
                    "top_k": params.top_k,
                    "top_p": params.top_p,
                    "repetition_penalty": params.repetition_penalty,
                    "max_new_tokens": params.max_new_tokens,
                    "cpu_threads": params.cpu_threads,
                },
                "online_learning": {
                    "passive_learning": passive_learning,
                    "passive_lr": passive_lr,
                }
            }),
        );
    }

    /// Set the CPU thread count used by the PyTorch backend.
    pub fn set_cpu_threads(&self, threads: u32) {
        self.send_rpc(
            "set_threads",
            serde_json::json!({
                "threads": threads,
            }),
        );
    }

    /// Start a training run using either the Python/PyTorch backend or the native Vulkan trainer.
    pub fn start_training(&self, config: TrainingConfig, model_path: String) {
        if !self.connected.load(Ordering::SeqCst) || !self.model_loaded.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Connect and load a model before training.".to_string(),
                ))
                .ok();
            self.event_tx
                .send(BridgeEvent::TrainingComplete {
                    status: "rejected".to_string(),
                })
                .ok();
            return;
        }
        if self.loading.load(Ordering::SeqCst) || self.generating.load(Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Error(
                    "Training is unavailable while loading or generating.".to_string(),
                ))
                .ok();
            self.event_tx
                .send(BridgeEvent::TrainingComplete {
                    status: "rejected".to_string(),
                })
                .ok();
            return;
        }
        if self.training.swap(true, Ordering::SeqCst) {
            self.event_tx
                .send(BridgeEvent::Status(
                    "Training is already in progress.".to_string(),
                ))
                .ok();
            return;
        }
        match config.backend {
            TrainingBackend::Pytorch => self.start_pytorch_training(config),
            TrainingBackend::Vulkan => self.start_vulkan_training(config, model_path),
        }
    }

    fn start_pytorch_training(&self, config: TrainingConfig) {
        self.send_rpc(
            "start_training",
            serde_json::json!({
                "data_path": config.data_path,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "min_lr": config.min_lr,
                "training_chunk_size": config.training_chunk_size,
                "full_sample_bptt": config.full_sample_bptt,
                "full_sample_activation_checkpointing": config.full_sample_activation_checkpointing,
                "full_sample_checkpoint_segment_size": config.full_sample_checkpoint_segment_size,
                "accumulation_steps": config.accumulation_steps,
                "grad_clip": config.grad_clip,
                "persist_state": config.persist_state,
                "amp": config.amp,
                "save_steps": config.save_steps,
                "out_dir": config.out_dir,
                "context_dim": config.context_dim,
                "h_hidden": config.h_hidden,
                "l_hidden": config.l_hidden,
                "persistent_dim": config.persistent_dim,
                "ltm_slots": config.ltm_slots,
                "ltm_key_dim": config.ltm_key_dim,
                "ltm_val_dim": config.ltm_val_dim,
                "ltm_topk": config.ltm_topk,
                "h_stride": config.h_stride,
                "max_h_steps": config.max_h_steps,
                "max_l_steps": config.max_l_steps,
                "max_length": config.max_length,
                "auto_max_length": config.auto_max_length,
            }),
        );
    }

    fn start_vulkan_training(&self, config: TrainingConfig, model_path: String) {
        let reject = |message: String| {
            self.training.store(false, Ordering::SeqCst);
            self.event_tx.send(BridgeEvent::Error(message)).ok();
            self.event_tx
                .send(BridgeEvent::TrainingComplete {
                    status: "rejected".to_string(),
                })
                .ok();
        };

        let Some(trainer) = find_vulkan_trainer() else {
            reject(
                "Native Vulkan trainer was not found. Build hierarchos-vulkan-train or install a Windows bundle that includes the vulkan runtime directory."
                    .to_string(),
            );
            return;
        };
        let model_dir = match resolve_vulkan_model_dir(&model_path) {
            Ok(path) => path,
            Err(err) => {
                reject(err);
                return;
            }
        };
        let dataset_path = PathBuf::from(config.data_path.trim());
        if !dataset_path.is_file() && !dataset_path.is_dir() {
            reject(format!(
                "Training data not found: {}",
                dataset_path.display()
            ));
            return;
        }
        if dataset_path.is_dir()
            && !["_SUCCESS", "tokens.bin", "index.safetensors"]
                .iter()
                .all(|name| dataset_path.join(name).is_file())
        {
            reject(format!(
                "Vulkan token-cache directory {} is incomplete or predates the cross-runtime index. Rebuild it with the current Hierarchos data pipeline so _SUCCESS, tokens.bin, and index.safetensors are present.",
                dataset_path.display()
            ));
            return;
        }
        let output_dir = PathBuf::from(config.out_dir.trim());
        if config.out_dir.trim().is_empty() {
            reject("Choose an output directory for native Vulkan checkpoints.".to_string());
            return;
        }
        if config.vulkan_exact_resume && !model_dir.join("training_state.json").is_file() {
            reject(format!(
                "Exact Vulkan resume requires training_state.json under {}. Disable exact resume to start a fresh optimizer/session from these weights.",
                model_dir.display()
            ));
            return;
        }
        let config = match resolve_vulkan_launch_config(&model_dir, &config) {
            Ok(config) => config,
            Err(err) => {
                reject(err);
                return;
            }
        };
        if output_dir == model_dir {
            reject(
                "Native Vulkan training output must be distinct from the loaded source package."
                    .to_string(),
            );
            return;
        }
        let device_indices = match parse_vulkan_device_indices(&config.vulkan_device_indices) {
            Ok(indices) => indices,
            Err(err) => {
                reject(err);
                return;
            }
        };
        if config.persist_state && device_indices.len() > 1 {
            reject(
                "Persistent recurrent state is currently single-device in the native Vulkan trainer. Select one Vulkan device or disable persisted state."
                    .to_string(),
            );
            return;
        }
        if config.persist_state && config.shuffle {
            reject(
                "Persistent recurrent state requires deterministic lane-contiguous ordering. Disable Vulkan dataset shuffling or turn off persisted state."
                    .to_string(),
            );
            return;
        }

        let mut command = Command::new(&trainer);
        if config.vulkan_exact_resume {
            command.arg("--resume-from-ckpt").arg(&model_dir);
        } else {
            command.arg("--model").arg(&model_dir);
        }
        command
            .arg("--dataset")
            .arg(&dataset_path)
            .arg("--output")
            .arg(&output_dir)
            .arg("--epochs")
            .arg(config.epochs.to_string())
            .arg("--batch-size")
            .arg(config.batch_size.to_string())
            .arg("--gradient-accumulation-steps")
            .arg(config.accumulation_steps.to_string())
            .arg("--lr")
            .arg(config.learning_rate.to_string())
            .arg("--min-lr")
            .arg(config.min_lr.to_string())
            .arg("--warmup-steps")
            .arg(config.warmup_steps.to_string())
            .arg("--warmup-ratio")
            .arg(config.warmup_ratio.to_string())
            .arg("--beta1")
            .arg(config.beta1.to_string())
            .arg("--beta2")
            .arg(config.beta2.to_string())
            .arg("--eps")
            .arg(config.eps.to_string())
            .arg("--weight-decay")
            .arg(config.weight_decay.to_string())
            .arg("--z-loss-weight")
            .arg(config.z_loss_weight.to_string())
            .arg("--ponder-loss-weight")
            .arg(config.ponder_loss_weight.to_string())
            .arg("--commitment-loss-weight")
            .arg(config.commitment_loss_weight.to_string())
            .arg("--max-ce-loss-for-backward")
            .arg(config.max_ce_loss_for_backward.to_string())
            .arg("--max-ponder-cost-for-backward")
            .arg(config.max_ponder_cost_for_backward.to_string())
            .arg("--max-commitment-cost-for-backward")
            .arg(config.max_commitment_cost_for_backward.to_string())
            .arg("--max-skipped-train-batches")
            .arg(config.max_skipped_train_batches.to_string())
            .arg("--seed")
            .arg(config.seed.to_string())
            .arg("--grad-clip")
            .arg(config.grad_clip.to_string())
            .arg("--save-steps")
            .arg(config.save_steps.to_string())
            .arg("--json-events")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        command.env(
            "HIERARCHOS_VULKAN_TRAINING_PRECISION",
            config.vulkan_precision.env_value(),
        );
        if config.disable_lr_schedule {
            command.arg("--disable-lr-schedule");
        }
        if config.vulkan_tbptt_enabled {
            command
                .arg("--tbptt-chunk-size")
                .arg(config.training_chunk_size.to_string());
        }
        if device_indices.len() == 1 {
            command
                .arg("--device-index")
                .arg(device_indices[0].to_string());
        } else {
            command.arg("--device-indices").arg(
                device_indices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if config.persist_state {
            command.arg("--persist-state");
        }
        if !config.shuffle {
            command.arg("--no-shuffle");
        }
        hide_backend_window(&mut command);

        let child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                reject(format!(
                    "Failed to start native Vulkan trainer {}: {err}",
                    trainer.display()
                ));
                return;
            }
        };

        let tx = self.event_tx.clone();
        let training = self.training.clone();
        let active = self.native_training_active.clone();
        let stop_requested = self.native_training_stop_requested.clone();
        let handle = self.native_training_handle.clone();
        stop_requested.store(false, Ordering::SeqCst);
        active.store(true, Ordering::SeqCst);
        tx.send(BridgeEvent::Status(format!(
            "Launching native Vulkan trainer with {} precision: {}",
            config.vulkan_precision.label(),
            trainer.display()
        )))
        .ok();

        self.runtime.spawn(async move {
            let mut child = child;
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            {
                let mut holder = handle.lock().await;
                *holder = Some(child);
            }

            if let Some(stdout) = stdout {
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stdout).lines();
                    while let Ok(Some(_line)) = reader.next_line().await {
                        // The trainer's final pretty JSON report remains available to CLI users.
                        // The GUI consumes compact structured events from stderr instead.
                    }
                });
            }
            if let Some(stderr) = stderr {
                let tx_stderr = tx.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stderr).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        if let Some(event) = parse_native_training_event(trimmed) {
                            tx_stderr.send(event).ok();
                        } else {
                            tx_stderr
                                .send(BridgeEvent::Status(format!("Vulkan: {trimmed}")))
                                .ok();
                        }
                    }
                });
            }

            let exit_status = loop {
                let result = {
                    let mut holder = handle.lock().await;
                    match holder.as_mut() {
                        Some(child) => child.try_wait(),
                        None => break None,
                    }
                };
                match result {
                    Ok(Some(status)) => break Some(Ok(status)),
                    Ok(None) => tokio::time::sleep(std::time::Duration::from_millis(100)).await,
                    Err(err) => break Some(Err(err)),
                }
            };

            let stopped = stop_requested.swap(false, Ordering::SeqCst);
            active.store(false, Ordering::SeqCst);
            training.store(false, Ordering::SeqCst);
            {
                let mut holder = handle.lock().await;
                *holder = None;
            }
            match exit_status {
                Some(Ok(status)) if stopped => {
                    tx.send(BridgeEvent::TrainingComplete {
                        status: "stopped".to_string(),
                    })
                    .ok();
                }
                Some(Ok(status)) if status.success() => {
                    tx.send(BridgeEvent::TrainingComplete {
                        status: "completed".to_string(),
                    })
                    .ok();
                }
                Some(Ok(status)) => {
                    tx.send(BridgeEvent::Error(format!(
                        "Native Vulkan trainer exited with status {status}."
                    )))
                    .ok();
                    tx.send(BridgeEvent::TrainingComplete {
                        status: "error".to_string(),
                    })
                    .ok();
                }
                Some(Err(err)) => {
                    tx.send(BridgeEvent::Error(format!(
                        "Failed to observe native Vulkan trainer: {err}"
                    )))
                    .ok();
                    tx.send(BridgeEvent::TrainingComplete {
                        status: "error".to_string(),
                    })
                    .ok();
                }
                None => {
                    tx.send(BridgeEvent::TrainingComplete {
                        status: if stopped { "stopped" } else { "error" }.to_string(),
                    })
                    .ok();
                }
            }
        });
    }

    /// Stop ongoing generation.
    pub fn stop_generation(&self) {
        self.send_rpc("stop_generation", serde_json::json!({}));
    }

    /// Stop ongoing training.
    pub fn stop_training(&self) {
        if self.native_training_active.load(Ordering::SeqCst) {
            let handle = self.native_training_handle.clone();
            let stop_requested = self.native_training_stop_requested.clone();
            let tx = self.event_tx.clone();
            stop_requested.store(true, Ordering::SeqCst);
            self.runtime.spawn(async move {
                let mut holder = handle.lock().await;
                if let Some(child) = holder.as_mut() {
                    if let Err(err) = child.kill().await {
                        tx.send(BridgeEvent::Error(format!(
                            "Failed to stop native Vulkan training: {err}"
                        )))
                        .ok();
                    }
                }
            });
        } else {
            self.send_rpc("stop_training", serde_json::json!({}));
        }
    }

    /// Request LTM memory snapshot.
    pub fn request_ltm_snapshot(&self) {
        self.send_rpc("get_ltm_snapshot", serde_json::json!({}));
    }

    /// Request model inspection data.
    pub fn request_model_info(&self) {
        self.send_rpc("get_model_info", serde_json::json!({}));
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded.load(Ordering::SeqCst)
    }

    pub fn is_generating(&self) -> bool {
        self.generating.load(Ordering::SeqCst)
    }

    pub fn is_training(&self) -> bool {
        self.training.load(Ordering::SeqCst)
    }

    pub fn is_loading(&self) -> bool {
        self.loading.load(Ordering::SeqCst)
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Send feedback for online learning.
    pub fn send_feedback(&self, positive: bool, learning_rate: f64) {
        self.send_rpc(
            "send_feedback",
            serde_json::json!({
                "positive": positive,
                "learning_rate": learning_rate,
            }),
        );
    }

    /// Execute a slash command.
    pub fn execute_command(&self, command: String) {
        self.send_rpc(
            "execute_command",
            serde_json::json!({
                "command": command,
            }),
        );
    }
}

// ── JSON Parsing Helpers ────────────────────────────────────────────────────

fn parse_nested_f32_vec(val: Option<&serde_json::Value>) -> Vec<Vec<f32>> {
    val.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|row| {
                    row.as_array().map(|r| {
                        r.iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect()
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_f32_vec(val: Option<&serde_json::Value>) -> Vec<f32> {
    val.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_i32_vec(val: Option<&serde_json::Value>) -> Vec<i32> {
    val.and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_i64().map(|i| i as i32))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_resume_rehydrates_vulkan_trajectory_but_keeps_runtime_choices() {
        let mut requested = TrainingConfig::default();
        requested.backend = TrainingBackend::Vulkan;
        requested.vulkan_exact_resume = true;
        requested.vulkan_precision = VulkanTrainingPrecision::Fp32;
        requested.vulkan_device_indices = "1,2".to_string();
        requested.data_path = "runtime-cache".to_string();
        requested.out_dir = "runtime-output".to_string();
        requested.epochs = 17;
        requested.save_steps = 23;
        requested.batch_size = 99;
        requested.accumulation_steps = 7;
        requested.learning_rate = 9.0e-4;
        requested.shuffle = true;

        let manifest = serde_json::json!({
            "training_precision_policy": "fp16-storage-parity",
            "training_session": {
                "effective_training_config": {
                    "batch_size": 4,
                    "gradient_accumulation_steps": 3,
                    "starting_lr": 0.0001,
                    "min_lr": 0.000001,
                    "warmup_steps": 12,
                    "warmup_ratio": 0.125,
                    "disable_lr_schedule": false,
                    "beta1": 0.8,
                    "beta2": 0.95,
                    "eps": 0.0000001,
                    "weight_decay": 0.2,
                    "z_loss_weight": 0.0002,
                    "ponder_loss_weight": 0.004,
                    "commitment_loss_weight": 0.6,
                    "max_ce_loss_for_backward": 12.0,
                    "max_ponder_cost_for_backward": 3.0,
                    "max_commitment_cost_for_backward": 1.5,
                    "max_skipped_train_batches": 8,
                    "seed": 42,
                    "shuffle": false,
                    "grad_clip": 0.75,
                    "tbptt_chunk_size": 128,
                    "persist_state": true
                }
            }
        });

        let resolved = resolve_vulkan_exact_resume_config_from_manifest(&manifest, &requested)
            .expect("exact resume configuration should hydrate");

        assert_eq!(
            resolved.vulkan_precision,
            VulkanTrainingPrecision::Fp16Parity
        );
        assert_eq!(resolved.batch_size, 4);
        assert_eq!(resolved.accumulation_steps, 3);
        assert_eq!(resolved.learning_rate, 1.0e-4);
        assert_eq!(resolved.warmup_steps, 12);
        assert_eq!(resolved.seed, 42);
        assert!(!resolved.shuffle);
        assert!(resolved.persist_state);
        assert!(resolved.vulkan_tbptt_enabled);
        assert_eq!(resolved.training_chunk_size, 128);

        assert_eq!(resolved.vulkan_device_indices, "1,2");
        assert_eq!(resolved.data_path, "runtime-cache");
        assert_eq!(resolved.out_dir, "runtime-output");
        assert_eq!(resolved.epochs, 17);
        assert_eq!(resolved.save_steps, 23);
    }

    #[test]
    fn exact_resume_restores_full_bptt_and_rejects_unknown_precision() {
        let requested = TrainingConfig::default();
        let full_bptt = serde_json::json!({
            "training_precision_policy": "fp16-storage-fp32-compute",
            "training_session": {
                "effective_training_config": {
                    "tbptt_chunk_size": null
                }
            }
        });
        let resolved = resolve_vulkan_exact_resume_config_from_manifest(&full_bptt, &requested)
            .expect("full-BPTT checkpoint should hydrate");
        assert!(!resolved.vulkan_tbptt_enabled);
        assert_eq!(
            resolved.vulkan_precision,
            VulkanTrainingPrecision::Fp16StorageFp32Compute
        );

        let unsupported = serde_json::json!({
            "training_precision_policy": "bf16-not-yet-supported",
            "training_session": {"effective_training_config": {}}
        });
        let error = resolve_vulkan_exact_resume_config_from_manifest(&unsupported, &requested)
            .expect_err("unsupported precision must be rejected");
        assert!(error.contains("unsupported Vulkan training precision"));
    }
}
