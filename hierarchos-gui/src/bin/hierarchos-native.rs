#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::{Duration, Instant};

use eframe::egui;
use hierarchos_inference::{HierarchosModel, RuntimeState, Sampler, SamplingConfig};
use tokenizers::Tokenizer;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
const NATIVE_VULKAN_EVENT_PREFIX: &str = "HIERARCHOS_EVENT ";

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Hierarchos Native — Rust Inference + Vulkan Training")
            .with_inner_size([1100.0, 760.0])
            .with_min_inner_size([760.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Hierarchos Native — Rust Inference + Vulkan Training",
        options,
        Box::new(|cc| Ok(Box::new(NativeApp::new(cc)))),
    )
}

struct NativeSession {
    model: HierarchosModel,
    tokenizer: Tokenizer,
    state: RuntimeState,
    eos_id: Option<u32>,
}

impl NativeSession {
    fn load(model_dir: &Path) -> Result<Self, String> {
        let model = HierarchosModel::load(model_dir)
            .map_err(|e| format!("Could not load Rust model: {e}"))?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(format!(
                "Missing {}. Add the local tokenizer.json to the native SafeTensors model package before loading it.",
                tokenizer_path.display()
            ));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Could not load tokenizer.json: {e}"))?;
        if tokenizer.get_vocab_size(true) != model.config().vocab_size {
            return Err(format!(
                "Tokenizer/model vocabulary mismatch: tokenizer has {} entries but model expects {}.",
                tokenizer.get_vocab_size(true),
                model.config().vocab_size
            ));
        }
        let eos_id = tokenizer.token_to_id("<|endoftext|>");
        let state = model.new_state();
        Ok(Self {
            model,
            tokenizer,
            state,
            eos_id,
        })
    }

    fn reset(&mut self) {
        self.state = self.model.new_state();
    }
}

#[derive(Clone)]
struct UiSampling {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    max_new_tokens: usize,
}

impl Default for UiSampling {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.2,
            max_new_tokens: 256,
        }
    }
}

#[derive(Clone, Copy)]
enum Role {
    User,
    Assistant,
    System,
}

struct ChatMessage {
    role: Role,
    text: String,
}

enum WorkerEvent {
    Loaded {
        path: PathBuf,
        summary: String,
    },
    LoadFailed(String),
    AssistantText(String),
    GenerationDone {
        elapsed: Duration,
        generated_tokens: usize,
        stopped: bool,
    },
    GenerationFailed(String),
    VulkanDevicesLoaded(Vec<VulkanDeviceInfo>),
    VulkanDevicesFailed(String),
    TrainingStarted {
        device: String,
        total_steps: u32,
    },
    TrainingMetrics {
        epoch: u32,
        step: u32,
        total_steps: u32,
        loss: f64,
        lr: f64,
        tokens_per_sec: Option<f64>,
    },
    TrainingStatus(String),
    TrainingDone {
        stopped: bool,
        success: bool,
        message: String,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VulkanDeviceInfo {
    index: usize,
    name: String,
    device_type: String,
    #[serde(default)]
    device_uuid: String,
    #[serde(default)]
    external_buffer: Option<VulkanExternalCapability>,
    #[serde(default)]
    external_semaphore: Option<VulkanExternalCapability>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct VulkanExternalCapability {
    #[serde(default)]
    platform_bidirectional_candidate: bool,
    #[serde(default)]
    platform_handle: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativePanel {
    Chat,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeTrainingPrecision {
    Fp32,
    Fp16StorageFp32Compute,
    Fp16Parity,
    Fp16LmBackward,
}

impl NativeTrainingPrecision {
    const ALL: [Self; 4] = [
        Self::Fp32,
        Self::Fp16StorageFp32Compute,
        Self::Fp16Parity,
        Self::Fp16LmBackward,
    ];

    fn label(self) -> &'static str {
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

#[derive(Clone)]
struct NativeTrainingConfig {
    dataset_path: String,
    output_dir: String,
    device_indices: String,
    epochs: u32,
    batch_size: u32,
    accumulation_steps: u32,
    learning_rate: f64,
    min_lr: f64,
    warmup_steps: u64,
    warmup_ratio: f64,
    disable_lr_schedule: bool,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    z_loss_weight: f32,
    ponder_loss_weight: f32,
    commitment_loss_weight: f32,
    max_ce_loss_for_backward: f32,
    max_ponder_cost_for_backward: f32,
    max_commitment_cost_for_backward: f32,
    max_skipped_train_batches: u64,
    seed: u64,
    shuffle: bool,
    grad_clip: f32,
    save_steps: u32,
    tbptt_chunk_size: u32,
    tbptt_enabled: bool,
    persist_state: bool,
    precision: NativeTrainingPrecision,
    exact_resume: bool,
}

impl Default for NativeTrainingConfig {
    fn default() -> Self {
        Self {
            dataset_path: String::new(),
            output_dir: "./hierarchos-vulkan-model".to_string(),
            device_indices: "0".to_string(),
            epochs: 3,
            batch_size: 1,
            accumulation_steps: 1,
            learning_rate: 1.0e-4,
            min_lr: 1.0e-6,
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
            grad_clip: 1.0,
            save_steps: 0,
            tbptt_chunk_size: 256,
            tbptt_enabled: true,
            persist_state: false,
            precision: NativeTrainingPrecision::Fp32,
            exact_resume: false,
        }
    }
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

fn exact_resume_training_config(
    package_dir: &Path,
    requested: &NativeTrainingConfig,
) -> Result<(NativeTrainingConfig, u64, u64), String> {
    let manifest_path = package_dir.join("training_state.json");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|error| format!("Could not read {}: {error}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Could not decode {}: {error}", manifest_path.display()))?;
    exact_resume_training_config_from_manifest(&manifest, requested)
}

fn exact_resume_training_config_from_manifest(
    manifest: &serde_json::Value,
    requested: &NativeTrainingConfig,
) -> Result<(NativeTrainingConfig, u64, u64), String> {
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
    let completed_epoch = session
        .get("completed_epoch")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let mid_epoch_step = session
        .get("mid_epoch_step")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);

    let precision_label = manifest
        .get("training_precision_policy")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            "Exact resume checkpoint is missing training_precision_policy.".to_string()
        })?;

    // Start from the user's runtime-only choices (dataset path, output path,
    // target epochs, save cadence, and device topology), then reconstruct every
    // trajectory-defining field that the native trainer binds into run_identity.
    let mut resolved = requested.clone();
    resolved.precision = NativeTrainingPrecision::from_manifest_label(precision_label)?;

    if let Some(value) = config_json_u64(config, "batch_size")? {
        resolved.batch_size = u32::try_from(value)
            .map_err(|_| "Checkpoint batch_size exceeds the native GUI range.".to_string())?;
    }
    if let Some(value) = config_json_u64(config, "gradient_accumulation_steps")? {
        resolved.accumulation_steps = u32::try_from(value).map_err(|_| {
            "Checkpoint gradient_accumulation_steps exceeds the native GUI range.".to_string()
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
        Some(serde_json::Value::Null) => resolved.tbptt_enabled = false,
        Some(value) => {
            let value = value.as_u64().ok_or_else(|| {
                "Checkpoint training field \"tbptt_chunk_size\" must be an unsigned integer or null."
                    .to_string()
            })?;
            resolved.tbptt_chunk_size = u32::try_from(value).map_err(|_| {
                "Checkpoint tbptt_chunk_size exceeds the native GUI range.".to_string()
            })?;
            resolved.tbptt_enabled = true;
        }
    }
    if let Some(value) = config_json_bool(config, "persist_state")? {
        resolved.persist_state = value;
    }

    Ok((resolved, completed_epoch, mid_epoch_step))
}

fn platform_executable(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

fn find_vulkan_executable(binary: &str) -> Option<PathBuf> {
    let name = platform_executable(binary);
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

fn find_vulkan_trainer() -> Option<PathBuf> {
    find_vulkan_executable("hierarchos-vulkan-train")
}

fn find_vulkan_device_probe() -> Option<PathBuf> {
    find_vulkan_executable("hierarchos-vulkan-devices")
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

fn selected_vulkan_device(raw: &str, index: usize) -> bool {
    raw.split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .any(|selected| selected == index)
}

fn set_vulkan_device_selected(raw: &mut String, index: usize, selected: bool) {
    let mut indices = raw
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    if selected {
        if !indices.contains(&index) {
            indices.push(index);
        }
    } else {
        indices.retain(|candidate| *candidate != index);
    }
    indices.sort_unstable();
    *raw = indices
        .into_iter()
        .map(|candidate| candidate.to_string())
        .collect::<Vec<_>>()
        .join(",");
}

fn hide_backend_window(command: &mut Command) {
    #[cfg(windows)]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

fn parse_native_training_event(line: &str) -> Option<WorkerEvent> {
    let payload = line.strip_prefix(NATIVE_VULKAN_EVENT_PREFIX)?;
    let msg: serde_json::Value = serde_json::from_str(payload).ok()?;
    match msg.get("event").and_then(|value| value.as_str())? {
        "training_started" => Some(WorkerEvent::TrainingStarted {
            device: msg
                .get("device")
                .and_then(|value| value.as_str())
                .unwrap_or("Vulkan")
                .to_string(),
            total_steps: msg
                .get("total_steps")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                .min(u64::from(u32::MAX)) as u32,
        }),
        "training_metrics" => Some(WorkerEvent::TrainingMetrics {
            epoch: msg
                .get("epoch")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                .min(u64::from(u32::MAX)) as u32,
            step: msg
                .get("step")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                .min(u64::from(u32::MAX)) as u32,
            total_steps: msg
                .get("total_steps")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                .min(u64::from(u32::MAX)) as u32,
            loss: msg
                .get("loss")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            lr: msg
                .get("lr")
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0),
            tokens_per_sec: msg.get("tokens_per_sec").and_then(|value| value.as_f64()),
        }),
        _ => None,
    }
}

struct NativeApp {
    session: Arc<Mutex<Option<NativeSession>>>,
    event_tx: mpsc::Sender<WorkerEvent>,
    event_rx: mpsc::Receiver<WorkerEvent>,
    stop_flag: Arc<AtomicBool>,
    training_stop_flag: Arc<AtomicBool>,
    vulkan_devices: Vec<VulkanDeviceInfo>,
    vulkan_devices_loading: bool,
    vulkan_devices_error: Option<String>,
    active_panel: NativePanel,
    model_path: String,
    prompt: String,
    messages: Vec<ChatMessage>,
    sampling: UiSampling,
    carry_chat_state: bool,
    loading: bool,
    generating: bool,
    training: bool,
    training_config: NativeTrainingConfig,
    training_epoch: u32,
    training_step: u32,
    training_total_steps: u32,
    training_loss: Option<f64>,
    training_lr: Option<f64>,
    training_tps: Option<f64>,
    training_log: Vec<String>,
    status: String,
    model_summary: Option<String>,
    tokens_per_second: Option<f64>,
}

impl NativeApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let (event_tx, event_rx) = mpsc::channel();
        let mut app = Self {
            session: Arc::new(Mutex::new(None)),
            event_tx,
            event_rx,
            stop_flag: Arc::new(AtomicBool::new(false)),
            training_stop_flag: Arc::new(AtomicBool::new(false)),
            vulkan_devices: Vec::new(),
            vulkan_devices_loading: false,
            vulkan_devices_error: None,
            active_panel: NativePanel::Chat,
            model_path: String::new(),
            prompt: String::new(),
            messages: vec![ChatMessage {
                role: Role::System,
                text: "Pure-Rust FP32 preview. Load a coherent-v9 Rust model package to begin."
                    .to_string(),
            }],
            sampling: UiSampling::default(),
            carry_chat_state: false,
            loading: false,
            generating: false,
            training: false,
            training_config: NativeTrainingConfig::default(),
            training_epoch: 0,
            training_step: 0,
            training_total_steps: 0,
            training_loss: None,
            training_lr: None,
            training_tps: None,
            training_log: Vec::new(),
            status: "Ready — no Python runtime is used by this executable.".to_string(),
            model_summary: None,
            tokens_per_second: None,
        };
        app.refresh_vulkan_devices();
        app
    }

    fn poll_worker(&mut self, ctx: &egui::Context) {
        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                WorkerEvent::Loaded { path, summary } => {
                    self.loading = false;
                    self.model_path = path.display().to_string();
                    self.model_summary = Some(summary.clone());
                    self.status = format!("Loaded {summary}");
                    self.messages.push(ChatMessage {
                        role: Role::System,
                        text: "Model loaded. Recurrent chat state starts fresh.".to_string(),
                    });
                }
                WorkerEvent::LoadFailed(error) => {
                    self.loading = false;
                    self.status = error.clone();
                    self.messages.push(ChatMessage {
                        role: Role::System,
                        text: error,
                    });
                }
                WorkerEvent::AssistantText(text) => {
                    if let Some(last) = self.messages.last_mut() {
                        if matches!(last.role, Role::Assistant) {
                            last.text = text;
                        }
                    }
                }
                WorkerEvent::GenerationDone {
                    elapsed,
                    generated_tokens,
                    stopped,
                } => {
                    self.generating = false;
                    self.tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
                        Some(generated_tokens as f64 / elapsed.as_secs_f64())
                    } else {
                        None
                    };
                    self.status = if stopped {
                        format!("Stopped after {generated_tokens} tokens")
                    } else {
                        format!("Generation complete: {generated_tokens} tokens")
                    };
                }
                WorkerEvent::GenerationFailed(error) => {
                    self.generating = false;
                    self.status = error.clone();
                    self.messages.push(ChatMessage {
                        role: Role::System,
                        text: error,
                    });
                }
                WorkerEvent::VulkanDevicesLoaded(devices) => {
                    self.vulkan_devices_loading = false;
                    self.vulkan_devices_error = None;
                    if devices.is_empty() {
                        self.status =
                            "No Vulkan compute adapters were reported by the native runtime."
                                .to_string();
                    } else {
                        self.status = format!(
                            "Vulkan runtime ready: {} compute adapter{} detected.",
                            devices.len(),
                            if devices.len() == 1 { "" } else { "s" }
                        );
                    }
                    self.vulkan_devices = devices;
                }
                WorkerEvent::VulkanDevicesFailed(error) => {
                    self.vulkan_devices_loading = false;
                    self.vulkan_devices_error = Some(error.clone());
                    self.status = error;
                }
                WorkerEvent::TrainingStarted {
                    device,
                    total_steps,
                } => {
                    self.training_total_steps = total_steps;
                    self.status = format!("Native Vulkan training on {device}");
                    self.push_training_log(format!(
                        "Training started on {device}; {total_steps} batches scheduled."
                    ));
                }
                WorkerEvent::TrainingMetrics {
                    epoch,
                    step,
                    total_steps,
                    loss,
                    lr,
                    tokens_per_sec,
                } => {
                    self.training_epoch = epoch;
                    self.training_step = step;
                    self.training_total_steps = total_steps;
                    self.training_loss = Some(loss);
                    self.training_lr = Some(lr);
                    self.training_tps = tokens_per_sec;
                }
                WorkerEvent::TrainingStatus(message) => {
                    self.push_training_log(message);
                }
                WorkerEvent::TrainingDone {
                    stopped,
                    success,
                    message,
                } => {
                    self.training = false;
                    self.training_stop_flag.store(false, Ordering::SeqCst);
                    self.status = message.clone();
                    self.push_training_log(message);
                    if success && !stopped {
                        self.push_training_log(format!(
                            "Checkpoint package is ready under {}.",
                            self.training_config.output_dir
                        ));
                    }
                }
            }
            ctx.request_repaint();
        }
    }

    fn push_training_log(&mut self, message: String) {
        self.training_log.push(message);
        if self.training_log.len() > 500 {
            let overflow = self.training_log.len() - 500;
            self.training_log.drain(0..overflow);
        }
    }

    fn refresh_vulkan_devices(&mut self) {
        if self.vulkan_devices_loading || self.training {
            return;
        }
        let Some(probe) = find_vulkan_device_probe() else {
            let message = "Native Vulkan device probe was not found. Build hierarchos-vulkan-devices or use a release bundle that includes the vulkan runtime directory.".to_string();
            self.vulkan_devices_error = Some(message.clone());
            self.status = message;
            return;
        };
        self.vulkan_devices_loading = true;
        self.vulkan_devices_error = None;
        let tx = self.event_tx.clone();
        std::thread::spawn(move || {
            let mut command = Command::new(&probe);
            command.stdout(Stdio::piped()).stderr(Stdio::piped());
            hide_backend_window(&mut command);
            let result = command.output().map_err(|error| {
                format!(
                    "Failed to run Vulkan device probe {}: {error}",
                    probe.display()
                )
            });
            let event = match result {
                Ok(output) if output.status.success() => {
                    match serde_json::from_slice::<Vec<VulkanDeviceInfo>>(&output.stdout) {
                        Ok(devices) => WorkerEvent::VulkanDevicesLoaded(devices),
                        Err(error) => WorkerEvent::VulkanDevicesFailed(format!(
                            "Vulkan device probe returned invalid JSON: {error}"
                        )),
                    }
                }
                Ok(output) => {
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
                    WorkerEvent::VulkanDevicesFailed(if stderr.is_empty() {
                        format!("Vulkan device probe exited with status {}.", output.status)
                    } else {
                        format!("Vulkan device probe failed: {stderr}")
                    })
                }
                Err(error) => WorkerEvent::VulkanDevicesFailed(error),
            };
            let _ = tx.send(event);
        });
    }

    fn load_model(&mut self, path: PathBuf) {
        if self.loading || self.generating || self.training {
            return;
        }
        self.loading = true;
        self.status = format!("Loading {} ...", path.display());
        let session = self.session.clone();
        let tx = self.event_tx.clone();
        std::thread::spawn(move || match NativeSession::load(&path) {
            Ok(loaded) => {
                let cfg = loaded.model.config();
                let summary = format!(
                    "{} | {}d | vocab {} | H/L {}/{} | CPU FP32",
                    cfg.architecture_revision,
                    cfg.context_dim,
                    cfg.vocab_size,
                    cfg.h_hidden,
                    cfg.l_hidden
                );
                match session.lock() {
                    Ok(mut guard) => {
                        *guard = Some(loaded);
                        let _ = tx.send(WorkerEvent::Loaded { path, summary });
                    }
                    Err(_) => {
                        let _ = tx.send(WorkerEvent::LoadFailed(
                            "Native model session lock was poisoned.".to_string(),
                        ));
                    }
                }
            }
            Err(error) => {
                let _ = tx.send(WorkerEvent::LoadFailed(error));
            }
        });
    }

    fn reset_context(&mut self) {
        if self.generating || self.training {
            return;
        }
        match self.session.lock() {
            Ok(mut guard) => {
                if let Some(session) = guard.as_mut() {
                    session.reset();
                    self.messages.clear();
                    self.messages.push(ChatMessage {
                        role: Role::System,
                        text: "Chat and recurrent model state reset.".to_string(),
                    });
                    self.status = "Context reset".to_string();
                }
            }
            Err(_) => self.status = "Native model session lock was poisoned.".to_string(),
        }
    }

    fn start_generation(&mut self) {
        let text = self.prompt.trim().to_string();
        if text.is_empty() || self.loading || self.generating || self.training {
            return;
        }
        let has_model = self
            .session
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        if !has_model {
            self.status = "Load a native Rust model package first.".to_string();
            return;
        }

        self.prompt.clear();
        self.messages.push(ChatMessage {
            role: Role::User,
            text: text.clone(),
        });
        self.messages.push(ChatMessage {
            role: Role::Assistant,
            text: String::new(),
        });
        self.generating = true;
        self.stop_flag.store(false, Ordering::SeqCst);
        self.status = "Generating with native FP32 runtime ...".to_string();

        let session_holder = self.session.clone();
        let tx = self.event_tx.clone();
        let stop = self.stop_flag.clone();
        let sampling = self.sampling.clone();
        let carry_chat_state = self.carry_chat_state;
        std::thread::spawn(move || {
            let started = Instant::now();
            let result = (|| -> Result<(usize, bool), String> {
                let mut guard = session_holder
                    .lock()
                    .map_err(|_| "Native model session lock was poisoned.".to_string())?;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| "Model was unloaded before generation began.".to_string())?;

                // Match the canonical Python chat default: each user turn is an
                // independent supervised-format sample unless recurrent carry is
                // explicitly enabled.  Keeping this opt-in avoids silently
                // changing the coherent-v9 learned inference contract.
                if !carry_chat_state {
                    session.reset();
                }

                let formatted = format!("User: {}\n\nAssistant: ", text.trim());
                let encoding = session
                    .tokenizer
                    .encode(formatted, true)
                    .map_err(|e| format!("Tokenizer encode failed: {e}"))?;
                let prompt_ids = encoding.get_ids();
                if prompt_ids.is_empty() {
                    return Err("Tokenizer produced an empty prompt.".to_string());
                }

                let eos_id = session.eos_id;
                let mut logits = session
                    .model
                    .prefill_last(prompt_ids, &mut session.state)
                    .map_err(|e| format!("Native prefill failed: {e}"))?;
                let mut sampler = Sampler::new(SamplingConfig {
                    temperature: sampling.temperature,
                    top_k: sampling.top_k,
                    top_p: sampling.top_p,
                    repetition_penalty: sampling.repetition_penalty,
                    seed: 0,
                });
                let mut response_ids = Vec::with_capacity(sampling.max_new_tokens);
                let mut stopped = false;

                for _ in 0..sampling.max_new_tokens {
                    if stop.load(Ordering::SeqCst) {
                        stopped = true;
                        break;
                    }
                    // Apply repetition penalty across the full recurrent context, not
                    // just tokens generated during this one assistant turn.
                    let token = sampler.sample(&logits, session.state.history());
                    let terminal = eos_id.is_some_and(|eos| eos == token);
                    if !terminal {
                        response_ids.push(token);
                        // Decode the complete generated prefix. Byte-level/BPE
                        // tokenizers can split one Unicode scalar across tokens, so
                        // decoding each token independently can emit replacement
                        // characters or otherwise corrupt the visible stream.
                        let text = session
                            .tokenizer
                            .decode(&response_ids, false)
                            .map_err(|e| format!("Tokenizer decode failed: {e}"))?;
                        let _ = tx.send(WorkerEvent::AssistantText(text));
                    }
                    logits = session
                        .model
                        .step(token, &mut session.state)
                        .map_err(|e| format!("Native decode step failed: {e}"))?;
                    if terminal {
                        break;
                    }
                }
                Ok((response_ids.len(), stopped))
            })();

            match result {
                Ok((generated_tokens, stopped)) => {
                    let _ = tx.send(WorkerEvent::GenerationDone {
                        elapsed: started.elapsed(),
                        generated_tokens,
                        stopped,
                    });
                }
                Err(error) => {
                    let _ = tx.send(WorkerEvent::GenerationFailed(error));
                }
            }
        });
    }

    fn start_vulkan_training(&mut self) {
        if self.loading || self.generating || self.training {
            return;
        }

        let has_model = self
            .session
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        if !has_model {
            self.status = "Load a native Rust model package before training.".to_string();
            return;
        }

        let model_dir = PathBuf::from(self.model_path.trim());
        if !model_dir.is_dir() || !model_dir.join("model.safetensors").is_file() {
            self.status = format!(
                "Native Vulkan training requires model.safetensors under {}.",
                model_dir.display()
            );
            return;
        }

        let mut config = self.training_config.clone();
        let resume_cursor = if config.exact_resume {
            if !model_dir.join("training_state.json").is_file() {
                self.status = format!(
                    "Exact Vulkan resume requires training_state.json under {}. Disable exact resume to start a fresh optimizer/session from these weights.",
                    model_dir.display()
                );
                return;
            }
            match exact_resume_training_config(&model_dir, &config) {
                Ok((resolved, completed_epoch, mid_epoch_step)) => {
                    config = resolved;
                    if u64::from(config.epochs) <= completed_epoch {
                        self.status = format!(
                            "Checkpoint already completed {completed_epoch} epoch(s). Set target epochs above {completed_epoch} to continue exact training."
                        );
                        return;
                    }
                    Some((completed_epoch, mid_epoch_step))
                }
                Err(error) => {
                    self.status = error;
                    return;
                }
            }
        } else {
            None
        };

        if config.persist_state {
            config.shuffle = false;
        }

        let dataset_path = PathBuf::from(config.dataset_path.trim());
        if !dataset_path.is_file() && !dataset_path.is_dir() {
            self.status = format!("Training data not found: {}", dataset_path.display());
            return;
        }
        if dataset_path.is_dir()
            && !["_SUCCESS", "tokens.bin", "index.safetensors"]
                .iter()
                .all(|name| dataset_path.join(name).is_file())
        {
            self.status = format!(
                "Vulkan token-cache directory {} is incomplete or predates the cross-runtime index. Rebuild it with the current Hierarchos data pipeline so _SUCCESS, tokens.bin, and index.safetensors are present.",
                dataset_path.display()
            );
            return;
        }

        if config.output_dir.trim().is_empty() {
            self.status = "Choose an output directory for Vulkan checkpoints.".to_string();
            return;
        }
        let output_dir = PathBuf::from(config.output_dir.trim());
        if output_dir == model_dir {
            self.status =
                "Training output must be distinct from the loaded source package.".to_string();
            return;
        }

        let device_indices = match parse_vulkan_device_indices(&config.device_indices) {
            Ok(indices) => indices,
            Err(error) => {
                self.status = error;
                return;
            }
        };
        if !self.vulkan_devices.is_empty() {
            if let Some(index) = device_indices.iter().copied().find(|index| {
                !self
                    .vulkan_devices
                    .iter()
                    .any(|device| device.index == *index)
            }) {
                self.status = format!(
                    "Vulkan adapter {index} is not present in the current device probe. Refresh adapters or select a detected device."
                );
                return;
            }
        }
        if config.persist_state && device_indices.len() > 1 {
            self.status = "Persistent recurrent state is currently single-device; select one Vulkan adapter or disable persisted state.".to_string();
            return;
        }

        let Some(trainer) = find_vulkan_trainer() else {
            self.status = "Native Vulkan trainer was not found. Build hierarchos-vulkan-train or use a release bundle that includes the vulkan runtime directory.".to_string();
            return;
        };

        let mut command = Command::new(&trainer);
        if config.exact_resume {
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
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        command.env(
            "HIERARCHOS_VULKAN_TRAINING_PRECISION",
            config.precision.env_value(),
        );
        if config.disable_lr_schedule {
            command.arg("--disable-lr-schedule");
        }
        if config.tbptt_enabled {
            command
                .arg("--tbptt-chunk-size")
                .arg(config.tbptt_chunk_size.to_string());
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

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                self.status = format!(
                    "Failed to start native Vulkan trainer {}: {error}",
                    trainer.display()
                );
                return;
            }
        };

        let stderr = child.stderr.take();
        let tx = self.event_tx.clone();
        let stop = self.training_stop_flag.clone();
        stop.store(false, Ordering::SeqCst);
        self.training = true;
        self.training_epoch = 0;
        self.training_step = 0;
        self.training_total_steps = 0;
        self.training_loss = None;
        self.training_lr = None;
        self.training_tps = None;
        self.training_log.clear();
        self.status = if let Some((completed_epoch, mid_epoch_step)) = resume_cursor {
            format!(
                "Resuming Vulkan trajectory from epoch {completed_epoch}, batch {mid_epoch_step} with {} precision: {}",
                config.precision.label(),
                trainer.display()
            )
        } else {
            format!(
                "Launching native Vulkan trainer with {} precision: {}",
                config.precision.label(),
                trainer.display()
            )
        };
        self.push_training_log(self.status.clone());

        std::thread::spawn(move || {
            let stderr_thread = stderr.map(|stderr| {
                let tx = tx.clone();
                std::thread::spawn(move || {
                    for line in BufReader::new(stderr).lines() {
                        let Ok(line) = line else {
                            break;
                        };
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        if let Some(event) = parse_native_training_event(trimmed) {
                            let _ = tx.send(event);
                        } else {
                            let _ =
                                tx.send(WorkerEvent::TrainingStatus(format!("Vulkan: {trimmed}")));
                        }
                    }
                })
            });

            let mut stopped = false;
            let exit_status = loop {
                if stop.load(Ordering::SeqCst) {
                    stopped = true;
                    let _ = child.kill();
                    break child.wait();
                }
                match child.try_wait() {
                    Ok(Some(status)) => break Ok(status),
                    Ok(None) => std::thread::sleep(Duration::from_millis(100)),
                    Err(error) => break Err(error),
                }
            };

            if let Some(stderr_thread) = stderr_thread {
                let _ = stderr_thread.join();
            }

            let (success, message) = match exit_status {
                Ok(status) if stopped => (false, "Vulkan training stopped safely.".to_string()),
                Ok(status) if status.success() => {
                    (true, "Native Vulkan training completed.".to_string())
                }
                Ok(status) => (
                    false,
                    format!("Native Vulkan trainer exited with status {status}."),
                ),
                Err(error) => (
                    false,
                    format!("Failed while observing native Vulkan trainer: {error}"),
                ),
            };
            let _ = tx.send(WorkerEvent::TrainingDone {
                stopped,
                success,
                message,
            });
        });
    }

    fn draw_top(&mut self, ui: &mut egui::Ui) {
        ui.heading("Hierarchos Native — Rust Inference + Vulkan Training");
        ui.label(
            "Pure-Rust FP32 inference and direct native Vulkan training. No Python runtime is used by this executable's training path.",
        );
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.active_panel,
                NativePanel::Chat,
                "Chat / Inference",
            );
            ui.selectable_value(
                &mut self.active_panel,
                NativePanel::Training,
                "Vulkan Training",
            );
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Model package");
            ui.add(
                egui::TextEdit::singleline(&mut self.model_path)
                    .desired_width((ui.available_width() - 230.0).max(180.0))
                    .hint_text("Folder containing model.safetensors + tokenizer.json"),
            );
            if ui.button("Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.model_path = path.display().to_string();
                }
            }
            let can_load = !self.loading
                && !self.generating
                && !self.training
                && !self.model_path.trim().is_empty();
            if ui
                .add_enabled(can_load, egui::Button::new("Load"))
                .clicked()
            {
                self.load_model(PathBuf::from(self.model_path.trim()));
            }
        });
        if let Some(summary) = &self.model_summary {
            ui.label(summary);
        }
        ui.horizontal(|ui| {
            ui.label(&self.status);
            if self.loading {
                ui.spinner();
            }
            if let Some(tps) = self.tokens_per_second {
                ui.separator();
                ui.label(format!("{tps:.2} tok/s"));
            }
        });
    }

    fn draw_training(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.heading("Native Vulkan Training");
                ui.label(
                    "The Vulkan runtime trains Hierarchos directly and exports FP32-master SafeTensors packages consumable by PyTorch CPU/CUDA and the native Rust inference engine.",
                );
                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label(egui::RichText::new("Training source").strong());
                    ui.label(if self.model_path.trim().is_empty() {
                        "No model package loaded.".to_string()
                    } else {
                        self.model_path.clone()
                    });
                    ui.small(
                        "Load a coherent Hierarchos package containing model.safetensors + tokenizer.json above. Final Vulkan checkpoints preserve package sidecars, including the tokenizer.",
                    );
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Dataset");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.training_config.dataset_path)
                            .desired_width((ui.available_width() - 180.0).max(180.0))
                            .hint_text("Token cache directory or tokenized JSONL"),
                    );
                    if ui.button("JSONL...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSONL", &["jsonl", "json"])
                            .pick_file()
                        {
                            self.training_config.dataset_path = path.display().to_string();
                        }
                    }
                    if ui.button("Cache...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.training_config.dataset_path = path.display().to_string();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Output ");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.training_config.output_dir)
                            .desired_width((ui.available_width() - 100.0).max(180.0))
                            .hint_text("New checkpoint package directory"),
                    );
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new().pick_folder() {
                            self.training_config.output_dir = path.display().to_string();
                        }
                    }
                });

                ui.add_space(8.0);
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Vulkan compute adapters").strong());
                        if ui
                            .add_enabled(
                                !self.vulkan_devices_loading && !self.training,
                                egui::Button::new("Refresh"),
                            )
                            .clicked()
                        {
                            self.refresh_vulkan_devices();
                        }
                        if self.vulkan_devices_loading {
                            ui.spinner();
                            ui.small("probing native Vulkan runtime...");
                        }
                    });
                    if let Some(error) = &self.vulkan_devices_error {
                        ui.colored_label(ui.visuals().warn_fg_color, error);
                        ui.small(
                            "Manual device indices remain available below for development builds where the probe binary is intentionally absent.",
                        );
                    } else if self.vulkan_devices.is_empty() && !self.vulkan_devices_loading {
                        ui.small("No Vulkan compute adapters detected yet.");
                    } else {
                        for device in self.vulkan_devices.clone() {
                            let mut selected = selected_vulkan_device(
                                &self.training_config.device_indices,
                                device.index,
                            );
                            let external_transport = device
                                .external_buffer
                                .as_ref()
                                .is_some_and(|capability| capability.platform_bidirectional_candidate)
                                && device
                                    .external_semaphore
                                    .as_ref()
                                    .is_some_and(|capability| {
                                        capability.platform_bidirectional_candidate
                                    });
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .checkbox(
                                        &mut selected,
                                        format!(
                                            "#{} {} ({})",
                                            device.index, device.name, device.device_type
                                        ),
                                    )
                                    .changed()
                                {
                                    set_vulkan_device_selected(
                                        &mut self.training_config.device_indices,
                                        device.index,
                                        selected,
                                    );
                                }
                                if !device.device_uuid.is_empty() {
                                    let short_uuid = device
                                        .device_uuid
                                        .chars()
                                        .take(12)
                                        .collect::<String>();
                                    ui.small(format!("uuid {short_uuid}"));
                                }
                                if external_transport {
                                    let handle = device
                                        .external_buffer
                                        .as_ref()
                                        .map(|capability| capability.platform_handle.as_str())
                                        .filter(|handle| !handle.is_empty())
                                        .unwrap_or("platform handle");
                                    ui.small(format!("• direct external transport candidate ({handle})"));
                                } else {
                                    ui.small("• staged/device-group transport selected by runtime");
                                }
                            });
                        }
                        ui.small(
                            "Select multiple adapters for synchronous Vulkan data-parallel training. The backend still performs its phase-aware VRAM and transport preflight before allocating the full graph.",
                        );
                    }
                });

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Training precision").strong());
                    ui.add_enabled_ui(!self.training_config.exact_resume, |ui| {
                        egui::ComboBox::from_id_salt("native_vulkan_training_precision")
                            .selected_text(self.training_config.precision.label())
                            .show_ui(ui, |ui| {
                                for precision in NativeTrainingPrecision::ALL {
                                    ui.selectable_value(
                                        &mut self.training_config.precision,
                                        precision,
                                        precision.label(),
                                    );
                                }
                            });
                    });
                    ui.small(
                        "All modes retain authoritative FP32 master weights and portable AdamW state; FP16 modes use rebuildable Vulkan execution mirrors.",
                    );
                });

                ui.add_space(8.0);
                egui::Grid::new("native_vulkan_training_grid")
                    .num_columns(4)
                    .spacing([16.0, 8.0])
                    .show(ui, |ui| {
                        ui.label("Vulkan devices");
                        ui.add(
                            egui::TextEdit::singleline(
                                &mut self.training_config.device_indices,
                            )
                            .desired_width(100.0)
                            .hint_text("0 or 0,1"),
                        )
                        .on_hover_text(
                            "Physical Vulkan adapter indices. Multiple indices enable synchronous native data-parallel training.",
                        );
                        ui.label("Epochs");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.epochs).range(1..=1000),
                        );
                        ui.end_row();

                        ui.label("Batch size");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.batch_size)
                                .range(1..=1024),
                        );
                        ui.label("Accumulation");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.accumulation_steps)
                                .range(1..=1024),
                        );
                        ui.end_row();

                        ui.label("Learning rate");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.learning_rate)
                                .speed(0.00001)
                                .range(0.0..=1.0)
                                .max_decimals(8),
                        );
                        ui.label("Min LR");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.min_lr)
                                .speed(0.000001)
                                .range(0.0..=1.0)
                                .max_decimals(9),
                        );
                        ui.end_row();

                        ui.label("Grad clip");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.grad_clip)
                                .speed(0.1)
                                .range(0.0..=100.0),
                        );
                        ui.label("TBPTT chunk");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.training_config.tbptt_enabled, "enabled");
                            ui.add_enabled(
                                self.training_config.tbptt_enabled,
                                egui::DragValue::new(&mut self.training_config.tbptt_chunk_size)
                                    .range(1..=32768),
                            );
                        });
                        ui.end_row();

                        ui.label("Save every N batches");
                        ui.add(
                            egui::DragValue::new(&mut self.training_config.save_steps)
                                .range(0..=1_000_000),
                        );
                        ui.label("Persistent state");
                        ui.checkbox(&mut self.training_config.persist_state, "Carry recurrence")
                            .on_hover_text(
                                "Carries recurrent/context/ROSA state across consecutive batches. This requires a single Vulkan device and deterministic no-shuffle ordering.",
                            );
                        ui.end_row();
                    });

                egui::CollapsingHeader::new("PyTorch-parity optimizer & objective controls")
                    .id_salt("native_vulkan_training_advanced")
                    .show(ui, |ui| {
                        ui.small(
                            "These values participate in the portable training trajectory. Exact resume reloads them from training_state.json; fresh runs pass them directly to the all-Vulkan trainer.",
                        );
                        egui::Grid::new("native_vulkan_training_advanced_grid")
                            .num_columns(4)
                            .spacing([16.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Warmup steps");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.warmup_steps)
                                        .range(0..=10_000_000),
                                );
                                ui.label("Warmup ratio");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.warmup_ratio)
                                        .speed(0.01)
                                        .range(0.0..=1.0),
                                );
                                ui.end_row();

                                ui.label("Adam beta1");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.beta1)
                                        .speed(0.001)
                                        .range(0.0..=0.999999),
                                );
                                ui.label("Adam beta2");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.beta2)
                                        .speed(0.0001)
                                        .range(0.0..=0.9999999),
                                );
                                ui.end_row();

                                ui.label("Adam eps");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.eps)
                                        .speed(1.0e-9)
                                        .range(1.0e-12..=1.0),
                                );
                                ui.label("Weight decay");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.weight_decay)
                                        .speed(0.001)
                                        .range(0.0..=10.0),
                                );
                                ui.end_row();

                                ui.label("Z-loss weight");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.z_loss_weight)
                                        .speed(0.00001)
                                        .range(0.0..=10.0),
                                );
                                ui.label("Ponder weight");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.ponder_loss_weight)
                                        .speed(0.0001)
                                        .range(0.0..=10.0),
                                );
                                ui.end_row();

                                ui.label("Commitment weight");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.commitment_loss_weight)
                                        .speed(0.01)
                                        .range(0.0..=100.0),
                                );
                                ui.label("CE backward cap");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.max_ce_loss_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=10_000.0),
                                );
                                ui.end_row();

                                ui.label("Ponder backward cap");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.max_ponder_cost_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=10_000.0),
                                );
                                ui.label("Commitment cap");
                                ui.add(
                                    egui::DragValue::new(&mut self.training_config.max_commitment_cost_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=10_000.0),
                                );
                                ui.end_row();

                                ui.label("Sampler seed");
                                ui.add(egui::DragValue::new(&mut self.training_config.seed));
                                ui.label("Skip budget");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut self.training_config.max_skipped_train_batches,
                                    )
                                    .range(0..=1_000_000),
                                );
                                ui.end_row();
                            });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.training_config.shuffle, "Shuffle each epoch")
                                .on_hover_text(
                                    "Persisted recurrent state forces deterministic no-shuffle ordering.",
                                );
                            ui.checkbox(
                                &mut self.training_config.disable_lr_schedule,
                                "Disable cosine LR schedule",
                            );
                        });
                    });

                ui.horizontal(|ui| {
                    let exact_resume_changed = ui.checkbox(
                        &mut self.training_config.exact_resume,
                        "Exact-resume loaded checkpoint",
                    )
                    .on_hover_text(
                        "Use --resume-from-ckpt so AdamW moments, pending gradients, scheduler/scaler, data cursor, recurrent replay state, precision, and all trajectory-defining numerical settings continue from the loaded package. Leave off to start a fresh optimizer/session from its weights.",
                    )
                    .changed();
                    if exact_resume_changed && self.training_config.exact_resume {
                        let model_dir = PathBuf::from(self.model_path.trim());
                        match exact_resume_training_config(&model_dir, &self.training_config) {
                            Ok((mut resolved, completed_epoch, mid_epoch_step)) => {
                                if u64::from(resolved.epochs) <= completed_epoch {
                                    if let Ok(next_epoch_target) =
                                        u32::try_from(completed_epoch.saturating_add(1))
                                    {
                                        resolved.epochs = next_epoch_target;
                                    }
                                }
                                self.training_config = resolved;
                                self.status = format!(
                                    "Loaded exact-resume trajectory: epoch {completed_epoch}, batch {mid_epoch_step}, {} precision.",
                                    self.training_config.precision.label()
                                );
                            }
                            Err(error) => self.status = error,
                        }
                    }
                    if self.training_config.exact_resume {
                        ui.label(
                            egui::RichText::new(
                                "checkpoint policy overrides trajectory-defining controls",
                            )
                                .color(ui.visuals().warn_fg_color),
                        );
                    }
                });

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if self.training {
                        if ui.button("Stop Vulkan Training").clicked() {
                            self.training_stop_flag.store(true, Ordering::SeqCst);
                            self.status =
                                "Stopping Vulkan trainer at the process boundary ...".to_string();
                        }
                        ui.spinner();
                    } else {
                        let can_start = !self.loading
                            && !self.generating
                            && self
                                .session
                                .lock()
                                .map(|guard| guard.is_some())
                                .unwrap_or(false)
                            && !self.training_config.dataset_path.trim().is_empty()
                            && !self.training_config.output_dir.trim().is_empty();
                        if ui
                            .add_enabled(can_start, egui::Button::new("Start Vulkan Training"))
                            .clicked()
                        {
                            self.start_vulkan_training();
                        }
                    }
                    ui.label(&self.status);
                });

                let trained_output = PathBuf::from(self.training_config.output_dir.trim());
                let output_is_inference_ready = trained_output.join("model.safetensors").is_file()
                    && trained_output.join("tokenizer.json").is_file();
                if ui
                    .add_enabled(
                        !self.training
                            && !self.loading
                            && !self.generating
                            && output_is_inference_ready,
                        egui::Button::new("Load trained output into Rust inference"),
                    )
                    .on_hover_text(
                        "Loads the Vulkan-written SafeTensors package directly into hierarchos-inference. The same model.safetensors is also the PyTorch CPU/CUDA interchange artifact.",
                    )
                    .clicked()
                {
                    self.active_panel = NativePanel::Chat;
                    self.model_path = trained_output.display().to_string();
                    self.load_model(trained_output);
                }

                if self.training_total_steps > 0 {
                    let progress =
                        self.training_step as f32 / self.training_total_steps.max(1) as f32;
                    ui.add(
                        egui::ProgressBar::new(progress.clamp(0.0, 1.0)).text(format!(
                            "batch {} / {}",
                            self.training_step, self.training_total_steps
                        )),
                    );
                }

                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Epoch: {}", self.training_epoch));
                    ui.separator();
                    ui.label(format!(
                        "Loss: {}",
                        self.training_loss
                            .map(|value| format!("{value:.6}"))
                            .unwrap_or_else(|| "—".to_string())
                    ));
                    ui.separator();
                    ui.label(format!(
                        "LR: {}",
                        self.training_lr
                            .map(|value| format!("{value:.8}"))
                            .unwrap_or_else(|| "—".to_string())
                    ));
                    ui.separator();
                    ui.label(format!(
                        "Throughput: {}",
                        self.training_tps
                            .map(|value| format!("{value:.1} tok/s"))
                            .unwrap_or_else(|| "—".to_string())
                    ));
                });

                ui.add_space(8.0);
                ui.group(|ui| {
                    ui.label(egui::RichText::new("Vulkan log").strong());
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            if self.training_log.is_empty() {
                                ui.small("No native training events yet.");
                            } else {
                                for line in &self.training_log {
                                    ui.small(line);
                                }
                            }
                        });
                });

                ui.add_space(8.0);
                ui.small(
                    "For PyTorch/Vulkan data parity, use a Hierarchos schema-v6 token-cache directory; the native trainer verifies its SafeTensors index, tokens.bin checksum, and canonical ordered-record identity before training. Legacy JSONL rows with input_ids remain supported. Periodic checkpoints and the final package use the same portable SafeTensors ABI as PyTorch and native Rust inference.",
                );
            });
    }

    fn draw_chat(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for message in &self.messages {
                    let (label, color) = match message.role {
                        Role::User => ("You", ui.visuals().hyperlink_color),
                        Role::Assistant => ("Hierarchos", ui.visuals().strong_text_color()),
                        Role::System => ("Runtime", ui.visuals().warn_fg_color),
                    };
                    ui.group(|ui| {
                        ui.label(egui::RichText::new(label).strong().color(color));
                        if message.text.is_empty() && matches!(message.role, Role::Assistant) {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("thinking...");
                            });
                        } else {
                            ui.label(&message.text);
                        }
                    });
                    ui.add_space(5.0);
                }
            });
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.sampling.temperature, 0.0..=2.0).text("temp"));
            ui.add(egui::Slider::new(&mut self.sampling.top_k, 0..=200).text("top-k"));
            ui.add(egui::Slider::new(&mut self.sampling.top_p, 0.01..=1.0).text("top-p"));
            ui.add(
                egui::Slider::new(&mut self.sampling.repetition_penalty, 0.5..=2.0).text("repeat"),
            );
            ui.add(
                egui::DragValue::new(&mut self.sampling.max_new_tokens)
                    .range(1..=4096)
                    .prefix("max "),
            );
            ui.checkbox(&mut self.carry_chat_state, "carry state")
                .on_hover_text(
                    "Carry recurrent/hierarchical state between user turns. Off matches the canonical Python chat default and training-format parity.",
                );
        });
        ui.horizontal(|ui| {
            let response = ui.add_sized(
                [ui.available_width() - 190.0, 70.0],
                egui::TextEdit::multiline(&mut self.prompt)
                    .hint_text("Message Hierarchos...")
                    .desired_rows(3),
            );
            ui.vertical(|ui| {
                let send_enabled =
                    !self.generating && !self.loading && !self.prompt.trim().is_empty();
                if ui
                    .add_enabled(send_enabled, egui::Button::new("Send"))
                    .clicked()
                {
                    self.start_generation();
                }
                if ui
                    .add_enabled(self.generating, egui::Button::new("Stop"))
                    .clicked()
                {
                    self.stop_flag.store(true, Ordering::SeqCst);
                    self.status = "Stopping after the current inference step ...".to_string();
                }
                if ui
                    .add_enabled(
                        !self.generating && !self.loading,
                        egui::Button::new("Reset chat"),
                    )
                    .clicked()
                {
                    self.reset_context();
                }
            });
            if response.has_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl)
            {
                self.start_generation();
            }
        });
        ui.small("Ctrl+Enter sends. Recurrent carry is opt-in; Vulkan training is available from the native Training tab.");
    }
}

impl eframe::App for NativeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker(ctx);
        if self.loading || self.generating || self.training {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        egui::TopBottomPanel::top("native_top").show(ctx, |ui| {
            ui.add_space(8.0);
            self.draw_top(ui);
            ui.add_space(8.0);
        });
        if self.active_panel == NativePanel::Chat {
            egui::TopBottomPanel::bottom("native_controls")
                .resizable(false)
                .show(ctx, |ui| {
                    ui.add_space(8.0);
                    self.draw_controls(ui);
                    ui.add_space(8.0);
                });
        }
        egui::CentralPanel::default().show(ctx, |ui| match self.active_panel {
            NativePanel::Chat => self.draw_chat(ui),
            NativePanel::Training => self.draw_training(ui),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vulkan_device_lists_and_rejects_ambiguous_ones() {
        assert_eq!(parse_vulkan_device_indices("0, 2").unwrap(), vec![0, 2]);
        assert!(parse_vulkan_device_indices("0,0").is_err());
        assert!(parse_vulkan_device_indices("0,").is_err());
        assert!(parse_vulkan_device_indices("gpu0").is_err());
    }

    #[test]
    fn vulkan_device_checkbox_selection_is_canonical_and_reversible() {
        let mut selected = "2,0".to_string();
        set_vulkan_device_selected(&mut selected, 1, true);
        assert_eq!(selected, "0,1,2");
        assert!(selected_vulkan_device(&selected, 1));
        set_vulkan_device_selected(&mut selected, 1, false);
        assert_eq!(selected, "0,2");
        assert!(!selected_vulkan_device(&selected, 1));
    }

    #[test]
    fn parses_vulkan_device_probe_payload() {
        let devices: Vec<VulkanDeviceInfo> = serde_json::from_str(
            r#"[{"index":0,"name":"GPU","device_type":"DISCRETE_GPU","device_uuid":"001122","external_buffer":{"platform_bidirectional_candidate":true,"platform_handle":"opaque-win32"},"external_semaphore":{"platform_bidirectional_candidate":true,"platform_handle":"opaque-win32"}}]"#,
        )
        .expect("device probe payload");
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[0].name, "GPU");
        assert!(
            devices[0]
                .external_buffer
                .as_ref()
                .expect("external buffer capability")
                .platform_bidirectional_candidate
        );
    }

    #[test]
    fn parses_native_training_metrics_event() {
        let event = parse_native_training_event(
            "HIERARCHOS_EVENT {\"event\":\"training_metrics\",\"epoch\":2,\"step\":9,\"total_steps\":12,\"loss\":1.25,\"lr\":0.0001,\"tokens_per_sec\":321.5}",
        )
        .expect("training event");
        match event {
            WorkerEvent::TrainingMetrics {
                epoch,
                step,
                total_steps,
                loss,
                lr,
                tokens_per_sec,
            } => {
                assert_eq!(epoch, 2);
                assert_eq!(step, 9);
                assert_eq!(total_steps, 12);
                assert_eq!(loss, 1.25);
                assert_eq!(lr, 0.0001);
                assert_eq!(tokens_per_sec, Some(321.5));
            }
            _ => panic!("unexpected event"),
        }
    }

    #[test]
    fn exact_resume_rehydrates_identity_bound_training_policy() {
        let mut requested = NativeTrainingConfig::default();
        requested.dataset_path = "dataset.jsonl".to_string();
        requested.output_dir = "next-output".to_string();
        requested.device_indices = "2,3".to_string();
        requested.epochs = 9;
        requested.save_steps = 17;
        requested.batch_size = 99;
        requested.precision = NativeTrainingPrecision::Fp32;

        let manifest = serde_json::json!({
            "training_precision_policy": "fp16-storage-parity",
            "training_session": {
                "completed_epoch": 4,
                "mid_epoch_step": 7,
                "effective_training_config": {
                    "batch_size": 3,
                    "gradient_accumulation_steps": 5,
                    "starting_lr": 0.0002,
                    "min_lr": 0.00001,
                    "warmup_steps": 12,
                    "warmup_ratio": 0.25,
                    "disable_lr_schedule": false,
                    "beta1": 0.8,
                    "beta2": 0.98,
                    "eps": 0.0000001,
                    "weight_decay": 0.02,
                    "z_loss_weight": 0.0003,
                    "ponder_loss_weight": 0.004,
                    "commitment_loss_weight": 0.6,
                    "max_ce_loss_for_backward": 0.0,
                    "max_ponder_cost_for_backward": 0.0,
                    "max_commitment_cost_for_backward": 1.5,
                    "max_skipped_train_batches": 2,
                    "seed": 1234,
                    "shuffle": false,
                    "grad_clip": 0.75,
                    "tbptt_chunk_size": 64,
                    "persist_state": true
                }
            }
        });

        let (resolved, completed_epoch, mid_epoch_step) =
            exact_resume_training_config_from_manifest(&manifest, &requested)
                .expect("exact resume policy");
        assert_eq!(completed_epoch, 4);
        assert_eq!(mid_epoch_step, 7);
        assert_eq!(resolved.batch_size, 3);
        assert_eq!(resolved.accumulation_steps, 5);
        assert_eq!(resolved.warmup_steps, 12);
        assert_eq!(resolved.seed, 1234);
        assert!(!resolved.shuffle);
        assert!(resolved.persist_state);
        assert!(resolved.tbptt_enabled);
        assert_eq!(resolved.tbptt_chunk_size, 64);
        assert_eq!(resolved.precision, NativeTrainingPrecision::Fp16Parity);
        assert_eq!(resolved.precision.env_value(), "fp16-storage-parity");

        // Runtime-only choices intentionally remain user-controlled on resume.
        assert_eq!(resolved.dataset_path, "dataset.jsonl");
        assert_eq!(resolved.output_dir, "next-output");
        assert_eq!(resolved.device_indices, "2,3");
        assert_eq!(resolved.epochs, 9);
        assert_eq!(resolved.save_steps, 17);
    }

    #[test]
    fn exact_resume_preserves_checkpoint_without_tbptt() {
        let requested = NativeTrainingConfig::default();
        let manifest = serde_json::json!({
            "training_precision_policy": "fp32",
            "training_session": {
                "completed_epoch": 0,
                "mid_epoch_step": 0,
                "effective_training_config": {
                    "tbptt_chunk_size": null,
                    "persist_state": false
                }
            }
        });
        let (resolved, _, _) = exact_resume_training_config_from_manifest(&manifest, &requested)
            .expect("checkpoint without TBPTT");
        assert!(!resolved.tbptt_enabled);
        assert_eq!(resolved.precision, NativeTrainingPrecision::Fp32);
    }
}
