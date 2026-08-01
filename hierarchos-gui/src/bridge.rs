// bridge.rs — Python ↔ Rust communication via subprocess + JSON-RPC
//
// Launches a Python subprocess running the Hierarchos bridge server,
// communicates via stdin/stdout JSON messages, and streams results
// back to the UI thread through tokio channels.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

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

/// Training configuration sent to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub data_path: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub min_lr: f64,
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
            data_path: String::new(),
            epochs: 3,
            batch_size: 64,
            learning_rate: 1e-4,
            min_lr: 1e-6,
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
            connected.store(false, Ordering::SeqCst);
            connecting.store(false, Ordering::SeqCst);
            model_loaded.store(false, Ordering::SeqCst);
            generating.store(false, Ordering::SeqCst);
            training.store(false, Ordering::SeqCst);
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

    /// Start a training run.
    pub fn start_training(&self, config: TrainingConfig) {
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

    /// Stop ongoing generation.
    pub fn stop_generation(&self) {
        self.send_rpc("stop_generation", serde_json::json!({}));
    }

    /// Stop ongoing training.
    pub fn stop_training(&self) {
        self.send_rpc("stop_training", serde_json::json!({}));
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
