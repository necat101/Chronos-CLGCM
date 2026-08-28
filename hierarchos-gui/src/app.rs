// app.rs — Core Application State & eframe::App Implementation
//
// Manages navigation, routes events from the Python bridge to panels,
// and renders the top bar, active panel, and status bar.

use egui::{self, Color32, CornerRadius as Rounding, RichText, Stroke};

use crate::bridge::{BridgeEvent, PythonBridge};
use crate::panels::chat::{draw_chat_panel, ChatState};
use crate::panels::hub::{draw_hub_panel, HubState};
use crate::panels::inspector::{draw_inspector_panel, InspectorState};
use crate::panels::memory::{draw_memory_panel, MemoryVisualizerState};
use crate::panels::settings::{draw_settings_panel, AppSettings};
use crate::panels::training::{draw_training_panel, TrainingState};
use crate::theme::{self, get_accent, HierarchosColors};
use crate::widgets::status_bar::{draw_status_bar, StatusBarInfo};

/// The active panel selection.
#[derive(PartialEq, Clone, Copy)]
pub enum Panel {
    Chat,
    Training,
    Inspector,
    Memory,
    Hub,
    Settings,
}

/// The main Hierarchos application.
pub struct HierarchosApp {
    // Navigation
    active_panel: Panel,

    // Backend
    bridge: PythonBridge,

    // Panel states
    chat_state: ChatState,
    training_state: TrainingState,
    inspector_state: InspectorState,
    memory_state: MemoryVisualizerState,
    hub_state: HubState,
    settings: AppSettings,

    // Global
    status_messages: Vec<String>,
    tokens_per_sec: Option<f64>,
    last_accent_idx: usize,
    close_dialog_open: bool,
    close_confirmed: bool,
    ltm_save_in_progress: bool,
    close_after_ltm_save: bool,
    ltm_save_error: Option<String>,
    load_progress: Option<LoadProgressState>,
}

struct LoadProgressState {
    progress: f32,
    label: String,
    hide_at: Option<f64>,
}

impl HierarchosApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Apply theme — apply_theme sets visuals/spacing, setup_fonts must come
        // LAST because it registers custom TextStyle::Name entries that would be
        // wiped if apply_theme ran again afterward.
        theme::apply_theme(&cc.egui_ctx);
        theme::setup_fonts(&cc.egui_ctx);

        // Track accent so we can detect changes
        let last_accent = theme::get_accent_index();

        Self {
            active_panel: Panel::Chat,
            bridge: PythonBridge::new(),
            chat_state: ChatState::default(),
            training_state: TrainingState::default(),
            inspector_state: InspectorState::default(),
            memory_state: MemoryVisualizerState::default(),
            hub_state: HubState::default(),
            settings: AppSettings::default(),
            status_messages: Vec::new(),
            tokens_per_sec: None,
            last_accent_idx: last_accent,
            close_dialog_open: false,
            close_confirmed: false,
            ltm_save_in_progress: false,
            close_after_ltm_save: false,
            ltm_save_error: None,
            load_progress: None,
        }
    }

    /// Process events from the Python bridge.
    fn process_bridge_events(&mut self, now: f64) {
        let events = self.bridge.poll_events();
        for event in events {
            match event {
                BridgeEvent::Token(token) => {
                    self.chat_state.on_token(token);
                }
                BridgeEvent::GenerationComplete => {
                    self.chat_state.on_generation_complete();
                    if self.bridge.is_connected()
                        && !self.bridge.is_training()
                        && !self.bridge.is_loading()
                    {
                        self.bridge
                            .save_chat_runtime_state(self.chat_state.runtime_state_path_string());
                    }
                }
                BridgeEvent::TrainingComplete { status } => {
                    self.training_state.is_training = false;
                    let message = match status.as_str() {
                        "completed" => "Training completed.".to_string(),
                        "stopped" => "Training stopped safely.".to_string(),
                        "rejected" => "Training request was rejected.".to_string(),
                        _ => "Training ended with an error.".to_string(),
                    };
                    self.training_state.log_messages.push(message);
                }
                BridgeEvent::TrainingMetrics {
                    epoch,
                    step,
                    total_steps,
                    loss,
                    lr,
                    ponder_cost,
                    commitment_cost,
                    tokens_per_sec,
                } => {
                    self.training_state.on_metrics(
                        epoch,
                        step,
                        total_steps,
                        loss,
                        lr,
                        ponder_cost,
                        commitment_cost,
                        tokens_per_sec,
                    );
                    if let Some(tps) = tokens_per_sec {
                        self.tokens_per_sec = Some(tps);
                    }
                }
                BridgeEvent::ModelLoaded(config) => {
                    self.training_state
                        .config
                        .sync_architecture_from_model(&config);
                    self.inspector_state.config = Some(config.clone());
                    let device_label = config
                        .device_label
                        .clone()
                        .unwrap_or_else(|| config.device.clone());
                    self.chat_state.on_status(format!(
                        "Model loaded: {:.1}M params | {} | {}",
                        config.total_params as f64 / 1e6,
                        device_label,
                        if config.is_quantized {
                            "Quantized"
                        } else {
                            "Full Precision"
                        }
                    ));
                    // Full per-parameter statistics require reductions over
                    // every weight tensor. Keep that cost on the Inspector's
                    // explicit Refresh action instead of every model load.
                    if self.chat_state.runtime_state_exists() {
                        self.bridge
                            .load_chat_runtime_state(self.chat_state.runtime_state_path_string());
                    } else {
                        self.bridge
                            .reset_chat_runtime_state(self.chat_state.runtime_state_path_string());
                    }
                }
                BridgeEvent::LoadProgress(progress) => {
                    let pct = progress.progress.clamp(0.0, 1.0);
                    self.load_progress = Some(LoadProgressState {
                        progress: pct,
                        label: progress.label,
                        hide_at: if pct >= 1.0 { Some(now + 1.2) } else { None },
                    });
                }
                BridgeEvent::ModelUnloaded => {
                    self.inspector_state.config = None;
                    self.inspector_state.inspection = None;
                    self.tokens_per_sec = None;
                    self.chat_state
                        .on_status("Previous model unloaded.".to_string());
                }
                BridgeEvent::LtmSnapshot {
                    fast_vals,
                    slow_vals,
                    timestamps,
                    sources,
                } => {
                    self.memory_state
                        .on_snapshot(fast_vals, slow_vals, timestamps, sources);
                }
                BridgeEvent::Status(msg) => {
                    self.status_messages.push(msg.clone());
                    if self.training_state.is_training {
                        self.training_state.log_messages.push(msg.clone());
                    }
                    self.chat_state.on_status(msg);
                }
                BridgeEvent::Error(err) => {
                    self.load_progress = None;
                    if self.ltm_save_in_progress {
                        self.ltm_save_in_progress = false;
                        self.close_after_ltm_save = false;
                        self.ltm_save_error = Some(err.clone());
                    }
                    self.chat_state.on_status(format!("❌ Error: {}", err));
                }
                BridgeEvent::ModelInfo(info) => {
                    self.inspector_state.inspection = Some(info);
                }
                BridgeEvent::LtmSaved(path) => {
                    self.ltm_save_in_progress = false;
                    self.ltm_save_error = None;
                    if path.is_empty() {
                        self.chat_state.on_status("LTM updates saved.".to_string());
                    } else {
                        self.chat_state
                            .on_status(format!("LTM updates saved to {}", path));
                    }
                }
                BridgeEvent::ConnectionLost(reason) => {
                    self.load_progress = None;
                    self.training_state.is_training = false;
                    self.chat_state.is_generating = false;
                    self.chat_state
                        .on_status(format!("⚠ Backend disconnected: {}", reason));
                }
            }
        }
    }
}

impl eframe::App for HierarchosApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = ctx.input(|i| i.time);
        // Process bridge events each frame
        self.process_bridge_events(now);

        if self
            .load_progress
            .as_ref()
            .and_then(|p| p.hide_at)
            .is_some_and(|hide_at| now >= hide_at)
        {
            self.load_progress = None;
        }

        if self.close_after_ltm_save && !self.ltm_save_in_progress && self.ltm_save_error.is_none()
        {
            self.close_confirmed = true;
            self.bridge.disconnect();
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        if ctx.input(|i| i.viewport().close_requested()) && !self.close_confirmed {
            let _ = self.chat_state.save_active_chat();
            if self.bridge.is_connected() {
                self.bridge
                    .save_chat_runtime_state(self.chat_state.runtime_state_path_string());
            }
            if self.bridge.is_model_loaded() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.close_dialog_open = true;
            } else {
                self.close_confirmed = true;
                self.bridge.disconnect();
            }
        }

        if self.settings.pending_load && self.bridge.is_connected() {
            self.bridge.load_model(
                self.settings.model_path.clone(),
                self.settings.device.clone(),
            );
            self.settings.pending_load = false;
        }

        if let Some(progress) = &mut self.load_progress {
            if progress.progress < 0.2
                && progress.hide_at.is_none()
                && self.bridge.is_connected()
                && !self.bridge.is_loading()
                && !self.settings.pending_load
            {
                progress.progress = 1.0;
                progress.label = "Backend ready".to_string();
                progress.hide_at = Some(now + 1.2);
            }
        }

        // Re-apply theme if accent color changed
        let current_accent = theme::get_accent_index();
        if current_accent != self.last_accent_idx {
            self.last_accent_idx = current_accent;
            theme::apply_theme(ctx);
        }

        // Top navigation bar
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            draw_top_bar(
                ui,
                &mut self.active_panel,
                &self.bridge,
                self.load_progress.as_ref(),
            );
        });

        // Bottom status bar
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            let info = StatusBarInfo {
                device: if self.bridge.is_model_loaded() {
                    self.inspector_state
                        .config
                        .as_ref()
                        .map(|c| c.device_label.clone().unwrap_or_else(|| c.device.clone()))
                        .unwrap_or_else(|| "CPU".to_string())
                } else {
                    "N/A".to_string()
                },
                model_status: if self.bridge.is_loading() {
                    "Loading model...".to_string()
                } else if self.bridge.is_model_loaded() {
                    if self.bridge.is_generating() {
                        "Generating...".to_string()
                    } else if self.bridge.is_training() {
                        "Training...".to_string()
                    } else {
                        "Ready".to_string()
                    }
                } else {
                    "No model loaded".to_string()
                },
                tokens_generated: self.chat_state.total_tokens,
                tokens_per_sec: self.tokens_per_sec,
                vram_total_mb: self
                    .inspector_state
                    .config
                    .as_ref()
                    .and_then(|c| c.vram_total_mb.map(|v| v as f64)),
                connected: self.bridge.is_connected(),
            };
            draw_status_bar(ui, &info);
        });

        // Main content area
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(HierarchosColors::BG_DARK)
                    .inner_margin(egui::Margin::same(16)),
            )
            .show(ctx, |ui| {
                match self.active_panel {
                    Panel::Chat => {
                        draw_chat_panel(ui, &mut self.chat_state, &self.bridge, &self.settings)
                    }
                    Panel::Training => draw_training_panel(
                        ui,
                        &mut self.training_state,
                        &self.bridge,
                        &self.settings.model_path,
                    ),
                    // Note: bridge.connect() is called from Settings panel when user loads a model
                    Panel::Inspector => {
                        draw_inspector_panel(ui, &mut self.inspector_state, &self.bridge)
                    }
                    Panel::Memory => draw_memory_panel(ui, &mut self.memory_state, &self.bridge),
                    Panel::Hub => {
                        draw_hub_panel(ui, &mut self.hub_state, &mut self.settings, &self.bridge)
                    }
                    Panel::Settings => draw_settings_panel(ui, &mut self.settings, &self.bridge),
                }
            });

        if self.close_dialog_open {
            draw_close_dialog(ctx, self);
        }
    }
}

fn draw_close_dialog(ctx: &egui::Context, app: &mut HierarchosApp) {
    egui::Window::new("Save LTM Updates?")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.set_width(430.0);
            ui.label(
                RichText::new("Do you want to save runtime LTM updates before exiting?")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(14.0),
            );
            ui.add_space(6.0);
            ui.label(
                RichText::new(
                    "Saving writes a small sidecar file next to the loaded model. Discarding closes without changing the model directory.",
                )
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
            );

            if app.ltm_save_in_progress {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        RichText::new("Saving LTM updates...")
                            .color(HierarchosColors::WARNING)
                            .size(12.0),
                    );
                });
            }

            if let Some(err) = &app.ltm_save_error {
                ui.add_space(10.0);
                ui.label(
                    RichText::new(format!("Save failed: {}", err))
                        .color(HierarchosColors::ERROR)
                        .size(12.0),
                );
            }

            ui.add_space(16.0);
            ui.horizontal(|ui| {
                let save_enabled = !app.ltm_save_in_progress && app.bridge.is_connected();
                let save = egui::Button::new(
                    RichText::new("Save & Exit")
                        .color(HierarchosColors::TEXT_ON_PRIMARY)
                        .size(13.0),
                )
                .fill(get_accent().primary)
                .corner_radius(Rounding::same(6));
                if ui.add_enabled(save_enabled, save).clicked() {
                    let _ = app.chat_state.save_active_chat();
                    app.bridge
                        .save_chat_runtime_state(app.chat_state.runtime_state_path_string());
                    app.ltm_save_in_progress = true;
                    app.close_after_ltm_save = true;
                    app.ltm_save_error = None;
                    app.bridge.save_ltm_updates();
                }

                let discard = egui::Button::new(
                    RichText::new("Discard & Exit")
                        .color(HierarchosColors::TEXT_PRIMARY)
                        .size(13.0),
                )
                .fill(HierarchosColors::BG_SURFACE)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(6));
                if ui.add_enabled(!app.ltm_save_in_progress, discard).clicked() {
                    let _ = app.chat_state.save_active_chat();
                    if app.bridge.is_connected() {
                        app.bridge
                            .save_chat_runtime_state(app.chat_state.runtime_state_path_string());
                    }
                    app.close_confirmed = true;
                    app.close_dialog_open = false;
                    app.close_after_ltm_save = false;
                    app.ltm_save_error = None;
                    app.bridge.disconnect();
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }

                if ui
                    .add_enabled(!app.ltm_save_in_progress, egui::Button::new("Cancel"))
                    .clicked()
                {
                    app.close_dialog_open = false;
                    app.close_after_ltm_save = false;
                    app.ltm_save_error = None;
                }
            });
        });
}

/// Draw the top navigation bar.
fn draw_top_bar(
    ui: &mut egui::Ui,
    active_panel: &mut Panel,
    bridge: &PythonBridge,
    load_progress: Option<&LoadProgressState>,
) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::GRADIENT_HEADER_START)
        .inner_margin(egui::Margin::symmetric(16, 8));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            // Logo / Title
            ui.label(RichText::new("◆").color(get_accent().primary).size(22.0));
            ui.label(
                RichText::new("HIERARCHOS")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(16.0)
                    .strong(),
            );
            ui.label(
                RichText::new("v0.1")
                    .color(HierarchosColors::TEXT_MUTED)
                    .size(10.0),
            );

            ui.add_space(24.0);

            // Navigation tabs
            let tabs = [
                (Panel::Chat, "💬 Chat"),
                (Panel::Training, "📊 Training"),
                (Panel::Inspector, "🔍 Inspector"),
                (Panel::Memory, "🧠 Memory"),
                (Panel::Hub, "🤗 Models"),
                (Panel::Settings, "⚙ Settings"),
            ];

            for (panel, label) in &tabs {
                let is_active = *active_panel == *panel;

                let btn = egui::Button::new(
                    RichText::new(*label)
                        .color(if is_active {
                            HierarchosColors::TEXT_ON_PRIMARY
                        } else {
                            HierarchosColors::TEXT_SECONDARY
                        })
                        .size(13.0),
                )
                .fill(if is_active {
                    get_accent().primary_dim
                } else {
                    Color32::TRANSPARENT
                })
                .stroke(if is_active {
                    Stroke::new(1.0, get_accent().primary)
                } else {
                    Stroke::NONE
                })
                .corner_radius(Rounding::same(6));

                if ui.add(btn).clicked() {
                    *active_panel = *panel;
                }
            }

            // Right side: connection status
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if bridge.is_model_loaded() {
                    ui.label(
                        RichText::new("● Connected")
                            .color(HierarchosColors::SUCCESS)
                            .size(11.0),
                    );
                } else {
                    ui.label(
                        RichText::new("○ Disconnected")
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0),
                    );
                }
            });
        });

        if let Some(progress) = load_progress {
            ui.add_space(8.0);
            draw_load_progress(ui, progress);
        }
    });
}

fn draw_load_progress(ui: &mut egui::Ui, progress: &LoadProgressState) {
    let pct = progress.progress.clamp(0.0, 1.0);
    let percent_text = format!("{:>3}%  {}", (pct * 100.0).round() as u32, progress.label);
    let progress_bar = egui::ProgressBar::new(pct)
        .desired_width(ui.available_width())
        .text(
            RichText::new(percent_text)
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(11.0),
        );

    ui.add(progress_bar);
}
