// panels/training.rs — Training Dashboard Panel
//
// Real-time training monitoring with live loss curves, metric cards,
// progress tracking, and configuration controls.

use crate::bridge::{PythonBridge, TrainingBackend, TrainingConfig, VulkanTrainingPrecision};
use crate::theme::{get_accent, HierarchosColors};
use crate::widgets::metric_card::{progress_bar_labeled, MetricCard};
use egui::{self, CornerRadius as Rounding, RichText, ScrollArea, Stroke};
use egui_plot::{Line, Plot, PlotPoints};

/// Training state.
pub struct TrainingState {
    pub config: TrainingConfig,
    pub is_training: bool,
    pub loss_history: Vec<f64>,
    pub lr_history: Vec<f64>,
    pub ponder_history: Vec<f64>,
    pub commitment_history: Vec<f64>,
    pub tps_history: Vec<f64>,
    pub current_epoch: u32,
    pub current_step: u32,
    pub total_steps: u32,
    pub current_loss: f64,
    pub current_lr: f64,
    pub current_tps: f64,
    pub log_messages: Vec<String>,
    pub show_config: bool,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            config: TrainingConfig::default(),
            is_training: false,
            loss_history: Vec::new(),
            lr_history: Vec::new(),
            ponder_history: Vec::new(),
            commitment_history: Vec::new(),
            tps_history: Vec::new(),
            current_epoch: 0,
            current_step: 0,
            total_steps: 100,
            current_loss: 0.0,
            current_lr: 0.0,
            current_tps: 0.0,
            log_messages: Vec::new(),
            show_config: true,
        }
    }
}

impl TrainingState {
    pub fn on_metrics(
        &mut self,
        epoch: u32,
        step: u32,
        total_steps: Option<u32>,
        loss: f64,
        lr: f64,
        ponder: Option<f64>,
        commitment: Option<f64>,
        tps: Option<f64>,
    ) {
        self.current_epoch = epoch;
        self.current_step = step;
        if let Some(total_steps) = total_steps.filter(|steps| *steps > 0) {
            self.total_steps = total_steps;
        }
        self.current_loss = loss;
        self.current_lr = lr;
        self.loss_history.push(loss);
        self.lr_history.push(lr);
        if let Some(p) = ponder {
            self.ponder_history.push(p);
        }
        if let Some(c) = commitment {
            self.commitment_history.push(c);
        }
        if let Some(t) = tps {
            self.current_tps = t;
            self.tps_history.push(t);
        }
        self.log_messages.push(format!(
            "[Epoch {} | Step {}] Loss: {:.4} | LR: {:.2e} | {:.0} tok/s",
            epoch,
            step,
            loss,
            lr,
            tps.unwrap_or(0.0)
        ));
    }
}

/// Draw the training dashboard.
pub fn draw_training_panel(
    ui: &mut egui::Ui,
    state: &mut TrainingState,
    bridge: &PythonBridge,
    model_path: &str,
) {
    // Wrap entire panel in a scroll area to prevent overflow crashes
    ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Header
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("📊 Training Dashboard")
                        .color(HierarchosColors::TEXT_PRIMARY)
                        .size(18.0)
                        .strong(),
                );

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if state.is_training {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("⏹ Stop Training")
                                        .color(HierarchosColors::ERROR)
                                        .size(13.0),
                                )
                                .fill(HierarchosColors::BG_CARD)
                                .stroke(Stroke::new(
                                    1.0,
                                    HierarchosColors::ERROR.linear_multiply(0.4),
                                ))
                                .corner_radius(Rounding::same(8)),
                            )
                            .clicked()
                        {
                            bridge.stop_training();
                            state.is_training = false;
                        }
                    } else {
                        let can_start = bridge.is_model_loaded()
                            && !bridge.is_generating()
                            && !bridge.is_loading()
                            && !bridge.is_training();
                        if ui
                            .add_enabled(
                                can_start,
                                egui::Button::new(
                                    RichText::new("▶ Start Training")
                                        .color(HierarchosColors::TEXT_ON_PRIMARY)
                                        .size(13.0),
                                )
                                .fill(HierarchosColors::SUCCESS)
                                .corner_radius(Rounding::same(8)),
                            )
                            .clicked()
                        {
                            state.is_training = true;
                            state.loss_history.clear();
                            state.lr_history.clear();
                            state.ponder_history.clear();
                            state.commitment_history.clear();
                            state.tps_history.clear();
                            state.log_messages.clear();
                            bridge.start_training(state.config.clone(), model_path.to_string());
                        }
                    }

                    // Toggle config panel
                    let config_text = if state.show_config {
                        "▼ Config"
                    } else {
                        "▶ Config"
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(config_text)
                                    .color(HierarchosColors::TEXT_SECONDARY)
                                    .size(12.0),
                            )
                            .fill(HierarchosColors::BG_CARD)
                            .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                            .corner_radius(Rounding::same(6)),
                        )
                        .clicked()
                    {
                        state.show_config = !state.show_config;
                    }
                });
            });

            ui.add_space(8.0);

            // Metric cards row
            ui.horizontal_wrapped(|ui| {
                let card_width = ((ui.available_width() - 32.0) / 4.0).max(150.0);

                MetricCard {
                    title: "Loss",
                    value: &format!("{:.4}", state.current_loss),
                    subtitle: Some(&format!("Step {}", state.current_step)),
                    accent_color: get_accent().primary,
                    icon: "📉",
                }
                .show(ui, card_width);

                MetricCard {
                    title: "Learning Rate",
                    value: &format!("{:.2e}", state.current_lr),
                    subtitle: Some(&format!("Epoch {}", state.current_epoch)),
                    accent_color: HierarchosColors::ACCENT_CYAN,
                    icon: "📐",
                }
                .show(ui, card_width);

                MetricCard {
                    title: "Throughput",
                    value: &format!("{:.0}", state.current_tps),
                    subtitle: Some("tokens/sec"),
                    accent_color: HierarchosColors::SUCCESS,
                    icon: "⚡",
                }
                .show(ui, card_width);

                MetricCard {
                    title: "Progress",
                    value: &format!("{}/{}", state.current_step, state.total_steps),
                    subtitle: Some(&format!("Epoch {}", state.current_epoch)),
                    accent_color: HierarchosColors::WARNING,
                    icon: "🎯",
                }
                .show(ui, card_width);
            });

            ui.add_space(8.0);

            // Progress bar
            if state.is_training && state.total_steps > 0 {
                let progress = state.current_step as f32 / state.total_steps as f32;
                progress_bar_labeled(ui, "Training Progress", progress, get_accent().primary);
                ui.add_space(8.0);
            }

            // Config section (collapsible)
            if state.show_config && !state.is_training {
                draw_config_section(ui, state);
                ui.add_space(8.0);
            }

            // Loss Curve Plot — fixed height
            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_CARD)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(8))
                .inner_margin(egui::Margin::same(8));

            frame.show(ui, |ui| {
                ui.label(
                    RichText::new("Loss Curve")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0)
                        .strong(),
                );

                let plot = Plot::new("loss_plot")
                    .height(250.0)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .show_grid(true)
                    .show_axes(true);

                plot.show(ui, |plot_ui| {
                    if !state.loss_history.is_empty() {
                        let points: PlotPoints = state
                            .loss_history
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect();
                        plot_ui.line(
                            Line::new(points)
                                .color(get_accent().primary)
                                .name("CE Loss")
                                .width(2.0),
                        );
                    }

                    if !state.ponder_history.is_empty() {
                        let points: PlotPoints = state
                            .ponder_history
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect();
                        plot_ui.line(
                            Line::new(points)
                                .color(HierarchosColors::WARNING)
                                .name("Ponder Cost")
                                .width(1.5),
                        );
                    }

                    if !state.commitment_history.is_empty() {
                        let points: PlotPoints = state
                            .commitment_history
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect();
                        plot_ui.line(
                            Line::new(points)
                                .color(HierarchosColors::ACCENT_CYAN)
                                .name("Commitment Cost")
                                .width(1.5),
                        );
                    }
                });
            });

            ui.add_space(4.0);

            // Training log
            let log_frame = egui::Frame::new()
                .fill(HierarchosColors::BG_INPUT)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(8))
                .inner_margin(egui::Margin::same(8));

            log_frame.show(ui, |ui| {
                ui.label(
                    RichText::new("Training Log")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0)
                        .strong(),
                );

                ScrollArea::vertical()
                    .max_height(180.0)
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for msg in &state.log_messages {
                            ui.label(
                                RichText::new(msg)
                                    .color(HierarchosColors::TEXT_MUTED)
                                    .size(11.0)
                                    .monospace(),
                            );
                        }

                        if state.log_messages.is_empty() {
                            ui.label(
                                RichText::new(
                                    "No training logs yet. Configure and start training.",
                                )
                                .color(HierarchosColors::TEXT_MUTED)
                                .size(12.0),
                            );
                        }
                    });
                if state.is_training {
                    ui.ctx().request_repaint();
                }
            });
        });
}

fn draw_config_section(ui: &mut egui::Ui, state: &mut TrainingState) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(8))
        .inner_margin(egui::Margin::same(14));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("Training Configuration")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(14.0)
                .strong(),
        );
        ui.add_space(8.0);

        ui.horizontal_wrapped(|ui| {
            ui.label(
                RichText::new("Training Backend")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            egui::ComboBox::from_id_salt("training_backend_selector")
                .selected_text(state.config.backend.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.config.backend,
                        TrainingBackend::Pytorch,
                        TrainingBackend::Pytorch.label(),
                    );
                    ui.selectable_value(
                        &mut state.config.backend,
                        TrainingBackend::Vulkan,
                        TrainingBackend::Vulkan.label(),
                    );
                });
            if state.config.backend == TrainingBackend::Vulkan {
                ui.separator();
                ui.label(
                    RichText::new("Vulkan devices")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(
                    egui::TextEdit::singleline(&mut state.config.vulkan_device_indices)
                        .desired_width(90.0)
                        .hint_text("0 or 0,1"),
                )
                .on_hover_text(
                    "Physical Vulkan adapter indices. A comma-separated list enables native synchronous data parallel training.",
                );
            }
        });
        if state.config.backend == TrainingBackend::Vulkan {
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "Native Vulkan uses the loaded local model.safetensors package directly and writes PyTorch/CUDA/native-Rust compatible SafeTensors checkpoints. For backend-parity training, select a Hierarchos token-cache directory (index.safetensors + tokens.bin); legacy tokenized JSONL remains supported.",
                )
                .color(HierarchosColors::ACCENT_CYAN)
                .size(11.0),
            );
            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut state.config.vulkan_exact_resume,
                    "Exact-resume loaded checkpoint",
                )
                .on_hover_text(
                    "Restore the Vulkan/PyTorch-portable optimizer, scheduler/scaler, data cursor, pending gradients, and recurrent replay state from the loaded package. Disable this to start a fresh optimizer/session from the same model weights.",
                );
                if state.config.vulkan_exact_resume {
                    ui.label(
                        RichText::new("requires training_state.json + portable replay state")
                            .color(HierarchosColors::WARNING)
                            .size(11.0),
                    );
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Training precision")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled_ui(!state.config.vulkan_exact_resume, |ui| {
                    egui::ComboBox::from_id_salt("vulkan_training_precision")
                        .selected_text(state.config.vulkan_precision.label())
                        .show_ui(ui, |ui| {
                            for precision in VulkanTrainingPrecision::ALL {
                                ui.selectable_value(
                                    &mut state.config.vulkan_precision,
                                    precision,
                                    precision.label(),
                                );
                            }
                        });
                });
                ui.label(
                    RichText::new(if state.config.vulkan_exact_resume {
                        "checkpoint-locked on launch"
                    } else {
                        "FP32 master weights stay portable"
                    })
                    .color(if state.config.vulkan_exact_resume {
                        HierarchosColors::WARNING
                    } else {
                        HierarchosColors::TEXT_MUTED
                    })
                    .size(11.0),
                );
            });
        }
        ui.add_space(8.0);

        // Use a grid instead of columns to avoid the height assertion
        egui::Grid::new("training_config_grid")
            .num_columns(4)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                // Row 1
                ui.label(
                    RichText::new("Epochs")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(egui::DragValue::new(&mut state.config.epochs).range(1..=100));
                ui.label(
                    RichText::new("TBPTT Chunk")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(
                    egui::DragValue::new(&mut state.config.training_chunk_size).range(16..=1024),
                );
                ui.end_row();

                // Row 2
                ui.label(
                    RichText::new("Batch Size")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(egui::DragValue::new(&mut state.config.batch_size).range(1..=128));
                ui.label(
                    RichText::new("Accum Steps")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(egui::DragValue::new(&mut state.config.accumulation_steps).range(1..=32));
                ui.end_row();

                // Row 3
                ui.label(
                    RichText::new("Learning Rate")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(
                    egui::DragValue::new(&mut state.config.learning_rate)
                        .speed(0.00001)
                        .range(0.0..=0.01)
                        .max_decimals(7),
                );
                ui.label(
                    RichText::new("Save Every N")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(egui::DragValue::new(&mut state.config.save_steps).range(0..=10000));
                ui.end_row();

                // Row 4
                ui.label(
                    RichText::new("Grad Clip")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add(
                    egui::DragValue::new(&mut state.config.grad_clip)
                        .speed(0.1)
                        .range(0.0..=10.0),
                );
                ui.label(
                    RichText::new("Mixed Precision")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    state.config.backend == TrainingBackend::Pytorch,
                    egui::Checkbox::new(&mut state.config.amp, "AMP"),
                )
                .on_hover_text(if state.config.backend == TrainingBackend::Vulkan {
                    "Native Vulkan precision is checkpoint/policy driven so PyTorch/CUDA interchange never changes the authoritative FP32 master weights."
                } else {
                    "Use PyTorch automatic mixed precision when supported."
                });
                ui.end_row();

                // Row 5
                ui.label(
                    RichText::new("Full-Sample BPTT")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                if ui
                    .add_enabled(
                        state.config.backend == TrainingBackend::Pytorch,
                        egui::Checkbox::new(&mut state.config.full_sample_bptt, "Exact coherence"),
                    )
                    .on_hover_text(
                        if state.config.backend == TrainingBackend::Vulkan {
                            "The native backend owns its recurrent Vulkan tape and TBPTT/checkpoint planner directly; this PyTorch autograd option does not apply."
                                .to_string()
                        } else {
                            "Use one attached gradient graph per trimmed sample. This disables cross-sample recurrent persistence and token-level detachment."
                                .to_string()
                        },
                    )
                    .changed()
                    && state.config.full_sample_bptt
                {
                    state.config.persist_state = false;
                    // Exact full-sample recurrence retains the entire graph.
                    // Start in the memory-safe mode; advanced users can still
                    // explicitly turn recomputation back off afterwards.
                    state.config.full_sample_activation_checkpointing = true;
                }
                ui.label(
                    RichText::new("Activation Recompute")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    state.config.backend == TrainingBackend::Pytorch && state.config.full_sample_bptt,
                    egui::Checkbox::new(
                        &mut state.config.full_sample_activation_checkpointing,
                        "Checkpoint",
                    ),
                )
                .on_hover_text("Recompute the full forward during backward to save VRAM without truncating gradients.");
                ui.end_row();

                ui.label(
                    RichText::new("Checkpoint Segment")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    state.config.backend == TrainingBackend::Pytorch
                        && state.config.full_sample_bptt
                        && state.config.full_sample_activation_checkpointing,
                    egui::DragValue::new(
                        &mut state.config.full_sample_checkpoint_segment_size,
                    )
                    .range(1..=4096),
                )
                .on_hover_text("Tokens per activation-recompute segment; smaller values reduce peak VRAM at extra compute cost.");
                ui.label("");
                ui.label("");
                ui.end_row();
            });

        if state.config.backend == TrainingBackend::Vulkan {
            ui.add_space(8.0);
            egui::CollapsingHeader::new("Vulkan PyTorch-parity controls")
                .id_salt("vulkan_pytorch_parity_controls")
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(
                            "These values map directly to the Rust trainer's portable optimizer, scheduler, objective, and deterministic data-order policy. Exact resume reconstructs trajectory-defining values from training_state.json before launch.",
                        )
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0),
                    );
                    ui.add_space(6.0);
                    ui.add_enabled_ui(!state.config.vulkan_exact_resume, |ui| {
                        egui::Grid::new("vulkan_parity_config_grid")
                            .num_columns(4)
                            .spacing([16.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Warmup steps");
                                ui.add(egui::DragValue::new(&mut state.config.warmup_steps).range(0..=1_000_000));
                                ui.label("Warmup ratio");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.warmup_ratio)
                                        .speed(0.001)
                                        .range(0.0..=1.0),
                                );
                                ui.end_row();

                                ui.label("Adam β1");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.beta1)
                                        .speed(0.001)
                                        .range(0.0..=0.99999),
                                );
                                ui.label("Adam β2");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.beta2)
                                        .speed(0.001)
                                        .range(0.0..=0.999999),
                                );
                                ui.end_row();

                                ui.label("Adam ε");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.eps)
                                        .speed(1.0e-9)
                                        .range(1.0e-12..=1.0e-2),
                                );
                                ui.label("Weight decay");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.weight_decay)
                                        .speed(0.001)
                                        .range(0.0..=10.0),
                                );
                                ui.end_row();

                                ui.label("Z-loss weight");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.z_loss_weight)
                                        .speed(1.0e-5)
                                        .range(0.0..=10.0),
                                );
                                ui.label("Ponder weight");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.ponder_loss_weight)
                                        .speed(1.0e-4)
                                        .range(0.0..=10.0),
                                );
                                ui.end_row();

                                ui.label("Commitment weight");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.commitment_loss_weight)
                                        .speed(0.01)
                                        .range(0.0..=10.0),
                                );
                                ui.label("Max skipped batches");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.max_skipped_train_batches)
                                        .range(0..=1_000_000),
                                );
                                ui.end_row();

                                ui.label("CE backward cap");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.max_ce_loss_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=1_000.0),
                                );
                                ui.label("Ponder backward cap");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.max_ponder_cost_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=1_000.0),
                                );
                                ui.end_row();

                                ui.label("Commitment cap");
                                ui.add(
                                    egui::DragValue::new(&mut state.config.max_commitment_cost_for_backward)
                                        .speed(0.1)
                                        .range(0.0..=1_000.0),
                                );
                                ui.label("Seed");
                                ui.add(egui::DragValue::new(&mut state.config.seed));
                                ui.end_row();
                            });

                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(
                                &mut state.config.disable_lr_schedule,
                                "Fixed learning rate",
                            );
                            ui.checkbox(&mut state.config.vulkan_tbptt_enabled, "TBPTT");
                            if ui
                                .checkbox(&mut state.config.persist_state, "Carry recurrent state")
                                .changed()
                                && state.config.persist_state
                            {
                                state.config.shuffle = false;
                            }
                            ui.add_enabled(
                                !state.config.persist_state,
                                egui::Checkbox::new(&mut state.config.shuffle, "Shuffle each epoch"),
                            );
                        });
                    });
                    if state.config.vulkan_exact_resume {
                        ui.small(
                            "Trajectory controls are locked here because the bridge rehydrates them from the checkpoint. Target epochs, output path, save cadence, and Vulkan device topology remain runtime choices.",
                        );
                    }
                });
        }

        ui.add_space(8.0);

        ui.separator();
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Loaded Model Architecture (locked)")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(13.0)
                    .strong(),
            );
        });
        ui.add_space(6.0);

        egui::Grid::new("training_arch_grid")
            .num_columns(4)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label(
                    RichText::new("Context Dim")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.context_dim).range(32..=4096),
                );
                ui.label(
                    RichText::new("Max Length")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !state.config.auto_max_length,
                        egui::DragValue::new(&mut state.config.max_length).range(32..=32768),
                    );
                    ui.checkbox(&mut state.config.auto_max_length, "Auto");
                });
                ui.end_row();

                ui.label(
                    RichText::new("H Hidden")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.h_hidden).range(32..=4096),
                );
                ui.label(
                    RichText::new("L Hidden")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.l_hidden).range(32..=4096),
                );
                ui.end_row();

                ui.label(
                    RichText::new("Persistent Dim")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.persistent_dim).range(1..=2048),
                );
                ui.label(
                    RichText::new("H Stride")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.h_stride).range(1..=64),
                );
                ui.end_row();

                ui.label(
                    RichText::new("LTM Slots")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.ltm_slots).range(1..=65536),
                );
                ui.label(
                    RichText::new("LTM Top-K")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.ltm_topk).range(1..=64),
                );
                ui.end_row();

                ui.label(
                    RichText::new("LTM Key Dim")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.ltm_key_dim).range(8..=2048),
                );
                ui.label(
                    RichText::new("LTM Val Dim")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.ltm_val_dim).range(8..=2048),
                );
                ui.end_row();

                ui.label(
                    RichText::new("Max H Steps")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.max_h_steps).range(1..=64),
                );
                ui.label(
                    RichText::new("Max L Steps")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );
                ui.add_enabled(
                    false,
                    egui::DragValue::new(&mut state.config.max_l_steps).range(1..=64),
                );
                ui.end_row();
            });

        ui.add_space(8.0);

        // Dataset path
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Dataset:")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::TextEdit::singleline(&mut state.config.data_path)
                    .desired_width(
                        ui.available_width()
                            - if state.config.backend == TrainingBackend::Vulkan { 150.0 } else { 80.0 },
                    )
                    .hint_text(if state.config.backend == TrainingBackend::Vulkan {
                        "Token cache directory or training JSONL"
                    } else {
                        "Path to training data (.jsonl)"
                    }),
            );
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Training Data", &["jsonl", "json", "pt"])
                    .pick_file()
                {
                    state.config.data_path = path.display().to_string();
                }
            }
            if state.config.backend == TrainingBackend::Vulkan && ui.button("Cache").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    state.config.data_path = path.display().to_string();
                }
            }
        });

        // Output dir
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Output:")
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(12.0),
            );
            ui.add(
                egui::TextEdit::singleline(&mut state.config.out_dir)
                    .desired_width(ui.available_width() - 80.0)
                    .hint_text("Output directory"),
            );
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    state.config.out_dir = path.display().to_string();
                }
            }
        });
    });
}
