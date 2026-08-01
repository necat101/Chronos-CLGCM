// panels/inspector.rs — Model Architecture Inspector
//
// Displays a tree view of model layers, parameter counts,
// weight statistics, and hyperparameter configuration.

use crate::bridge::{ModelConfig, ModelInspection, PythonBridge};
use crate::theme::{get_accent, HierarchosColors};
use egui::{self, CornerRadius as Rounding, RichText, ScrollArea, Stroke};

/// Inspector state.
pub struct InspectorState {
    pub config: Option<ModelConfig>,
    pub inspection: Option<ModelInspection>,
    pub selected_layer: Option<usize>,
    pub show_config: bool,
    pub show_layers: bool,
}

impl Default for InspectorState {
    fn default() -> Self {
        Self {
            config: None,
            inspection: None,
            selected_layer: None,
            show_config: true,
            show_layers: true,
        }
    }
}

/// Draw the model inspector panel.
pub fn draw_inspector_panel(ui: &mut egui::Ui, state: &mut InspectorState, bridge: &PythonBridge) {
    let can_inspect = bridge.is_model_loaded()
        && !bridge.is_generating()
        && !bridge.is_training()
        && !bridge.is_loading();
    ui.vertical(|ui| {
        // Header
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("🔍 Model Inspector")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(18.0)
                    .strong(),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        can_inspect,
                        egui::Button::new(
                            RichText::new("↻ Refresh")
                                .color(HierarchosColors::TEXT_SECONDARY)
                                .size(12.0),
                        )
                        .fill(HierarchosColors::BG_CARD)
                        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                        .corner_radius(Rounding::same(6)),
                    )
                    .clicked()
                {
                    bridge.request_model_info();
                }
            });
        });

        ui.add_space(8.0);

        if !bridge.is_model_loaded() {
            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_CARD)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(12))
                .inner_margin(egui::Margin::same(40));

            frame.show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("🧠").size(48.0));
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("No model loaded")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(16.0),
                    );
                    ui.label(
                        RichText::new(
                            "Load a model from the Settings panel to inspect its architecture.",
                        )
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(13.0),
                    );
                });
            });
            return;
        }

        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Architecture Overview
                if let Some(config) = &state.config {
                    draw_architecture_overview(ui, config);
                    ui.add_space(12.0);
                }

                // Hyperparameters Table
                if state.show_config {
                    if let Some(config) = &state.config {
                        draw_config_table(ui, config);
                        ui.add_space(12.0);
                    }
                }

                // Layer tree
                if state.show_layers {
                    if let Some(inspection) = &state.inspection {
                        draw_layer_tree(ui, inspection, &mut state.selected_layer);
                    }
                }
            });
    });
}

fn draw_architecture_overview(ui: &mut egui::Ui, config: &ModelConfig) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, get_accent().primary.linear_multiply(0.2)))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("Architecture Overview")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(15.0)
                .strong(),
        );
        ui.add_space(8.0);

        // Architecture diagram (text-based)
        let diagram = format!(
            "┌─────────────────────────────────────────────┐\n\
             │  HierarchosCore  ({:.1}M params)              │\n\
             │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │\n\
             │  │ Manager  │  │ Worker   │  │  LTM      │  │\n\
             │  │ (H-RNN)  │──│ (L-RNN)  │──│ (Titans)  │  │\n\
             │  │ dim={}  │  │ dim={}  │  │ {}×{} │  │\n\
             │  └──────────┘  └──────────┘  └───────────┘  │\n\
             │  Stride: {}  |  TopK: {}  |  Device: {}     │\n\
             └─────────────────────────────────────────────┘",
            config.total_params as f64 / 1e6,
            config.h_hidden,
            config.l_hidden,
            config.ltm_slots,
            config.ltm_val_dim,
            config.h_stride,
            config.ltm_topk,
            config.device_label.as_deref().unwrap_or(&config.device),
        );

        ui.label(
            RichText::new(diagram)
                .color(HierarchosColors::ACCENT_CYAN)
                .size(12.0)
                .monospace(),
        );
    });
}

fn draw_config_table(ui: &mut egui::Ui, config: &ModelConfig) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(14));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("Hyperparameters")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(14.0)
                .strong(),
        );
        ui.add_space(8.0);

        egui::Grid::new("config_grid")
            .num_columns(4)
            .spacing([20.0, 6.0])
            .striped(true)
            .show(ui, |ui| {
                let row = |ui: &mut egui::Ui, label: &str, value: &str| {
                    ui.label(
                        RichText::new(label)
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    );
                    ui.label(
                        RichText::new(value)
                            .color(HierarchosColors::TEXT_PRIMARY)
                            .size(12.0)
                            .monospace(),
                    );
                };

                row(ui, "Context Dim", &config.context_dim.to_string());
                row(ui, "Vocab Size", &config.vocab_size.to_string());
                ui.end_row();

                row(ui, "H-RNN Hidden", &config.h_hidden.to_string());
                row(ui, "L-RNN Hidden", &config.l_hidden.to_string());
                ui.end_row();

                row(ui, "LTM Slots", &config.ltm_slots.to_string());
                row(ui, "LTM Key Dim", &config.ltm_key_dim.to_string());
                ui.end_row();

                row(ui, "LTM Val Dim", &config.ltm_val_dim.to_string());
                row(ui, "LTM Top-K", &config.ltm_topk.to_string());
                ui.end_row();

                row(ui, "H Stride", &config.h_stride.to_string());
                row(ui, "Max H Steps", &config.max_h_steps.to_string());
                ui.end_row();

                row(ui, "Max L Steps", &config.max_l_steps.to_string());
                row(ui, "Max Length", &config.max_length.to_string());
                ui.end_row();

                row(ui, "Persistent Dim", &config.persistent_dim.to_string());
                row(ui, "Quantized", &config.is_quantized.to_string());
                ui.end_row();

                row(
                    ui,
                    "Device",
                    config.device_label.as_deref().unwrap_or(&config.device),
                );
                row(
                    ui,
                    "PyTorch",
                    config.torch_version.as_deref().unwrap_or("unknown"),
                );
                ui.end_row();

                row(ui, "CUDA", config.cuda_version.as_deref().unwrap_or("none"));
                let gpu_label =
                    config
                        .cuda_device_name
                        .as_deref()
                        .unwrap_or(if config.cuda_available {
                            "available"
                        } else {
                            "inactive"
                        });
                row(ui, "GPU", gpu_label);
                ui.end_row();
            });
    });
}

fn draw_layer_tree(ui: &mut egui::Ui, inspection: &ModelInspection, selected: &mut Option<usize>) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(14));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Layer Parameters")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(14.0)
                    .strong(),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!(
                        "Total: {:.2}M | Trainable: {:.2}M",
                        inspection.total_params as f64 / 1e6,
                        inspection.trainable_params as f64 / 1e6
                    ))
                    .color(HierarchosColors::TEXT_MUTED)
                    .size(11.0),
                );
            });
        });
        ui.add_space(8.0);

        // Table header
        egui::Grid::new("layer_grid")
            .num_columns(6)
            .spacing([16.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.label(
                    RichText::new("Layer")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.label(
                    RichText::new("Params")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.label(
                    RichText::new("Shape")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.label(
                    RichText::new("Mean")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.label(
                    RichText::new("Std")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.label(
                    RichText::new("Range")
                        .color(HierarchosColors::TEXT_MUTED)
                        .size(11.0)
                        .strong(),
                );
                ui.end_row();

                for (i, layer) in inspection.layers.iter().enumerate() {
                    let is_selected = *selected == Some(i);
                    let text_color = if is_selected {
                        get_accent().primary
                    } else {
                        HierarchosColors::TEXT_PRIMARY
                    };

                    if ui
                        .selectable_label(
                            is_selected,
                            RichText::new(&layer.name)
                                .color(text_color)
                                .size(12.0)
                                .monospace(),
                        )
                        .clicked()
                    {
                        *selected = Some(i);
                    }

                    ui.label(
                        RichText::new(format_params(layer.param_count))
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0)
                            .monospace(),
                    );

                    let shape_str: Vec<String> =
                        layer.shape.iter().map(|s| s.to_string()).collect();
                    ui.label(
                        RichText::new(format!("[{}]", shape_str.join("×")))
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0)
                            .monospace(),
                    );

                    ui.label(
                        RichText::new(format!("{:.4}", layer.mean))
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0)
                            .monospace(),
                    );

                    ui.label(
                        RichText::new(format!("{:.4}", layer.std))
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0)
                            .monospace(),
                    );

                    ui.label(
                        RichText::new(format!("[{:.3}, {:.3}]", layer.min, layer.max))
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0)
                            .monospace(),
                    );

                    ui.end_row();
                }
            });
    });
}

fn format_params(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1e3)
    } else {
        count.to_string()
    }
}
