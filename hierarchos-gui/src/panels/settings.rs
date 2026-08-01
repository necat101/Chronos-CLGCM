// panels/settings.rs — Global Settings Panel
//
// Model loading, device selection, inference settings, and appearance options.

use crate::bridge::PythonBridge;
use crate::theme::{self, get_accent, HierarchosColors};
use egui::{self, Color32, CornerRadius as Rounding, RichText, ScrollArea, Stroke, Vec2};

/// Application settings state.
pub struct AppSettings {
    pub model_path: String,
    pub device: String,
    pub available_devices: Vec<String>,
    pub python_path: String,
    pub pending_load: bool,

    // Online Learning
    pub passive_learning: bool,
    pub surprise_threshold: f32,
    pub passive_lr: f64,
    pub ltm_lr: f64,

    // Appearance
    pub font_size: f32,
    pub accent_color_idx: usize,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            device: "Auto".to_string(),
            available_devices: vec![
                "Auto".to_string(),
                "CUDA".to_string(),
                "DirectML".to_string(),
                "CPU".to_string(),
            ],
            python_path: "bundled".to_string(),
            pending_load: false,
            passive_learning: true,
            surprise_threshold: 1.0,
            passive_lr: 5e-6,
            ltm_lr: 1e-3,
            font_size: 14.0,
            accent_color_idx: 0,
        }
    }
}

/// Draw the settings panel.
pub fn draw_settings_panel(ui: &mut egui::Ui, settings: &mut AppSettings, bridge: &PythonBridge) {
    ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.vertical(|ui| {
                // Header
                ui.label(
                    RichText::new("⚙ Settings")
                        .color(HierarchosColors::TEXT_PRIMARY)
                        .size(18.0)
                        .strong(),
                );
                ui.add_space(12.0);

                // Model Loading Section
                draw_model_section(ui, settings, bridge);
                ui.add_space(16.0);

                // Online Learning Section
                draw_learning_section(ui, settings);
                ui.add_space(16.0);

                // Appearance Section
                draw_appearance_section(ui, settings);
                ui.add_space(16.0);

                // About Section
                draw_about_section(ui);
            });
        });
}

fn draw_model_section(ui: &mut egui::Ui, settings: &mut AppSettings, bridge: &PythonBridge) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("🗂 Model")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(15.0)
                .strong(),
        );
        ui.add_space(10.0);

        // Model path
        ui.label(
            RichText::new("Model Source")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut settings.model_path)
                    .desired_width(ui.available_width() - 260.0)
                    .hint_text("Model folder, direct .pt file, or HF repo id"),
            );
            if ui
                .button("Folder")
                .on_hover_text("Choose a model directory")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    settings.model_path = path.display().to_string();
                }
            }
            if ui
                .button(".pt")
                .on_hover_text("Choose an inference checkpoint")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PyTorch checkpoint", &["pt"])
                    .pick_file()
                {
                    settings.model_path = path.display().to_string();
                }
            }
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(if bridge.is_connected() {
                            "Load"
                        } else {
                            "Connect & Load"
                        })
                        .color(HierarchosColors::TEXT_ON_PRIMARY)
                        .size(13.0),
                    )
                    .fill(get_accent().primary)
                    .corner_radius(Rounding::same(6)),
                )
                .clicked()
            {
                if !settings.model_path.is_empty() {
                    if bridge.is_connected() {
                        bridge.load_model(settings.model_path.clone(), settings.device.clone());
                    } else {
                        settings.pending_load = true;
                        bridge.connect(&settings.python_path);
                    }
                }
            }
        });

        ui.add_space(8.0);

        // Python interpreter path
        ui.label(
            RichText::new("Python Interpreter")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut settings.python_path)
                    .desired_width(ui.available_width() - 240.0)
                    .hint_text("bundled, auto, python, or full path to interpreter"),
            );

            if bridge.is_connected() {
                ui.label(
                    RichText::new("● Connected")
                        .color(HierarchosColors::SUCCESS)
                        .size(12.0),
                );
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Disconnect")
                                .color(HierarchosColors::ERROR)
                                .size(12.0),
                        )
                        .fill(HierarchosColors::BG_SURFACE)
                        .stroke(Stroke::new(
                            1.0,
                            HierarchosColors::ERROR.linear_multiply(0.4),
                        ))
                        .corner_radius(Rounding::same(6)),
                    )
                    .clicked()
                {
                    bridge.disconnect();
                }
            } else {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Connect Backend")
                                .color(HierarchosColors::TEXT_ON_PRIMARY)
                                .size(12.0),
                        )
                        .fill(HierarchosColors::SUCCESS)
                        .corner_radius(Rounding::same(6)),
                    )
                    .clicked()
                {
                    bridge.connect(&settings.python_path);
                }
            }
        });

        // Device selection
        ui.add_space(8.0);
        ui.label(
            RichText::new("Compute Device")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.horizontal(|ui| {
            for device in &settings.available_devices.clone() {
                let is_selected = settings.device == *device;
                let btn_fill = if is_selected {
                    get_accent().primary_dim
                } else {
                    HierarchosColors::BG_SURFACE
                };
                let text_color = if is_selected {
                    HierarchosColors::TEXT_ON_PRIMARY
                } else {
                    HierarchosColors::TEXT_SECONDARY
                };

                let hover = match device.as_str() {
                    "Auto" => "Uses CUDA on NVIDIA GPUs when available; otherwise CPU.",
                    "CUDA" => "NVIDIA CUDA acceleration. Requires a CUDA-capable GPU and driver.",
                    "CPU" => "Portable fallback for non-NVIDIA PCs and handhelds.",
                    "DirectML" => "Optional DirectML backend if torch-directml is installed.",
                    _ => "",
                };

                if ui
                    .add(
                        egui::Button::new(RichText::new(device).color(text_color).size(12.0))
                            .fill(btn_fill)
                            .stroke(Stroke::new(
                                1.0,
                                if is_selected {
                                    get_accent().primary
                                } else {
                                    HierarchosColors::BORDER_SUBTLE
                                },
                            ))
                            .corner_radius(Rounding::same(6)),
                    )
                    .on_hover_text(hover)
                    .clicked()
                {
                    settings.device = device.clone();
                }
            }
        });

        // Status
        ui.add_space(8.0);
        let status_text = if bridge.is_model_loaded() {
            ("● Model loaded", HierarchosColors::SUCCESS)
        } else if settings.pending_load {
            (
                "● Waiting for backend, then model will load",
                HierarchosColors::WARNING,
            )
        } else {
            ("○ No model loaded", HierarchosColors::TEXT_MUTED)
        };
        ui.label(RichText::new(status_text.0).color(status_text.1).size(12.0));

        // Auto-connect + load hint
        if !bridge.is_connected() && !settings.model_path.is_empty() {
            ui.add_space(4.0);
            ui.label(
                RichText::new("Use Connect & Load to start the backend and load this source.")
                    .color(HierarchosColors::WARNING)
                    .size(11.0),
            );
        }
    });
}

fn draw_learning_section(ui: &mut egui::Ui, settings: &mut AppSettings) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("🧠 Online Learning")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(15.0)
                .strong(),
        );
        ui.add_space(10.0);

        // Passive learning toggle
        ui.horizontal(|ui| {
            ui.checkbox(&mut settings.passive_learning, "");
            ui.label(
                RichText::new("Passive Learning")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(13.0),
            );
        });
        ui.label(
            RichText::new("Automatically update LTM after each generation turn based on surprise.")
                .color(HierarchosColors::TEXT_MUTED)
                .size(11.0),
        );

        ui.add_space(8.0);

        // Surprise threshold
        ui.label(
            RichText::new("Surprise Threshold")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.add(
            egui::Slider::new(&mut settings.surprise_threshold, 0.0..=5.0)
                .step_by(0.1)
                .text(""),
        );
        ui.label(
            RichText::new("Only learn when loss > threshold (higher = more conservative)")
                .color(HierarchosColors::TEXT_MUTED)
                .size(10.0),
        );

        ui.add_space(8.0);

        // Passive LR
        ui.label(
            RichText::new("Passive Learning Rate")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.add(
            egui::DragValue::new(&mut settings.passive_lr)
                .speed(1e-7)
                .range(1e-8..=1e-3)
                .max_decimals(8),
        );

        ui.add_space(8.0);

        // LTM LR
        ui.label(
            RichText::new("LTM Learning Rate")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.add(
            egui::DragValue::new(&mut settings.ltm_lr)
                .speed(1e-5)
                .range(1e-6..=0.1)
                .max_decimals(6),
        );
    });
}

fn draw_appearance_section(ui: &mut egui::Ui, settings: &mut AppSettings) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("🎨 Appearance")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(15.0)
                .strong(),
        );
        ui.add_space(10.0);

        // Font size
        ui.label(
            RichText::new("Font Size")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.add(egui::Slider::new(&mut settings.font_size, 10.0..=20.0).step_by(0.5));

        ui.add_space(8.0);

        // Accent color picker
        ui.label(
            RichText::new("Accent Color")
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.horizontal(|ui| {
            let colors = [
                ("Indigo", Color32::from_rgb(99, 102, 241)),
                ("Violet", Color32::from_rgb(139, 92, 246)),
                ("Cyan", Color32::from_rgb(34, 211, 238)),
                ("Emerald", Color32::from_rgb(16, 185, 129)),
                ("Rose", Color32::from_rgb(244, 63, 94)),
            ];

            for (i, (name, color)) in colors.iter().enumerate() {
                let is_selected = settings.accent_color_idx == i;
                let size = if is_selected { 24.0 } else { 20.0 };
                let (rect, response) =
                    ui.allocate_exact_size(Vec2::splat(size), egui::Sense::click());

                ui.painter().rect_filled(rect, Rounding::same(4), *color);
                if is_selected {
                    ui.painter().rect_stroke(
                        rect.expand(2.0),
                        Rounding::same(6),
                        Stroke::new(2.0, HierarchosColors::TEXT_PRIMARY),
                        egui::StrokeKind::Outside,
                    );
                }

                if response.clicked() {
                    settings.accent_color_idx = i;
                    theme::set_accent_index(i);
                }
                response.on_hover_text(*name);
            }
        });
    });
}

fn draw_about_section(ui: &mut egui::Ui) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_CARD)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .corner_radius(Rounding::same(10))
        .inner_margin(egui::Margin::same(16));

    frame.show(ui, |ui| {
        ui.label(
            RichText::new("ℹ About Hierarchos")
                .color(HierarchosColors::TEXT_PRIMARY)
                .size(15.0)
                .strong(),
        );
        ui.add_space(8.0);

        ui.label(
            RichText::new(
                "A Linear-Complexity Hierarchical Agent with Titans Memory and Vulkan Acceleration",
            )
            .color(HierarchosColors::TEXT_SECONDARY)
            .size(12.0)
            .italics(),
        );
        ui.add_space(4.0);

        ui.label(
            RichText::new(concat!(
                "Architecture: RWKV backbone + Titans Neural LTM + HRM Manager-Worker topology\n",
                "Inference: O(1) per token | Infinite context via surprise-based memory\n",
                "Backend: Vulkan compute shaders | INT4/Q4_0/Q2_K quantization\n",
                "Training: LoRA fine-tuning | Truncated BPTT | Gradient checkpointing"
            ))
            .color(HierarchosColors::TEXT_MUTED)
            .size(11.0)
            .monospace(),
        );
    });
}
