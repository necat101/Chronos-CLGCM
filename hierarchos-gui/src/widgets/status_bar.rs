// widgets/status_bar.rs — Bottom status bar with device info and metrics

use crate::theme::HierarchosColors;
use egui::{self, RichText, Stroke, Ui};

pub struct StatusBarInfo {
    pub device: String,
    pub model_status: String,
    pub tokens_generated: u64,
    pub tokens_per_sec: Option<f64>,
    pub vram_total_mb: Option<f64>,
    pub connected: bool,
}

pub fn draw_status_bar(ui: &mut Ui, info: &StatusBarInfo) {
    let frame = egui::Frame::new()
        .fill(HierarchosColors::BG_DARKEST)
        .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
        .inner_margin(egui::Margin::symmetric(16, 6));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            // Connection indicator
            let (dot_color, status_text) = if info.connected {
                (HierarchosColors::SUCCESS, "●")
            } else {
                (HierarchosColors::ERROR, "○")
            };
            ui.label(RichText::new(status_text).color(dot_color).size(10.0));
            ui.label(
                RichText::new(&info.model_status)
                    .color(HierarchosColors::TEXT_SECONDARY)
                    .size(11.0),
            );

            ui.separator();

            // Device
            ui.label(
                RichText::new(format!("⚡ {}", info.device))
                    .color(HierarchosColors::ACCENT_CYAN)
                    .size(11.0),
            );

            ui.separator();

            // Token count
            ui.label(
                RichText::new(format!("Tokens: {}", format_number(info.tokens_generated)))
                    .color(HierarchosColors::TEXT_MUTED)
                    .size(11.0),
            );

            // Fill remaining space
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // VRAM
                if let Some(vram) = info.vram_total_mb {
                    ui.label(
                        RichText::new(format!("VRAM: {:.1} GB", vram / 1024.0))
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0),
                    );
                    ui.separator();
                }

                // Tokens/sec
                if let Some(tps) = info.tokens_per_sec {
                    ui.label(
                        RichText::new(format!("{:.1} tok/s", tps))
                            .color(HierarchosColors::SUCCESS)
                            .size(11.0),
                    );
                    ui.separator();
                }
            });
        });
    });
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
