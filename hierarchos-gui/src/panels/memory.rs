// panels/memory.rs — LTM Memory Visualizer Panel
//
// Visualizes the Titans-style neural memory with heatmaps for
// fast vals (working memory) and slow vals (consolidated memory).

use crate::bridge::PythonBridge;
use crate::theme::{get_accent, HierarchosColors};
use crate::widgets::heatmap::draw_heatmap;
use egui::{self, CornerRadius as Rounding, RichText, ScrollArea, Stroke, Vec2};

/// Memory visualizer state.
pub struct MemoryVisualizerState {
    pub fast_vals: Vec<Vec<f32>>,
    pub slow_vals: Vec<Vec<f32>>,
    pub timestamps: Vec<f32>,
    pub sources: Vec<i32>,
    pub view_mode: MemoryView,
    pub auto_refresh: bool,
    pub refresh_interval_ms: u64,
    pub last_refresh: std::time::Instant,
}

#[derive(PartialEq, Clone, Copy)]
pub enum MemoryView {
    FastVals,
    SlowVals,
    Combined,
    Difference,
}

impl Default for MemoryVisualizerState {
    fn default() -> Self {
        Self {
            fast_vals: Vec::new(),
            slow_vals: Vec::new(),
            timestamps: Vec::new(),
            sources: Vec::new(),
            view_mode: MemoryView::FastVals,
            auto_refresh: false,
            refresh_interval_ms: 2000,
            last_refresh: std::time::Instant::now(),
        }
    }
}

impl MemoryVisualizerState {
    pub fn on_snapshot(
        &mut self,
        fast: Vec<Vec<f32>>,
        slow: Vec<Vec<f32>>,
        timestamps: Vec<f32>,
        sources: Vec<i32>,
    ) {
        self.fast_vals = fast;
        self.slow_vals = slow;
        self.timestamps = timestamps;
        self.sources = sources;
        self.last_refresh = std::time::Instant::now();
    }

    fn combined_vals(&self) -> Vec<Vec<f32>> {
        if self.fast_vals.len() != self.slow_vals.len() {
            return self.fast_vals.clone();
        }
        self.fast_vals
            .iter()
            .zip(&self.slow_vals)
            .map(|(f, s)| f.iter().zip(s).map(|(fv, sv)| fv + sv).collect())
            .collect()
    }

    fn difference_vals(&self) -> Vec<Vec<f32>> {
        if self.fast_vals.len() != self.slow_vals.len() {
            return self.fast_vals.clone();
        }
        self.fast_vals
            .iter()
            .zip(&self.slow_vals)
            .map(|(f, s)| f.iter().zip(s).map(|(fv, sv)| fv - sv).collect())
            .collect()
    }
}

/// Draw the memory visualizer panel.
pub fn draw_memory_panel(
    ui: &mut egui::Ui,
    state: &mut MemoryVisualizerState,
    bridge: &PythonBridge,
) {
    let can_snapshot = bridge.is_model_loaded()
        && !bridge.is_generating()
        && !bridge.is_training()
        && !bridge.is_loading();
    ui.vertical(|ui| {
        // Header
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("🧠 LTM Memory Visualizer")
                    .color(HierarchosColors::TEXT_PRIMARY)
                    .size(18.0)
                    .strong(),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Auto-refresh toggle
                ui.checkbox(&mut state.auto_refresh, "");
                ui.label(
                    RichText::new("Auto-refresh")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                );

                ui.separator();

                // Manual refresh
                if ui.add_enabled(can_snapshot, egui::Button::new(
                    RichText::new("↻ Snapshot")
                        .color(HierarchosColors::TEXT_SECONDARY)
                        .size(12.0),
                ).fill(HierarchosColors::BG_CARD)
                 .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                 .corner_radius(Rounding::same(6)))
                .clicked() {
                    bridge.request_ltm_snapshot();
                }
            });
        });

        ui.add_space(8.0);

        // Auto-refresh logic
        if can_snapshot
            && state.auto_refresh
            && state.last_refresh.elapsed().as_millis() > state.refresh_interval_ms as u128
        {
            bridge.request_ltm_snapshot();
        }

        // View mode selector
        ui.horizontal(|ui| {
            let modes = [
                (MemoryView::FastVals, "Fast Vals (Working Memory)"),
                (MemoryView::SlowVals, "Slow Vals (Consolidated)"),
                (MemoryView::Combined, "Combined (Fast + Slow)"),
                (MemoryView::Difference, "Difference (Fast − Slow)"),
            ];

            for (mode, label) in &modes {
                let is_selected = state.view_mode == *mode;
                let btn_fill = if is_selected { get_accent().primary_dim } else { HierarchosColors::BG_CARD };
                let text_color = if is_selected { HierarchosColors::TEXT_ON_PRIMARY } else { HierarchosColors::TEXT_SECONDARY };

                if ui.add(egui::Button::new(
                    RichText::new(*label).color(text_color).size(11.0),
                ).fill(btn_fill)
                 .stroke(Stroke::new(1.0, if is_selected { get_accent().primary } else { HierarchosColors::BORDER_SUBTLE }))
                 .corner_radius(Rounding::same(6)))
                .clicked() {
                    state.view_mode = *mode;
                }
            }
        });

        ui.add_space(8.0);

        if state.fast_vals.is_empty() {
            // Empty state
            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_CARD)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(12))
                .inner_margin(egui::Margin::same(40));

            frame.show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("🔮").size(48.0));
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("No memory data available")
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(16.0),
                    );
                    ui.label(
                        RichText::new("Click 'Snapshot' to capture the current LTM state, or enable auto-refresh.")
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(13.0),
                    );
                });
            });
            return;
        }

        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let available_width = ui.available_width() - 20.0;
            let heatmap_height = 300.0;

            // Main heatmap
            let _data = match state.view_mode {
                MemoryView::FastVals => &state.fast_vals,
                MemoryView::SlowVals => &state.slow_vals,
                MemoryView::Combined => &state.combined_vals(),
                MemoryView::Difference => &state.difference_vals(),
            };

            let title = match state.view_mode {
                MemoryView::FastVals => "Fast Values (Working Memory) — Actively updated at test time",
                MemoryView::SlowVals => "Slow Values (Consolidated) — Trained parameters",
                MemoryView::Combined => "Combined Memory (Slow + Fast) — Effective retrieval state",
                MemoryView::Difference => "Delta (Fast − Slow) — Runtime learning signal",
            };

            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_CARD)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(10))
                .inner_margin(egui::Margin::same(14));

            frame.show(ui, |ui| {
                // Need to clone for Difference/Combined since they return owned Vecs
                let display_data = match state.view_mode {
                    MemoryView::FastVals => state.fast_vals.clone(),
                    MemoryView::SlowVals => state.slow_vals.clone(),
                    MemoryView::Combined => state.combined_vals(),
                    MemoryView::Difference => state.difference_vals(),
                };
                draw_heatmap(ui, &display_data, title, available_width - 28.0, heatmap_height);
            });

            ui.add_space(12.0);

            // Slot metadata
            let frame = egui::Frame::new()
                .fill(HierarchosColors::BG_CARD)
                .stroke(Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE))
                .corner_radius(Rounding::same(10))
                .inner_margin(egui::Margin::same(14));

            frame.show(ui, |ui| {
                ui.label(
                    RichText::new("Slot Metadata")
                        .color(HierarchosColors::TEXT_PRIMARY)
                        .size(14.0)
                        .strong(),
                );
                ui.add_space(6.0);

                // Summary stats
                ui.horizontal(|ui| {
                    let total_slots = state.fast_vals.len();
                    let active_slots = state.timestamps.iter().filter(|&&t| t > 0.0).count();
                    let user_slots = state.sources.iter().filter(|&&s| s == 1).count();
                    let training_slots = state.sources.iter().filter(|&&s| s == 2).count();

                    ui.label(RichText::new(format!("Total: {}", total_slots))
                        .color(HierarchosColors::TEXT_SECONDARY).size(12.0));
                    ui.separator();
                    ui.label(RichText::new(format!("Active: {}", active_slots))
                        .color(HierarchosColors::SUCCESS).size(12.0));
                    ui.separator();
                    ui.label(RichText::new(format!("User: {}", user_slots))
                        .color(get_accent().primary).size(12.0));
                    ui.separator();
                    ui.label(RichText::new(format!("Training: {}", training_slots))
                        .color(HierarchosColors::WARNING).size(12.0));
                });

                ui.add_space(4.0);

                // Source distribution bar
                let total = state.sources.len().max(1) as f32;
                let user_frac = state.sources.iter().filter(|&&s| s == 1).count() as f32 / total;
                let train_frac = state.sources.iter().filter(|&&s| s == 2).count() as f32 / total;
                let _unknown_frac = 1.0 - user_frac - train_frac;

                let bar_height = 8.0;
                let bar_width = ui.available_width();
                let (rect, _) = ui.allocate_exact_size(Vec2::new(bar_width, bar_height), egui::Sense::hover());

                let painter = ui.painter();
                painter.rect_filled(rect, Rounding::same(4), HierarchosColors::BG_INPUT);

                // User slots (indigo)
                if user_frac > 0.0 {
                    let user_rect = egui::Rect::from_min_size(
                        rect.min,
                        Vec2::new(bar_width * user_frac, bar_height),
                    );
                    painter.rect_filled(user_rect, Rounding::same(4), get_accent().primary);
                }

                // Training slots (amber)
                if train_frac > 0.0 {
                    let train_rect = egui::Rect::from_min_size(
                        rect.min + Vec2::new(bar_width * user_frac, 0.0),
                        Vec2::new(bar_width * train_frac, bar_height),
                    );
                    painter.rect_filled(train_rect, Rounding::ZERO, HierarchosColors::WARNING);
                }
            });
        });
    });
}
