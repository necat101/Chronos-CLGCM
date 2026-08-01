// widgets/metric_card.rs — Dashboard metric card with glassmorphism styling

use crate::theme::HierarchosColors;
use egui::{self, Color32, CornerRadius as Rounding, RichText, Stroke, TextStyle, Ui, Vec2};

pub struct MetricCard<'a> {
    pub title: &'a str,
    pub value: &'a str,
    pub subtitle: Option<&'a str>,
    pub accent_color: Color32,
    pub icon: &'a str,
}

impl<'a> MetricCard<'a> {
    pub fn show(&self, ui: &mut Ui, width: f32) {
        let frame = egui::Frame::new()
            .fill(HierarchosColors::BG_CARD)
            .stroke(Stroke::new(1.0, self.accent_color.linear_multiply(0.2)))
            .corner_radius(Rounding::same(12))
            .inner_margin(egui::Margin::same(16));

        ui.allocate_ui(Vec2::new(width, 0.0), |ui| {
            frame.show(ui, |ui| {
                ui.set_width(width - 34.0);

                // Icon and title row
                ui.horizontal(|ui| {
                    ui.label(RichText::new(self.icon).size(16.0).color(self.accent_color));
                    ui.label(
                        RichText::new(self.title)
                            .color(HierarchosColors::TEXT_SECONDARY)
                            .size(12.0),
                    );
                });

                ui.add_space(6.0);

                // Value
                ui.label(
                    RichText::new(self.value)
                        .color(HierarchosColors::TEXT_PRIMARY)
                        .text_style(TextStyle::Name("metric_value".into())),
                );

                // Optional subtitle
                if let Some(sub) = self.subtitle {
                    ui.add_space(2.0);
                    ui.label(
                        RichText::new(sub)
                            .color(HierarchosColors::TEXT_MUTED)
                            .size(11.0),
                    );
                }
            });
        });
    }
}

/// A compact progress bar with label.
pub fn progress_bar_labeled(ui: &mut Ui, label: &str, progress: f32, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .color(HierarchosColors::TEXT_SECONDARY)
                .size(12.0),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{:.1}%", progress * 100.0))
                    .color(HierarchosColors::TEXT_MUTED)
                    .size(11.0),
            );
        });
    });

    let desired_size = Vec2::new(ui.available_width(), 6.0);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    // Background
    ui.painter()
        .rect_filled(rect, Rounding::same(3), HierarchosColors::BG_INPUT);

    // Fill
    let fill_width = rect.width() * progress.clamp(0.0, 1.0);
    let fill_rect = egui::Rect::from_min_size(rect.min, Vec2::new(fill_width, rect.height()));
    ui.painter()
        .rect_filled(fill_rect, Rounding::same(3), color);
}
