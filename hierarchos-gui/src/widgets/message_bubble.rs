// widgets/message_bubble.rs — Chat message bubble with glassmorphism styling

use crate::theme::{get_accent, HierarchosColors};
use egui::{self, Color32, CornerRadius as Rounding, RichText, Stroke, Ui, Vec2};

pub enum MessageRole {
    User,
    Assistant,
    System,
}

pub struct MessageBubble<'a> {
    pub role: MessageRole,
    pub content: &'a str,
    pub timestamp: &'a str,
    pub is_streaming: bool,
}

impl<'a> MessageBubble<'a> {
    pub fn show(&self, ui: &mut Ui) -> egui::Response {
        let available_width = ui.available_width();
        let max_bubble_width = (available_width * 0.80).min(800.0);

        let (bg_color, border_color, label_text, label_color, align_right) = match self.role {
            MessageRole::User => (
                HierarchosColors::USER_BUBBLE,
                get_accent().primary,
                "You",
                get_accent().primary,
                true,
            ),
            MessageRole::Assistant => (
                HierarchosColors::BOT_BUBBLE,
                get_accent().secondary,
                "Hierarchos",
                get_accent().secondary,
                false,
            ),
            MessageRole::System => (
                Color32::from_rgb(20, 25, 20),
                HierarchosColors::SUCCESS,
                "System",
                HierarchosColors::SUCCESS,
                false,
            ),
        };

        let response = ui.allocate_ui_with_layout(
            Vec2::new(available_width, 0.0),
            if align_right {
                egui::Layout::right_to_left(egui::Align::TOP)
            } else {
                egui::Layout::left_to_right(egui::Align::TOP)
            },
            |ui| {
                ui.allocate_ui_with_layout(
                    Vec2::new(max_bubble_width, 0.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let frame = egui::Frame::new()
                            .fill(bg_color)
                            .stroke(Stroke::new(1.0, border_color.linear_multiply(0.3)))
                            .corner_radius(Rounding {
                                nw: if align_right { 12 } else { 4 },
                                ne: if align_right { 4 } else { 12 },
                                sw: 12,
                                se: 12,
                            })
                            .inner_margin(egui::Margin {
                                left: 14,
                                right: 14,
                                top: 10,
                                bottom: 10,
                            });

                        frame.show(ui, |ui| {
                            let content_width = (max_bubble_width - 28.0).max(96.0);
                            ui.set_min_width(content_width);
                            ui.set_max_width(content_width);

                            // Role label
                            ui.horizontal(|ui| {
                                ui.set_width(content_width);
                                ui.label(
                                    RichText::new(label_text)
                                        .color(label_color)
                                        .size(11.0)
                                        .strong(),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.label(
                                            RichText::new(self.timestamp)
                                                .color(HierarchosColors::TEXT_MUTED)
                                                .size(10.0),
                                        );
                                    },
                                );
                            });

                            ui.add_space(4.0);

                            // Message content
                            let text = if self.is_streaming {
                                format!("{}▊", self.content)
                            } else {
                                self.content.to_string()
                            };

                            ui.add(
                                egui::Label::new(
                                    RichText::new(&text)
                                        .color(HierarchosColors::TEXT_PRIMARY)
                                        .size(14.0),
                                )
                                .wrap(),
                            );
                        });
                    },
                );
            },
        );

        response.response
    }
}

/// Draw a typing indicator animation (three pulsing dots).
pub fn typing_indicator(ui: &mut Ui) {
    let time = ui.input(|i| i.time);
    let available_width = ui.available_width();

    ui.allocate_ui(Vec2::new(available_width.min(120.0), 30.0), |ui| {
        let frame = egui::Frame::new()
            .fill(HierarchosColors::BOT_BUBBLE)
            .stroke(Stroke::new(
                1.0,
                get_accent().secondary.linear_multiply(0.2),
            ))
            .corner_radius(Rounding::same(12))
            .inner_margin(egui::Margin::symmetric(14, 8));

        frame.show(ui, |ui| {
            ui.horizontal(|ui| {
                for i in 0..3 {
                    let phase = time * 3.0 + i as f64 * 0.8;
                    let alpha = ((phase.sin() + 1.0) * 0.5 * 200.0) as u8 + 55;
                    let dot_color = Color32::from_rgba_premultiplied(
                        get_accent().secondary.r(),
                        get_accent().secondary.g(),
                        get_accent().secondary.b(),
                        alpha,
                    );

                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                }
            });
        });
    });

    ui.ctx().request_repaint();
}
