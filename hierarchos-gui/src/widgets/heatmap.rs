// widgets/heatmap.rs — LTM Memory Heatmap Visualization

use crate::theme::HierarchosColors;
use egui::{self, Color32, CornerRadius as Rounding, Pos2, Rect, RichText, Stroke, Ui, Vec2};

/// Renders a heatmap grid for LTM memory values.
pub fn draw_heatmap(ui: &mut Ui, data: &[Vec<f32>], title: &str, width: f32, height: f32) {
    if data.is_empty() {
        ui.label(
            RichText::new("No memory data available")
                .color(HierarchosColors::TEXT_MUTED)
                .size(13.0),
        );
        return;
    }

    let rows = data.len();
    let cols = data[0].len();

    // Find min/max for normalization
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for row in data {
        for &val in row {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
    }
    let range = (max_val - min_val).max(1e-8);

    // Title
    ui.label(
        RichText::new(title)
            .color(HierarchosColors::TEXT_SECONDARY)
            .size(13.0)
            .strong(),
    );
    ui.add_space(4.0);

    let desired_size = Vec2::new(width, height);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    // Paint main heatmap
    {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, Rounding::same(6), HierarchosColors::BG_INPUT);

        let cell_w = rect.width() / cols as f32;
        let cell_h = rect.height() / rows as f32;

        for (r, row) in data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let norm = ((val - min_val) / range).clamp(0.0, 1.0);
                let color = heatmap_color(norm);

                let cell_rect = Rect::from_min_size(
                    Pos2::new(
                        rect.min.x + c as f32 * cell_w,
                        rect.min.y + r as f32 * cell_h,
                    ),
                    Vec2::new(cell_w, cell_h),
                );

                painter.rect_filled(cell_rect, Rounding::ZERO, color);
            }
        }

        // Border
        painter.rect_stroke(
            rect,
            Rounding::same(6),
            Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE),
            egui::StrokeKind::Outside,
        );
    }

    // Tooltip on hover
    let cell_w = rect.width() / cols as f32;
    let cell_h = rect.height() / rows as f32;
    if let Some(pos) = response.hover_pos() {
        let col = ((pos.x - rect.min.x) / cell_w) as usize;
        let row = ((pos.y - rect.min.y) / cell_h) as usize;
        if row < rows && col < cols {
            let val = data[row][col];
            egui::show_tooltip(
                ui.ctx(),
                ui.layer_id(),
                response.id.with("heatmap_tt"),
                |ui| {
                    ui.label(
                        RichText::new(format!("Slot {} | Dim {} | Value: {:.4}", row, col, val))
                            .color(HierarchosColors::TEXT_PRIMARY)
                            .size(12.0),
                    );
                },
            );
        }
    }

    // Color legend
    ui.add_space(4.0);
    let legend_height = 12.0;
    let legend_width = width.min(300.0);
    let (legend_rect, _) =
        ui.allocate_exact_size(Vec2::new(legend_width, legend_height), egui::Sense::hover());

    {
        let painter = ui.painter();
        for x in 0..legend_width as u32 {
            let t = x as f32 / legend_width;
            let color = heatmap_color(t);
            let pixel_rect = Rect::from_min_size(
                Pos2::new(legend_rect.min.x + x as f32, legend_rect.min.y),
                Vec2::new(1.0, legend_height),
            );
            painter.rect_filled(pixel_rect, Rounding::ZERO, color);
        }
    }

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(format!("{:.3}", min_val))
                .color(HierarchosColors::TEXT_MUTED)
                .size(10.0),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                RichText::new(format!("{:.3}", max_val))
                    .color(HierarchosColors::TEXT_MUTED)
                    .size(10.0),
            );
        });
    });
}

/// Convert a normalized value [0, 1] to a heatmap color.
fn heatmap_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);

    // 5-stop gradient: deep blue → indigo → cyan → green → gold
    let stops: [(f32, (u8, u8, u8)); 5] = [
        (0.0, (8, 8, 40)),      // Deep blue
        (0.25, (60, 50, 160)),  // Indigo
        (0.5, (30, 180, 220)),  // Cyan
        (0.75, (16, 185, 129)), // Emerald
        (1.0, (245, 158, 11)),  // Amber
    ];

    // Find the two stops to interpolate between
    let mut lower = 0;
    for i in 0..stops.len() - 1 {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            lower = i;
            break;
        }
    }

    let (t0, c0) = stops[lower];
    let (t1, c1) = stops[lower + 1];
    let local_t = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);

    let r = (c0.0 as f32 + (c1.0 as f32 - c0.0 as f32) * local_t) as u8;
    let g = (c0.1 as f32 + (c1.1 as f32 - c0.1 as f32) * local_t) as u8;
    let b = (c0.2 as f32 + (c1.2 as f32 - c0.2 as f32) * local_t) as u8;

    Color32::from_rgb(r, g, b)
}
