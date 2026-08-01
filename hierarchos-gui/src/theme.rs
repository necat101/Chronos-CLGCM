// theme.rs — Hierarchos Premium Dark Theme Engine
//
// Runtime accent color system with 5 curated palettes.
// All accent-dependent colors are resolved through get_accent().

use egui::{
    epaint::Shadow, Color32, CornerRadius as Rounding, FontFamily, FontId, Stroke, Style, Vec2,
    Visuals,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global accent palette index — updated from Settings, read everywhere.
static ACCENT_IDX: AtomicUsize = AtomicUsize::new(0);

/// An accent color palette (primary + derived shades).
#[derive(Clone, Copy)]
pub struct AccentPalette {
    pub primary: Color32,
    pub primary_dim: Color32,
    pub secondary: Color32,
    pub border: Color32,
    pub border_active: Color32,
    pub selection_bg: Color32,
}

/// The 5 curated accent palettes.
pub const ACCENT_PALETTES: [AccentPalette; 5] = [
    // 0: Indigo (default)
    AccentPalette {
        primary: Color32::from_rgb(99, 102, 241),
        primary_dim: Color32::from_rgb(67, 69, 180),
        secondary: Color32::from_rgb(139, 92, 246),
        border: Color32::from_rgba_premultiplied(99, 102, 241, 40),
        border_active: Color32::from_rgba_premultiplied(99, 102, 241, 100),
        selection_bg: Color32::from_rgba_premultiplied(99, 102, 241, 60),
    },
    // 1: Violet
    AccentPalette {
        primary: Color32::from_rgb(139, 92, 246),
        primary_dim: Color32::from_rgb(109, 62, 206),
        secondary: Color32::from_rgb(192, 132, 252),
        border: Color32::from_rgba_premultiplied(139, 92, 246, 40),
        border_active: Color32::from_rgba_premultiplied(139, 92, 246, 100),
        selection_bg: Color32::from_rgba_premultiplied(139, 92, 246, 60),
    },
    // 2: Cyan
    AccentPalette {
        primary: Color32::from_rgb(34, 211, 238),
        primary_dim: Color32::from_rgb(22, 163, 184),
        secondary: Color32::from_rgb(6, 182, 212),
        border: Color32::from_rgba_premultiplied(34, 211, 238, 40),
        border_active: Color32::from_rgba_premultiplied(34, 211, 238, 100),
        selection_bg: Color32::from_rgba_premultiplied(34, 211, 238, 60),
    },
    // 3: Emerald
    AccentPalette {
        primary: Color32::from_rgb(16, 185, 129),
        primary_dim: Color32::from_rgb(5, 150, 105),
        secondary: Color32::from_rgb(110, 231, 183),
        border: Color32::from_rgba_premultiplied(16, 185, 129, 40),
        border_active: Color32::from_rgba_premultiplied(16, 185, 129, 100),
        selection_bg: Color32::from_rgba_premultiplied(16, 185, 129, 60),
    },
    // 4: Rose
    AccentPalette {
        primary: Color32::from_rgb(244, 63, 94),
        primary_dim: Color32::from_rgb(190, 40, 70),
        secondary: Color32::from_rgb(253, 164, 175),
        border: Color32::from_rgba_premultiplied(244, 63, 94, 40),
        border_active: Color32::from_rgba_premultiplied(244, 63, 94, 100),
        selection_bg: Color32::from_rgba_premultiplied(244, 63, 94, 60),
    },
];

/// Get the currently active accent palette.
pub fn get_accent() -> &'static AccentPalette {
    let idx = ACCENT_IDX.load(Ordering::Relaxed);
    &ACCENT_PALETTES[idx.min(ACCENT_PALETTES.len() - 1)]
}

/// Set the accent palette index (called from Settings).
pub fn set_accent_index(idx: usize) {
    ACCENT_IDX.store(idx.min(ACCENT_PALETTES.len() - 1), Ordering::Relaxed);
}

/// Get the current accent index.
pub fn get_accent_index() -> usize {
    ACCENT_IDX.load(Ordering::Relaxed)
}

/// All non-accent colors used across the Hierarchos GUI.
pub struct HierarchosColors;

impl HierarchosColors {
    // Backgrounds
    pub const BG_DARKEST: Color32 = Color32::from_rgb(8, 8, 16);
    pub const BG_DARK: Color32 = Color32::from_rgb(12, 12, 22);
    pub const BG_SURFACE: Color32 = Color32::from_rgb(18, 18, 32);
    pub const BG_CARD: Color32 = Color32::from_rgb(22, 22, 40);
    pub const BG_CARD_HOVER: Color32 = Color32::from_rgb(28, 28, 50);
    pub const BG_INPUT: Color32 = Color32::from_rgb(14, 14, 28);

    // Semantic
    pub const SUCCESS: Color32 = Color32::from_rgb(16, 185, 129);
    pub const WARNING: Color32 = Color32::from_rgb(245, 158, 11);
    pub const ERROR: Color32 = Color32::from_rgb(239, 68, 68);
    pub const ACCENT_CYAN: Color32 = Color32::from_rgb(34, 211, 238);

    // Text
    pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(241, 245, 249);
    pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(148, 163, 184);
    pub const TEXT_MUTED: Color32 = Color32::from_rgb(100, 116, 139);
    pub const TEXT_ON_PRIMARY: Color32 = Color32::from_rgb(255, 255, 255);

    // Borders (non-accent)
    pub const BORDER_SUBTLE: Color32 = Color32::from_rgba_premultiplied(60, 60, 90, 60);

    // Chat
    pub const USER_BUBBLE: Color32 = Color32::from_rgb(30, 30, 55);
    pub const BOT_BUBBLE: Color32 = Color32::from_rgb(20, 20, 38);

    // Gradients
    pub const GRADIENT_HEADER_START: Color32 = Color32::from_rgb(15, 15, 30);
}

/// Apply the Hierarchos theme with the current accent palette.
pub fn apply_theme(ctx: &egui::Context) {
    let accent = get_accent();
    let mut style = Style::default();

    // Spacing
    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(16);
    style.spacing.button_padding = Vec2::new(14.0, 7.0);
    style.spacing.indent = 18.0;

    // Dark visuals base
    style.visuals = Visuals::dark();
    let v = &mut style.visuals;

    v.panel_fill = HierarchosColors::BG_DARK;
    v.window_fill = HierarchosColors::BG_SURFACE;
    v.extreme_bg_color = HierarchosColors::BG_INPUT;
    v.faint_bg_color = HierarchosColors::BG_CARD;
    v.code_bg_color = Color32::from_rgb(16, 16, 30);

    // Widget visuals — use accent colors
    v.widgets.noninteractive.bg_fill = HierarchosColors::BG_CARD;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, HierarchosColors::TEXT_SECONDARY);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE);
    v.widgets.noninteractive.corner_radius = Rounding::same(8);

    v.widgets.inactive.bg_fill = HierarchosColors::BG_CARD;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, HierarchosColors::TEXT_PRIMARY);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, HierarchosColors::BORDER_SUBTLE);
    v.widgets.inactive.corner_radius = Rounding::same(8);

    v.widgets.hovered.bg_fill = HierarchosColors::BG_CARD_HOVER;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, HierarchosColors::TEXT_PRIMARY);
    v.widgets.hovered.bg_stroke = Stroke::new(1.5, accent.primary);
    v.widgets.hovered.corner_radius = Rounding::same(8);

    v.widgets.active.bg_fill = accent.primary_dim;
    v.widgets.active.fg_stroke = Stroke::new(1.0, HierarchosColors::TEXT_ON_PRIMARY);
    v.widgets.active.bg_stroke = Stroke::new(1.5, accent.primary);
    v.widgets.active.corner_radius = Rounding::same(8);

    v.widgets.open.bg_fill = HierarchosColors::BG_CARD;
    v.widgets.open.fg_stroke = Stroke::new(1.0, accent.primary);
    v.widgets.open.bg_stroke = Stroke::new(1.0, accent.border_active);
    v.widgets.open.corner_radius = Rounding::same(8);

    // Selection
    v.selection.bg_fill = accent.selection_bg;
    v.selection.stroke = Stroke::new(1.0, accent.primary);

    // Shadows
    v.popup_shadow = Shadow {
        offset: [0, 4],
        blur: 16,
        spread: 2,
        color: Color32::from_rgba_premultiplied(0, 0, 0, 80),
    };
    v.window_shadow = Shadow {
        offset: [0, 8],
        blur: 24,
        spread: 4,
        color: Color32::from_rgba_premultiplied(0, 0, 0, 100),
    };

    // Window rounding
    v.window_corner_radius = Rounding::same(12);
    v.menu_corner_radius = Rounding::same(8);

    // Text cursor — accent colored
    v.text_cursor.stroke = Stroke::new(2.0, accent.primary);

    // Striped rows
    v.striped = true;

    // Scrollbar
    v.handle_shape = egui::style::HandleShape::Rect { aspect_ratio: 0.5 };

    ctx.set_style(style);
}

/// Set up custom fonts.
pub fn setup_fonts(ctx: &egui::Context) {
    let fonts = egui::FontDefinitions::default();
    ctx.set_fonts(fonts);

    use egui::TextStyle;
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (
            TextStyle::Heading,
            FontId::new(22.0, FontFamily::Proportional),
        ),
        (TextStyle::Body, FontId::new(14.0, FontFamily::Proportional)),
        (
            TextStyle::Monospace,
            FontId::new(13.0, FontFamily::Monospace),
        ),
        (
            TextStyle::Button,
            FontId::new(14.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Small,
            FontId::new(11.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Name("heading2".into()),
            FontId::new(18.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Name("heading3".into()),
            FontId::new(15.0, FontFamily::Proportional),
        ),
        (
            TextStyle::Name("large_mono".into()),
            FontId::new(16.0, FontFamily::Monospace),
        ),
        (
            TextStyle::Name("metric_value".into()),
            FontId::new(28.0, FontFamily::Monospace),
        ),
    ]
    .into();
    ctx.set_style(style);
}
