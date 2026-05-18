use egui::{Color32, Painter, Pos2, Rect, Stroke};

use crate::acoustics::GridPoint;
use crate::renderer::viridis_color;

/// Default dynamic range for legend dB span (0 dB top, -60 dB bottom).
pub const DEFAULT_LEGEND_DB_RANGE: f32 = 60.0;

/// Vertical colorbar configuration. `min_db`/`max_db` are absolute log-dB
/// boundaries (max_db = top of bar, min_db = bottom). `use_log` toggles
/// linear-vs-log mapping for the tick labels and value→t function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorLegend {
    pub min_db: f32,
    pub max_db: f32,
    pub use_log: bool,
}

impl Default for ColorLegend {
    fn default() -> Self {
        Self {
            min_db: -DEFAULT_LEGEND_DB_RANGE,
            max_db: 0.0,
            use_log: true,
        }
    }
}

impl ColorLegend {
    /// Auto-calibrate range to fit the scene's energy span. `max_grid_energy`
    /// is taken as the 0-dB reference. min_db is fixed at the default range
    /// below that.
    pub fn calibrate_to_grid(grid: &[GridPoint]) -> Self {
        let max_e = grid
            .iter()
            .map(|p| p.energy)
            .fold(0.0_f32, f32::max)
            .max(1e-12);
        // After calibration, max_e is mapped to 0 dB and min_e to -range dB.
        let mut legend = Self::default();
        // Compute actual min (non-zero) for diagnostics — but bound at -range.
        let min_nonzero = grid
            .iter()
            .map(|p| p.energy)
            .filter(|e| *e > 0.0)
            .fold(f32::MAX, f32::min);
        if min_nonzero < f32::MAX {
            let ratio = (min_nonzero / max_e).max(1e-12);
            let min_db = 10.0 * ratio.log10();
            legend.min_db = min_db.max(-DEFAULT_LEGEND_DB_RANGE * 2.0);
        }
        legend
    }

    /// Map an energy value (linear) to a 0..1 t parameter for colormap lookup.
    /// `max_energy` is the reference (0 dB) value.
    pub fn energy_to_t(&self, energy: f32, max_energy: f32) -> f32 {
        if energy <= 0.0 || max_energy <= 0.0 {
            return 0.0;
        }
        let span = (self.max_db - self.min_db).max(1e-6);
        if self.use_log {
            let ratio = (energy / max_energy).max(1e-12);
            let db = 10.0 * ratio.log10();
            ((db - self.min_db) / span).clamp(0.0, 1.0)
        } else {
            // Linear: energy / max maps to (0..1) directly, ignoring dB bounds.
            (energy / max_energy).clamp(0.0, 1.0)
        }
    }

    /// Map a dB value to a 0..1 t for tick labels.
    pub fn db_to_t(&self, db: f32) -> f32 {
        let span = (self.max_db - self.min_db).max(1e-6);
        ((db - self.min_db) / span).clamp(0.0, 1.0)
    }

    /// Generate evenly-spaced dB tick labels for the colorbar. Always includes
    /// min_db and max_db.
    pub fn tick_labels(&self, count: usize) -> Vec<f32> {
        if count < 2 {
            return vec![self.max_db];
        }
        (0..count)
            .map(|i| {
                let t = i as f32 / (count - 1) as f32;
                self.min_db + t * (self.max_db - self.min_db)
            })
            .collect()
    }
}

/// Render the vertical colorbar (right edge) into `bar_rect`. Caller supplies
/// the egui painter and the bar's screen-space rect; this draws filled segments
/// stepping through the colormap.
pub fn render_color_legend(legend: &ColorLegend, painter: &Painter, bar_rect: Rect, steps: usize) {
    let steps = steps.max(2);
    let h = bar_rect.height();
    for i in 0..steps {
        let t_lo = i as f32 / steps as f32;
        let t_hi = (i + 1) as f32 / steps as f32;
        let t_mid = (t_lo + t_hi) * 0.5;
        let color = viridis_color(t_mid);
        // Bar is drawn top=high, bottom=low → flip y.
        let y_top = bar_rect.bottom() - t_hi * h;
        let y_bot = bar_rect.bottom() - t_lo * h;
        let seg = Rect::from_min_max(
            Pos2::new(bar_rect.left(), y_top),
            Pos2::new(bar_rect.right(), y_bot.min(bar_rect.bottom())),
        );
        painter.rect_filled(seg, 0.0, color);
    }
    // Outline for legibility on any background.
    painter.rect_stroke(
        bar_rect,
        0.0,
        Stroke::new(1.0, Color32::from_rgb(220, 220, 220)),
        egui::epaint::StrokeKind::Outside,
    );
    let _ = legend; // axis labels rendered by the surrounding UI code
}

/// One row for the material absorption legend: material name + absorption
/// coefficient per band (6 values, 125Hz..4kHz).
#[derive(Clone, Debug)]
pub struct MaterialLegendRow {
    pub name: String,
    pub absorption: [f32; 6],
}

/// Render the material legend panel: one row per material, with 6 horizontal
/// bars (one per band) whose width is proportional to the absorption coefficient
/// (0..1).
pub fn render_material_legend(rows: &[MaterialLegendRow], painter: &Painter, panel_rect: Rect) {
    if rows.is_empty() {
        return;
    }
    let row_h = (panel_rect.height() / rows.len() as f32).max(8.0);
    for (i, row) in rows.iter().enumerate() {
        let y_top = panel_rect.top() + i as f32 * row_h;
        let row_rect = Rect::from_min_max(
            Pos2::new(panel_rect.left(), y_top),
            Pos2::new(panel_rect.right(), y_top + row_h),
        );
        // 6 horizontal bars per row, evenly stacked
        let band_h = (row_rect.height() / 6.0).max(2.0);
        for (band_idx, &abs) in row.absorption.iter().enumerate() {
            let abs = abs.clamp(0.0, 1.0);
            let by_top = row_rect.top() + band_idx as f32 * band_h;
            let bar_w = abs * row_rect.width().max(1.0);
            let bar = Rect::from_min_max(
                Pos2::new(row_rect.left(), by_top),
                Pos2::new(row_rect.left() + bar_w, by_top + band_h),
            );
            let t = band_idx as f32 / 5.0;
            painter.rect_filled(bar, 0.0, viridis_color(t));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn gp(energy: f32, x: f32) -> GridPoint {
        GridPoint {
            position: Vec3::new(x, 0.0, 0.0),
            energy,
        }
    }

    #[test]
    fn test_color_legend_default_range_60db() {
        let l = ColorLegend::default();
        assert!((l.max_db - 0.0).abs() < 1e-6);
        assert!((l.min_db - -60.0).abs() < 1e-6);
        assert!(l.use_log);
    }

    #[test]
    fn test_color_legend_calibrate_empty_grid_uses_default() {
        let l = ColorLegend::calibrate_to_grid(&[]);
        assert_eq!(l, ColorLegend::default());
    }

    #[test]
    fn test_color_legend_calibrate_clamps_min_to_double_range() {
        // A grid with a wildly low non-zero (1e-40) would push min_db < -range*2;
        // legend should clamp.
        let grid = vec![gp(1.0, 0.0), gp(1e-40, 1.0)];
        let l = ColorLegend::calibrate_to_grid(&grid);
        assert!(
            l.min_db >= -DEFAULT_LEGEND_DB_RANGE * 2.0 - 1e-3,
            "min_db should be clamped: {}",
            l.min_db
        );
    }

    #[test]
    fn test_color_legend_energy_to_t_zero_returns_zero() {
        let l = ColorLegend::default();
        assert_eq!(l.energy_to_t(0.0, 1.0), 0.0);
    }

    #[test]
    fn test_color_legend_energy_to_t_max_returns_one() {
        let l = ColorLegend::default();
        let t = l.energy_to_t(1.0, 1.0);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_color_legend_energy_to_t_log_30db_returns_half() {
        let l = ColorLegend::default();
        // 1e-3 = -30 dB → t = 0.5 with -60..0 range
        let t = l.energy_to_t(1e-3, 1.0);
        assert!((t - 0.5).abs() < 1e-3, "expected 0.5, got {t}");
    }

    #[test]
    fn test_color_legend_energy_to_t_linear_is_ratio() {
        let l = ColorLegend {
            use_log: false,
            ..ColorLegend::default()
        };
        assert!((l.energy_to_t(0.5, 1.0) - 0.5).abs() < 1e-5);
        assert!((l.energy_to_t(0.25, 1.0) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_color_legend_db_to_t_endpoints() {
        let l = ColorLegend::default();
        assert!(l.db_to_t(l.max_db) > 0.999);
        assert!(l.db_to_t(l.min_db) < 0.001);
    }

    #[test]
    fn test_color_legend_db_to_t_clamps_out_of_range() {
        let l = ColorLegend::default();
        assert_eq!(l.db_to_t(100.0), 1.0);
        assert_eq!(l.db_to_t(-1000.0), 0.0);
    }

    #[test]
    fn test_color_legend_tick_labels_include_endpoints() {
        let l = ColorLegend::default();
        let labels = l.tick_labels(5);
        assert_eq!(labels.len(), 5);
        assert!((labels[0] - l.min_db).abs() < 1e-5);
        assert!((labels[4] - l.max_db).abs() < 1e-5);
    }

    #[test]
    fn test_color_legend_tick_labels_evenly_spaced() {
        let l = ColorLegend::default();
        let labels = l.tick_labels(5);
        let span = labels[1] - labels[0];
        for w in labels.windows(2) {
            assert!(
                (w[1] - w[0] - span).abs() < 1e-3,
                "labels not evenly spaced"
            );
        }
    }

    #[test]
    fn test_material_legend_row_holds_six_bands() {
        let row = MaterialLegendRow {
            name: "concrete".into(),
            absorption: [0.01, 0.02, 0.04, 0.05, 0.06, 0.07],
        };
        assert_eq!(row.absorption.len(), 6);
    }

    #[test]
    fn test_color_legend_calibrate_grid_no_panic_with_zero_energy() {
        let grid = vec![gp(0.0, 0.0), gp(0.0, 1.0)];
        let _ = ColorLegend::calibrate_to_grid(&grid);
    }
}
