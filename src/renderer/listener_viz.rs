use egui::{Color32, Painter, Pos2, Rect, Stroke};
use glam::Vec3;

use crate::acoustics::GridPoint;
use crate::renderer::{project_3d, Camera};

/// Default listener capture sphere radius (meters). Energy from grid samples
/// within this radius is integrated to estimate captured broadband SPL.
pub const DEFAULT_LISTENER_CAPTURE_RADIUS: f32 = 0.5;

/// Dynamic range over which the listener's pulse and color span the 0..1 t
/// parameter. -60 dB to 0 dB relative to scene `max_energy` maps to t=0..1.
pub const LISTENER_DB_RANGE: f32 = 60.0;

/// Sample the broadband energy captured by a listener at `position`. Energy
/// from grid points within `capture_radius` is summed (geometric proxy until
/// goal 005 lands per-ray streaming). Returns raw energy (not normalized).
pub fn capture_listener_energy(position: Vec3, capture_radius: f32, grid: &[GridPoint]) -> f32 {
    let r2 = capture_radius * capture_radius;
    let mut sum = 0.0_f32;
    for gp in grid {
        let d2 = (gp.position - position).length_squared();
        if d2 <= r2 {
            sum += gp.energy.iter().copied().fold(0.0_f32, f32::max);
        }
    }
    sum
}

/// Convert captured energy to a normalized 0..1 SPL value using log-dB scaling
/// against the scene's max grid energy. Output drives pulse radius + hot/cold
/// color shift.
pub fn normalized_spl(captured_energy: f32, max_grid_energy: f32) -> f32 {
    if captured_energy <= 0.0 || max_grid_energy <= 0.0 {
        return 0.0;
    }
    let ratio = (captured_energy / max_grid_energy).max(1e-12);
    let db = 10.0 * ratio.log10();
    let t = 1.0 + db / LISTENER_DB_RANGE;
    t.clamp(0.0, 1.0)
}

/// Cold-to-hot color map for listener visualization. t=0 → cool blue,
/// t=1 → hot red. Distinct from the surface heatmap (viridis) so listeners
/// pop visually against the room.
pub fn spl_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    // Two-stop cyan → magenta → yellow → red ramp for high contrast.
    let r = if t < 0.5 {
        // 64..240 over 0..0.5
        64.0 + t * 2.0 * (240.0 - 64.0)
    } else {
        // 240..255 over 0.5..1.0
        240.0 + (t - 0.5) * 2.0 * (255.0 - 240.0)
    };
    let g = if t < 0.5 {
        180.0 + t * 2.0 * (60.0 - 180.0)
    } else {
        60.0 + (t - 0.5) * 2.0 * (40.0 - 60.0)
    };
    let b = if t < 0.5 {
        240.0 + t * 2.0 * (90.0 - 240.0)
    } else {
        90.0 + (t - 0.5) * 2.0 * (40.0 - 90.0)
    };
    Color32::from_rgb(r as u8, g as u8, b as u8)
}

/// Pulse radius for a listener sphere, in world units. Base radius scales 1x..2x
/// with normalized SPL — fully silent listeners stay at base size, loud ones
/// expand to roughly 2x for visual emphasis.
pub fn pulse_radius(base_radius: f32, normalized_spl: f32) -> f32 {
    let t = normalized_spl.clamp(0.0, 1.0);
    base_radius * (1.0 + t)
}

/// Render a listener pulse + transparent capture-radius shell.
///
/// - Inner filled disk at `position`, radius `pulse_radius(base, spl)`, color
///   shifted by `spl_color`.
/// - Outer ring at `capture_radius`, semi-transparent, indicates the energy
///   integration sphere.
#[allow(clippy::too_many_arguments)]
pub fn render_listener_pulse(
    position: Vec3,
    capture_radius: f32,
    normalized_spl: f32,
    base_pulse_radius: f32,
    painter: &Painter,
    camera: &Camera,
    screen_center: Pos2,
    scale: f32,
    clip_rect: Rect,
) {
    let p_screen = project_3d(position, camera, screen_center, scale);
    if !clip_rect.contains(p_screen) {
        return;
    }

    // Outer shell — capture radius
    let edge_world = position + Vec3::new(capture_radius, 0.0, 0.0);
    let edge_screen = project_3d(edge_world, camera, screen_center, scale);
    let shell_radius_px = (edge_screen - p_screen).length().max(2.0);
    let shell_color = Color32::from_rgba_unmultiplied(180, 200, 220, 40);
    painter.circle_stroke(p_screen, shell_radius_px, Stroke::new(1.5, shell_color));

    // Inner pulse — colored by SPL, sized by pulse_radius
    let pulse_world =
        position + Vec3::new(pulse_radius(base_pulse_radius, normalized_spl), 0.0, 0.0);
    let pulse_screen = project_3d(pulse_world, camera, screen_center, scale);
    let pulse_px = (pulse_screen - p_screen).length().max(3.0);
    let core_color = spl_color(normalized_spl);
    let core_alpha =
        Color32::from_rgba_unmultiplied(core_color.r(), core_color.g(), core_color.b(), 200);
    painter.circle_filled(p_screen, pulse_px, core_alpha);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_listener_capture_radius_default_is_half_meter() {
        assert!((DEFAULT_LISTENER_CAPTURE_RADIUS - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_listener_capture_empty_grid_returns_zero() {
        let e = capture_listener_energy(Vec3::ZERO, 0.5, &[]);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn test_listener_capture_sums_within_radius() {
        let grid = vec![
            GridPoint {
                position: Vec3::new(0.1, 0.0, 0.0),
                energy: [2.0; 6],
            },
            GridPoint {
                position: Vec3::new(0.3, 0.0, 0.0),
                energy: [3.0; 6],
            },
            GridPoint {
                // Outside 0.5m radius
                position: Vec3::new(2.0, 0.0, 0.0),
                energy: [100.0; 6],
            },
        ];
        let e = capture_listener_energy(Vec3::ZERO, 0.5, &grid);
        assert!((e - 5.0).abs() < 1e-5, "expected 5.0, got {e}");
    }

    #[test]
    fn test_listener_normalized_spl_zero_energy_is_zero() {
        assert_eq!(normalized_spl(0.0, 1.0), 0.0);
    }

    #[test]
    fn test_listener_normalized_spl_max_is_one() {
        let t = normalized_spl(1.0, 1.0);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_listener_normalized_spl_60db_below_is_zero() {
        let t = normalized_spl(1e-6, 1.0);
        assert!(t.abs() < 1e-5, "expected ~0 at -60dB, got {t}");
    }

    #[test]
    fn test_listener_pulse_radius_silent_is_base() {
        let r = pulse_radius(1.0, 0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_listener_pulse_radius_loud_is_2x_base() {
        let r = pulse_radius(1.0, 1.0);
        assert!((r - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_listener_pulse_radius_monotonic() {
        let mut prev = 0.0;
        for i in 0..10 {
            let r = pulse_radius(1.0, i as f32 / 9.0);
            assert!(
                r >= prev,
                "pulse must not shrink with SPL: prev={prev} r={r}"
            );
            prev = r;
        }
    }

    #[test]
    fn test_listener_spl_color_cold_at_t0() {
        let c = spl_color(0.0);
        // Cool quadrant: blue dominant
        assert!(c.b() > 200);
        assert!(c.r() < 100);
    }

    #[test]
    fn test_listener_spl_color_hot_at_t1() {
        let c = spl_color(1.0);
        // Hot quadrant: red dominant
        assert!(c.r() > 240);
        assert!(c.b() < 100);
    }

    #[test]
    fn test_listener_spl_color_distinct_endpoints() {
        let cold = spl_color(0.0);
        let hot = spl_color(1.0);
        assert_ne!(cold, hot);
        // Cold and hot should be in different hue regions
        assert!(hot.r() > cold.r());
        assert!(hot.b() < cold.b());
    }

    #[test]
    fn test_listener_spl_color_clamps() {
        assert_eq!(spl_color(-0.5), spl_color(0.0));
        assert_eq!(spl_color(2.0), spl_color(1.0));
    }
}
