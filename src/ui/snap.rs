//! Snap helpers — grid, surface, and angle.
//!
//! The convention in EchoMap matches Blender + most DCC tools: hold **Shift**
//! while transforming to enable snap. Without Shift, transforms are free.
//! For users who want snap-on-by-default behaviour, [`SnapConfig::always_on`]
//! is exposed — Settings can flip it.
//!
//! This module is intentionally pure: it computes snap targets given a
//! candidate position / angle and config. Hooking it up to gizmo state and
//! modifier-key polling lives in `ui/mod.rs`.

use glam::Vec3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SnapMode {
    None,
    /// Round each axis to the nearest multiple of `grid_size`.
    #[default]
    Grid,
    /// Project onto the ground plane (y=0). Future: raycast onto meshes.
    Surface,
    /// Round angle (degrees) to the nearest multiple of `angle_step_deg`.
    Angle,
}

impl SnapMode {
    pub fn label(&self) -> &'static str {
        match self {
            SnapMode::None => "off",
            SnapMode::Grid => "grid",
            SnapMode::Surface => "surface",
            SnapMode::Angle => "angle",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SnapConfig {
    pub mode: SnapMode,
    pub grid_size: f32,
    pub angle_step_deg: f32,
    /// If true, snap applies without holding Shift (toggleable from
    /// Settings). Default false matches Blender/Maya conventions.
    pub always_on: bool,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            mode: SnapMode::Grid,
            grid_size: 0.25,
            angle_step_deg: 15.0,
            always_on: false,
        }
    }
}

impl SnapConfig {
    /// Whether the snap should apply right now given the live `shift_held`
    /// modifier state. With `always_on = false` (default), Shift enables
    /// snap. With `always_on = true`, Shift inverts and disables it. XOR
    /// captures both.
    pub fn active(&self, shift_held: bool) -> bool {
        self.mode != SnapMode::None && (self.always_on ^ shift_held)
    }
}

/// Snap a position vector to the nearest multiple of `step` on each axis.
///
/// `step <= 0` returns the input unchanged so callers don't divide by zero.
pub fn snap_grid(pos: Vec3, step: f32) -> Vec3 {
    if step <= 0.0 || !step.is_finite() {
        return pos;
    }
    Vec3::new(
        (pos.x / step).round() * step,
        (pos.y / step).round() * step,
        (pos.z / step).round() * step,
    )
}

/// Snap only the components that are non-zero in `mask` (1.0 = snap this
/// axis, 0.0 = leave unchanged). Used to combine with the gizmo's axis lock
/// — only snap the axes the user is actually moving.
pub fn snap_grid_masked(pos: Vec3, step: f32, mask: Vec3) -> Vec3 {
    if step <= 0.0 || !step.is_finite() {
        return pos;
    }
    let snapped = snap_grid(pos, step);
    Vec3::new(
        if mask.x.abs() > 1e-6 {
            snapped.x
        } else {
            pos.x
        },
        if mask.y.abs() > 1e-6 {
            snapped.y
        } else {
            pos.y
        },
        if mask.z.abs() > 1e-6 {
            snapped.z
        } else {
            pos.z
        },
    )
}

/// Project a position onto the ground plane (y = 0). For future deliverables
/// this can be extended to raycast onto mesh surfaces and snap to the hit
/// point.
pub fn snap_surface_ground(pos: Vec3) -> Vec3 {
    Vec3::new(pos.x, 0.0, pos.z)
}

/// Round an angle (degrees) to the nearest multiple of `step_deg`.
pub fn snap_angle_deg(angle_deg: f32, step_deg: f32) -> f32 {
    if step_deg <= 0.0 || !step_deg.is_finite() {
        return angle_deg;
    }
    (angle_deg / step_deg).round() * step_deg
}

/// Resolve which snap function to use given the config + return the snapped
/// position. `angle_deg` is ignored unless mode == Angle (then the function
/// returns Vec3::new(snapped_angle, 0, 0) for the caller to extract).
pub fn apply_snap(pos: Vec3, config: &SnapConfig) -> Vec3 {
    match config.mode {
        SnapMode::None => pos,
        SnapMode::Grid => snap_grid(pos, config.grid_size),
        SnapMode::Surface => snap_surface_ground(pos),
        SnapMode::Angle => pos, // angle mode operates on rotations, not positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_grid_rounds_each_axis_to_nearest_step() {
        let p = Vec3::new(0.12, 0.78, -0.34);
        let s = snap_grid(p, 0.25);
        // 0.12 / 0.25 = 0.48 → round 0 → 0.0
        // 0.78 / 0.25 = 3.12 → round 3 → 0.75
        // -0.34 / 0.25 = -1.36 → round -1 → -0.25
        assert!((s.x - 0.0).abs() < 1e-6, "x got {}", s.x);
        assert!((s.y - 0.75).abs() < 1e-6, "y got {}", s.y);
        assert!((s.z - -0.25).abs() < 1e-6, "z got {}", s.z);
    }

    #[test]
    fn snap_grid_handles_zero_step_safely() {
        let p = Vec3::new(1.5, 2.5, 3.5);
        assert_eq!(snap_grid(p, 0.0), p);
        assert_eq!(snap_grid(p, -1.0), p);
        assert_eq!(snap_grid(p, f32::NAN), p);
    }

    #[test]
    fn snap_grid_origin_is_invariant() {
        let p = Vec3::ZERO;
        for step in [0.1, 0.25, 1.0, 5.0] {
            assert_eq!(snap_grid(p, step), Vec3::ZERO);
        }
    }

    #[test]
    fn snap_grid_negative_values_snap_correctly() {
        let p = Vec3::new(-1.13, -0.6, -0.4);
        let s = snap_grid(p, 0.5);
        // -1.13 / 0.5 = -2.26, round = -2, * 0.5 = -1.0
        // -0.6 / 0.5 = -1.2, round = -1, * 0.5 = -0.5
        // -0.4 / 0.5 = -0.8, round = -1, * 0.5 = -0.5
        assert!((s.x - -1.0).abs() < 1e-6);
        assert!((s.y - -0.5).abs() < 1e-6);
        assert!((s.z - -0.5).abs() < 1e-6);
    }

    #[test]
    fn snap_grid_masked_only_snaps_selected_axes() {
        let p = Vec3::new(0.12, 0.78, -0.34);
        // Snap X only — X rounds to 0.0; Y and Z untouched.
        let s = snap_grid_masked(p, 0.25, Vec3::new(1.0, 0.0, 0.0));
        assert!((s.x - 0.0).abs() < 1e-6, "x got {}", s.x);
        assert!((s.y - 0.78).abs() < 1e-6);
        assert!((s.z - -0.34).abs() < 1e-6);
    }

    #[test]
    fn snap_surface_ground_zeros_y() {
        let s = snap_surface_ground(Vec3::new(1.5, 2.7, -3.2));
        assert!((s.x - 1.5).abs() < 1e-6);
        assert!(s.y.abs() < 1e-6);
        assert!((s.z - -3.2).abs() < 1e-6);
    }

    #[test]
    fn snap_angle_deg_rounds_to_step() {
        assert!((snap_angle_deg(17.0, 15.0) - 15.0).abs() < 1e-4);
        assert!((snap_angle_deg(23.0, 15.0) - 30.0).abs() < 1e-4);
        assert!((snap_angle_deg(-7.0, 15.0) - 0.0).abs() < 1e-4);
        assert!((snap_angle_deg(-23.0, 15.0) - -30.0).abs() < 1e-4);
        assert!((snap_angle_deg(180.0, 90.0) - 180.0).abs() < 1e-4);
    }

    #[test]
    fn snap_angle_zero_step_passes_through() {
        assert_eq!(snap_angle_deg(17.0, 0.0), 17.0);
        assert_eq!(snap_angle_deg(17.0, f32::NAN), 17.0);
    }

    #[test]
    fn config_default_is_grid_quarter_meter() {
        let c = SnapConfig::default();
        assert_eq!(c.mode, SnapMode::Grid);
        assert!((c.grid_size - 0.25).abs() < 1e-6);
        assert!((c.angle_step_deg - 15.0).abs() < 1e-4);
        assert!(!c.always_on);
    }

    #[test]
    fn config_active_respects_shift_unless_always_on() {
        let mut c = SnapConfig::default();
        // Default: shift toggles snap.
        assert!(c.active(true));
        assert!(!c.active(false));

        // always_on: shift inverts → off.
        c.always_on = true;
        assert!(c.active(false));
        // The XOR convention is "shift inverts the default" — see implementation.
        assert!(!c.active(true));
    }

    #[test]
    fn config_active_off_when_mode_none() {
        let c = SnapConfig {
            mode: SnapMode::None,
            ..Default::default()
        };
        assert!(!c.active(true));
        assert!(!c.active(false));
    }

    #[test]
    fn apply_snap_routes_by_mode() {
        let p = Vec3::new(0.12, 1.7, -0.34);
        let mut c = SnapConfig {
            mode: SnapMode::Grid,
            ..Default::default()
        };
        let s = apply_snap(p, &c);
        assert!((s.x - 0.0).abs() < 1e-6 || (s.x - 0.25).abs() < 1e-6);

        c.mode = SnapMode::Surface;
        let s = apply_snap(p, &c);
        assert!(s.y.abs() < 1e-6);

        c.mode = SnapMode::None;
        assert_eq!(apply_snap(p, &c), p);
    }

    #[test]
    fn snap_mode_labels_are_human_readable() {
        assert_eq!(SnapMode::None.label(), "off");
        assert_eq!(SnapMode::Grid.label(), "grid");
        assert_eq!(SnapMode::Surface.label(), "surface");
        assert_eq!(SnapMode::Angle.label(), "angle");
    }
}
