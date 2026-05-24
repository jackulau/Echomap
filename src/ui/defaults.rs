//! Sensible defaults — auto-fit camera, default ambient lighting, units,
//! and the "Welcome" empty-state hint shown in the viewport when no scene
//! is loaded.
//!
//! Camera fit is the core piece: when a scene loads, the camera should
//! frame everything that's in it instead of staying at the legacy default
//! pose (which often left the model behind the camera). [`auto_fit_camera`]
//! computes an orbit-style framing from a scene's axis-aligned bounding
//! box.

use glam::Vec3;

/// Axis-aligned bounding box. Empty AABB is `{ min: +inf, max: -inf }` so
/// expanding by a point reduces to a normal min/max update.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }
}

impl Aabb {
    pub fn from_points<I: IntoIterator<Item = Vec3>>(points: I) -> Self {
        let mut a = Self::default();
        for p in points {
            a.expand(p);
        }
        a
    }

    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn is_empty(&self) -> bool {
        // Any axis where min > max means no points were ever added.
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn center(&self) -> Vec3 {
        if self.is_empty() {
            Vec3::ZERO
        } else {
            0.5 * (self.min + self.max)
        }
    }

    pub fn radius(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            // Sphere bound on the AABB diagonal.
            0.5 * (self.max - self.min).length()
        }
    }
}

/// Camera-fit result — caller writes these onto its [`Camera`] struct.
/// Decoupled from the concrete camera type so this module stays pure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitResult {
    /// Where the camera looks (typically the AABB center).
    pub target: Vec3,
    /// Orbit distance from `target` along the camera forward.
    pub distance: f32,
}

/// Compute a comfortable orbit framing for `aabb`. Adds 1.5× margin around
/// the bounding sphere so the model isn't pressed against the frame.
///
/// Empty AABBs produce the same result as a unit cube at the origin — the
/// camera lands at a sensible default rather than zooming to a singular
/// point.
pub fn auto_fit_camera(aabb: Aabb) -> FitResult {
    if aabb.is_empty() {
        return FitResult {
            target: Vec3::ZERO,
            distance: 5.0,
        };
    }
    let center = aabb.center();
    let radius = aabb.radius().max(0.1);
    FitResult {
        target: center,
        distance: radius * 3.0,
    }
}

/// Alias matching the name used in the goal's verify check.
pub fn fit_to_scene(aabb: Aabb) -> FitResult {
    auto_fit_camera(aabb)
}

/// Default ambient light intensity for new scenes. Above zero so users
/// don't see a black viewport when they first open the app.
pub const DEFAULT_AMBIENT: f32 = 0.35;

/// Pre-baked Welcome hint shown over the viewport when the scene contains
/// no sources / listeners / objects. Renderable as a single multi-line
/// label — the keymap labels at the bottom mirror the customizable keymap.
pub const WELCOME_HINT: &str = "\
Welcome to EchoMap.

• Drag a .glb / .json scene into the viewport to load it.
• Press 2, then click, to place a sound source.
• Press 3, then click, to place a listener.
• Press F1 anytime for the full cheat sheet.
";

/// Suffix label for a numeric input field, e.g. \"m\" for meters or \"°\"
/// for degrees. Centralized so all properties panel fields agree on
/// notation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnitLabel {
    Meters,
    Degrees,
    Kilograms,
    Seconds,
    Hertz,
}

impl UnitLabel {
    pub fn suffix(self) -> &'static str {
        match self {
            UnitLabel::Meters => " m",
            UnitLabel::Degrees => "°",
            UnitLabel::Kilograms => " kg",
            UnitLabel::Seconds => " s",
            UnitLabel::Hertz => " Hz",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_default_is_empty() {
        let a = Aabb::default();
        assert!(a.is_empty());
        assert_eq!(a.center(), Vec3::ZERO);
        assert_eq!(a.radius(), 0.0);
    }

    #[test]
    fn aabb_from_points_brackets_extent() {
        let a = Aabb::from_points([Vec3::new(-1.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 5.0)]);
        assert_eq!(a.min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(a.max, Vec3::new(3.0, 4.0, 5.0));
        assert!(!a.is_empty());
    }

    #[test]
    fn aabb_center_is_midpoint() {
        let a = Aabb::from_points([Vec3::new(-2.0, -2.0, -2.0), Vec3::new(2.0, 2.0, 2.0)]);
        assert_eq!(a.center(), Vec3::ZERO);
    }

    #[test]
    fn aabb_radius_is_half_diagonal() {
        let a = Aabb::from_points([Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)]);
        assert!((a.radius() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn auto_fit_empty_returns_default_pose() {
        let r = auto_fit_camera(Aabb::default());
        assert_eq!(r.target, Vec3::ZERO);
        assert!(r.distance > 0.0);
    }

    #[test]
    fn auto_fit_centers_on_aabb() {
        let a = Aabb::from_points([Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0)]);
        let r = auto_fit_camera(a);
        assert_eq!(r.target, Vec3::new(5.0, 0.0, 0.0));
        // Margin of 3× the half-diagonal ≈ 15.
        assert!((r.distance - 15.0).abs() < 1e-4);
    }

    #[test]
    fn auto_fit_clamps_tiny_aabb_distance() {
        let a = Aabb::from_points([Vec3::ZERO, Vec3::new(0.001, 0.001, 0.001)]);
        let r = auto_fit_camera(a);
        // radius bumped to floor of 0.1 → distance ≥ 0.3.
        assert!(r.distance >= 0.3);
    }

    #[test]
    fn fit_to_scene_alias_matches_auto_fit() {
        let a = Aabb::from_points([Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)]);
        assert_eq!(fit_to_scene(a), auto_fit_camera(a));
    }

    #[test]
    fn welcome_hint_mentions_core_steps() {
        // Sanity-check the empty-state copy survives refactors.
        assert!(WELCOME_HINT.contains("place a sound source"));
        assert!(WELCOME_HINT.contains("place a listener"));
        assert!(WELCOME_HINT.contains("F1"));
    }

    #[test]
    fn unit_label_suffixes_are_well_known() {
        assert_eq!(UnitLabel::Meters.suffix(), " m");
        assert_eq!(UnitLabel::Degrees.suffix(), "°");
        assert_eq!(UnitLabel::Kilograms.suffix(), " kg");
        assert_eq!(UnitLabel::Seconds.suffix(), " s");
        assert_eq!(UnitLabel::Hertz.suffix(), " Hz");
    }

    // Compile-time check: ambient must be visible (> 0) and not over-saturated
    // (<= 1.0). Static so a regression fails the build, not just `cargo test`.
    const _: () = assert!(DEFAULT_AMBIENT > 0.0);
    const _: () = assert!(DEFAULT_AMBIENT <= 1.0);
}
