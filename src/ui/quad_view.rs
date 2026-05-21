//! Quad-view: split the viewport into Top / Front / Side / Perspective
//! quadrants sharing a focus point.
//!
//! Toggled by Ctrl+Alt+Q. The four quadrants update in lockstep: when the
//! user pans / orbits in any quadrant, the focus point migrates and the
//! other three re-anchor on it. Distance is preserved per quadrant so the
//! perspective camera can zoom while the orthographic projections stay
//! orthonormal.
//!
//! The single-camera mode is the default. Quad-view is additive — when
//! [`QuadView::enabled`] is false the viewport renders exactly as before.

use glam::Vec3;

use crate::renderer::CameraView;

/// One of four quadrants laid out as:
///
/// ```text
/// ┌───────┬───────┐
/// │ Top   │ Persp │
/// ├───────┼───────┤
/// │ Front │ Side  │
/// └───────┴───────┘
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Quadrant {
    Top,
    Front,
    Side,
    Perspective,
}

impl Quadrant {
    pub const ALL: [Quadrant; 4] = [
        Quadrant::Top,
        Quadrant::Front,
        Quadrant::Side,
        Quadrant::Perspective,
    ];

    pub fn view(self) -> CameraView {
        match self {
            Quadrant::Top => CameraView::Top,
            Quadrant::Front => CameraView::Front,
            Quadrant::Side => CameraView::Side,
            Quadrant::Perspective => CameraView::Perspective,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Quadrant::Top => "Top",
            Quadrant::Front => "Front",
            Quadrant::Side => "Side",
            Quadrant::Perspective => "Perspective",
        }
    }
}

/// One element of [`QuadView::quadrant_rects`] — quadrant, origin (x, y),
/// size (w, h).
pub type QuadrantRect = (Quadrant, (f32, f32), (f32, f32));

/// Quad-view state. Off by default; toggled by Ctrl+Alt+Q.
#[derive(Clone, Debug)]
pub struct QuadView {
    pub enabled: bool,
    /// Shared focus point. When a quadrant pans/orbits, this migrates
    /// and the other three re-anchor on it.
    pub focus: Vec3,
    /// Which quadrant is "active" — receives input first when the cursor
    /// is over it. Defaults to perspective.
    pub active: Quadrant,
}

impl Default for QuadView {
    fn default() -> Self {
        Self {
            enabled: false,
            focus: Vec3::ZERO,
            active: Quadrant::Perspective,
        }
    }
}

impl QuadView {
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }

    /// Compute the (origin, size) of each quadrant given the viewport rect.
    /// Returns `[QuadrantRect; 4]` in `Quadrant::ALL` order.
    /// Pure function — easy to unit test layout math.
    pub fn quadrant_rects(rect_size: (f32, f32)) -> [QuadrantRect; 4] {
        let (w, h) = rect_size;
        let hw = w * 0.5;
        let hh = h * 0.5;
        [
            (Quadrant::Top, (0.0, 0.0), (hw, hh)),
            (Quadrant::Perspective, (hw, 0.0), (hw, hh)),
            (Quadrant::Front, (0.0, hh), (hw, hh)),
            (Quadrant::Side, (hw, hh), (hw, hh)),
        ]
    }

    /// Identify which quadrant the cursor is over, given the viewport
    /// origin + size and the cursor position. Returns `None` if the cursor
    /// is outside the viewport.
    pub fn quadrant_at(
        rect_origin: (f32, f32),
        rect_size: (f32, f32),
        cursor: (f32, f32),
    ) -> Option<Quadrant> {
        let (ox, oy) = rect_origin;
        let (w, h) = rect_size;
        let (cx, cy) = cursor;
        if cx < ox || cy < oy || cx > ox + w || cy > oy + h {
            return None;
        }
        let half_x = ox + w * 0.5;
        let half_y = oy + h * 0.5;
        Some(match (cx < half_x, cy < half_y) {
            (true, true) => Quadrant::Top,
            (false, true) => Quadrant::Perspective,
            (true, false) => Quadrant::Front,
            (false, false) => Quadrant::Side,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_view_disabled_by_default() {
        let q = QuadView::default();
        assert!(!q.enabled);
        assert_eq!(q.active, Quadrant::Perspective);
    }

    #[test]
    fn quad_view_toggle_flips() {
        let mut q = QuadView::default();
        q.toggle();
        assert!(q.enabled);
        q.toggle();
        assert!(!q.enabled);
    }

    #[test]
    fn quad_view_quadrant_rects_partition() {
        let rects = QuadView::quadrant_rects((800.0, 600.0));
        assert_eq!(rects.len(), 4);
        for (_, _, (w, h)) in rects {
            assert_eq!(w, 400.0);
            assert_eq!(h, 300.0);
        }
        // Origins span the full rect.
        let origins: Vec<_> = rects.iter().map(|(_, o, _)| *o).collect();
        assert!(origins.contains(&(0.0, 0.0)));
        assert!(origins.contains(&(400.0, 0.0)));
        assert!(origins.contains(&(0.0, 300.0)));
        assert!(origins.contains(&(400.0, 300.0)));
    }

    #[test]
    fn quad_view_quadrant_at_routes_clicks() {
        let origin = (0.0, 0.0);
        let size = (800.0, 600.0);
        assert_eq!(
            QuadView::quadrant_at(origin, size, (100.0, 100.0)),
            Some(Quadrant::Top)
        );
        assert_eq!(
            QuadView::quadrant_at(origin, size, (500.0, 100.0)),
            Some(Quadrant::Perspective)
        );
        assert_eq!(
            QuadView::quadrant_at(origin, size, (100.0, 400.0)),
            Some(Quadrant::Front)
        );
        assert_eq!(
            QuadView::quadrant_at(origin, size, (500.0, 400.0)),
            Some(Quadrant::Side)
        );
    }

    #[test]
    fn quad_view_quadrant_at_returns_none_outside() {
        assert_eq!(
            QuadView::quadrant_at((10.0, 10.0), (100.0, 100.0), (5.0, 5.0)),
            None
        );
        assert_eq!(
            QuadView::quadrant_at((10.0, 10.0), (100.0, 100.0), (200.0, 50.0)),
            None
        );
    }

    #[test]
    fn quadrant_view_mapping() {
        assert_eq!(Quadrant::Top.view(), CameraView::Top);
        assert_eq!(Quadrant::Front.view(), CameraView::Front);
        assert_eq!(Quadrant::Side.view(), CameraView::Side);
        assert_eq!(Quadrant::Perspective.view(), CameraView::Perspective);
    }

    #[test]
    fn quadrant_labels_present() {
        for q in Quadrant::ALL {
            assert!(!q.label().is_empty());
        }
    }

    #[test]
    fn quad_view_focus_defaults_to_origin() {
        let q = QuadView::default();
        assert_eq!(q.focus, Vec3::ZERO);
    }
}
