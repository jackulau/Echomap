//! Trackpad gesture math — pinch-zoom + two-finger pan applied to a
//! [`crate::renderer::Camera`].
//!
//! egui exposes `InputState::zoom_delta()` (≈1.0 = no change, >1.0 = pinch
//! out / zoom in, <1.0 = pinch in / zoom out) and `InputState::multi_touch`
//! for raw events. The viewport hooks consume these and call the helpers
//! below to mutate the camera. Keeping the math here as pure functions
//! lets us unit-test gesture handling without spinning up an egui context.

/// A multi-touch event reduced to viewport-relevant quantities. Built from
/// `egui::MultiTouchInfo` in the viewport path.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GestureFrame {
    /// `1.0` = no pinch. `>1.0` zooms in, `<1.0` zooms out.
    pub zoom_delta: f32,
    /// Two-finger translation delta in screen pixels (centered).
    pub translation: (f32, f32),
    /// Number of fingers currently down (1 = single tap/drag, 2 = pinch/pan).
    pub fingers: u8,
}

impl Default for GestureFrame {
    fn default() -> Self {
        Self {
            zoom_delta: 1.0,
            translation: (0.0, 0.0),
            fingers: 0,
        }
    }
}

impl GestureFrame {
    pub fn is_idle(&self) -> bool {
        self.fingers == 0 && (self.zoom_delta - 1.0).abs() < 1e-4 && self.translation == (0.0, 0.0)
    }

    pub fn is_pinch(&self) -> bool {
        self.fingers >= 2 && (self.zoom_delta - 1.0).abs() > 1e-4
    }

    pub fn is_two_finger_pan(&self) -> bool {
        self.fingers >= 2 && self.translation != (0.0, 0.0) && (self.zoom_delta - 1.0).abs() < 1e-3
    }
}

/// Apply a pinch zoom_delta to a camera "distance" (zoom radius). Clamped
/// so the camera can't flip through the focal point.
///
/// `zoom_delta` > 1.0 → pinch out → camera moves closer (smaller distance).
pub fn apply_pinch_zoom(current: f32, zoom_delta: f32) -> f32 {
    let zd = zoom_delta.max(1e-3);
    (current / zd).clamp(0.1, 1000.0)
}

/// Apply two-finger pan: project screen-space pixel delta into a world-space
/// nudge given a sensitivity scale. Returns a `(dx, dy)` world-space offset
/// that the caller should add to the camera target.
///
/// `pixels_per_unit` should be the projector's scale (pixels per world unit)
/// so 100px drag maps to one world-unit at the default scale.
pub fn pan_pixels_to_world(translation_px: (f32, f32), pixels_per_unit: f32) -> (f32, f32) {
    let s = pixels_per_unit.max(1.0);
    (-translation_px.0 / s, translation_px.1 / s)
}

/// Sensitivity scaling for trackpad zoom. egui's raw zoom_delta is centred
/// around 1.0; values like 1.05 / 0.95 per frame come from real macOS
/// pinches. We bias it slightly toward unity so single-finger drift doesn't
/// micro-zoom: `output = 1 + (zoom_delta - 1) * gain`.
pub fn shape_zoom_delta(zoom_delta: f32, gain: f32) -> f32 {
    1.0 + (zoom_delta - 1.0) * gain.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gestures_default_is_idle() {
        let g = GestureFrame::default();
        assert!(g.is_idle());
        assert!(!g.is_pinch());
        assert!(!g.is_two_finger_pan());
    }

    #[test]
    fn gestures_pinch_detection() {
        let g = GestureFrame {
            zoom_delta: 1.05,
            translation: (0.0, 0.0),
            fingers: 2,
        };
        assert!(g.is_pinch());
        assert!(!g.is_two_finger_pan());
        assert!(!g.is_idle());
    }

    #[test]
    fn gestures_pan_detection() {
        let g = GestureFrame {
            zoom_delta: 1.0,
            translation: (10.0, 0.0),
            fingers: 2,
        };
        assert!(g.is_two_finger_pan());
        assert!(!g.is_pinch());
    }

    #[test]
    fn gestures_single_finger_does_not_pinch_or_pan() {
        let g = GestureFrame {
            zoom_delta: 1.05,
            translation: (10.0, 5.0),
            fingers: 1,
        };
        assert!(!g.is_pinch());
        assert!(!g.is_two_finger_pan());
    }

    #[test]
    fn apply_pinch_zoom_in_makes_camera_closer() {
        let z = apply_pinch_zoom(10.0, 1.25);
        assert!(z < 10.0);
        assert!(z > 0.1);
    }

    #[test]
    fn apply_pinch_zoom_out_pushes_camera_back() {
        let z = apply_pinch_zoom(10.0, 0.8);
        assert!(z > 10.0);
    }

    #[test]
    fn apply_pinch_zoom_clamped() {
        let near = apply_pinch_zoom(0.05, 100.0); // should clamp to 0.1
        assert_eq!(near, 0.1);
        let far = apply_pinch_zoom(10000.0, 0.0001); // would explode; clamp
        assert!(far <= 1000.0);
    }

    #[test]
    fn apply_pinch_zoom_unit_delta_is_noop() {
        let z = apply_pinch_zoom(7.3, 1.0);
        assert!((z - 7.3).abs() < 1e-6);
    }

    #[test]
    fn pan_pixels_to_world_inverts_x_axis() {
        // Drag finger right (+x screen) → camera target moves left
        // (so the world appears to move with the finger).
        let (dx, dy) = pan_pixels_to_world((100.0, 0.0), 100.0);
        assert_eq!(dx, -1.0);
        assert_eq!(dy, 0.0);
    }

    #[test]
    fn pan_pixels_to_world_scaled_by_zoom() {
        let (dx, _) = pan_pixels_to_world((50.0, 0.0), 50.0);
        assert_eq!(dx, -1.0);
    }

    #[test]
    fn pan_pixels_to_world_guards_against_zero_scale() {
        // Should not panic / divide by zero
        let _ = pan_pixels_to_world((10.0, 10.0), 0.0);
    }

    #[test]
    fn shape_zoom_delta_unity_for_no_input() {
        assert_eq!(shape_zoom_delta(1.0, 1.0), 1.0);
        assert_eq!(shape_zoom_delta(1.0, 0.5), 1.0);
    }

    #[test]
    fn shape_zoom_delta_dampens_with_low_gain() {
        let shaped = shape_zoom_delta(1.2, 0.5);
        // (1.2 - 1.0) * 0.5 = 0.1 → 1.1
        assert!((shaped - 1.1).abs() < 1e-6);
    }
}
