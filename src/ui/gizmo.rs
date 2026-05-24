//! Modal transform gizmo state — Blender-style G / R / S workflow.
//!
//! User flow:
//! 1. Make a selection.
//! 2. Press `G` (translate), `R` (rotate), or `S` (scale).
//! 3. Optionally constrain to an axis with `X`, `Y`, or `Z`.
//! 4. Either:
//!    - Drag the mouse to update the transform freely; or
//!    - Type a numeric value (e.g. `2.5`) to set the magnitude exactly.
//! 5. Confirm with `Enter` / `LMB` or cancel with `Esc` / `RMB`.
//!
//! This module owns the *state machine* — geometry mutation is up to the
//! caller (viewport_3d), which on confirmation reads `delta()` and routes
//! the change through `vp.history.push(MoveSource/MoveListener/...)`.

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

impl TransformMode {
    pub fn label(&self) -> &'static str {
        match self {
            TransformMode::Translate => "Translate",
            TransformMode::Rotate => "Rotate",
            TransformMode::Scale => "Scale",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AxisLock {
    #[default]
    None,
    X,
    Y,
    Z,
}

impl AxisLock {
    pub fn label(&self) -> &'static str {
        match self {
            AxisLock::None => "all",
            AxisLock::X => "X",
            AxisLock::Y => "Y",
            AxisLock::Z => "Z",
        }
    }

    /// Project `delta` onto the locked axis. `None` leaves it unchanged.
    pub fn constrain(&self, delta: Vec3) -> Vec3 {
        match self {
            AxisLock::None => delta,
            AxisLock::X => Vec3::new(delta.x, 0.0, 0.0),
            AxisLock::Y => Vec3::new(0.0, delta.y, 0.0),
            AxisLock::Z => Vec3::new(0.0, 0.0, delta.z),
        }
    }
}

/// Active gizmo state. `mode == None` ↔ no gizmo running.
#[derive(Default)]
pub struct GizmoState {
    pub mode: Option<TransformMode>,
    pub axis: AxisLock,
    /// Position of the selected item at the moment the gizmo was activated.
    /// On cancel, the caller restores this. On confirm, this + `delta()` is
    /// the new position.
    pub start_position: Vec3,
    /// Free-drag accumulated delta (in world units). The mouse handler in
    /// the viewport feeds [`Self::accumulate_drag`] each frame.
    pub drag_delta: Vec3,
    /// Optional user-typed numeric magnitude. While non-empty, takes
    /// precedence over `drag_delta`: the active axis is set to this value
    /// and other axes are zero. Without an axis lock, falls back to drag.
    pub numeric_input: String,
}

impl GizmoState {
    pub fn is_active(&self) -> bool {
        self.mode.is_some()
    }

    /// Enter a transform mode. Resets axis, delta, and numeric input.
    pub fn begin(&mut self, mode: TransformMode, start_position: Vec3) {
        self.mode = Some(mode);
        self.axis = AxisLock::None;
        self.start_position = start_position;
        self.drag_delta = Vec3::ZERO;
        self.numeric_input.clear();
    }

    /// Toggle axis lock. Pressing the same axis twice clears it.
    pub fn set_axis(&mut self, axis: AxisLock) {
        if self.axis == axis {
            self.axis = AxisLock::None;
        } else {
            self.axis = axis;
        }
    }

    /// Accumulate a free-drag delta this frame (e.g. converted from a
    /// `egui::Response::drag_delta()`).
    pub fn accumulate_drag(&mut self, world_delta: Vec3) {
        self.drag_delta += world_delta;
    }

    /// Append a typed character to the numeric input buffer. Accepts
    /// digits + `. - + * / ^ ( )` and letters (for `pi`, `e`, `sin`, etc.) —
    /// the buffer is fed to [`crate::ui::expr::evaluate_expression`] on read,
    /// so any expression that grammar accepts is valid input here.
    pub fn type_char(&mut self, c: char) {
        if c.is_ascii_digit()
            || matches!(c, '.' | '-' | '+' | '*' | '/' | '^' | '(' | ')' | ' ')
            || c.is_ascii_alphabetic()
        {
            self.numeric_input.push(c);
        }
    }

    /// Delete one character from the numeric input (Backspace).
    pub fn backspace(&mut self) {
        self.numeric_input.pop();
    }

    /// Parse the numeric input as an arithmetic expression (so users can
    /// type `2*3.14` or `1+sin(0)` directly into the gizmo). Returns `None`
    /// if empty or malformed.
    pub fn parsed_numeric(&self) -> Option<f32> {
        if self.numeric_input.is_empty() {
            return None;
        }
        crate::ui::expr::evaluate_expression(&self.numeric_input)
            .ok()
            .map(|v| v as f32)
    }

    /// Effective delta this frame, accounting for axis lock and numeric
    /// input override. Returns `Vec3::ZERO` if the gizmo isn't active.
    pub fn delta(&self) -> Vec3 {
        if !self.is_active() {
            return Vec3::ZERO;
        }
        if let Some(v) = self.parsed_numeric() {
            // Numeric input applies to the locked axis. If no axis locked,
            // fall back to the X axis (Blender's same convention).
            let axis = if self.axis == AxisLock::None {
                AxisLock::X
            } else {
                self.axis
            };
            return axis.constrain(Vec3::splat(v));
        }
        self.axis.constrain(self.drag_delta)
    }

    /// Confirm the in-progress transform. Returns the final delta and
    /// resets the gizmo. Caller uses the delta to construct a history
    /// command.
    pub fn confirm(&mut self) -> Vec3 {
        let d = self.delta();
        self.reset();
        d
    }

    /// Cancel without applying. Resets state.
    pub fn cancel(&mut self) {
        self.reset();
    }

    fn reset(&mut self) {
        self.mode = None;
        self.axis = AxisLock::None;
        self.start_position = Vec3::ZERO;
        self.drag_delta = Vec3::ZERO;
        self.numeric_input.clear();
    }

    /// Human-readable HUD line shown in the viewport while active:
    /// e.g. `"Translate X: 2.50"` or `"Translate: drag"`.
    pub fn hud_text(&self) -> Option<String> {
        let mode = self.mode?;
        let axis = self.axis.label();
        if let Some(n) = self.parsed_numeric() {
            Some(format!("{}: {} {:.3}", mode.label(), axis, n))
        } else if self.drag_delta.length() > 0.0 {
            let d = self.delta();
            Some(format!(
                "{}: {} ({:+.2}, {:+.2}, {:+.2})",
                mode.label(),
                axis,
                d.x,
                d.y,
                d.z
            ))
        } else {
            Some(format!("{}: {} (drag or type value)", mode.label(), axis))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gizmo_starts_inactive() {
        let g = GizmoState::default();
        assert!(!g.is_active());
        assert_eq!(g.delta(), Vec3::ZERO);
        assert!(g.hud_text().is_none());
    }

    #[test]
    fn gizmo_begin_records_start_position() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::new(1.0, 2.0, 3.0));
        assert!(g.is_active());
        assert_eq!(g.mode, Some(TransformMode::Translate));
        assert_eq!(g.start_position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(g.axis, AxisLock::None);
    }

    #[test]
    fn gizmo_axis_lock_constrains_delta() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.accumulate_drag(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(g.delta(), Vec3::new(1.0, 2.0, 3.0));

        g.set_axis(AxisLock::X);
        assert_eq!(g.delta(), Vec3::new(1.0, 0.0, 0.0));

        g.set_axis(AxisLock::Y);
        assert_eq!(g.delta(), Vec3::new(0.0, 2.0, 0.0));

        g.set_axis(AxisLock::Z);
        assert_eq!(g.delta(), Vec3::new(0.0, 0.0, 3.0));
    }

    #[test]
    fn gizmo_axis_lock_toggle_clears() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.set_axis(AxisLock::X);
        assert_eq!(g.axis, AxisLock::X);
        g.set_axis(AxisLock::X);
        assert_eq!(g.axis, AxisLock::None);
    }

    #[test]
    fn gizmo_drag_accumulates() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.accumulate_drag(Vec3::new(0.5, 0.0, 0.0));
        g.accumulate_drag(Vec3::new(0.3, 0.2, 0.0));
        assert_eq!(g.drag_delta, Vec3::new(0.8, 0.2, 0.0));
    }

    #[test]
    fn gizmo_numeric_input_overrides_drag() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.set_axis(AxisLock::Y);
        g.accumulate_drag(Vec3::new(10.0, 10.0, 10.0));

        g.type_char('2');
        g.type_char('.');
        g.type_char('5');
        assert_eq!(g.parsed_numeric(), Some(2.5));
        assert_eq!(g.delta(), Vec3::new(0.0, 2.5, 0.0));
    }

    #[test]
    fn gizmo_numeric_input_filters_non_expr_garbage() {
        // Type_char accepts digits + letters + arithmetic operators (so
        // users can type `sin(0)` or `2*3`), but rejects punctuation that
        // can't appear in expressions.
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        for c in "1!@#$.2,;3".chars() {
            g.type_char(c);
        }
        assert_eq!(g.numeric_input, "1.23");
        assert_eq!(g.parsed_numeric(), Some(1.23));
    }

    #[test]
    #[allow(clippy::approx_constant)] // 6.28 is the deliberate parse output we're asserting on
    fn gizmo_numeric_accepts_arithmetic_expressions() {
        // `2 * 3.14` should evaluate to 6.28 — the gizmo no longer filters
        // operators, since the expr evaluator handles them.
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        for c in "2 * 3.14".chars() {
            g.type_char(c);
        }
        assert_eq!(g.numeric_input, "2 * 3.14");
        let v = g.parsed_numeric().unwrap();
        assert!((v - 6.28).abs() < 1e-4);
    }

    #[test]
    fn gizmo_numeric_accepts_function_calls() {
        // Common math functions land in the gizmo unchanged, so the
        // existing expr evaluator can resolve them.
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        for c in "sqrt(9) + 1".chars() {
            g.type_char(c);
        }
        let v = g.parsed_numeric().unwrap();
        assert!((v - 4.0).abs() < 1e-5);
    }

    #[test]
    fn gizmo_numeric_supports_leading_minus_and_single_dot() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.type_char('-');
        g.type_char('1');
        g.type_char('.');
        g.type_char('5');
        assert_eq!(g.numeric_input, "-1.5");
        assert_eq!(g.parsed_numeric(), Some(-1.5));
    }

    #[test]
    fn gizmo_backspace_removes_last_char() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.type_char('1');
        g.type_char('2');
        g.backspace();
        assert_eq!(g.numeric_input, "1");
        g.backspace();
        assert_eq!(g.numeric_input, "");
        g.backspace(); // no panic on empty
        assert_eq!(g.numeric_input, "");
    }

    #[test]
    fn gizmo_confirm_returns_delta_and_resets() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::new(5.0, 0.0, 0.0));
        g.set_axis(AxisLock::X);
        g.accumulate_drag(Vec3::new(2.0, 99.0, 99.0));
        let d = g.confirm();
        assert_eq!(d, Vec3::new(2.0, 0.0, 0.0));
        assert!(!g.is_active());
        assert_eq!(g.drag_delta, Vec3::ZERO);
        assert_eq!(g.numeric_input, "");
    }

    #[test]
    fn gizmo_cancel_resets_without_returning_delta() {
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.accumulate_drag(Vec3::new(5.0, 0.0, 0.0));
        g.cancel();
        assert!(!g.is_active());
        assert_eq!(g.drag_delta, Vec3::ZERO);
    }

    #[test]
    fn gizmo_numeric_without_axis_defaults_to_x() {
        // Blender convention: typing "2" with no axis lock acts as X delta.
        let mut g = GizmoState::default();
        g.begin(TransformMode::Translate, Vec3::ZERO);
        g.type_char('2');
        assert_eq!(g.delta(), Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn gizmo_hud_text_changes_with_state() {
        let mut g = GizmoState::default();
        assert!(g.hud_text().is_none());

        g.begin(TransformMode::Rotate, Vec3::ZERO);
        let hud = g.hud_text().unwrap();
        assert!(hud.contains("Rotate"));

        g.set_axis(AxisLock::Z);
        g.type_char('9');
        g.type_char('0');
        let hud = g.hud_text().unwrap();
        assert!(hud.contains("Rotate"));
        assert!(hud.contains("Z"));
        assert!(hud.contains("90"));
    }

    #[test]
    fn gizmo_modes_round_trip_labels() {
        assert_eq!(TransformMode::Translate.label(), "Translate");
        assert_eq!(TransformMode::Rotate.label(), "Rotate");
        assert_eq!(TransformMode::Scale.label(), "Scale");
        assert_eq!(AxisLock::None.label(), "all");
        assert_eq!(AxisLock::X.label(), "X");
        assert_eq!(AxisLock::Y.label(), "Y");
        assert_eq!(AxisLock::Z.label(), "Z");
    }
}
