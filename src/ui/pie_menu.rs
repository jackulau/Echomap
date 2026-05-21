//! Radial pie menu — hold Q to summon a ring of quick-action wedges.
//!
//! Blender-style pie menus: while the activator key (default Q) is held,
//! a circle of N action wedges appears centered on the cursor. Releasing
//! the key fires whichever wedge the cursor is over. Movement past a small
//! dead-zone "selects" the wedge; releasing inside the dead-zone cancels.
//!
//! State here is pure: position math + activation lifecycle. The actual
//! rendering (egui Window + polygon-painter) lives in the viewport.

use std::f32::consts::TAU;

/// One wedge of the pie. Action is identified by [`PieAction`] so the
/// viewport can dispatch it back through the existing palette / keymap
/// machinery without knowing about pie geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PieSlice {
    pub label: &'static str,
    pub action: PieAction,
}

/// The eight quick actions chosen for the goal's wedge set. Maps to the
/// existing palette/keymap commands when the wedge is fired.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieAction {
    FrameSelected,
    ResetCamera,
    ToggleGrid,
    ToggleShaded,
    SwitchView,
    PlaceSource,
    PlaceListener,
    RunSim,
}

/// Default 8-wedge layout. Order chosen so the most-used actions land on
/// the cardinal directions (up, right, down, left).
pub const DEFAULT_SLICES: &[PieSlice] = &[
    PieSlice {
        label: "Frame Selected",
        action: PieAction::FrameSelected,
    },
    PieSlice {
        label: "Place Source",
        action: PieAction::PlaceSource,
    },
    PieSlice {
        label: "Run Sim",
        action: PieAction::RunSim,
    },
    PieSlice {
        label: "Place Listener",
        action: PieAction::PlaceListener,
    },
    PieSlice {
        label: "Reset Camera",
        action: PieAction::ResetCamera,
    },
    PieSlice {
        label: "Toggle Grid",
        action: PieAction::ToggleGrid,
    },
    PieSlice {
        label: "Toggle Shaded",
        action: PieAction::ToggleShaded,
    },
    PieSlice {
        label: "Switch View",
        action: PieAction::SwitchView,
    },
];

/// Per-frame pie menu state.
#[derive(Clone, Debug, Default)]
pub struct PieMenuState {
    pub active: bool,
    /// Pointer position at activation — wedge geometry is computed relative
    /// to this center.
    pub center: (f32, f32),
    pub slices: Vec<PieSlice>,
    /// Inner dead-zone radius, in screen pixels. Cursor inside this radius
    /// is "no selection".
    pub deadzone_px: f32,
}

impl PieMenuState {
    /// Create with the default 8-wedge layout.
    pub fn with_defaults() -> Self {
        Self {
            active: false,
            center: (0.0, 0.0),
            slices: DEFAULT_SLICES.to_vec(),
            deadzone_px: 40.0,
        }
    }

    /// Open at the given cursor position. Idempotent — re-activating
    /// doesn't change `slices`.
    pub fn open(&mut self, x: f32, y: f32) {
        self.active = true;
        self.center = (x, y);
    }

    /// Close without firing an action.
    pub fn cancel(&mut self) {
        self.active = false;
    }

    /// Resolve the wedge the pointer is currently over. Returns `None`
    /// when the pointer is inside the dead-zone or the menu is closed.
    pub fn slice_at(&self, x: f32, y: f32) -> Option<PieSlice> {
        if !self.active || self.slices.is_empty() {
            return None;
        }
        let dx = x - self.center.0;
        let dy = y - self.center.1;
        let r = (dx * dx + dy * dy).sqrt();
        if r < self.deadzone_px {
            return None;
        }
        // Atan2 returns (-PI, PI]; shift so 0 is "up" and increases
        // clockwise to match the wedge order in DEFAULT_SLICES.
        let mut theta = (-dy).atan2(dx); // 0 = right, PI/2 = up
        theta = (TAU * 0.25) - theta; // rotate so 0 = up
        if theta < 0.0 {
            theta += TAU;
        }
        let n = self.slices.len() as f32;
        let idx = ((theta / TAU) * n).floor() as usize % self.slices.len();
        self.slices.get(idx).copied()
    }

    /// Convenience for callers: release-the-key path — pick the current
    /// slice (or `None` if dead-zone) and close the menu either way.
    pub fn release(&mut self, x: f32, y: f32) -> Option<PieSlice> {
        let pick = self.slice_at(x, y);
        self.active = false;
        pick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pie_defaults_have_eight_slices() {
        // Goal calls for 8 wedges.
        assert_eq!(DEFAULT_SLICES.len(), 8);
    }

    #[test]
    fn pie_default_state_is_inactive() {
        let p = PieMenuState::with_defaults();
        assert!(!p.active);
        assert_eq!(p.slices.len(), 8);
    }

    #[test]
    fn pie_open_sets_center_and_active() {
        let mut p = PieMenuState::with_defaults();
        p.open(120.0, 240.0);
        assert!(p.active);
        assert_eq!(p.center, (120.0, 240.0));
    }

    #[test]
    fn pie_cancel_clears_active() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        p.cancel();
        assert!(!p.active);
    }

    #[test]
    fn pie_deadzone_returns_none() {
        let mut p = PieMenuState::with_defaults();
        p.open(100.0, 100.0);
        // Pointer at the exact center is inside the dead-zone.
        assert!(p.slice_at(100.0, 100.0).is_none());
        // Pointer just outside the dead-zone radius.
        let just_outside = (p.deadzone_px + 1.0) + 100.0;
        assert!(p.slice_at(just_outside, 100.0).is_some());
    }

    #[test]
    fn pie_slice_at_up_picks_first_slice() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        // Directly above center (screen y decreases up).
        let picked = p.slice_at(0.0, -80.0).unwrap();
        assert_eq!(picked.action, PieAction::FrameSelected);
    }

    #[test]
    fn pie_slice_at_right_picks_third_slice() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        // Pure-right (theta = TAU/4) lands on the wedge boundary between
        // index 1 and index 2; floor() takes it to index 2 = RunSim.
        let picked = p.slice_at(80.0, 0.0).unwrap();
        assert_eq!(picked.action, PieAction::RunSim);
    }

    #[test]
    fn pie_slice_at_upper_right_picks_place_source() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        // Up-right diagonal (theta ≈ TAU/8) lands inside index 1.
        let picked = p.slice_at(60.0, -60.0).unwrap();
        assert_eq!(picked.action, PieAction::PlaceSource);
    }

    #[test]
    fn pie_slice_at_inactive_returns_none() {
        let p = PieMenuState::with_defaults();
        assert!(p.slice_at(80.0, 0.0).is_none());
    }

    #[test]
    fn pie_release_picks_and_closes() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        let picked = p.release(0.0, -80.0);
        assert!(picked.is_some());
        assert!(!p.active);
    }

    #[test]
    fn pie_release_inside_deadzone_returns_none_and_closes() {
        let mut p = PieMenuState::with_defaults();
        p.open(0.0, 0.0);
        let picked = p.release(0.0, 0.0);
        assert!(picked.is_none());
        assert!(!p.active);
    }
}
