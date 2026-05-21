//! Per-row outliner state: visibility + lock flags for sources, listeners,
//! and objects.
//!
//! Objects already have `MeshObject::visible`; sources and listeners don't
//! own a `visible` field, so the outliner needs its own side-table. Lock
//! state is purely UI — when a row is locked, clicks in the outliner are
//! treated as a no-op so the user can't accidentally select & nudge a
//! pinned reference.
//!
//! Stored as `BTreeSet<usize>` of *hidden* / *locked* indices: empty set ==
//! default state, which keeps serialization small if persisted later.

use std::collections::BTreeSet;

use crate::ui::Selection;

#[derive(Default, Clone, Debug)]
pub struct OutlinerRows {
    pub hidden_sources: BTreeSet<usize>,
    pub hidden_listeners: BTreeSet<usize>,
    pub locked_sources: BTreeSet<usize>,
    pub locked_listeners: BTreeSet<usize>,
    pub locked_objects: BTreeSet<usize>,
}

impl OutlinerRows {
    pub fn is_source_visible(&self, idx: usize) -> bool {
        !self.hidden_sources.contains(&idx)
    }
    pub fn is_listener_visible(&self, idx: usize) -> bool {
        !self.hidden_listeners.contains(&idx)
    }

    pub fn toggle_source_visibility(&mut self, idx: usize) {
        if !self.hidden_sources.insert(idx) {
            self.hidden_sources.remove(&idx);
        }
    }
    pub fn toggle_listener_visibility(&mut self, idx: usize) {
        if !self.hidden_listeners.insert(idx) {
            self.hidden_listeners.remove(&idx);
        }
    }

    pub fn is_source_locked(&self, idx: usize) -> bool {
        self.locked_sources.contains(&idx)
    }
    pub fn is_listener_locked(&self, idx: usize) -> bool {
        self.locked_listeners.contains(&idx)
    }
    pub fn is_object_locked(&self, idx: usize) -> bool {
        self.locked_objects.contains(&idx)
    }

    pub fn toggle_source_lock(&mut self, idx: usize) {
        if !self.locked_sources.insert(idx) {
            self.locked_sources.remove(&idx);
        }
    }
    pub fn toggle_listener_lock(&mut self, idx: usize) {
        if !self.locked_listeners.insert(idx) {
            self.locked_listeners.remove(&idx);
        }
    }
    pub fn toggle_object_lock(&mut self, idx: usize) {
        if !self.locked_objects.insert(idx) {
            self.locked_objects.remove(&idx);
        }
    }

    /// Returns true when the given selection is locked and therefore a click
    /// in the outliner should be a no-op. Locks Robot/RobotLink are not
    /// supported (links are structural).
    pub fn selection_locked(&self, sel: Selection) -> bool {
        match sel {
            Selection::Source(i) => self.is_source_locked(i),
            Selection::Listener(i) => self.is_listener_locked(i),
            Selection::Object(i) => self.is_object_locked(i),
            _ => false,
        }
    }
}

/// Glyphs for the outliner eye/lock buttons. Stored as `&'static str` so
/// labels can be built without allocation.
pub const ICON_EYE_OPEN: &str = "\u{1F441}"; // 👁
pub const ICON_EYE_CLOSED: &str = "—"; // visual cue: hidden
pub const ICON_LOCK_OPEN: &str = "\u{1F513}"; // 🔓
pub const ICON_LOCK_CLOSED: &str = "\u{1F512}"; // 🔒

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outliner_visibility_defaults_to_visible() {
        let r = OutlinerRows::default();
        assert!(r.is_source_visible(0));
        assert!(r.is_listener_visible(99));
    }

    #[test]
    fn outliner_toggle_source_visibility_roundtrips() {
        let mut r = OutlinerRows::default();
        r.toggle_source_visibility(0);
        assert!(!r.is_source_visible(0));
        r.toggle_source_visibility(0);
        assert!(r.is_source_visible(0));
    }

    #[test]
    fn outliner_toggle_listener_visibility_roundtrips() {
        let mut r = OutlinerRows::default();
        r.toggle_listener_visibility(2);
        assert!(!r.is_listener_visible(2));
        r.toggle_listener_visibility(2);
        assert!(r.is_listener_visible(2));
    }

    #[test]
    fn outliner_lock_defaults_to_unlocked() {
        let r = OutlinerRows::default();
        assert!(!r.is_source_locked(0));
        assert!(!r.is_listener_locked(1));
        assert!(!r.is_object_locked(2));
    }

    #[test]
    fn outliner_toggle_source_lock_roundtrips() {
        let mut r = OutlinerRows::default();
        r.toggle_source_lock(0);
        assert!(r.is_source_locked(0));
        r.toggle_source_lock(0);
        assert!(!r.is_source_locked(0));
    }

    #[test]
    fn outliner_toggle_listener_lock_roundtrips() {
        let mut r = OutlinerRows::default();
        r.toggle_listener_lock(0);
        assert!(r.is_listener_locked(0));
        r.toggle_listener_lock(0);
        assert!(!r.is_listener_locked(0));
    }

    #[test]
    fn outliner_toggle_object_lock_roundtrips() {
        let mut r = OutlinerRows::default();
        r.toggle_object_lock(5);
        assert!(r.is_object_locked(5));
        r.toggle_object_lock(5);
        assert!(!r.is_object_locked(5));
    }

    #[test]
    fn outliner_selection_locked_routes_per_variant() {
        let mut r = OutlinerRows::default();
        r.toggle_source_lock(0);
        r.toggle_listener_lock(1);
        r.toggle_object_lock(2);
        assert!(r.selection_locked(Selection::Source(0)));
        assert!(!r.selection_locked(Selection::Source(1)));
        assert!(r.selection_locked(Selection::Listener(1)));
        assert!(r.selection_locked(Selection::Object(2)));
        assert!(!r.selection_locked(Selection::None));
        assert!(!r.selection_locked(Selection::Robot(0)));
        assert!(!r.selection_locked(Selection::RobotLink(0, 0)));
    }

    #[test]
    fn outliner_visibility_independent_per_index() {
        let mut r = OutlinerRows::default();
        r.toggle_source_visibility(0);
        r.toggle_source_visibility(2);
        assert!(!r.is_source_visible(0));
        assert!(r.is_source_visible(1));
        assert!(!r.is_source_visible(2));
    }

    #[test]
    fn outliner_visibility_lock_orthogonal() {
        let mut r = OutlinerRows::default();
        r.toggle_source_lock(0);
        assert!(r.is_source_visible(0));
        assert!(r.is_source_locked(0));
        r.toggle_source_visibility(0);
        assert!(!r.is_source_visible(0));
        assert!(r.is_source_locked(0));
    }
}
