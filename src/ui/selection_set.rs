//! Multi-selection set + hide/isolate state.
//!
//! EchoMap's legacy [`Selection`] is single-item (one variant chosen).
//! `SelectionSet` extends to a multi-item selection with the conventional
//! editor operations: add, remove, toggle, contains, clear, union, diff.
//!
//! Hide/Isolate state is stored as three index sets — one per pickable
//! collection. Sources/listeners don't have a `visible` field of their own,
//! so the viewport renderer queries [`HiddenState::is_hidden`] before
//! drawing.

use std::collections::BTreeSet;

use crate::ui::Selection;

/// Ordered set of [`Selection`] values. Ordered so primary-selection
/// (the "active" item, e.g. for properties panel) is always the head.
#[derive(Default, Clone, Debug)]
pub struct SelectionSet {
    items: Vec<Selection>,
}

impl SelectionSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Wrap a single selection. `Selection::None` produces an empty set.
    pub fn single(sel: Selection) -> Self {
        if matches!(sel, Selection::None) {
            Self::default()
        } else {
            Self { items: vec![sel] }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn contains(&self, sel: Selection) -> bool {
        self.items.iter().any(|s| selection_eq(*s, sel))
    }

    /// Primary selection — the first item, returned as `Selection::None`
    /// when empty. Useful for legacy single-select code paths.
    pub fn primary(&self) -> Selection {
        self.items.first().copied().unwrap_or(Selection::None)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Selection> {
        self.items.iter()
    }

    /// Add a selection. Skips duplicates and `Selection::None`.
    pub fn add(&mut self, sel: Selection) {
        if matches!(sel, Selection::None) {
            return;
        }
        if !self.contains(sel) {
            self.items.push(sel);
        }
    }

    /// Remove a selection. No-op if not present.
    pub fn remove(&mut self, sel: Selection) {
        self.items.retain(|s| !selection_eq(*s, sel));
    }

    /// Toggle: add if absent, remove if present. Used by Ctrl-click.
    pub fn toggle(&mut self, sel: Selection) {
        if matches!(sel, Selection::None) {
            return;
        }
        if self.contains(sel) {
            self.remove(sel);
        } else {
            self.add(sel);
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Replace contents with a single selection.
    pub fn set_single(&mut self, sel: Selection) {
        self.items.clear();
        self.add(sel);
    }
}

/// Selection variants don't derive Eq (Float content), so we compare by
/// variant + indices.
fn selection_eq(a: Selection, b: Selection) -> bool {
    use Selection::*;
    match (a, b) {
        (None, None) => true,
        (Source(i), Source(j)) => i == j,
        (Listener(i), Listener(j)) => i == j,
        (Object(i), Object(j)) => i == j,
        (Robot(i), Robot(j)) => i == j,
        (RobotLink(r1, l1), RobotLink(r2, l2)) => r1 == r2 && l1 == l2,
        _ => false,
    }
}

/// Per-collection hidden-index sets. The viewport draw path checks
/// [`HiddenState::is_*_hidden`] before rendering each item.
///
/// Isolate is implemented by hiding everything not in the selection set —
/// stored as additional bits on top of explicit user-hide so toggling
/// isolate off doesn't lose the user's prior hides.
#[derive(Default, Clone, Debug)]
pub struct HiddenState {
    pub hidden_sources: BTreeSet<usize>,
    pub hidden_listeners: BTreeSet<usize>,
    pub hidden_objects: BTreeSet<usize>,
    pub hidden_robots: BTreeSet<usize>,
    /// True when isolate mode is on. Renderer treats unselected items as
    /// hidden when this is set.
    pub isolate: bool,
}

impl HiddenState {
    pub fn is_source_hidden(&self, idx: usize) -> bool {
        self.hidden_sources.contains(&idx)
    }
    pub fn is_listener_hidden(&self, idx: usize) -> bool {
        self.hidden_listeners.contains(&idx)
    }
    pub fn is_object_hidden(&self, idx: usize) -> bool {
        self.hidden_objects.contains(&idx)
    }
    pub fn is_robot_hidden(&self, idx: usize) -> bool {
        self.hidden_robots.contains(&idx)
    }

    /// Hide each item in the selection set.
    pub fn hide_selection(&mut self, sel: &SelectionSet) {
        for s in sel.iter() {
            match s {
                Selection::Source(i) => {
                    self.hidden_sources.insert(*i);
                }
                Selection::Listener(i) => {
                    self.hidden_listeners.insert(*i);
                }
                Selection::Object(i) => {
                    self.hidden_objects.insert(*i);
                }
                Selection::Robot(i) => {
                    self.hidden_robots.insert(*i);
                }
                _ => {}
            }
        }
    }

    /// Clear every hidden flag. Used by Alt+H "unhide all".
    pub fn unhide_all(&mut self) {
        self.hidden_sources.clear();
        self.hidden_listeners.clear();
        self.hidden_objects.clear();
        self.hidden_robots.clear();
        self.isolate = false;
    }

    /// Toggle isolate. Does NOT change explicit user-hide state — when
    /// isolate is off, those persist.
    pub fn toggle_isolate(&mut self) {
        self.isolate = !self.isolate;
    }

    /// Reports whether a source should render given hide + isolate state
    /// against the current selection set.
    pub fn render_source(&self, idx: usize, selection: &SelectionSet) -> bool {
        if self.is_source_hidden(idx) {
            return false;
        }
        if self.isolate && !selection.contains(Selection::Source(idx)) {
            return false;
        }
        true
    }
    pub fn render_listener(&self, idx: usize, selection: &SelectionSet) -> bool {
        if self.is_listener_hidden(idx) {
            return false;
        }
        if self.isolate && !selection.contains(Selection::Listener(idx)) {
            return false;
        }
        true
    }
    pub fn render_object(&self, idx: usize, selection: &SelectionSet) -> bool {
        if self.is_object_hidden(idx) {
            return false;
        }
        if self.isolate && !selection.contains(Selection::Object(idx)) {
            return false;
        }
        true
    }
}

/// Inputs needed to enumerate pickable items for "select all".
/// Counts only — every concrete index becomes a [`Selection`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PickableCounts {
    pub sources: usize,
    pub listeners: usize,
    pub objects: usize,
    pub robots: usize,
}

/// Replace `set` with every concrete pickable item. Used by `A` (select-all).
pub fn select_all(set: &mut SelectionSet, counts: PickableCounts) {
    set.clear();
    for i in 0..counts.sources {
        set.add(Selection::Source(i));
    }
    for i in 0..counts.listeners {
        set.add(Selection::Listener(i));
    }
    for i in 0..counts.objects {
        set.add(Selection::Object(i));
    }
    for i in 0..counts.robots {
        set.add(Selection::Robot(i));
    }
}

/// Compute the Shift-click range between `anchor` and `target`, inclusive,
/// when both selections are the same variant. Cross-variant ranges produce
/// just the target (range semantics undefined). `None` anchors equivalently
/// produce `single(target)`.
pub fn range_between(anchor: Selection, target: Selection) -> Vec<Selection> {
    use Selection::*;
    let (a, b) = match (anchor, target) {
        (Source(a), Source(b)) => (a, b),
        (Listener(a), Listener(b)) => (a, b),
        (Object(a), Object(b)) => (a, b),
        (Robot(a), Robot(b)) => (a, b),
        _ => return vec![target],
    };
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let ctor: fn(usize) -> Selection = match target {
        Source(_) => Source,
        Listener(_) => Listener,
        Object(_) => Object,
        Robot(_) => Robot,
        _ => return vec![target],
    };
    (lo..=hi).map(ctor).collect()
}

/// Apply the result of a viewport pick under modifier-aware semantics:
/// plain click = set_single, Ctrl/Cmd = toggle, Shift = range from anchor.
/// Returns the new anchor (= target for plain/shift, unchanged for ctrl).
pub fn apply_pick(
    set: &mut SelectionSet,
    anchor: Selection,
    target: Selection,
    ctrl_or_cmd: bool,
    shift: bool,
) -> Selection {
    if matches!(target, Selection::None) {
        if !ctrl_or_cmd && !shift {
            set.clear();
        }
        return Selection::None;
    }
    if shift {
        for sel in range_between(anchor, target) {
            set.add(sel);
        }
        return target;
    }
    if ctrl_or_cmd {
        set.toggle(target);
        return anchor;
    }
    set.set_single(target);
    target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_set_starts_empty() {
        let s = SelectionSet::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(matches!(s.primary(), Selection::None));
    }

    #[test]
    fn selection_set_single_skips_none() {
        let s = SelectionSet::single(Selection::None);
        assert!(s.is_empty());
        let s = SelectionSet::single(Selection::Source(5));
        assert_eq!(s.len(), 1);
        assert!(s.contains(Selection::Source(5)));
    }

    #[test]
    fn selection_set_add_dedupes() {
        let mut s = SelectionSet::new();
        s.add(Selection::Source(0));
        s.add(Selection::Source(0));
        s.add(Selection::Listener(1));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn selection_set_add_ignores_none() {
        let mut s = SelectionSet::new();
        s.add(Selection::None);
        assert!(s.is_empty());
    }

    #[test]
    fn selection_set_remove_works() {
        let mut s = SelectionSet::new();
        s.add(Selection::Source(0));
        s.add(Selection::Listener(1));
        s.remove(Selection::Source(0));
        assert_eq!(s.len(), 1);
        assert!(s.contains(Selection::Listener(1)));
    }

    #[test]
    fn selection_set_toggle_round_trips() {
        let mut s = SelectionSet::new();
        s.toggle(Selection::Source(2));
        assert!(s.contains(Selection::Source(2)));
        s.toggle(Selection::Source(2));
        assert!(!s.contains(Selection::Source(2)));
    }

    #[test]
    fn selection_set_clear_empties() {
        let mut s = SelectionSet::new();
        s.add(Selection::Source(0));
        s.add(Selection::Listener(0));
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn selection_set_set_single_replaces() {
        let mut s = SelectionSet::new();
        s.add(Selection::Source(0));
        s.add(Selection::Listener(1));
        s.set_single(Selection::Object(3));
        assert_eq!(s.len(), 1);
        assert!(s.contains(Selection::Object(3)));
    }

    #[test]
    fn selection_set_primary_is_first_added() {
        let mut s = SelectionSet::new();
        s.add(Selection::Source(5));
        s.add(Selection::Listener(2));
        assert!(matches!(s.primary(), Selection::Source(5)));
    }

    #[test]
    fn hidden_state_hide_selection_marks_indices() {
        let mut sel = SelectionSet::new();
        sel.add(Selection::Source(0));
        sel.add(Selection::Source(2));
        sel.add(Selection::Listener(1));

        let mut h = HiddenState::default();
        h.hide_selection(&sel);
        assert!(h.is_source_hidden(0));
        assert!(!h.is_source_hidden(1));
        assert!(h.is_source_hidden(2));
        assert!(h.is_listener_hidden(1));
    }

    #[test]
    fn hidden_state_unhide_all_clears() {
        let mut h = HiddenState::default();
        h.hidden_sources.insert(0);
        h.hidden_listeners.insert(1);
        h.hidden_objects.insert(2);
        h.isolate = true;
        h.unhide_all();
        assert!(h.hidden_sources.is_empty());
        assert!(h.hidden_listeners.is_empty());
        assert!(h.hidden_objects.is_empty());
        assert!(!h.isolate);
    }

    #[test]
    fn hidden_state_isolate_hides_unselected() {
        let mut sel = SelectionSet::new();
        sel.add(Selection::Source(1));

        let mut h = HiddenState::default();
        h.toggle_isolate();
        assert!(h.isolate);

        // Selected = visible.
        assert!(h.render_source(1, &sel));
        // Unselected = hidden by isolate.
        assert!(!h.render_source(0, &sel));
        assert!(!h.render_source(2, &sel));
    }

    #[test]
    fn hidden_state_isolate_preserves_explicit_hide() {
        let mut sel = SelectionSet::new();
        sel.add(Selection::Source(0));

        let mut h = HiddenState::default();
        h.hidden_sources.insert(0); // user hid even the selected one
        h.toggle_isolate();

        // Even with isolate showing only selection, explicit hide wins.
        assert!(!h.render_source(0, &sel));
    }

    #[test]
    fn hidden_state_isolate_off_shows_unselected() {
        let mut sel = SelectionSet::new();
        sel.add(Selection::Source(0));

        let h = HiddenState::default();
        assert!(h.render_source(0, &sel));
        assert!(h.render_source(99, &sel));
    }

    #[test]
    fn hidden_state_render_listener_and_object_follow_same_rules() {
        let mut sel = SelectionSet::new();
        sel.add(Selection::Listener(3));
        sel.add(Selection::Object(4));

        let mut h = HiddenState::default();
        h.toggle_isolate();
        assert!(h.render_listener(3, &sel));
        assert!(!h.render_listener(0, &sel));
        assert!(h.render_object(4, &sel));
        assert!(!h.render_object(0, &sel));
    }
}
