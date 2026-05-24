//! Status bar action hints — always-visible context: current mode,
//! next-step hint, active modifiers, and a last-action label with an
//! undo affordance.
//!
//! Pure data: the actual bottom-bar rendering reads [`StatusHints`] each
//! frame in `viewport_3d` and lays out the cells. The renderer never
//! computes hint strings — only the layout.

use crate::ui::InteractionMode;

/// Modifier keys held *this frame*. Mirrors `egui::Modifiers` but lives
/// here so this module is unit-testable without an egui context.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActiveModifiers {
    pub shift: bool,
    pub ctrl_or_cmd: bool,
    pub alt: bool,
}

/// Snapshot of everything the status bar shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusHints {
    /// Current interaction mode (Select / PlaceSource / PlaceListener).
    pub mode_label: &'static str,
    /// Next-step prompt (e.g. "Click viewport to place").
    pub next_step_hint: String,
    /// One-line note describing the last user-driven action ("Moved
    /// Source 1") with a hint that Cmd/Ctrl+Z would undo it.
    pub action_hint: Option<String>,
    /// Live modifier badges — populated each frame from the input state.
    pub modifiers: ActiveModifiers,
    /// Current perf governor label ("perf: healthy" etc.). Empty string
    /// if the host hasn't wired the governor in.
    pub perf_label: &'static str,
}

impl StatusHints {
    /// Compute hints from runtime state. Pure — no egui calls.
    pub fn compute(
        mode: InteractionMode,
        modifiers: ActiveModifiers,
        last_action: Option<&str>,
        selection_count: usize,
    ) -> Self {
        Self::compute_with_perf(mode, modifiers, last_action, selection_count, "")
    }

    /// Compute hints including a `perf_label` from `PerfGovernor::class().label()`.
    pub fn compute_with_perf(
        mode: InteractionMode,
        modifiers: ActiveModifiers,
        last_action: Option<&str>,
        selection_count: usize,
        perf_label: &'static str,
    ) -> Self {
        let mode_label = mode_label(mode);
        let next_step_hint = next_step_for(mode, selection_count);
        let action_hint = last_action.map(|name| format!("{name} (Cmd/Ctrl+Z to undo)"));
        Self {
            mode_label,
            next_step_hint,
            action_hint,
            modifiers,
            perf_label,
        }
    }

    /// Human-readable summary of held modifiers — populates the badge
    /// strip in the status bar. Empty string when no modifier is down.
    pub fn modifier_summary(&self) -> String {
        let mut parts: Vec<&'static str> = Vec::new();
        if self.modifiers.shift {
            parts.push("Shift=snap");
        }
        if self.modifiers.ctrl_or_cmd {
            parts.push("Ctrl=multi");
        }
        if self.modifiers.alt {
            parts.push("Alt=alt");
        }
        parts.join("  •  ")
    }
}

fn mode_label(mode: InteractionMode) -> &'static str {
    match mode {
        InteractionMode::Select => "Select",
        InteractionMode::PlaceSource => "Place Source",
        InteractionMode::PlaceListener => "Place Listener",
    }
}

fn next_step_for(mode: InteractionMode, selection_count: usize) -> String {
    match (mode, selection_count) {
        (InteractionMode::Select, 0) => "Click to select. Press G/R/S to transform.".into(),
        (InteractionMode::Select, n) => {
            format!("{n} selected. G=move, R=rotate, S=scale, H=hide, /=isolate.")
        }
        (InteractionMode::PlaceSource, _) => "Click in viewport to place a sound source.".into(),
        (InteractionMode::PlaceListener, _) => "Click in viewport to place a listener.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_hint_select_mode_no_selection() {
        let h = StatusHints::compute(InteractionMode::Select, ActiveModifiers::default(), None, 0);
        assert_eq!(h.mode_label, "Select");
        assert!(h.next_step_hint.contains("Click to select"));
        assert!(h.action_hint.is_none());
    }

    #[test]
    fn status_hint_select_mode_with_selection_lists_keys() {
        let h = StatusHints::compute(InteractionMode::Select, ActiveModifiers::default(), None, 3);
        assert!(h.next_step_hint.contains("3 selected"));
        assert!(h.next_step_hint.contains("G=move"));
    }

    #[test]
    fn status_hint_place_source_prompts_click() {
        let h = StatusHints::compute(
            InteractionMode::PlaceSource,
            ActiveModifiers::default(),
            None,
            0,
        );
        assert_eq!(h.mode_label, "Place Source");
        assert!(h.next_step_hint.contains("place a sound source"));
    }

    #[test]
    fn status_hint_place_listener_prompts_click() {
        let h = StatusHints::compute(
            InteractionMode::PlaceListener,
            ActiveModifiers::default(),
            None,
            0,
        );
        assert_eq!(h.mode_label, "Place Listener");
        assert!(h.next_step_hint.contains("place a listener"));
    }

    #[test]
    fn status_hint_action_includes_undo_affordance() {
        let h = StatusHints::compute(
            InteractionMode::Select,
            ActiveModifiers::default(),
            Some("Moved Source 1"),
            1,
        );
        assert_eq!(
            h.action_hint.as_deref(),
            Some("Moved Source 1 (Cmd/Ctrl+Z to undo)")
        );
    }

    #[test]
    fn status_modifier_summary_empty_when_nothing_held() {
        let h = StatusHints::compute(InteractionMode::Select, ActiveModifiers::default(), None, 0);
        assert_eq!(h.modifier_summary(), "");
    }

    #[test]
    fn status_modifier_summary_lists_active_modifiers() {
        let mods = ActiveModifiers {
            shift: true,
            ctrl_or_cmd: true,
            alt: false,
        };
        let h = StatusHints::compute(InteractionMode::Select, mods, None, 0);
        let summary = h.modifier_summary();
        assert!(summary.contains("Shift=snap"));
        assert!(summary.contains("Ctrl=multi"));
        assert!(!summary.contains("Alt"));
    }

    #[test]
    fn status_modifier_summary_handles_all_three() {
        let mods = ActiveModifiers {
            shift: true,
            ctrl_or_cmd: true,
            alt: true,
        };
        let h = StatusHints::compute(InteractionMode::Select, mods, None, 0);
        let summary = h.modifier_summary();
        assert!(summary.contains("Shift=snap"));
        assert!(summary.contains("Ctrl=multi"));
        assert!(summary.contains("Alt=alt"));
    }

    #[test]
    fn status_compute_perf_label_propagates() {
        let h = StatusHints::compute_with_perf(
            InteractionMode::Select,
            ActiveModifiers::default(),
            None,
            0,
            "perf: throttled",
        );
        assert_eq!(h.perf_label, "perf: throttled");
    }

    #[test]
    fn status_default_compute_has_empty_perf_label() {
        let h = StatusHints::compute(InteractionMode::Select, ActiveModifiers::default(), None, 0);
        assert!(h.perf_label.is_empty());
    }

    #[test]
    fn status_perf_label_for_each_class_distinct() {
        use crate::renderer::PerfClass;
        let labels = [
            PerfClass::Healthy.label(),
            PerfClass::Degraded.label(),
            PerfClass::Critical.label(),
        ];
        for l in labels {
            let h = StatusHints::compute_with_perf(
                InteractionMode::Select,
                ActiveModifiers::default(),
                None,
                0,
                l,
            );
            assert_eq!(h.perf_label, l);
        }
    }
}
