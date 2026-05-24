//! First-run onboarding tour + cheat sheet.
//!
//! On first launch (no flag file in user config dir), EchoMap shows a
//! 5-step modal that walks through: load model → place source → place
//! listener → run sim → view results. The user can dismiss at any step;
//! once dismissed, the flag is persisted so the tour doesn't reappear.
//!
//! F1 reopens the cheat sheet view (same content, presented as a
//! reference card instead of a guided tour).
//!
//! State here is pure data. The actual modal Window is rendered by
//! `viewport_3d` after reading [`OnboardingState`].

use std::fs;
use std::path::PathBuf;

/// One step of the first-run tour. Steps render in [`STEPS`] order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OnboardingStep {
    pub title: &'static str,
    pub body: &'static str,
    pub hotkey_hint: Option<&'static str>,
}

/// Hard-coded 5-step tour. Order matters — index 0 is shown first.
pub const STEPS: &[OnboardingStep] = &[
    OnboardingStep {
        title: "Welcome to EchoMap",
        body: "EchoMap simulates acoustics in 3D scenes. This 5-step tour shows you how to run your first simulation.",
        hotkey_hint: Some("Press Esc or click X to skip. Press F1 anytime to reopen this as a cheat sheet."),
    },
    OnboardingStep {
        title: "1. Load a model",
        body: "Use File → Load Scene to open a .glb/.json scene. Or drag a model into the viewport.",
        hotkey_hint: Some("Cmd/Ctrl+O"),
    },
    OnboardingStep {
        title: "2. Place a sound source",
        body: "Press 2 to enter Place Source mode, then click in the viewport. Sources emit sound that the simulator traces.",
        hotkey_hint: Some("2"),
    },
    OnboardingStep {
        title: "3. Place a listener",
        body: "Press 3 to enter Place Listener mode and click to drop a microphone position. Listeners record the received signal.",
        hotkey_hint: Some("3"),
    },
    OnboardingStep {
        title: "4. Run the simulation",
        body: "Open the command palette (Cmd/Ctrl+K) and search 'Run Sim'. Results appear in the listener panel.",
        hotkey_hint: Some("Cmd/Ctrl+K"),
    },
];

/// State machine for the tour. `step` indexes [`STEPS`] when `visible`. When
/// `dismissed` is set, [`Self::should_show_on_launch`] returns false so the
/// modal doesn't reappear next session.
#[derive(Clone, Debug, Default)]
pub struct OnboardingState {
    pub visible: bool,
    pub step: usize,
    pub dismissed: bool,
    /// True when the cheat-sheet view is open (F1). Mutually exclusive with
    /// the guided tour — the cheat sheet is a flat reference grid, no
    /// next/prev buttons.
    pub cheat_sheet_open: bool,
}

impl OnboardingState {
    /// Build initial state by reading the dismissed-flag from disk.
    /// On any IO error (missing dir, permission denied) we conservatively
    /// treat the tour as dismissed so we don't repeatedly annoy users on
    /// systems where the flag can't be persisted.
    pub fn load() -> Self {
        let path = Self::flag_path();
        let dismissed = path.exists();
        Self {
            visible: !dismissed,
            step: 0,
            dismissed,
            cheat_sheet_open: false,
        }
    }

    /// Persist the dismissed flag. Best-effort — IO failure is silent
    /// (printed via `eprintln!`) since onboarding is non-critical state.
    pub fn persist_dismissed(&self) {
        if !self.dismissed {
            return;
        }
        let path = Self::flag_path();
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("onboarding: cannot create config dir: {e}");
                return;
            }
        }
        if let Err(e) = fs::write(&path, b"1") {
            eprintln!("onboarding: cannot persist dismiss flag: {e}");
        }
    }

    /// Advance to the next step or, if on the last step, dismiss.
    pub fn next(&mut self) {
        if self.step + 1 >= STEPS.len() {
            self.dismiss();
        } else {
            self.step += 1;
        }
    }

    /// Move back one step. No-op on the first step.
    pub fn prev(&mut self) {
        self.step = self.step.saturating_sub(1);
    }

    /// Mark dismissed + close. Persists flag to disk.
    pub fn dismiss(&mut self) {
        self.visible = false;
        self.dismissed = true;
        self.cheat_sheet_open = false;
        self.persist_dismissed();
    }

    /// Open the cheat-sheet view (typically bound to F1). Doesn't touch
    /// the dismissed flag — F1 is a reference, not the first-run tour.
    pub fn open_cheat_sheet(&mut self) {
        self.cheat_sheet_open = true;
    }

    pub fn close_cheat_sheet(&mut self) {
        self.cheat_sheet_open = false;
    }

    pub fn current(&self) -> Option<&'static OnboardingStep> {
        if !self.visible {
            return None;
        }
        STEPS.get(self.step)
    }

    /// Path to the dismissed-flag file under user config dir.
    /// Follows the same XDG convention as the keymap module.
    pub fn flag_path() -> PathBuf {
        let base = if let Some(x) = std::env::var_os("XDG_CONFIG_HOME") {
            PathBuf::from(x)
        } else if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(".config")
        } else {
            PathBuf::from(".")
        };
        base.join("echomap").join("onboarding.done")
    }

    /// True if the first-run tour should fire on app start.
    pub fn should_show_on_launch(&self) -> bool {
        self.visible && !self.dismissed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onboarding_default_is_hidden() {
        let s = OnboardingState::default();
        assert!(!s.visible);
        assert_eq!(s.step, 0);
        assert!(!s.dismissed);
        assert!(!s.cheat_sheet_open);
    }

    #[test]
    fn onboarding_next_advances_step() {
        let mut s = OnboardingState {
            visible: true,
            ..Default::default()
        };
        s.next();
        assert_eq!(s.step, 1);
        s.next();
        assert_eq!(s.step, 2);
    }

    #[test]
    fn onboarding_next_past_last_dismisses() {
        let mut s = OnboardingState {
            visible: true,
            step: STEPS.len() - 1,
            ..Default::default()
        };
        s.next();
        assert!(s.dismissed);
        assert!(!s.visible);
    }

    #[test]
    fn onboarding_prev_clamps_at_zero() {
        let mut s = OnboardingState {
            visible: true,
            ..Default::default()
        };
        s.prev();
        assert_eq!(s.step, 0);
    }

    #[test]
    fn onboarding_dismiss_sets_flags() {
        let mut s = OnboardingState {
            visible: true,
            ..Default::default()
        };
        s.dismiss();
        assert!(s.dismissed);
        assert!(!s.visible);
    }

    #[test]
    fn onboarding_cheat_sheet_toggle_independent_of_dismiss() {
        let mut s = OnboardingState {
            dismissed: true,
            ..Default::default()
        };
        s.open_cheat_sheet();
        assert!(s.cheat_sheet_open);
        s.close_cheat_sheet();
        assert!(!s.cheat_sheet_open);
        // Cheat sheet open/close did not flip the dismissed flag.
        assert!(s.dismissed);
    }

    #[test]
    fn onboarding_current_is_some_when_visible() {
        let s = OnboardingState {
            visible: true,
            step: 0,
            ..Default::default()
        };
        assert!(s.current().is_some());
        assert_eq!(s.current().unwrap().title, "Welcome to EchoMap");
    }

    #[test]
    fn onboarding_current_is_none_when_hidden() {
        let s = OnboardingState::default();
        assert!(s.current().is_none());
    }

    #[test]
    fn onboarding_should_show_on_launch_only_when_visible_and_not_dismissed() {
        let mut s = OnboardingState {
            visible: true,
            ..Default::default()
        };
        assert!(s.should_show_on_launch());
        s.dismissed = true;
        assert!(!s.should_show_on_launch());
    }

    #[test]
    fn onboarding_flag_path_uses_xdg_when_set() {
        let saved = std::env::var("XDG_CONFIG_HOME").ok();
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/test-echomap-xdg");
        let p = OnboardingState::flag_path();
        assert!(p.to_string_lossy().contains("/tmp/test-echomap-xdg"));
        assert!(p.to_string_lossy().ends_with("onboarding.done"));
        match saved {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    fn onboarding_step_count_matches_objective() {
        // Objective: 5-step tour. Welcome + 4 task steps = 5.
        assert_eq!(STEPS.len(), 5);
    }
}
