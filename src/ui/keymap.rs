//! User-customizable keymap (JSON-persisted).
//!
//! Every interactive action in EchoMap has a stable [`ActionId`]. The
//! [`Keymap`] maps each ActionId to a list of [`KeyBinding`]s — typically
//! one, sometimes two (e.g. Redo is bound to both Cmd+Shift+Z and Cmd+Y).
//!
//! Load order at app startup:
//! 1. Built-in defaults from [`Keymap::default`].
//! 2. Overlay user's JSON file from
//!    `$HOME/.config/echomap/keymap.json` (or platform equivalent) if it
//!    exists — overrides any default bindings for actions it mentions.
//!
//! The user can edit this file by hand or via a future in-app editor
//! (Settings → Keymap, follow-up).
//!
//! Match policy: a binding matches the *first frame* its key is pressed
//! with all listed modifiers held and no extra modifiers held. We use
//! egui's `modifiers.matches_exact` semantics (matching the convention
//! Blender uses — `Shift+G` does NOT fire on `Ctrl+Shift+G`).

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use eframe::egui;
use serde::{Deserialize, Serialize};

/// Stable identifier for every customizable action. New actions append to
/// the end — existing serialized keymaps must keep working when the enum
/// grows.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum ActionId {
    // History
    Undo,
    Redo,
    // Palette
    OpenCommandPalette,
    // Camera / view
    ResetCamera,
    FocusSelection,
    ViewTop,
    ViewFront,
    ViewSide,
    ViewPerspective,
    ViewIsometric,
    // Mode
    ModeSelect,
    ModePlaceSource,
    ModePlaceListener,
    // Gizmo
    GizmoTranslate,
    GizmoRotate,
    GizmoScale,
    GizmoAxisX,
    GizmoAxisY,
    GizmoAxisZ,
    GizmoConfirm,
    GizmoCancel,
    // Selection
    DeleteSelection,
    EscapeSelection,
    // Toggles
    ToggleGrid,
    ToggleFlyMode,
    ToggleTeleop,
    // Scene
    SaveScene,
    LoadScene,
    // Multi-selection (D7)
    SelectAll,
    DeselectAll,
    HideSelection,
    UnhideAll,
    ToggleIsolate,
    BoxSelect,
}

impl ActionId {
    pub fn label(&self) -> &'static str {
        use ActionId::*;
        match self {
            Undo => "Undo",
            Redo => "Redo",
            OpenCommandPalette => "Open Command Palette",
            ResetCamera => "Reset Camera",
            FocusSelection => "Focus Selection",
            ViewTop => "View: Top",
            ViewFront => "View: Front",
            ViewSide => "View: Side",
            ViewPerspective => "View: Perspective",
            ViewIsometric => "View: Isometric",
            ModeSelect => "Mode: Select",
            ModePlaceSource => "Mode: Place Source",
            ModePlaceListener => "Mode: Place Listener",
            GizmoTranslate => "Gizmo: Translate",
            GizmoRotate => "Gizmo: Rotate",
            GizmoScale => "Gizmo: Scale",
            GizmoAxisX => "Gizmo: Axis X",
            GizmoAxisY => "Gizmo: Axis Y",
            GizmoAxisZ => "Gizmo: Axis Z",
            GizmoConfirm => "Gizmo: Confirm",
            GizmoCancel => "Gizmo: Cancel",
            DeleteSelection => "Delete Selection",
            EscapeSelection => "Clear Selection",
            ToggleGrid => "Toggle Grid",
            ToggleFlyMode => "Toggle Fly Mode",
            ToggleTeleop => "Toggle Tele-op",
            SaveScene => "Save Scene",
            LoadScene => "Load Scene",
            SelectAll => "Select All",
            DeselectAll => "Deselect All",
            HideSelection => "Hide Selection",
            UnhideAll => "Unhide All",
            ToggleIsolate => "Toggle Isolate",
            BoxSelect => "Box Select",
        }
    }
}

/// A keybinding — key + modifier set.
///
/// Serialized as JSON like `{"key":"Z","cmd":true,"shift":true}` —
/// modifier fields are omitted when false to keep the file readable.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KeyBinding {
    pub key: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub cmd: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub ctrl: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub shift: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub alt: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl KeyBinding {
    pub fn key(key: &str) -> Self {
        Self {
            key: key.to_string(),
            cmd: false,
            ctrl: false,
            shift: false,
            alt: false,
        }
    }
    pub fn with_cmd(mut self) -> Self {
        self.cmd = true;
        self
    }
    pub fn with_ctrl(mut self) -> Self {
        self.ctrl = true;
        self
    }
    pub fn with_shift(mut self) -> Self {
        self.shift = true;
        self
    }
    pub fn with_alt(mut self) -> Self {
        self.alt = true;
        self
    }

    /// Human-readable representation: `"Cmd+Shift+Z"`. Stable across runs;
    /// safe to show in tooltips and the future keymap editor.
    pub fn pretty(&self) -> String {
        let mut parts = Vec::new();
        if self.cmd {
            parts.push("Cmd");
        }
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        parts.push(&self.key);
        parts.join("+")
    }

    /// Test whether `input.key_pressed(key)` AND modifiers match exactly
    /// this binding.
    pub fn matches(&self, input: &egui::InputState) -> bool {
        let Some(key) = egui_key_from_str(&self.key) else {
            return false;
        };
        if !input.key_pressed(key) {
            return false;
        }
        let m = input.modifiers;
        // `command` is mac-cmd or ctrl-on-others. Distinguish from raw ctrl
        // so users on Mac can bind Cmd+Z while preserving Ctrl+Z as a
        // separate, non-conflicting binding if desired.
        let want_cmd = self.cmd;
        let want_ctrl = self.ctrl;
        let got_cmd = m.command;
        let got_ctrl = m.ctrl && !m.command; // ctrl on non-mac is also command
        if cfg!(target_os = "macos") {
            // mac: command is Cmd, ctrl is raw Ctrl; treat them separately.
            if want_cmd != got_cmd {
                return false;
            }
            if want_ctrl != m.ctrl {
                return false;
            }
        } else {
            // other: command == ctrl. Honour either bind.
            let want_either = want_cmd || want_ctrl;
            if want_either != got_cmd {
                return false;
            }
            // Ignore raw ctrl/cmd distinction.
            let _ = got_ctrl;
        }
        if self.shift != m.shift {
            return false;
        }
        if self.alt != m.alt {
            return false;
        }
        true
    }
}

/// JSON-serializable mapping from ActionId → list of bindings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Keymap {
    pub bindings: BTreeMap<ActionId, Vec<KeyBinding>>,
}

impl Keymap {
    /// Built-in default bindings — match the hard-coded behaviour
    /// shipped before the keymap module existed, plus the new gizmo + snap
    /// + palette additions from deliverables 2–4.
    pub fn defaults() -> Self {
        let mut m = BTreeMap::new();
        m.insert(ActionId::Undo, vec![KeyBinding::key("Z").with_cmd()]);
        m.insert(
            ActionId::Redo,
            vec![
                KeyBinding::key("Z").with_cmd().with_shift(),
                KeyBinding::key("Y").with_cmd(),
            ],
        );
        m.insert(
            ActionId::OpenCommandPalette,
            vec![KeyBinding::key("K").with_cmd()],
        );
        m.insert(ActionId::ResetCamera, vec![KeyBinding::key("R")]);
        m.insert(ActionId::FocusSelection, vec![KeyBinding::key("F")]);
        m.insert(ActionId::ViewTop, vec![KeyBinding::key("Num7")]);
        m.insert(
            ActionId::ViewFront,
            vec![KeyBinding::key("Num1").with_ctrl()],
        );
        m.insert(
            ActionId::ViewSide,
            vec![KeyBinding::key("Num3").with_ctrl()],
        );
        m.insert(ActionId::ViewPerspective, vec![KeyBinding::key("Num0")]);
        m.insert(ActionId::ViewIsometric, vec![KeyBinding::key("Num5")]);
        m.insert(ActionId::ModeSelect, vec![KeyBinding::key("Num1")]);
        m.insert(ActionId::ModePlaceSource, vec![KeyBinding::key("Num2")]);
        m.insert(ActionId::ModePlaceListener, vec![KeyBinding::key("Num3")]);
        m.insert(ActionId::GizmoTranslate, vec![KeyBinding::key("G")]);
        m.insert(ActionId::GizmoRotate, vec![KeyBinding::key("R")]);
        m.insert(ActionId::GizmoScale, vec![KeyBinding::key("S")]);
        m.insert(ActionId::GizmoAxisX, vec![KeyBinding::key("X")]);
        m.insert(ActionId::GizmoAxisY, vec![KeyBinding::key("Y")]);
        m.insert(ActionId::GizmoAxisZ, vec![KeyBinding::key("Z")]);
        m.insert(ActionId::GizmoConfirm, vec![KeyBinding::key("Enter")]);
        m.insert(ActionId::GizmoCancel, vec![KeyBinding::key("Escape")]);
        m.insert(
            ActionId::DeleteSelection,
            vec![KeyBinding::key("Delete"), KeyBinding::key("Backspace")],
        );
        m.insert(ActionId::EscapeSelection, vec![KeyBinding::key("Escape")]);
        m.insert(ActionId::ToggleGrid, vec![KeyBinding::key("G").with_alt()]);
        m.insert(ActionId::ToggleFlyMode, vec![KeyBinding::key("Tab")]);
        m.insert(
            ActionId::ToggleTeleop,
            vec![KeyBinding::key("T").with_cmd()],
        );
        m.insert(ActionId::SaveScene, vec![KeyBinding::key("S").with_cmd()]);
        m.insert(ActionId::LoadScene, vec![KeyBinding::key("O").with_cmd()]);
        // Multi-selection (D7) — Blender-style: A select all, Alt+A deselect,
        // B box-select, H hide, Alt+H unhide, / isolate.
        m.insert(ActionId::SelectAll, vec![KeyBinding::key("A")]);
        m.insert(ActionId::DeselectAll, vec![KeyBinding::key("A").with_alt()]);
        m.insert(ActionId::HideSelection, vec![KeyBinding::key("H")]);
        m.insert(ActionId::UnhideAll, vec![KeyBinding::key("H").with_alt()]);
        m.insert(ActionId::ToggleIsolate, vec![KeyBinding::key("Slash")]);
        m.insert(ActionId::BoxSelect, vec![KeyBinding::key("B")]);
        Self { bindings: m }
    }

    /// Read a keymap JSON file and overlay it on the built-in defaults.
    /// Returns the defaults unchanged if the file doesn't exist.
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut km = Self::defaults();
        if !path.exists() {
            return Ok(km);
        }
        let s = fs::read_to_string(path)?;
        let overlay: Keymap =
            serde_json::from_str(&s).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        for (action, bindings) in overlay.bindings {
            km.bindings.insert(action, bindings);
        }
        Ok(km)
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// True if the current input frame triggers `action`.
    pub fn triggered(&self, action: ActionId, input: &egui::InputState) -> bool {
        match self.bindings.get(&action) {
            Some(binds) => binds.iter().any(|b| b.matches(input)),
            None => false,
        }
    }

    pub fn bindings_for(&self, action: ActionId) -> &[KeyBinding] {
        self.bindings
            .get(&action)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn reset_to_defaults(&mut self) {
        *self = Self::defaults();
    }

    /// Set the bindings for a single action (overwriting any prior list).
    pub fn rebind(&mut self, action: ActionId, bindings: Vec<KeyBinding>) {
        self.bindings.insert(action, bindings);
    }

    /// Suggested filesystem path for the user's keymap. Honours
    /// `$XDG_CONFIG_HOME` on Linux, falls back to `$HOME/.config/echomap`.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            std::path::PathBuf::from(xdg).join("echomap/keymap.json")
        } else if let Ok(home) = std::env::var("HOME") {
            std::path::PathBuf::from(home).join(".config/echomap/keymap.json")
        } else {
            std::path::PathBuf::from("echomap-keymap.json")
        }
    }
}

/// Map a string key name (as used in JSON) to an [`egui::Key`]. Returns
/// None for unrecognised names — keymap users see a silent ignore rather
/// than a crash.
fn egui_key_from_str(s: &str) -> Option<egui::Key> {
    use egui::Key;
    Some(match s {
        "A" => Key::A,
        "B" => Key::B,
        "C" => Key::C,
        "D" => Key::D,
        "E" => Key::E,
        "F" => Key::F,
        "G" => Key::G,
        "H" => Key::H,
        "I" => Key::I,
        "J" => Key::J,
        "K" => Key::K,
        "L" => Key::L,
        "M" => Key::M,
        "N" => Key::N,
        "O" => Key::O,
        "P" => Key::P,
        "Q" => Key::Q,
        "R" => Key::R,
        "S" => Key::S,
        "T" => Key::T,
        "U" => Key::U,
        "V" => Key::V,
        "W" => Key::W,
        "X" => Key::X,
        "Y" => Key::Y,
        "Z" => Key::Z,
        "Num0" => Key::Num0,
        "Num1" => Key::Num1,
        "Num2" => Key::Num2,
        "Num3" => Key::Num3,
        "Num4" => Key::Num4,
        "Num5" => Key::Num5,
        "Num6" => Key::Num6,
        "Num7" => Key::Num7,
        "Num8" => Key::Num8,
        "Num9" => Key::Num9,
        "Space" => Key::Space,
        "Tab" => Key::Tab,
        "Enter" => Key::Enter,
        "Escape" | "Esc" => Key::Escape,
        "Delete" | "Del" => Key::Delete,
        "Backspace" => Key::Backspace,
        "Home" => Key::Home,
        "End" => Key::End,
        "PageUp" => Key::PageUp,
        "PageDown" => Key::PageDown,
        "ArrowUp" | "Up" => Key::ArrowUp,
        "ArrowDown" | "Down" => Key::ArrowDown,
        "ArrowLeft" | "Left" => Key::ArrowLeft,
        "ArrowRight" | "Right" => Key::ArrowRight,
        "OpenBracket" | "[" => Key::OpenBracket,
        "CloseBracket" | "]" => Key::CloseBracket,
        "Slash" | "/" => Key::Slash,
        "F1" => Key::F1,
        "F2" => Key::F2,
        "F3" => Key::F3,
        "F4" => Key::F4,
        "F5" => Key::F5,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn keymap_defaults_have_core_actions() {
        let km = Keymap::defaults();
        // Every ActionId should be in the default keymap.
        for id in [
            ActionId::Undo,
            ActionId::Redo,
            ActionId::OpenCommandPalette,
            ActionId::ResetCamera,
            ActionId::FocusSelection,
            ActionId::GizmoTranslate,
            ActionId::DeleteSelection,
        ] {
            assert!(
                !km.bindings_for(id).is_empty(),
                "default keymap missing {:?}",
                id
            );
        }
    }

    #[test]
    fn keymap_undo_default_is_cmd_z() {
        let km = Keymap::defaults();
        let b = &km.bindings_for(ActionId::Undo)[0];
        assert_eq!(b.key, "Z");
        assert!(b.cmd);
        assert!(!b.shift);
    }

    #[test]
    fn keymap_redo_has_two_bindings() {
        let km = Keymap::defaults();
        let b = km.bindings_for(ActionId::Redo);
        assert_eq!(b.len(), 2, "redo should bind both Cmd+Shift+Z and Cmd+Y");
        assert!(b.iter().any(|bind| bind.key == "Z" && bind.shift));
        assert!(b.iter().any(|bind| bind.key == "Y" && !bind.shift));
    }

    #[test]
    fn keybinding_pretty_includes_all_modifiers() {
        let b = KeyBinding::key("Z").with_cmd().with_shift();
        assert_eq!(b.pretty(), "Cmd+Shift+Z");
        let b = KeyBinding::key("Tab");
        assert_eq!(b.pretty(), "Tab");
        let b = KeyBinding::key("F1").with_ctrl().with_alt();
        assert_eq!(b.pretty(), "Ctrl+Alt+F1");
    }

    #[test]
    fn keymap_roundtrips_through_json() {
        let km = Keymap::defaults();
        let json = serde_json::to_string(&km).unwrap();
        let parsed: Keymap = serde_json::from_str(&json).unwrap();
        for id in [
            ActionId::Undo,
            ActionId::GizmoTranslate,
            ActionId::ToggleFlyMode,
        ] {
            assert_eq!(
                km.bindings_for(id).len(),
                parsed.bindings_for(id).len(),
                "binding count for {:?} should match",
                id
            );
        }
    }

    #[test]
    fn keymap_load_returns_defaults_if_no_file() {
        let km = Keymap::load(Path::new("/tmp/echomap-nonexistent-keymap.json")).unwrap();
        assert!(!km.bindings_for(ActionId::Undo).is_empty());
    }

    #[test]
    fn keymap_load_overlays_user_overrides_on_defaults() {
        // Write a partial overlay file — only overrides Undo. The other
        // defaults must remain.
        let tmp = std::env::temp_dir().join("echomap-test-keymap.json");
        std::fs::write(&tmp, r#"{"bindings":{"Undo":[{"key":"Backspace"}]}}"#).unwrap();

        let km = Keymap::load(&tmp).unwrap();
        let undo = km.bindings_for(ActionId::Undo);
        assert_eq!(undo.len(), 1);
        assert_eq!(undo[0].key, "Backspace");

        // ResetCamera default must still be present.
        assert!(!km.bindings_for(ActionId::ResetCamera).is_empty());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn keymap_save_and_reload_roundtrip() {
        let tmp = std::env::temp_dir().join("echomap-test-keymap-save.json");
        let mut km = Keymap::defaults();
        km.rebind(ActionId::Undo, vec![KeyBinding::key("U").with_cmd()]);
        km.save(&tmp).unwrap();

        let reloaded = Keymap::load(&tmp).unwrap();
        let undo = reloaded.bindings_for(ActionId::Undo);
        assert_eq!(undo.len(), 1);
        assert_eq!(undo[0].key, "U");
        assert!(undo[0].cmd);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn keymap_reset_to_defaults_restores() {
        let mut km = Keymap::defaults();
        km.rebind(ActionId::Undo, vec![]);
        assert!(km.bindings_for(ActionId::Undo).is_empty());
        km.reset_to_defaults();
        assert!(!km.bindings_for(ActionId::Undo).is_empty());
    }

    #[test]
    fn keymap_action_labels_unique() {
        // Bare-minimum check: action labels distinguishable in editor UI.
        let labels: HashSet<&'static str> = [
            ActionId::Undo,
            ActionId::Redo,
            ActionId::OpenCommandPalette,
            ActionId::ResetCamera,
            ActionId::FocusSelection,
            ActionId::GizmoTranslate,
            ActionId::GizmoRotate,
            ActionId::GizmoScale,
            ActionId::DeleteSelection,
        ]
        .iter()
        .map(|a| a.label())
        .collect();
        assert!(labels.len() >= 9);
    }

    #[test]
    fn keymap_default_path_uses_xdg_when_set() {
        let saved = std::env::var("XDG_CONFIG_HOME").ok();
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/test-xdg");
        let p = Keymap::default_path();
        assert!(p.to_string_lossy().contains("/tmp/test-xdg"));
        assert!(p.to_string_lossy().ends_with("keymap.json"));
        match saved {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    fn keymap_load_rejects_malformed_json() {
        let tmp = std::env::temp_dir().join("echomap-test-keymap-bad.json");
        std::fs::write(&tmp, "{not valid json").unwrap();
        let r = Keymap::load(&tmp);
        assert!(r.is_err());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn keymap_serialized_omits_default_modifier_fields() {
        let b = KeyBinding::key("Tab");
        let json = serde_json::to_string(&b).unwrap();
        // No modifier flags set → no cmd/ctrl/shift/alt keys in JSON.
        assert!(!json.contains("cmd"));
        assert!(!json.contains("shift"));
        assert!(!json.contains("ctrl"));
        assert!(!json.contains("alt"));
        assert!(json.contains("Tab"));
    }
}
