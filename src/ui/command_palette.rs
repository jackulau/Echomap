//! Fuzzy-search command palette (Cmd/Ctrl+K).
//!
//! Surfaces every action in the app behind a single shortcut. Inspired by
//! Blender's F3 search, Cinema 4D's Cmd+E command finder, and SolidWorks's
//! "Search Commands" box.
//!
//! Architecture: this module owns the palette UI + fuzzy match logic. The
//! list of actions is defined here, but *executing* an action is the caller's
//! job — [`CommandPalette::show`] returns the picked [`Action`] and the
//! viewport dispatches it. This keeps the palette free of references to
//! scene/sim/agent state.

use eframe::egui;

use crate::renderer::CameraView;
use crate::ui::InteractionMode;

/// Every command the palette can issue. Add a variant + entry in
/// [`Action::ALL`] to register a new command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    // History
    Undo,
    Redo,

    // Camera / view
    ResetCamera,
    FocusSelection,
    SetView(CameraView),
    ToggleFlyMode,

    // Visibility toggles
    ToggleGrid,
    ToggleShaded,
    ToggleRays,
    ToggleRobots,
    ToggleSourcesVisibility,
    ToggleListenersVisibility,
    ToggleMeshesVisibility,

    // Placement / modes
    SetMode(InteractionMode),

    // Scene authoring
    AddSource,
    AddListener,
    AddPartitionWall,
    AddPlatform,
    DeleteSelected,
    NewScene,

    // Simulation
    RunSimulation,

    // Misc
    ToggleTeleop,
    ToggleSettings,
    ToggleAbout,
}

impl Action {
    /// Stable registry of every command. Order here is the default display
    /// order when the user opens the palette with an empty query.
    pub const ALL: &'static [Action] = &[
        Action::Undo,
        Action::Redo,
        Action::ResetCamera,
        Action::FocusSelection,
        Action::SetView(CameraView::Perspective),
        Action::SetView(CameraView::Top),
        Action::SetView(CameraView::Front),
        Action::SetView(CameraView::Side),
        Action::SetView(CameraView::Isometric),
        Action::ToggleFlyMode,
        Action::ToggleGrid,
        Action::ToggleShaded,
        Action::ToggleRays,
        Action::ToggleRobots,
        Action::ToggleSourcesVisibility,
        Action::ToggleListenersVisibility,
        Action::ToggleMeshesVisibility,
        Action::SetMode(InteractionMode::Select),
        Action::SetMode(InteractionMode::PlaceSource),
        Action::SetMode(InteractionMode::PlaceListener),
        Action::AddSource,
        Action::AddListener,
        Action::AddPartitionWall,
        Action::AddPlatform,
        Action::DeleteSelected,
        Action::NewScene,
        Action::RunSimulation,
        Action::ToggleTeleop,
        Action::ToggleSettings,
        Action::ToggleAbout,
    ];

    pub fn label(&self) -> &'static str {
        use Action::*;
        match self {
            Undo => "Undo",
            Redo => "Redo",
            ResetCamera => "Reset Camera",
            FocusSelection => "Focus Selection",
            SetView(CameraView::Perspective) => "View: Perspective",
            SetView(CameraView::Top) => "View: Top",
            SetView(CameraView::Front) => "View: Front",
            SetView(CameraView::Side) => "View: Side",
            SetView(CameraView::Isometric) => "View: Isometric",
            SetView(_) => "View: Custom",
            ToggleFlyMode => "Toggle Fly Mode",
            ToggleGrid => "Toggle Grid",
            ToggleShaded => "Toggle Shaded Rendering",
            ToggleRays => "Toggle Ray Paths",
            ToggleRobots => "Toggle Robots Visibility",
            ToggleSourcesVisibility => "Toggle Sources Visibility",
            ToggleListenersVisibility => "Toggle Listeners Visibility",
            ToggleMeshesVisibility => "Toggle Meshes Visibility",
            SetMode(InteractionMode::Select) => "Mode: Select",
            SetMode(InteractionMode::PlaceSource) => "Mode: Place Source",
            SetMode(InteractionMode::PlaceListener) => "Mode: Place Listener",
            AddSource => "Add: Sound Source",
            AddListener => "Add: Listener",
            AddPartitionWall => "Add: Partition Wall",
            AddPlatform => "Add: Platform / Stage",
            DeleteSelected => "Delete Selection",
            NewScene => "New Scene",
            RunSimulation => "Run Simulation",
            ToggleTeleop => "Toggle Tele-op Mode",
            ToggleSettings => "Open Settings",
            ToggleAbout => "About EchoMap",
        }
    }

    pub fn category(&self) -> &'static str {
        use Action::*;
        match self {
            Undo | Redo => "history",
            ResetCamera | FocusSelection | SetView(_) | ToggleFlyMode => "view",
            ToggleGrid
            | ToggleShaded
            | ToggleRays
            | ToggleRobots
            | ToggleSourcesVisibility
            | ToggleListenersVisibility
            | ToggleMeshesVisibility => "display",
            SetMode(_) => "mode",
            AddSource | AddListener | AddPartitionWall | AddPlatform | NewScene => "add",
            DeleteSelected => "edit",
            RunSimulation => "sim",
            ToggleTeleop => "robot",
            ToggleSettings | ToggleAbout => "misc",
        }
    }

    /// Pretty "Category › Label" for the palette row.
    pub fn display(&self) -> String {
        format!("{} › {}", self.category(), self.label())
    }

    /// Optional shortcut hint shown right-aligned in the palette row.
    pub fn shortcut_hint(&self) -> Option<&'static str> {
        use Action::*;
        match self {
            Undo => Some("Cmd+Z"),
            Redo => Some("Cmd+Shift+Z"),
            ResetCamera => Some("R"),
            FocusSelection => Some("F"),
            SetView(CameraView::Top) => Some("Num7"),
            SetView(CameraView::Perspective) => Some("Num0"),
            SetView(CameraView::Isometric) => Some("Num5"),
            DeleteSelected => Some("Del"),
            SetMode(InteractionMode::Select) => Some("1"),
            SetMode(InteractionMode::PlaceSource) => Some("2"),
            SetMode(InteractionMode::PlaceListener) => Some("3"),
            ToggleFlyMode => Some("Tab"),
            ToggleTeleop => Some("Cmd+T"),
            _ => None,
        }
    }
}

/// Palette state — small enough to live inside `ViewportState`.
#[derive(Default)]
pub struct CommandPalette {
    pub open: bool,
    pub query: String,
    /// Index into the *filtered* results list (not into `Action::ALL`).
    pub selected: usize,
    /// Set true on the frame the palette opens so the text input grabs focus.
    pub request_focus: bool,
}

impl CommandPalette {
    /// Toggle visibility. Used by the Cmd/Ctrl+K handler.
    pub fn toggle(&mut self) {
        self.open = !self.open;
        if self.open {
            self.query.clear();
            self.selected = 0;
            self.request_focus = true;
        }
    }

    pub fn close(&mut self) {
        self.open = false;
        self.query.clear();
        self.selected = 0;
    }

    /// Return the list of action indices that match the current query,
    /// sorted by ascending score (best match first).
    pub fn filter(&self) -> Vec<usize> {
        let mut scored: Vec<(i32, usize)> = Action::ALL
            .iter()
            .enumerate()
            .filter_map(|(i, a)| fuzzy_score(&self.query, a.label()).map(|s| (s, i)))
            .collect();
        // Stable sort so equal-scoring entries keep registry order.
        scored.sort_by_key(|(s, i)| (*s, *i));
        scored.into_iter().map(|(_, i)| i).collect()
    }
}

/// Subsequence fuzzy score. Returns `None` if `query` is not a subsequence
/// of `label`. Lower scores rank higher.
///
/// Scoring heuristics:
/// - Earlier matches win (last-match-position component)
/// - Consecutive-character matches earn a bonus (e.g. "und" matches "undo"
///   contiguously and beats "uno" in "uno reverse")
/// - Word-boundary matches (start of label or after space / punctuation)
///   earn a bigger bonus
pub fn fuzzy_score(query: &str, label: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let q: Vec<char> = query.chars().flat_map(|c| c.to_lowercase()).collect();
    let l: Vec<char> = label.chars().flat_map(|c| c.to_lowercase()).collect();

    let mut qi = 0;
    let mut score: i32 = 0;
    let mut last_match: i32 = 0;
    let mut prev_was_match = false;
    let mut prev_char: Option<char> = None;

    for (i, ch) in l.iter().enumerate() {
        if qi < q.len() && *ch == q[qi] {
            // Word-boundary bonus.
            let at_word_boundary = match prev_char {
                None => true,
                Some(p) => p == ' ' || p == '_' || p == '-' || p == ':' || p == '›',
            };
            if at_word_boundary {
                score -= 8;
            }
            if prev_was_match {
                score -= 4;
            }
            last_match = i as i32;
            qi += 1;
            prev_was_match = true;
        } else {
            prev_was_match = false;
        }
        prev_char = Some(*ch);
    }

    if qi == q.len() {
        // Earlier matches rank better; longer labels with same prefix get
        // penalized slightly.
        Some(score + last_match + (l.len() as i32) / 8)
    } else {
        None
    }
}

/// Render the palette modal. Returns `Some(Action)` if the user picked one
/// this frame.
///
/// The caller wires `Cmd/Ctrl+K` to [`CommandPalette::toggle`] and dispatches
/// the returned Action against scene/vp/sim/app state.
pub fn show(ctx: &egui::Context, palette: &mut CommandPalette) -> Option<Action> {
    if !palette.open {
        return None;
    }

    let mut chosen: Option<Action> = None;
    let mut close_after = false;

    // Esc closes (consume before egui scrolls or moves focus).
    let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));
    if esc {
        close_after = true;
    }

    egui::Window::new("Command Palette")
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 80.0))
        .resizable(false)
        .collapsible(false)
        .title_bar(false)
        .fixed_size(egui::vec2(520.0, 360.0))
        .show(ctx, |ui| {
            ui.add_space(6.0);
            let resp = ui.add(
                egui::TextEdit::singleline(&mut palette.query)
                    .hint_text("Type a command — fuzzy search")
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Heading),
            );
            if palette.request_focus {
                resp.request_focus();
                palette.request_focus = false;
            }

            let filtered = palette.filter();
            if palette.selected >= filtered.len() {
                palette.selected = filtered.len().saturating_sub(1);
            }

            // Arrow keys cycle selection.
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowDown) && !filtered.is_empty() {
                    palette.selected = (palette.selected + 1) % filtered.len();
                }
                if i.key_pressed(egui::Key::ArrowUp) && !filtered.is_empty() {
                    palette.selected = if palette.selected == 0 {
                        filtered.len() - 1
                    } else {
                        palette.selected - 1
                    };
                }
                if i.key_pressed(egui::Key::Enter) && !filtered.is_empty() {
                    chosen = Some(Action::ALL[filtered[palette.selected]]);
                }
            });

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            if filtered.is_empty() {
                ui.label(
                    egui::RichText::new("No matching commands.")
                        .italics()
                        .color(egui::Color32::GRAY),
                );
            } else {
                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        for (row, &action_idx) in filtered.iter().enumerate() {
                            let action = Action::ALL[action_idx];
                            let selected = row == palette.selected;
                            let row_resp = ui.horizontal(|ui| {
                                let label =
                                    egui::RichText::new(action.display()).color(if selected {
                                        egui::Color32::WHITE
                                    } else {
                                        egui::Color32::LIGHT_GRAY
                                    });
                                ui.add(egui::Label::new(label).selectable(false));
                                if let Some(hint) = action.shortcut_hint() {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(
                                                egui::RichText::new(hint)
                                                    .small()
                                                    .color(egui::Color32::DARK_GRAY),
                                            );
                                        },
                                    );
                                }
                            });
                            if selected {
                                ui.painter().rect_stroke(
                                    row_resp.response.rect.expand(2.0),
                                    2.0,
                                    egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 160, 220)),
                                    egui::StrokeKind::Outside,
                                );
                            }
                            if row_resp.response.interact(egui::Sense::click()).clicked() {
                                chosen = Some(action);
                            }
                        }
                    });
            }

            ui.add_space(4.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.small(
                    egui::RichText::new(format!("{} commands", filtered.len()))
                        .color(egui::Color32::DARK_GRAY),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.small(
                        egui::RichText::new("↑↓ navigate  ↵ run  Esc close")
                            .color(egui::Color32::DARK_GRAY),
                    );
                });
            });
        });

    if chosen.is_some() || close_after {
        palette.close();
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_palette_starts_closed() {
        let p = CommandPalette::default();
        assert!(!p.open);
        assert!(p.query.is_empty());
    }

    #[test]
    fn command_palette_toggle_opens_and_clears() {
        let mut p = CommandPalette {
            query: "stale".into(),
            selected: 3,
            ..Default::default()
        };
        p.toggle();
        assert!(p.open);
        assert!(p.query.is_empty());
        assert_eq!(p.selected, 0);
        assert!(p.request_focus);
    }

    #[test]
    fn command_palette_close_resets() {
        let mut p = CommandPalette::default();
        p.toggle();
        p.query = "undo".into();
        p.selected = 2;
        p.close();
        assert!(!p.open);
        assert!(p.query.is_empty());
        assert_eq!(p.selected, 0);
    }

    #[test]
    fn fuzzy_empty_query_matches_everything() {
        for a in Action::ALL {
            assert_eq!(fuzzy_score("", a.label()), Some(0));
        }
    }

    #[test]
    fn fuzzy_exact_subsequence_matches() {
        assert!(fuzzy_score("undo", "Undo").is_some());
        assert!(fuzzy_score("rst", "Reset Camera").is_some());
        assert!(fuzzy_score("addsrc", "Add: Sound Source").is_some());
    }

    #[test]
    fn fuzzy_no_match_returns_none() {
        assert!(fuzzy_score("zzz", "Undo").is_none());
        assert!(fuzzy_score("xq", "Reset Camera").is_none());
    }

    #[test]
    fn fuzzy_word_boundary_outranks_inline() {
        // "set" should match "Set View" better than "Settings" because
        // "Settings" buries the query inside the word, while "Set View"
        // hits word boundary right away (well, both do; the point is
        // verifying word-start matches earn a bonus). Use an asymmetric
        // case instead.
        let exact_start = fuzzy_score("undo", "Undo").unwrap();
        let buried = fuzzy_score("undo", "Run undo handler check").unwrap();
        assert!(
            exact_start < buried,
            "exact-start match {exact_start} should rank above buried match {buried}"
        );
    }

    #[test]
    fn fuzzy_consecutive_better_than_spread() {
        let consec = fuzzy_score("res", "Reset Camera").unwrap();
        // Force a non-consecutive subsequence on the same letters: r..e..s
        let spread = fuzzy_score("res", "Render Sources").unwrap();
        assert!(
            consec < spread,
            "consecutive match {consec} should beat spread match {spread}"
        );
    }

    #[test]
    fn filter_returns_indices_in_score_order() {
        let p = CommandPalette {
            query: "undo".into(),
            ..Default::default()
        };
        let result = p.filter();
        assert!(!result.is_empty(), "should find at least 'Undo'");
        // First hit must be the literal Undo action.
        assert_eq!(Action::ALL[result[0]], Action::Undo);
    }

    #[test]
    fn filter_with_empty_query_returns_all() {
        let p = CommandPalette::default();
        let result = p.filter();
        assert_eq!(result.len(), Action::ALL.len());
    }

    #[test]
    fn action_label_unique_per_variant() {
        use std::collections::HashSet;
        let labels: HashSet<&'static str> = Action::ALL.iter().map(|a| a.label()).collect();
        assert_eq!(
            labels.len(),
            Action::ALL.len(),
            "every action must have a unique label"
        );
    }

    #[test]
    fn action_display_contains_category_and_label() {
        for a in Action::ALL {
            let d = a.display();
            assert!(d.contains(a.category()));
            assert!(d.contains(a.label()));
        }
    }
}
