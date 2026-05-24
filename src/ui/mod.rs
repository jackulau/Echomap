pub mod command_palette;
pub mod config_validation;
pub mod defaults;
pub mod expr;
pub mod gestures;
pub mod gizmo;
pub mod keymap;
pub mod onboarding;
pub mod outliner;
pub mod pie_menu;
pub mod quad_view;
pub mod scene_io;
pub mod selection_set;
pub mod snap;
pub mod status_hints;

pub use outliner::OutlinerRows;
pub use quad_view::{QuadView, Quadrant};

pub use selection_set::{HiddenState, SelectionSet};
pub use status_hints::{ActiveModifiers, StatusHints};

/// Convenience wrapper: build a [`StatusHints`] from a viewport snapshot.
/// Surfaces `next_step_hint` + `action_hint` so the bottom bar always
/// shows a mode label, a next-step prompt, and (when set) the last
/// action with its undo affordance.
pub fn compute_status_hints(
    mode: InteractionMode,
    modifiers: ActiveModifiers,
    last_action: Option<&str>,
    selection_count: usize,
) -> StatusHints {
    StatusHints::compute(mode, modifiers, last_action, selection_count)
}

pub use command_palette::{Action as PaletteAction, CommandPalette};
pub use gizmo::{AxisLock, GizmoState, TransformMode};
pub use keymap::{ActionId, KeyBinding, Keymap};
pub use snap::{SnapConfig, SnapMode};

/// A right-click context menu entry. Labels are static so the menu can be
/// built without allocation. The `action` is a [`PaletteAction`] so we
/// reuse the existing dispatcher rather than maintain a parallel registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextMenuItem {
    pub label: &'static str,
    pub action: PaletteAction,
}

impl ContextMenuItem {
    pub const fn new(label: &'static str, action: PaletteAction) -> Self {
        Self { label, action }
    }
}

/// Build the context menu for a selection. Pure function — easy to test.
///
/// Returns an empty vec when nothing is selected, signalling the caller
/// should fall back to a scene-level menu.
pub fn context_menu_items_for(selection: Selection) -> Vec<ContextMenuItem> {
    use PaletteAction::*;
    match selection {
        Selection::None => vec![
            ContextMenuItem::new("Add Source", AddSource),
            ContextMenuItem::new("Add Listener", AddListener),
            ContextMenuItem::new("Add Partition Wall", AddPartitionWall),
            ContextMenuItem::new("Add Platform", AddPlatform),
            ContextMenuItem::new("Reset Camera", ResetCamera),
        ],
        Selection::Source(_) => vec![
            ContextMenuItem::new("Focus", FocusSelection),
            ContextMenuItem::new("Delete", DeleteSelected),
        ],
        Selection::Listener(_) => vec![
            ContextMenuItem::new("Focus", FocusSelection),
            ContextMenuItem::new("Delete", DeleteSelected),
        ],
        Selection::Object(_) => vec![
            ContextMenuItem::new("Focus", FocusSelection),
            ContextMenuItem::new("Delete", DeleteSelected),
        ],
        Selection::Robot(_) | Selection::RobotLink(_, _) => {
            vec![ContextMenuItem::new("Focus", FocusSelection)]
        }
    }
}

use glam::Vec3;

use crate::acoustics::SimulationState;
use crate::agent::bridge::{
    create_bridge, AgentActivityLog, AgentEventKind, MessageDirection, SimBridgeClient,
    SimBridgeServer,
};
use crate::agent::demo::{DemoAgentHandle, DemoBehavior};
use crate::agent::{AgentServerConfig, AgentServerHandle};
use crate::fluids::FluidSimulation;
use crate::gas::GasSimulation;
use crate::io::DeviceCaps;
use crate::renderer::{
    energy_to_color, ground_shadow_polygon, project_3d, ray_ground_intersect, render_fluid_slice,
    render_gas_slice, scene_light_dir, screen_to_ray, shade_color, Camera, CameraView,
    FluidVisualizationMode, GasVisualizationMode, PerfGovernor,
};
use crate::robot::definition::{CollisionShape, RobotDefinition, SensorDefinition};
use crate::robot::sensors::sensor_world_pose;
use crate::robot::state::ActuatorCommand;
use crate::robot::RobotManager;
use crate::scene::{
    History, Listener, MaterialLibrary, MediumLibrary, Scene, SceneCommand, SoundSource,
};

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionMode {
    #[default]
    Select,
    PlaceSource,
    PlaceListener,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Selection {
    #[default]
    None,
    Source(usize),
    Listener(usize),
    Object(usize),
    Robot(usize),
    RobotLink(usize, usize),
}

impl Selection {
    /// Human-readable breadcrumb path for the status bar / tooltips.
    pub fn breadcrumb(&self, scene: &Scene, robot_manager: &RobotManager) -> String {
        match self {
            Selection::None => "Scene".to_string(),
            Selection::Source(i) => format!("Scene > Source {}", i + 1),
            Selection::Listener(i) => {
                let name = scene
                    .listeners
                    .get(*i)
                    .map(|l| l.name.as_str())
                    .unwrap_or("?");
                format!("Scene > {}", name)
            }
            Selection::Object(i) => {
                let name = scene.meshes.get(*i).map(|m| m.name.as_str()).unwrap_or("?");
                format!("Scene > {}", name)
            }
            Selection::Robot(r) => {
                let name = robot_manager
                    .robots
                    .get(*r)
                    .map(|rb| rb.definition.name.as_str())
                    .unwrap_or("?");
                format!("Scene > Robot {} ({})", r, name)
            }
            Selection::RobotLink(r, l) => {
                let (rname, lname) = robot_manager
                    .robots
                    .get(*r)
                    .map(|rb| {
                        let ln = rb
                            .definition
                            .links
                            .get(*l)
                            .map(|ld| ld.name.as_str())
                            .unwrap_or("?");
                        (rb.definition.name.as_str(), ln)
                    })
                    .unwrap_or(("?", "?"));
                format!("Scene > Robot {} ({}) > {}", r, rname, lname)
            }
        }
    }
}

pub struct ViewportState {
    pub camera: Camera,
    pub mode: InteractionMode,
    pub selection: Selection,
    pub show_grid: bool,
    pub show_rays: bool,
    pub dragging: bool,
    pub hover_world: Option<Vec3>,
    pub material_lib: MaterialLibrary,
    pub medium_lib: MediumLibrary,
    pub show_fluid: bool,
    pub fluid_viz_mode: FluidVisualizationMode,
    pub fluid_slice_y: usize,
    pub show_gas: bool,
    pub gas_viz_mode: GasVisualizationMode,
    pub gas_slice_y: usize,
    pub gas_species_idx: usize,
    pub selected_robot: usize,
    pub show_robots: bool,
    pub show_sensor_rays: bool,
    pub hit_flash_timer: f32,
    pub hit_flash_robot: Option<usize>,
    pub camera_auto_track: bool,
    pub show_boxing_hud: bool,
    pub boxing_messages: Vec<(f32, String, usize)>,
    /// FPS-style fly camera mode (WASD + right-drag look).
    pub fly_mode: bool,
    /// Movement speed (units/sec) while in fly mode.
    pub fly_speed: f32,
    /// Current named viewpoint; updated when user picks a preset.
    pub current_view: CameraView,
    /// Shaded surface rendering toggle (Lambert fills + drop shadow).
    pub shaded: bool,
    /// Visibility toggles per outliner category.
    pub show_meshes: bool,
    pub show_sources: bool,
    pub show_listeners: bool,
    /// Last hover tooltip target — populated by hit_test_hover each frame.
    pub hover_label: Option<(egui::Pos2, String)>,
    /// Outliner search filter (case-insensitive substring).
    pub outliner_filter: String,
    /// Active frequency band for acoustic heatmap rendering. `Broadband`
    /// averages all 6 octave bands; specific bands select one of [125, 250,
    /// 500, 1k, 2k, 4k] Hz.
    pub current_band: crate::renderer::FrequencyBand,
    /// Surface-overlay vs floor-grid heatmap mode.
    pub heatmap_mode: crate::renderer::HeatmapMode,
    /// Toggle for ray-path debug visualization. False = zero perf cost.
    pub show_debug_rays: bool,
    /// Number of ray paths to sample for debug viz when `show_debug_rays = true`.
    pub debug_ray_count: usize,
    /// Tele-op keyboard mode (toggled by Ctrl+T). While active, the viewport
    /// consumes W/A/S/D/Q/E and emits a [`RobotAction`] into
    /// [`Self::teleop_pending`] each frame for the app loop to apply to
    /// `robot/0`. The fly-camera handler is gated off in this mode so the
    /// WASD keys go to the robot, not the camera.
    pub teleop_mode: bool,
    /// Latest action computed by the tele-op key handler. Set every frame
    /// when [`Self::teleop_mode`] is on (even when nothing is pressed — the
    /// next frame must zero stale commands). Drained by the main loop.
    pub teleop_pending: Option<crate::robot::state::RobotAction>,
    /// Undo/redo history for user-driven scene edits. Add/remove/move
    /// operations should funnel through `vp.history.push(cmd, scene)` rather
    /// than mutating `scene` directly. Programmatic mutations (sim ticks,
    /// agent server, tele-op) bypass history by design.
    pub history: History,
    /// Transient one-shot message from the last undo/redo action — drained
    /// by the status bar each frame so the user sees "Undid: Move source".
    pub last_history_msg: Option<String>,
    /// Fuzzy-search command palette state. Cmd/Ctrl+K toggles `.open`.
    pub palette: CommandPalette,
    /// Palette action picked this frame that the main loop must handle
    /// because viewport_3d doesn't own the affected state (e.g.
    /// `show_settings`, sim run). Set by viewport, drained by main update().
    pub pending_palette_action: Option<PaletteAction>,
    /// Modal transform gizmo state — G/R/S activates, X/Y/Z constrains,
    /// Enter / LMB confirms, Esc / RMB cancels. `mode == None` ↔ inactive.
    pub gizmo: GizmoState,
    /// Snap configuration. Default grid mode at 0.25 m increments. Hold
    /// Shift during gizmo confirm to apply.
    pub snap: SnapConfig,
    /// Multi-selection set. Primary item (head) mirrors [`Self::selection`]
    /// for legacy single-select code paths. Ctrl/Cmd-click toggles, plain
    /// click resets to single.
    pub selection_set: SelectionSet,
    /// Per-collection hidden indices + isolate flag. Hidden items are
    /// skipped by the viewport renderer; isolate mode shows only items in
    /// [`Self::selection_set`].
    pub hidden_state: HiddenState,
    /// Per-row outliner visibility + lock flags. Distinct from
    /// [`Self::hidden_state`] because outliner-driven hide is row-scoped UI
    /// (eye icon next to each row) and lock blocks accidental click-select.
    pub outliner_rows: OutlinerRows,
    /// Toggle for the per-row outliner eye/lock icons. On by default; off
    /// hides the icons for a leaner panel.
    pub show_visibility_icons: bool,
    /// Anchor for Shift-click range selection. Updated on plain/shift clicks,
    /// preserved on Ctrl/Cmd-click.
    pub selection_anchor: Selection,
    /// True while the next viewport drag should draw a rubber-band rectangle
    /// and select every pickable item inside on release (B-keyed box select).
    pub box_select_armed: bool,
    /// Active rubber-band rectangle while a box-select drag is in progress.
    pub box_select_rect: Option<(egui::Pos2, egui::Pos2)>,
    /// Quad-view state — Ctrl+Alt+Q toggles a Top/Front/Side/Persp split.
    /// Off by default; additive feature.
    pub quad_view: QuadView,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            mode: InteractionMode::Select,
            selection: Selection::None,
            show_grid: true,
            show_rays: true,
            dragging: false,
            hover_world: None,
            material_lib: MaterialLibrary::with_defaults(),
            medium_lib: MediumLibrary::with_defaults(),
            show_fluid: false,
            fluid_viz_mode: FluidVisualizationMode::default(),
            fluid_slice_y: 0,
            show_gas: false,
            gas_viz_mode: GasVisualizationMode::default(),
            gas_slice_y: 0,
            gas_species_idx: 0,
            selected_robot: 0,
            show_robots: true,
            show_sensor_rays: true,
            hit_flash_timer: 0.0,
            hit_flash_robot: None,
            camera_auto_track: false,
            show_boxing_hud: true,
            boxing_messages: Vec::new(),
            fly_mode: false,
            fly_speed: 4.0,
            current_view: CameraView::Perspective,
            shaded: true,
            show_meshes: true,
            show_sources: true,
            show_listeners: true,
            hover_label: None,
            outliner_filter: String::new(),
            current_band: crate::renderer::FrequencyBand::default(),
            heatmap_mode: crate::renderer::HeatmapMode::default(),
            show_debug_rays: false,
            debug_ray_count: crate::renderer::DEFAULT_DEBUG_RAY_COUNT,
            teleop_mode: false,
            teleop_pending: None,
            history: History::default(),
            last_history_msg: None,
            palette: CommandPalette::default(),
            pending_palette_action: None,
            gizmo: GizmoState::default(),
            snap: SnapConfig::default(),
            selection_set: SelectionSet::default(),
            hidden_state: HiddenState::default(),
            outliner_rows: OutlinerRows::default(),
            show_visibility_icons: true,
            selection_anchor: Selection::None,
            box_select_armed: false,
            box_select_rect: None,
            quad_view: QuadView::default(),
        }
    }
}

/// Pressed-state snapshot of the W/A/S/D/Q/E tele-op keys.
///
/// Bit order mirrors the field order so a `[bool; 6]` from a test mirrors
/// what `viewport_3d` reads from `egui::InputState::key_down`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TeleopKeys {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub q: bool,
    pub e: bool,
}

/// Pure helper that maps the tele-op key state to a [`RobotAction`].
///
/// Mapping (joints 0..3 of `robot/0`; joints 3..6 reserved at zero so the
/// action is still legal for 6-DoF robots and forward-compatible for richer
/// keymaps):
/// * `W` / `S` → joint 0 motor velocity `+1.0` / `-1.0`
/// * `A` / `D` → joint 1 motor velocity `-1.0` / `+1.0`
/// * `Q` / `E` → joint 2 motor velocity `-1.0` / `+1.0`
///
/// `num_motors` clamps the action vector length, matching the robot's
/// `ActionSpace::num_motors`. Released keys produce `0.0` so the next frame
/// zeros stale commands without extra bookkeeping.
pub fn compute_teleop_action(
    keys: TeleopKeys,
    num_motors: usize,
) -> crate::robot::state::RobotAction {
    let mut velocities = vec![0.0_f32; num_motors];
    let mut set = |joint: usize, v: f32| {
        if joint < velocities.len() {
            velocities[joint] = v;
        }
    };
    let axis = |pos: bool, neg: bool| -> f32 {
        match (pos, neg) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        }
    };
    set(0, axis(keys.w, keys.s));
    set(1, axis(keys.d, keys.a));
    set(2, axis(keys.e, keys.q));
    crate::robot::state::RobotAction {
        motor_velocities: velocities,
        gripper_commands: Vec::new(),
        base_velocity: [0.0, 0.0],
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod teleop_tests {
    use super::*;

    #[test]
    fn teleop_keymap_zero_when_no_keys() {
        let action = compute_teleop_action(TeleopKeys::default(), 3);
        assert_eq!(action.motor_velocities, vec![0.0, 0.0, 0.0]);
        assert!(action.gripper_commands.is_empty());
        assert_eq!(action.base_velocity, [0.0, 0.0]);
    }

    #[test]
    fn teleop_keymap_w_drives_joint_0_positive() {
        let mut keys = TeleopKeys::default();
        keys.w = true;
        let action = compute_teleop_action(keys, 3);
        assert_eq!(action.motor_velocities, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn teleop_keymap_s_drives_joint_0_negative() {
        let mut keys = TeleopKeys::default();
        keys.s = true;
        let action = compute_teleop_action(keys, 3);
        assert_eq!(action.motor_velocities, vec![-1.0, 0.0, 0.0]);
    }

    #[test]
    fn teleop_keymap_a_d_axis_joint_1() {
        let mut keys = TeleopKeys::default();
        keys.a = true;
        let neg = compute_teleop_action(keys, 3);
        assert_eq!(neg.motor_velocities[1], -1.0);

        let mut keys = TeleopKeys::default();
        keys.d = true;
        let pos = compute_teleop_action(keys, 3);
        assert_eq!(pos.motor_velocities[1], 1.0);
    }

    #[test]
    fn teleop_keymap_q_e_axis_joint_2() {
        let mut keys = TeleopKeys::default();
        keys.q = true;
        let neg = compute_teleop_action(keys, 3);
        assert_eq!(neg.motor_velocities[2], -1.0);

        let mut keys = TeleopKeys::default();
        keys.e = true;
        let pos = compute_teleop_action(keys, 3);
        assert_eq!(pos.motor_velocities[2], 1.0);
    }

    #[test]
    fn teleop_keymap_opposite_keys_cancel() {
        let mut keys = TeleopKeys::default();
        keys.w = true;
        keys.s = true;
        let action = compute_teleop_action(keys, 3);
        assert_eq!(action.motor_velocities[0], 0.0);
    }

    #[test]
    fn teleop_keymap_clamps_to_num_motors() {
        // Smaller robot: only 1 motor exposed → vector length matches.
        let mut keys = TeleopKeys::default();
        keys.w = true;
        let action = compute_teleop_action(keys, 1);
        assert_eq!(action.motor_velocities.len(), 1);
        assert_eq!(action.motor_velocities[0], 1.0);

        // No motors → empty vector, no panic.
        let action = compute_teleop_action(keys, 0);
        assert!(action.motor_velocities.is_empty());
    }

    #[test]
    fn teleop_keymap_pads_to_six_for_richer_robots() {
        let mut keys = TeleopKeys::default();
        keys.e = true;
        let action = compute_teleop_action(keys, 6);
        assert_eq!(action.motor_velocities.len(), 6);
        assert_eq!(action.motor_velocities[2], 1.0);
        for i in [0, 1, 3, 4, 5] {
            assert_eq!(
                action.motor_velocities[i], 0.0,
                "joint {} should be zero",
                i
            );
        }
    }

    #[test]
    fn teleop_mode_field_defaults_off() {
        let vp = ViewportState::default();
        assert!(!vp.teleop_mode);
        assert!(vp.teleop_pending.is_none());
    }
}

#[allow(clippy::too_many_arguments)]
pub fn menu_bar(
    ctx: &egui::Context,
    show_settings: &mut bool,
    show_about: &mut bool,
    status: &mut AppStatus,
    sim: &mut SimulationState,
    show_agent_inspector: &mut bool,
    scene: &mut Scene,
    vp: &mut ViewportState,
) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui
                    .button("New Scene")
                    .on_hover_text("Clear all objects, sources, and listeners")
                    .clicked()
                {
                    *scene = scene_io::new_scene();
                    vp.selection = Selection::None;
                    status.info("New scene");
                    ui.close_menu();
                }
                if ui
                    .button("Open STEP File...")
                    .on_hover_text("Import a .step / .stp CAD model as scene geometry")
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("STEP", &["step", "stp", "STEP", "STP"])
                        .pick_file()
                    {
                        match crate::io::load_step_file(&path) {
                            Ok(load) => {
                                let count = load.objects.len();
                                scene.meshes.extend(load.objects);
                                focus_on_scene(&mut vp.camera, scene);
                                let warn_note = if load.warnings.is_empty() {
                                    String::new()
                                } else {
                                    format!(" ({} warnings)", load.warnings.len())
                                };
                                status.info(format!(
                                    "Loaded STEP: {} ({count} objects){warn_note}",
                                    path.display()
                                ));
                            }
                            Err(e) => {
                                status.error(format!("Failed to load STEP: {e}"));
                            }
                        }
                    }
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button("Save Scene...")
                    .on_hover_text("Save sources, listeners, geometry, and sim config to JSON")
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("EchoMap Scene", &["json"])
                        .set_file_name("scene.json")
                        .save_file()
                    {
                        match scene_io::save_scene_to_string(scene, &sim.config) {
                            Ok(data) => match std::fs::write(&path, data) {
                                Ok(_) => status.info(format!("Saved {}", path.display())),
                                Err(e) => status.error(format!("Save failed: {e}")),
                            },
                            Err(e) => status.error(e),
                        }
                    }
                    ui.close_menu();
                }
                if ui
                    .button("Load Scene...")
                    .on_hover_text("Restore a previously-saved EchoMap scene")
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("EchoMap Scene", &["json"])
                        .pick_file()
                    {
                        match std::fs::read_to_string(&path) {
                            Ok(data) => {
                                match scene_io::load_scene_from_string(&data, &vp.medium_lib) {
                                    Ok((loaded, cfg)) => {
                                        *scene = loaded;
                                        sim.config = cfg;
                                        vp.selection = Selection::None;
                                        focus_on_scene(&mut vp.camera, scene);
                                        status.info(format!("Loaded {}", path.display()));
                                    }
                                    Err(e) => status.error(e),
                                }
                            }
                            Err(e) => status.error(format!("Read failed: {e}")),
                        }
                    }
                    ui.close_menu();
                }
                ui.separator();
                let has_results = sim.result().is_some();
                let resp = ui
                    .add_enabled(has_results, egui::Button::new("Export Results..."))
                    .on_hover_text(if has_results {
                        "Write listener readings (CSV) and a text report"
                    } else {
                        "Run a simulation first to enable export"
                    });
                if resp.clicked() {
                    if let Some(result) = sim.result() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .set_file_name("results.csv")
                            .save_file()
                        {
                            let csv = scene_io::export_results_csv(result, &scene.listeners);
                            let report =
                                scene_io::export_results_report(result, scene, &sim.config);
                            let report_path = path.with_extension("report.md");
                            let csv_ok = std::fs::write(&path, csv);
                            let rep_ok = std::fs::write(&report_path, report);
                            match (csv_ok, rep_ok) {
                                (Ok(_), Ok(_)) => status.info(format!(
                                    "Exported {} + {}",
                                    path.display(),
                                    report_path.display()
                                )),
                                (Err(e), _) | (_, Err(e)) => {
                                    status.error(format!("Export failed: {e}"))
                                }
                            }
                        }
                    }
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("Add", |ui| {
                if ui
                    .button("Box Room (5x4x3m)")
                    .on_hover_text("Add a closed 5×4×3 m rectangular room")
                    .clicked()
                {
                    scene
                        .meshes
                        .push(crate::scene::primitives::box_room(5.0, 4.0, 3.0));
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                if ui
                    .button("L-Room (8x6x3m)")
                    .on_hover_text("Add an L-shaped room composed of two boxes")
                    .clicked()
                {
                    scene
                        .meshes
                        .extend(crate::scene::primitives::l_room(8.0, 6.0, 3.0, 3.0, 3.0));
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                if ui
                    .button("Partition Wall")
                    .on_hover_text("Insert a free-standing 2×2.5 m wall")
                    .clicked()
                {
                    let obj = crate::scene::primitives::partition_wall(
                        Vec3::new(2.0, 0.0, 1.0),
                        2.0,
                        2.5,
                        0.15,
                    );
                    let _ = vp.history.push(
                        SceneCommand::InsertObject {
                            idx: scene.meshes.len(),
                            obj,
                        },
                        scene,
                    );
                    ui.close_menu();
                }
                if ui
                    .button("Platform / Stage")
                    .on_hover_text("Add a 2×2×0.5 m raised platform")
                    .clicked()
                {
                    let obj =
                        crate::scene::primitives::platform(Vec3::new(1.0, 0.0, 1.0), 2.0, 2.0, 0.5);
                    let _ = vp.history.push(
                        SceneCommand::InsertObject {
                            idx: scene.meshes.len(),
                            obj,
                        },
                        scene,
                    );
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .button("Sound Source")
                    .on_hover_text("Add a new omnidirectional source at the origin")
                    .clicked()
                {
                    let _ = vp.history.push(
                        SceneCommand::InsertSource {
                            idx: scene.sound_sources.len(),
                            src: SoundSource::default(),
                        },
                        scene,
                    );
                    vp.selection = Selection::Source(scene.sound_sources.len() - 1);
                    ui.close_menu();
                }
                if ui
                    .button("Listener")
                    .on_hover_text("Add a new listener probe at the origin")
                    .clicked()
                {
                    let n = scene.listeners.len() + 1;
                    let listener = Listener {
                        name: format!("Listener {n}"),
                        ..Default::default()
                    };
                    let _ = vp.history.push(
                        SceneCommand::InsertListener {
                            idx: scene.listeners.len(),
                            listener,
                        },
                        scene,
                    );
                    vp.selection = Selection::Listener(scene.listeners.len() - 1);
                    ui.close_menu();
                }
            });

            ui.menu_button("View", |ui| {
                ui.checkbox(&mut vp.show_grid, "Show Grid")
                    .on_hover_text("Toggle the floor grid overlay");
                ui.checkbox(&mut vp.show_rays, "Show Ray Paths")
                    .on_hover_text("Render simulated ray paths in the viewport");
                ui.checkbox(&mut vp.show_robots, "Show Robots")
                    .on_hover_text("Show or hide robot bodies in the scene");
                ui.checkbox(&mut vp.show_sensor_rays, "Show Sensor Rays")
                    .on_hover_text("Visualize each robot's sensor ray casts");
                ui.checkbox(&mut vp.shaded, "Shaded Surfaces")
                    .on_hover_text("Lambert-shade surfaces with a drop shadow");
                ui.separator();
                ui.menu_button("Camera Presets", |ui| {
                    let presets = [
                        ("Perspective (0)", CameraView::Perspective),
                        ("Top (7)", CameraView::Top),
                        ("Front (1)", CameraView::Front),
                        ("Side (3)", CameraView::Side),
                        ("Isometric (5)", CameraView::Isometric),
                        ("Ringside A ([)", CameraView::RingsideA),
                        ("Ringside B (])", CameraView::RingsideB),
                    ];
                    for (label, view) in presets {
                        if ui.button(label).clicked() {
                            vp.camera.set_view(view);
                            vp.current_view = view;
                            ui.close_menu();
                        }
                    }
                });
                ui.checkbox(&mut vp.fly_mode, "Fly Mode (Tab)")
                    .on_hover_text("Toggle WASD + right-drag fly camera");
                ui.add(
                    egui::Slider::new(&mut vp.fly_speed, 0.5..=20.0)
                        .text("Fly Speed")
                        .clamping(egui::SliderClamping::Always),
                );
                ui.separator();
                if ui
                    .button("Reset Camera")
                    .on_hover_text("Restore the camera to its default orbit (also: R)")
                    .clicked()
                {
                    vp.camera = Camera::default();
                    if !scene.meshes.is_empty() {
                        focus_on_scene(&mut vp.camera, scene);
                    }
                    ui.close_menu();
                }
                if ui
                    .button("Focus on Scene (F)")
                    .on_hover_text("Frame the camera on all scene geometry")
                    .clicked()
                {
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                ui.separator();
                if ui
                    .checkbox(show_agent_inspector, "Agent Inspector")
                    .on_hover_text("Live observation/action message inspector")
                    .clicked()
                {
                    ui.close_menu();
                }
                if ui
                    .button("Settings...")
                    .on_hover_text("Open the Simulation Settings window")
                    .clicked()
                {
                    *show_settings = true;
                    ui.close_menu();
                }
            });

            ui.menu_button("Help", |ui| {
                if ui
                    .button("About EchoMap")
                    .on_hover_text("Version and credits")
                    .clicked()
                {
                    *show_about = true;
                    ui.close_menu();
                }
            });
        });
    });
}

/// Last status / warning / error displayed in the bottom status bar.
/// Owned by `main.rs` and threaded into `menu_bar`, `side_panel`, and `status_bar`.
#[derive(Default, Clone, Debug)]
pub struct AppStatus {
    /// Active message; cleared when the user starts a new action that succeeds.
    pub message: String,
    /// Severity drives the colour shown in the status bar.
    pub severity: StatusSeverity,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusSeverity {
    #[default]
    Info,
    Warn,
    Error,
}

impl AppStatus {
    pub fn info(&mut self, msg: impl Into<String>) {
        self.message = msg.into();
        self.severity = StatusSeverity::Info;
    }
    pub fn warn(&mut self, msg: impl Into<String>) {
        self.message = msg.into();
        self.severity = StatusSeverity::Warn;
    }
    pub fn error(&mut self, msg: impl Into<String>) {
        self.message = msg.into();
        self.severity = StatusSeverity::Error;
    }
    pub fn color(&self) -> egui::Color32 {
        match self.severity {
            StatusSeverity::Info => egui::Color32::from_rgb(180, 220, 180),
            StatusSeverity::Warn => egui::Color32::from_rgb(255, 200, 80),
            StatusSeverity::Error => egui::Color32::from_rgb(255, 100, 100),
        }
    }
}

pub fn toolbar(ctx: &egui::Context, vp: &mut ViewportState) {
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.label("Mode:");
            ui.selectable_value(&mut vp.mode, InteractionMode::Select, "Select (1)");
            ui.selectable_value(
                &mut vp.mode,
                InteractionMode::PlaceSource,
                "Place Source (2)",
            );
            ui.selectable_value(
                &mut vp.mode,
                InteractionMode::PlaceListener,
                "Place Listener (3)",
            );

            ui.separator();
            ui.label("View:");
            let presets = [
                ("Persp", CameraView::Perspective),
                ("Top", CameraView::Top),
                ("Front", CameraView::Front),
                ("Side", CameraView::Side),
                ("Iso", CameraView::Isometric),
                ("Ring-A", CameraView::RingsideA),
                ("Ring-B", CameraView::RingsideB),
            ];
            for (label, view) in presets {
                let selected = vp.current_view == view;
                if ui.selectable_label(selected, label).clicked() {
                    vp.camera.set_view(view);
                    vp.current_view = view;
                }
            }

            ui.separator();
            if ui
                .selectable_label(vp.fly_mode, "Fly (Tab)")
                .on_hover_text("WASD move, QE up/down, Right-drag look")
                .clicked()
            {
                vp.fly_mode = !vp.fly_mode;
            }

            ui.separator();
            let hint = if vp.fly_mode {
                "Fly: WASD=Move  QE=Up/Down  Shift=Sprint  Right-drag=Look  Tab=Exit"
            } else {
                "Orbit: Alt/MMB drag   Pan: Right-drag   Zoom: Scroll   Focus: F   Frame: Home"
            };
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(hint);
            });
        });
    });
}

/// Inspector for whatever is currently selected: shows world position, joint
/// values for robot links, HP/stamina for combat, position editors for sources.
fn render_inspector(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    vp: &mut ViewportState,
    robot_manager: &RobotManager,
) {
    egui::CollapsingHeader::new(
        egui::RichText::new(format!(
            "Inspector — {}",
            vp.selection.breadcrumb(scene, robot_manager)
        ))
        .strong(),
    )
    .id_salt("inspector_section")
    .default_open(true)
    .show(ui, |ui| match vp.selection {
        Selection::None => {
            ui.label("Nothing selected. Click an object or use the outliner.");
        }
        Selection::Source(i) if i < scene.sound_sources.len() => {
            let src = &mut scene.sound_sources[i];
            ui.label(format!("Source {}", i + 1));
            ui.horizontal(|ui| {
                ui.label("X");
                ui.add(egui::DragValue::new(&mut src.position.x).speed(0.05));
                ui.label("Y");
                ui.add(egui::DragValue::new(&mut src.position.y).speed(0.05));
                ui.label("Z");
                ui.add(egui::DragValue::new(&mut src.position.z).speed(0.05));
            });
            ui.checkbox(&mut src.enabled, "Enabled");
            if ui.button("📍 Focus camera").clicked() {
                vp.camera.smooth_focus(src.position, 1.5);
            }
        }
        Selection::Listener(i) if i < scene.listeners.len() => {
            let lis = &mut scene.listeners[i];
            ui.label(&lis.name);
            ui.horizontal(|ui| {
                ui.label("X");
                ui.add(egui::DragValue::new(&mut lis.position.x).speed(0.05));
                ui.label("Y");
                ui.add(egui::DragValue::new(&mut lis.position.y).speed(0.05));
                ui.label("Z");
                ui.add(egui::DragValue::new(&mut lis.position.z).speed(0.05));
            });
            if ui.button("📍 Focus camera").clicked() {
                vp.camera.smooth_focus(lis.position, 1.5);
            }
        }
        Selection::Object(i) if i < scene.meshes.len() => {
            let obj = &scene.meshes[i];
            ui.label(&obj.name);
            ui.label(format!("Material: {}", obj.material.name));
            let tris = obj.mesh.triangles.len();
            ui.label(format!("Triangles: {tris}"));
            let (mn, mx) = obj.mesh.bounds();
            ui.label(format!(
                "AABB: ({:.2},{:.2},{:.2}) → ({:.2},{:.2},{:.2})",
                mn.x, mn.y, mn.z, mx.x, mx.y, mx.z
            ));
        }
        Selection::Robot(ri) => {
            if let Some(robot) = robot_manager.robots.get(ri) {
                inspector_robot_summary(ui, ri, robot, &mut vp.camera);
            }
        }
        Selection::RobotLink(ri, li) => {
            if let Some(robot) = robot_manager.robots.get(ri) {
                inspector_robot_link(ui, ri, li, robot, &mut vp.camera);
            }
        }
        _ => {
            ui.label("Selection out of range.");
        }
    });
}

fn inspector_robot_summary(
    ui: &mut egui::Ui,
    ri: usize,
    robot: &crate::robot::ManagedRobot,
    cam: &mut Camera,
) {
    ui.colored_label(
        robot_color(ri),
        format!("Robot {} — {}", ri, robot.definition.name),
    );
    let poses = robot.state.link_poses_as_mat4();
    if let Some(p) = poses.first() {
        ui.label(format!(
            "Root: ({:.2}, {:.2}, {:.2})",
            p.w_axis.x, p.w_axis.y, p.w_axis.z
        ));
    }
    ui.label(format!("Links: {}", robot.definition.links.len()));
    ui.label(format!("Joints: {}", robot.definition.joints.len()));
    if let Some(ref combat) = robot.state.combat {
        ui.horizontal(|ui| {
            ui.label("HP");
            let frac = (combat.health / combat.max_health).clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("{:.0}/{:.0}", combat.health, combat.max_health))
                    .desired_width(140.0),
            );
        });
        ui.horizontal(|ui| {
            ui.label("Stam");
            let frac = (combat.stamina / combat.max_stamina).clamp(0.0, 1.0);
            ui.add(
                egui::ProgressBar::new(frac)
                    .text(format!("{:.0}/{:.0}", combat.stamina, combat.max_stamina))
                    .desired_width(140.0),
            );
        });
    }
    if ui.button("📍 Focus camera").clicked() {
        if let Some(p) = poses.first() {
            cam.smooth_focus(Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z), 2.0);
        }
    }
}

fn inspector_robot_link(
    ui: &mut egui::Ui,
    ri: usize,
    li: usize,
    robot: &crate::robot::ManagedRobot,
    cam: &mut Camera,
) {
    let link_def = match robot.definition.links.get(li) {
        Some(l) => l,
        None => {
            ui.label("Link out of range.");
            return;
        }
    };
    let poses = robot.state.link_poses_as_mat4();
    let pose = poses.get(li);
    ui.colored_label(robot_color(ri), format!("R{} · {}", ri, link_def.name));
    if let Some(p) = pose {
        ui.label(format!(
            "World pos: ({:.3}, {:.3}, {:.3})",
            p.w_axis.x, p.w_axis.y, p.w_axis.z
        ));
    }
    let shape_str = match &link_def.collision_shape {
        crate::robot::definition::CollisionShape::Cuboid { half_extents } => format!(
            "Cuboid {:.2}×{:.2}×{:.2}",
            half_extents.x * 2.0,
            half_extents.y * 2.0,
            half_extents.z * 2.0
        ),
        crate::robot::definition::CollisionShape::Cylinder { radius, height } => {
            format!("Cylinder r={:.2} h={:.2}", radius, height)
        }
        crate::robot::definition::CollisionShape::Sphere { radius } => {
            format!("Sphere r={:.2}", radius)
        }
    };
    ui.label(format!("Shape: {}", shape_str));

    // Show parent joint info if any.
    let parent_joint = robot
        .definition
        .joints
        .iter()
        .enumerate()
        .find(|(_, j)| j.child_link == li);
    if let Some((ji, joint_def)) = parent_joint {
        ui.separator();
        ui.label(format!("Joint #{}: {}", ji, joint_def.name));
        if let Some(&q) = robot.state.joint_positions.get(ji) {
            ui.label(format!("  position: {:.3} rad ({:.1}°)", q, q.to_degrees()));
        }
        if let Some(&qd) = robot.state.joint_velocities.get(ji) {
            ui.label(format!("  velocity: {:.3} rad/s", qd));
        }
    }
    if ui.button("📍 Focus camera").clicked() {
        if let Some(p) = pose {
            cam.smooth_focus(Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z), 1.0);
        }
    }
}

/// Scene outliner — tree view of everything pickable in the scene, with search,
/// visibility toggles, and click-to-select that smooth-focuses the camera.
pub fn outliner_panel(
    ctx: &egui::Context,
    scene: &mut Scene,
    vp: &mut ViewportState,
    robot_manager: &RobotManager,
) {
    egui::SidePanel::left("outliner_panel")
        .default_width(240.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Outliner");
            ui.horizontal(|ui| {
                ui.label("Find:");
                ui.add(
                    egui::TextEdit::singleline(&mut vp.outliner_filter)
                        .hint_text("name…")
                        .desired_width(ui.available_width() - 30.0),
                );
                if ui.small_button("×").clicked() {
                    vp.outliner_filter.clear();
                }
            });
            ui.separator();

            // --- Inspector for current selection ---
            render_inspector(ui, scene, vp, robot_manager);
            ui.separator();

            let filter = vp.outliner_filter.to_lowercase();
            let matches = |name: &str| -> bool {
                filter.is_empty() || name.to_lowercase().contains(filter.as_str())
            };

            egui::ScrollArea::vertical().show(ui, |ui| {
                // --- Robots ---
                ui.horizontal(|ui| {
                    let eye = if vp.show_robots { "●" } else { "○" };
                    if ui
                        .small_button(eye)
                        .on_hover_text("Toggle robot visibility")
                        .clicked()
                    {
                        vp.show_robots = !vp.show_robots;
                    }
                    ui.label(
                        egui::RichText::new(format!("Robots ({})", robot_manager.robots.len()))
                            .strong(),
                    );
                });
                for (ri, robot) in robot_manager.robots.iter().enumerate() {
                    let robot_label = format!("R{}  {}", ri, robot.definition.name);
                    let visible = matches(&robot_label)
                        || robot.definition.links.iter().any(|l| matches(&l.name));
                    if !visible {
                        continue;
                    }
                    let robot_selected = vp.selection == Selection::Robot(ri)
                        || matches!(vp.selection, Selection::RobotLink(r, _) if r == ri);
                    egui::CollapsingHeader::new(
                        egui::RichText::new(&robot_label).color(robot_color(ri)),
                    )
                    .id_salt(format!("outliner_robot_{}", ri))
                    .default_open(robot_selected)
                    .show(ui, |ui| {
                        if ui
                            .selectable_label(vp.selection == Selection::Robot(ri), "📍 Focus body")
                            .clicked()
                        {
                            vp.selection = Selection::Robot(ri);
                            if let Some(p) = robot.state.link_poses_as_mat4().first() {
                                vp.camera.smooth_focus(
                                    Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z),
                                    2.0,
                                );
                            }
                        }
                        let poses = robot.state.link_poses_as_mat4();
                        for (li, link_def) in robot.definition.links.iter().enumerate() {
                            if !matches(&link_def.name) && !matches(&robot_label) {
                                continue;
                            }
                            let sel = vp.selection == Selection::RobotLink(ri, li);
                            let label = format!("  • {}", link_def.name);
                            if ui.selectable_label(sel, label).clicked() {
                                vp.selection = Selection::RobotLink(ri, li);
                                if let Some(p) = poses.get(li) {
                                    vp.camera.smooth_focus(
                                        Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z),
                                        1.0,
                                    );
                                }
                            }
                        }
                    });
                }
                ui.separator();

                // --- Sources ---
                ui.horizontal(|ui| {
                    let eye = if vp.show_sources { "●" } else { "○" };
                    if ui.small_button(eye).clicked() {
                        vp.show_sources = !vp.show_sources;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Sound Sources ({})",
                            scene.sound_sources.len()
                        ))
                        .strong(),
                    );
                });
                let mut to_select: Option<Selection> = None;
                let mut toggle_src_vis: Option<usize> = None;
                let mut toggle_src_lock: Option<usize> = None;
                for (i, _) in scene.sound_sources.iter().enumerate() {
                    let label = format!("Source {}", i + 1);
                    if !matches(&label) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        if vp.show_visibility_icons {
                            let eye = if vp.outliner_rows.is_source_visible(i) {
                                outliner::ICON_EYE_OPEN
                            } else {
                                outliner::ICON_EYE_CLOSED
                            };
                            if ui
                                .small_button(eye)
                                .on_hover_text("Toggle row visibility")
                                .clicked()
                            {
                                toggle_src_vis = Some(i);
                            }
                            let lock = if vp.outliner_rows.is_source_locked(i) {
                                outliner::ICON_LOCK_CLOSED
                            } else {
                                outliner::ICON_LOCK_OPEN
                            };
                            if ui
                                .small_button(lock)
                                .on_hover_text("Lock row (blocks select)")
                                .clicked()
                            {
                                toggle_src_lock = Some(i);
                            }
                        }
                        let sel = vp.selection == Selection::Source(i);
                        let locked = vp.outliner_rows.is_source_locked(i);
                        let row_label = if locked {
                            egui::RichText::new(&label).weak()
                        } else {
                            egui::RichText::new(&label)
                        };
                        if ui.selectable_label(sel, row_label).clicked() && !locked {
                            to_select = Some(Selection::Source(i));
                        }
                    });
                }
                if let Some(i) = toggle_src_vis {
                    vp.outliner_rows.toggle_source_visibility(i);
                }
                if let Some(i) = toggle_src_lock {
                    vp.outliner_rows.toggle_source_lock(i);
                }
                if let Some(Selection::Source(i)) = to_select {
                    vp.selection = Selection::Source(i);
                    vp.selection_set.set_single(Selection::Source(i));
                    if let Some(s) = scene.sound_sources.get(i) {
                        vp.camera.smooth_focus(s.position, 1.5);
                    }
                }
                ui.separator();

                // --- Listeners ---
                ui.horizontal(|ui| {
                    let eye = if vp.show_listeners { "●" } else { "○" };
                    if ui.small_button(eye).clicked() {
                        vp.show_listeners = !vp.show_listeners;
                    }
                    ui.label(
                        egui::RichText::new(format!("Listeners ({})", scene.listeners.len()))
                            .strong(),
                    );
                });
                let mut focus_listener: Option<usize> = None;
                let mut toggle_lis_vis: Option<usize> = None;
                let mut toggle_lis_lock: Option<usize> = None;
                for (i, listener) in scene.listeners.iter().enumerate() {
                    if !matches(&listener.name) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        if vp.show_visibility_icons {
                            let eye = if vp.outliner_rows.is_listener_visible(i) {
                                outliner::ICON_EYE_OPEN
                            } else {
                                outliner::ICON_EYE_CLOSED
                            };
                            if ui
                                .small_button(eye)
                                .on_hover_text("Toggle row visibility")
                                .clicked()
                            {
                                toggle_lis_vis = Some(i);
                            }
                            let lock = if vp.outliner_rows.is_listener_locked(i) {
                                outliner::ICON_LOCK_CLOSED
                            } else {
                                outliner::ICON_LOCK_OPEN
                            };
                            if ui
                                .small_button(lock)
                                .on_hover_text("Lock row (blocks select)")
                                .clicked()
                            {
                                toggle_lis_lock = Some(i);
                            }
                        }
                        let sel = vp.selection == Selection::Listener(i);
                        let locked = vp.outliner_rows.is_listener_locked(i);
                        let row_label = if locked {
                            egui::RichText::new(&listener.name).weak()
                        } else {
                            egui::RichText::new(&listener.name)
                        };
                        if ui.selectable_label(sel, row_label).clicked() && !locked {
                            vp.selection = Selection::Listener(i);
                            vp.selection_set.set_single(Selection::Listener(i));
                            focus_listener = Some(i);
                        }
                    });
                }
                if let Some(i) = toggle_lis_vis {
                    vp.outliner_rows.toggle_listener_visibility(i);
                }
                if let Some(i) = toggle_lis_lock {
                    vp.outliner_rows.toggle_listener_lock(i);
                }
                if let Some(i) = focus_listener {
                    if let Some(l) = scene.listeners.get(i) {
                        vp.camera.smooth_focus(l.position, 1.5);
                    }
                }
                ui.separator();

                // --- Objects (meshes) ---
                ui.horizontal(|ui| {
                    let eye = if vp.show_meshes { "●" } else { "○" };
                    if ui.small_button(eye).clicked() {
                        vp.show_meshes = !vp.show_meshes;
                    }
                    ui.label(
                        egui::RichText::new(format!("Objects ({})", scene.meshes.len())).strong(),
                    );
                });
                let mut focus_obj: Option<usize> = None;
                let mut toggle_obj_lock: Option<usize> = None;
                for (i, obj) in scene.meshes.iter_mut().enumerate() {
                    if !matches(&obj.name) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        if vp.show_visibility_icons {
                            let eye = if obj.visible {
                                outliner::ICON_EYE_OPEN
                            } else {
                                outliner::ICON_EYE_CLOSED
                            };
                            if ui
                                .small_button(eye)
                                .on_hover_text("Toggle row visibility")
                                .clicked()
                            {
                                obj.visible = !obj.visible;
                            }
                            let lock = if vp.outliner_rows.is_object_locked(i) {
                                outliner::ICON_LOCK_CLOSED
                            } else {
                                outliner::ICON_LOCK_OPEN
                            };
                            if ui
                                .small_button(lock)
                                .on_hover_text("Lock row (blocks select)")
                                .clicked()
                            {
                                toggle_obj_lock = Some(i);
                            }
                        }
                        let sel = vp.selection == Selection::Object(i);
                        let locked = vp.outliner_rows.is_object_locked(i);
                        let row_label = if locked {
                            egui::RichText::new(&obj.name).weak()
                        } else {
                            egui::RichText::new(&obj.name)
                        };
                        if ui.selectable_label(sel, row_label).clicked() && !locked {
                            vp.selection = Selection::Object(i);
                            vp.selection_set.set_single(Selection::Object(i));
                            focus_obj = Some(i);
                        }
                    });
                }
                if let Some(i) = toggle_obj_lock {
                    vp.outliner_rows.toggle_object_lock(i);
                }
                if let Some(i) = focus_obj {
                    if let Some(obj) = scene.meshes.get(i) {
                        let (mn, mx) = obj.mesh.bounds();
                        let c = (mn + mx) * 0.5;
                        let r = (mx - mn).length() * 0.5;
                        vp.camera.smooth_focus(c, r.max(0.5));
                    }
                }
            });
        });
}

#[allow(clippy::too_many_arguments)]
pub fn side_panel(
    ctx: &egui::Context,
    scene: &mut Scene,
    sim: &mut SimulationState,
    vp: &mut ViewportState,
    fluid_sim: &mut FluidSimulation,
    gas_sim: &mut GasSimulation,
    robot_manager: &mut RobotManager,
    agent_config: &mut AgentServerConfig,
    agent_handle: &mut Option<AgentServerHandle>,
    bridge_client: &mut SimBridgeClient,
    bridge_server: &mut Option<SimBridgeServer>,
) {
    egui::SidePanel::left("side_panel")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("Properties");
            ui.separator();

            // --- Background Medium ---
            egui::CollapsingHeader::new("Background Medium")
                .default_open(false)
                .show(ui, |ui| {
                    let medium_names: Vec<String> = vp.medium_lib.media.keys().cloned().collect();
                    egui::ComboBox::from_id_salt("bg_medium_combo")
                        .selected_text(&scene.background_medium.name)
                        .show_ui(ui, |ui| {
                            for name in &medium_names {
                                if ui
                                    .selectable_label(scene.background_medium.name == *name, name)
                                    .clicked()
                                {
                                    if let Some(m) = vp.medium_lib.get(name) {
                                        scene.background_medium = m.clone();
                                    }
                                }
                            }
                        });
                    ui.label(format!(
                        "  Density: {:.3} kg/m\u{b3}",
                        scene.background_medium.density
                    ));
                    ui.label(format!(
                        "  Speed of sound: {:.1} m/s",
                        scene.background_medium.speed_of_sound
                    ));
                    ui.label(format!(
                        "  Impedance: {:.1} Pa\u{b7}s/m",
                        scene.background_medium.impedance
                    ));
                });

            ui.separator();

            egui::CollapsingHeader::new(format!("Scene Objects ({})", scene.meshes.len()))
                .default_open(true)
                .show(ui, |ui| {
                    if scene.meshes.is_empty() {
                        ui.label("No objects. Use Add menu or File > Open STEP.");
                    }
                    let mut to_remove = None;
                    for (i, obj) in scene.meshes.iter_mut().enumerate() {
                        let selected = vp.selection == Selection::Object(i);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut obj.visible, "");
                            if ui.selectable_label(selected, &obj.name).clicked() {
                                vp.selection = Selection::Object(i);
                            }
                        });
                    }
                    if let Selection::Object(i) = vp.selection {
                        if i < scene.meshes.len() {
                            ui.separator();
                            ui.label("Material:");
                            let mat_names: Vec<String> =
                                vp.material_lib.materials.keys().cloned().collect();
                            egui::ComboBox::from_id_salt("mat_combo")
                                .selected_text(&scene.meshes[i].material.name)
                                .show_ui(ui, |ui| {
                                    for name in &mat_names {
                                        if ui
                                            .selectable_label(
                                                scene.meshes[i].material.name == *name,
                                                name,
                                            )
                                            .clicked()
                                        {
                                            if let Some(mat) = vp.material_lib.materials.get(name) {
                                                scene.meshes[i].material = mat.clone();
                                            }
                                        }
                                    }
                                });

                            let abs = &scene.meshes[i].material.absorption;
                            ui.label(format!("  Absorption: {:.2} avg", abs.average()));
                            ui.add(
                                egui::Slider::new(
                                    &mut scene.meshes[i].material.scattering,
                                    0.0..=1.0,
                                )
                                .text("Scatter"),
                            );

                            // --- Interior Medium ---
                            ui.separator();
                            ui.label("Interior Medium:");
                            let int_medium_names: Vec<String> =
                                vp.medium_lib.media.keys().cloned().collect();
                            let selected_int_name = scene.meshes[i]
                                .interior_medium
                                .as_ref()
                                .map_or("None".to_string(), |m| m.name.clone());
                            egui::ComboBox::from_id_salt(format!("int_medium_combo_{i}"))
                                .selected_text(&selected_int_name)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(
                                            scene.meshes[i].interior_medium.is_none(),
                                            "None",
                                        )
                                        .clicked()
                                    {
                                        scene.meshes[i].interior_medium = None;
                                    }
                                    for name in &int_medium_names {
                                        let is_selected = scene.meshes[i]
                                            .interior_medium
                                            .as_ref()
                                            .is_some_and(|m| m.name == *name);
                                        if ui.selectable_label(is_selected, name).clicked() {
                                            if let Some(m) = vp.medium_lib.get(name) {
                                                scene.meshes[i].interior_medium = Some(m.clone());
                                            }
                                        }
                                    }
                                });
                            if let Some(ref med) = scene.meshes[i].interior_medium {
                                ui.label(format!("  Density: {:.3} kg/m\u{b3}", med.density));
                                ui.label(format!(
                                    "  Speed of sound: {:.1} m/s",
                                    med.speed_of_sound
                                ));
                                ui.label(format!("  Impedance: {:.1} Pa\u{b7}s/m", med.impedance));
                            }

                            // --- Surface Properties ---
                            ui.separator();
                            egui::CollapsingHeader::new("Surface Properties")
                                .id_salt(format!("surface_props_{i}"))
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.friction_static,
                                            0.0..=2.0,
                                        )
                                        .text("Static Friction"),
                                    );
                                    let max_kinetic = scene.meshes[i].material.friction_static;
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.friction_kinetic,
                                            0.0..=max_kinetic,
                                        )
                                        .text("Kinetic Friction"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.roughness,
                                            0.0..=0.1,
                                        )
                                        .text("Roughness (m)"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.porosity,
                                            0.0..=1.0,
                                        )
                                        .text("Porosity"),
                                    );
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.permeability,
                                            0.0..=1e-10,
                                        )
                                        .text("Permeability (m\u{b2})"),
                                    );
                                    let degrees =
                                        scene.meshes[i].material.contact_angle.to_degrees();
                                    ui.add(
                                        egui::Slider::new(
                                            &mut scene.meshes[i].material.contact_angle,
                                            0.0..=std::f32::consts::PI,
                                        )
                                        .text(format!("Contact Angle ({degrees:.1}\u{b0})")),
                                    );
                                });

                            if ui.button("Delete Object").clicked() {
                                to_remove = Some(i);
                            }
                        }
                    }
                    if let Some(i) = to_remove {
                        if i < scene.meshes.len() {
                            let snap = scene.meshes[i].clone();
                            let _ = vp.history.push(
                                SceneCommand::RemoveObject { idx: i, snapshot: snap },
                                scene,
                            );
                        }
                        vp.selection = Selection::None;
                    }
                });

            ui.separator();

            egui::CollapsingHeader::new(format!("Sound Sources ({})", scene.sound_sources.len()))
                .default_open(true)
                .show(ui, |ui| {
                    let mut to_remove = None;
                    for (i, source) in scene.sound_sources.iter_mut().enumerate() {
                        let selected = vp.selection == Selection::Source(i);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut source.enabled, "");
                            if ui
                                .selectable_label(selected, format!("Source {}", i + 1))
                                .clicked()
                            {
                                vp.selection = Selection::Source(i);
                            }
                        });

                        if selected {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Pos:");
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.x)
                                            .prefix("x:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.y)
                                            .prefix("y:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.z)
                                            .prefix("z:")
                                            .speed(0.1),
                                    );
                                });
                                ui.add(
                                    egui::Slider::new(&mut source.frequency_hz, 20.0..=20000.0)
                                        .text("Hz")
                                        .logarithmic(true),
                                );
                                ui.add(
                                    egui::Slider::new(&mut source.power_db, 40.0..=120.0)
                                        .text("dB"),
                                );
                                if ui.button("Delete Source").clicked() {
                                    to_remove = Some(i);
                                }
                            });
                        }
                    }
                    if let Some(i) = to_remove {
                        if i < scene.sound_sources.len() {
                            let snap = scene.sound_sources[i].clone();
                            let _ = vp.history.push(
                                SceneCommand::RemoveSource { idx: i, snapshot: snap },
                                scene,
                            );
                        }
                        vp.selection = Selection::None;
                    }
                });

            ui.separator();

            egui::CollapsingHeader::new(format!("Listeners ({})", scene.listeners.len()))
                .default_open(true)
                .show(ui, |ui| {
                    let mut to_remove = None;
                    for (i, listener) in scene.listeners.iter_mut().enumerate() {
                        let selected = vp.selection == Selection::Listener(i);
                        if ui.selectable_label(selected, &listener.name).clicked() {
                            vp.selection = Selection::Listener(i);
                        }

                        if selected {
                            ui.group(|ui| {
                                ui.text_edit_singleline(&mut listener.name);
                                ui.horizontal(|ui| {
                                    ui.label("Pos:");
                                    ui.add(
                                        egui::DragValue::new(&mut listener.position.x)
                                            .prefix("x:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut listener.position.y)
                                            .prefix("y:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut listener.position.z)
                                            .prefix("z:")
                                            .speed(0.1),
                                    );
                                });
                                if ui.button("Delete Listener").clicked() {
                                    to_remove = Some(i);
                                }
                            });
                        }
                    }
                    if let Some(i) = to_remove {
                        if i < scene.listeners.len() {
                            let snap = scene.listeners[i].clone();
                            let _ = vp.history.push(
                                SceneCommand::RemoveListener { idx: i, snapshot: snap },
                                scene,
                            );
                        }
                        vp.selection = Selection::None;
                    }
                });

            if sim.is_running() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Simulating… {:.0}%", sim.progress() * 100.0));
                    if ui.button("Cancel").clicked() {
                        sim.cancel();
                    }
                });
            }

            ui.separator();

            // --- Fluid Simulation Controls ---
            egui::CollapsingHeader::new("Fluid Simulation")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let label = if fluid_sim.running { "Stop" } else { "Start" };
                        if ui.button(label).clicked() {
                            fluid_sim.running = !fluid_sim.running;
                        }
                        if ui.button("Step").clicked() {
                            fluid_sim.step();
                        }
                        if ui.button("Reset").clicked() {
                            fluid_sim.reset();
                        }
                    });
                    ui.label(format!("Frame: {}", fluid_sim.frame));
                });

            ui.separator();

            // --- Fluid Visualization ---
            egui::CollapsingHeader::new("Fluid Visualization")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut vp.show_fluid, "Show Fluid");

                    let mode_label = match vp.fluid_viz_mode {
                        FluidVisualizationMode::VelocityMagnitude => "Velocity",
                        FluidVisualizationMode::Pressure => "Pressure",
                        FluidVisualizationMode::Density => "Density",
                        FluidVisualizationMode::LevelSet => "LevelSet",
                    };
                    egui::ComboBox::from_id_salt("fluid_viz_mode")
                        .selected_text(mode_label)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut vp.fluid_viz_mode,
                                FluidVisualizationMode::VelocityMagnitude,
                                "Velocity",
                            );
                            ui.selectable_value(
                                &mut vp.fluid_viz_mode,
                                FluidVisualizationMode::Pressure,
                                "Pressure",
                            );
                            ui.selectable_value(
                                &mut vp.fluid_viz_mode,
                                FluidVisualizationMode::Density,
                                "Density",
                            );
                            ui.selectable_value(
                                &mut vp.fluid_viz_mode,
                                FluidVisualizationMode::LevelSet,
                                "LevelSet",
                            );
                        });

                    let max_y = fluid_sim
                        .grid
                        .as_ref()
                        .map_or(0, |g| g.ny.saturating_sub(1));
                    ui.add(
                        egui::Slider::new(&mut vp.fluid_slice_y, 0..=max_y.max(1)).text("Slice Y"),
                    );
                });

            ui.separator();

            // --- Gas Simulation Controls ---
            egui::CollapsingHeader::new("Gas Simulation")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let label = if gas_sim.running { "Stop" } else { "Start" };
                        if ui.button(label).clicked() {
                            gas_sim.running = !gas_sim.running;
                        }
                        if ui.button("Step").clicked() {
                            gas_sim.step();
                        }
                        if ui.button("Reset").clicked() {
                            gas_sim.reset();
                        }
                    });
                    ui.label(format!("Frame: {}", gas_sim.frame));

                    ui.separator();
                    ui.label("Settings:");

                    ui.add(
                        egui::Slider::new(&mut gas_sim.config.dt, 0.001..=0.1).text("Timestep (s)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut gas_sim.config.ambient_temperature, 200.0..=500.0)
                            .text("Ambient Temp (K)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut gas_sim.config.thermal_diffusivity, 0.0..=0.1)
                            .text("Thermal Diff"),
                    );
                    ui.add(
                        egui::Slider::new(&mut gas_sim.config.buoyancy_coefficient, 0.0..=1.0)
                            .text("Buoyancy Coeff"),
                    );

                    ui.label("Gravity:");
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut gas_sim.config.gravity.x)
                                .prefix("x: ")
                                .speed(0.1),
                        );
                        ui.add(
                            egui::DragValue::new(&mut gas_sim.config.gravity.y)
                                .prefix("y: ")
                                .speed(0.1),
                        );
                        ui.add(
                            egui::DragValue::new(&mut gas_sim.config.gravity.z)
                                .prefix("z: ")
                                .speed(0.1),
                        );
                    });

                    // --- Gas Source Controls ---
                    ui.separator();
                    ui.label(format!("Gas Sources ({}):", gas_sim.sources.len()));
                    if ui.button("Add Source").clicked() {
                        gas_sim.sources.push(crate::gas::boundary::GasSource {
                            position: Vec3::ZERO,
                            species_index: 0,
                            rate: 1.0,
                            radius: 0.5,
                        });
                    }

                    let species_count = gas_sim.grid.as_ref().map_or(0, |g| g.species.len());
                    let mut to_remove_src = None;
                    for (si, source) in gas_sim.sources.iter_mut().enumerate() {
                        egui::CollapsingHeader::new(format!("Source {}", si + 1))
                            .id_salt(format!("gas_source_{si}"))
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Pos:");
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.x)
                                            .prefix("x:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.y)
                                            .prefix("y:")
                                            .speed(0.1),
                                    );
                                    ui.add(
                                        egui::DragValue::new(&mut source.position.z)
                                            .prefix("z:")
                                            .speed(0.1),
                                    );
                                });

                                if species_count > 0 {
                                    let sp_name = gas_sim
                                        .grid
                                        .as_ref()
                                        .and_then(|g| g.species.get(source.species_index))
                                        .map_or("None".to_string(), |s| s.name.clone());
                                    egui::ComboBox::from_id_salt(format!("gas_src_species_{si}"))
                                        .selected_text(sp_name)
                                        .show_ui(ui, |ui| {
                                            if let Some(ref grid) = gas_sim.grid {
                                                for (idx, sp) in grid.species.iter().enumerate() {
                                                    ui.selectable_value(
                                                        &mut source.species_index,
                                                        idx,
                                                        &sp.name,
                                                    );
                                                }
                                            }
                                        });
                                }

                                ui.add(
                                    egui::Slider::new(&mut source.rate, 0.0..=100.0).text("Rate"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut source.radius, 0.01..=5.0)
                                        .text("Radius"),
                                );

                                if ui.button("Delete Source").clicked() {
                                    to_remove_src = Some(si);
                                }
                            });
                    }
                    if let Some(idx) = to_remove_src {
                        gas_sim.sources.remove(idx);
                    }
                });

            ui.separator();

            // --- Gas Visualization ---
            egui::CollapsingHeader::new("Gas Visualization")
                .default_open(false)
                .show(ui, |ui| {
                    ui.checkbox(&mut vp.show_gas, "Show Gas");

                    let gas_mode_label = match vp.gas_viz_mode {
                        GasVisualizationMode::Concentration => "Concentration",
                        GasVisualizationMode::Temperature => "Temperature",
                        GasVisualizationMode::Pressure => "Pressure",
                        GasVisualizationMode::VelocityMagnitude => "Velocity",
                    };
                    egui::ComboBox::from_id_salt("gas_viz_mode")
                        .selected_text(gas_mode_label)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut vp.gas_viz_mode,
                                GasVisualizationMode::Concentration,
                                "Concentration",
                            );
                            ui.selectable_value(
                                &mut vp.gas_viz_mode,
                                GasVisualizationMode::Temperature,
                                "Temperature",
                            );
                            ui.selectable_value(
                                &mut vp.gas_viz_mode,
                                GasVisualizationMode::Pressure,
                                "Pressure",
                            );
                            ui.selectable_value(
                                &mut vp.gas_viz_mode,
                                GasVisualizationMode::VelocityMagnitude,
                                "Velocity",
                            );
                        });

                    // Species selector
                    let species_count = gas_sim.grid.as_ref().map_or(0, |g| g.species.len());
                    if species_count > 0 {
                        let species_name = gas_sim
                            .grid
                            .as_ref()
                            .and_then(|g| g.species.get(vp.gas_species_idx))
                            .map_or("None".to_string(), |s| s.name.clone());
                        egui::ComboBox::from_id_salt("gas_species_sel")
                            .selected_text(species_name)
                            .show_ui(ui, |ui| {
                                if let Some(ref grid) = gas_sim.grid {
                                    for (idx, sp) in grid.species.iter().enumerate() {
                                        ui.selectable_value(&mut vp.gas_species_idx, idx, &sp.name);
                                    }
                                }
                            });
                    }

                    let gas_max_y = gas_sim.grid.as_ref().map_or(0, |g| g.ny.saturating_sub(1));
                    ui.add(
                        egui::Slider::new(&mut vp.gas_slice_y, 0..=gas_max_y.max(1))
                            .text("Slice Y"),
                    );
                });

            ui.separator();

            // --- Robot Control ---
            egui::CollapsingHeader::new(format!("Robot Control ({})", robot_manager.robots.len()))
                .id_salt("robot_control")
                .default_open(false)
                .show(ui, |ui| {
                    // Start/Stop toggle
                    ui.horizontal(|ui| {
                        let label = if robot_manager.running {
                            "Stop"
                        } else {
                            "Start"
                        };
                        if ui.button(label).clicked() {
                            robot_manager.running = !robot_manager.running;
                        }
                    });

                    // Add Simple Arm button
                    if ui.button("Add Simple Arm").clicked() {
                        let def = RobotDefinition::simple_arm(3);
                        robot_manager.add_robot(def, glam::Mat4::IDENTITY);
                    }

                    if robot_manager.robots.is_empty() {
                        ui.label("No robots. Click 'Add Simple Arm' to add one.");
                        return;
                    }

                    // Robot selector (dropdown if multiple robots)
                    let robot_count = robot_manager.robots.len();
                    let selected_idx = vp.selected_robot.min(robot_count.saturating_sub(1));
                    vp.selected_robot = selected_idx;

                    if robot_count > 1 {
                        let selected_name = robot_manager
                            .robots
                            .get(selected_idx)
                            .map_or("None".to_string(), |r| {
                                format!("{} [{}]", r.definition.name, selected_idx)
                            });
                        egui::ComboBox::from_id_salt("robot_selector")
                            .selected_text(selected_name)
                            .show_ui(ui, |ui| {
                                for (i, robot) in robot_manager.robots.iter().enumerate() {
                                    ui.selectable_value(
                                        &mut vp.selected_robot,
                                        i,
                                        format!("{} [{}]", robot.definition.name, i),
                                    );
                                }
                            });
                    }

                    // Display selected robot details
                    if let Some(robot) = robot_manager.robots.get(selected_idx) {
                        ui.label(format!("Name: {}", robot.definition.name));
                        ui.label(format!(
                            "Joints: {} | Links: {} | Sensors: {}",
                            robot.definition.joints.len(),
                            robot.definition.links.len(),
                            robot.definition.sensors.len()
                        ));

                        // Collect joint info for sliders (need to separate borrow)
                        let joint_infos: Vec<(String, f32, f32, f32, f32)> = robot
                            .definition
                            .joints
                            .iter()
                            .enumerate()
                            .map(|(i, jd)| {
                                let pos =
                                    robot.state.joint_positions.get(i).copied().unwrap_or(0.0);
                                let vel =
                                    robot.state.joint_velocities.get(i).copied().unwrap_or(0.0);
                                (jd.name.clone(), jd.limit_min, jd.limit_max, pos, vel)
                            })
                            .collect();

                        // Sensor reading snapshot
                        let sensor_readings: Vec<_> = robot.state.sensor_readings.to_vec();
                        let sensor_defs: Vec<_> = robot.definition.sensors.to_vec();

                        // Joint angles with sliders
                        if !joint_infos.is_empty() {
                            ui.separator();
                            ui.label("Joint Angles:");
                            let mut commands: Vec<(usize, ActuatorCommand)> = Vec::new();
                            for (i, (name, limit_min, limit_max, pos, vel)) in
                                joint_infos.iter().enumerate()
                            {
                                egui::CollapsingHeader::new(name)
                                    .id_salt(format!("joint_{}", i))
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        let mut current_pos = *pos;
                                        if ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut current_pos,
                                                    *limit_min..=*limit_max,
                                                )
                                                .text("Position"),
                                            )
                                            .changed()
                                        {
                                            commands
                                                .push((i, ActuatorCommand::Position(current_pos)));
                                        }
                                        ui.label(format!("Velocity: {:.3} rad/s", vel));
                                    });
                            }
                            // Apply slider commands
                            for (joint_idx, cmd) in commands {
                                robot_manager.set_command(selected_idx, joint_idx, cmd);
                            }
                        }

                        // Sensor readings display
                        if !sensor_readings.is_empty() {
                            ui.separator();
                            ui.label("Sensor Readings:");
                            for (i, reading) in sensor_readings.iter().enumerate() {
                                let sensor_name =
                                    sensor_defs.get(i).map_or(format!("Sensor {}", i), |sd| {
                                        match &sd.sensor {
                                        crate::robot::definition::SensorDefinition::Distance {
                                            ..
                                        } => format!("Distance [{}]", i),
                                        crate::robot::definition::SensorDefinition::Lidar {
                                            ..
                                        } => format!("Lidar [{}]", i),
                                        crate::robot::definition::SensorDefinition::Contact => {
                                            format!("Contact [{}]", i)
                                        }
                                        crate::robot::definition::SensorDefinition::Imu => {
                                            format!("IMU [{}]", i)
                                        }
                                    }
                                    });
                                match reading {
                                    crate::robot::state::SensorReading::Distance(d) => {
                                        ui.label(format!("  {}: {:.3} m", sensor_name, d));
                                    }
                                    crate::robot::state::SensorReading::Lidar(rays) => {
                                        ui.label(format!("  {}: {} rays", sensor_name, rays.len()));
                                    }
                                    crate::robot::state::SensorReading::Contact(c) => {
                                        ui.label(format!("  {}: {}", sensor_name, c));
                                    }
                                    crate::robot::state::SensorReading::Imu {
                                        linear_accel,
                                        angular_vel,
                                    } => {
                                        ui.label(format!(
                                            "  {}: accel=({:.2},{:.2},{:.2})",
                                            sensor_name,
                                            linear_accel.x,
                                            linear_accel.y,
                                            linear_accel.z
                                        ));
                                        ui.label(format!(
                                            "    gyro=({:.2},{:.2},{:.2})",
                                            angular_vel.x, angular_vel.y, angular_vel.z
                                        ));
                                    }
                                }
                            }
                        }
                    }
                });

            ui.separator();

            // --- Simulation Config (acoustic ray tracing) ---
            egui::CollapsingHeader::new("Simulation Config")
                .id_salt("sim_config_group")
                .default_open(true)
                .show(ui, |ui| {
                    use config_validation as cv;
                    let validation = cv::validate_sim_config(&sim.config);

                    // ray_count slider (log)
                    let mut ray_count = sim.config.ray_count as f64;
                    let ray_resp = ui
                        .add(
                            egui::Slider::new(
                                &mut ray_count,
                                (cv::RAY_COUNT_MIN as f64)..=(cv::RAY_COUNT_MAX as f64),
                            )
                            .text("ray_count")
                            .logarithmic(true)
                            .integer(),
                        )
                        .on_hover_text(cv::RAY_COUNT_HELP);
                    if ray_resp.changed() {
                        sim.config.ray_count = ray_count.round() as u32;
                    }
                    ui.small(cv::RAY_COUNT_HELP);
                    if let Some(err) = &validation.ray_count {
                        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
                    }

                    // max_bounces slider
                    let mut max_bounces = sim.config.max_bounces as i32;
                    let mb_resp = ui
                        .add(
                            egui::Slider::new(
                                &mut max_bounces,
                                (cv::MAX_BOUNCES_MIN as i32)..=(cv::MAX_BOUNCES_MAX as i32),
                            )
                            .text("max_bounces"),
                        )
                        .on_hover_text(cv::MAX_BOUNCES_HELP);
                    if mb_resp.changed() {
                        sim.config.max_bounces = max_bounces.max(0) as u32;
                    }
                    ui.small(cv::MAX_BOUNCES_HELP);
                    if let Some(err) = &validation.max_bounces {
                        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
                    }

                    // grid_resolution slider (log, metres)
                    ui.add(
                        egui::Slider::new(
                            &mut sim.config.grid_resolution,
                            cv::GRID_RES_MIN..=cv::GRID_RES_MAX,
                        )
                        .text("grid_resolution (m)")
                        .logarithmic(true),
                    )
                    .on_hover_text(cv::GRID_RES_HELP);
                    ui.small(cv::GRID_RES_HELP);
                    if let Some(err) = &validation.grid_resolution {
                        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), err);
                    }

                    ui.separator();
                    let can_run = validation.is_valid()
                        && !scene.sound_sources.is_empty()
                        && !scene.meshes.is_empty()
                        && !sim.is_running();
                    let run_resp = ui
                        .add_enabled(can_run, egui::Button::new("Run Simulation"))
                        .on_hover_text(if !validation.is_valid() {
                            "Fix the highlighted parameter errors before running"
                        } else if scene.sound_sources.is_empty() {
                            "Add at least one sound source"
                        } else if scene.meshes.is_empty() {
                            "Add at least one mesh / room"
                        } else if sim.is_running() {
                            "Simulation already running"
                        } else {
                            "Trace rays for the current scene"
                        });
                    if run_resp.clicked() {
                        sim.start(scene);
                    }
                });

            ui.separator();

            // --- Results (simulation output summary) ---
            egui::CollapsingHeader::new("Results")
                .id_salt("results_group")
                .default_open(true)
                .show(ui, |ui| match sim.result() {
                    None => {
                        ui.label("No results yet. Run a simulation above.");
                    }
                    Some(r) => {
                        ui.label(format!("Grid samples: {}", r.energy_grid.len()))
                            .on_hover_text("Energy samples in the spatial grid");
                        ui.label(format!("Ray paths: {}", r.ray_paths.len()));
                        let broadband_max =
                            r.max_energy.iter().copied().fold(0.0_f32, f32::max);
                        ui.label(format!("Max energy: {:.4e}", broadband_max));
                    }
                });

            ui.separator();

            // --- Agent Server Control ---
            egui::CollapsingHeader::new("Agent Server")
                .id_salt("agent_server_control")
                .default_open(false)
                .show(ui, |ui| {
                    let is_running = agent_handle.as_ref().is_some_and(|h| h.status().running);

                    // Enabled toggle
                    let mut enabled = is_running;
                    if ui.checkbox(&mut enabled, "Server Enabled").changed() {
                        if enabled && !is_running {
                            // Start the server: create a new bridge and start.
                            let (bridge_server, new_client) = create_bridge();
                            *bridge_client = new_client;
                            let handle = crate::agent::start_agent_server(
                                agent_config.clone(),
                                bridge_server,
                            );
                            log::info!("Agent server started via UI toggle");
                            *agent_handle = Some(handle);
                        } else if !enabled && is_running {
                            // Stop the server.
                            if let Some(ref mut h) = agent_handle {
                                h.stop();
                            }
                            *agent_handle = None;
                            log::info!("Agent server stopped via UI toggle");
                        }
                    }

                    // Port configuration (only editable when stopped)
                    ui.add_enabled_ui(!is_running, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("TCP Port:");
                            let mut port = agent_config.tcp_port as u32;
                            if ui
                                .add(egui::DragValue::new(&mut port).range(1024..=65535))
                                .changed()
                            {
                                agent_config.tcp_port = port as u16;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("WS Port:");
                            let mut port = agent_config.ws_port as u32;
                            if ui
                                .add(egui::DragValue::new(&mut port).range(1024..=65535))
                                .changed()
                            {
                                agent_config.ws_port = port as u16;
                            }
                        });
                    });

                    // Status display
                    if let Some(ref h) = agent_handle {
                        let status = h.status();
                        if status.running {
                            ui.label(format!("TCP: port {}", status.tcp_port));
                            ui.label(format!("WS:  port {}", status.ws_port));
                            ui.label(format!(
                                "Connections: {} TCP, {} WS",
                                status.tcp_connections, status.ws_connections
                            ));
                        }
                    } else {
                        ui.label("Server stopped");
                    }

                    ui.separator();

                    if ui
                        .button("Start Boxing Match")
                        .on_hover_text("Load boxing scenario with two humanoid fighters and start the agent server")
                        .clicked()
                    {
                        if let Some(ref mut h) = agent_handle {
                            h.stop();
                        }
                        *agent_handle = None;

                        let boxing_config = crate::robot::boxing::BoxingMatchConfig {
                            round_duration: 15.0,
                            num_rounds: 3,
                            ..crate::robot::boxing::BoxingMatchConfig::default()
                        };
                        let (scenario, new_manager) =
                            crate::robot::boxing::BoxingScenario::new(boxing_config);

                        *robot_manager = new_manager;
                        scene.meshes = scenario.ring.meshes;

                        let (new_bridge_server, new_client) =
                            crate::agent::bridge::create_bridge_with_boxing(scenario.boxing_match);
                        *bridge_client = new_client;

                        let handle = crate::agent::start_agent_server(
                            agent_config.clone(),
                            new_bridge_server.clone(),
                        );
                        *bridge_server = Some(new_bridge_server);
                        *agent_handle = Some(handle);
                        log::info!("Boxing match started via UI");
                    }
                });
        });
}

/// Map a (modifiers + key) pair to a camera preset.
/// Used inside the viewport input loop AND in unit tests so the bindings can be
/// asserted without spinning up egui.
pub fn camera_view_for_key(
    modifiers: &egui::Modifiers,
    input: &egui::InputState,
) -> Option<CameraView> {
    if !modifiers.shift {
        return None;
    }
    if input.key_pressed(egui::Key::Num1) {
        return Some(CameraView::Front);
    }
    if input.key_pressed(egui::Key::Num2) {
        return Some(CameraView::Top);
    }
    if input.key_pressed(egui::Key::Num3) {
        return Some(CameraView::Side);
    }
    None
}

/// Pure-data version of `camera_view_for_key` for unit tests — looks up by key
/// + shift bool, no egui InputState dependency.
pub fn camera_view_for_shortcut(shift: bool, key: egui::Key) -> Option<CameraView> {
    if !shift {
        return None;
    }
    match key {
        egui::Key::Num1 => Some(CameraView::Front),
        egui::Key::Num2 => Some(CameraView::Top),
        egui::Key::Num3 => Some(CameraView::Side),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn viewport_3d(
    ctx: &egui::Context,
    scene: &mut Scene,
    sim: &SimulationState,
    vp: &mut ViewportState,
    fluid_sim: &crate::fluids::FluidSimulation,
    gas_sim: &GasSimulation,
    robot_manager: &RobotManager,
    activity_log: &AgentActivityLog,
    bridge: &SimBridgeClient,
) {
    // --- Command palette (Cmd/Ctrl+K) ---
    // Handled before CentralPanel so the modal floats above everything and
    // the palette's text input gets keyboard focus without competing with
    // viewport shortcuts. modifiers.command resolves to Cmd on macOS and
    // Ctrl elsewhere.
    let (cmd_k, modifiers_palette) = ctx.input(|i| (i.key_pressed(egui::Key::K), i.modifiers));
    if cmd_k && modifiers_palette.command {
        vp.palette.toggle();
    }
    if let Some(action) = command_palette::show(ctx, &mut vp.palette) {
        dispatch_palette_action(action, scene, vp);
    }

    egui::CentralPanel::default().show(ctx, |ui| {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

        let rect = response.rect;
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(30, 30, 35));

        // Tele-op top banner — drawn first so any 3D overlays show beneath.
        if vp.teleop_mode {
            let banner_h = 24.0;
            let banner_rect =
                egui::Rect::from_min_size(rect.left_top(), egui::vec2(rect.width(), banner_h));
            painter.rect_filled(
                banner_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(180, 40, 40, 220),
            );
            painter.text(
                banner_rect.center(),
                egui::Align2::CENTER_CENTER,
                "TELE-OP (Ctrl+T to exit) — W/S A/D Q/E → robot/0 joints 0..3",
                egui::FontId::proportional(14.0),
                egui::Color32::WHITE,
            );
        }

        let cam = &mut vp.camera;
        let center = rect.center();

        // --- Smooth focus interpolation tick ---
        let dt = ctx.input(|i| i.predicted_dt).max(1e-3);
        cam.tick_focus(dt);

        // --- Keyboard shortcuts ---
        let modifiers = ui.input(|i| i.modifiers);
        let robot_link_focus =
            |cam: &mut Camera, robot_manager: &RobotManager, ri: usize, li: usize| {
                if let Some(robot) = robot_manager.robots.get(ri) {
                    let poses = robot.state.link_poses_as_mat4();
                    if let Some(p) = poses.get(li) {
                        cam.smooth_focus(Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z), 1.5);
                    }
                }
            };
        ui.input(|i| {
            if i.key_pressed(egui::Key::Num1) {
                vp.mode = InteractionMode::Select;
            }
            if i.key_pressed(egui::Key::Num2) {
                vp.mode = InteractionMode::PlaceSource;
            }
            if i.key_pressed(egui::Key::Num3) {
                vp.mode = InteractionMode::PlaceListener;
            }
            if i.key_pressed(egui::Key::Escape) {
                vp.selection = Selection::None;
                vp.selection_set.clear();
            }
            if i.key_pressed(egui::Key::Tab) {
                vp.fly_mode = !vp.fly_mode;
            }
            // --- Selection-set + hide/isolate shortcuts (deliverable 7) ---
            // A = select all, Alt+A = deselect all, H = hide selection,
            // Alt+H = unhide all, / = toggle isolate. Gated so they don't
            // collide with fly-mode WASD or teleop-mode joint motion.
            let selection_keys_ok = !vp.gizmo.is_active()
                && !vp.fly_mode
                && !vp.teleop_mode
                && !modifiers.command
                && !modifiers.ctrl;
            if selection_keys_ok {
                if i.key_pressed(egui::Key::A) {
                    if modifiers.alt {
                        vp.selection_set.clear();
                        vp.selection = Selection::None;
                    } else {
                        vp.selection_set.clear();
                        for idx in 0..scene.sound_sources.len() {
                            vp.selection_set.add(Selection::Source(idx));
                        }
                        for idx in 0..scene.listeners.len() {
                            vp.selection_set.add(Selection::Listener(idx));
                        }
                        for idx in 0..scene.meshes.len() {
                            vp.selection_set.add(Selection::Object(idx));
                        }
                        vp.selection = vp.selection_set.primary();
                    }
                }
                if i.key_pressed(egui::Key::H) {
                    if modifiers.alt {
                        vp.hidden_state.unhide_all();
                    } else {
                        vp.hidden_state.hide_selection(&vp.selection_set);
                    }
                }
                if i.key_pressed(egui::Key::Slash) {
                    vp.hidden_state.toggle_isolate();
                }
            }
            // Ctrl+Alt+Q toggles quad-view (Top / Front / Side / Persp).
            if (modifiers.ctrl || modifiers.command) && modifiers.alt && i.key_pressed(egui::Key::Q)
            {
                vp.quad_view.toggle();
            }
            // Ctrl+T toggles tele-op mode. Turning it on suppresses fly-mode so
            // WASD/QE feed the robot rather than the camera.
            if modifiers.ctrl && i.key_pressed(egui::Key::T) {
                vp.teleop_mode = !vp.teleop_mode;
                if vp.teleop_mode {
                    vp.fly_mode = false;
                } else {
                    // Emit a zero action immediately so motors stop on next frame.
                    vp.teleop_pending = Some(compute_teleop_action(TeleopKeys::default(), 6));
                }
            }
            // Camera view presets (Blender-style numpad)
            let view_keys = [
                (egui::Key::Num0, CameraView::Perspective),
                (egui::Key::Num7, CameraView::Top),
                (egui::Key::Num5, CameraView::Isometric),
                (egui::Key::OpenBracket, CameraView::RingsideA),
                (egui::Key::CloseBracket, CameraView::RingsideB),
            ];
            // Note: Num1/3 are bound to interaction modes above; we use them only
            // when modifiers.ctrl is held to disambiguate as Front/Side.
            for (k, v) in view_keys {
                if i.key_pressed(k) {
                    cam.set_view(v);
                    vp.current_view = v;
                }
            }
            if modifiers.ctrl && i.key_pressed(egui::Key::Num1) {
                cam.set_view(CameraView::Front);
                vp.current_view = CameraView::Front;
            }
            if modifiers.ctrl && i.key_pressed(egui::Key::Num3) {
                cam.set_view(CameraView::Side);
                vp.current_view = CameraView::Side;
            }
            // Shift+1/2/3 = Front/Top/Side (Blender-style; non-numpad-friendly).
            if let Some(view) = camera_view_for_key(&modifiers, i) {
                cam.set_view(view);
                vp.current_view = view;
            }
            // --- Transform gizmo modal ---
            // Selection-aware G/R/S activate the gizmo. With no selection
            // (or non-position selection like Robot), R falls through to
            // reset-camera.
            let gizmo_target_pos: Option<Vec3> = match vp.selection {
                Selection::Source(idx) if idx < scene.sound_sources.len() => {
                    Some(scene.sound_sources[idx].position)
                }
                Selection::Listener(idx) if idx < scene.listeners.len() => {
                    Some(scene.listeners[idx].position)
                }
                _ => None,
            };

            if vp.gizmo.is_active() {
                // While the gizmo is running, every keypress feeds it —
                // don't fall through to other handlers.
                if i.key_pressed(egui::Key::Escape) {
                    vp.gizmo.cancel();
                    vp.last_history_msg = Some("Gizmo cancelled".into());
                } else if i.key_pressed(egui::Key::Enter) {
                    let mode = vp.gizmo.mode.expect("active gizmo has mode");
                    let delta = vp.gizmo.delta();
                    let snap_active = vp.snap.active(modifiers.shift);
                    let applied = match (mode, vp.selection) {
                        (TransformMode::Translate, Selection::Source(idx))
                            if idx < scene.sound_sources.len() =>
                        {
                            let from = scene.sound_sources[idx].position;
                            let raw_to = from + delta;
                            let to = if snap_active {
                                snap::apply_snap(raw_to, &vp.snap)
                            } else {
                                raw_to
                            };
                            if (to - from).length() > 1e-9 {
                                let _ = vp
                                    .history
                                    .push(SceneCommand::MoveSource { idx, from, to }, scene);
                                let actual = to - from;
                                Some(format!(
                                    "Moved source by ({:+.2}, {:+.2}, {:+.2}){}",
                                    actual.x,
                                    actual.y,
                                    actual.z,
                                    if snap_active { " · snap" } else { "" }
                                ))
                            } else {
                                None
                            }
                        }
                        (TransformMode::Translate, Selection::Listener(idx))
                            if idx < scene.listeners.len() =>
                        {
                            let from = scene.listeners[idx].position;
                            let raw_to = from + delta;
                            let to = if snap_active {
                                snap::apply_snap(raw_to, &vp.snap)
                            } else {
                                raw_to
                            };
                            if (to - from).length() > 1e-9 {
                                let _ = vp
                                    .history
                                    .push(SceneCommand::MoveListener { idx, from, to }, scene);
                                let actual = to - from;
                                Some(format!(
                                    "Moved listener by ({:+.2}, {:+.2}, {:+.2}){}",
                                    actual.x,
                                    actual.y,
                                    actual.z,
                                    if snap_active { " · snap" } else { "" }
                                ))
                            } else {
                                None
                            }
                        }
                        (TransformMode::Rotate, _) | (TransformMode::Scale, _) => {
                            // Rotate / Scale apply paths are deliverable
                            // 9's job (properties polish + per-object
                            // transform field). The state machine + UI is
                            // ready; the apply step no-ops gracefully.
                            Some(format!(
                                "{} not yet applied to this selection",
                                mode.label()
                            ))
                        }
                        _ => None,
                    };
                    if let Some(msg) = applied {
                        vp.last_history_msg = Some(msg);
                    }
                    vp.gizmo.confirm();
                } else if i.key_pressed(egui::Key::X) {
                    vp.gizmo.set_axis(AxisLock::X);
                } else if i.key_pressed(egui::Key::Y) {
                    vp.gizmo.set_axis(AxisLock::Y);
                } else if i.key_pressed(egui::Key::Z) {
                    vp.gizmo.set_axis(AxisLock::Z);
                } else if i.key_pressed(egui::Key::Backspace) {
                    vp.gizmo.backspace();
                } else {
                    // Numeric typing — egui::Event::Text carries pressed chars.
                    for ev in &i.events {
                        if let egui::Event::Text(s) = ev {
                            for c in s.chars() {
                                vp.gizmo.type_char(c);
                            }
                        }
                    }
                }
            } else {
                // Gizmo inactive — selection-bearing keys begin a gizmo;
                // otherwise R still resets camera.
                let can_begin = gizmo_target_pos.is_some()
                    && !vp.teleop_mode
                    && !vp.fly_mode
                    && !modifiers.command
                    && !modifiers.ctrl
                    && !modifiers.shift
                    && !modifiers.alt;
                if can_begin {
                    let p = gizmo_target_pos.unwrap();
                    if i.key_pressed(egui::Key::G) {
                        vp.gizmo.begin(TransformMode::Translate, p);
                    } else if i.key_pressed(egui::Key::R) {
                        vp.gizmo.begin(TransformMode::Rotate, p);
                    } else if i.key_pressed(egui::Key::S) {
                        vp.gizmo.begin(TransformMode::Scale, p);
                    }
                }
                // R = reset camera fallback (only if gizmo path didn't fire).
                if !vp.gizmo.is_active() && i.key_pressed(egui::Key::R) {
                    *cam = Camera::default();
                    if !scene.meshes.is_empty() {
                        focus_on_scene(cam, scene);
                    }
                }
            }
            if i.key_pressed(egui::Key::Home) {
                focus_on_scene(cam, scene);
            }
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                match vp.selection {
                    Selection::Source(idx) if idx < scene.sound_sources.len() => {
                        let snap = scene.sound_sources[idx].clone();
                        let _ = vp.history.push(
                            SceneCommand::RemoveSource {
                                idx,
                                snapshot: snap,
                            },
                            scene,
                        );
                        vp.selection = Selection::None;
                    }
                    Selection::Listener(idx) if idx < scene.listeners.len() => {
                        let snap = scene.listeners[idx].clone();
                        let _ = vp.history.push(
                            SceneCommand::RemoveListener {
                                idx,
                                snapshot: snap,
                            },
                            scene,
                        );
                        vp.selection = Selection::None;
                    }
                    Selection::Object(idx) if idx < scene.meshes.len() => {
                        let snap = scene.meshes[idx].clone();
                        let _ = vp.history.push(
                            SceneCommand::RemoveObject {
                                idx,
                                snapshot: snap,
                            },
                            scene,
                        );
                        vp.selection = Selection::None;
                    }
                    _ => {}
                }
            }
            // Undo / Redo — Cmd+Z (Mac) / Ctrl+Z (others), shift adds redo.
            // `modifiers.command` resolves to Cmd on macOS, Ctrl elsewhere.
            if modifiers.command && i.key_pressed(egui::Key::Z) {
                if modifiers.shift {
                    if let Some(name) = vp.history.redo(scene) {
                        vp.last_history_msg = Some(format!("Redid: {name}"));
                    }
                } else if let Some(name) = vp.history.undo(scene) {
                    vp.last_history_msg = Some(format!("Undid: {name}"));
                    // Selection may now point at a removed item; defensively
                    // reset rather than render against stale indices.
                    vp.selection = Selection::None;
                }
            }
            // Ctrl+Y = redo (Windows-style alternative).
            if modifiers.command && i.key_pressed(egui::Key::Y) {
                if let Some(name) = vp.history.redo(scene) {
                    vp.last_history_msg = Some(format!("Redid: {name}"));
                }
            }
            if i.key_pressed(egui::Key::F) {
                match vp.selection {
                    Selection::Source(idx) if idx < scene.sound_sources.len() => {
                        cam.smooth_focus(scene.sound_sources[idx].position, 1.5);
                    }
                    Selection::Listener(idx) if idx < scene.listeners.len() => {
                        cam.smooth_focus(scene.listeners[idx].position, 1.5);
                    }
                    Selection::Robot(ri) => {
                        if let Some(robot) = robot_manager.robots.get(ri) {
                            let poses = robot.state.link_poses_as_mat4();
                            if let Some(p) = poses.first() {
                                cam.smooth_focus(
                                    Vec3::new(p.w_axis.x, p.w_axis.y, p.w_axis.z),
                                    2.0,
                                );
                            }
                        }
                    }
                    Selection::RobotLink(ri, li) => {
                        robot_link_focus(cam, robot_manager, ri, li);
                    }
                    _ => {
                        focus_on_scene(cam, scene);
                    }
                }
            }

            // --- D7 selection hotkeys (gated off when fly/teleop modes own
            // the same keys to avoid stealing input). ---
            let selection_keys_active = !vp.fly_mode && !vp.teleop_mode && !vp.palette.open;
            if selection_keys_active {
                // A — Select All (plain). Alt+A — Deselect All.
                if i.key_pressed(egui::Key::A) {
                    if modifiers.alt {
                        vp.selection_set.clear();
                        vp.selection = Selection::None;
                        vp.selection_anchor = Selection::None;
                    } else if !modifiers.command && !modifiers.ctrl && !modifiers.shift {
                        let counts = selection_set::PickableCounts {
                            sources: scene.sound_sources.len(),
                            listeners: scene.listeners.len(),
                            objects: scene.meshes.len(),
                            robots: robot_manager.robots.len(),
                        };
                        selection_set::select_all(&mut vp.selection_set, counts);
                        vp.selection = vp.selection_set.primary();
                    }
                }
                // H — Hide selection. Alt+H — Unhide all.
                if i.key_pressed(egui::Key::H) {
                    if modifiers.alt {
                        vp.hidden_state.unhide_all();
                    } else {
                        vp.hidden_state.hide_selection(&vp.selection_set);
                    }
                }
                // / — Toggle isolate.
                if i.key_pressed(egui::Key::Slash) {
                    vp.hidden_state.toggle_isolate();
                }
                // B — Arm box select. Next viewport drag draws a rubber-band
                // and selects everything inside on release.
                if i.key_pressed(egui::Key::B) {
                    vp.box_select_armed = !vp.box_select_armed;
                }
            }

            // --- Tele-op key sampling (Ctrl+T to toggle) ---
            if vp.teleop_mode {
                let num_motors = robot_manager
                    .get_robot(0)
                    .map(|r| r.definition.joints.len())
                    .unwrap_or(0);
                let keys = TeleopKeys {
                    w: i.key_down(egui::Key::W),
                    a: i.key_down(egui::Key::A),
                    s: i.key_down(egui::Key::S),
                    d: i.key_down(egui::Key::D),
                    q: i.key_down(egui::Key::Q),
                    e: i.key_down(egui::Key::E),
                };
                vp.teleop_pending = Some(compute_teleop_action(keys, num_motors));
            }

            // --- Fly-mode movement ---
            if vp.fly_mode && !vp.teleop_mode {
                let mut fwd = 0.0;
                let mut rt = 0.0;
                let mut up = 0.0;
                if i.key_down(egui::Key::W) {
                    fwd += 1.0;
                }
                if i.key_down(egui::Key::S) {
                    fwd -= 1.0;
                }
                if i.key_down(egui::Key::D) {
                    rt += 1.0;
                }
                if i.key_down(egui::Key::A) {
                    rt -= 1.0;
                }
                if i.key_down(egui::Key::E) {
                    up += 1.0;
                }
                if i.key_down(egui::Key::Q) {
                    up -= 1.0;
                }
                let sprint = if modifiers.shift { 3.0 } else { 1.0 };
                let step = vp.fly_speed * dt * sprint;
                if fwd != 0.0 || rt != 0.0 || up != 0.0 {
                    cam.fly(fwd * step, rt * step, up * step);
                }
            }
        });

        // --- Camera controls ---
        let is_orbit = !vp.fly_mode
            && (response.dragged_by(egui::PointerButton::Middle)
                || (response.dragged_by(egui::PointerButton::Primary) && modifiers.alt));
        let is_pan = !vp.fly_mode && response.dragged_by(egui::PointerButton::Secondary);
        let is_fly_look = vp.fly_mode && response.dragged_by(egui::PointerButton::Secondary);

        if is_orbit {
            let d = response.drag_delta();
            cam.orbit(d.x, d.y);
        }
        if is_pan {
            let d = response.drag_delta();
            cam.pan(d.x, d.y);
        }
        if is_fly_look {
            let d = response.drag_delta();
            cam.look(d.x, d.y);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                // Zoom-to-cursor when we have a valid ground hover point.
                let prev_scale = rect.height() * 0.4 / cam.distance;
                let pivot = response.hover_pos().and_then(|hp| {
                    let (origin, dir) = screen_to_ray(hp, cam, center, prev_scale);
                    ray_ground_intersect(origin, dir)
                });
                match pivot {
                    Some(p) => cam.zoom_toward(p, scroll * 0.1),
                    None => cam.zoom(scroll * 0.1),
                }
            }
            // Trackpad pinch — egui exposes zoom_delta() centred on 1.0.
            // We dampen via shape_zoom_delta so single-finger drift doesn't
            // micro-zoom, then apply through gestures::apply_pinch_zoom to
            // the camera distance directly (independent of scroll-wheel).
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if (zoom_delta - 1.0).abs() > 1e-3 {
                let shaped = gestures::shape_zoom_delta(zoom_delta, 0.8);
                cam.distance = gestures::apply_pinch_zoom(cam.distance, shaped);
            }
        }

        // Recalculate scale after camera changes
        let scale = rect.height() * 0.4 / cam.distance;

        // --- Hover world position + hover label ---
        vp.hover_world = None;
        vp.hover_label = None;
        if let Some(hover_pos) = response.hover_pos() {
            let (origin, dir) = screen_to_ray(hover_pos, cam, center, scale);
            vp.hover_world = ray_ground_intersect(origin, dir);
            if let Some(label) =
                hover_label_test(hover_pos, scene, robot_manager, cam, center, scale)
            {
                vp.hover_label = Some((hover_pos, label));
            }
        }

        // --- Object interaction ---
        if !is_orbit && !is_pan && !is_fly_look {
            if response.drag_started_by(egui::PointerButton::Primary) && !modifiers.alt {
                if let Some(hover) = response.hover_pos() {
                    let sel = hit_test(hover, scene, robot_manager, cam, center, scale);
                    if sel != Selection::None {
                        vp.selection = sel;
                        vp.dragging = true;
                    }
                }
            }

            if response.dragged_by(egui::PointerButton::Primary) && vp.dragging && !modifiers.alt {
                if let Some(hover) = response.hover_pos() {
                    let (origin, dir) = screen_to_ray(hover, cam, center, scale);
                    if let Some(gp) = ray_ground_intersect(origin, dir) {
                        match vp.selection {
                            Selection::Source(i) if i < scene.sound_sources.len() => {
                                scene.sound_sources[i].position.x = gp.x;
                                scene.sound_sources[i].position.z = gp.z;
                            }
                            Selection::Listener(i) if i < scene.listeners.len() => {
                                scene.listeners[i].position.x = gp.x;
                                scene.listeners[i].position.z = gp.z;
                            }
                            _ => {}
                        }
                    }
                }
            }

            if response.clicked_by(egui::PointerButton::Primary) && !modifiers.alt {
                if let Some(hover) = response.hover_pos() {
                    let (origin, dir) = screen_to_ray(hover, cam, center, scale);
                    let ground = ray_ground_intersect(origin, dir);

                    match vp.mode {
                        InteractionMode::Select => {
                            let hit = hit_test(hover, scene, robot_manager, cam, center, scale);
                            let new_anchor = selection_set::apply_pick(
                                &mut vp.selection_set,
                                vp.selection_anchor,
                                hit,
                                modifiers.command || modifiers.ctrl,
                                modifiers.shift,
                            );
                            vp.selection_anchor = new_anchor;
                            vp.selection = vp.selection_set.primary();
                        }
                        InteractionMode::PlaceSource => {
                            if let Some(gp) = ground {
                                let src = SoundSource {
                                    position: Vec3::new(gp.x, 1.0, gp.z),
                                    ..Default::default()
                                };
                                let _ = vp.history.push(
                                    SceneCommand::InsertSource {
                                        idx: scene.sound_sources.len(),
                                        src,
                                    },
                                    scene,
                                );
                                vp.selection = Selection::Source(scene.sound_sources.len() - 1);
                            }
                        }
                        InteractionMode::PlaceListener => {
                            if let Some(gp) = ground {
                                let n = scene.listeners.len() + 1;
                                let listener = Listener {
                                    position: Vec3::new(gp.x, 1.0, gp.z),
                                    name: format!("Listener {n}"),
                                    ..Listener::default()
                                };
                                let _ = vp.history.push(
                                    SceneCommand::InsertListener {
                                        idx: scene.listeners.len(),
                                        listener,
                                    },
                                    scene,
                                );
                                vp.selection = Selection::Listener(scene.listeners.len() - 1);
                            }
                        }
                    }
                }
            }
        }

        if response.drag_stopped() {
            vp.dragging = false;
        }

        // --- Right-click context menu ---
        // Selecting on right-click (before showing the menu) lets the menu
        // be selection-aware. If the right-click was a drag (panning), it
        // won't trigger context_menu — egui treats drag and click as
        // distinct on the secondary button.
        if response.secondary_clicked() && !vp.gizmo.is_active() {
            if let Some(hover) = response.hover_pos() {
                vp.selection = hit_test(hover, scene, robot_manager, cam, center, scale);
            }
        }
        let menu_selection = vp.selection;
        // Field-scoped borrow so the closure doesn't clash with `cam`
        // (which holds `&mut vp.camera` across this block).
        let pending = &mut vp.pending_palette_action;
        response.context_menu(|ui| {
            let items = context_menu_items_for(menu_selection);
            if items.is_empty() {
                ui.label("(no actions)");
                return;
            }
            for item in items {
                if ui.button(item.label).clicked() {
                    *pending = Some(item.action);
                    ui.close_menu();
                }
            }
        });

        // --- Drawing ---

        // Grid (distance-faded, axis-tinted)
        if vp.show_grid {
            let half = 12i32;
            let cam_xz = glam::Vec2::new(cam.target.x, cam.target.z);
            let alpha_for = |w: glam::Vec2| -> u8 {
                let d = (w - cam_xz).length();
                let max_d = (half as f32) * 1.2;
                let t = (1.0 - (d / max_d)).clamp(0.0, 1.0);
                (t * 75.0) as u8
            };
            for i in -half..=half {
                let f = i as f32;
                // X-axis line (red tint when i==0)
                let is_axis = i == 0;
                let base_x = if is_axis {
                    egui::Color32::from_rgb(220, 80, 80)
                } else {
                    egui::Color32::from_rgb(120, 120, 130)
                };
                let base_z = if is_axis {
                    egui::Color32::from_rgb(80, 140, 220)
                } else {
                    egui::Color32::from_rgb(120, 120, 130)
                };
                let wp1 = glam::Vec2::new(f, -half as f32);
                let wp2 = glam::Vec2::new(f, half as f32);
                let wp3 = glam::Vec2::new(-half as f32, f);
                let wp4 = glam::Vec2::new(half as f32, f);
                let a_z = alpha_for(wp1).min(alpha_for(wp2));
                let a_x = alpha_for(wp3).min(alpha_for(wp4));
                let stroke_w = if is_axis { 1.5 } else { 0.6 };
                let cz =
                    egui::Color32::from_rgba_unmultiplied(base_z.r(), base_z.g(), base_z.b(), a_z);
                let cx =
                    egui::Color32::from_rgba_unmultiplied(base_x.r(), base_x.g(), base_x.b(), a_x);
                let p1 = project_3d(Vec3::new(f, 0.0, -half as f32), cam, center, scale);
                let p2 = project_3d(Vec3::new(f, 0.0, half as f32), cam, center, scale);
                painter.line_segment([p1, p2], egui::Stroke::new(stroke_w, cz));
                let p3 = project_3d(Vec3::new(-half as f32, 0.0, f), cam, center, scale);
                let p4 = project_3d(Vec3::new(half as f32, 0.0, f), cam, center, scale);
                painter.line_segment([p3, p4], egui::Stroke::new(stroke_w, cx));
            }
            // Origin marker
            let o = project_3d(Vec3::ZERO, cam, center, scale);
            painter.circle_stroke(
                o,
                3.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 180, 180)),
            );
        }

        // Scene meshes (wireframe)
        if vp.show_meshes {
            for (i, obj) in scene.meshes.iter().enumerate() {
                if !obj.visible {
                    continue;
                }
                let is_selected = vp.selection == Selection::Object(i);
                let base_color = egui::Color32::from_rgb(
                    (obj.material.color[0] * 255.0) as u8,
                    (obj.material.color[1] * 255.0) as u8,
                    (obj.material.color[2] * 255.0) as u8,
                );
                let stroke_width = if is_selected { 2.0 } else { 1.0 };
                let color = if is_selected {
                    egui::Color32::from_rgb(100, 200, 255)
                } else {
                    base_color
                };
                for tri in &obj.mesh.triangles {
                    let p0 = project_3d(tri.vertices[0].position, cam, center, scale);
                    let p1 = project_3d(tri.vertices[1].position, cam, center, scale);
                    let p2 = project_3d(tri.vertices[2].position, cam, center, scale);
                    painter.line_segment([p0, p1], egui::Stroke::new(stroke_width, color));
                    painter.line_segment([p1, p2], egui::Stroke::new(stroke_width, color));
                    painter.line_segment([p2, p0], egui::Stroke::new(stroke_width, color));
                }
            }
        }

        // Ray paths — render the running stream as it arrives, then the
        // final result once the worker has joined. This is the visible
        // non-blocking behaviour: rays keep appearing while the worker is
        // still tracing.
        if vp.show_rays {
            let ray_color = egui::Color32::from_rgba_premultiplied(255, 200, 50, 30);
            let live_paths = sim.partial_paths();
            let max_draw = 500.min(live_paths.len());
            for path in &live_paths[..max_draw] {
                for segment in path.windows(2) {
                    let p1 = project_3d(segment[0], cam, center, scale);
                    let p2 = project_3d(segment[1], cam, center, scale);
                    painter.line_segment([p1, p2], egui::Stroke::new(0.5, ray_color));
                }
            }

            if let Some(result) = sim.result() {
                // Visualise the band-summed total. SimulationResult now
                // carries per-band data; the 2D viewport collapses it to a
                // single scalar so the heat map stays meaningful until a
                // band-aware UI surfaces it explicitly.
                let max_total = result.max_energy.iter().sum::<f32>();
                for gp in &result.energy_grid {
                    let total = gp.energy_total();
                    if total > 0.01 {
                        let color = energy_to_color(total, max_total);
                        let p = project_3d(gp.position, cam, center, scale);
                        if rect.contains(p) {
                            painter.circle_filled(p, 2.0, color);
                        }
                    }
                }
            }
        }

        // Fluid slice visualization
        if vp.show_fluid {
            if let Some(ref grid) = fluid_sim.grid {
                render_fluid_slice(
                    grid,
                    vp.fluid_slice_y,
                    vp.fluid_viz_mode,
                    &painter,
                    cam,
                    center,
                    scale,
                    rect,
                );
            }
        }

        // Gas slice visualization
        if vp.show_gas {
            if let Some(ref grid) = gas_sim.grid {
                render_gas_slice(
                    grid,
                    vp.gas_slice_y,
                    vp.gas_species_idx,
                    vp.gas_viz_mode,
                    &painter,
                    cam,
                    center,
                    scale,
                    rect,
                );
            }
        }

        // --- Robot rendering ---
        if vp.show_robots {
            render_robots(
                &painter,
                robot_manager,
                vp.show_sensor_rays,
                vp.shaded,
                vp.selection,
                activity_log,
                cam,
                center,
                scale,
                rect,
            );

            // --- Environment interaction overlays ---
            render_environment_interactions(
                &painter,
                robot_manager,
                scene,
                fluid_sim,
                gas_sim,
                cam,
                center,
                scale,
                rect,
            );

            // --- Boxing HUD overlays ---
            if vp.show_boxing_hud && bridge.boxing_match.is_some() {
                render_health_bars(&painter, robot_manager, cam, center, scale, rect);
                render_boxing_score_overlay(&painter, bridge, rect);
                render_message_feed(&painter, &vp.boxing_messages, activity_log.elapsed, rect);
                render_hit_flash(&painter, vp.hit_flash_timer, rect);
            }
        }

        // Update hit flash timer
        let dt = ctx.input(|i| i.predicted_dt);
        if vp.hit_flash_timer > 0.0 {
            vp.hit_flash_timer = (vp.hit_flash_timer - dt * 4.0).max(0.0);
        }

        // Trigger flash on new hit events
        if !robot_manager.last_hit_events.is_empty() {
            vp.hit_flash_timer = 1.0;
            if let Some(hit) = robot_manager.last_hit_events.first() {
                vp.hit_flash_robot = Some(hit.target_robot);
            }
        }

        // Camera auto-track: follow midpoint between two boxing robots
        if vp.camera_auto_track {
            if let Some(ref bm) = bridge.boxing_match {
                let robot_a_id = bm.robot_a;
                let robot_b_id = bm.robot_b;
                if let (Some(ra), Some(rb)) = (
                    robot_manager.get_robot(robot_a_id),
                    robot_manager.get_robot(robot_b_id),
                ) {
                    let poses_a = ra.state.link_poses_as_mat4();
                    let poses_b = rb.state.link_poses_as_mat4();
                    if !poses_a.is_empty() && !poses_b.is_empty() {
                        let pos_a = Vec3::new(
                            poses_a[0].w_axis.x,
                            poses_a[0].w_axis.y,
                            poses_a[0].w_axis.z,
                        );
                        let pos_b = Vec3::new(
                            poses_b[0].w_axis.x,
                            poses_b[0].w_axis.y,
                            poses_b[0].w_axis.z,
                        );
                        let mid = (pos_a + pos_b) * 0.5;
                        let dist = (pos_a - pos_b).length();
                        let target = Vec3::new(mid.x, mid.y + 0.3, mid.z);
                        let lerp_f = 0.05;
                        cam.target = cam.target + (target - cam.target) * lerp_f;
                        let desired_dist = (dist * 2.5).max(4.0);
                        cam.distance += (desired_dist - cam.distance) * lerp_f;
                        cam.update_position();
                    }
                }
            }
        }

        // Collect messages from activity log for the message feed
        for event in activity_log.iter() {
            if event.kind == AgentEventKind::Message {
                let robot_id = event.robot_id.unwrap_or(0);
                let already_has = vp.boxing_messages.iter().any(|(t, msg, rid)| {
                    (*t - event.timestamp).abs() < 0.01
                        && msg == &event.description
                        && *rid == robot_id
                });
                if !already_has {
                    vp.boxing_messages
                        .push((event.timestamp, event.description.clone(), robot_id));
                    if vp.boxing_messages.len() > 50 {
                        vp.boxing_messages.remove(0);
                    }
                }
            }
        }

        // Sound sources
        if vp.show_sources {
            for (i, source) in scene.sound_sources.iter().enumerate() {
                if !source.enabled {
                    continue;
                }
                let p = project_3d(source.position, cam, center, scale);
                let is_selected = vp.selection == Selection::Source(i);
                if is_selected {
                    painter.circle_stroke(
                        p,
                        12.0,
                        egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 220, 80)),
                    );
                }
                // Glow
                painter.circle_filled(
                    p,
                    9.0,
                    egui::Color32::from_rgba_unmultiplied(255, 100, 50, 70),
                );
                painter.circle_filled(p, 6.0, egui::Color32::from_rgb(255, 100, 50));
                painter.text(
                    p + egui::vec2(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("S{}", i + 1),
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
            }
        }

        // Listeners
        if vp.show_listeners {
            for (i, listener) in scene.listeners.iter().enumerate() {
                let p = project_3d(listener.position, cam, center, scale);
                let is_selected = vp.selection == Selection::Listener(i);
                if is_selected {
                    painter.circle_stroke(
                        p,
                        11.0,
                        egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 220, 80)),
                    );
                }
                painter.circle_filled(
                    p,
                    8.0,
                    egui::Color32::from_rgba_unmultiplied(50, 150, 255, 70),
                );
                painter.circle_filled(p, 5.0, egui::Color32::from_rgb(50, 150, 255));
                painter.text(
                    p + egui::vec2(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    &listener.name,
                    egui::FontId::proportional(12.0),
                    egui::Color32::WHITE,
                );
            }
        }

        // Placement preview
        if vp.mode != InteractionMode::Select {
            if let Some(gp) = vp.hover_world {
                let preview_pos = Vec3::new(gp.x, 1.0, gp.z);
                let p = project_3d(preview_pos, cam, center, scale);
                let color = match vp.mode {
                    InteractionMode::PlaceSource => {
                        egui::Color32::from_rgba_premultiplied(255, 100, 50, 100)
                    }
                    InteractionMode::PlaceListener => {
                        egui::Color32::from_rgba_premultiplied(50, 150, 255, 100)
                    }
                    _ => egui::Color32::TRANSPARENT,
                };
                painter.circle_stroke(p, 8.0, egui::Stroke::new(2.0, color));
                // Ground marker
                let ground_p = project_3d(gp, cam, center, scale);
                painter.circle_stroke(
                    ground_p,
                    4.0,
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 100, 100)),
                );
                painter.line_segment(
                    [ground_p, p],
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_premultiplied(150, 150, 150, 80),
                    ),
                );
            }
        }

        // Gizmo HUD — top-center banner while a transform is in flight.
        if let Some(hud) = vp.gizmo.hud_text() {
            let h = 26.0;
            let banner_rect = egui::Rect::from_min_size(
                rect.left_top() + egui::vec2(0.0, if vp.teleop_mode { 24.0 } else { 0.0 }),
                egui::vec2(rect.width(), h),
            );
            painter.rect_filled(
                banner_rect,
                0.0,
                egui::Color32::from_rgba_unmultiplied(60, 90, 140, 220),
            );
            painter.text(
                banner_rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("{hud} · X/Y/Z lock · type value · Enter confirm · Esc cancel"),
                egui::FontId::proportional(13.0),
                egui::Color32::WHITE,
            );
        }

        // Hover tooltip
        if let Some((pos, label)) = &vp.hover_label {
            let offset = egui::vec2(14.0, -8.0);
            let pad = egui::vec2(6.0, 3.0);
            let font = egui::FontId::proportional(11.0);
            let galley = painter.layout_no_wrap(label.clone(), font.clone(), egui::Color32::WHITE);
            let text_size = galley.size();
            let anchor = *pos + offset;
            let bg = egui::Rect::from_min_size(anchor, text_size + pad * 2.0);
            painter.rect_filled(bg, 3.0, egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200));
            painter.rect_stroke(
                bg,
                3.0,
                egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(255, 220, 80, 160),
                ),
                egui::StrokeKind::Outside,
            );
            painter.galley(anchor + pad, galley, egui::Color32::WHITE);
        }

        // Camera/mode indicator (top-left)
        let cam_info = if vp.fly_mode {
            format!("FLY · speed {:.1} · {:?}", vp.fly_speed, vp.current_view)
        } else {
            format!("{:?}", vp.current_view)
        };
        painter.text(
            rect.min + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            cam_info,
            egui::FontId::proportional(11.0),
            egui::Color32::from_rgba_unmultiplied(200, 200, 210, 180),
        );

        // Empty state
        if scene.meshes.is_empty() && scene.sound_sources.is_empty() && scene.listeners.is_empty() {
            painter.text(
                center,
                egui::Align2::CENTER_CENTER,
                "EchoMap\n\nAdd > Box Room  to start\nor  File > Open STEP File",
                egui::FontId::proportional(18.0),
                egui::Color32::from_rgb(120, 120, 120),
            );
        }
    });
}

/// Adapter exposing the current PerfGovernor class label to the status
/// bar. Kept as a free fn so unit tests don't need an egui context.
pub fn perf_label_for(gov: &PerfGovernor) -> &'static str {
    gov.class().label()
}

/// Standalone Performance window — Settings → Performance equivalent
/// that doesn't require threading caps + governor through every
/// existing settings call site.
pub fn performance_window(
    ctx: &egui::Context,
    open: &mut bool,
    caps: &DeviceCaps,
    gov: &PerfGovernor,
) {
    egui::Window::new("Performance")
        .open(open)
        .resizable(false)
        .show(ctx, |ui| perf_settings_section(ui, caps, gov));
}

/// Render a Performance section into an existing settings window. Pure
/// read of [`DeviceCaps`] and [`PerfGovernor`]; never mutates anything
/// (overrides happen via env vars at startup).
pub fn perf_settings_section(ui: &mut egui::Ui, caps: &DeviceCaps, gov: &PerfGovernor) {
    ui.heading("Performance");
    ui.label(format!("Device: {}", caps.summary()));
    ui.label(format!(
        "Governor: {}  ·  avg frame {:.1} ms over {} samples",
        gov.class().label(),
        gov.avg_frame_time().as_secs_f32() * 1000.0,
        gov.sample_count(),
    ));
    ui.small(
        "Override defaults with ECHOMAP_SIM_THREADS, ECHOMAP_RAY_PATHS, \
         ECHOMAP_HEATMAP_RES env vars; ECHOMAP_STRESS=1 enables crash-injection smoke.",
    );
}

pub fn status_bar(
    ctx: &egui::Context,
    vp: &ViewportState,
    scene: &Scene,
    robot_manager: &RobotManager,
    sim: &SimulationState,
    status: &AppStatus,
) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        // Three-column layout: left=status/breadcrumb, centre=progress, right=Cancel.
        // Wrap so narrow windows don't overlap the right-aligned counts onto the left breadcrumb.
        ui.horizontal_wrapped(|ui| {
            // Left: severity message OR breadcrumb fallback.
            if !status.message.is_empty() {
                ui.colored_label(status.color(), &status.message)
                    .on_hover_text("Latest status message — also visible in the log");
            } else {
                let crumb = vp.selection.breadcrumb(scene, robot_manager);
                ui.colored_label(egui::Color32::from_rgb(255, 220, 120), &crumb)
                    .on_hover_text("Selected object path");
            }

            ui.separator();
            let mode_str = match vp.mode {
                InteractionMode::Select => "Select",
                InteractionMode::PlaceSource => "Place Source",
                InteractionMode::PlaceListener => "Place Listener",
            };
            ui.label(format!("Mode: {mode_str}"));

            if let Some(pos) = vp.hover_world {
                ui.separator();
                ui.label(format!("World: ({:.2}, {:.2})", pos.x, pos.z));
            }

            // Centre/right: progress bar with "N/M rays" overlay, then Cancel button.
            // Lay out from the right edge so the Cancel sits flush right.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let total_rays = sim.config.ray_count.max(1) as f32;
                let sim_progress = sim.progress();
                let done_rays = (sim_progress.clamp(0.0, 1.0) * total_rays) as u32;

                let cancel_enabled = sim.is_running();
                let resp = ui
                    .add_enabled(cancel_enabled, egui::Button::new("Cancel"))
                    .on_hover_text(if cancel_enabled {
                        "Request stop of the running simulation"
                    } else {
                        "Disabled — no simulation in flight"
                    });
                if resp.clicked() {
                    log::info!("Cancel requested via status bar");
                }

                ui.separator();
                ui.label(format!(
                    "Objects: {} | Sources: {} | Listeners: {} | Robots: {}",
                    scene.meshes.len(),
                    scene.sound_sources.len(),
                    scene.listeners.len(),
                    robot_manager.robots.len()
                ));

                ui.separator();
                if sim.is_running() {
                    ui.add(
                        egui::ProgressBar::new(sim_progress)
                            .show_percentage()
                            .desired_width(180.0)
                            .text(format!(
                                "{:.0}% ({}/{} rays)",
                                sim_progress * 100.0,
                                done_rays,
                                sim.config.ray_count
                            )),
                    )
                    .on_hover_text("Live simulation progress");
                } else {
                    ui.allocate_exact_size(egui::vec2(180.0, 12.0), egui::Sense::hover());
                    ui.label("Idle").on_hover_text(
                        "No simulation running — press Run in the Simulation Config group",
                    );
                }
            });
        });
    });
}

/// About dialog — shows version from `CARGO_PKG_VERSION` and project link.
pub fn about_window(ctx: &egui::Context, open: &mut bool) {
    let version = env!("CARGO_PKG_VERSION");
    egui::Window::new("About EchoMap")
        .open(open)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("EchoMap");
                ui.label(format!("Version {version}"));
                ui.separator();
                ui.label("Desktop acoustic visualization tool");
                ui.label("STEP-driven sound propagation simulator");
                ui.separator();
                ui.label("Press F1 for the keyboard cheat sheet.");
            });
        });
}

/// Agent activity panel: shows live connection status, command log, sensor
/// readings, step count, and reward per robot. Also controls the demo agent.
#[allow(clippy::too_many_arguments)]
pub fn agent_activity_panel(
    ctx: &egui::Context,
    activity_log: &AgentActivityLog,
    robot_manager: &RobotManager,
    agent_handle: &Option<AgentServerHandle>,
    demo_handle: &mut Option<DemoAgentHandle>,
    demo_behavior: &mut DemoBehavior,
    bridge_server: &Option<SimBridgeServer>,
) {
    egui::SidePanel::right("agent_activity_panel")
        .default_width(260.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Agent Activity");
            ui.separator();

            // --- Demo Agent Controls ---
            egui::CollapsingHeader::new("Demo Agent")
                .id_salt("demo_agent_ctrl")
                .default_open(true)
                .show(ui, |ui| {
                    let is_running = demo_handle.as_ref().is_some_and(|h| h.is_running());

                    if is_running {
                        ui.colored_label(egui::Color32::from_rgb(80, 220, 80), "Running");

                        // Behavior selector
                        ui.horizontal(|ui| {
                            ui.label("Behavior:");
                        });
                        let behaviors = [
                            DemoBehavior::ReachTarget,
                            DemoBehavior::ExploreRoom,
                            DemoBehavior::AvoidObstacles,
                        ];
                        for b in &behaviors {
                            if ui
                                .selectable_label(*demo_behavior == *b, b.label())
                                .clicked()
                            {
                                *demo_behavior = *b;
                                if let Some(ref h) = demo_handle {
                                    h.set_behavior(*b);
                                }
                            }
                        }

                        if ui.button("Stop Demo Agent").clicked() {
                            if let Some(ref mut h) = demo_handle {
                                h.stop();
                            }
                            *demo_handle = None;
                        }
                    } else {
                        ui.colored_label(egui::Color32::from_rgb(180, 180, 180), "Stopped");
                        if let Some(ref bridge) = bridge_server {
                            if ui.button("Start Demo Agent").clicked() {
                                let handle = crate::agent::demo::start_demo_agent(
                                    bridge.clone(),
                                    *demo_behavior,
                                );
                                *demo_handle = Some(handle);
                            }
                        } else {
                            ui.label("Enable Agent Server first.");
                        }
                    }
                });

            ui.separator();

            // --- Connection Status ---
            let server_running = agent_handle.as_ref().is_some_and(|h| h.status().running);
            let status_text = if server_running { "Running" } else { "Stopped" };
            let status_color = if server_running {
                egui::Color32::from_rgb(80, 220, 80)
            } else {
                egui::Color32::from_rgb(180, 180, 180)
            };
            ui.horizontal(|ui| {
                ui.label("Server:");
                ui.colored_label(status_color, status_text);
            });

            if let Some(ref h) = agent_handle {
                let status = h.status();
                if status.running {
                    ui.label(format!(
                        "  TCP:{} ({} conn)  WS:{} ({} conn)",
                        status.tcp_port,
                        status.tcp_connections,
                        status.ws_port,
                        status.ws_connections
                    ));
                }
            }

            ui.separator();

            // --- Per-robot status ---
            egui::CollapsingHeader::new(format!("Robot Status ({})", robot_manager.robots.len()))
                .id_salt("agent_robot_status")
                .default_open(true)
                .show(ui, |ui| {
                    if robot_manager.robots.is_empty() {
                        ui.label("No robots active.");
                        return;
                    }
                    for (i, robot) in robot_manager.robots.iter().enumerate() {
                        let color = robot_color(i);
                        let connected = activity_log.is_connected(i);
                        let conn_label = if connected { "connected" } else { "idle" };
                        let conn_color = if connected {
                            egui::Color32::from_rgb(80, 220, 80)
                        } else {
                            egui::Color32::from_rgb(150, 150, 150)
                        };
                        ui.horizontal(|ui| {
                            ui.colored_label(color, format!("R{} {}", i, robot.definition.name));
                            ui.colored_label(conn_color, conn_label);
                        });

                        let steps = activity_log.step_counts.get(i).copied().unwrap_or(0);
                        let reward = activity_log.latest_rewards.get(i).copied().unwrap_or(0.0);
                        ui.label(format!("  Steps: {}  Reward: {:.3}", steps, reward));

                        // Show first few sensor readings inline
                        let max_show = 3.min(robot.state.sensor_readings.len());
                        for s_idx in 0..max_show {
                            let reading = &robot.state.sensor_readings[s_idx];
                            let sensor_type = robot
                                .definition
                                .sensors
                                .get(s_idx)
                                .map(|m| match &m.sensor {
                                    SensorDefinition::Distance { .. } => "Dist",
                                    SensorDefinition::Lidar { .. } => "Lidar",
                                    SensorDefinition::Contact => "Contact",
                                    SensorDefinition::Imu => "IMU",
                                })
                                .unwrap_or("?");
                            let val = match reading {
                                crate::robot::state::SensorReading::Distance(d) => {
                                    format!("{:.2}m", d)
                                }
                                crate::robot::state::SensorReading::Lidar(rays) => {
                                    format!("{} rays", rays.len())
                                }
                                crate::robot::state::SensorReading::Contact(c) => {
                                    format!("{}", c)
                                }
                                crate::robot::state::SensorReading::Imu {
                                    linear_accel, ..
                                } => {
                                    format!(
                                        "({:.1},{:.1},{:.1})",
                                        linear_accel.x, linear_accel.y, linear_accel.z
                                    )
                                }
                            };
                            ui.label(format!("    {sensor_type}: {val}"));
                        }
                        if robot.state.sensor_readings.len() > max_show {
                            ui.label(format!(
                                "    +{} more sensors",
                                robot.state.sensor_readings.len() - max_show
                            ));
                        }

                        ui.add_space(4.0);
                    }
                });

            ui.separator();

            // --- Command Log ---
            egui::CollapsingHeader::new(format!("Command Log ({})", activity_log.len()))
                .id_salt("agent_command_log")
                .default_open(true)
                .show(ui, |ui| {
                    if activity_log.is_empty() {
                        ui.label("No commands received yet.");
                        return;
                    }

                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            // Show newest events at the bottom (natural scroll)
                            for event in activity_log.iter() {
                                let event_color = match event.kind {
                                    AgentEventKind::Connect => egui::Color32::from_rgb(80, 200, 80),
                                    AgentEventKind::Step => egui::Color32::from_rgb(180, 180, 220),
                                    AgentEventKind::Observe => {
                                        egui::Color32::from_rgb(150, 200, 255)
                                    }
                                    AgentEventKind::Reset => egui::Color32::from_rgb(255, 200, 80),
                                    AgentEventKind::Remove => egui::Color32::from_rgb(200, 150, 80),
                                    AgentEventKind::Error => egui::Color32::from_rgb(255, 80, 80),
                                    AgentEventKind::Message => {
                                        egui::Color32::from_rgb(200, 150, 255)
                                    }
                                };
                                let robot_str = event
                                    .robot_id
                                    .map_or(String::new(), |id| format!("R{} ", id));
                                ui.colored_label(
                                    event_color,
                                    format!(
                                        "[{:.1}s] {}{}",
                                        event.timestamp, robot_str, event.description
                                    ),
                                );
                            }
                        });
                });
        });
}

pub fn settings_window(
    ctx: &egui::Context,
    open: &mut bool,
    sim: &mut SimulationState,
    fluid_sim: &mut FluidSimulation,
    gas_sim: &mut GasSimulation,
) {
    egui::Window::new("Simulation Settings")
        .open(open)
        .resizable(false)
        .show(ctx, |ui| {
            ui.heading("Acoustics");
            ui.add(
                egui::Slider::new(&mut sim.config.grid_resolution, 0.05..=2.0)
                    .text("Grid Resolution (m)")
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut sim.config.energy_threshold, 0.0001..=0.1)
                    .text("Energy Threshold")
                    .logarithmic(true),
            );

            ui.separator();
            ui.heading("Fluid");

            let old_resolution = fluid_sim.grid.as_ref().map(|g| g.dx);

            let mut grid_res = fluid_sim.grid.as_ref().map_or(0.1_f32, |g| g.dx);
            let res_changed = ui
                .add(egui::Slider::new(&mut grid_res, 0.05..=1.0).text("Grid Resolution (m)"))
                .changed();

            ui.add(egui::Slider::new(&mut fluid_sim.config.dt, 0.001..=0.1).text("Timestep (s)"));
            ui.add(egui::Slider::new(&mut fluid_sim.config.viscosity, 0.0..=1.0).text("Viscosity"));

            ui.label("Gravity:");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut fluid_sim.config.gravity.x)
                        .prefix("x: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut fluid_sim.config.gravity.y)
                        .prefix("y: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut fluid_sim.config.gravity.z)
                        .prefix("z: ")
                        .speed(0.1),
                );
            });

            let mut iters = fluid_sim.config.jacobi_iterations as i32;
            ui.add(egui::Slider::new(&mut iters, 10..=200).text("Jacobi Iterations"));
            fluid_sim.config.jacobi_iterations = iters as u32;

            // If grid resolution changed and a grid exists, trigger re-init
            if res_changed {
                if let Some(old_dx) = old_resolution {
                    if (grid_res - old_dx).abs() > 1e-6 {
                        if let Some(ref grid) = fluid_sim.grid {
                            let origin = grid.origin;
                            let extent = Vec3::new(
                                grid.nx as f32 * old_dx,
                                grid.ny as f32 * old_dx,
                                grid.nz as f32 * old_dx,
                            );
                            let bounds = (origin, origin + extent);
                            fluid_sim.initialize(bounds, grid_res, &[]);
                        }
                    }
                }
            }

            ui.separator();
            ui.heading("Gas");

            ui.add(egui::Slider::new(&mut gas_sim.config.dt, 0.001..=0.1).text("Timestep (s)"));
            ui.add(
                egui::Slider::new(&mut gas_sim.config.ambient_temperature, 200.0..=500.0)
                    .text("Ambient Temp (K)"),
            );
            ui.add(
                egui::Slider::new(&mut gas_sim.config.thermal_diffusivity, 0.0..=0.1)
                    .text("Thermal Diffusivity"),
            );
            ui.add(
                egui::Slider::new(&mut gas_sim.config.buoyancy_coefficient, 0.0..=1.0)
                    .text("Buoyancy Coefficient"),
            );

            ui.label("Gravity:");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut gas_sim.config.gravity.x)
                        .prefix("x: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut gas_sim.config.gravity.y)
                        .prefix("y: ")
                        .speed(0.1),
                );
                ui.add(
                    egui::DragValue::new(&mut gas_sim.config.gravity.z)
                        .prefix("z: ")
                        .speed(0.1),
                );
            });
        });
}

fn hit_test(
    screen_pos: egui::Pos2,
    scene: &Scene,
    robot_manager: &RobotManager,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
) -> Selection {
    let hit_radius = 14.0;

    for (i, source) in scene.sound_sources.iter().enumerate() {
        let p = project_3d(source.position, cam, center, scale);
        if p.distance(screen_pos) < hit_radius {
            return Selection::Source(i);
        }
    }

    for (i, listener) in scene.listeners.iter().enumerate() {
        let p = project_3d(listener.position, cam, center, scale);
        if p.distance(screen_pos) < hit_radius {
            return Selection::Listener(i);
        }
    }

    // Robot link hit-testing: pick closest link within radius (camera-depth aware).
    let mut best: Option<(f32, Selection)> = None;
    for (ri, robot) in robot_manager.robots.iter().enumerate() {
        let poses = robot.state.link_poses_as_mat4();
        for (li, pose) in poses.iter().enumerate() {
            let world = Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);
            let p = project_3d(world, cam, center, scale);
            let d = p.distance(screen_pos);
            if d < hit_radius * 1.3 {
                let depth = (world - cam.position).length();
                let cand = (depth, Selection::RobotLink(ri, li));
                if best.as_ref().is_none_or(|b| depth < b.0) {
                    best = Some(cand);
                }
            }
        }
    }
    if let Some((_, sel)) = best {
        return sel;
    }

    Selection::None
}

/// Lightweight hover hit test: returns (Selection, label) of closest pickable.
fn hover_label_test(
    screen_pos: egui::Pos2,
    scene: &Scene,
    robot_manager: &RobotManager,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
) -> Option<String> {
    let hit_radius = 16.0;
    for (i, source) in scene.sound_sources.iter().enumerate() {
        let p = project_3d(source.position, cam, center, scale);
        if p.distance(screen_pos) < hit_radius {
            return Some(format!("Source {}", i + 1));
        }
    }
    for listener in &scene.listeners {
        let p = project_3d(listener.position, cam, center, scale);
        if p.distance(screen_pos) < hit_radius {
            return Some(listener.name.clone());
        }
    }
    let mut best: Option<(f32, String)> = None;
    for (ri, robot) in robot_manager.robots.iter().enumerate() {
        let poses = robot.state.link_poses_as_mat4();
        for (li, pose) in poses.iter().enumerate() {
            let world = Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);
            let p = project_3d(world, cam, center, scale);
            if p.distance(screen_pos) < hit_radius * 1.3 {
                let depth = (world - cam.position).length();
                let link_name = robot
                    .definition
                    .links
                    .get(li)
                    .map(|l| l.name.as_str())
                    .unwrap_or("link");
                let lbl = format!("R{} {} · {}", ri, robot.definition.name, link_name);
                if best.as_ref().is_none_or(|b| depth < b.0) {
                    best = Some((depth, lbl));
                }
            }
        }
    }
    best.map(|(_, s)| s)
}

fn focus_on_scene(cam: &mut Camera, scene: &Scene) {
    if scene.meshes.is_empty() {
        return;
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for obj in &scene.meshes {
        let (obj_min, obj_max) = obj.mesh.bounds();
        min = min.min(obj_min);
        max = max.max(obj_max);
    }

    let center = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;
    cam.focus_on(center, radius);
}

/// Apply a [`PaletteAction`] picked by the user via the command palette.
///
/// Actions whose effects are local to viewport state (toggles, view changes,
/// scene mutations routed through `vp.history`) are handled here directly.
/// Actions that touch app-level state we don't have access to in viewport
/// (settings/about windows, sim run, scene reset) are recorded as
/// `vp.pending_palette_action` for the main update loop to drain.
fn dispatch_palette_action(action: PaletteAction, scene: &mut Scene, vp: &mut ViewportState) {
    use PaletteAction::*;
    match action {
        Undo => {
            if let Some(name) = vp.history.undo(scene) {
                vp.last_history_msg = Some(format!("Undid: {name}"));
                vp.selection = Selection::None;
            }
        }
        Redo => {
            if let Some(name) = vp.history.redo(scene) {
                vp.last_history_msg = Some(format!("Redid: {name}"));
            }
        }
        ResetCamera => {
            vp.camera = Camera::default();
            if !scene.meshes.is_empty() {
                focus_on_scene(&mut vp.camera, scene);
            }
        }
        FocusSelection => match vp.selection {
            Selection::Source(idx) if idx < scene.sound_sources.len() => {
                vp.camera
                    .smooth_focus(scene.sound_sources[idx].position, 1.5);
            }
            Selection::Listener(idx) if idx < scene.listeners.len() => {
                vp.camera.smooth_focus(scene.listeners[idx].position, 1.5);
            }
            _ => focus_on_scene(&mut vp.camera, scene),
        },
        SetView(view) => {
            vp.camera.set_view(view);
            vp.current_view = view;
        }
        ToggleFlyMode => vp.fly_mode = !vp.fly_mode,
        ToggleGrid => vp.show_grid = !vp.show_grid,
        ToggleShaded => vp.shaded = !vp.shaded,
        ToggleRays => vp.show_rays = !vp.show_rays,
        ToggleRobots => vp.show_robots = !vp.show_robots,
        ToggleSourcesVisibility => vp.show_sources = !vp.show_sources,
        ToggleListenersVisibility => vp.show_listeners = !vp.show_listeners,
        ToggleMeshesVisibility => vp.show_meshes = !vp.show_meshes,
        SetMode(m) => vp.mode = m,

        AddSource => {
            let _ = vp.history.push(
                SceneCommand::InsertSource {
                    idx: scene.sound_sources.len(),
                    src: SoundSource::default(),
                },
                scene,
            );
            vp.selection = Selection::Source(scene.sound_sources.len() - 1);
        }
        AddListener => {
            let n = scene.listeners.len() + 1;
            let listener = Listener {
                name: format!("Listener {n}"),
                ..Default::default()
            };
            let _ = vp.history.push(
                SceneCommand::InsertListener {
                    idx: scene.listeners.len(),
                    listener,
                },
                scene,
            );
            vp.selection = Selection::Listener(scene.listeners.len() - 1);
        }
        AddPartitionWall => {
            let obj =
                crate::scene::primitives::partition_wall(Vec3::new(2.0, 0.0, 1.0), 2.0, 2.5, 0.15);
            let _ = vp.history.push(
                SceneCommand::InsertObject {
                    idx: scene.meshes.len(),
                    obj,
                },
                scene,
            );
        }
        AddPlatform => {
            let obj = crate::scene::primitives::platform(Vec3::new(1.0, 0.0, 1.0), 2.0, 2.0, 0.5);
            let _ = vp.history.push(
                SceneCommand::InsertObject {
                    idx: scene.meshes.len(),
                    obj,
                },
                scene,
            );
        }
        DeleteSelected => match vp.selection {
            Selection::Source(idx) if idx < scene.sound_sources.len() => {
                let snap = scene.sound_sources[idx].clone();
                let _ = vp.history.push(
                    SceneCommand::RemoveSource {
                        idx,
                        snapshot: snap,
                    },
                    scene,
                );
                vp.selection = Selection::None;
            }
            Selection::Listener(idx) if idx < scene.listeners.len() => {
                let snap = scene.listeners[idx].clone();
                let _ = vp.history.push(
                    SceneCommand::RemoveListener {
                        idx,
                        snapshot: snap,
                    },
                    scene,
                );
                vp.selection = Selection::None;
            }
            Selection::Object(idx) if idx < scene.meshes.len() => {
                let snap = scene.meshes[idx].clone();
                let _ = vp.history.push(
                    SceneCommand::RemoveObject {
                        idx,
                        snapshot: snap,
                    },
                    scene,
                );
                vp.selection = Selection::None;
            }
            _ => {}
        },
        ToggleTeleop => {
            vp.teleop_mode = !vp.teleop_mode;
            if vp.teleop_mode {
                vp.fly_mode = false;
            }
        }
        // App-level actions — main update() drains pending_palette_action.
        NewScene | RunSimulation | ToggleSettings | ToggleAbout => {
            vp.pending_palette_action = Some(action);
        }
    }
}

// ---------------------------------------------------------------------------
// Boxing HUD overlays
// ---------------------------------------------------------------------------

fn render_health_bars(
    painter: &egui::Painter,
    robot_manager: &RobotManager,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    for robot in &robot_manager.robots {
        let combat = match &robot.state.combat {
            Some(c) => c,
            None => continue,
        };

        let link_poses = robot.state.link_poses_as_mat4();
        if link_poses.is_empty() {
            continue;
        }

        let base_pos = Vec3::new(
            link_poses[0].w_axis.x,
            link_poses[0].w_axis.y + 0.55,
            link_poses[0].w_axis.z,
        );
        let sp = project_3d(base_pos, cam, center, scale);
        if !rect.contains(sp) {
            continue;
        }

        let bar_w = 50.0;
        let bar_h = 6.0;
        let bar_x = sp.x - bar_w / 2.0;

        // Health bar background
        let bg_rect =
            egui::Rect::from_min_size(egui::pos2(bar_x, sp.y - bar_h), egui::vec2(bar_w, bar_h));
        painter.rect_filled(bg_rect, 2.0, egui::Color32::from_rgb(40, 40, 40));

        // Health fill
        let health_pct = (combat.health / combat.max_health).clamp(0.0, 1.0);
        let health_color = if health_pct > 0.5 {
            egui::Color32::from_rgb(80, 220, 80)
        } else if health_pct > 0.25 {
            egui::Color32::from_rgb(220, 200, 60)
        } else {
            egui::Color32::from_rgb(220, 60, 60)
        };
        let fill_rect = egui::Rect::from_min_size(
            egui::pos2(bar_x, sp.y - bar_h),
            egui::vec2(bar_w * health_pct, bar_h),
        );
        painter.rect_filled(fill_rect, 2.0, health_color);

        // Stamina bar (smaller, below health)
        let stam_y = sp.y + 1.0;
        let stam_h = 3.0;
        let stam_bg =
            egui::Rect::from_min_size(egui::pos2(bar_x, stam_y), egui::vec2(bar_w, stam_h));
        painter.rect_filled(stam_bg, 1.0, egui::Color32::from_rgb(30, 30, 30));

        let stam_pct = (combat.stamina / combat.max_stamina).clamp(0.0, 1.0);
        let stam_fill = egui::Rect::from_min_size(
            egui::pos2(bar_x, stam_y),
            egui::vec2(bar_w * stam_pct, stam_h),
        );
        painter.rect_filled(stam_fill, 1.0, egui::Color32::from_rgb(60, 150, 255));

        // Health text
        painter.text(
            egui::pos2(sp.x, sp.y - bar_h - 2.0),
            egui::Align2::CENTER_BOTTOM,
            format!("{:.0}", combat.health),
            egui::FontId::proportional(10.0),
            egui::Color32::WHITE,
        );
    }
}

fn render_boxing_score_overlay(
    painter: &egui::Painter,
    bridge: &SimBridgeClient,
    rect: egui::Rect,
) {
    let bm = match &bridge.boxing_match {
        Some(bm) => bm,
        None => return,
    };

    let snapshot = bm.snapshot(0, None);
    let phase_display = match snapshot.phase.as_str() {
        p if p.starts_with("countdown_") => {
            let secs = p.strip_prefix("countdown_").unwrap_or("?");
            format!("Countdown: {}", secs)
        }
        "fighting" => format!(
            "Round {} — {:.0}s",
            snapshot.current_round,
            snapshot.round_duration - snapshot.round_time
        ),
        p if p.starts_with("round_end_") => "Round End".to_string(),
        "match_end" => "Match Over".to_string(),
        "waiting_for_agents" => "Waiting for Agents...".to_string(),
        other => other.to_string(),
    };

    let top_center = egui::pos2(rect.center().x, rect.min.y + 15.0);

    // Phase banner
    painter.text(
        top_center,
        egui::Align2::CENTER_TOP,
        &phase_display,
        egui::FontId::proportional(16.0),
        egui::Color32::WHITE,
    );

    // Score
    let score_pos = egui::pos2(rect.center().x, rect.min.y + 35.0);
    let score_text = format!(
        "Robot A: {}  —  Robot B: {}",
        snapshot.total_score_a, snapshot.total_score_b
    );
    painter.text(
        score_pos,
        egui::Align2::CENTER_TOP,
        &score_text,
        egui::FontId::proportional(13.0),
        egui::Color32::from_rgb(200, 200, 200),
    );

    // Round scores
    if !snapshot.scores.is_empty() {
        let rounds_pos = egui::pos2(rect.center().x, rect.min.y + 52.0);
        let rounds: Vec<String> = snapshot
            .scores
            .iter()
            .enumerate()
            .map(|(i, s)| format!("R{}: {}-{}", i + 1, s[0], s[1]))
            .collect();
        painter.text(
            rounds_pos,
            egui::Align2::CENTER_TOP,
            rounds.join("  "),
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgb(150, 150, 150),
        );
    }
}

fn render_message_feed(
    painter: &egui::Painter,
    messages: &[(f32, String, usize)],
    elapsed: f32,
    rect: egui::Rect,
) {
    let feed_x = rect.min.x + 10.0;
    let feed_y = rect.max.y - 20.0;
    let max_age = 8.0;
    let max_show = 6;

    let recent: Vec<_> = messages
        .iter()
        .filter(|(t, _, _)| elapsed - t < max_age)
        .rev()
        .take(max_show)
        .collect();

    for (i, (timestamp, text, robot_id)) in recent.iter().enumerate() {
        let age = elapsed - timestamp;
        let alpha = ((1.0 - age / max_age) * 255.0) as u8;
        let color = if *robot_id == 0 {
            egui::Color32::from_rgba_premultiplied(80, 180, 255, alpha)
        } else {
            egui::Color32::from_rgba_premultiplied(255, 120, 80, alpha)
        };
        let y = feed_y - (i as f32) * 16.0;
        let label = format!("R{}: {}", robot_id, text);
        painter.text(
            egui::pos2(feed_x, y),
            egui::Align2::LEFT_BOTTOM,
            &label,
            egui::FontId::proportional(11.0),
            color,
        );
    }
}

fn render_hit_flash(painter: &egui::Painter, flash_timer: f32, rect: egui::Rect) {
    if flash_timer <= 0.0 {
        return;
    }
    let alpha = (flash_timer.min(1.0) * 80.0) as u8;
    let flash_color = egui::Color32::from_rgba_premultiplied(255, 50, 50, alpha);
    painter.rect_filled(rect, 0.0, flash_color);
}

// ---------------------------------------------------------------------------
// Robot rendering helpers
// ---------------------------------------------------------------------------

/// Distinct colors for up to 8 robots; wraps around for more.
const ROBOT_COLORS: [[u8; 3]; 8] = [
    [80, 180, 255],  // blue
    [255, 120, 80],  // orange
    [100, 220, 100], // green
    [220, 100, 220], // purple
    [255, 220, 80],  // yellow
    [80, 220, 220],  // cyan
    [255, 100, 150], // pink
    [180, 180, 100], // olive
];

/// Return the base color for a robot at the given index.
fn robot_color(robot_idx: usize) -> egui::Color32 {
    let c = ROBOT_COLORS[robot_idx % ROBOT_COLORS.len()];
    egui::Color32::from_rgb(c[0], c[1], c[2])
}

/// Render all robots in the viewport: links as wireframe shapes, joints as
/// spheres, and optionally sensor rays as colored lines.
#[allow(clippy::too_many_arguments)]
fn render_robots(
    painter: &egui::Painter,
    robot_manager: &RobotManager,
    show_sensor_rays: bool,
    shaded: bool,
    selection: Selection,
    activity_log: &AgentActivityLog,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    for (robot_idx, robot) in robot_manager.robots.iter().enumerate() {
        let color = robot_color(robot_idx);

        let link_poses = robot.state.link_poses_as_mat4();

        // Drop shadow under root link (y=0 plane)
        if shaded && !link_poses.is_empty() {
            let root_pos = Vec3::new(
                link_poses[0].w_axis.x,
                link_poses[0].w_axis.y,
                link_poses[0].w_axis.z,
            );
            let shadow_radius = (root_pos.y + 0.3).max(0.4);
            let shadow_pts = ground_shadow_polygon(root_pos, shadow_radius, cam, center, scale);
            painter.add(egui::Shape::convex_polygon(
                shadow_pts,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 70),
                egui::Stroke::NONE,
            ));
        }

        let robot_selected = selection == Selection::Robot(robot_idx);
        let selected_link = match selection {
            Selection::RobotLink(r, l) if r == robot_idx => Some(l),
            _ => None,
        };

        if robot.definition.name == "boxing_humanoid" && link_poses.len() >= 4 {
            render_boxing_humanoid(
                painter,
                robot,
                robot_idx,
                &link_poses,
                color,
                robot_selected || selected_link.is_some(),
                activity_log,
                cam,
                center,
                scale,
                rect,
            );
        } else {
            render_generic_robot(
                painter,
                robot,
                robot_idx,
                &link_poses,
                color,
                shaded,
                selected_link,
                activity_log,
                show_sensor_rays,
                cam,
                center,
                scale,
                rect,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn render_boxing_humanoid(
    painter: &egui::Painter,
    robot: &crate::robot::ManagedRobot,
    robot_idx: usize,
    link_poses: &[glam::Mat4],
    color: egui::Color32,
    selected: bool,
    activity_log: &AgentActivityLog,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let pos_of = |pose: glam::Mat4| Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);
    let proj = |p: Vec3| project_3d(p, cam, center, scale);

    let torso_pos = pos_of(link_poses[0]);
    let head_pos = pos_of(link_poses[1]);
    let left_fist_pos = pos_of(link_poses[2]);
    let right_fist_pos = pos_of(link_poses[3]);

    let head_sp = proj(head_pos);
    let left_fist_sp = proj(left_fist_pos);
    let right_fist_sp = proj(right_fist_pos);

    let fill = egui::Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 100);
    let bright = egui::Color32::from_rgb(
        color.r().saturating_add(40),
        color.g().saturating_add(40),
        color.b().saturating_add(40),
    );

    // Shoulder positions (on torso, offset toward each arm)
    let left_shoulder = torso_pos + Vec3::new(0.0, 0.15, 0.12);
    let right_shoulder = torso_pos + Vec3::new(0.0, 0.15, -0.12);
    let left_shoulder_sp = proj(left_shoulder);
    let right_shoulder_sp = proj(right_shoulder);
    let hip_pos = torso_pos - Vec3::new(0.0, 0.25, 0.0);
    let left_foot = hip_pos + Vec3::new(0.0, -0.4, 0.12);
    let right_foot = hip_pos + Vec3::new(0.0, -0.4, -0.12);
    let hip_sp = proj(hip_pos);
    let left_foot_sp = proj(left_foot);
    let right_foot_sp = proj(right_foot);

    let limb_stroke = egui::Stroke::new(4.0, color);
    let body_stroke = egui::Stroke::new(5.0, color);

    // Legs
    painter.line_segment([hip_sp, left_foot_sp], limb_stroke);
    painter.line_segment([hip_sp, right_foot_sp], limb_stroke);
    painter.circle_filled(left_foot_sp, 4.0, color);
    painter.circle_filled(right_foot_sp, 4.0, color);

    // Torso (thick line from hip to neck area)
    let neck_pos = torso_pos + Vec3::new(0.0, 0.3, 0.0);
    let neck_sp = proj(neck_pos);
    painter.line_segment([hip_sp, neck_sp], body_stroke);

    // Shoulder bar
    painter.line_segment([left_shoulder_sp, right_shoulder_sp], body_stroke);

    // Torso fill — rectangle between shoulders and hips
    let torso_rect_points = [
        proj(torso_pos + Vec3::new(0.0, 0.15, 0.15)),
        proj(torso_pos + Vec3::new(0.0, 0.15, -0.15)),
        proj(torso_pos + Vec3::new(0.0, -0.25, -0.12)),
        proj(torso_pos + Vec3::new(0.0, -0.25, 0.12)),
    ];
    let torso_mesh = egui::Shape::convex_polygon(
        torso_rect_points.to_vec(),
        fill,
        egui::Stroke::new(1.5, color),
    );
    painter.add(torso_mesh);

    // Arms — thick lines from shoulder to fist
    painter.line_segment([left_shoulder_sp, left_fist_sp], limb_stroke);
    painter.line_segment([right_shoulder_sp, right_fist_sp], limb_stroke);

    // Fists — larger filled circles
    let fist_radius = (0.15 * scale * 5.0).clamp(8.0, 28.0);
    painter.circle_filled(left_fist_sp, fist_radius, fill);
    painter.circle_stroke(left_fist_sp, fist_radius, egui::Stroke::new(2.5, bright));
    painter.circle_filled(right_fist_sp, fist_radius, fill);
    painter.circle_stroke(right_fist_sp, fist_radius, egui::Stroke::new(2.5, bright));

    // Neck line
    painter.line_segment([neck_sp, head_sp], egui::Stroke::new(3.0, color));

    // Head — filled circle
    let head_radius = (0.1 * scale * 5.0).clamp(8.0, 24.0);
    painter.circle_filled(head_sp, head_radius, fill);
    painter.circle_stroke(head_sp, head_radius, egui::Stroke::new(2.0, bright));

    // Eyes (two small dots based on facing direction)
    let face_dir = if robot_idx == 0 { 1.0_f32 } else { -1.0 };
    let eye_offset = Vec3::new(0.0, 0.02, 0.04 * face_dir);
    let left_eye_sp = proj(head_pos + eye_offset + Vec3::new(0.0, 0.0, -0.025 * face_dir));
    let right_eye_sp = proj(head_pos + eye_offset + Vec3::new(0.0, 0.0, 0.025 * face_dir));
    painter.circle_filled(left_eye_sp, 2.0, egui::Color32::WHITE);
    painter.circle_filled(right_eye_sp, 2.0, egui::Color32::WHITE);

    // Health bar above head
    if let Some(ref combat) = robot.state.combat {
        let bar_center = proj(head_pos + Vec3::new(0.0, 0.18, 0.0));
        let bar_w = 40.0_f32;
        let bar_h = 5.0_f32;
        let hp_frac = (combat.health / combat.max_health).clamp(0.0, 1.0);

        let bg_rect = egui::Rect::from_center_size(bar_center, egui::vec2(bar_w, bar_h));
        painter.rect_filled(bg_rect, 2.0, egui::Color32::from_rgb(60, 20, 20));

        let hp_color = if hp_frac > 0.5 {
            egui::Color32::from_rgb(60, 220, 60)
        } else if hp_frac > 0.25 {
            egui::Color32::from_rgb(220, 180, 40)
        } else {
            egui::Color32::from_rgb(220, 40, 40)
        };
        let hp_rect =
            egui::Rect::from_min_size(bg_rect.left_top(), egui::vec2(bar_w * hp_frac, bar_h));
        painter.rect_filled(hp_rect, 2.0, hp_color);
        painter.rect_stroke(
            bg_rect,
            2.0,
            egui::Stroke::new(1.0, egui::Color32::WHITE),
            egui::StrokeKind::Outside,
        );
    }

    // Name label above health bar
    let label_pos = proj(head_pos + Vec3::new(0.0, 0.26, 0.0));
    if rect.contains(label_pos) {
        let connected = activity_log.is_connected(robot_idx);
        let tag = if connected { "AI" } else { "--" };
        let label = format!("[{}] R{}", tag, robot_idx);
        painter.text(
            label_pos,
            egui::Align2::CENTER_BOTTOM,
            &label,
            egui::FontId::proportional(12.0),
            bright,
        );
    }

    // Selection halo around torso
    if selected {
        let halo_color = egui::Color32::from_rgb(255, 220, 80);
        let torso_screen = proj(torso_pos);
        let halo_radius = (0.6 * scale * 5.0).clamp(28.0, 90.0);
        painter.circle_stroke(
            torso_screen,
            halo_radius,
            egui::Stroke::new(2.5, halo_color),
        );
        painter.circle_stroke(
            torso_screen,
            halo_radius + 3.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 220, 80, 90)),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn render_generic_robot(
    painter: &egui::Painter,
    robot: &crate::robot::ManagedRobot,
    robot_idx: usize,
    link_poses: &[glam::Mat4],
    color: egui::Color32,
    shaded: bool,
    selected_link: Option<usize>,
    activity_log: &AgentActivityLog,
    show_sensor_rays: bool,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let joint_color = egui::Color32::from_rgb(
        (color.r() as u16 * 3 / 4) as u8 + 60,
        (color.g() as u16 * 3 / 4) as u8 + 60,
        (color.b() as u16 * 3 / 4) as u8 + 60,
    );

    for (link_idx, link_def) in robot.definition.links.iter().enumerate() {
        if link_idx >= link_poses.len() {
            break;
        }
        let pose = link_poses[link_idx];
        let is_selected_link = selected_link == Some(link_idx);
        let link_color = if is_selected_link {
            egui::Color32::from_rgb(255, 220, 80)
        } else {
            color
        };
        render_link_shape(
            painter,
            &link_def.collision_shape,
            pose,
            link_color,
            shaded,
            cam,
            center,
            scale,
            rect,
        );
        if is_selected_link {
            // Outline ring around link center
            let p = project_3d(
                Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z),
                cam,
                center,
                scale,
            );
            painter.circle_stroke(
                p,
                18.0,
                egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 220, 80)),
            );
        }
    }

    for joint_def in &robot.definition.joints {
        if joint_def.child_link >= link_poses.len() {
            continue;
        }
        let child_pose = link_poses[joint_def.child_link];
        let joint_pos = Vec3::new(
            child_pose.w_axis.x,
            child_pose.w_axis.y,
            child_pose.w_axis.z,
        );
        let sp = project_3d(joint_pos, cam, center, scale);
        if rect.contains(sp) {
            painter.circle_filled(sp, 4.0, joint_color);
            painter.circle_stroke(sp, 4.0, egui::Stroke::new(1.0, color));
        }
    }

    if show_sensor_rays {
        let sensor_color =
            egui::Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 120);
        render_sensor_rays(painter, robot, sensor_color, cam, center, scale);
    }

    if !link_poses.is_empty() {
        let base_pos = Vec3::new(
            link_poses[0].w_axis.x,
            link_poses[0].w_axis.y + 0.3,
            link_poses[0].w_axis.z,
        );
        let lp = project_3d(base_pos, cam, center, scale);
        if rect.contains(lp) {
            let connected = activity_log.is_connected(robot_idx);
            let status_icon = if connected { "[A]" } else { "[-]" };
            let status_color = if connected {
                egui::Color32::from_rgb(80, 220, 80)
            } else {
                egui::Color32::from_rgb(150, 150, 150)
            };

            let label = format!("{} {} R{}", status_icon, robot.definition.name, robot_idx);
            painter.text(
                lp,
                egui::Align2::CENTER_BOTTOM,
                &label,
                egui::FontId::proportional(11.0),
                color,
            );
            let dot_pos = egui::Pos2::new(lp.x - 30.0, lp.y - 4.0);
            painter.circle_filled(dot_pos, 3.0, status_color);
        }
    }
}

/// Draw a single link's collision shape at the given world pose.
/// When `shaded`, draws filled Lambert-shaded faces with depth-sort + outline.
#[allow(clippy::too_many_arguments)]
fn render_link_shape(
    painter: &egui::Painter,
    shape: &CollisionShape,
    pose: glam::Mat4,
    color: egui::Color32,
    shaded: bool,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let origin = Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);

    match shape {
        CollisionShape::Cuboid { half_extents } => {
            if shaded {
                draw_shaded_cuboid(
                    painter,
                    pose,
                    *half_extents,
                    color,
                    cam,
                    center,
                    scale,
                    rect,
                );
            } else {
                draw_wireframe_cuboid(
                    painter,
                    pose,
                    *half_extents,
                    color,
                    cam,
                    center,
                    scale,
                    rect,
                );
            }
        }
        CollisionShape::Cylinder { radius, height } => {
            if shaded {
                draw_shaded_cylinder(
                    painter, pose, *radius, *height, color, cam, center, scale, rect,
                );
            } else {
                draw_wireframe_cylinder(
                    painter, pose, *radius, *height, color, cam, center, scale, rect,
                );
            }
        }
        CollisionShape::Sphere { radius } => {
            let sp = project_3d(origin, cam, center, scale);
            if rect.contains(sp) {
                let screen_radius = (*radius * scale * 5.0).clamp(3.0, 40.0);
                if shaded {
                    // Approx a sphere with a radial gradient via 3 concentric disks.
                    let lit = shade_color(color, Vec3::Y, scene_light_dir(), 0.4);
                    let mid = shade_color(color, Vec3::ZERO, scene_light_dir(), 0.25);
                    painter.circle_filled(sp, screen_radius, mid);
                    painter.circle_filled(
                        sp + egui::vec2(-screen_radius * 0.25, -screen_radius * 0.3),
                        screen_radius * 0.55,
                        lit,
                    );
                    painter.circle_stroke(sp, screen_radius, egui::Stroke::new(1.0, color));
                } else {
                    painter.circle_stroke(sp, screen_radius, egui::Stroke::new(1.5, color));
                }
            }
        }
    }
}

/// Draw a cuboid with Lambert-shaded face fills (back-to-front depth-sorted) and outline.
#[allow(clippy::too_many_arguments)]
fn draw_shaded_cuboid(
    painter: &egui::Painter,
    pose: glam::Mat4,
    half: Vec3,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let corners_local = [
        Vec3::new(-half.x, -half.y, -half.z),
        Vec3::new(half.x, -half.y, -half.z),
        Vec3::new(half.x, half.y, -half.z),
        Vec3::new(-half.x, half.y, -half.z),
        Vec3::new(-half.x, -half.y, half.z),
        Vec3::new(half.x, -half.y, half.z),
        Vec3::new(half.x, half.y, half.z),
        Vec3::new(-half.x, half.y, half.z),
    ];
    let world_corners: [Vec3; 8] = std::array::from_fn(|i| pose.transform_point3(corners_local[i]));
    let screen_corners: [egui::Pos2; 8] =
        std::array::from_fn(|i| project_3d(world_corners[i], cam, center, scale));

    if !screen_corners
        .iter()
        .any(|p| rect.expand(50.0).contains(*p))
    {
        return;
    }

    // Faces: (vertex idxs in CCW outward order, local normal)
    let faces: [([usize; 4], Vec3); 6] = [
        ([0, 3, 2, 1], Vec3::new(0.0, 0.0, -1.0)), // -Z
        ([4, 5, 6, 7], Vec3::new(0.0, 0.0, 1.0)),  // +Z
        ([0, 4, 7, 3], Vec3::new(-1.0, 0.0, 0.0)), // -X
        ([1, 2, 6, 5], Vec3::new(1.0, 0.0, 0.0)),  // +X
        ([0, 1, 5, 4], Vec3::new(0.0, -1.0, 0.0)), // -Y
        ([3, 7, 6, 2], Vec3::new(0.0, 1.0, 0.0)),  // +Y
    ];

    let light = scene_light_dir();
    let view_origin = cam.position;
    let mut draw_list: Vec<(f32, Vec<egui::Pos2>, egui::Color32)> = Vec::with_capacity(6);

    for (verts, normal_local) in faces {
        let face_center_world = verts
            .iter()
            .map(|&i| world_corners[i])
            .fold(Vec3::ZERO, |a, b| a + b)
            / 4.0;
        let to_cam = (view_origin - face_center_world).normalize_or_zero();
        let normal_world = pose.transform_vector3(normal_local).normalize_or_zero();
        // Back-face cull
        if normal_world.dot(to_cam) <= 0.02 {
            continue;
        }
        let depth = (face_center_world - view_origin).length();
        let pts: Vec<egui::Pos2> = verts.iter().map(|&i| screen_corners[i]).collect();
        let fill = shade_color(color, normal_world, light, 0.28);
        draw_list.push((depth, pts, fill));
    }
    draw_list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let outline =
        egui::Color32::from_rgba_unmultiplied(color.r() / 2, color.g() / 2, color.b() / 2, 220);
    for (_, pts, fill) in draw_list {
        painter.add(egui::Shape::convex_polygon(
            pts,
            fill,
            egui::Stroke::new(1.0, outline),
        ));
    }
}

/// Draw a cylinder with Lambert-shaded side quads + end caps.
#[allow(clippy::too_many_arguments)]
fn draw_shaded_cylinder(
    painter: &egui::Painter,
    pose: glam::Mat4,
    radius: f32,
    height: f32,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let half_h = height / 2.0;
    let segments = 16usize;
    let mut bottom_w = Vec::with_capacity(segments);
    let mut top_w = Vec::with_capacity(segments);
    let mut bottom_sp = Vec::with_capacity(segments);
    let mut top_sp = Vec::with_capacity(segments);
    for i in 0..segments {
        let a = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let x = radius * a.cos();
        let z = radius * a.sin();
        let bw = pose.transform_point3(Vec3::new(x, -half_h, z));
        let tw = pose.transform_point3(Vec3::new(x, half_h, z));
        bottom_w.push(bw);
        top_w.push(tw);
        bottom_sp.push(project_3d(bw, cam, center, scale));
        top_sp.push(project_3d(tw, cam, center, scale));
    }
    if !bottom_sp
        .iter()
        .chain(top_sp.iter())
        .any(|p| rect.expand(50.0).contains(*p))
    {
        return;
    }

    let light = scene_light_dir();
    let view_origin = cam.position;
    let mut draw_list: Vec<(f32, Vec<egui::Pos2>, egui::Color32)> =
        Vec::with_capacity(segments + 2);

    // Side quads
    for i in 0..segments {
        let j = (i + 1) % segments;
        let mid_w = (bottom_w[i] + bottom_w[j] + top_w[i] + top_w[j]) * 0.25;
        let outward_local = Vec3::new(
            (bottom_w[i] - pose.w_axis.truncate()).x,
            0.0,
            (bottom_w[i] - pose.w_axis.truncate()).z,
        );
        let normal = (mid_w - Vec3::new(pose.w_axis.x, mid_w.y, pose.w_axis.z)).normalize_or_zero();
        let _ = outward_local; // suppress; normal already in world
        let to_cam = (view_origin - mid_w).normalize_or_zero();
        if normal.dot(to_cam) <= 0.0 {
            continue;
        }
        let depth = (mid_w - view_origin).length();
        let pts = vec![bottom_sp[i], bottom_sp[j], top_sp[j], top_sp[i]];
        let fill = shade_color(color, normal, light, 0.28);
        draw_list.push((depth, pts, fill));
    }

    // Caps
    let cap_normal_top = pose.transform_vector3(Vec3::Y).normalize_or_zero();
    let top_center_w = pose.transform_point3(Vec3::new(0.0, half_h, 0.0));
    if cap_normal_top.dot((view_origin - top_center_w).normalize_or_zero()) > 0.0 {
        let depth = (top_center_w - view_origin).length();
        let fill = shade_color(color, cap_normal_top, light, 0.28);
        draw_list.push((depth, top_sp.clone(), fill));
    }
    let bottom_center_w = pose.transform_point3(Vec3::new(0.0, -half_h, 0.0));
    let cap_normal_bot = -cap_normal_top;
    if cap_normal_bot.dot((view_origin - bottom_center_w).normalize_or_zero()) > 0.0 {
        let depth = (bottom_center_w - view_origin).length();
        let fill = shade_color(color, cap_normal_bot, light, 0.28);
        // reverse for CCW
        let mut pts = bottom_sp.clone();
        pts.reverse();
        draw_list.push((depth, pts, fill));
    }

    draw_list.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let outline =
        egui::Color32::from_rgba_unmultiplied(color.r() / 2, color.g() / 2, color.b() / 2, 220);
    for (_, pts, fill) in draw_list {
        painter.add(egui::Shape::convex_polygon(
            pts,
            fill,
            egui::Stroke::new(0.8, outline),
        ));
    }
}

/// Draw an axis-aligned cuboid wireframe transformed by `pose`.
#[allow(clippy::too_many_arguments)]
fn draw_wireframe_cuboid(
    painter: &egui::Painter,
    pose: glam::Mat4,
    half: Vec3,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    // 8 corners of the cuboid in local space
    let corners_local = [
        Vec3::new(-half.x, -half.y, -half.z),
        Vec3::new(half.x, -half.y, -half.z),
        Vec3::new(half.x, half.y, -half.z),
        Vec3::new(-half.x, half.y, -half.z),
        Vec3::new(-half.x, -half.y, half.z),
        Vec3::new(half.x, -half.y, half.z),
        Vec3::new(half.x, half.y, half.z),
        Vec3::new(-half.x, half.y, half.z),
    ];

    // Transform to world space
    let corners: Vec<egui::Pos2> = corners_local
        .iter()
        .map(|c| {
            let world = pose.transform_point3(*c);
            project_3d(world, cam, center, scale)
        })
        .collect();

    // Check if any corner is in view
    if !corners.iter().any(|c| rect.contains(*c)) {
        return;
    }

    // 12 edges of a cuboid
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // vertical edges
    ];

    let stroke = egui::Stroke::new(1.5, color);
    for (a, b) in edges {
        painter.line_segment([corners[a], corners[b]], stroke);
    }
}

/// Draw a cylinder wireframe (two ellipses + vertical lines) transformed by `pose`.
#[allow(clippy::too_many_arguments)]
fn draw_wireframe_cylinder(
    painter: &egui::Painter,
    pose: glam::Mat4,
    radius: f32,
    height: f32,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let half_h = height / 2.0;
    let segments = 12;
    let stroke = egui::Stroke::new(1.5, color);

    // Generate circle points at bottom and top
    let mut bottom_pts = Vec::with_capacity(segments);
    let mut top_pts = Vec::with_capacity(segments);

    for i in 0..segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let local_x = radius * angle.cos();
        let local_z = radius * angle.sin();

        let bottom_local = Vec3::new(local_x, -half_h, local_z);
        let top_local = Vec3::new(local_x, half_h, local_z);

        let bottom_world = pose.transform_point3(bottom_local);
        let top_world = pose.transform_point3(top_local);

        bottom_pts.push(project_3d(bottom_world, cam, center, scale));
        top_pts.push(project_3d(top_world, cam, center, scale));
    }

    // Check if any point is in view
    let in_view = bottom_pts
        .iter()
        .chain(top_pts.iter())
        .any(|p| rect.contains(*p));
    if !in_view {
        return;
    }

    // Draw bottom and top circles
    for i in 0..segments {
        let next = (i + 1) % segments;
        painter.line_segment([bottom_pts[i], bottom_pts[next]], stroke);
        painter.line_segment([top_pts[i], top_pts[next]], stroke);
    }

    // Draw vertical lines at 4 cardinal points
    for i in (0..segments).step_by(segments / 4) {
        painter.line_segment([bottom_pts[i], top_pts[i]], stroke);
    }
}

/// Draw sensor rays as lines from sensor world position along sensor direction.
fn render_sensor_rays(
    painter: &egui::Painter,
    robot: &crate::robot::ManagedRobot,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
) {
    for (sensor_idx, mount) in robot.definition.sensors.iter().enumerate() {
        let (world_pos, world_dir) = sensor_world_pose(mount, &robot.state);

        match &mount.sensor {
            SensorDefinition::Distance { max_range, .. } => {
                // Use actual reading if available, else max_range
                let dist = match robot.state.sensor_readings.get(sensor_idx) {
                    Some(crate::robot::state::SensorReading::Distance(d)) => *d,
                    _ => *max_range,
                };
                let end = world_pos + world_dir * dist;
                let p1 = project_3d(world_pos, cam, center, scale);
                let p2 = project_3d(end, cam, center, scale);
                painter.line_segment([p1, p2], egui::Stroke::new(1.0, color));

                // Draw hit indicator if distance < max_range
                if dist < *max_range - 0.01 {
                    let hit_color = egui::Color32::from_rgb(255, 60, 60);
                    painter.circle_filled(p2, 3.0, hit_color);
                }
            }
            SensorDefinition::Lidar {
                num_rays,
                fov_rad,
                max_range,
            } => {
                let readings = match robot.state.sensor_readings.get(sensor_idx) {
                    Some(crate::robot::state::SensorReading::Lidar(rays)) => Some(rays.as_slice()),
                    _ => None,
                };

                // Draw fan of rays
                let num = *num_rays;
                if num == 0 {
                    continue;
                }
                let half_fov = fov_rad / 2.0;
                // Compute a perpendicular axis for the fan plane
                let perp = if world_dir.cross(Vec3::Y).length() > 0.01 {
                    world_dir.cross(Vec3::Y).normalize()
                } else {
                    world_dir.cross(Vec3::X).normalize()
                };

                let faded =
                    egui::Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 60);

                for ray_i in 0..num {
                    let t = if num > 1 {
                        ray_i as f32 / (num - 1) as f32
                    } else {
                        0.5
                    };
                    let angle = -half_fov + t * fov_rad;
                    let rot = glam::Quat::from_axis_angle(perp, angle);
                    let ray_dir = rot.mul_vec3(world_dir);

                    let dist = readings
                        .and_then(|r| r.get(ray_i).copied())
                        .unwrap_or(*max_range);
                    let end = world_pos + ray_dir * dist;

                    let p1 = project_3d(world_pos, cam, center, scale);
                    let p2 = project_3d(end, cam, center, scale);
                    painter.line_segment([p1, p2], egui::Stroke::new(0.5, faded));
                }
            }
            // Contact and IMU sensors have no visual ray
            SensorDefinition::Contact | SensorDefinition::Imu => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Environment interaction visualization
// ---------------------------------------------------------------------------

/// Draw contact force arrows, fluid disturbance indicators, and gas
/// displacement effects near robot links.
#[allow(clippy::too_many_arguments)]
fn render_environment_interactions(
    painter: &egui::Painter,
    robot_manager: &RobotManager,
    scene: &Scene,
    fluid_sim: &crate::fluids::FluidSimulation,
    gas_sim: &GasSimulation,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    for (robot_idx, robot) in robot_manager.robots.iter().enumerate() {
        let link_poses = robot.state.link_poses_as_mat4();
        let color = robot_color(robot_idx);

        for (link_idx, link_def) in robot.definition.links.iter().enumerate() {
            if link_idx >= link_poses.len() {
                break;
            }
            let pose = link_poses[link_idx];
            let link_pos = Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);

            // --- Contact force arrows ---
            render_contact_arrows(
                painter,
                link_pos,
                &link_def.collision_shape,
                scene,
                cam,
                center,
                scale,
                rect,
            );

            // --- Fluid disturbance indicators ---
            if let Some(ref grid) = fluid_sim.grid {
                render_fluid_disturbance(painter, link_pos, grid, color, cam, center, scale, rect);
            }

            // --- Gas displacement visualization ---
            if let Some(ref grid) = gas_sim.grid {
                render_gas_displacement(painter, link_pos, grid, cam, center, scale, rect);
            }
        }
    }
}

/// Draw contact force arrows when a robot link is near scene geometry.
///
/// For each scene triangle, if the link center is within a threshold
/// distance, draw an arrow from the triangle surface toward the link
/// center, indicating a repulsion/contact force.
#[allow(clippy::too_many_arguments)]
fn render_contact_arrows(
    painter: &egui::Painter,
    link_pos: Vec3,
    shape: &CollisionShape,
    scene: &Scene,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let contact_radius = match shape {
        CollisionShape::Sphere { radius } => *radius,
        CollisionShape::Cuboid { half_extents } => half_extents.length(),
        CollisionShape::Cylinder { radius, height } => {
            (radius * radius + (height / 2.0).powi(2)).sqrt()
        }
    };

    // Extend the detection radius slightly for visual feedback.
    let detect_radius = contact_radius * 1.5;
    let arrow_color = egui::Color32::from_rgb(255, 80, 60);

    for obj in &scene.meshes {
        if !obj.visible {
            continue;
        }
        for tri in &obj.mesh.triangles {
            // Use triangle centroid as proxy for closest point.
            let centroid =
                (tri.vertices[0].position + tri.vertices[1].position + tri.vertices[2].position)
                    / 3.0;
            let diff = link_pos - centroid;
            let dist = diff.length();

            if dist < detect_radius && dist > 1e-6 {
                // Draw an arrow from centroid toward link (contact normal direction).
                let arrow_dir = diff / dist;
                let arrow_len = (detect_radius - dist).min(0.3);
                let arrow_end = centroid + arrow_dir * arrow_len;

                let p1 = project_3d(centroid, cam, center, scale);
                let p2 = project_3d(arrow_end, cam, center, scale);

                if rect.contains(p1) || rect.contains(p2) {
                    // Arrow shaft.
                    let intensity = 1.0 - (dist / detect_radius);
                    let alpha = (intensity * 200.0) as u8;
                    let c = egui::Color32::from_rgba_premultiplied(
                        arrow_color.r(),
                        arrow_color.g(),
                        arrow_color.b(),
                        alpha,
                    );
                    painter.line_segment([p1, p2], egui::Stroke::new(2.0, c));

                    // Arrowhead (small triangle).
                    painter.circle_filled(p2, 3.0, c);
                }
            }
        }
    }
}

/// Draw fluid disturbance indicators near a robot link.
///
/// Samples the fluid velocity field at the link's grid cell and nearby
/// cells, drawing small velocity arrows to show fluid being disturbed.
#[allow(clippy::too_many_arguments)]
fn render_fluid_disturbance(
    painter: &egui::Painter,
    link_pos: Vec3,
    grid: &crate::fluids::grid::FluidGrid,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    // Convert link position to grid coordinates.
    let rel = link_pos - grid.origin;
    let gi = (rel.x / grid.dx) as i32;
    let gj = (rel.y / grid.dx) as i32;
    let gk = (rel.z / grid.dx) as i32;

    let nx = grid.nx as i32;
    let ny = grid.ny as i32;
    let nz = grid.nz as i32;

    // Sample a 3x3x3 neighborhood around the link cell.
    let arrow_color = egui::Color32::from_rgba_premultiplied(
        color.r() / 2 + 60,
        color.g() / 2 + 100,
        color.b() / 2 + 120,
        140,
    );

    for di in -1..=1 {
        for dj in -1..=1 {
            for dk in -1..=1 {
                let ci = gi + di;
                let cj = gj + dj;
                let ck = gk + dk;

                if ci < 0 || cj < 0 || ck < 0 || ci >= nx || cj >= ny || ck >= nz {
                    continue;
                }

                let ci = ci as usize;
                let cj = cj as usize;
                let ck = ck as usize;

                // Sample cell-centered velocity (average of face values).
                let u_avg = sample_u(grid, ci, cj, ck);
                let v_avg = sample_v(grid, ci, cj, ck);
                let w_avg = sample_w(grid, ci, cj, ck);

                let speed = (u_avg * u_avg + v_avg * v_avg + w_avg * w_avg).sqrt();
                if speed < 0.01 {
                    continue;
                }

                // World position of cell center.
                let cell_pos = grid.origin
                    + Vec3::new(
                        (ci as f32 + 0.5) * grid.dx,
                        (cj as f32 + 0.5) * grid.dx,
                        (ck as f32 + 0.5) * grid.dx,
                    );

                let vel = Vec3::new(u_avg, v_avg, w_avg);
                let arrow_len = (speed * 0.5).min(grid.dx * 2.0);
                let end = cell_pos + vel.normalize() * arrow_len;

                let p1 = project_3d(cell_pos, cam, center, scale);
                let p2 = project_3d(end, cam, center, scale);

                if rect.contains(p1) || rect.contains(p2) {
                    painter.line_segment([p1, p2], egui::Stroke::new(1.5, arrow_color));
                    painter.circle_filled(p2, 2.0, arrow_color);
                }
            }
        }
    }
}

/// Sample cell-centered x-velocity by averaging adjacent face values.
fn sample_u(grid: &crate::fluids::grid::FluidGrid, i: usize, j: usize, k: usize) -> f32 {
    let idx_lo = i * grid.ny * grid.nz + j * grid.nz + k;
    let idx_hi = (i + 1) * grid.ny * grid.nz + j * grid.nz + k;
    let u_lo = grid.u.get(idx_lo).copied().unwrap_or(0.0);
    let u_hi = grid.u.get(idx_hi).copied().unwrap_or(0.0);
    (u_lo + u_hi) * 0.5
}

/// Sample cell-centered y-velocity by averaging adjacent face values.
fn sample_v(grid: &crate::fluids::grid::FluidGrid, i: usize, j: usize, k: usize) -> f32 {
    let idx_lo = i * (grid.ny + 1) * grid.nz + j * grid.nz + k;
    let idx_hi = i * (grid.ny + 1) * grid.nz + (j + 1) * grid.nz + k;
    let v_lo = grid.v.get(idx_lo).copied().unwrap_or(0.0);
    let v_hi = grid.v.get(idx_hi).copied().unwrap_or(0.0);
    (v_lo + v_hi) * 0.5
}

/// Sample cell-centered z-velocity by averaging adjacent face values.
fn sample_w(grid: &crate::fluids::grid::FluidGrid, i: usize, j: usize, k: usize) -> f32 {
    let idx_lo = i * grid.ny * (grid.nz + 1) + j * (grid.nz + 1) + k;
    let idx_hi = i * grid.ny * (grid.nz + 1) + j * (grid.nz + 1) + (k + 1);
    let w_lo = grid.w.get(idx_lo).copied().unwrap_or(0.0);
    let w_hi = grid.w.get(idx_hi).copied().unwrap_or(0.0);
    (w_lo + w_hi) * 0.5
}

/// Draw gas displacement visualization near a robot link.
///
/// Shows concentration gradient disturbance as small colored circles
/// where the gas density differs significantly from average.
#[allow(clippy::too_many_arguments)]
fn render_gas_displacement(
    painter: &egui::Painter,
    link_pos: Vec3,
    grid: &crate::gas::grid::GasGrid,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let rel = link_pos - grid.origin;
    let gi = (rel.x / grid.dx) as i32;
    let gj = (rel.y / grid.dx) as i32;
    let gk = (rel.z / grid.dx) as i32;

    let nx = grid.nx as i32;
    let ny = grid.ny as i32;
    let nz = grid.nz as i32;

    if grid.concentrations.is_empty() {
        return;
    }

    // Use first species for visualization.
    let conc = &grid.concentrations[0];

    // Sample 3x3x3 neighborhood and compute average concentration.
    let mut sum = 0.0_f32;
    let mut count = 0_u32;
    let mut cells: Vec<(usize, usize, usize)> = Vec::new();

    for di in -1..=1 {
        for dj in -1..=1 {
            for dk in -1..=1 {
                let ci = gi + di;
                let cj = gj + dj;
                let ck = gk + dk;
                if ci < 0 || cj < 0 || ck < 0 || ci >= nx || cj >= ny || ck >= nz {
                    continue;
                }
                let ci = ci as usize;
                let cj = cj as usize;
                let ck = ck as usize;
                let idx = ci * grid.ny * grid.nz + cj * grid.nz + ck;
                if let Some(&c) = conc.get(idx) {
                    sum += c;
                    count += 1;
                    cells.push((ci, cj, ck));
                }
            }
        }
    }

    if count == 0 {
        return;
    }

    let avg = sum / count as f32;
    if avg < 1e-6 {
        return;
    }

    // Draw indicators where concentration deviates from average.
    for (ci, cj, ck) in cells {
        let idx = ci * grid.ny * grid.nz + cj * grid.nz + ck;
        let c = conc.get(idx).copied().unwrap_or(0.0);
        let deviation = (c - avg).abs() / avg;

        if deviation < 0.05 {
            continue;
        }

        let cell_pos = grid.origin
            + Vec3::new(
                (ci as f32 + 0.5) * grid.dx,
                (cj as f32 + 0.5) * grid.dx,
                (ck as f32 + 0.5) * grid.dx,
            );

        let p = project_3d(cell_pos, cam, center, scale);
        if !rect.contains(p) {
            continue;
        }

        // Color: green = higher concentration, yellow = lower.
        let intensity = deviation.min(1.0);
        let gas_color = if c > avg {
            egui::Color32::from_rgba_premultiplied(
                80,
                (180.0 + intensity * 75.0) as u8,
                80,
                (100.0 + intensity * 100.0) as u8,
            )
        } else {
            egui::Color32::from_rgba_premultiplied(
                (200.0 + intensity * 55.0) as u8,
                (200.0 + intensity * 55.0) as u8,
                60,
                (80.0 + intensity * 100.0) as u8,
            )
        };

        let radius = 2.0 + intensity * 4.0;
        painter.circle_filled(p, radius, gas_color);
    }
}

/// Canonical groups rendered in the left side panel.
/// Drives the goal-007 `side_panel_groups` test and keeps the docs walkthrough
/// truthful when new panes are added.
pub const SIDE_PANEL_GROUPS: &[&str] = &[
    "Scene",
    "Sources",
    "Listeners",
    "Materials",
    "Simulation Config",
    "Results",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::SimulationState;

    #[test]
    fn side_panel_groups_canonical_list() {
        for needle in [
            "Scene",
            "Sources",
            "Listeners",
            "Materials",
            "Simulation Config",
            "Results",
        ] {
            assert!(
                SIDE_PANEL_GROUPS.contains(&needle),
                "SIDE_PANEL_GROUPS missing {needle}"
            );
        }
        assert_eq!(SIDE_PANEL_GROUPS.len(), 6);
    }

    #[test]
    fn status_bar_app_status_severity_default_is_info() {
        let s = AppStatus::default();
        assert_eq!(s.severity, StatusSeverity::Info);
        assert!(s.message.is_empty());
    }

    #[test]
    fn status_bar_error_message_paints_red() {
        let mut s = AppStatus::default();
        s.error("boom");
        assert_eq!(s.severity, StatusSeverity::Error);
        assert_eq!(s.message, "boom");
        let c = s.color();
        // Red dominant for errors.
        assert!(c.r() > c.g() && c.r() > c.b());
    }

    #[test]
    fn status_bar_progress_fraction_is_clamped() {
        let sim = SimulationState::default();
        let total = sim.config.ray_count.max(1) as f32;
        let raw_progress: f32 = 1.5;
        let done = (raw_progress.clamp(0.0, 1.0) * total) as u32;
        assert_eq!(done, sim.config.ray_count);
    }

    #[test]
    fn camera_shortcuts_shift_num1_is_front() {
        assert_eq!(
            camera_view_for_shortcut(true, egui::Key::Num1),
            Some(CameraView::Front)
        );
        assert_eq!(
            camera_view_for_shortcut(true, egui::Key::Num2),
            Some(CameraView::Top)
        );
        assert_eq!(
            camera_view_for_shortcut(true, egui::Key::Num3),
            Some(CameraView::Side)
        );
    }

    #[test]
    fn camera_shortcuts_no_shift_returns_none() {
        assert_eq!(camera_view_for_shortcut(false, egui::Key::Num1), None);
        assert_eq!(camera_view_for_shortcut(false, egui::Key::Num2), None);
    }

    #[test]
    fn camera_shortcuts_other_keys_are_none() {
        assert_eq!(camera_view_for_shortcut(true, egui::Key::Q), None);
        assert_eq!(camera_view_for_shortcut(true, egui::Key::A), None);
    }

    /// Lower bound for tooltips ensures the UI explains itself.
    /// Counts every `.on_hover_text(` in this file — should grow over time, not shrink.
    #[test]
    fn tooltips_minimum_count() {
        let src = include_str!("mod.rs");
        let count = src.matches(".on_hover_text(").count();
        assert!(
            count >= 30,
            "expected >= 30 on_hover_text calls in src/ui/mod.rs, found {count}"
        );
    }

    // ---- Right-click context menu ----

    #[test]
    fn context_menu_no_selection_offers_scene_level_actions() {
        let items = context_menu_items_for(Selection::None);
        assert!(!items.is_empty());
        let labels: Vec<&str> = items.iter().map(|i| i.label).collect();
        assert!(labels.contains(&"Add Source"));
        assert!(labels.contains(&"Add Listener"));
        assert!(labels.contains(&"Reset Camera"));
    }

    #[test]
    fn context_menu_source_offers_focus_and_delete() {
        let items = context_menu_items_for(Selection::Source(0));
        let labels: Vec<&str> = items.iter().map(|i| i.label).collect();
        assert!(labels.contains(&"Focus"));
        assert!(labels.contains(&"Delete"));
    }

    #[test]
    fn context_menu_listener_offers_focus_and_delete() {
        let items = context_menu_items_for(Selection::Listener(2));
        let labels: Vec<&str> = items.iter().map(|i| i.label).collect();
        assert!(labels.contains(&"Focus"));
        assert!(labels.contains(&"Delete"));
    }

    #[test]
    fn context_menu_object_offers_focus_and_delete() {
        let items = context_menu_items_for(Selection::Object(1));
        let labels: Vec<&str> = items.iter().map(|i| i.label).collect();
        assert!(labels.contains(&"Focus"));
        assert!(labels.contains(&"Delete"));
    }

    #[test]
    fn context_menu_robot_omits_delete() {
        // Robots aren't user-deletable from the viewport (they're owned by
        // the robot manager). Focus-only is the expected behaviour.
        let items = context_menu_items_for(Selection::Robot(0));
        let labels: Vec<&str> = items.iter().map(|i| i.label).collect();
        assert!(labels.contains(&"Focus"));
        assert!(!labels.contains(&"Delete"));
    }

    #[test]
    fn context_menu_robot_link_focus_only() {
        let items = context_menu_items_for(Selection::RobotLink(0, 3));
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "Focus");
    }

    #[test]
    fn context_menu_each_item_maps_to_palette_action() {
        // Sanity: every item's action variant survives a clone. Catches
        // panics if PaletteAction grows new variants with !Copy data.
        for sel in [
            Selection::None,
            Selection::Source(0),
            Selection::Listener(0),
            Selection::Object(0),
            Selection::Robot(0),
            Selection::RobotLink(0, 0),
        ] {
            for item in context_menu_items_for(sel) {
                let _ = item.action;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agent Inspector (D8) — live observation/action message lane window.
// ---------------------------------------------------------------------------

/// Per-session state for the Agent Inspector window.
///
/// Holds the inspector's filter/UI state separately from the rolling event
/// log so opening + closing the window does not lose user preferences and
/// so the same log can be inspected from multiple windows in the future.
#[derive(Clone, Debug)]
pub struct AgentInspectorState {
    pub show_incoming: bool,
    pub show_outgoing: bool,
    pub show_internal: bool,
    pub show_errors_only: bool,
    pub robot_filter: Option<usize>,
    pub autoscroll: bool,
    pub expanded_index: Option<usize>,
}

impl Default for AgentInspectorState {
    fn default() -> Self {
        Self {
            show_incoming: true,
            show_outgoing: true,
            show_internal: false,
            show_errors_only: false,
            robot_filter: None,
            autoscroll: true,
            expanded_index: None,
        }
    }
}

impl AgentInspectorState {
    pub fn matches(&self, ev: &crate::agent::bridge::AgentEvent) -> bool {
        if self.show_errors_only && ev.kind != AgentEventKind::Error {
            return false;
        }
        match ev.direction {
            MessageDirection::Incoming if !self.show_incoming => return false,
            MessageDirection::Outgoing if !self.show_outgoing => return false,
            MessageDirection::Internal if !self.show_internal => return false,
            _ => {}
        }
        if let Some(r) = self.robot_filter {
            if ev.robot_id != Some(r) {
                return false;
            }
        }
        true
    }
}

/// Color the inspector uses for each protocol-message direction. Chosen
/// to match the existing `agent_activity_panel` event palette so the two
/// surfaces stay visually consistent.
fn direction_color(d: MessageDirection) -> egui::Color32 {
    match d {
        MessageDirection::Incoming => egui::Color32::from_rgb(120, 180, 255), // azure
        MessageDirection::Outgoing => egui::Color32::from_rgb(140, 230, 160), // mint
        MessageDirection::Internal => egui::Color32::from_rgb(160, 160, 180), // slate
    }
}

fn direction_glyph(d: MessageDirection) -> &'static str {
    match d {
        MessageDirection::Incoming => "▶",
        MessageDirection::Outgoing => "◀",
        MessageDirection::Internal => "·",
    }
}

fn kind_chip(k: &AgentEventKind) -> (&'static str, egui::Color32) {
    match k {
        AgentEventKind::Connect => ("bind", egui::Color32::from_rgb(100, 200, 130)),
        AgentEventKind::Step => ("step", egui::Color32::from_rgb(150, 200, 255)),
        AgentEventKind::Observe => ("obs", egui::Color32::from_rgb(180, 220, 240)),
        AgentEventKind::Reset => ("reset", egui::Color32::from_rgb(240, 180, 100)),
        AgentEventKind::Remove => ("close", egui::Color32::from_rgb(220, 150, 90)),
        AgentEventKind::Error => ("err", egui::Color32::from_rgb(240, 100, 100)),
        AgentEventKind::Message => ("msg", egui::Color32::from_rgb(200, 160, 240)),
    }
}

/// Format a JSON string with two-space indentation. Falls back to the raw
/// payload when parsing fails so the inspector still shows *something*
/// for malformed packets.
fn pretty_json(raw: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| raw.to_string()),
        Err(_) => raw.to_string(),
    }
}

/// Live observation/action inspector window — the user-visible artifact of
/// D8. Renders a tight, lane-style log: time | direction arrow | kind chip |
/// summary | byte size, with click-to-expand pretty-printed JSON per row.
///
/// The window is toggled by `*open` so the caller can drive visibility
/// from a menu item or sidebar checkbox.
pub fn agent_inspector_window(
    ctx: &egui::Context,
    state: &mut AgentInspectorState,
    log: &AgentActivityLog,
    capabilities: &[String],
    open: &mut bool,
) {
    if !*open {
        return;
    }
    let mut window_open = true;
    egui::Window::new("Agent Inspector")
        .open(&mut window_open)
        .default_pos(egui::pos2(80.0, 90.0))
        .default_width(520.0)
        .default_height(360.0)
        .resizable(true)
        .collapsible(true)
        .show(ctx, |ui| {
            // --- header row: capability badges + connected robot count ---
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("CAPS").small().weak());
                if capabilities.is_empty() {
                    ui.label(egui::RichText::new("none advertised yet").italics().weak());
                } else {
                    for cap in capabilities {
                        ui.label(
                            egui::RichText::new(cap)
                                .small()
                                .monospace()
                                .background_color(egui::Color32::from_rgb(40, 60, 90))
                                .color(egui::Color32::from_rgb(180, 220, 255)),
                        );
                    }
                }
            });
            ui.separator();

            // --- filter row ---
            ui.horizontal(|ui| {
                ui.checkbox(&mut state.show_incoming, "▶ in");
                ui.checkbox(&mut state.show_outgoing, "◀ out");
                ui.checkbox(&mut state.show_internal, "· int");
                ui.separator();
                ui.checkbox(&mut state.show_errors_only, "errors only");
                ui.separator();
                ui.checkbox(&mut state.autoscroll, "follow tail");
                ui.separator();
                let mut label = state
                    .robot_filter
                    .map(|r| format!("robot/{r}"))
                    .unwrap_or_else(|| "all robots".to_string());
                egui::ComboBox::from_id_salt("agent_inspector_robot_filter")
                    .selected_text(&label)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(state.robot_filter.is_none(), "all robots")
                            .clicked()
                        {
                            state.robot_filter = None;
                            label = "all robots".to_string();
                        }
                        for id in 0..log.connected_robots.len() {
                            let txt = format!("robot/{id}");
                            if ui
                                .selectable_label(state.robot_filter == Some(id), &txt)
                                .clicked()
                            {
                                state.robot_filter = Some(id);
                            }
                        }
                    });
            });
            ui.separator();

            // --- message lane ---
            let scroll = egui::ScrollArea::vertical().auto_shrink([false, false]);
            let scroll = if state.autoscroll {
                scroll.stick_to_bottom(true)
            } else {
                scroll
            };
            scroll.show(ui, |ui| {
                let mut shown = 0usize;
                let filter_state = state.clone();
                let mut to_expand: Option<usize> = None;
                for (idx, ev) in log
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| filter_state.matches(e))
                {
                    let (chip_text, chip_color) = kind_chip(&ev.kind);
                    let dir_color = direction_color(ev.direction);
                    let header = ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("{:>7.2}s", ev.timestamp))
                                .monospace()
                                .weak()
                                .small(),
                        );
                        ui.label(
                            egui::RichText::new(direction_glyph(ev.direction))
                                .color(dir_color)
                                .monospace(),
                        );
                        ui.label(
                            egui::RichText::new(chip_text)
                                .small()
                                .monospace()
                                .color(chip_color),
                        );
                        if let Some(r) = ev.robot_id {
                            ui.label(egui::RichText::new(format!("r/{r}")).small().weak());
                        }
                        let summary = if ev.description.len() > 60 {
                            format!("{}…", &ev.description[..60])
                        } else {
                            ev.description.clone()
                        };
                        ui.label(egui::RichText::new(summary).small());
                        if let Some(p) = &ev.payload_json {
                            ui.label(
                                egui::RichText::new(format!("{}B", p.len()))
                                    .small()
                                    .weak()
                                    .monospace(),
                            );
                        }
                    });
                    if header.response.clicked() {
                        to_expand = Some(idx);
                    }
                    // Inline expanded JSON for the selected row.
                    if filter_state.expanded_index == Some(idx) {
                        if let Some(payload) = &ev.payload_json {
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(pretty_json(payload))
                                        .monospace()
                                        .small(),
                                );
                            });
                        }
                    }
                    shown += 1;
                }
                if let Some(idx) = to_expand {
                    state.expanded_index = if state.expanded_index == Some(idx) {
                        None
                    } else {
                        Some(idx)
                    };
                }
                if shown == 0 {
                    ui.vertical_centered(|ui| {
                        ui.add_space(40.0);
                        ui.label(
                            egui::RichText::new("no traffic matches the current filters")
                                .italics()
                                .weak(),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(
                                "bind an agent or start the demo to see live messages",
                            )
                            .small()
                            .weak(),
                        );
                    });
                }
            });
        });
    if !window_open {
        *open = false;
    }
}

#[cfg(test)]
mod agent_inspector_tests {
    use super::*;
    use crate::agent::bridge::{AgentActivityLog, AgentEventKind, MessageDirection};

    fn log_with_events() -> AgentActivityLog {
        let mut log = AgentActivityLog::new(50);
        log.elapsed = 0.0;
        log.push_message(
            AgentEventKind::Connect,
            Some(0),
            "Bind robot/0".into(),
            MessageDirection::Incoming,
            Some(r#"{"type":"bind_target","robot_id":0}"#.to_string()),
        );
        log.push_message(
            AgentEventKind::Step,
            Some(0),
            "Step (motors=3)".into(),
            MessageDirection::Incoming,
            Some(r#"{"type":"step","action":{"motor_velocities":[0,0,0]}}"#.to_string()),
        );
        log.push_message(
            AgentEventKind::Observe,
            Some(0),
            "Observation (joints=3)".into(),
            MessageDirection::Outgoing,
            Some(r#"{"type":"observation","state":{}}"#.to_string()),
        );
        log.push_message(
            AgentEventKind::Error,
            Some(0),
            "Error: malformed action".into(),
            MessageDirection::Outgoing,
            Some(r#"{"type":"error","message":"malformed action"}"#.to_string()),
        );
        log
    }

    #[test]
    fn agent_inspector_state_defaults_to_user_friendly_filters() {
        let s = AgentInspectorState::default();
        assert!(s.show_incoming);
        assert!(s.show_outgoing);
        assert!(!s.show_internal, "internal events too noisy by default");
        assert!(s.autoscroll);
        assert!(!s.show_errors_only);
        assert!(s.robot_filter.is_none());
    }

    #[test]
    fn agent_inspector_filter_errors_only_drops_observations() {
        let log = log_with_events();
        let mut s = AgentInspectorState::default();
        s.show_errors_only = true;
        let kept: Vec<_> = log.iter().filter(|e| s.matches(e)).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].kind, AgentEventKind::Error);
    }

    #[test]
    fn agent_inspector_filter_direction_excludes_outgoing() {
        let log = log_with_events();
        let mut s = AgentInspectorState::default();
        s.show_outgoing = false;
        let kept: Vec<_> = log.iter().filter(|e| s.matches(e)).collect();
        // 2 incoming events (bind, step) — both observation + error were outgoing.
        assert_eq!(kept.len(), 2);
        assert!(kept
            .iter()
            .all(|e| e.direction == MessageDirection::Incoming));
    }

    #[test]
    fn agent_inspector_robot_filter_scopes_to_one_robot() {
        let mut log = log_with_events();
        // Add an event for a different robot.
        log.push_message(
            AgentEventKind::Step,
            Some(1),
            "Step robot 1".into(),
            MessageDirection::Incoming,
            Some(r#"{"type":"step"}"#.to_string()),
        );
        let mut s = AgentInspectorState::default();
        s.robot_filter = Some(1);
        let kept: Vec<_> = log.iter().filter(|e| s.matches(e)).collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].robot_id, Some(1));
    }

    #[test]
    fn agent_inspector_iter_messages_skips_payloadless_events() {
        let mut log = AgentActivityLog::new(10);
        log.push(AgentEventKind::Connect, Some(0), "no payload".into());
        log.push_message(
            AgentEventKind::Step,
            Some(0),
            "with payload".into(),
            MessageDirection::Incoming,
            Some(r#"{"x":1}"#.to_string()),
        );
        let msgs: Vec<_> = log.iter_messages().collect();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].description, "with payload");
    }

    #[test]
    fn agent_inspector_pretty_json_indents_well_formed_input() {
        let p = pretty_json(r#"{"type":"observation","reward":1.0}"#);
        assert!(p.contains("\n"));
        assert!(p.contains("\"type\": \"observation\""));
    }

    #[test]
    fn agent_inspector_pretty_json_passes_through_malformed_input() {
        let p = pretty_json("not json at all");
        assert_eq!(p, "not json at all");
    }

    #[test]
    fn agent_inspector_window_renders_without_panic() {
        // Headless egui smoke test — ensures the window function can run
        // through one frame against a non-empty log without panicking.
        let ctx = egui::Context::default();
        let mut state = AgentInspectorState::default();
        let log = log_with_events();
        let caps = vec![
            "observe".to_string(),
            "step".to_string(),
            "motors".to_string(),
        ];
        let mut open = true;
        let _ = ctx.run(Default::default(), |ctx| {
            agent_inspector_window(ctx, &mut state, &log, &caps, &mut open);
        });
        assert!(open, "window should remain open through one frame");
    }
}
