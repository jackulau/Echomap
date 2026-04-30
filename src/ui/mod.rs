use glam::Vec3;

use crate::acoustics::SimulationState;
use crate::renderer::{energy_to_color, project_3d, ray_ground_intersect, screen_to_ray, Camera};
use crate::scene::{Listener, MaterialLibrary, Scene, SoundSource};

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum InteractionMode {
    #[default]
    Select,
    PlaceSource,
    PlaceListener,
}

#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    #[default]
    None,
    Source(usize),
    Listener(usize),
    Object(usize),
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
        }
    }
}

pub fn menu_bar(
    ctx: &egui::Context,
    show_settings: &mut bool,
    scene: &mut Scene,
    vp: &mut ViewportState,
) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open STEP File...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("STEP", &["step", "stp", "STEP", "STP"])
                        .pick_file()
                    {
                        match crate::io::load_step_file(&path) {
                            Ok(objects) => {
                                scene.meshes.extend(objects);
                                focus_on_scene(&mut vp.camera, scene);
                                log::info!("Loaded STEP: {}", path.display());
                            }
                            Err(e) => {
                                log::error!("Failed to load STEP: {e}");
                            }
                        }
                    }
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Exit").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("Add", |ui| {
                if ui.button("Box Room (5x4x3m)").clicked() {
                    scene
                        .meshes
                        .push(crate::scene::primitives::box_room(5.0, 4.0, 3.0));
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                if ui.button("L-Room (8x6x3m)").clicked() {
                    scene
                        .meshes
                        .extend(crate::scene::primitives::l_room(8.0, 6.0, 3.0, 3.0, 3.0));
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                if ui.button("Partition Wall").clicked() {
                    scene.meshes.push(crate::scene::primitives::partition_wall(
                        Vec3::new(2.0, 0.0, 1.0),
                        2.0,
                        2.5,
                        0.15,
                    ));
                    ui.close_menu();
                }
                if ui.button("Platform / Stage").clicked() {
                    scene.meshes.push(crate::scene::primitives::platform(
                        Vec3::new(1.0, 0.0, 1.0),
                        2.0,
                        2.0,
                        0.5,
                    ));
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Sound Source").clicked() {
                    scene.sound_sources.push(SoundSource::default());
                    vp.selection = Selection::Source(scene.sound_sources.len() - 1);
                    ui.close_menu();
                }
                if ui.button("Listener").clicked() {
                    let n = scene.listeners.len() + 1;
                    scene.listeners.push(Listener {
                        name: format!("Listener {n}"),
                        ..Default::default()
                    });
                    vp.selection = Selection::Listener(scene.listeners.len() - 1);
                    ui.close_menu();
                }
            });

            ui.menu_button("View", |ui| {
                ui.checkbox(&mut vp.show_grid, "Show Grid");
                ui.checkbox(&mut vp.show_rays, "Show Ray Paths");
                ui.separator();
                if ui.button("Reset Camera").clicked() {
                    vp.camera = Camera::default();
                    if !scene.meshes.is_empty() {
                        focus_on_scene(&mut vp.camera, scene);
                    }
                    ui.close_menu();
                }
                if ui.button("Focus on Scene").clicked() {
                    focus_on_scene(&mut vp.camera, scene);
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Settings...").clicked() {
                    *show_settings = true;
                    ui.close_menu();
                }
            });
        });
    });
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

            ui.label("Navigate: Alt+Drag=Orbit  RightDrag=Pan  Scroll=Zoom");
        });
    });
}

pub fn side_panel(
    ctx: &egui::Context,
    scene: &mut Scene,
    sim: &mut SimulationState,
    vp: &mut ViewportState,
) {
    egui::SidePanel::left("side_panel")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("EchoMap");
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

                            if ui.button("Delete Object").clicked() {
                                to_remove = Some(i);
                            }
                        }
                    }
                    if let Some(i) = to_remove {
                        scene.meshes.remove(i);
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
                        scene.sound_sources.remove(i);
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
                        scene.listeners.remove(i);
                        vp.selection = Selection::None;
                    }
                });

            ui.separator();

            ui.heading("Simulation");
            ui.add(
                egui::Slider::new(&mut sim.config.ray_count, 100..=100_000)
                    .text("Rays")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut sim.config.max_bounces, 1..=200).text("Max Bounces"));

            ui.add_space(8.0);

            let can_run =
                !sim.running && !scene.meshes.is_empty() && !scene.sound_sources.is_empty();
            if ui
                .add_enabled(can_run, egui::Button::new("Run Simulation"))
                .clicked()
            {
                sim.run(scene);
            }

            if sim.running {
                ui.spinner();
            }

            if let Some(ref result) = sim.result {
                ui.label(format!(
                    "Rays: {} | Grid: {}",
                    result.ray_paths.len(),
                    result.energy_grid.len()
                ));
            }
        });
}

pub fn viewport_3d(
    ctx: &egui::Context,
    scene: &mut Scene,
    sim: &SimulationState,
    vp: &mut ViewportState,
) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

        let rect = response.rect;
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(30, 30, 35));

        let cam = &mut vp.camera;
        let center = rect.center();

        // --- Keyboard shortcuts ---
        let modifiers = ui.input(|i| i.modifiers);
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
            }
            if i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace) {
                match vp.selection {
                    Selection::Source(idx) if idx < scene.sound_sources.len() => {
                        scene.sound_sources.remove(idx);
                        vp.selection = Selection::None;
                    }
                    Selection::Listener(idx) if idx < scene.listeners.len() => {
                        scene.listeners.remove(idx);
                        vp.selection = Selection::None;
                    }
                    Selection::Object(idx) if idx < scene.meshes.len() => {
                        scene.meshes.remove(idx);
                        vp.selection = Selection::None;
                    }
                    _ => {}
                }
            }
            if i.key_pressed(egui::Key::F) {
                match vp.selection {
                    Selection::Source(idx) if idx < scene.sound_sources.len() => {
                        cam.focus_on(scene.sound_sources[idx].position, 3.0);
                    }
                    Selection::Listener(idx) if idx < scene.listeners.len() => {
                        cam.focus_on(scene.listeners[idx].position, 3.0);
                    }
                    _ => {
                        focus_on_scene(cam, scene);
                    }
                }
            }
        });

        // --- Camera controls ---
        let is_orbit = response.dragged_by(egui::PointerButton::Middle)
            || (response.dragged_by(egui::PointerButton::Primary) && modifiers.alt);
        let is_pan = response.dragged_by(egui::PointerButton::Secondary);

        if is_orbit {
            let d = response.drag_delta();
            cam.orbit(d.x, d.y);
        }
        if is_pan {
            let d = response.drag_delta();
            cam.pan(d.x, d.y);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                cam.zoom(scroll * 0.1);
            }
        }

        // Recalculate scale after camera changes
        let scale = rect.height() * 0.4 / cam.distance;

        // --- Hover world position ---
        vp.hover_world = None;
        if let Some(hover_pos) = response.hover_pos() {
            let (origin, dir) = screen_to_ray(hover_pos, cam, center, scale);
            vp.hover_world = ray_ground_intersect(origin, dir);
        }

        // --- Object interaction ---
        if !is_orbit && !is_pan {
            if response.drag_started_by(egui::PointerButton::Primary) && !modifiers.alt {
                if let Some(hover) = response.hover_pos() {
                    let sel = hit_test(hover, scene, cam, center, scale);
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
                            vp.selection = hit_test(hover, scene, cam, center, scale);
                        }
                        InteractionMode::PlaceSource => {
                            if let Some(gp) = ground {
                                scene.sound_sources.push(SoundSource {
                                    position: Vec3::new(gp.x, 1.0, gp.z),
                                    ..Default::default()
                                });
                                vp.selection = Selection::Source(scene.sound_sources.len() - 1);
                            }
                        }
                        InteractionMode::PlaceListener => {
                            if let Some(gp) = ground {
                                let n = scene.listeners.len() + 1;
                                scene.listeners.push(Listener {
                                    position: Vec3::new(gp.x, 1.0, gp.z),
                                    name: format!("Listener {n}"),
                                });
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

        // --- Drawing ---

        // Grid
        if vp.show_grid {
            let grid_color = egui::Color32::from_rgba_premultiplied(80, 80, 80, 40);
            let axis_color = egui::Color32::from_rgba_premultiplied(120, 120, 120, 60);
            for i in -10..=10 {
                let f = i as f32;
                let color = if i == 0 { axis_color } else { grid_color };
                let p1 = project_3d(Vec3::new(f, 0.0, -10.0), cam, center, scale);
                let p2 = project_3d(Vec3::new(f, 0.0, 10.0), cam, center, scale);
                painter.line_segment([p1, p2], egui::Stroke::new(0.5, color));

                let p3 = project_3d(Vec3::new(-10.0, 0.0, f), cam, center, scale);
                let p4 = project_3d(Vec3::new(10.0, 0.0, f), cam, center, scale);
                painter.line_segment([p3, p4], egui::Stroke::new(0.5, color));
            }
        }

        // Scene meshes (wireframe)
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

        // Ray paths
        if vp.show_rays {
            if let Some(ref result) = sim.result {
                let ray_color = egui::Color32::from_rgba_premultiplied(255, 200, 50, 30);
                let max_draw = 500.min(result.ray_paths.len());
                for path in &result.ray_paths[..max_draw] {
                    for segment in path.windows(2) {
                        let p1 = project_3d(segment[0], cam, center, scale);
                        let p2 = project_3d(segment[1], cam, center, scale);
                        painter.line_segment([p1, p2], egui::Stroke::new(0.5, ray_color));
                    }
                }

                for gp in &result.energy_grid {
                    if gp.energy > 0.01 {
                        let color = energy_to_color(gp.energy, result.max_energy);
                        let p = project_3d(gp.position, cam, center, scale);
                        if rect.contains(p) {
                            painter.circle_filled(p, 2.0, color);
                        }
                    }
                }
            }
        }

        // Sound sources
        for (i, source) in scene.sound_sources.iter().enumerate() {
            if !source.enabled {
                continue;
            }
            let p = project_3d(source.position, cam, center, scale);
            let is_selected = vp.selection == Selection::Source(i);
            if is_selected {
                painter.circle_stroke(p, 10.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
            }
            painter.circle_filled(p, 6.0, egui::Color32::from_rgb(255, 100, 50));
            painter.text(
                p + egui::vec2(10.0, -10.0),
                egui::Align2::LEFT_BOTTOM,
                format!("S{}", i + 1),
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
        }

        // Listeners
        for (i, listener) in scene.listeners.iter().enumerate() {
            let p = project_3d(listener.position, cam, center, scale);
            let is_selected = vp.selection == Selection::Listener(i);
            if is_selected {
                painter.circle_stroke(p, 9.0, egui::Stroke::new(2.0, egui::Color32::WHITE));
            }
            painter.circle_filled(p, 5.0, egui::Color32::from_rgb(50, 150, 255));
            painter.text(
                p + egui::vec2(10.0, -10.0),
                egui::Align2::LEFT_BOTTOM,
                &listener.name,
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
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

pub fn status_bar(ctx: &egui::Context, vp: &ViewportState, scene: &Scene) {
    egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let mode_str = match vp.mode {
                InteractionMode::Select => "Select",
                InteractionMode::PlaceSource => "Place Source",
                InteractionMode::PlaceListener => "Place Listener",
            };
            ui.label(format!("Mode: {mode_str}"));
            ui.separator();

            if let Some(pos) = vp.hover_world {
                ui.label(format!("World: ({:.2}, {:.2})", pos.x, pos.z));
                ui.separator();
            }

            ui.label(format!(
                "Objects: {} | Sources: {} | Listeners: {}",
                scene.meshes.len(),
                scene.sound_sources.len(),
                scene.listeners.len()
            ));
        });
    });
}

pub fn settings_window(ctx: &egui::Context, open: &mut bool, sim: &mut SimulationState) {
    egui::Window::new("Simulation Settings")
        .open(open)
        .resizable(false)
        .show(ctx, |ui| {
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
        });
}

fn hit_test(
    screen_pos: egui::Pos2,
    scene: &Scene,
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

    Selection::None
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

use egui;
