use glam::Vec3;

use crate::acoustics::SimulationState;
use crate::agent::bridge::{
    create_bridge, AgentActivityLog, AgentEventKind, SimBridgeClient, SimBridgeServer,
};
use crate::agent::demo::{DemoAgentHandle, DemoBehavior};
use crate::agent::{AgentServerConfig, AgentServerHandle};
use crate::fluids::FluidSimulation;
use crate::gas::GasSimulation;
use crate::renderer::{
    energy_to_color, project_3d, ray_ground_intersect, render_fluid_slice, render_gas_slice,
    screen_to_ray, Camera, FluidVisualizationMode, GasVisualizationMode,
};
use crate::robot::definition::{CollisionShape, RobotDefinition, SensorDefinition};
use crate::robot::sensors::sensor_world_pose;
use crate::robot::state::ActuatorCommand;
use crate::robot::RobotManager;
use crate::scene::{Listener, MaterialLibrary, MediumLibrary, Scene, SoundSource};

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
                ui.checkbox(&mut vp.show_robots, "Show Robots");
                ui.checkbox(&mut vp.show_sensor_rays, "Show Sensor Rays");
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
) {
    egui::SidePanel::left("side_panel")
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.heading("EchoMap");
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
                });
        });
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
    activity_log: &AgentActivityLog,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    for (robot_idx, robot) in robot_manager.robots.iter().enumerate() {
        let color = robot_color(robot_idx);
        let joint_color = egui::Color32::from_rgb(
            (color.r() as u16 * 3 / 4) as u8 + 60,
            (color.g() as u16 * 3 / 4) as u8 + 60,
            (color.b() as u16 * 3 / 4) as u8 + 60,
        );

        let link_poses = robot.state.link_poses_as_mat4();

        // --- Draw links ---
        for (link_idx, link_def) in robot.definition.links.iter().enumerate() {
            if link_idx >= link_poses.len() {
                break;
            }
            let pose = link_poses[link_idx];
            render_link_shape(
                painter,
                &link_def.collision_shape,
                pose,
                color,
                cam,
                center,
                scale,
                rect,
            );
        }

        // --- Draw joints as spheres at the child link origin ---
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

        // --- Draw sensor rays ---
        if show_sensor_rays {
            let sensor_color =
                egui::Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), 120);
            render_sensor_rays(painter, robot, sensor_color, cam, center, scale);
        }

        // --- Label: robot name + connection status above base link ---
        if !link_poses.is_empty() {
            let base_pos = Vec3::new(
                link_poses[0].w_axis.x,
                link_poses[0].w_axis.y + 0.3,
                link_poses[0].w_axis.z,
            );
            let lp = project_3d(base_pos, cam, center, scale);
            if rect.contains(lp) {
                // Connection status indicator
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

                // Small status dot next to the label
                let dot_pos = egui::Pos2::new(lp.x - 30.0, lp.y - 4.0);
                painter.circle_filled(dot_pos, 3.0, status_color);
            }
        }
    }
}

/// Draw a single link's collision shape as a wireframe at the given world pose.
#[allow(clippy::too_many_arguments)]
fn render_link_shape(
    painter: &egui::Painter,
    shape: &CollisionShape,
    pose: glam::Mat4,
    color: egui::Color32,
    cam: &Camera,
    center: egui::Pos2,
    scale: f32,
    rect: egui::Rect,
) {
    let origin = Vec3::new(pose.w_axis.x, pose.w_axis.y, pose.w_axis.z);

    match shape {
        CollisionShape::Cuboid { half_extents } => {
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
        CollisionShape::Cylinder { radius, height } => {
            draw_wireframe_cylinder(
                painter, pose, *radius, *height, color, cam, center, scale, rect,
            );
        }
        CollisionShape::Sphere { radius } => {
            let sp = project_3d(origin, cam, center, scale);
            if rect.contains(sp) {
                let screen_radius = (*radius * scale * 5.0).clamp(3.0, 40.0);
                painter.circle_stroke(sp, screen_radius, egui::Stroke::new(1.5, color));
            }
        }
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
