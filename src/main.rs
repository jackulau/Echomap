use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let boxing = std::env::args().any(|a| a == "--boxing");
    let test_frames = std::env::var("ECHOMAP_TEST_FRAMES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("EchoMap")
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "EchoMap",
        options,
        Box::new(move |cc| Ok(Box::new(app::EchoMapApp::new(cc, boxing, test_frames)))),
    )
}

mod app {
    use echomap::acoustics::SimulationState;
    use echomap::agent::bridge::{
        create_bridge, create_bridge_with_boxing, AgentActivityLog, SimBridgeClient,
        SimBridgeServer,
    };
    use echomap::agent::demo::{DemoAgentHandle, DemoBehavior};
    use echomap::agent::{AgentServerConfig, AgentServerHandle};
    use echomap::fluids::FluidSimulation;
    use echomap::gas::GasSimulation;
    use echomap::robot::RobotManager;
    use echomap::scene::Scene;
    use echomap::ui::{AppStatus, ViewportState};
    use eframe::egui;
    use std::time::{Duration, Instant};

    /// Soft per-frame budget when ECHOMAP_TEST_FRAMES is active.
    /// Going over once is noisy-not-fatal — the perf governor downshifts
    /// and the harness keeps going. Only `MAX_CONSECUTIVE_OVERAGE`
    /// over-budget frames in a row triggers exit 2.
    const TEST_FRAME_BUDGET: Duration = Duration::from_millis(500);

    /// How many consecutive over-budget frames the harness tolerates
    /// before declaring the governor cannot recover and exiting 2.
    /// Picked to give PerfGovernor's STICKY_FRAMES window time to take
    /// effect (governor downshifts within ~30 samples).
    const MAX_CONSECUTIVE_OVERAGE: u32 = 30;

    pub struct EchoMapApp {
        scene: Scene,
        simulation: SimulationState,
        fluid_sim: FluidSimulation,
        gas_sim: GasSimulation,
        robot_manager: RobotManager,
        viewport: ViewportState,
        show_settings: bool,
        show_about: bool,
        show_performance: bool,
        show_agent_inspector: bool,
        agent_inspector_state: echomap::ui::AgentInspectorState,
        status: AppStatus,
        bridge_client: SimBridgeClient,
        bridge_server: Option<SimBridgeServer>,
        agent_server_config: AgentServerConfig,
        agent_server_handle: Option<AgentServerHandle>,
        activity_log: AgentActivityLog,
        demo_handle: Option<DemoAgentHandle>,
        demo_behavior: DemoBehavior,
        // ECHOMAP_TEST_FRAMES support: when Some(N), the app runs exactly N
        // update() ticks and then exits 0 (or exits 2 if any frame exceeded
        // TEST_FRAME_BUDGET). None = normal interactive mode.
        test_frame_limit: Option<usize>,
        test_frames_done: std::sync::atomic::AtomicUsize,
        test_consecutive_overage: std::sync::atomic::AtomicU32,
        device_caps: echomap::io::DeviceCaps,
        perf_governor: echomap::renderer::PerfGovernor,
    }

    impl EchoMapApp {
        pub fn new(
            cc: &eframe::CreationContext<'_>,
            boxing: bool,
            test_frames: Option<usize>,
        ) -> Self {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            let agent_server_config = AgentServerConfig::default();

            if boxing {
                let mut boxing_config = echomap::robot::boxing::BoxingMatchConfig::default();
                if let Ok(v) = std::env::var("ROUND_DURATION") {
                    if let Ok(s) = v.parse::<f32>() {
                        boxing_config.round_duration = s;
                    }
                }
                if let Ok(v) = std::env::var("NUM_ROUNDS") {
                    if let Ok(n) = v.parse::<u8>() {
                        boxing_config.num_rounds = n;
                    }
                }
                let (scenario, robot_manager) =
                    echomap::robot::boxing::BoxingScenario::new(boxing_config);

                let (bridge_server, bridge_client) =
                    create_bridge_with_boxing(scenario.boxing_match);

                let handle = echomap::agent::start_agent_server(
                    agent_server_config.clone(),
                    bridge_server.clone(),
                );
                log::info!(
                    "Boxing mode: server started on WS:{}",
                    agent_server_config.ws_port
                );

                let scene = Scene {
                    meshes: scenario.ring.meshes,
                    ..Scene::default()
                };

                let mut viewport = ViewportState::default();
                viewport.camera.distance = 3.0;
                viewport.camera.pitch = 15.0_f32.to_radians();
                viewport.camera.yaw = 0.0;
                viewport.camera.target = glam::Vec3::new(0.0, 0.3, 0.0);
                viewport.camera.update_position();
                viewport.camera_auto_track = true;

                return Self {
                    scene,
                    simulation: SimulationState::default(),
                    fluid_sim: FluidSimulation::default(),
                    gas_sim: GasSimulation::default(),
                    robot_manager,
                    viewport,
                    show_settings: false,
                    show_about: false,
                    show_performance: false,
                    show_agent_inspector: false, // user toggles via View > Agent Inspector
                    agent_inspector_state: echomap::ui::AgentInspectorState::default(),
                    status: AppStatus::default(),
                    bridge_client,
                    bridge_server: Some(bridge_server),
                    agent_server_config,
                    agent_server_handle: Some(handle),
                    activity_log: AgentActivityLog::default(),
                    demo_handle: None,
                    demo_behavior: DemoBehavior::ReachTarget,
                    test_frame_limit: test_frames,
                    test_frames_done: std::sync::atomic::AtomicUsize::new(0),
                    test_consecutive_overage: std::sync::atomic::AtomicU32::new(0),
                    device_caps: echomap::io::DeviceCaps::detect(),
                    perf_governor: echomap::renderer::PerfGovernor::new(),
                };
            }

            let (bridge_server, bridge_client) = create_bridge();

            let (agent_server_handle, retained_bridge) = if agent_server_config.enabled {
                log::info!("Starting agent server (enabled by default config)");
                let handle = echomap::agent::start_agent_server(
                    agent_server_config.clone(),
                    bridge_server.clone(),
                );
                (Some(handle), Some(bridge_server))
            } else {
                (None, Some(bridge_server))
            };

            let device_caps = echomap::io::DeviceCaps::detect();
            log::info!("Device capabilities: {}", device_caps.summary());

            let mut scene = Scene::default();
            if device_caps.stress_mode {
                // ECHOMAP_STRESS=1 — synthesize 50 listeners around origin
                // and bias the test budget. The intent is to surface
                // perf regressions early without crashing the harness.
                for i in 0..50u32 {
                    let angle = (i as f32) * std::f32::consts::TAU / 50.0;
                    scene.listeners.push(echomap::scene::Listener {
                        position: glam::Vec3::new(angle.cos() * 5.0, 0.5, angle.sin() * 5.0),
                        name: format!("stress_listener_{i}"),
                        capture_radius: 0.3,
                    });
                }
                log::warn!(
                    "ECHOMAP_STRESS=1: pre-loaded {} listeners for stress smoke",
                    scene.listeners.len()
                );
            }

            Self {
                scene,
                simulation: SimulationState::default(),
                fluid_sim: FluidSimulation::default(),
                gas_sim: GasSimulation::default(),
                robot_manager: RobotManager::default(),
                viewport: ViewportState::default(),
                show_settings: false,
                show_about: false,
                show_performance: false,
                show_agent_inspector: false,
                agent_inspector_state: echomap::ui::AgentInspectorState::default(),
                status: AppStatus::default(),
                bridge_client,
                bridge_server: retained_bridge,
                agent_server_config,
                agent_server_handle,
                activity_log: AgentActivityLog::default(),
                demo_handle: None,
                demo_behavior: DemoBehavior::ReachTarget,
                test_frame_limit: test_frames,
                test_frames_done: std::sync::atomic::AtomicUsize::new(0),
                test_consecutive_overage: std::sync::atomic::AtomicU32::new(0),
                device_caps,
                perf_governor: echomap::renderer::PerfGovernor::new(),
            }
        }
    }

    impl eframe::App for EchoMapApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // Time every frame's compute so the PerfGovernor reflects real
            // interactive load, not just ECHOMAP_TEST_FRAMES smoke runs.
            let frame_start = Instant::now();

            // Feed the previous frame's class into this frame's render budgets
            // (the governor records at end-of-frame, so this reflects 0..N-1).
            self.viewport.perf_work_scale = self.perf_governor.class().work_scale();

            // Bump activity log elapsed time.
            self.activity_log.elapsed += ctx.input(|i| i.predicted_dt);

            // Process pending agent bridge commands each frame, logging events.
            self.bridge_client.process_pending_with_log(
                &mut self.robot_manager,
                &self.scene.meshes,
                &mut self.activity_log,
            );

            // Drain the acoustic sim's bounded mpsc — new ray paths arrive
            // from the worker thread here. If anything arrived, schedule
            // another paint so the new geometry shows up immediately.
            // While the sim is still running we keep requesting repaints
            // so progress and live ray paths update at the display rate.
            let drained = self.simulation.tick();
            if drained || self.simulation.is_running() {
                ctx.request_repaint();
            }

            echomap::ui::menu_bar(
                ctx,
                &mut self.show_settings,
                &mut self.show_about,
                &mut self.status,
                &mut self.simulation,
                &mut self.show_agent_inspector,
                &mut self.scene,
                &mut self.viewport,
            );
            echomap::ui::toolbar(ctx, &mut self.viewport);
            echomap::ui::outliner_panel(
                ctx,
                &mut self.scene,
                &mut self.viewport,
                &self.robot_manager,
            );
            echomap::ui::side_panel(
                ctx,
                &mut self.scene,
                &mut self.simulation,
                &mut self.viewport,
                &mut self.fluid_sim,
                &mut self.gas_sim,
                &mut self.robot_manager,
                &mut self.agent_server_config,
                &mut self.agent_server_handle,
                &mut self.bridge_client,
                &mut self.bridge_server,
            );
            echomap::ui::agent_activity_panel(
                ctx,
                &self.activity_log,
                &self.robot_manager,
                &self.agent_server_handle,
                &mut self.demo_handle,
                &mut self.demo_behavior,
                &self.bridge_server,
            );
            echomap::ui::viewport_3d(
                ctx,
                &mut self.scene,
                &self.simulation,
                &mut self.viewport,
                &self.fluid_sim,
                &self.gas_sim,
                &self.robot_manager,
                &self.activity_log,
                &self.bridge_client,
            );
            echomap::ui::status_bar(
                ctx,
                &self.viewport,
                &self.scene,
                &self.robot_manager,
                &self.simulation,
                &self.status,
                echomap::ui::perf_label_for(&self.perf_governor),
            );

            // Drain app-level palette actions (those viewport_3d can't
            // handle because they touch settings/about window flags or
            // need a mutable Sim handle).
            if let Some(action) = self.viewport.pending_palette_action.take() {
                use echomap::ui::PaletteAction;
                match action {
                    PaletteAction::ToggleSettings => self.show_settings = !self.show_settings,
                    PaletteAction::ToggleAbout => self.show_about = !self.show_about,
                    PaletteAction::NewScene => {
                        self.scene = echomap::scene::Scene::default();
                        self.viewport.selection = echomap::ui::Selection::None;
                        self.viewport.history.clear();
                    }
                    PaletteAction::RunSimulation => {
                        if !self.simulation.is_running() {
                            self.simulation.start(&self.scene);
                        }
                    }
                    _ => {} // viewport_3d handled it
                }
            }

            // Drain tele-op pending action (Ctrl+T mode) onto robot/0.
            if let Some(action) = self.viewport.teleop_pending.take() {
                if let Some(robot) = self.robot_manager.get_robot_mut(0) {
                    echomap::robot::state::apply_action(
                        &robot.definition,
                        &mut robot.state,
                        &action,
                    );
                }
                ctx.request_repaint();
            }

            // Step robot simulation (skip when agent server owns stepping via bridge)
            if self.agent_server_handle.is_none() {
                let dt = 1.0 / 60.0;
                self.robot_manager.step(dt, &self.scene.meshes);
            }

            if self.show_settings {
                echomap::ui::settings_window(
                    ctx,
                    &mut self.show_settings,
                    &mut self.simulation,
                    &mut self.fluid_sim,
                    &mut self.gas_sim,
                );
            }

            if self.show_about {
                echomap::ui::about_window(ctx, &mut self.show_about);
            }

            if self.show_performance {
                echomap::ui::performance_window(
                    ctx,
                    &mut self.show_performance,
                    &self.device_caps,
                    &self.perf_governor,
                );
            }

            // Agent Inspector — live observation/action message lane.
            // Capabilities derived from the first connected robot so the
            // inspector reflects what the actively bound agent sees.
            let inspector_capabilities = self
                .robot_manager
                .get_robot(0)
                .map(|r| {
                    let obs =
                        echomap::robot::state::ObservationSpace::from_definition(&r.definition);
                    let act = echomap::robot::state::ActionSpace::from_definition(&r.definition);
                    echomap::agent::protocol::capabilities_from_spaces(
                        &obs,
                        &act,
                        r.state.combat.is_some(),
                    )
                })
                .unwrap_or_default();
            echomap::ui::agent_inspector_window(
                ctx,
                &mut self.agent_inspector_state,
                &self.activity_log,
                &inspector_capabilities,
                &mut self.show_agent_inspector,
            );

            // Request continuous repainting when demo agent or agent server is running.
            if self.demo_handle.as_ref().is_some_and(|h| h.is_running())
                || self
                    .agent_server_handle
                    .as_ref()
                    .is_some_and(|h| h.status().running)
            {
                ctx.request_repaint();
            }

            // Record this frame's compute time on EVERY frame so the governor
            // (and the Performance window / status-bar label it drives) tracks
            // real interactive load — not only ECHOMAP_TEST_FRAMES runs.
            let elapsed = frame_start.elapsed();
            self.perf_governor.record_frame(elapsed);

            // ECHOMAP_TEST_FRAMES: governor-driven soft budget. Single
            // overruns log + downshift PerfGovernor; only
            // MAX_CONSECUTIVE_OVERAGE in a row triggers exit 2. This
            // mirrors real-user behaviour: a slow device degrades
            // gracefully instead of crashing the process.
            if let Some(limit) = self.test_frame_limit {
                if elapsed > TEST_FRAME_BUDGET {
                    let prev = self
                        .test_consecutive_overage
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let streak = prev + 1;
                    eprintln!(
                        "ECHOMAP_TEST_FRAMES: frame over budget {:?} (took {:?}), \
                         streak {}/{}, governor: {:?}",
                        TEST_FRAME_BUDGET,
                        elapsed,
                        streak,
                        MAX_CONSECUTIVE_OVERAGE,
                        self.perf_governor.class()
                    );
                    if streak >= MAX_CONSECUTIVE_OVERAGE {
                        eprintln!(
                            "ECHOMAP_TEST_FRAMES: governor could not recover after {} \
                             consecutive over-budget frames; exiting 2",
                            streak
                        );
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        std::process::exit(2);
                    }
                } else {
                    // Reset streak on any in-budget frame.
                    self.test_consecutive_overage
                        .store(0, std::sync::atomic::Ordering::SeqCst);
                }
                let prev = self
                    .test_frames_done
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let done = prev + 1;
                // Force the next frame so we hit the limit without waiting on input.
                ctx.request_repaint();
                if done >= limit {
                    eprintln!(
                        "ECHOMAP_TEST_FRAMES: completed {}/{} frames (last {:?}, governor: {:?}); exiting 0",
                        done,
                        limit,
                        elapsed,
                        self.perf_governor.class()
                    );
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    std::process::exit(0);
                }
            }
        }
    }
}
