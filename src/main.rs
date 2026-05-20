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

    /// Per-frame budget when ECHOMAP_TEST_FRAMES is active.
    /// If any single update() exceeds this, the test harness exits 2.
    const TEST_FRAME_BUDGET: Duration = Duration::from_millis(500);

    pub struct EchoMapApp {
        scene: Scene,
        simulation: SimulationState,
        fluid_sim: FluidSimulation,
        gas_sim: GasSimulation,
        robot_manager: RobotManager,
        viewport: ViewportState,
        show_settings: bool,
        show_about: bool,
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
                    show_agent_inspector: true, // open by default in boxing mode
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

            Self {
                scene: Scene::default(),
                simulation: SimulationState::default(),
                fluid_sim: FluidSimulation::default(),
                gas_sim: GasSimulation::default(),
                robot_manager: RobotManager::default(),
                viewport: ViewportState::default(),
                show_settings: false,
                show_about: false,
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
            }
        }
    }

    impl eframe::App for EchoMapApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            let frame_start = self.test_frame_limit.map(|_| Instant::now());

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
            );

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

            // ECHOMAP_TEST_FRAMES: enforce frame budget + bounded run.
            if let (Some(limit), Some(start)) = (self.test_frame_limit, frame_start) {
                let elapsed = start.elapsed();
                if elapsed > TEST_FRAME_BUDGET {
                    eprintln!(
                        "ECHOMAP_TEST_FRAMES: frame exceeded budget {:?} (took {:?}); exiting 2",
                        TEST_FRAME_BUDGET, elapsed
                    );
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    std::process::exit(2);
                }
                let prev = self
                    .test_frames_done
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let done = prev + 1;
                // Force the next frame so we hit the limit without waiting on input.
                ctx.request_repaint();
                if done >= limit {
                    eprintln!(
                        "ECHOMAP_TEST_FRAMES: completed {}/{} frames (last {:?}); exiting 0",
                        done, limit, elapsed
                    );
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    std::process::exit(0);
                }
            }
        }
    }
}
