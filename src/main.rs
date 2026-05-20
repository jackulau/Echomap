use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let boxing = std::env::args().any(|a| a == "--boxing");

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
        Box::new(move |cc| Ok(Box::new(app::EchoMapApp::new(cc, boxing)))),
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
    use echomap::ui::ViewportState;
    use eframe::egui;

    pub struct EchoMapApp {
        scene: Scene,
        simulation: SimulationState,
        fluid_sim: FluidSimulation,
        gas_sim: GasSimulation,
        robot_manager: RobotManager,
        viewport: ViewportState,
        show_settings: bool,
        show_agent_inspector: bool,
        agent_inspector_state: echomap::ui::AgentInspectorState,
        bridge_client: SimBridgeClient,
        bridge_server: Option<SimBridgeServer>,
        agent_server_config: AgentServerConfig,
        agent_server_handle: Option<AgentServerHandle>,
        activity_log: AgentActivityLog,
        demo_handle: Option<DemoAgentHandle>,
        demo_behavior: DemoBehavior,
    }

    impl EchoMapApp {
        pub fn new(_cc: &eframe::CreationContext<'_>, boxing: bool) -> Self {
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
                    show_agent_inspector: true, // open by default in boxing mode
                    agent_inspector_state: echomap::ui::AgentInspectorState::default(),
                    bridge_client,
                    bridge_server: Some(bridge_server),
                    agent_server_config,
                    agent_server_handle: Some(handle),
                    activity_log: AgentActivityLog::default(),
                    demo_handle: None,
                    demo_behavior: DemoBehavior::ReachTarget,
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
                show_agent_inspector: false,
                agent_inspector_state: echomap::ui::AgentInspectorState::default(),
                bridge_client,
                bridge_server: retained_bridge,
                agent_server_config,
                agent_server_handle,
                activity_log: AgentActivityLog::default(),
                demo_handle: None,
                demo_behavior: DemoBehavior::ReachTarget,
            }
        }
    }

    impl eframe::App for EchoMapApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // Bump activity log elapsed time.
            self.activity_log.elapsed += ctx.input(|i| i.predicted_dt);

            // Process pending agent bridge commands each frame, logging events.
            self.bridge_client.process_pending_with_log(
                &mut self.robot_manager,
                &self.scene.meshes,
                &mut self.activity_log,
            );

            echomap::ui::menu_bar(
                ctx,
                &mut self.show_settings,
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
            echomap::ui::status_bar(ctx, &self.viewport, &self.scene, &self.robot_manager);

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
        }
    }
}
