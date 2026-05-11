use eframe::egui;

fn main() -> eframe::Result<()> {
    env_logger::init();

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
        Box::new(|cc| Ok(Box::new(app::EchoMapApp::new(cc)))),
    )
}

mod app {
    use echomap::acoustics::SimulationState;
    use echomap::agent::bridge::{
        create_bridge, AgentActivityLog, SimBridgeClient, SimBridgeServer,
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
        bridge_client: SimBridgeClient,
        bridge_server: Option<SimBridgeServer>,
        agent_server_config: AgentServerConfig,
        agent_server_handle: Option<AgentServerHandle>,
        activity_log: AgentActivityLog,
        demo_handle: Option<DemoAgentHandle>,
        demo_behavior: DemoBehavior,
    }

    impl EchoMapApp {
        pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
            let (bridge_server, bridge_client) = create_bridge();

            let agent_server_config = AgentServerConfig::default();
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
                &mut self.scene,
                &mut self.viewport,
            );
            echomap::ui::toolbar(ctx, &mut self.viewport);
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
            );
            echomap::ui::status_bar(ctx, &self.viewport, &self.scene);

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

            // Request continuous repainting when demo agent is running.
            if self.demo_handle.as_ref().is_some_and(|h| h.is_running()) {
                ctx.request_repaint();
            }
        }
    }
}
