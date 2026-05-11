mod acoustics;
mod agent;
mod fluids;
mod gas;
mod io;
mod renderer;
mod robot;
mod scene;
mod surface;
mod ui;

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
    use crate::acoustics::SimulationState;
    use crate::agent::bridge::{create_bridge, SimBridgeClient};
    use crate::agent::{AgentServerConfig, AgentServerHandle};
    use crate::fluids::FluidSimulation;
    use crate::gas::GasSimulation;
    use crate::robot::RobotManager;
    use crate::scene::Scene;
    use crate::ui::ViewportState;
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
        agent_server_config: AgentServerConfig,
        agent_server_handle: Option<AgentServerHandle>,
    }

    impl EchoMapApp {
        pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
            let (bridge_server, bridge_client) = create_bridge();

            let agent_server_config = AgentServerConfig::default();
            let agent_server_handle = if agent_server_config.enabled {
                log::info!("Starting agent server (enabled by default config)");
                Some(crate::agent::start_agent_server(
                    agent_server_config.clone(),
                    bridge_server,
                ))
            } else {
                // Store the bridge server for later use when toggled on.
                // We need to keep it alive; store via Option dance.
                // Since bridge_server is Clone, we just drop it here — a new
                // bridge will be created when the server is toggled on.
                drop(bridge_server);
                None
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
                agent_server_config,
                agent_server_handle,
            }
        }
    }

    impl eframe::App for EchoMapApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // Process pending agent bridge commands each frame.
            self.bridge_client
                .process_pending(&mut self.robot_manager, &self.scene.meshes);

            crate::ui::menu_bar(
                ctx,
                &mut self.show_settings,
                &mut self.scene,
                &mut self.viewport,
            );
            crate::ui::toolbar(ctx, &mut self.viewport);
            crate::ui::side_panel(
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
            crate::ui::viewport_3d(
                ctx,
                &mut self.scene,
                &self.simulation,
                &mut self.viewport,
                &self.fluid_sim,
                &self.gas_sim,
            );
            crate::ui::status_bar(ctx, &self.viewport, &self.scene);

            // Step robot simulation (skip when agent server owns stepping via bridge)
            if self.agent_server_handle.is_none() {
                let dt = 1.0 / 60.0;
                self.robot_manager.step(dt, &self.scene.meshes);
            }

            if self.show_settings {
                crate::ui::settings_window(
                    ctx,
                    &mut self.show_settings,
                    &mut self.simulation,
                    &mut self.fluid_sim,
                    &mut self.gas_sim,
                );
            }
        }
    }
}
