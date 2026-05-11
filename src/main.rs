mod acoustics;
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
    }

    impl EchoMapApp {
        pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
            Self {
                scene: Scene::default(),
                simulation: SimulationState::default(),
                fluid_sim: FluidSimulation::default(),
                gas_sim: GasSimulation::default(),
                robot_manager: RobotManager::default(),
                viewport: ViewportState::default(),
                show_settings: false,
            }
        }
    }

    impl eframe::App for EchoMapApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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

            // Step robot simulation
            let dt = 1.0 / 60.0;
            self.robot_manager.step(dt, &self.scene.meshes);

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
