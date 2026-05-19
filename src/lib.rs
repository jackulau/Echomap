pub mod acoustics;
pub mod agent;
pub mod fluids;
pub mod gas;
pub mod io;
pub mod renderer;
pub mod robot;
pub mod scenarios;
pub mod scene;
pub mod surface;
pub mod teleop;
pub mod ui;

pub mod benchmarks;

#[cfg(test)]
mod tests {
    use super::fluids::FluidSimulation;
    use super::gas::GasSimulation;
    use super::scene::material::AcousticMaterial;
    use super::scene::Scene;
    use super::surface::SurfaceInteraction;

    #[test]
    fn test_lib_modules_accessible() {
        // Verify key types from each module are importable by constructing them.
        let _scene = Scene::default();
        let _fluid = FluidSimulation::default();
        let _gas = GasSimulation::default();
        let _surface = SurfaceInteraction::from_material(&AcousticMaterial::default());
    }

    #[test]
    fn test_scene_default() {
        let scene = Scene::default();
        assert!(
            scene.meshes.is_empty(),
            "default scene should have no meshes"
        );
    }

    #[test]
    fn test_fluid_sim_default() {
        let sim = FluidSimulation::default();
        assert!(!sim.running, "default fluid sim should not be running");
    }
}
