pub mod boundary;
pub mod grid;
pub mod solver;

use glam::Vec3;

use self::boundary::{apply_sources, classify_cells, voxelize_scene, GasSource};
use self::grid::{GasGrid, GasSpecies};
use self::solver::GasConfig;
use crate::scene::SceneObject;

/// Top-level gas simulation state, following the same pattern as
/// `FluidSimulation` in the fluids module.
pub struct GasSimulation {
    pub config: GasConfig,
    pub grid: Option<GasGrid>,
    pub running: bool,
    pub frame: u32,
    pub elapsed_time: f32,
    pub sources: Vec<GasSource>,
}

#[allow(dead_code)]
impl GasSimulation {
    /// Create a new simulation with the given config but no grid allocated.
    pub fn new(config: GasConfig) -> Self {
        Self {
            config,
            grid: None,
            running: false,
            frame: 0,
            elapsed_time: 0.0,
            sources: Vec::new(),
        }
    }

    /// Initialize the simulation grid from scene bounds.
    ///
    /// - `bounds`: (min, max) of the simulation domain in world space.
    /// - `resolution`: cell size (dx).
    /// - `species`: gas species to simulate.
    /// - `meshes`: scene objects to voxelize as solid obstacles.
    ///
    /// After initialization the grid is voxelized, classified, and initial
    /// temperature is set to the config's ambient temperature.
    pub fn initialize(
        &mut self,
        bounds: (Vec3, Vec3),
        resolution: f32,
        species: Vec<GasSpecies>,
        meshes: &[SceneObject],
    ) {
        let (min, max) = bounds;
        let extent = max - min;

        // Compute grid dimensions from bounds and resolution.
        let nx = ((extent.x / resolution).ceil() as usize).max(1);
        let ny = ((extent.y / resolution).ceil() as usize).max(1);
        let nz = ((extent.z / resolution).ceil() as usize).max(1);

        let mut grid = GasGrid::new(nx, ny, nz, resolution, min, species);

        // Voxelize scene obstacles into solid cells.
        voxelize_scene(&mut grid, meshes);

        // Classify remaining cells as Gas.
        classify_cells(&mut grid);

        // Set initial temperature to ambient for all Gas cells.
        for idx in 0..grid.temperature.len() {
            if grid.cell_types[idx] == grid::GasCellType::Gas {
                grid.temperature[idx] = self.config.ambient_temperature;
            }
        }

        self.grid = Some(grid);
        self.frame = 0;
        self.elapsed_time = 0.0;
    }

    /// Advance the simulation by one timestep.
    ///
    /// Applies gas sources first, then runs the solver step.
    pub fn step(&mut self) {
        if let Some(ref mut grid) = self.grid {
            // Apply sources to inject concentration.
            apply_sources(grid, &self.sources, self.config.dt);
            // Run the full solver step (advect -> diffuse -> buoyancy -> pressure).
            solver::step(grid, &self.config);
            self.frame += 1;
            self.elapsed_time += self.config.dt;
        }
    }

    /// Reset to initial conditions: drop the grid and zero counters.
    pub fn reset(&mut self) {
        self.grid = None;
        self.frame = 0;
        self.elapsed_time = 0.0;
        self.running = false;
    }
}

impl Default for GasSimulation {
    fn default() -> Self {
        Self {
            config: GasConfig::default(),
            grid: None,
            running: false,
            frame: 0,
            elapsed_time: 0.0,
            sources: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_species(name: &str) -> GasSpecies {
        GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: 0.2,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn test_gas_simulation_new() {
        let sim = GasSimulation::new(GasConfig::default());
        assert!(sim.grid.is_none(), "New simulation should have no grid");
        assert!(!sim.running, "New simulation should not be running");
        assert_eq!(sim.frame, 0, "New simulation frame should be 0");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "New simulation elapsed_time should be 0.0"
        );
        assert!(
            sim.sources.is_empty(),
            "New simulation should have no sources"
        );
    }

    #[test]
    fn test_gas_simulation_initialize() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        let resolution = 0.5;
        let species = vec![make_species("CO2"), make_species("CH4")];

        sim.initialize(bounds, resolution, species, &[]);

        assert!(
            sim.grid.is_some(),
            "Grid should be allocated after initialize"
        );
        let grid = sim.grid.as_ref().unwrap();

        // 2.0 / 0.5 = 4, 3.0 / 0.5 = 6, 4.0 / 0.5 = 8
        assert_eq!(grid.nx, 4, "nx should be ceil(2.0/0.5) = 4");
        assert_eq!(grid.ny, 6, "ny should be ceil(3.0/0.5) = 6");
        assert_eq!(grid.nz, 8, "nz should be ceil(4.0/0.5) = 8");

        assert!(
            (grid.origin - Vec3::ZERO).length() < 1e-6,
            "Grid origin should match bounds min"
        );
        assert!(
            (grid.dx - 0.5).abs() < 1e-6,
            "Grid dx should match resolution"
        );

        // Should have 2 species
        assert_eq!(grid.species.len(), 2, "Should have 2 species");
        assert_eq!(
            grid.concentrations.len(),
            2,
            "Should have 2 concentration arrays"
        );
    }

    #[test]
    fn test_gas_simulation_step_advances_frame() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let species = vec![make_species("Air")];
        sim.initialize(bounds, 0.5, species, &[]);

        assert_eq!(sim.frame, 0, "Frame should start at 0");

        sim.step();
        assert_eq!(sim.frame, 1, "Frame should be 1 after one step");

        sim.step();
        assert_eq!(sim.frame, 2, "Frame should be 2 after two steps");

        assert!(
            (sim.elapsed_time - 2.0 * sim.config.dt).abs() < 1e-6,
            "Elapsed time should be 2 * dt"
        );
    }

    #[test]
    fn test_gas_simulation_reset() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        let species = vec![make_species("Air")];
        sim.initialize(bounds, 0.5, species, &[]);
        sim.running = true;
        sim.step();
        sim.step();

        assert!(sim.grid.is_some(), "Grid should exist before reset");
        assert!(sim.frame > 0, "Frame should be > 0 before reset");

        sim.reset();

        assert!(sim.grid.is_none(), "Grid should be None after reset");
        assert_eq!(sim.frame, 0, "Frame should be 0 after reset");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "Elapsed time should be 0.0 after reset"
        );
        assert!(!sim.running, "Simulation should not be running after reset");
    }

    // ---- Q3 Edge Case Tests ----

    #[test]
    fn test_edge_step_no_grid() {
        // step() with no grid initialized should be a no-op (not panic)
        let mut sim = GasSimulation::new(GasConfig::default());
        assert!(sim.grid.is_none());

        sim.step();

        assert_eq!(sim.frame, 0, "Frame should stay 0 with no grid");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "Elapsed time should stay 0 with no grid"
        );
    }

    #[test]
    fn test_edge_default_simulation() {
        let sim = GasSimulation::default();
        assert!(sim.grid.is_none());
        assert!(!sim.running);
        assert_eq!(sim.frame, 0);
        assert!((sim.elapsed_time - 0.0).abs() < 1e-6);
        assert!(sim.sources.is_empty());
        // Verify default config values
        assert!((sim.config.dt - 0.016).abs() < 1e-6);
        assert!((sim.config.ambient_temperature - 293.15).abs() < 1e-3);
    }

    #[test]
    fn test_edge_reinitialize() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let species1 = vec![make_species("CO2")];
        let bounds1 = (Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        sim.initialize(bounds1, 0.5, species1, &[]);
        sim.step();
        sim.step();
        assert_eq!(sim.frame, 2);

        // Re-initialize with different parameters
        let species2 = vec![make_species("CH4"), make_species("N2O")];
        let bounds2 = (Vec3::ZERO, Vec3::new(3.0, 3.0, 3.0));
        sim.initialize(bounds2, 1.0, species2, &[]);

        // Frame and elapsed time should be reset
        assert_eq!(sim.frame, 0, "Frame should reset on re-initialize");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "Elapsed time should reset on re-initialize"
        );

        let grid = sim.grid.as_ref().unwrap();
        assert_eq!(grid.species.len(), 2, "Should have new species count");
        assert_eq!(grid.nx, 3, "Should have new dimensions");
    }

    #[test]
    fn test_edge_step_with_sources() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let species = vec![make_species("CO2")];
        let bounds = (Vec3::ZERO, Vec3::new(4.0, 4.0, 4.0));
        sim.initialize(bounds, 1.0, species, &[]);

        // Add a source
        let grid = sim.grid.as_ref().unwrap();
        let source_pos = grid.cell_center(2, 2, 2);
        sim.sources.push(GasSource {
            position: source_pos,
            species_index: 0,
            rate: 10.0,
            radius: 1.5,
        });

        // Step should apply sources then run solver
        sim.step();

        let grid = sim.grid.as_ref().unwrap();
        let center_idx = grid.idx(2, 2, 2);
        assert!(
            grid.concentrations[0][center_idx] > 0.0,
            "Source should inject concentration during step"
        );
        assert_eq!(sim.frame, 1);
    }

    #[test]
    fn test_edge_initialize_tiny_extent() {
        // Very small extent should still produce at least 1x1x1 grid
        let mut sim = GasSimulation::new(GasConfig::default());
        let species = vec![make_species("Air")];
        let bounds = (Vec3::ZERO, Vec3::new(0.01, 0.01, 0.01));
        sim.initialize(bounds, 1.0, species, &[]);

        let grid = sim.grid.as_ref().unwrap();
        assert!(grid.nx >= 1, "nx should be at least 1");
        assert!(grid.ny >= 1, "ny should be at least 1");
        assert!(grid.nz >= 1, "nz should be at least 1");
    }

    #[test]
    fn test_edge_initialize_sets_ambient_temperature() {
        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 350.0,
            thermal_diffusivity: 0.02,
            buoyancy_coefficient: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };
        let mut sim = GasSimulation::new(config);
        let species = vec![make_species("Air")];
        let bounds = (Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        sim.initialize(bounds, 1.0, species, &[]);

        let grid = sim.grid.as_ref().unwrap();
        // All Gas cells should have ambient temperature
        for idx in 0..grid.temperature.len() {
            if grid.cell_types[idx] == grid::GasCellType::Gas {
                assert!(
                    (grid.temperature[idx] - 350.0).abs() < 1e-3,
                    "Gas cell should have ambient temp 350.0, got {}",
                    grid.temperature[idx]
                );
            }
        }
    }

    #[test]
    fn test_edge_reset_then_step() {
        let mut sim = GasSimulation::new(GasConfig::default());
        let species = vec![make_species("Air")];
        sim.initialize((Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0)), 1.0, species, &[]);
        sim.step();
        sim.reset();

        // Step after reset should be a no-op (no grid)
        sim.step();
        assert_eq!(
            sim.frame, 0,
            "Step after reset with no grid should be no-op"
        );
    }
}
