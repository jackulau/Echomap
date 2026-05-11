pub mod boundary;
pub mod grid;
pub mod solver;

use glam::Vec3;

use self::boundary::{classify_cells, voxelize_scene};
use self::grid::FluidGrid;
use self::solver::FluidConfig;
use crate::scene::SceneObject;

/// Top-level fluid simulation state, following the same pattern as
/// `SimulationState` in the acoustics module.
pub struct FluidSimulation {
    pub config: FluidConfig,
    pub grid: Option<FluidGrid>,
    pub running: bool,
    pub frame: u32,
    pub elapsed_time: f32,
}

impl FluidSimulation {
    /// Create a new simulation with the given config but no grid allocated.
    pub fn new(config: FluidConfig) -> Self {
        Self {
            config,
            grid: None,
            running: false,
            frame: 0,
            elapsed_time: 0.0,
        }
    }

    /// Initialize the simulation grid from scene bounds.
    ///
    /// - `bounds`: (min, max) of the simulation domain in world space.
    /// - `resolution`: cell size (dx).
    /// - `meshes`: scene objects to voxelize as solid obstacles.
    ///
    /// After initialization the grid is classified (level set -> cell types)
    /// and ready for stepping.
    pub fn initialize(&mut self, bounds: (Vec3, Vec3), resolution: f32, meshes: &[SceneObject]) {
        let (min, max) = bounds;
        let extent = max - min;

        // Compute grid dimensions from bounds and resolution.
        let nx = ((extent.x / resolution).ceil() as usize).max(1);
        let ny = ((extent.y / resolution).ceil() as usize).max(1);
        let nz = ((extent.z / resolution).ceil() as usize).max(1);

        let mut grid = FluidGrid::new(nx, ny, nz, resolution, min);

        // Voxelize scene obstacles into solid cells.
        voxelize_scene(&mut grid, meshes);

        // Initialize level set: all cells negative (fluid) by default, then
        // classify to set cell types.
        for ls in grid.level_set.iter_mut() {
            *ls = -1.0;
        }
        classify_cells(&mut grid);

        // Set density to reference density for all fluid cells.
        for i in 0..grid.density.len() {
            if grid.cell_types[i] == grid::CellType::Fluid {
                grid.density[i] = self.config.density;
            }
        }

        self.grid = Some(grid);
        self.frame = 0;
        self.elapsed_time = 0.0;
    }

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) {
        if let Some(ref mut grid) = self.grid {
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

impl Default for FluidSimulation {
    fn default() -> Self {
        Self {
            config: FluidConfig::default(),
            grid: None,
            running: false,
            frame: 0,
            elapsed_time: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_simulation_new() {
        let sim = FluidSimulation::new(FluidConfig::default());
        assert!(sim.grid.is_none(), "New simulation should have no grid");
        assert!(!sim.running, "New simulation should not be running");
        assert_eq!(sim.frame, 0, "New simulation frame should be 0");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "New simulation elapsed_time should be 0.0"
        );
    }

    #[test]
    fn test_fluid_simulation_initialize() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(2.0, 3.0, 4.0));
        let resolution = 0.5;

        sim.initialize(bounds, resolution, &[]);

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
    }

    #[test]
    fn test_fluid_simulation_step_advances_frame() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        sim.initialize(bounds, 0.5, &[]);

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
    fn test_fluid_simulation_reset_clears_state() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        let bounds = (Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        sim.initialize(bounds, 0.5, &[]);
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

    // =========================================================================
    // Edge case tests
    // =========================================================================

    // --- Default construction ---

    #[test]
    fn test_fluid_simulation_default() {
        let sim = FluidSimulation::default();
        assert!(sim.grid.is_none());
        assert!(!sim.running);
        assert_eq!(sim.frame, 0);
        assert!((sim.elapsed_time - 0.0).abs() < 1e-6);
    }

    // --- Step without initialize (no grid) is a no-op ---

    #[test]
    fn test_step_without_grid_is_noop() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        assert!(sim.grid.is_none());
        sim.step();
        assert_eq!(sim.frame, 0, "Step with no grid should not advance frame");
        assert!(
            (sim.elapsed_time - 0.0).abs() < 1e-6,
            "Step with no grid should not advance time"
        );
    }

    // --- Double reset ---

    #[test]
    fn test_double_reset() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        sim.initialize((Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)), 0.5, &[]);
        sim.step();
        sim.reset();
        sim.reset(); // second reset should be safe
        assert!(sim.grid.is_none());
        assert_eq!(sim.frame, 0);
    }

    // --- Initialize with very small extent ---

    #[test]
    fn test_initialize_tiny_extent() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        // Extent = (0.001, 0.001, 0.001), resolution = 0.5
        // Each axis: ceil(0.001/0.5) = 1 => 1x1x1 grid
        sim.initialize((Vec3::ZERO, Vec3::new(0.001, 0.001, 0.001)), 0.5, &[]);
        let grid = sim.grid.as_ref().unwrap();
        assert_eq!(grid.nx, 1);
        assert_eq!(grid.ny, 1);
        assert_eq!(grid.nz, 1);
    }

    // --- Initialize then step on a 1x1x1 grid ---

    #[test]
    fn test_initialize_and_step_1x1x1() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        sim.initialize((Vec3::ZERO, Vec3::new(0.5, 0.5, 0.5)), 0.5, &[]);
        sim.step();
        assert_eq!(sim.frame, 1);
        let grid = sim.grid.as_ref().unwrap();
        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "u should be finite after step on 1x1x1"
        );
        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "v should be finite after step on 1x1x1"
        );
        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "pressure should be finite after step on 1x1x1"
        );
    }

    // --- Initialize, step, reset, re-initialize, step ---

    #[test]
    fn test_reinitialize_after_reset() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        sim.initialize((Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)), 0.5, &[]);
        sim.step();
        sim.step();
        assert_eq!(sim.frame, 2);

        sim.reset();
        assert!(sim.grid.is_none());

        sim.initialize((Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0)), 0.5, &[]);
        assert_eq!(sim.frame, 0);
        let grid = sim.grid.as_ref().unwrap();
        assert_eq!(grid.nx, 4);
        assert_eq!(grid.ny, 4);
        assert_eq!(grid.nz, 4);

        sim.step();
        assert_eq!(sim.frame, 1);
    }

    // --- Elapsed time accumulation ---

    #[test]
    fn test_elapsed_time_accumulates() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        sim.initialize((Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0)), 0.5, &[]);
        let dt = sim.config.dt;
        for i in 1..=10 {
            sim.step();
            let expected = dt * i as f32;
            assert!(
                (sim.elapsed_time - expected).abs() < 1e-4,
                "After {i} steps, elapsed_time should be {expected}, got {}",
                sim.elapsed_time
            );
        }
    }

    // --- Initialize with non-zero origin ---

    #[test]
    fn test_initialize_with_offset_origin() {
        let mut sim = FluidSimulation::new(FluidConfig::default());
        let min = Vec3::new(10.0, 20.0, 30.0);
        let max = Vec3::new(12.0, 23.0, 34.0);
        sim.initialize((min, max), 1.0, &[]);
        let grid = sim.grid.as_ref().unwrap();
        assert_eq!(grid.nx, 2);
        assert_eq!(grid.ny, 3);
        assert_eq!(grid.nz, 4);
        assert!(
            (grid.origin - min).length() < 1e-6,
            "Grid origin should match bounds min"
        );
    }
}
