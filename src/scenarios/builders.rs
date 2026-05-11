//! Dedicated scenario builder types for integration testing.
//!
//! Each builder produces a fully initialized, deterministic scenario
//! with small grids (8x8x8 by default) suitable for CI.

use glam::Vec3;

use crate::fluids::solver::FluidConfig;
use crate::fluids::FluidSimulation;
use crate::gas::boundary::GasSource;
use crate::gas::grid::GasSpecies;
use crate::gas::solver::GasConfig;
use crate::gas::GasSimulation;
use crate::scene::material::{AcousticMaterial, MaterialLibrary, MediumLibrary};
use crate::scene::{Mesh, Scene, SceneObject, Triangle, Vertex};

// ---------------------------------------------------------------------------
// ScenarioConfig
// ---------------------------------------------------------------------------

/// Shared configuration parameters for scenario builders.
#[derive(Clone, Debug)]
pub struct ScenarioConfig {
    /// Grid cell size / spatial resolution (meters).
    pub resolution: f32,
    /// Simulation timestep (seconds).
    pub dt: f32,
    /// Number of grid cells per axis.
    pub grid_size: usize,
    /// Tolerance for numerical comparisons in tests.
    pub tolerance: f64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            resolution: 0.5,
            dt: 0.016,
            grid_size: 8,
            tolerance: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// FluidRoomScenario
// ---------------------------------------------------------------------------

/// A room filled with fluid, ready for simulation.
pub struct FluidRoomScenario {
    pub simulation: FluidSimulation,
    pub scene: Scene,
    pub config: ScenarioConfig,
}

impl FluidRoomScenario {
    /// Build a fluid room scenario with the given configuration.
    ///
    /// Creates a cubic bounding box from `(0,0,0)` to
    /// `(grid_size * resolution, grid_size * resolution, grid_size * resolution)`
    /// and initializes the fluid simulation grid inside it.
    pub fn build(config: &ScenarioConfig) -> Self {
        let extent = config.grid_size as f32 * config.resolution;
        let bounds = (Vec3::ZERO, Vec3::splat(extent));

        let scene = super::make_test_room(extent);

        let fluid_config = FluidConfig {
            dt: config.dt,
            ..FluidConfig::default()
        };

        let mut simulation = FluidSimulation::new(fluid_config);
        simulation.initialize(bounds, config.resolution, &[]);

        Self {
            simulation,
            scene,
            config: config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// GasLeakScenario
// ---------------------------------------------------------------------------

/// A gas leak scenario with a single CO2 source at the center.
pub struct GasLeakScenario {
    pub simulation: GasSimulation,
    pub scene: Scene,
    pub source_position: Vec3,
    pub config: ScenarioConfig,
}

impl GasLeakScenario {
    /// Build a gas leak scenario with the given configuration.
    ///
    /// Creates a gas simulation with a single CO2 species and places a
    /// source at the center of the domain.
    pub fn build(config: &ScenarioConfig) -> Self {
        let extent = config.grid_size as f32 * config.resolution;
        let bounds = (Vec3::ZERO, Vec3::splat(extent));
        let center = Vec3::splat(extent / 2.0);

        let scene = super::make_test_room(extent);

        let species = vec![GasSpecies {
            name: "CO2".to_string(),
            diffusion_coefficient: 0.16,
            molecular_weight: 44.0,
            density_at_stp: 1.842,
            color: [1.0, 0.0, 0.0],
        }];

        let gas_config = GasConfig {
            dt: config.dt,
            ..GasConfig::default()
        };

        let mut simulation = GasSimulation::new(gas_config);
        simulation.initialize(bounds, config.resolution, species, &[]);
        simulation.sources.push(GasSource {
            position: center,
            species_index: 0,
            rate: 10.0,
            radius: config.resolution * 1.5,
        });

        Self {
            simulation,
            scene,
            source_position: center,
            config: config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// UnderwaterAcousticsScenario
// ---------------------------------------------------------------------------

/// A scene configured with water as the background medium for underwater
/// acoustics testing.
pub struct UnderwaterAcousticsScenario {
    pub scene: Scene,
    pub config: ScenarioConfig,
}

impl UnderwaterAcousticsScenario {
    /// Build an underwater acoustics scenario.
    ///
    /// Creates a Scene with `background_medium` set to water from
    /// the default `MediumLibrary`.
    pub fn build(config: &ScenarioConfig) -> Self {
        let extent = config.grid_size as f32 * config.resolution;
        let mut scene = super::make_test_room(extent);

        let medium_lib = MediumLibrary::with_defaults();
        let water = medium_lib.get("Water").unwrap().clone();
        scene.background_medium = water;

        Self {
            scene,
            config: config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// SurfaceTraversalScenario
// ---------------------------------------------------------------------------

/// A scene with multiple surfaces of different materials for surface
/// interaction testing.
pub struct SurfaceTraversalScenario {
    pub scene: Scene,
    pub config: ScenarioConfig,
}

impl SurfaceTraversalScenario {
    /// Build a surface traversal scenario.
    ///
    /// Creates a Scene with three floor panels side by side, each using a
    /// different material preset (Concrete, Carpet, Glass).
    pub fn build(config: &ScenarioConfig) -> Self {
        let mat_lib = MaterialLibrary::with_defaults();
        let concrete = mat_lib.materials.get("Concrete").unwrap().clone();
        let carpet = mat_lib.materials.get("Carpet").unwrap().clone();
        let glass = mat_lib.materials.get("Glass").unwrap().clone();

        let panel_width = config.grid_size as f32 * config.resolution;
        let panel_depth = panel_width;

        let make_floor_panel =
            |name: &str, x_offset: f32, material: AcousticMaterial| -> SceneObject {
                let v = |x: f32, z: f32| Vertex {
                    position: Vec3::new(x, 0.0, z),
                    normal: Vec3::Y,
                };
                SceneObject {
                    name: name.to_string(),
                    mesh: Mesh {
                        triangles: vec![
                            Triangle {
                                vertices: [
                                    v(x_offset, 0.0),
                                    v(x_offset + panel_width, 0.0),
                                    v(x_offset + panel_width, panel_depth),
                                ],
                            },
                            Triangle {
                                vertices: [
                                    v(x_offset, 0.0),
                                    v(x_offset + panel_width, panel_depth),
                                    v(x_offset, panel_depth),
                                ],
                            },
                        ],
                    },
                    material,
                    visible: true,
                    interior_medium: None,
                }
            };

        let meshes = vec![
            make_floor_panel("Concrete Floor", 0.0, concrete),
            make_floor_panel("Carpet Floor", panel_width, carpet),
            make_floor_panel("Glass Floor", panel_width * 2.0, glass),
        ];

        let scene = Scene {
            meshes,
            ..Scene::default()
        };

        Self {
            scene,
            config: config.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::Medium;

    #[test]
    fn test_scenario_config_defaults() {
        let config = ScenarioConfig::default();
        assert!(
            (config.dt - 0.016).abs() < 1e-6,
            "Default dt should be 0.016, got {}",
            config.dt
        );
        assert!(
            (config.resolution - 0.5).abs() < 1e-6,
            "Default resolution should be 0.5, got {}",
            config.resolution
        );
        assert_eq!(
            config.grid_size, 8,
            "Default grid_size should be 8, got {}",
            config.grid_size
        );
        assert!(
            (config.tolerance - 0.1).abs() < 1e-10,
            "Default tolerance should be 0.1, got {}",
            config.tolerance
        );
    }

    #[test]
    fn test_fluid_room_builder() {
        let config = ScenarioConfig::default();
        let scenario = FluidRoomScenario::build(&config);

        // Grid should be allocated and initialized.
        assert!(
            scenario.simulation.grid.is_some(),
            "FluidRoomScenario should have an initialized grid"
        );

        let grid = scenario.simulation.grid.as_ref().unwrap();

        // With extent = 8 * 0.5 = 4.0, resolution = 0.5:
        // nx = ceil(4.0 / 0.5) = 8
        assert_eq!(
            grid.nx, config.grid_size,
            "Grid nx should match config grid_size"
        );
        assert_eq!(
            grid.ny, config.grid_size,
            "Grid ny should match config grid_size"
        );
        assert_eq!(
            grid.nz, config.grid_size,
            "Grid nz should match config grid_size"
        );

        // Scene should have 6 walls (box room).
        assert_eq!(
            scenario.scene.meshes.len(),
            6,
            "FluidRoomScenario scene should have 6 walls"
        );

        // Simulation dt should match config.
        assert!(
            (scenario.simulation.config.dt - config.dt).abs() < 1e-6,
            "Simulation dt should match config dt"
        );
    }

    #[test]
    fn test_gas_leak_builder() {
        let config = ScenarioConfig::default();
        let scenario = GasLeakScenario::build(&config);

        // Grid should be allocated.
        assert!(
            scenario.simulation.grid.is_some(),
            "GasLeakScenario should have an initialized grid"
        );

        let grid = scenario.simulation.grid.as_ref().unwrap();

        // Grid dimensions should match config.
        assert_eq!(grid.nx, config.grid_size);
        assert_eq!(grid.ny, config.grid_size);
        assert_eq!(grid.nz, config.grid_size);

        // Should have exactly one species (CO2).
        assert_eq!(
            grid.species.len(),
            1,
            "GasLeakScenario should have 1 species"
        );
        assert_eq!(grid.species[0].name, "CO2");

        // Source should be at the center of the domain.
        let extent = config.grid_size as f32 * config.resolution;
        let expected_center = Vec3::splat(extent / 2.0);
        assert!(
            (scenario.source_position - expected_center).length() < 1e-6,
            "Source should be at center: expected {:?}, got {:?}",
            expected_center,
            scenario.source_position
        );

        // Should have one source configured.
        assert_eq!(
            scenario.simulation.sources.len(),
            1,
            "GasLeakScenario should have 1 source"
        );
        assert_eq!(scenario.simulation.sources[0].species_index, 0);
    }

    #[test]
    fn test_underwater_acoustics_builder() {
        let config = ScenarioConfig::default();
        let scenario = UnderwaterAcousticsScenario::build(&config);

        // Background medium should be water.
        assert_eq!(
            scenario.scene.background_medium.medium_type,
            Medium::Liquid,
            "Background medium should be Liquid (water)"
        );
        assert_eq!(
            scenario.scene.background_medium.name, "Water",
            "Background medium name should be Water"
        );
        assert!(
            (scenario.scene.background_medium.density - 998.0).abs() < 0.1,
            "Water density should be ~998 kg/m3, got {}",
            scenario.scene.background_medium.density
        );
        assert!(
            (scenario.scene.background_medium.speed_of_sound - 1481.0).abs() < 0.1,
            "Water speed of sound should be ~1481 m/s, got {}",
            scenario.scene.background_medium.speed_of_sound
        );

        // Scene should have room walls.
        assert_eq!(
            scenario.scene.meshes.len(),
            6,
            "UnderwaterAcousticsScenario should have 6 walls"
        );
    }

    #[test]
    fn test_surface_traversal_builder() {
        let config = ScenarioConfig::default();
        let scenario = SurfaceTraversalScenario::build(&config);

        // Should have 3 floor panels with distinct materials.
        assert_eq!(
            scenario.scene.meshes.len(),
            3,
            "SurfaceTraversalScenario should have 3 material surfaces"
        );

        let names: Vec<&str> = scenario
            .scene
            .meshes
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert!(
            names.contains(&"Concrete Floor"),
            "Should contain Concrete Floor"
        );
        assert!(
            names.contains(&"Carpet Floor"),
            "Should contain Carpet Floor"
        );
        assert!(names.contains(&"Glass Floor"), "Should contain Glass Floor");

        // Materials should be distinct (different absorption profiles).
        let mat_names: Vec<&str> = scenario
            .scene
            .meshes
            .iter()
            .map(|m| m.material.name.as_str())
            .collect();
        assert_eq!(mat_names[0], "Concrete");
        assert_eq!(mat_names[1], "Carpet");
        assert_eq!(mat_names[2], "Glass");

        // Each panel should have 2 triangles.
        for obj in &scenario.scene.meshes {
            assert_eq!(
                obj.mesh.triangles.len(),
                2,
                "Each floor panel should have 2 triangles"
            );
        }
    }

    #[test]
    fn test_builders_deterministic() {
        let config = ScenarioConfig::default();

        // Build each scenario twice and compare key fields.

        // FluidRoomScenario
        let fluid_a = FluidRoomScenario::build(&config);
        let fluid_b = FluidRoomScenario::build(&config);
        let grid_a = fluid_a.simulation.grid.as_ref().unwrap();
        let grid_b = fluid_b.simulation.grid.as_ref().unwrap();
        assert_eq!(grid_a.nx, grid_b.nx, "FluidRoom: nx should be identical");
        assert_eq!(grid_a.ny, grid_b.ny, "FluidRoom: ny should be identical");
        assert_eq!(grid_a.nz, grid_b.nz, "FluidRoom: nz should be identical");
        assert!(
            (grid_a.dx - grid_b.dx).abs() < 1e-10,
            "FluidRoom: dx should be identical"
        );
        assert_eq!(
            fluid_a.scene.meshes.len(),
            fluid_b.scene.meshes.len(),
            "FluidRoom: mesh count should be identical"
        );

        // GasLeakScenario
        let gas_a = GasLeakScenario::build(&config);
        let gas_b = GasLeakScenario::build(&config);
        assert!(
            (gas_a.source_position - gas_b.source_position).length() < 1e-10,
            "GasLeak: source position should be identical"
        );
        let gg_a = gas_a.simulation.grid.as_ref().unwrap();
        let gg_b = gas_b.simulation.grid.as_ref().unwrap();
        assert_eq!(gg_a.species.len(), gg_b.species.len());
        assert_eq!(gg_a.species[0].name, gg_b.species[0].name);

        // UnderwaterAcousticsScenario
        let uw_a = UnderwaterAcousticsScenario::build(&config);
        let uw_b = UnderwaterAcousticsScenario::build(&config);
        assert!(
            (uw_a.scene.background_medium.density - uw_b.scene.background_medium.density).abs()
                < 1e-10,
            "Underwater: background density should be identical"
        );
        assert!(
            (uw_a.scene.background_medium.speed_of_sound
                - uw_b.scene.background_medium.speed_of_sound)
                .abs()
                < 1e-10,
            "Underwater: background speed_of_sound should be identical"
        );

        // SurfaceTraversalScenario
        let st_a = SurfaceTraversalScenario::build(&config);
        let st_b = SurfaceTraversalScenario::build(&config);
        assert_eq!(st_a.scene.meshes.len(), st_b.scene.meshes.len());
        for (a, b) in st_a.scene.meshes.iter().zip(st_b.scene.meshes.iter()) {
            assert_eq!(a.name, b.name, "Surface: mesh names should be identical");
            assert_eq!(
                a.material.name, b.material.name,
                "Surface: material names should be identical"
            );
            assert!(
                (a.material.friction_static - b.material.friction_static).abs() < 1e-10,
                "Surface: material friction should be identical"
            );
        }
    }
}
