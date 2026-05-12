//! Scenario presets and factory functions for integration testing.
//!
//! Provides pre-configured scenes, materials, fluid/gas configs, and robot
//! definitions that exercise multiple subsystems together.

pub mod builders;
pub mod cross_system;
pub mod fluid_validation;
pub mod gas_validation;
pub mod surface_validation;
pub mod underwater_acoustics;

use glam::Vec3;

use crate::fluids::solver::FluidConfig;
use crate::gas::grid::GasSpecies;
use crate::gas::solver::GasConfig;
use crate::robot::definition::{
    CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition, SensorDefinition,
    SensorMount,
};
use crate::scene::material::{AcousticMaterial, FrequencyBands};
use crate::scene::{Mesh, Scene, SceneObject, Triangle, Vertex};

// ---------------------------------------------------------------------------
// ScenarioPreset
// ---------------------------------------------------------------------------

/// A complete preset scenario bundling a scene with fluid, gas, and robot
/// configuration for integration testing.
pub struct ScenarioPreset {
    pub scene: Scene,
    pub fluid_config: FluidConfig,
    pub gas_config: GasConfig,
    pub robot_definition: RobotDefinition,
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create a box room scene with 6 wall meshes (floor, ceiling, 4 walls).
///
/// Each wall is a separate `SceneObject` with a triangulated quad (2 triangles).
/// The room spans from `(0, 0, 0)` to `(size, size, size)`.
///
/// A `size` of 0.0 produces a degenerate room with zero-area triangles.
pub fn make_test_room(size: f32) -> Scene {
    let s = size;
    let mat = make_default_material();

    // 8 corners of the box
    let p = [
        Vec3::new(0.0, 0.0, 0.0), // 0
        Vec3::new(s, 0.0, 0.0),   // 1
        Vec3::new(s, 0.0, s),     // 2
        Vec3::new(0.0, 0.0, s),   // 3
        Vec3::new(0.0, s, 0.0),   // 4
        Vec3::new(s, s, 0.0),     // 5
        Vec3::new(s, s, s),       // 6
        Vec3::new(0.0, s, s),     // 7
    ];

    let make_wall = |name: &str, a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3| -> SceneObject {
        let v = |pos: Vec3| Vertex {
            position: pos,
            normal,
        };
        SceneObject {
            name: name.to_string(),
            mesh: Mesh {
                triangles: vec![
                    Triangle {
                        vertices: [v(a), v(b), v(c)],
                    },
                    Triangle {
                        vertices: [v(a), v(c), v(d)],
                    },
                ],
            },
            material: mat.clone(),
            visible: true,
            interior_medium: None,
        }
    };

    let meshes = vec![
        // Floor (y=0, normal up)
        make_wall("Floor", p[0], p[1], p[2], p[3], Vec3::Y),
        // Ceiling (y=s, normal down)
        make_wall("Ceiling", p[4], p[7], p[6], p[5], Vec3::NEG_Y),
        // Front wall (z=0, normal +Z)
        make_wall("Front Wall", p[0], p[4], p[5], p[1], Vec3::Z),
        // Back wall (z=s, normal -Z)
        make_wall("Back Wall", p[2], p[6], p[7], p[3], Vec3::NEG_Z),
        // Left wall (x=0, normal +X)
        make_wall("Left Wall", p[3], p[7], p[4], p[0], Vec3::X),
        // Right wall (x=s, normal -X)
        make_wall("Right Wall", p[1], p[5], p[6], p[2], Vec3::NEG_X),
    ];

    Scene {
        meshes,
        ..Scene::default()
    }
}

/// Create a boxing ring scene centered at the origin.
///
/// The ring is a flat floor with 4 low walls (1.0m high) forming a square
/// boundary, representing ropes. The ring spans from `(-size/2, 0, -size/2)`
/// to `(size/2, wall_height, size/2)` on the XZ plane with the floor at y=0.
///
/// Returns a `Scene` with 5 `SceneObject`s: 1 floor + 4 walls.
/// A default `size` of 6.0 meters is typical for a boxing ring.
pub fn make_boxing_ring(size: f32) -> Scene {
    let h = size / 2.0;
    let wall_height: f32 = 1.0;
    let mat = make_default_material();

    // Floor corners (y=0)
    let f0 = Vec3::new(-h, 0.0, -h);
    let f1 = Vec3::new(h, 0.0, -h);
    let f2 = Vec3::new(h, 0.0, h);
    let f3 = Vec3::new(-h, 0.0, h);

    // Top-of-wall corners (y=wall_height)
    let t0 = Vec3::new(-h, wall_height, -h);
    let t1 = Vec3::new(h, wall_height, -h);
    let t2 = Vec3::new(h, wall_height, h);
    let t3 = Vec3::new(-h, wall_height, h);

    let make_wall = |name: &str, a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3| -> SceneObject {
        let v = |pos: Vec3| Vertex {
            position: pos,
            normal,
        };
        SceneObject {
            name: name.to_string(),
            mesh: Mesh {
                triangles: vec![
                    Triangle {
                        vertices: [v(a), v(b), v(c)],
                    },
                    Triangle {
                        vertices: [v(a), v(c), v(d)],
                    },
                ],
            },
            material: mat.clone(),
            visible: true,
            interior_medium: None,
        }
    };

    let meshes = vec![
        // Floor (y=0, normal up)
        make_wall("Floor", f0, f1, f2, f3, Vec3::Y),
        // Front wall (z=-h, normal +Z into ring)
        make_wall("Front Wall", f0, t0, t1, f1, Vec3::Z),
        // Back wall (z=+h, normal -Z into ring)
        make_wall("Back Wall", f2, t2, t3, f3, Vec3::NEG_Z),
        // Left wall (x=-h, normal +X into ring)
        make_wall("Left Wall", f3, t3, t0, f0, Vec3::X),
        // Right wall (x=+h, normal -X into ring)
        make_wall("Right Wall", f1, t1, t2, f2, Vec3::NEG_X),
    ];

    Scene {
        meshes,
        ..Scene::default()
    }
}

/// Create a default acoustic material with known physical properties.
///
/// Uses concrete-like values matching `AcousticMaterial::default()` but
/// with explicit, documented values for test assertions.
pub fn make_default_material() -> AcousticMaterial {
    AcousticMaterial {
        name: "Test Concrete".into(),
        absorption: FrequencyBands {
            hz_125: 0.01,
            hz_250: 0.01,
            hz_500: 0.02,
            hz_1000: 0.02,
            hz_2000: 0.02,
            hz_4000: 0.03,
        },
        scattering: 0.1,
        color: [0.7, 0.7, 0.7],
        friction_static: 0.6,
        friction_kinetic: 0.5,
        roughness: 0.002,
        porosity: 0.15,
        permeability: 1e-15,
        contact_angle: std::f32::consts::FRAC_PI_4,
    }
}

/// Create a simple robot definition with 3 links, 2 joints, and 1 distance sensor.
///
/// - Link 0: base (cuboid)
/// - Link 1: arm segment (cylinder), connected by revolute joint 0 around Y
/// - Link 2: end effector (sphere), connected by revolute joint 1 around Y
/// - Sensor: distance sensor on link 2 pointing along +Z with 50m range
pub fn make_simple_robot() -> RobotDefinition {
    RobotDefinition {
        name: "test_robot".to_string(),
        links: vec![
            LinkDefinition {
                name: "base".to_string(),
                mass: 5.0,
                inertia: 1.0,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: Vec3::splat(0.1),
                },
                parent_joint: None,
                body_zone: None,
            },
            LinkDefinition {
                name: "arm".to_string(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.5,
                },
                parent_joint: Some(0),
                body_zone: None,
            },
            LinkDefinition {
                name: "end_effector".to_string(),
                mass: 0.5,
                inertia: 0.05,
                collision_shape: CollisionShape::Sphere { radius: 0.03 },
                parent_joint: Some(1),
                body_zone: None,
            },
        ],
        joints: vec![
            JointDefinition {
                name: "shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
            },
            JointDefinition {
                name: "elbow".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 1,
                child_link: 2,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 5.0,
                damping: 0.1,
            },
        ],
        sensors: vec![SensorMount {
            link_index: 2,
            local_offset: Vec3::ZERO,
            sensor: SensorDefinition::Distance {
                direction: Vec3::Z,
                max_range: 50.0,
            },
        }],
    }
}

/// Create a fluid config with sensible test defaults.
///
/// Uses small timestep and moderate resolution suitable for integration tests.
pub fn make_fluid_config() -> FluidConfig {
    FluidConfig {
        dt: 0.016,
        viscosity: 0.001,
        density: 1000.0,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        surface_tension: 0.0,
        jacobi_iterations: 40,
    }
}

/// Create a gas config with sensible test defaults.
///
/// Uses standard temperature and small timestep suitable for integration tests.
pub fn make_gas_config() -> GasConfig {
    GasConfig {
        dt: 0.016,
        ambient_temperature: 293.15,
        thermal_diffusivity: 2.2e-5,
        buoyancy_coefficient: 0.01,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    }
}

/// Create a default gas species for testing (CO2-like).
pub fn make_test_species() -> GasSpecies {
    GasSpecies {
        name: "CO2".to_string(),
        diffusion_coefficient: 0.16,
        molecular_weight: 44.0,
        density_at_stp: 1.842,
        color: [1.0, 0.0, 0.0],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_test_room_has_walls() {
        let scene = make_test_room(2.0);
        assert_eq!(
            scene.meshes.len(),
            6,
            "Room should have 6 wall SceneObjects (floor, ceiling, 4 walls)"
        );

        // Each wall has 2 triangles, all with nonzero area
        for (i, obj) in scene.meshes.iter().enumerate() {
            assert_eq!(
                obj.mesh.triangles.len(),
                2,
                "Wall {i} ({}) should have 2 triangles",
                obj.name
            );
            for (j, tri) in obj.mesh.triangles.iter().enumerate() {
                let area = tri.area();
                assert!(
                    area > 0.0,
                    "Wall {i} ({}) triangle {j} should have nonzero area, got {area}",
                    obj.name
                );
            }
        }
    }

    #[test]
    fn test_make_default_material_properties() {
        let mat = make_default_material();
        assert!(
            mat.friction_static > 0.0,
            "friction_static should be positive, got {}",
            mat.friction_static
        );
        assert!(
            mat.roughness > 0.0,
            "roughness should be positive, got {}",
            mat.roughness
        );
        // All numeric properties should be finite
        assert!(mat.friction_static.is_finite());
        assert!(mat.friction_kinetic.is_finite());
        assert!(mat.roughness.is_finite());
        assert!(mat.porosity.is_finite());
        assert!(mat.permeability.is_finite());
        assert!(mat.contact_angle.is_finite());
        assert!(mat.scattering.is_finite());
        // Static friction >= kinetic friction
        assert!(
            mat.friction_static >= mat.friction_kinetic,
            "friction_static ({}) should be >= friction_kinetic ({})",
            mat.friction_static,
            mat.friction_kinetic
        );
    }

    #[test]
    fn test_make_simple_robot_structure() {
        let def = make_simple_robot();
        assert_eq!(def.links.len(), 3, "Should have 3 links");
        assert_eq!(def.joints.len(), 2, "Should have 2 joints");
        assert!(!def.sensors.is_empty(), "Should have at least 1 sensor");

        // Verify base link has no parent joint
        assert!(
            def.links[0].parent_joint.is_none(),
            "Base link should have no parent joint"
        );

        // Verify child links have parent joints
        for link in &def.links[1..] {
            assert!(
                link.parent_joint.is_some(),
                "Non-base link '{}' should have a parent joint",
                link.name
            );
        }

        // Verify sensor is a distance sensor
        match &def.sensors[0].sensor {
            SensorDefinition::Distance { max_range, .. } => {
                assert!(
                    *max_range > 0.0,
                    "Distance sensor max_range should be positive"
                );
            }
            other => panic!("Expected Distance sensor, got {:?}", other),
        }
    }

    #[test]
    fn test_scenario_preset_construction() {
        let preset = ScenarioPreset {
            scene: make_test_room(3.0),
            fluid_config: make_fluid_config(),
            gas_config: make_gas_config(),
            robot_definition: make_simple_robot(),
        };

        // All fields accessible without panic
        assert_eq!(preset.scene.meshes.len(), 6);
        assert!(preset.fluid_config.dt > 0.0);
        assert!(preset.gas_config.dt > 0.0);
        assert_eq!(preset.robot_definition.joints.len(), 2);
    }

    #[test]
    fn test_make_fluid_config_defaults() {
        let config = make_fluid_config();
        assert!(
            config.density > 0.0,
            "density should be positive, got {}",
            config.density
        );
        assert!(
            config.viscosity > 0.0,
            "viscosity should be positive, got {}",
            config.viscosity
        );
        assert!(config.dt > 0.0, "dt should be positive, got {}", config.dt);
        assert!(config.density.is_finite());
        assert!(config.viscosity.is_finite());
        assert!(config.dt.is_finite());
    }

    #[test]
    fn test_make_gas_config_defaults() {
        let config = make_gas_config();
        assert!(config.dt > 0.0, "dt should be positive, got {}", config.dt);
        assert!(
            config.ambient_temperature > 0.0,
            "ambient_temperature should be positive, got {}",
            config.ambient_temperature
        );
        assert!(config.dt.is_finite());
        assert!(config.ambient_temperature.is_finite());
        assert!(config.thermal_diffusivity.is_finite());
        assert!(config.buoyancy_coefficient.is_finite());
    }

    #[test]
    fn test_make_test_room_empty_scene() {
        // Degenerate case: size=0 should not panic, no negative dimensions
        let scene = make_test_room(0.0);
        assert_eq!(
            scene.meshes.len(),
            6,
            "Even degenerate room should have 6 walls"
        );

        // All vertex positions should be non-negative (room starts at origin)
        for obj in &scene.meshes {
            for tri in &obj.mesh.triangles {
                for v in &tri.vertices {
                    assert!(
                        v.position.x >= 0.0 && v.position.y >= 0.0 && v.position.z >= 0.0,
                        "Degenerate room should not have negative coordinates, got {:?}",
                        v.position
                    );
                }
            }
        }
    }

    #[test]
    fn test_boxing_ring_has_floor_and_walls() {
        let scene = make_boxing_ring(6.0);
        assert_eq!(
            scene.meshes.len(),
            5,
            "Boxing ring should have 5 SceneObjects (1 floor + 4 walls)"
        );

        // Verify names
        assert_eq!(scene.meshes[0].name, "Floor");
        assert_eq!(scene.meshes[1].name, "Front Wall");
        assert_eq!(scene.meshes[2].name, "Back Wall");
        assert_eq!(scene.meshes[3].name, "Left Wall");
        assert_eq!(scene.meshes[4].name, "Right Wall");

        // Each object has 2 triangles with nonzero area
        for (i, obj) in scene.meshes.iter().enumerate() {
            assert_eq!(
                obj.mesh.triangles.len(),
                2,
                "Object {i} ({}) should have 2 triangles",
                obj.name
            );
            for (j, tri) in obj.mesh.triangles.iter().enumerate() {
                let area = tri.area();
                assert!(
                    area > 0.0,
                    "Object {i} ({}) triangle {j} should have nonzero area, got {area}",
                    obj.name
                );
            }
        }
    }

    #[test]
    fn test_boxing_ring_dimensions() {
        let size = 8.0_f32;
        let half = size / 2.0;
        let wall_height = 1.0_f32;
        let scene = make_boxing_ring(size);

        // All vertex positions should be within [-half, half] on X and Z,
        // and within [0, wall_height] on Y.
        for obj in &scene.meshes {
            for tri in &obj.mesh.triangles {
                for v in &tri.vertices {
                    assert!(
                        v.position.x >= -half && v.position.x <= half,
                        "Vertex x={} should be within [-{half}, {half}]",
                        v.position.x
                    );
                    assert!(
                        v.position.z >= -half && v.position.z <= half,
                        "Vertex z={} should be within [-{half}, {half}]",
                        v.position.z
                    );
                    assert!(
                        v.position.y >= 0.0 && v.position.y <= wall_height,
                        "Vertex y={} should be within [0, {wall_height}]",
                        v.position.y
                    );
                }
            }
        }

        // Verify ring spans the full extent: collect all unique x and z values
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for obj in &scene.meshes {
            for tri in &obj.mesh.triangles {
                for v in &tri.vertices {
                    min_x = min_x.min(v.position.x);
                    max_x = max_x.max(v.position.x);
                    min_z = min_z.min(v.position.z);
                    max_z = max_z.max(v.position.z);
                }
            }
        }
        assert!(
            (min_x - (-half)).abs() < f32::EPSILON,
            "Min x should be -{half}, got {min_x}"
        );
        assert!(
            (max_x - half).abs() < f32::EPSILON,
            "Max x should be {half}, got {max_x}"
        );
        assert!(
            (min_z - (-half)).abs() < f32::EPSILON,
            "Min z should be -{half}, got {min_z}"
        );
        assert!(
            (max_z - half).abs() < f32::EPSILON,
            "Max z should be {half}, got {max_z}"
        );
    }
}
