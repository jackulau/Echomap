#[allow(dead_code)]
pub mod actuators;
#[allow(dead_code)]
pub mod body;
#[allow(dead_code)]
pub mod collision;
#[allow(dead_code)]
pub mod definition;
#[allow(dead_code)]
pub mod dynamics;
#[allow(dead_code)]
pub mod kinematics;
#[allow(dead_code)]
pub mod sensors;
#[allow(dead_code)]
pub mod state;

#[allow(unused_imports)]
pub use actuators::*;
#[allow(unused_imports)]
pub use body::*;
#[allow(unused_imports)]
pub use collision::*;
#[allow(unused_imports)]
pub use definition::*;
#[allow(unused_imports)]
pub use dynamics::*;
#[allow(unused_imports)]
pub use kinematics::*;
#[allow(unused_imports)]
pub use sensors::*;
#[allow(unused_imports)]
pub use state::*;

use glam::Mat4;
use serde::{Deserialize, Serialize};

use crate::scene::SceneObject;
use definition::RobotDefinition;
use dynamics::step_dynamics;
use kinematics::forward_kinematics;
use sensors::simulate_sensors;
use state::{ActuatorCommand, RobotState};

// ---------------------------------------------------------------------------
// ManagedRobot — wraps a definition + mutable state + base pose
// ---------------------------------------------------------------------------

/// A robot instance managed by the RobotManager.
///
/// Bundles a static `RobotDefinition` with the mutable `RobotState` and a
/// world-space base pose. The base pose is stored as `[f32; 16]` for serde
/// compatibility; use `base_pose_mat4()` to get a `glam::Mat4`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ManagedRobot {
    pub definition: RobotDefinition,
    pub state: RobotState,
    pub base_pose: [f32; 16],
}

#[allow(dead_code)]
impl ManagedRobot {
    /// Return the base pose as a `glam::Mat4`.
    pub fn base_pose_mat4(&self) -> Mat4 {
        Mat4::from_cols_array(&self.base_pose)
    }
}

// ---------------------------------------------------------------------------
// RobotManager — owns multiple ManagedRobots and steps them each frame
// ---------------------------------------------------------------------------

/// Central manager for all robots in the simulation.
///
/// Holds a vector of `ManagedRobot` instances and provides methods to add,
/// query, command, and step them.
#[allow(dead_code)]
pub struct RobotManager {
    pub robots: Vec<ManagedRobot>,
    pub running: bool,
}

impl Default for RobotManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl RobotManager {
    /// Create an empty RobotManager with no robots.
    pub fn new() -> Self {
        Self {
            robots: Vec::new(),
            running: true,
        }
    }

    /// Add a robot from a definition and a world-space base pose.
    ///
    /// Returns the index of the newly added robot.
    pub fn add_robot(&mut self, definition: RobotDefinition, base_pose: Mat4) -> usize {
        let state = RobotState::new(&definition);
        let robot = ManagedRobot {
            definition,
            state,
            base_pose: base_pose.to_cols_array(),
        };
        let index = self.robots.len();
        self.robots.push(robot);
        index
    }

    /// Step all robots forward by `dt` seconds.
    ///
    /// For each robot: step dynamics, compute forward kinematics, then
    /// simulate sensors against the provided scene meshes.
    pub fn step(&mut self, dt: f32, scene_meshes: &[SceneObject]) {
        if !self.running {
            return;
        }
        for robot in &mut self.robots {
            step_dynamics(&robot.definition, &mut robot.state, dt);
            let bp = Mat4::from_cols_array(&robot.base_pose);
            forward_kinematics(&robot.definition, &mut robot.state, bp);
            simulate_sensors(&robot.definition, &mut robot.state, scene_meshes);
        }
    }

    /// Get an immutable reference to a robot by index.
    pub fn get_robot(&self, index: usize) -> Option<&ManagedRobot> {
        self.robots.get(index)
    }

    /// Get a mutable reference to a robot by index.
    pub fn get_robot_mut(&mut self, index: usize) -> Option<&mut ManagedRobot> {
        self.robots.get_mut(index)
    }

    /// Set an actuator command on a specific joint of a specific robot.
    pub fn set_command(
        &mut self,
        robot_index: usize,
        joint_index: usize,
        command: ActuatorCommand,
    ) {
        if let Some(robot) = self.robots.get_mut(robot_index) {
            // Ensure the actuator_commands vector is large enough.
            let num_joints = robot.definition.joints.len();
            if robot.state.actuator_commands.len() < num_joints {
                robot
                    .state
                    .actuator_commands
                    .resize(num_joints, ActuatorCommand::Torque(0.0));
            }
            if joint_index < robot.state.actuator_commands.len() {
                robot.state.actuator_commands[joint_index] = command;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RobotSimulation — high-level simulation wrapper with fixed timestep
// ---------------------------------------------------------------------------

/// High-level robot simulation that wraps a `RobotManager` with a fixed
/// timestep. Provides a `step` method that advances all robots by `dt`.
#[allow(dead_code)]
pub struct RobotSimulation {
    pub manager: RobotManager,
    pub running: bool,
    pub dt: f32,
}

impl Default for RobotSimulation {
    fn default() -> Self {
        Self {
            manager: RobotManager::default(),
            running: true,
            dt: 1.0 / 60.0,
        }
    }
}

#[allow(dead_code)]
impl RobotSimulation {
    /// Step all robots forward by `self.dt` seconds.
    ///
    /// For each robot: apply actuator dynamics, compute forward kinematics,
    /// then simulate sensors against the provided scene meshes.
    pub fn step(&mut self, scene_meshes: &[SceneObject]) {
        if !self.running {
            return;
        }
        self.manager.step(self.dt, scene_meshes);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::definition::RobotDefinition;
    use crate::robot::state::ActuatorCommand;
    use glam::Mat4;

    #[test]
    fn test_manager_default() {
        let manager = RobotManager::default();
        assert!(
            manager.robots.is_empty(),
            "default manager should have no robots"
        );
        assert!(manager.running, "default manager should be running");
    }

    #[test]
    fn test_add_robot() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        assert_eq!(idx, 0, "first robot should have index 0");
        assert!(
            manager.get_robot(idx).is_some(),
            "robot should be accessible by index"
        );

        let robot = manager.get_robot(idx).unwrap();
        assert_eq!(robot.definition.name, "simple_arm");
        assert_eq!(robot.state.joint_positions.len(), 2);
    }

    #[test]
    fn test_step_updates_state() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Set a velocity command so dynamics actually move the joint
        manager.set_command(idx, 0, ActuatorCommand::Velocity(2.0));

        let pos_before = manager.get_robot(idx).unwrap().state.joint_positions[0];

        // Step several times
        for _ in 0..10 {
            manager.step(0.01, &[]);
        }

        let pos_after = manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            (pos_after - pos_before).abs() > 1e-6,
            "joint position should change after stepping with a velocity command, before={} after={}",
            pos_before,
            pos_after
        );
    }

    #[test]
    fn test_multiple_robots() {
        let mut manager = RobotManager::new();

        let def1 = RobotDefinition::simple_arm(1);
        let def2 = RobotDefinition::simple_arm(3);
        let idx1 = manager.add_robot(def1, Mat4::IDENTITY);
        let idx2 = manager.add_robot(def2, Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)));

        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);
        assert_eq!(manager.robots.len(), 2);

        // Robots have independent state
        let r1 = manager.get_robot(idx1).unwrap();
        let r2 = manager.get_robot(idx2).unwrap();
        assert_eq!(r1.state.joint_positions.len(), 1);
        assert_eq!(r2.state.joint_positions.len(), 3);

        // Command only robot 1
        manager.set_command(idx1, 0, ActuatorCommand::Velocity(5.0));

        manager.step(0.01, &[]);

        // Robot 1 should have moved
        let r1_pos = manager.get_robot(idx1).unwrap().state.joint_positions[0];
        assert!(
            r1_pos.abs() > 1e-9,
            "robot 1 should have moved, pos={}",
            r1_pos
        );

        // Robot 2 joint 0 should still be at zero (no command set)
        let r2_pos = manager.get_robot(idx2).unwrap().state.joint_positions[0];
        assert!(
            r2_pos.abs() < 1e-6,
            "robot 2 should not have moved without a command, pos={}",
            r2_pos
        );
    }

    // ------------------------------------------------------------------
    // Task 7: RobotSimulation tests
    // ------------------------------------------------------------------

    #[test]
    fn test_robot_simulation_step() {
        let mut sim = RobotSimulation::default();
        let def = RobotDefinition::simple_arm(1);
        let idx = sim.manager.add_robot(def, Mat4::IDENTITY);

        // Set a velocity command so dynamics actually move the joint
        sim.manager
            .set_command(idx, 0, ActuatorCommand::Velocity(2.0));

        let pos_before = sim.manager.get_robot(idx).unwrap().state.joint_positions[0];

        // Step several times
        let meshes: Vec<crate::scene::SceneObject> = vec![];
        for _ in 0..10 {
            sim.step(&meshes);
        }

        let pos_after = sim.manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            (pos_after - pos_before).abs() > 1e-6,
            "joint position should change after stepping with a velocity command, before={} after={}",
            pos_before,
            pos_after
        );
    }

    #[test]
    fn test_robot_simulation_sensors_update() {
        use crate::robot::definition::{SensorDefinition, SensorMount};

        // Build a robot definition with a distance sensor
        let def = RobotDefinition {
            name: "sensor_bot".to_string(),
            links: vec![crate::robot::definition::LinkDefinition {
                name: "base".to_string(),
                mass: 5.0,
                inertia: 1.0,
                collision_shape: crate::robot::definition::CollisionShape::Cuboid {
                    half_extents: glam::Vec3::splat(0.1),
                },
                parent_joint: None,
            }],
            joints: vec![],
            sensors: vec![SensorMount {
                link_index: 0,
                local_offset: glam::Vec3::ZERO,
                sensor: SensorDefinition::Distance {
                    direction: glam::Vec3::Z,
                    max_range: 50.0,
                },
            }],
        };

        let mut sim = RobotSimulation::default();
        sim.manager.add_robot(def, Mat4::IDENTITY);

        // Step with no scene objects
        let meshes: Vec<crate::scene::SceneObject> = vec![];
        sim.step(&meshes);

        // Sensor should have a reading (max_range since no objects)
        let robot = sim.manager.get_robot(0).unwrap();
        assert_eq!(
            robot.state.sensor_readings.len(),
            1,
            "should have one sensor reading"
        );
        match &robot.state.sensor_readings[0] {
            crate::robot::state::SensorReading::Distance(d) => {
                assert!(
                    (*d - 50.0).abs() < 1e-4,
                    "distance sensor should read max_range (50.0) with no objects, got {}",
                    d
                );
            }
            other => panic!("Expected Distance reading, got {:?}", other),
        }
    }

    #[test]
    fn test_set_command() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Set command on joint 1
        manager.set_command(idx, 1, ActuatorCommand::Position(0.5));

        let robot = manager.get_robot(idx).unwrap();
        assert_eq!(robot.state.actuator_commands.len(), 2);
        assert_eq!(
            robot.state.actuator_commands[1],
            ActuatorCommand::Position(0.5)
        );

        // Overwrite with a different command
        manager.set_command(idx, 0, ActuatorCommand::Torque(3.0));
        let robot = manager.get_robot(idx).unwrap();
        assert_eq!(
            robot.state.actuator_commands[0],
            ActuatorCommand::Torque(3.0)
        );

        // Out-of-bounds robot index is a no-op
        manager.set_command(999, 0, ActuatorCommand::Velocity(1.0));

        // Out-of-bounds joint index is a no-op
        manager.set_command(idx, 999, ActuatorCommand::Velocity(1.0));
    }

    // ------------------------------------------------------------------
    // Task 8: Robot system integration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_integration_simple_arm_full_pipeline() {
        // Create a 3-DOF simple arm, set position commands, step, verify
        // kinematics produces non-identity link poses and sensor readings
        // are populated.
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(3);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Set position commands on each joint to different targets.
        manager.set_command(idx, 0, ActuatorCommand::Position(0.5));
        manager.set_command(idx, 1, ActuatorCommand::Position(-0.3));
        manager.set_command(idx, 2, ActuatorCommand::Position(0.8));

        // Step multiple times so dynamics move joints toward targets.
        for _ in 0..100 {
            manager.step(0.01, &[]);
        }

        let robot = manager.get_robot(idx).unwrap();

        // Verify joint positions have moved from zero.
        let moved = robot.state.joint_positions.iter().any(|p| p.abs() > 1e-4);
        assert!(
            moved,
            "at least one joint should have moved from zero after position commands"
        );

        // Verify link poses are non-identity (kinematics was computed).
        let identity = Mat4::IDENTITY.to_cols_array();
        let mut has_non_identity = false;
        // Skip link 0 (base) which stays at identity; check child links.
        for pose in robot.state.link_poses.iter().skip(1) {
            let diff: f32 = pose
                .iter()
                .zip(identity.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff > 1e-4 {
                has_non_identity = true;
                break;
            }
        }
        assert!(
            has_non_identity,
            "child link poses should differ from identity after stepping with position commands"
        );
    }

    #[test]
    fn test_integration_serialization_round_trip() {
        // Create a robot, step it so state is non-trivial, then serialize
        // and deserialize both RobotState and RobotDefinition.
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        manager.set_command(idx, 0, ActuatorCommand::Velocity(1.0));
        manager.set_command(idx, 1, ActuatorCommand::Position(0.5));

        for _ in 0..50 {
            manager.step(0.01, &[]);
        }

        let robot = manager.get_robot(idx).unwrap();

        // Serialize RobotState
        let state_json =
            serde_json::to_string(&robot.state).expect("RobotState serialization failed");
        let state_deser: crate::robot::state::RobotState =
            serde_json::from_str(&state_json).expect("RobotState deserialization failed");

        // Verify state fields match
        assert_eq!(
            robot.state.joint_positions.len(),
            state_deser.joint_positions.len()
        );
        for (i, (&a, &b)) in robot
            .state
            .joint_positions
            .iter()
            .zip(state_deser.joint_positions.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "joint_positions[{}] mismatch: {} vs {}",
                i,
                a,
                b
            );
        }
        for (i, (&a, &b)) in robot
            .state
            .joint_velocities
            .iter()
            .zip(state_deser.joint_velocities.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "joint_velocities[{}] mismatch: {} vs {}",
                i,
                a,
                b
            );
        }
        assert_eq!(
            robot.state.link_poses.len(),
            state_deser.link_poses.len(),
            "link_poses length mismatch"
        );

        // Serialize RobotDefinition
        let def_json =
            serde_json::to_string(&robot.definition).expect("RobotDefinition serialization failed");
        let def_deser: crate::robot::definition::RobotDefinition =
            serde_json::from_str(&def_json).expect("RobotDefinition deserialization failed");

        assert_eq!(robot.definition.name, def_deser.name);
        assert_eq!(robot.definition.links.len(), def_deser.links.len());
        assert_eq!(robot.definition.joints.len(), def_deser.joints.len());
    }

    #[test]
    fn test_integration_multi_robot() {
        // Add 2 robots with different commands, step, verify independent state.
        let mut manager = RobotManager::new();
        let def_a = RobotDefinition::simple_arm(2);
        let def_b = RobotDefinition::simple_arm(2);
        let idx_a = manager.add_robot(def_a, Mat4::IDENTITY);
        let idx_b = manager.add_robot(
            def_b,
            Mat4::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)),
        );

        // Give robot A a positive velocity on joint 0.
        manager.set_command(idx_a, 0, ActuatorCommand::Velocity(3.0));
        // Give robot B a negative velocity on joint 1.
        manager.set_command(idx_b, 1, ActuatorCommand::Velocity(-3.0));

        for _ in 0..50 {
            manager.step(0.01, &[]);
        }

        let ra = manager.get_robot(idx_a).unwrap();
        let rb = manager.get_robot(idx_b).unwrap();

        // Robot A joint 0 should have moved positively.
        assert!(
            ra.state.joint_positions[0] > 0.01,
            "robot A joint 0 should be positive, got {}",
            ra.state.joint_positions[0]
        );
        // Robot A joint 1 should still be near zero (no command).
        assert!(
            ra.state.joint_positions[1].abs() < 1e-4,
            "robot A joint 1 should be near zero, got {}",
            ra.state.joint_positions[1]
        );

        // Robot B joint 0 should still be near zero (no command).
        assert!(
            rb.state.joint_positions[0].abs() < 1e-4,
            "robot B joint 0 should be near zero, got {}",
            rb.state.joint_positions[0]
        );
        // Robot B joint 1 should have moved negatively.
        assert!(
            rb.state.joint_positions[1] < -0.01,
            "robot B joint 1 should be negative, got {}",
            rb.state.joint_positions[1]
        );

        // Verify base poses are different.
        let bp_a = ra.base_pose;
        let bp_b = rb.base_pose;
        let diff: f32 = bp_a
            .iter()
            .zip(bp_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0, "base poses should differ between robot A and B");
    }

    #[test]
    fn test_integration_joint_limits_respected() {
        // Command a joint far beyond its limits, step 200+ times, verify
        // position stays within [limit_min, limit_max].
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let limit_min = def.joints[0].limit_min;
        let limit_max = def.joints[0].limit_max;

        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Command far beyond the positive limit.
        manager.set_command(idx, 0, ActuatorCommand::Position(100.0));

        for _ in 0..200 {
            manager.step(0.01, &[]);
        }

        let pos = manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            pos <= limit_max + 1e-6,
            "position {} should not exceed limit_max {}",
            pos,
            limit_max
        );
        assert!(
            pos >= limit_min - 1e-6,
            "position {} should not go below limit_min {}",
            pos,
            limit_min
        );

        // Now command far below the negative limit.
        manager.set_command(idx, 0, ActuatorCommand::Position(-100.0));

        for _ in 0..200 {
            manager.step(0.01, &[]);
        }

        let pos = manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            pos <= limit_max + 1e-6,
            "position {} should not exceed limit_max {} after negative command",
            pos,
            limit_max
        );
        assert!(
            pos >= limit_min - 1e-6,
            "position {} should not go below limit_min {} after negative command",
            pos,
            limit_min
        );
    }

    #[test]
    fn test_integration_sensor_with_scene() {
        use crate::robot::definition::{
            CollisionShape, LinkDefinition, SensorDefinition, SensorMount,
        };
        use crate::scene::material::AcousticMaterial;
        use crate::scene::{Mesh, SceneObject, Triangle, Vertex};

        // Build a robot with a distance sensor pointing along +Z.
        let def = RobotDefinition {
            name: "sensor_arm".to_string(),
            links: vec![LinkDefinition {
                name: "base".to_string(),
                mass: 5.0,
                inertia: 1.0,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: glam::Vec3::splat(0.1),
                },
                parent_joint: None,
            }],
            joints: vec![],
            sensors: vec![SensorMount {
                link_index: 0,
                local_offset: glam::Vec3::ZERO,
                sensor: SensorDefinition::Distance {
                    direction: glam::Vec3::Z,
                    max_range: 100.0,
                },
            }],
        };

        let mut manager = RobotManager::new();
        manager.add_robot(def, Mat4::IDENTITY);

        // Create a SceneObject with a triangle at z=3 (a wall).
        let wall_tri = Triangle {
            vertices: [
                Vertex {
                    position: glam::Vec3::new(-2.0, -2.0, 3.0),
                    normal: glam::Vec3::NEG_Z,
                },
                Vertex {
                    position: glam::Vec3::new(2.0, -2.0, 3.0),
                    normal: glam::Vec3::NEG_Z,
                },
                Vertex {
                    position: glam::Vec3::new(0.0, 2.0, 3.0),
                    normal: glam::Vec3::NEG_Z,
                },
            ],
        };
        let scene_obj = SceneObject {
            name: "wall".to_string(),
            mesh: Mesh {
                triangles: vec![wall_tri],
            },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        };

        // Step with the scene object so sensors are simulated.
        manager.step(0.01, &[scene_obj]);

        let robot = manager.get_robot(0).unwrap();
        assert_eq!(
            robot.state.sensor_readings.len(),
            1,
            "should have one sensor reading"
        );

        match &robot.state.sensor_readings[0] {
            crate::robot::state::SensorReading::Distance(d) => {
                assert!(
                    (*d - 3.0).abs() < 0.1,
                    "distance sensor should read ~3.0 for wall at z=3, got {}",
                    d
                );
            }
            other => panic!("Expected Distance reading, got {:?}", other),
        }
    }

    #[test]
    fn test_integration_dynamics_convergence() {
        // Set a position command to a target, step many times, verify the
        // joint position converges within tolerance of the target.
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        let target = 1.0_f32;
        manager.set_command(idx, 0, ActuatorCommand::Position(target));

        // Step 1500 times at 0.001s (1.5 seconds of simulation).
        for _ in 0..1500 {
            manager.step(0.001, &[]);
        }

        let pos = manager.get_robot(idx).unwrap().state.joint_positions[0];
        let tolerance = 0.1;
        assert!(
            (pos - target).abs() < tolerance,
            "joint position {} should converge within {} of target {}, error = {}",
            pos,
            tolerance,
            target,
            (pos - target).abs()
        );
    }
}
