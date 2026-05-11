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

    // ------------------------------------------------------------------
    // Task 8: Spec-required integration tests (6 tests)
    // ------------------------------------------------------------------

    #[test]
    fn test_integration_3dof_arm() {
        // Build a 3-DOF arm using the definition-based system, set joint
        // angles directly, compute FK, verify end-effector position matches
        // an analytical solution.
        //
        // Robot: 3 revolute joints around Y axis. simple_arm creates
        // sequential parent-child links. All joints at 0 means all links
        // are stacked at identity (no displacement since joint transforms
        // are identity at position=0).
        //
        // Set joint 0 to pi/2 around Y. This should rotate the subtree.
        // Since definition-based FK computes child_pose = parent_pose * joint_transform,
        // and simple_arm joints use Y axis, a pi/2 rotation around Y swaps X and Z.
        // Use the body-based FK which has known-good local offsets.
        use crate::robot::body::{Joint, JointType as BodyJointType, Link, Robot};
        use crate::robot::kinematics::compute_forward_kinematics;
        use glam::{Quat, Vec3};
        use std::f32::consts::FRAC_PI_2;

        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("3dof_arm", Vec3::ZERO, Quat::IDENTITY, base);

        // Link 1: offset 1.0 along X, joint 0 revolute around Z at 0
        let j0 = Joint::new(
            BodyJointType::Revolute,
            Vec3::Z,
            0.0,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let l1 = Link::new(
            "link1",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j0, l1);

        // Link 2: offset 1.0 along X, joint 1 revolute around Z at pi/2
        let j1 = Joint::new(
            BodyJointType::Revolute,
            Vec3::Z,
            FRAC_PI_2,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let l2 = Link::new(
            "link2",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j1, l2);

        // Link 3: offset 1.0 along X, joint 2 revolute around Z at -pi/2
        let j2 = Joint::new(
            BodyJointType::Revolute,
            Vec3::Z,
            -FRAC_PI_2,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let l3 = Link::new(
            "link3",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j2, l3);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(
            transforms.len(),
            4,
            "should have 4 link transforms (base + 3)"
        );

        // Analytical solution:
        // Link 0 (base): at origin (0, 0, 0)
        // Link 1: joint 0 at 0 deg, offset (1,0,0) -> (1, 0, 0)
        // Link 2: joint 1 at 90 deg around Z. At link1, rotate local X by 90 -> becomes Y.
        //         offset (1,0,0) in rotated frame = (0,1,0). World pos = (1,0,0) + (0,1,0) = (1, 1, 0)
        // Link 3: joint 2 at -90 deg around Z (cumulative rotation = 90 - 90 = 0).
        //         offset (1,0,0) in frame rotated 0 deg from world = (1,0,0).
        //         World pos = (1, 1, 0) + (1, 0, 0) = (2, 1, 0)
        let epsilon = 1e-4;
        let end_effector = transforms[3].position;
        assert!(
            (end_effector - Vec3::new(2.0, 1.0, 0.0)).length() < epsilon,
            "end-effector should be at (2, 1, 0), got {:?}",
            end_effector
        );

        // Also verify intermediate positions.
        assert!(
            (transforms[1].position - Vec3::new(1.0, 0.0, 0.0)).length() < epsilon,
            "link1 should be at (1, 0, 0), got {:?}",
            transforms[1].position
        );
        assert!(
            (transforms[2].position - Vec3::new(1.0, 1.0, 0.0)).length() < epsilon,
            "link2 should be at (1, 1, 0), got {:?}",
            transforms[2].position
        );
    }

    #[test]
    fn test_integration_sensor_sweep() {
        // Create a robot with a revolute joint and a distance sensor on
        // the child link. Place a wall at z=5. Rotate the sensor 360
        // degrees in discrete steps, verifying readings change — the
        // sensor should detect the wall when pointing toward it and
        // return max_range when pointing away.
        use crate::robot::definition::{
            CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
            SensorDefinition, SensorMount,
        };
        use crate::scene::material::AcousticMaterial;
        use crate::scene::{Mesh, SceneObject, Triangle, Vertex};
        use glam::Vec3;

        // Robot: base link at origin + child link connected by revolute
        // joint around Y. Distance sensor on child link pointing along +Z.
        let def = RobotDefinition {
            name: "sweep_bot".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".into(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: Vec3::splat(0.1),
                    },
                    parent_joint: None,
                },
                LinkDefinition {
                    name: "sensor_arm".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.05 },
                    parent_joint: Some(0),
                },
            ],
            joints: vec![JointDefinition {
                name: "sweep_joint".into(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 100.0,
                damping: 0.0,
            }],
            sensors: vec![SensorMount {
                link_index: 1,
                local_offset: Vec3::ZERO,
                sensor: SensorDefinition::Distance {
                    direction: Vec3::Z,
                    max_range: 50.0,
                },
            }],
        };

        // Large wall at z=5, spanning x=[-5,5], y=[-5,5].
        let wall = SceneObject {
            name: "wall".into(),
            mesh: Mesh {
                triangles: vec![
                    Triangle {
                        vertices: [
                            Vertex {
                                position: Vec3::new(-5.0, -5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                            Vertex {
                                position: Vec3::new(5.0, -5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                            Vertex {
                                position: Vec3::new(0.0, 5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                        ],
                    },
                    Triangle {
                        vertices: [
                            Vertex {
                                position: Vec3::new(5.0, -5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                            Vertex {
                                position: Vec3::new(5.0, 5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                            Vertex {
                                position: Vec3::new(-5.0, 5.0, 5.0),
                                normal: Vec3::NEG_Z,
                            },
                        ],
                    },
                ],
            },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        };
        let scene_meshes = vec![wall];

        let mut manager = RobotManager::new();
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Sweep: collect distance readings at various angles.
        let num_steps = 8;
        let mut readings = Vec::new();

        for step in 0..num_steps {
            let angle = -std::f32::consts::PI
                + 2.0 * std::f32::consts::PI * (step as f32) / (num_steps as f32);

            // Directly set the joint position for sweep.
            if let Some(robot) = manager.get_robot_mut(idx) {
                robot.state.joint_positions[0] = angle;
                robot.state.joint_velocities[0] = 0.0;
            }

            // Compute FK and sensors.
            {
                let robot = manager.get_robot_mut(idx).unwrap();
                let bp = Mat4::from_cols_array(&robot.base_pose);
                forward_kinematics(&robot.definition, &mut robot.state, bp);
                crate::robot::sensors::simulate_sensors(
                    &robot.definition,
                    &mut robot.state,
                    &scene_meshes,
                );
            }

            let robot = manager.get_robot(idx).unwrap();
            if let crate::robot::state::SensorReading::Distance(d) = &robot.state.sensor_readings[0]
            {
                readings.push(*d);
            }
        }

        // Some readings should be ~5 (pointing at wall), others should be 50 (max_range).
        let has_near = readings.iter().any(|&d| d < 10.0);
        let has_far = readings.iter().any(|&d| d > 40.0);
        assert!(
            has_near,
            "should detect wall at ~5m for some angles, readings: {:?}",
            readings
        );
        assert!(
            has_far,
            "should return max_range for angles pointing away, readings: {:?}",
            readings
        );

        // Verify not all readings are the same (sweep actually changes readings).
        let first = readings[0];
        let all_same = readings.iter().all(|&r| (r - first).abs() < 0.1);
        assert!(
            !all_same,
            "readings should vary across sweep angles, but all are ~{:.2}",
            first
        );
    }

    #[test]
    fn test_integration_motor_step_sequence() {
        // Apply a velocity command over multiple steps, then switch to a
        // position command and verify the joint converges to the target.
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Phase 1: Apply velocity command to get the joint moving.
        manager.set_command(idx, 0, ActuatorCommand::Velocity(2.0));
        for _ in 0..100 {
            manager.step(0.01, &[]);
        }

        let pos_after_velocity = manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            pos_after_velocity > 0.1,
            "joint should have moved positively with velocity command, got {}",
            pos_after_velocity
        );

        // Phase 2: Switch to a position command targeting 0.5 rad.
        let target = 0.5_f32;
        manager.set_command(idx, 0, ActuatorCommand::Position(target));
        for _ in 0..2000 {
            manager.step(0.001, &[]);
        }

        let pos_final = manager.get_robot(idx).unwrap().state.joint_positions[0];
        let tolerance = 0.15;
        assert!(
            (pos_final - target).abs() < tolerance,
            "joint position {} should converge to target {} within {}, error = {}",
            pos_final,
            target,
            tolerance,
            (pos_final - target).abs()
        );
    }

    #[test]
    fn test_integration_gripper_pick_place() {
        // Use the body-based Robot + GripperActuator to close the gripper
        // near an object, verify attachment, move the robot, then open
        // the gripper and verify the object is released.
        use crate::robot::actuators::GripperActuator;
        use crate::robot::body::{Joint, JointType as BodyJointType, Link, Robot};
        use crate::robot::kinematics::compute_forward_kinematics;
        use crate::scene::material::AcousticMaterial;
        use crate::scene::{Mesh, Scene, SceneObject, Triangle, Vertex};
        use glam::{Quat, Vec3};

        // Build a simple robot with a base + one arm link.
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("gripper_bot", Vec3::ZERO, Quat::IDENTITY, base);
        let joint = Joint::new(
            BodyJointType::Revolute,
            Vec3::Y,
            0.0,
            0.0,
            (-3.14, 3.14),
            10.0,
        );
        let gripper_link = Link::new(
            "gripper_link",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.5), // large half-extents for overlap detection
            1.0,
        );
        robot.add_joint_and_link(joint, gripper_link);

        let transforms = compute_forward_kinematics(&robot);
        let gripper_pos = transforms[1].position; // link 1 world position

        // Place a scene object overlapping the gripper link.
        let obj = SceneObject {
            name: "target_box".into(),
            mesh: Mesh {
                triangles: vec![Triangle {
                    vertices: [
                        Vertex {
                            position: gripper_pos - Vec3::splat(0.2),
                            normal: Vec3::Y,
                        },
                        Vertex {
                            position: gripper_pos + Vec3::new(0.2, -0.2, -0.2),
                            normal: Vec3::Y,
                        },
                        Vertex {
                            position: gripper_pos + Vec3::splat(0.2),
                            normal: Vec3::Y,
                        },
                    ],
                }],
            },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        };

        let mut scene = Scene::default();
        scene.meshes.push(obj);

        // Create the gripper and close it.
        let mut gripper = GripperActuator {
            link_index: 1,
            is_open: true,
            attached_object: None,
            grip_strength: 10.0,
        };

        gripper.close(&transforms, &robot, &scene);

        // Verify object is attached.
        assert!(!gripper.is_open, "gripper should be closed");
        assert_eq!(
            gripper.attached_object,
            Some(0),
            "gripper should have attached the nearby object"
        );

        // Compute attached transform — object should follow the gripper.
        let attached = gripper.compute_attached_transform(&transforms, &robot);
        assert!(attached.is_some(), "should compute attached transform");
        let (obj_idx, obj_pos, _obj_rot) = attached.unwrap();
        assert_eq!(obj_idx, 0);
        assert!(
            (obj_pos - gripper_pos).length() < 1e-4,
            "attached object should follow gripper position"
        );

        // Open the gripper — object should be released.
        gripper.open();
        assert!(gripper.is_open, "gripper should be open");
        assert!(
            gripper.attached_object.is_none(),
            "object should be released after opening gripper"
        );

        // After opening, compute_attached_transform should return None.
        let detached = gripper.compute_attached_transform(&transforms, &robot);
        assert!(
            detached.is_none(),
            "should not compute transform after release"
        );
    }

    #[test]
    fn test_integration_robot_state_roundtrip() {
        // Create a robot with sensors and actuators, step it to populate
        // state with non-trivial values, then serialize and deserialize
        // the full ManagedRobot (definition + state), verifying all
        // fields survive the round-trip.
        use crate::robot::definition::{
            CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
            SensorDefinition, SensorMount,
        };
        use glam::Vec3;

        let def = RobotDefinition {
            name: "roundtrip_bot".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".into(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: Vec3::splat(0.1),
                    },
                    parent_joint: None,
                },
                LinkDefinition {
                    name: "link_1".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Cylinder {
                        radius: 0.05,
                        height: 0.5,
                    },
                    parent_joint: Some(0),
                },
            ],
            joints: vec![JointDefinition {
                name: "joint_0".into(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
            }],
            sensors: vec![
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Distance {
                        direction: Vec3::Z,
                        max_range: 50.0,
                    },
                },
                SensorMount {
                    link_index: 1,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Imu,
                },
            ],
        };

        let mut manager = RobotManager::new();
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // Set commands and step to produce non-trivial state.
        manager.set_command(idx, 0, ActuatorCommand::Velocity(1.5));
        for _ in 0..50 {
            manager.step(0.01, &[]);
        }

        let robot = manager.get_robot(idx).unwrap();

        // Serialize the entire ManagedRobot.
        let json = serde_json::to_string(robot).expect("ManagedRobot serialization failed");
        let deser: ManagedRobot =
            serde_json::from_str(&json).expect("ManagedRobot deserialization failed");

        // Verify definition fields.
        assert_eq!(deser.definition.name, robot.definition.name);
        assert_eq!(deser.definition.links.len(), robot.definition.links.len());
        assert_eq!(deser.definition.joints.len(), robot.definition.joints.len());
        assert_eq!(
            deser.definition.sensors.len(),
            robot.definition.sensors.len()
        );

        // Verify state fields.
        assert_eq!(
            deser.state.joint_positions.len(),
            robot.state.joint_positions.len()
        );
        for (i, (&a, &b)) in robot
            .state
            .joint_positions
            .iter()
            .zip(deser.state.joint_positions.iter())
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
            .zip(deser.state.joint_velocities.iter())
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
            deser.state.link_poses.len(),
            robot.state.link_poses.len(),
            "link_poses length mismatch"
        );
        assert_eq!(
            deser.state.sensor_readings.len(),
            robot.state.sensor_readings.len(),
            "sensor_readings length mismatch"
        );

        // Verify base_pose round-trips.
        for (i, (&a, &b)) in robot
            .base_pose
            .iter()
            .zip(deser.base_pose.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "base_pose[{}] mismatch: {} vs {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_integration_all_joint_types() {
        // Build a robot with all three joint types (revolute, prismatic,
        // fixed) and verify FK produces correct transforms for each.
        use crate::robot::body::{Joint, JointType as BodyJointType, Link, Robot};
        use crate::robot::kinematics::compute_forward_kinematics;
        use glam::{Quat, Vec3};

        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("all_joints", Vec3::ZERO, Quat::IDENTITY, base);

        // Joint 0: Revolute around Z at pi/2 — rotates child offset (1,0,0) to (0,1,0).
        let j_revolute = Joint::new(
            BodyJointType::Revolute,
            Vec3::Z,
            std::f32::consts::FRAC_PI_2,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let l1 = Link::new(
            "revolute_child",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j_revolute, l1);

        // Joint 1: Prismatic along X, extended by 2.0.
        // Because parent link 1 has 90-deg rotation around Z, local X of child
        // is now world Y. So prismatic along X at parent = translation along world Y.
        let j_prismatic = Joint::new(
            BodyJointType::Prismatic,
            Vec3::X,
            2.0,
            0.0,
            (-5.0, 5.0),
            10.0,
        );
        let l2 = Link::new(
            "prismatic_child",
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j_prismatic, l2);

        // Joint 2: Fixed — no additional transform, child just inherits parent.
        let j_fixed = Joint::new(BodyJointType::Fixed, Vec3::Y, 0.0, 0.0, (0.0, 0.0), 0.0);
        let l3 = Link::new(
            "fixed_child",
            Vec3::new(0.0, 0.0, 1.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(j_fixed, l3);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 4, "should have 4 link transforms");

        let epsilon = 1e-4;

        // Link 0 (base): at origin.
        assert!(
            (transforms[0].position - Vec3::ZERO).length() < epsilon,
            "base should be at origin, got {:?}",
            transforms[0].position
        );

        // Link 1 (revolute child): offset (1,0,0) rotated 90 deg around Z = (0,1,0).
        assert!(
            (transforms[1].position - Vec3::new(0.0, 1.0, 0.0)).length() < epsilon,
            "revolute child should be at (0,1,0), got {:?}",
            transforms[1].position
        );

        // Link 2 (prismatic child): parent at (0,1,0) with 90-deg Z rotation.
        // Prismatic along X in parent frame = along Y in world frame.
        // Translation = parent_rot * (axis * position) = Rot_Z(90) * (2,0,0) = (0,2,0).
        // World pos = (0,1,0) + (0,2,0) = (0,3,0).
        assert!(
            (transforms[2].position - Vec3::new(0.0, 3.0, 0.0)).length() < epsilon,
            "prismatic child should be at (0,3,0), got {:?}",
            transforms[2].position
        );

        // Link 3 (fixed child): parent at (0,3,0) with 90-deg Z rotation.
        // Fixed joint adds no joint transform. Child local offset (0,0,1) is
        // rotated by parent rotation: Rot_Z(90) * (0,0,1) = (0,0,1).
        // World pos = (0,3,0) + (0,0,1) = (0,3,1).
        assert!(
            (transforms[3].position - Vec3::new(0.0, 3.0, 1.0)).length() < epsilon,
            "fixed child should be at (0,3,1), got {:?}",
            transforms[3].position
        );
    }
}
