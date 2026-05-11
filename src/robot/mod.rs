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
}
