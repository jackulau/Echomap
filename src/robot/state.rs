use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use super::definition::RobotDefinition;

// ---- Enums ----

/// A single sensor reading from one of the robot's sensors.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SensorReading {
    Distance(f32),
    Lidar(Vec<f32>),
    Contact(bool),
    Imu {
        linear_accel: Vec3,
        angular_vel: Vec3,
    },
}

/// A command sent to one of the robot's actuators.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ActuatorCommand {
    Position(f32),
    Velocity(f32),
    Torque(f32),
}

// ---- Structs ----

/// Serializable snapshot of the full robot state at a point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotState {
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub link_poses: Vec<[f32; 16]>,
    pub sensor_readings: Vec<SensorReading>,
    pub actuator_commands: Vec<ActuatorCommand>,
    pub timestamp: f32,
}

impl RobotState {
    /// Create a new RobotState sized to match the given definition.
    /// All values are initialized to zero/default.
    pub fn new(definition: &RobotDefinition) -> Self {
        let num_joints = definition.joints.len();
        let num_links = definition.links.len();
        let num_sensors = definition.sensors.len();

        // Default sensor readings based on sensor type
        let sensor_readings = definition
            .sensors
            .iter()
            .map(|mount| match &mount.sensor {
                super::definition::SensorDefinition::Distance { max_range, .. } => {
                    SensorReading::Distance(*max_range)
                }
                super::definition::SensorDefinition::Lidar { num_rays, .. } => {
                    SensorReading::Lidar(vec![0.0; *num_rays])
                }
                super::definition::SensorDefinition::Contact => SensorReading::Contact(false),
                super::definition::SensorDefinition::Imu => SensorReading::Imu {
                    linear_accel: Vec3::ZERO,
                    angular_vel: Vec3::ZERO,
                },
            })
            .collect();

        Self {
            joint_positions: vec![0.0; num_joints],
            joint_velocities: vec![0.0; num_joints],
            link_poses: vec![Mat4::IDENTITY.to_cols_array(); num_links],
            sensor_readings,
            actuator_commands: Vec::with_capacity(num_sensors),
            timestamp: 0.0,
        }
    }

    /// Set a joint position, clamping to the provided limits.
    pub fn set_joint_position(&mut self, index: usize, value: f32, min: f32, max: f32) {
        if index < self.joint_positions.len() {
            self.joint_positions[index] = value.clamp(min, max);
        }
    }

    /// Convert link_poses to glam::Mat4 values.
    pub fn link_poses_as_mat4(&self) -> Vec<Mat4> {
        self.link_poses.iter().map(Mat4::from_cols_array).collect()
    }

    /// Set a link pose from a glam::Mat4.
    pub fn set_link_pose(&mut self, index: usize, mat: Mat4) {
        if index < self.link_poses.len() {
            self.link_poses[index] = mat.to_cols_array();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::definition::{
        CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition, SensorMount,
    };

    /// Helper to create a test robot definition with known sizes.
    fn test_definition() -> RobotDefinition {
        RobotDefinition {
            name: "test_bot".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".to_string(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: Vec3::new(0.1, 0.1, 0.1),
                    },
                    parent_joint: None,
                },
                LinkDefinition {
                    name: "link_1".to_string(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Cylinder {
                        radius: 0.05,
                        height: 0.5,
                    },
                    parent_joint: Some(0),
                },
                LinkDefinition {
                    name: "link_2".to_string(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: Some(1),
                },
            ],
            joints: vec![
                JointDefinition {
                    name: "joint_0".to_string(),
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
                    name: "joint_1".to_string(),
                    joint_type: JointType::Prismatic,
                    axis: Vec3::X,
                    parent_link: 1,
                    child_link: 2,
                    limit_min: 0.0,
                    limit_max: 1.0,
                    max_torque: 5.0,
                    damping: 0.05,
                },
            ],
            sensors: vec![SensorMount {
                link_index: 1,
                local_offset: Vec3::ZERO,
                sensor: crate::robot::definition::SensorDefinition::Contact,
            }],
        }
    }

    #[test]
    fn test_state_new_sizes() {
        let def = test_definition();
        let state = RobotState::new(&def);

        // 2 joints => 2 positions, 2 velocities
        assert_eq!(
            state.joint_positions.len(),
            def.joints.len(),
            "joint_positions length should match number of joints"
        );
        assert_eq!(
            state.joint_velocities.len(),
            def.joints.len(),
            "joint_velocities length should match number of joints"
        );
        // 3 links => 3 link poses
        assert_eq!(
            state.link_poses.len(),
            def.links.len(),
            "link_poses length should match number of links"
        );
        // 1 sensor => 1 sensor reading
        assert_eq!(
            state.sensor_readings.len(),
            def.sensors.len(),
            "sensor_readings length should match number of sensors"
        );
    }

    #[test]
    fn test_state_initial_zeros() {
        let def = test_definition();
        let state = RobotState::new(&def);

        for (i, &pos) in state.joint_positions.iter().enumerate() {
            assert!(
                pos.abs() < 1e-6,
                "joint_positions[{}] should be 0, got {}",
                i,
                pos
            );
        }
        for (i, &vel) in state.joint_velocities.iter().enumerate() {
            assert!(
                vel.abs() < 1e-6,
                "joint_velocities[{}] should be 0, got {}",
                i,
                vel
            );
        }
        assert!(
            (state.timestamp).abs() < 1e-6,
            "timestamp should be 0, got {}",
            state.timestamp
        );

        // Each link pose should be identity
        let identity = Mat4::IDENTITY.to_cols_array();
        for (i, pose) in state.link_poses.iter().enumerate() {
            for (j, (&a, &b)) in pose.iter().zip(identity.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "link_poses[{}][{}] should be identity element {}, got {}",
                    i,
                    j,
                    b,
                    a
                );
            }
        }
    }

    #[test]
    fn test_set_joint_position_clamped() {
        let def = test_definition();
        let mut state = RobotState::new(&def);

        // Joint 1 has limits [0.0, 1.0]
        // Value within range
        state.set_joint_position(1, 0.5, 0.0, 1.0);
        assert!(
            (state.joint_positions[1] - 0.5).abs() < 1e-6,
            "Expected 0.5, got {}",
            state.joint_positions[1]
        );

        // Value above max -> clamped to 1.0
        state.set_joint_position(1, 5.0, 0.0, 1.0);
        assert!(
            (state.joint_positions[1] - 1.0).abs() < 1e-6,
            "Expected clamped to 1.0, got {}",
            state.joint_positions[1]
        );

        // Value below min -> clamped to 0.0
        state.set_joint_position(1, -2.0, 0.0, 1.0);
        assert!(
            (state.joint_positions[1]).abs() < 1e-6,
            "Expected clamped to 0.0, got {}",
            state.joint_positions[1]
        );
    }

    #[test]
    fn test_state_serialization() {
        let def = test_definition();
        let mut state = RobotState::new(&def);

        // Set some non-trivial values
        state.set_joint_position(0, 1.0, -std::f32::consts::PI, std::f32::consts::PI);
        state.set_joint_position(1, 0.5, 0.0, 1.0);
        state.joint_velocities[0] = 0.3;
        state.timestamp = 1.5;
        state.sensor_readings = vec![SensorReading::Distance(2.5)];
        state.actuator_commands = vec![ActuatorCommand::Velocity(1.0)];

        let json = serde_json::to_string(&state).expect("serialization failed");
        let deserialized: RobotState = serde_json::from_str(&json).expect("deserialization failed");

        // Verify all fields survived round-trip
        assert_eq!(
            state.joint_positions.len(),
            deserialized.joint_positions.len()
        );
        for (i, (&a, &b)) in state
            .joint_positions
            .iter()
            .zip(deserialized.joint_positions.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "joint_positions[{}]: expected {}, got {}",
                i,
                a,
                b
            );
        }

        assert_eq!(
            state.joint_velocities.len(),
            deserialized.joint_velocities.len()
        );
        for (i, (&a, &b)) in state
            .joint_velocities
            .iter()
            .zip(deserialized.joint_velocities.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "joint_velocities[{}]: expected {}, got {}",
                i,
                a,
                b
            );
        }

        assert!(
            (state.timestamp - deserialized.timestamp).abs() < 1e-6,
            "timestamp: expected {}, got {}",
            state.timestamp,
            deserialized.timestamp
        );

        assert_eq!(state.link_poses.len(), deserialized.link_poses.len());
        assert_eq!(
            state.sensor_readings.len(),
            deserialized.sensor_readings.len()
        );
        assert_eq!(
            state.actuator_commands.len(),
            deserialized.actuator_commands.len()
        );

        // Verify sensor reading values
        assert_eq!(state.sensor_readings, deserialized.sensor_readings);
        assert_eq!(state.actuator_commands, deserialized.actuator_commands);
    }

    #[test]
    fn test_actuator_command_variants() {
        let pos_cmd = ActuatorCommand::Position(1.5);
        let vel_cmd = ActuatorCommand::Velocity(-0.5);
        let torque_cmd = ActuatorCommand::Torque(10.0);

        // Verify distinct
        assert_ne!(pos_cmd, vel_cmd);
        assert_ne!(vel_cmd, torque_cmd);
        assert_ne!(pos_cmd, torque_cmd);

        // Verify values
        match &pos_cmd {
            ActuatorCommand::Position(v) => assert!((v - 1.5).abs() < 1e-6),
            _ => panic!("Expected Position"),
        }
        match &vel_cmd {
            ActuatorCommand::Velocity(v) => assert!((v - (-0.5)).abs() < 1e-6),
            _ => panic!("Expected Velocity"),
        }
        match &torque_cmd {
            ActuatorCommand::Torque(v) => assert!((v - 10.0).abs() < 1e-6),
            _ => panic!("Expected Torque"),
        }

        // Verify Clone
        let cloned = pos_cmd.clone();
        assert_eq!(cloned, pos_cmd);

        // Verify serde round-trip for each variant
        for cmd in &[pos_cmd, vel_cmd, torque_cmd] {
            let json = serde_json::to_string(cmd).unwrap();
            let deser: ActuatorCommand = serde_json::from_str(&json).unwrap();
            assert_eq!(cmd, &deser);
        }
    }

    #[test]
    fn test_sensor_reading_variants() {
        let dist = SensorReading::Distance(3.5);
        let lidar = SensorReading::Lidar(vec![1.0, 2.0, 3.0]);
        let contact = SensorReading::Contact(true);
        let imu = SensorReading::Imu {
            linear_accel: Vec3::new(0.0, -9.81, 0.0),
            angular_vel: Vec3::new(0.1, 0.0, 0.0),
        };

        // Verify distinct
        assert_ne!(dist, lidar);
        assert_ne!(lidar, contact);
        assert_ne!(contact, imu);

        // Verify values
        match &dist {
            SensorReading::Distance(v) => assert!((v - 3.5).abs() < 1e-6),
            _ => panic!("Expected Distance"),
        }
        match &lidar {
            SensorReading::Lidar(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0] - 1.0).abs() < 1e-6);
                assert!((v[1] - 2.0).abs() < 1e-6);
                assert!((v[2] - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected Lidar"),
        }
        match &contact {
            SensorReading::Contact(v) => assert!(*v),
            _ => panic!("Expected Contact"),
        }
        match &imu {
            SensorReading::Imu {
                linear_accel,
                angular_vel,
            } => {
                assert!((linear_accel.y - (-9.81)).abs() < 1e-4);
                assert!((angular_vel.x - 0.1).abs() < 1e-6);
            }
            _ => panic!("Expected Imu"),
        }

        // Verify Clone
        let cloned = imu.clone();
        assert_eq!(cloned, imu);

        // Verify serde round-trip for each variant
        for reading in &[dist, lidar, contact, imu] {
            let json = serde_json::to_string(reading).unwrap();
            let deser: SensorReading = serde_json::from_str(&json).unwrap();
            assert_eq!(reading, &deser);
        }
    }
}
