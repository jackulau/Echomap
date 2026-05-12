use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use super::collision::HitEvent;
use super::definition::RobotDefinition;
use super::sensors::ImuReading;

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

/// Combat state tracking health, stamina, and damage statistics for a robot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombatState {
    pub health: f32,
    pub max_health: f32,
    pub stamina: f32,
    pub max_stamina: f32,
    pub recent_hits: Vec<HitEvent>,
    pub total_damage_dealt: f32,
    pub total_damage_received: f32,
    pub knockdown: bool,
}

impl CombatState {
    /// Create a new CombatState with the given max health and stamina.
    pub fn new(max_health: f32, max_stamina: f32) -> Self {
        Self {
            health: max_health,
            max_health,
            stamina: max_stamina,
            max_stamina,
            recent_hits: Vec::new(),
            total_damage_dealt: 0.0,
            total_damage_received: 0.0,
            knockdown: false,
        }
    }

    /// Apply damage, reducing health and setting knockdown if health reaches 0.
    pub fn apply_damage(&mut self, amount: f32) {
        self.health = (self.health - amount).max(0.0);
        if self.health <= 0.0 {
            self.knockdown = true;
        }
    }

    /// Consume stamina for an action. Returns true if enough stamina was available.
    pub fn consume_stamina(&mut self, amount: f32) -> bool {
        if self.stamina >= amount {
            self.stamina -= amount;
            true
        } else {
            false
        }
    }

    /// Regenerate stamina over time at a rate of 5.0 per second.
    pub fn regenerate_stamina(&mut self, dt: f32) {
        self.stamina = (self.stamina + 5.0 * dt).min(self.max_stamina);
    }
}

/// Serializable snapshot of the full robot state at a point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotState {
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub link_poses: Vec<[f32; 16]>,
    #[serde(default)]
    pub prev_link_poses: Vec<[f32; 16]>,
    pub sensor_readings: Vec<SensorReading>,
    pub actuator_commands: Vec<ActuatorCommand>,
    pub timestamp: f32,
    #[serde(default)]
    pub combat: Option<CombatState>,
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
            prev_link_poses: Vec::new(),
            sensor_readings,
            actuator_commands: Vec::with_capacity(num_sensors),
            timestamp: 0.0,
            combat: None,
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

    /// Save the current link poses as the previous poses (for velocity computation).
    pub fn save_previous_poses(&mut self) {
        self.prev_link_poses = self.link_poses.clone();
    }

    /// Compute per-link velocities from the difference between current and previous poses.
    ///
    /// If `prev_link_poses` is empty (first simulation step), returns zero velocities
    /// for every link. Otherwise, extracts the translation column (indices 12, 13, 14)
    /// from each column-major Mat4 and divides by `dt`.
    pub fn compute_link_velocities(&self, dt: f32) -> Vec<Vec3> {
        if self.prev_link_poses.is_empty() {
            return vec![Vec3::ZERO; self.link_poses.len()];
        }

        self.link_poses
            .iter()
            .zip(self.prev_link_poses.iter())
            .map(|(cur, prev)| {
                let cur_pos = Vec3::new(cur[12], cur[13], cur[14]);
                let prev_pos = Vec3::new(prev[12], prev[13], prev[14]);
                (cur_pos - prev_pos) / dt
            })
            .collect()
    }

    /// Set a link pose from a glam::Mat4.
    pub fn set_link_pose(&mut self, index: usize, mat: Mat4) {
        if index < self.link_poses.len() {
            self.link_poses[index] = mat.to_cols_array();
        }
    }
}

// ---------------------------------------------------------------------------
// Task 6: Gym-compatible robot state types
// ---------------------------------------------------------------------------

/// Aggregated sensor readings decomposed by sensor type for gym interfaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GymSensorReadings {
    pub distances: Vec<f32>,
    pub contacts: Vec<bool>,
    pub imu: Vec<ImuReading>,
    pub camera_visible: Vec<Vec<usize>>,
}

/// Gripper open/close state with optional attached object index.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GripperState {
    pub is_open: bool,
    pub attached_object: Option<usize>,
}

/// Combat state snapshot for gym-compatible agent interfaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GymCombatState {
    pub health: f32,
    pub max_health: f32,
    pub stamina: f32,
    pub max_stamina: f32,
    pub knockdown: bool,
    pub recent_hits: Vec<super::collision::HitEvent>,
    pub total_damage_dealt: f32,
    pub total_damage_received: f32,
}

/// Gym-compatible snapshot of a robot's full state for agent communication.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GymRobotState {
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub sensor_readings: GymSensorReadings,
    pub gripper_states: Vec<GripperState>,
    #[serde(default)]
    pub combat: Option<GymCombatState>,
}

impl GymRobotState {
    /// Build a GymRobotState from a low-level RobotState and its definition.
    ///
    /// Decomposes the flat `sensor_readings` vector into typed buckets
    /// (distances, contacts, imu, camera_visible). Camera visible is always
    /// empty here because the base RobotState does not carry camera data.
    pub fn from_robot_state(state: &RobotState, _def: &RobotDefinition) -> Self {
        let mut distances = Vec::new();
        let mut contacts = Vec::new();
        let mut imu = Vec::new();
        let camera_visible: Vec<Vec<usize>> = Vec::new();

        for reading in &state.sensor_readings {
            match reading {
                SensorReading::Distance(d) => distances.push(*d),
                SensorReading::Contact(c) => contacts.push(*c),
                SensorReading::Imu {
                    linear_accel,
                    angular_vel,
                } => imu.push(ImuReading {
                    linear_acceleration: *linear_accel,
                    angular_velocity: *angular_vel,
                }),
                SensorReading::Lidar(_) => {
                    // LIDAR readings are not decomposed into distances here
                }
            }
        }

        Self {
            joint_positions: state.joint_positions.clone(),
            joint_velocities: state.joint_velocities.clone(),
            sensor_readings: GymSensorReadings {
                distances,
                contacts,
                imu,
                camera_visible,
            },
            gripper_states: Vec::new(),
            combat: state.combat.as_ref().map(|c| GymCombatState {
                health: c.health,
                max_health: c.max_health,
                stamina: c.stamina,
                max_stamina: c.max_stamina,
                knockdown: c.knockdown,
                recent_hits: c.recent_hits.clone(),
                total_damage_dealt: c.total_damage_dealt,
                total_damage_received: c.total_damage_received,
            }),
        }
    }

    pub fn from_robot_state_into(
        state: &RobotState,
        _def: &RobotDefinition,
        buf: &mut GymStateBuffer,
    ) -> Self {
        buf.distances.clear();
        buf.contacts.clear();
        buf.imu.clear();
        buf.joint_positions.clear();
        buf.joint_velocities.clear();

        for reading in &state.sensor_readings {
            match reading {
                SensorReading::Distance(d) => buf.distances.push(*d),
                SensorReading::Contact(c) => buf.contacts.push(*c),
                SensorReading::Imu {
                    linear_accel,
                    angular_vel,
                } => buf.imu.push(ImuReading {
                    linear_acceleration: *linear_accel,
                    angular_velocity: *angular_vel,
                }),
                SensorReading::Lidar(_) => {}
            }
        }

        buf.joint_positions
            .extend_from_slice(&state.joint_positions);
        buf.joint_velocities
            .extend_from_slice(&state.joint_velocities);

        Self {
            joint_positions: buf.joint_positions.clone(),
            joint_velocities: buf.joint_velocities.clone(),
            sensor_readings: GymSensorReadings {
                distances: buf.distances.clone(),
                contacts: buf.contacts.clone(),
                imu: buf.imu.clone(),
                camera_visible: Vec::new(),
            },
            gripper_states: Vec::new(),
            combat: state.combat.as_ref().map(|c| GymCombatState {
                health: c.health,
                max_health: c.max_health,
                stamina: c.stamina,
                max_stamina: c.max_stamina,
                knockdown: c.knockdown,
                recent_hits: c.recent_hits.clone(),
                total_damage_dealt: c.total_damage_dealt,
                total_damage_received: c.total_damage_received,
            }),
        }
    }
}

#[derive(Default)]
pub struct GymStateBuffer {
    pub distances: Vec<f32>,
    pub contacts: Vec<bool>,
    pub imu: Vec<ImuReading>,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
}

impl GymStateBuffer {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Describes the dimensions and ranges of sensor outputs (observation space)
/// for gym-compatible agent interfaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservationSpace {
    /// Number of joint position dimensions.
    pub num_joint_positions: usize,
    /// Number of joint velocity dimensions.
    pub num_joint_velocities: usize,
    /// Total number of sensors.
    pub num_sensors: usize,
    /// Per-joint position limits: (min, max).
    pub joint_position_limits: Vec<(f32, f32)>,
}

impl ObservationSpace {
    /// Build an ObservationSpace from a robot definition.
    pub fn from_definition(def: &RobotDefinition) -> Self {
        let joint_position_limits: Vec<(f32, f32)> = def
            .joints
            .iter()
            .map(|j| (j.limit_min, j.limit_max))
            .collect();

        Self {
            num_joint_positions: def.joints.len(),
            num_joint_velocities: def.joints.len(),
            num_sensors: def.sensors.len(),
            joint_position_limits,
        }
    }
}

/// Describes the dimensions and ranges of motor/gripper commands (action space)
/// for gym-compatible agent interfaces.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionSpace {
    /// Number of motor velocity dimensions (one per joint).
    pub num_motors: usize,
    /// Per-motor velocity limits: (min, max) derived from joint max_torque.
    pub motor_limits: Vec<(f32, f32)>,
    /// Number of gripper command dimensions.
    pub num_grippers: usize,
}

impl ActionSpace {
    /// Build an ActionSpace from a robot definition.
    pub fn from_definition(def: &RobotDefinition) -> Self {
        let motor_limits: Vec<(f32, f32)> = def
            .joints
            .iter()
            .map(|j| (-j.max_torque, j.max_torque))
            .collect();

        Self {
            num_motors: def.joints.len(),
            motor_limits,
            num_grippers: 0, // grippers are not defined in RobotDefinition
        }
    }
}

/// An action to apply to a robot: motor velocity targets and gripper commands.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotAction {
    /// Target velocity for each motor (one per joint).
    pub motor_velocities: Vec<f32>,
    /// Gripper commands: true = close, false = open.
    pub gripper_commands: Vec<bool>,
}

/// Apply a RobotAction to a RobotState by setting actuator commands.
///
/// Each motor velocity becomes an `ActuatorCommand::Velocity`. The number of
/// motor velocities is clamped to the number of joints in the definition.
pub fn apply_action(def: &RobotDefinition, state: &mut RobotState, action: &RobotAction) {
    let num_joints = def.joints.len();
    let num_motors = action.motor_velocities.len().min(num_joints);

    state.actuator_commands.clear();
    state.actuator_commands.extend(
        action.motor_velocities[..num_motors]
            .iter()
            .map(|&v| ActuatorCommand::Velocity(v)),
    );
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
                    body_zone: None,
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
                    body_zone: None,
                },
                LinkDefinition {
                    name: "link_2".to_string(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: Some(1),
                    body_zone: None,
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

    // ---- Task 6: Robot state serialization tests ----

    use crate::robot::sensors::ImuReading;

    /// Helper: build a RobotDefinition with 2 joints, 1 distance sensor, 1 contact
    /// sensor, and 1 IMU sensor for gym-interface tests.
    fn gym_definition() -> RobotDefinition {
        use crate::robot::definition::SensorDefinition;
        RobotDefinition {
            name: "gym_bot".to_string(),
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
                    name: "link_1".to_string(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: Some(0),
                    body_zone: None,
                },
                LinkDefinition {
                    name: "link_2".to_string(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: Some(1),
                    body_zone: None,
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
                    joint_type: JointType::Revolute,
                    axis: Vec3::Y,
                    parent_link: 1,
                    child_link: 2,
                    limit_min: -std::f32::consts::PI,
                    limit_max: std::f32::consts::PI,
                    max_torque: 5.0,
                    damping: 0.05,
                },
            ],
            sensors: vec![
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Distance {
                        direction: Vec3::Z,
                        max_range: 10.0,
                    },
                },
                SensorMount {
                    link_index: 1,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Contact,
                },
                SensorMount {
                    link_index: 2,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Imu,
                },
            ],
        }
    }

    #[test]
    fn test_robot_state_json_roundtrip() {
        // Build a GymRobotState with non-trivial values and verify JSON round-trip.
        let state = GymRobotState {
            joint_positions: vec![1.0, -0.5],
            joint_velocities: vec![0.1, -0.2],
            sensor_readings: GymSensorReadings {
                distances: vec![5.0, 3.2],
                contacts: vec![true, false],
                imu: vec![ImuReading {
                    linear_acceleration: Vec3::new(0.0, -9.81, 0.0),
                    angular_velocity: Vec3::new(0.1, 0.0, 0.0),
                }],
                camera_visible: vec![vec![0, 2], vec![1]],
            },
            gripper_states: vec![GripperState {
                is_open: false,
                attached_object: Some(3),
            }],
            combat: None,
        };

        let json = serde_json::to_string(&state).expect("serialization failed");
        let deser: GymRobotState = serde_json::from_str(&json).expect("deserialization failed");

        // Joint positions
        assert_eq!(deser.joint_positions.len(), state.joint_positions.len());
        for (a, b) in state
            .joint_positions
            .iter()
            .zip(deser.joint_positions.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
        // Joint velocities
        for (a, b) in state
            .joint_velocities
            .iter()
            .zip(deser.joint_velocities.iter())
        {
            assert!((a - b).abs() < 1e-6);
        }
        // Sensor readings
        assert_eq!(
            deser.sensor_readings.distances.len(),
            state.sensor_readings.distances.len()
        );
        assert_eq!(
            deser.sensor_readings.contacts,
            state.sensor_readings.contacts
        );
        assert_eq!(
            deser.sensor_readings.camera_visible,
            state.sensor_readings.camera_visible
        );
        assert_eq!(
            deser.sensor_readings.imu.len(),
            state.sensor_readings.imu.len()
        );
        // Gripper states
        assert_eq!(deser.gripper_states.len(), 1);
        assert_eq!(deser.gripper_states[0].is_open, false);
        assert_eq!(deser.gripper_states[0].attached_object, Some(3));
    }

    #[test]
    fn test_observation_space_describes_robot() {
        let def = gym_definition();
        let obs = ObservationSpace::from_definition(&def);

        // 2 joints => 2 joint position dims + 2 joint velocity dims
        assert_eq!(
            obs.num_joint_positions, 2,
            "should have 2 joint position dimensions"
        );
        assert_eq!(
            obs.num_joint_velocities, 2,
            "should have 2 joint velocity dimensions"
        );
        // 3 sensors
        assert_eq!(
            obs.num_sensors,
            def.sensors.len(),
            "should match number of sensors"
        );
    }

    #[test]
    fn test_action_space_describes_robot() {
        let def = gym_definition();
        let action = ActionSpace::from_definition(&def);

        // 2 joints => 2 motor velocity dimensions
        assert_eq!(
            action.num_motors, 2,
            "should have 2 motor dimensions (one per joint)"
        );
        // Limits should match joint limits
        for (i, joint) in def.joints.iter().enumerate() {
            assert!(
                (action.motor_limits[i].0 - (-joint.max_torque)).abs() < 1e-6,
                "motor limit min should be -max_torque"
            );
            assert!(
                (action.motor_limits[i].1 - joint.max_torque).abs() < 1e-6,
                "motor limit max should be max_torque"
            );
        }
    }

    #[test]
    fn test_apply_action_sets_motors() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);

        let action = RobotAction {
            motor_velocities: vec![1.5, -2.0],
            gripper_commands: vec![],
        };

        apply_action(&def, &mut state, &action);

        // Actuator commands should be set to Velocity for each motor
        assert_eq!(state.actuator_commands.len(), 2);
        assert_eq!(state.actuator_commands[0], ActuatorCommand::Velocity(1.5));
        assert_eq!(state.actuator_commands[1], ActuatorCommand::Velocity(-2.0));
    }

    #[test]
    fn test_apply_action_sets_grippers() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);

        let action = RobotAction {
            motor_velocities: vec![0.0, 0.0],
            gripper_commands: vec![true, false],
        };

        apply_action(&def, &mut state, &action);

        // Motor commands applied
        assert_eq!(state.actuator_commands.len(), 2);
        // Gripper commands are stored in the action and accessible
        assert_eq!(action.gripper_commands[0], true);
        assert_eq!(action.gripper_commands[1], false);
    }

    #[test]
    fn test_empty_robot_state() {
        // Robot with no joints and no sensors should have empty state vectors.
        let def = RobotDefinition {
            name: "empty_bot".to_string(),
            links: vec![LinkDefinition {
                name: "base".to_string(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![],
        };

        let state = GymRobotState::from_robot_state(&RobotState::new(&def), &def);

        assert!(
            state.joint_positions.is_empty(),
            "no joints => empty joint_positions"
        );
        assert!(
            state.joint_velocities.is_empty(),
            "no joints => empty joint_velocities"
        );
        assert!(
            state.sensor_readings.distances.is_empty(),
            "no sensors => empty distances"
        );
        assert!(
            state.sensor_readings.contacts.is_empty(),
            "no sensors => empty contacts"
        );
        assert!(
            state.sensor_readings.imu.is_empty(),
            "no sensors => empty imu"
        );
        assert!(
            state.sensor_readings.camera_visible.is_empty(),
            "no sensors => empty camera_visible"
        );
        assert!(
            state.gripper_states.is_empty(),
            "no grippers => empty gripper_states"
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_set_joint_position_out_of_bounds_index() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        let pos_before = state.joint_positions.clone();

        state.set_joint_position(999, 1.0, -1.0, 1.0);

        assert_eq!(
            state.joint_positions, pos_before,
            "OOB index should be no-op"
        );
    }

    #[test]
    fn test_set_link_pose_out_of_bounds() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        let poses_before = state.link_poses.clone();

        state.set_link_pose(999, Mat4::from_translation(Vec3::ONE));

        assert_eq!(
            state.link_poses, poses_before,
            "OOB set_link_pose should be no-op"
        );
    }

    #[test]
    fn test_link_poses_as_mat4_roundtrip() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        let rot = Mat4::from_rotation_y(1.0);
        state.set_link_pose(0, rot);

        let mats = state.link_poses_as_mat4();
        let diff: f32 = mats[0]
            .to_cols_array()
            .iter()
            .zip(rot.to_cols_array().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-6, "link_poses_as_mat4 roundtrip failed");
    }

    #[test]
    fn test_apply_action_extra_motors() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);

        let action = RobotAction {
            motor_velocities: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            gripper_commands: vec![],
        };

        apply_action(&def, &mut state, &action);

        assert_eq!(
            state.actuator_commands.len(),
            2,
            "extra motor velocities should be truncated to joint count"
        );
    }

    #[test]
    fn test_apply_action_empty_motors() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);

        let action = RobotAction {
            motor_velocities: vec![],
            gripper_commands: vec![],
        };

        apply_action(&def, &mut state, &action);

        assert!(
            state.actuator_commands.is_empty(),
            "zero motor velocities should produce empty commands"
        );
    }

    #[test]
    fn test_gym_robot_state_from_all_sensor_types() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);
        state.sensor_readings = vec![
            SensorReading::Distance(5.0),
            SensorReading::Contact(true),
            SensorReading::Imu {
                linear_accel: Vec3::new(0.0, -9.81, 0.0),
                angular_vel: Vec3::new(0.1, 0.0, 0.0),
            },
        ];

        let gym_state = GymRobotState::from_robot_state(&state, &def);

        assert_eq!(gym_state.sensor_readings.distances.len(), 1);
        assert!((gym_state.sensor_readings.distances[0] - 5.0).abs() < 1e-6);
        assert_eq!(gym_state.sensor_readings.contacts, vec![true]);
        assert_eq!(gym_state.sensor_readings.imu.len(), 1);
    }

    #[test]
    fn test_gym_robot_state_with_lidar() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);
        state.sensor_readings = vec![SensorReading::Lidar(vec![1.0, 2.0, 3.0])];

        let gym_state = GymRobotState::from_robot_state(&state, &def);

        // LIDAR readings are not decomposed into distances
        assert!(
            gym_state.sensor_readings.distances.is_empty(),
            "LIDAR should not appear in distances"
        );
    }

    #[test]
    fn test_observation_space_empty_robot() {
        let def = RobotDefinition {
            name: "empty".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![],
        };

        let obs = ObservationSpace::from_definition(&def);
        assert_eq!(obs.num_joint_positions, 0);
        assert_eq!(obs.num_joint_velocities, 0);
        assert_eq!(obs.num_sensors, 0);
        assert!(obs.joint_position_limits.is_empty());
    }

    #[test]
    fn test_action_space_empty_robot() {
        let def = RobotDefinition {
            name: "empty".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![],
        };

        let action = ActionSpace::from_definition(&def);
        assert_eq!(action.num_motors, 0);
        assert!(action.motor_limits.is_empty());
        assert_eq!(action.num_grippers, 0);
    }

    // ---- Task 2: CombatState tests ----

    #[test]
    fn test_combat_state_new() {
        let cs = CombatState::new(100.0, 100.0);
        assert!((cs.health - 100.0).abs() < 1e-6);
        assert!((cs.max_health - 100.0).abs() < 1e-6);
        assert!((cs.stamina - 100.0).abs() < 1e-6);
        assert!((cs.max_stamina - 100.0).abs() < 1e-6);
        assert!(cs.recent_hits.is_empty());
        assert!((cs.total_damage_dealt - 0.0).abs() < 1e-6);
        assert!((cs.total_damage_received - 0.0).abs() < 1e-6);
        assert!(!cs.knockdown);
    }

    #[test]
    fn test_apply_damage_reduces_health() {
        let mut cs = CombatState::new(100.0, 100.0);
        cs.apply_damage(30.0);
        assert!(
            (cs.health - 70.0).abs() < 1e-6,
            "health should be 70 after 30 damage"
        );
        // Damage should not go below 0
        cs.apply_damage(200.0);
        assert!((cs.health - 0.0).abs() < 1e-6, "health should floor at 0");
    }

    #[test]
    fn test_knockdown_on_zero_health() {
        let mut cs = CombatState::new(50.0, 100.0);
        assert!(!cs.knockdown);
        cs.apply_damage(50.0);
        assert!(
            cs.knockdown,
            "knockdown should be true when health reaches 0"
        );
    }

    #[test]
    fn test_consume_stamina_success() {
        let mut cs = CombatState::new(100.0, 100.0);
        let ok = cs.consume_stamina(30.0);
        assert!(ok, "should return true when enough stamina");
        assert!(
            (cs.stamina - 70.0).abs() < 1e-6,
            "stamina should be 70 after consuming 30"
        );
    }

    #[test]
    fn test_consume_stamina_insufficient() {
        let mut cs = CombatState::new(100.0, 20.0);
        let ok = cs.consume_stamina(30.0);
        assert!(!ok, "should return false when insufficient stamina");
        assert!(
            (cs.stamina - 20.0).abs() < 1e-6,
            "stamina should be unchanged"
        );
    }

    #[test]
    fn test_regenerate_stamina() {
        let mut cs = CombatState::new(100.0, 100.0);
        cs.consume_stamina(50.0);
        assert!((cs.stamina - 50.0).abs() < 1e-6);
        // Regen at 5.0/sec for 2 seconds = +10
        cs.regenerate_stamina(2.0);
        assert!(
            (cs.stamina - 60.0).abs() < 1e-6,
            "stamina should be 60 after 2s regen"
        );
        // Should not exceed max
        cs.regenerate_stamina(100.0);
        assert!(
            (cs.stamina - 100.0).abs() < 1e-6,
            "stamina should cap at max"
        );
    }

    #[test]
    fn test_robot_state_combat_default_none() {
        // Deserialize a RobotState JSON that does not include a "combat" field.
        // The serde default should make it None.
        let def = test_definition();
        let state = RobotState::new(&def);
        let json = serde_json::to_string(&state).expect("serialize");
        let deser: RobotState = serde_json::from_str(&json).expect("deserialize");
        assert!(deser.combat.is_none(), "combat should default to None");
    }

    // ---- Task 4: Link velocity tests ----

    #[test]
    fn test_compute_link_velocities_stationary() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        // Both current and previous poses are identity (stationary)
        state.prev_link_poses = state.link_poses.clone();

        let vels = state.compute_link_velocities(1.0);
        assert_eq!(vels.len(), def.links.len());
        for (i, v) in vels.iter().enumerate() {
            assert!(
                v.length() < 1e-6,
                "link {} velocity should be zero for stationary poses, got {:?}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_compute_link_velocities_moving() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        // Previous poses: identity
        state.prev_link_poses = state.link_poses.clone();
        // Move link 0 by (1, 0, 0)
        state.set_link_pose(0, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)));

        let vels = state.compute_link_velocities(1.0);
        assert!(
            (vels[0] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6,
            "link 0 velocity should be (1,0,0), got {:?}",
            vels[0]
        );
        // Other links stayed at identity
        for v in &vels[1..] {
            assert!(
                v.length() < 1e-6,
                "stationary links should have zero velocity, got {:?}",
                v
            );
        }
    }

    #[test]
    fn test_save_previous_poses() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        assert!(
            state.prev_link_poses.is_empty(),
            "prev_link_poses should start empty"
        );

        state.save_previous_poses();
        assert_eq!(
            state.prev_link_poses, state.link_poses,
            "after save, prev_link_poses should equal link_poses"
        );
    }

    #[test]
    fn test_compute_velocities_first_step() {
        let def = test_definition();
        let state = RobotState::new(&def);
        // prev_link_poses is empty on first step
        assert!(state.prev_link_poses.is_empty());

        let vels = state.compute_link_velocities(1.0);
        assert_eq!(vels.len(), def.links.len());
        for (i, v) in vels.iter().enumerate() {
            assert!(
                v.length() < 1e-6,
                "link {} velocity should be zero on first step, got {:?}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_state_serialization_with_all_reading_types() {
        let def = test_definition();
        let mut state = RobotState::new(&def);
        state.sensor_readings = vec![
            SensorReading::Distance(1.5),
            SensorReading::Lidar(vec![1.0, 2.0]),
            SensorReading::Contact(true),
            SensorReading::Imu {
                linear_accel: Vec3::new(0.0, -9.81, 0.0),
                angular_vel: Vec3::ZERO,
            },
        ];

        let json = serde_json::to_string(&state).unwrap();
        let deser: RobotState = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.sensor_readings.len(), 4);
        assert_eq!(deser.sensor_readings[0], SensorReading::Distance(1.5));
        assert_eq!(
            deser.sensor_readings[1],
            SensorReading::Lidar(vec![1.0, 2.0])
        );
        assert_eq!(deser.sensor_readings[2], SensorReading::Contact(true));
    }

    // ---- Task 6: GymCombatState integration tests ----

    #[test]
    fn test_gym_state_with_combat() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);
        state.combat = Some(CombatState::new(100.0, 100.0));

        let gym_state = GymRobotState::from_robot_state(&state, &def);
        assert!(gym_state.combat.is_some(), "combat should be Some");
        let combat = gym_state.combat.unwrap();
        assert!((combat.health - 100.0).abs() < 1e-6);
        assert!((combat.max_health - 100.0).abs() < 1e-6);
        assert!((combat.stamina - 100.0).abs() < 1e-6);
        assert!((combat.max_stamina - 100.0).abs() < 1e-6);
        assert!(!combat.knockdown);
        assert!(combat.recent_hits.is_empty());
        assert!((combat.total_damage_dealt - 0.0).abs() < 1e-6);
        assert!((combat.total_damage_received - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gym_state_without_combat() {
        let def = gym_definition();
        let state = RobotState::new(&def);
        assert!(state.combat.is_none(), "default combat should be None");

        let gym_state = GymRobotState::from_robot_state(&state, &def);
        assert!(
            gym_state.combat.is_none(),
            "gym combat should be None when source is None"
        );
    }

    #[test]
    fn test_gym_combat_state_serialization() {
        let combat = GymCombatState {
            health: 75.0,
            max_health: 100.0,
            stamina: 40.0,
            max_stamina: 80.0,
            knockdown: false,
            recent_hits: vec![],
            total_damage_dealt: 25.0,
            total_damage_received: 10.0,
        };

        let json = serde_json::to_string(&combat).expect("serialization failed");
        let deser: GymCombatState = serde_json::from_str(&json).expect("deserialization failed");

        assert!((deser.health - 75.0).abs() < 1e-6);
        assert!((deser.max_health - 100.0).abs() < 1e-6);
        assert!((deser.stamina - 40.0).abs() < 1e-6);
        assert!((deser.max_stamina - 80.0).abs() < 1e-6);
        assert!(!deser.knockdown);
        assert!(deser.recent_hits.is_empty());
        assert!((deser.total_damage_dealt - 25.0).abs() < 1e-6);
        assert!((deser.total_damage_received - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_gym_state_combat_via_buffered_constructor() {
        let def = gym_definition();
        let mut state = RobotState::new(&def);
        state.combat = Some(CombatState::new(80.0, 60.0));
        // Apply some damage to verify it propagates
        state.combat.as_mut().unwrap().apply_damage(20.0);

        let mut buf = GymStateBuffer::new();
        let gym_state = GymRobotState::from_robot_state_into(&state, &def, &mut buf);

        assert!(gym_state.combat.is_some());
        let combat = gym_state.combat.unwrap();
        assert!(
            (combat.health - 60.0).abs() < 1e-6,
            "health should be 80 - 20 = 60"
        );
        assert!((combat.max_health - 80.0).abs() < 1e-6);
        assert!((combat.stamina - 60.0).abs() < 1e-6);
        assert!(!combat.knockdown);
    }
}
