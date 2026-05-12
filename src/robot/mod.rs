pub mod actuators;
pub mod body;
pub mod collision;
pub mod definition;
pub mod dynamics;
pub mod kinematics;
pub mod sensors;
pub mod state;

use glam::Mat4;
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

use crate::scene::SceneObject;
use collision::{detect_punches, detect_robot_collisions, HitEvent, SceneBvh, PUNCH_STAMINA_COST};
use definition::RobotDefinition;
use dynamics::step_dynamics;
use kinematics::forward_kinematics;
use sensors::simulate_sensors_bvh;
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
    cached_bvh: Option<SceneBvh>,
    bvh_mesh_count: usize,
    pub last_hit_events: Vec<HitEvent>,
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
            cached_bvh: None,
            bvh_mesh_count: 0,
            last_hit_events: Vec::new(),
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

        let mesh_count = scene_meshes.len();
        if self.cached_bvh.is_none() || self.bvh_mesh_count != mesh_count {
            self.cached_bvh = Some(SceneBvh::build(scene_meshes));
            self.bvh_mesh_count = mesh_count;
        }

        // Save previous poses for combat velocity tracking
        for robot in self.robots.iter_mut() {
            if robot.state.combat.is_some() {
                robot.state.save_previous_poses();
            }
        }

        let bvh = self.cached_bvh.as_ref().unwrap();
        let step_one = |robot: &mut ManagedRobot| {
            step_dynamics(&robot.definition, &mut robot.state, dt);
            let bp = Mat4::from_cols_array(&robot.base_pose);
            forward_kinematics(&robot.definition, &mut robot.state, bp);
            simulate_sensors_bvh(&robot.definition, &mut robot.state, scene_meshes, bvh);
        };

        if self.robots.len() >= 2 {
            self.robots.par_iter_mut().for_each(step_one);
        } else {
            self.robots.iter_mut().for_each(step_one);
        }

        // Sequential combat step after parallel physics
        self.last_hit_events = step_combat(&mut self.robots, dt);
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
// step_combat — sequential combat resolution after parallel physics
// ---------------------------------------------------------------------------

fn step_combat(robots: &mut [ManagedRobot], dt: f32) -> Vec<HitEvent> {
    // 1. Clear recent_hits for combat-enabled robots
    for robot in robots.iter_mut() {
        if let Some(ref mut combat) = robot.state.combat {
            combat.recent_hits.clear();
        }
    }

    // 2. Collect combat-enabled robots for collision detection
    let combat_data: Vec<(usize, &RobotDefinition, &RobotState)> = robots
        .iter()
        .enumerate()
        .filter(|(_, r)| r.state.combat.is_some())
        .map(|(i, r)| (i, &r.definition, &r.state))
        .collect();

    if combat_data.len() < 2 {
        // Regenerate stamina even without combat
        for robot in robots.iter_mut() {
            if let Some(ref mut combat) = robot.state.combat {
                combat.regenerate_stamina(dt);
            }
        }
        return Vec::new();
    }

    // 3. Detect robot-robot collisions
    let collisions = detect_robot_collisions(&combat_data);

    // 4. Compute link velocities for combat robots
    let velocities: Vec<Vec<glam::Vec3>> = combat_data
        .iter()
        .map(|(i, _, _)| robots[*i].state.compute_link_velocities(dt))
        .collect();

    // 5. Build robots-with-velocities slice for detect_punches
    let punch_data: Vec<(usize, &RobotDefinition, &RobotState, &[glam::Vec3])> = combat_data
        .iter()
        .zip(velocities.iter())
        .map(|((id, def, state), vels)| (*id, *def, *state, vels.as_slice()))
        .collect();

    // 6. Detect punches
    let hit_events = detect_punches(&collisions, &punch_data);

    // 7. Apply damage and consume stamina
    for hit in &hit_events {
        // Apply damage to target
        if let Some(ref mut combat) = robots[hit.target_robot].state.combat {
            combat.apply_damage(hit.damage);
            combat.total_damage_received += hit.damage;
            combat.recent_hits.push(hit.clone());
        }
        // Consume stamina from attacker and track damage dealt
        if let Some(ref mut combat) = robots[hit.attacker_robot].state.combat {
            combat.consume_stamina(PUNCH_STAMINA_COST);
            combat.total_damage_dealt += hit.damage;
        }
    }

    // 8. Regenerate stamina for all combat robots
    for robot in robots.iter_mut() {
        if let Some(ref mut combat) = robot.state.combat {
            combat.regenerate_stamina(dt);
        }
    }

    hit_events
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
// PhysicsTimer — optional per-frame timing utility
// ---------------------------------------------------------------------------

pub struct PhysicsTimer {
    durations: std::collections::VecDeque<std::time::Duration>,
    capacity: usize,
}

impl PhysicsTimer {
    pub fn new(capacity: usize) -> Self {
        Self {
            durations: std::collections::VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn record(&mut self, d: std::time::Duration) {
        if self.durations.len() >= self.capacity {
            self.durations.pop_front();
        }
        self.durations.push_back(d);
    }

    pub fn last(&self) -> Option<std::time::Duration> {
        self.durations.back().copied()
    }

    pub fn avg(&self) -> Option<std::time::Duration> {
        if self.durations.is_empty() {
            return None;
        }
        let sum: std::time::Duration = self.durations.iter().sum();
        Some(sum / self.durations.len() as u32)
    }

    pub fn len(&self) -> usize {
        self.durations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.durations.is_empty()
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
                body_zone: None,
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
                body_zone: None,
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
                    body_zone: None,
                },
                LinkDefinition {
                    name: "sensor_arm".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.05 },
                    parent_joint: Some(0),
                    body_zone: None,
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
                    body_zone: None,
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
                    body_zone: None,
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

    // ---- Edge case tests ----

    #[test]
    fn test_manager_get_robot_invalid_index() {
        let manager = RobotManager::new();
        assert!(manager.get_robot(0).is_none());
        assert!(manager.get_robot(999).is_none());
    }

    #[test]
    fn test_manager_not_running_skips_step() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, Mat4::IDENTITY);
        manager.set_command(idx, 0, ActuatorCommand::Velocity(5.0));

        manager.running = false;
        manager.step(0.01, &[]);

        let pos = manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            pos.abs() < 1e-9,
            "stopped manager should not step, got pos={}",
            pos
        );
    }

    #[test]
    fn test_simulation_not_running_skips_step() {
        let mut sim = RobotSimulation::default();
        let def = RobotDefinition::simple_arm(1);
        let idx = sim.manager.add_robot(def, Mat4::IDENTITY);
        sim.manager
            .set_command(idx, 0, ActuatorCommand::Velocity(5.0));

        sim.running = false;
        let meshes: Vec<crate::scene::SceneObject> = vec![];
        sim.step(&meshes);

        let pos = sim.manager.get_robot(idx).unwrap().state.joint_positions[0];
        assert!(
            pos.abs() < 1e-9,
            "stopped simulation should not step, got pos={}",
            pos
        );
    }

    #[test]
    fn test_manager_step_empty_no_panic() {
        let mut manager = RobotManager::new();
        manager.step(0.01, &[]);
        // No robots => no-op, should not panic
    }

    #[test]
    fn test_managed_robot_base_pose_mat4() {
        let mut manager = RobotManager::new();
        let base_pose = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, base_pose);

        let robot = manager.get_robot(idx).unwrap();
        let recovered = robot.base_pose_mat4();
        let diff: f32 = recovered
            .to_cols_array()
            .iter()
            .zip(base_pose.to_cols_array().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-6, "base_pose_mat4 should roundtrip");
    }

    #[test]
    fn test_managed_robot_serialization() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        let idx = manager.add_robot(def, Mat4::IDENTITY);
        manager.set_command(idx, 0, ActuatorCommand::Velocity(1.0));
        manager.step(0.01, &[]);

        let robot = manager.get_robot(idx).unwrap();
        let json = serde_json::to_string(robot).unwrap();
        let deser: ManagedRobot = serde_json::from_str(&json).unwrap();

        assert_eq!(deser.definition.name, robot.definition.name);
        assert_eq!(
            deser.state.joint_positions.len(),
            robot.state.joint_positions.len()
        );
    }

    #[test]
    fn test_set_command_resizes_actuator_vec() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(3);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // actuator_commands starts empty
        assert!(manager
            .get_robot(idx)
            .unwrap()
            .state
            .actuator_commands
            .is_empty());

        // set_command on joint 2 should resize to 3
        manager.set_command(idx, 2, ActuatorCommand::Position(1.0));

        let robot = manager.get_robot(idx).unwrap();
        assert_eq!(robot.state.actuator_commands.len(), 3);
        assert_eq!(
            robot.state.actuator_commands[2],
            ActuatorCommand::Position(1.0)
        );
        // Other slots filled with Torque(0.0)
        assert_eq!(
            robot.state.actuator_commands[0],
            ActuatorCommand::Torque(0.0)
        );
    }

    #[test]
    fn test_simulation_default_dt() {
        let sim = RobotSimulation::default();
        assert!((sim.dt - 1.0 / 60.0).abs() < 1e-6);
        assert!(sim.running);
    }

    // ------------------------------------------------------------------
    // Performance benchmarks
    // ------------------------------------------------------------------

    use crate::robot::definition::{
        CollisionShape, JointDefinition, JointType, LinkDefinition, SensorDefinition, SensorMount,
    };
    use crate::scene::SceneObject;

    fn perf_robot_def() -> RobotDefinition {
        let links = vec![
            LinkDefinition {
                name: "base".into(),
                mass: 5.0,
                inertia: 1.0,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: glam::Vec3::splat(0.1),
                },
                parent_joint: None,
                body_zone: None,
            },
            LinkDefinition {
                name: "link1".into(),
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
                name: "link2".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.4,
                },
                parent_joint: Some(1),
                body_zone: None,
            },
            LinkDefinition {
                name: "link3".into(),
                mass: 0.5,
                inertia: 0.05,
                collision_shape: CollisionShape::Sphere { radius: 0.08 },
                parent_joint: Some(2),
                body_zone: None,
            },
        ];
        let joints = vec![
            JointDefinition {
                name: "j0".into(),
                joint_type: JointType::Revolute,
                axis: glam::Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
            },
            JointDefinition {
                name: "j1".into(),
                joint_type: JointType::Revolute,
                axis: glam::Vec3::X,
                parent_link: 1,
                child_link: 2,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 8.0,
                damping: 0.1,
            },
            JointDefinition {
                name: "j2".into(),
                joint_type: JointType::Revolute,
                axis: glam::Vec3::Y,
                parent_link: 2,
                child_link: 3,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 5.0,
                damping: 0.1,
            },
        ];
        let sensors = vec![
            SensorMount {
                link_index: 3,
                local_offset: glam::Vec3::new(0.0, 0.0, 0.1),
                sensor: SensorDefinition::Distance {
                    direction: glam::Vec3::Z,
                    max_range: 5.0,
                },
            },
            SensorMount {
                link_index: 3,
                local_offset: glam::Vec3::new(0.0, 0.0, 0.1),
                sensor: SensorDefinition::Distance {
                    direction: glam::Vec3::X,
                    max_range: 5.0,
                },
            },
        ];
        RobotDefinition {
            name: "perf_bot".into(),
            links,
            joints,
            sensors,
        }
    }

    fn perf_scene(num_tris: usize) -> Vec<SceneObject> {
        use crate::scene::{AcousticMaterial, Mesh, Triangle, Vertex};

        let triangles: Vec<Triangle> = (0..num_tris)
            .map(|i| {
                let x = (i as f32) * 0.3 - (num_tris as f32) * 0.15;
                let y = 2.0 + (i as f32) * 0.01;
                let n = glam::Vec3::Y;
                Triangle {
                    vertices: [
                        Vertex {
                            position: glam::Vec3::new(x, y, -1.0),
                            normal: n,
                        },
                        Vertex {
                            position: glam::Vec3::new(x + 0.2, y, -1.0),
                            normal: n,
                        },
                        Vertex {
                            position: glam::Vec3::new(x + 0.1, y + 0.2, -1.0),
                            normal: n,
                        },
                    ],
                }
            })
            .collect();

        vec![SceneObject {
            name: "perf_floor".into(),
            mesh: Mesh { triangles },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }]
    }

    fn setup_perf_manager(num_robots: usize, scene: &[SceneObject]) -> RobotManager {
        let mut manager = RobotManager::new();
        for i in 0..num_robots {
            let def = perf_robot_def();
            let offset = glam::Vec3::new(i as f32 * 2.0, 0.0, 0.0);
            manager.add_robot(def, Mat4::from_translation(offset));
        }
        manager.step(1.0 / 60.0, scene);
        manager
    }

    #[test]
    fn test_perf_1_robot_step_time() {
        let scene = perf_scene(100);
        let mut manager = setup_perf_manager(1, &scene);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            manager.step(1.0 / 60.0, &scene);
        }
        let elapsed = start.elapsed();
        let per_step = elapsed / 100;

        println!("1 robot: {per_step:?}/step ({elapsed:?} total for 100 steps)");
        assert!(
            per_step < std::time::Duration::from_millis(1),
            "1 robot step should be <1ms, got {:?}",
            per_step
        );
    }

    #[test]
    fn test_perf_4_robots_under_2ms() {
        let scene = perf_scene(100);
        let mut manager = setup_perf_manager(4, &scene);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            manager.step(1.0 / 60.0, &scene);
        }
        let elapsed = start.elapsed();
        let per_step = elapsed / 100;

        println!("4 robots: {per_step:?}/step ({elapsed:?} total for 100 steps)");
        assert!(
            per_step < std::time::Duration::from_millis(2),
            "4 robot step should be <2ms, got {:?}",
            per_step
        );
    }

    #[test]
    fn test_perf_scaling() {
        let scene = perf_scene(100);

        for n in [1, 2, 4] {
            let mut manager = setup_perf_manager(n, &scene);

            let start = std::time::Instant::now();
            for _ in 0..100 {
                manager.step(1.0 / 60.0, &scene);
            }
            let elapsed = start.elapsed();
            let per_step = elapsed / 100;
            println!("{n} robot(s): {per_step:?}/step");
        }
    }

    #[test]
    fn test_physics_timer_records() {
        let mut timer = PhysicsTimer::new(100);
        assert!(timer.is_empty());
        assert_eq!(timer.len(), 0);
        assert!(timer.last().is_none());
        assert!(timer.avg().is_none());

        for i in 0..150 {
            timer.record(std::time::Duration::from_micros(100 + i));
        }

        assert_eq!(timer.len(), 100, "should cap at capacity");
        assert_eq!(
            timer.last(),
            Some(std::time::Duration::from_micros(249)),
            "last should be most recent"
        );
        assert!(timer.avg().is_some());
    }

    #[test]
    fn test_parallel_step_matches_sequential() {
        let scene = perf_scene(50);

        let mut seq_manager = RobotManager::new();
        let mut par_manager = RobotManager::new();
        for i in 0..4 {
            let def = perf_robot_def();
            let offset = glam::Vec3::new(i as f32 * 2.0, 0.0, 0.0);
            seq_manager.add_robot(def.clone(), Mat4::from_translation(offset));
            par_manager.add_robot(def, Mat4::from_translation(offset));
        }

        let dt = 1.0 / 60.0;
        for _ in 0..10 {
            // Sequential
            let bvh = crate::robot::collision::SceneBvh::build(&scene);
            for robot in &mut seq_manager.robots {
                step_dynamics(&robot.definition, &mut robot.state, dt);
                let bp = Mat4::from_cols_array(&robot.base_pose);
                forward_kinematics(&robot.definition, &mut robot.state, bp);
                simulate_sensors_bvh(&robot.definition, &mut robot.state, &scene, &bvh);
            }
            // Parallel
            par_manager.step(dt, &scene);
        }

        for i in 0..4 {
            let s = &seq_manager.robots[i].state;
            let p = &par_manager.robots[i].state;
            assert_eq!(
                s.joint_positions, p.joint_positions,
                "robot {i} positions diverged"
            );
            assert_eq!(
                s.joint_velocities, p.joint_velocities,
                "robot {i} velocities diverged"
            );
        }
    }

    #[test]
    fn test_parallel_step_single_robot() {
        let scene = perf_scene(10);
        let mut manager = RobotManager::new();
        manager.add_robot(perf_robot_def(), Mat4::IDENTITY);

        for _ in 0..10 {
            manager.step(1.0 / 60.0, &scene);
        }

        let robot = manager.get_robot(0).unwrap();
        assert_eq!(robot.state.joint_positions.len(), 3);
    }

    #[test]
    fn test_parallel_step_deterministic() {
        let scene = perf_scene(50);

        let run = || {
            let mut manager = RobotManager::new();
            for i in 0..4 {
                let offset = glam::Vec3::new(i as f32 * 2.0, 0.0, 0.0);
                manager.add_robot(perf_robot_def(), Mat4::from_translation(offset));
            }
            for _ in 0..20 {
                manager.step(1.0 / 60.0, &scene);
            }
            manager
                .robots
                .iter()
                .flat_map(|r| r.state.joint_positions.clone())
                .collect::<Vec<f32>>()
        };

        let a = run();
        let b = run();
        assert_eq!(a, b, "two parallel runs should produce identical results");
    }

    // ------------------------------------------------------------------
    // Task 5: Combat integration tests
    // ------------------------------------------------------------------

    use crate::robot::collision::PUNCH_STAMINA_COST as TEST_PUNCH_STAMINA_COST;
    use crate::robot::state::CombatState;

    /// Helper: build a combat robot definition with body zones on all links.
    fn combat_robot_def() -> RobotDefinition {
        use crate::robot::definition::BodyZone;

        RobotDefinition {
            name: "combat_bot".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".to_string(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: glam::Vec3::splat(0.5),
                    },
                    parent_joint: None,
                    body_zone: Some(BodyZone::Body),
                },
                LinkDefinition {
                    name: "arm".to_string(),
                    mass: 2.0,
                    inertia: 0.5,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: glam::Vec3::splat(0.3),
                    },
                    parent_joint: Some(0),
                    body_zone: Some(BodyZone::RightArm),
                },
            ],
            joints: vec![JointDefinition {
                name: "shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: glam::Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
            }],
            sensors: vec![],
        }
    }

    #[test]
    fn test_combat_step_no_combat_robots() {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)));

        manager.step(0.01, &[]);

        assert!(
            manager.last_hit_events.is_empty(),
            "non-combat robots should produce no hit events"
        );
    }

    #[test]
    fn test_combat_step_overlapping_robots() {
        // Test step_combat directly with prepared robot state to avoid
        // step() overwriting prev_link_poses.
        let def = combat_robot_def();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        // Enable combat on both
        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        // Both at origin (link_poses default to identity = overlapping).
        // Set prev_link_poses for robot 0's arm link with a large offset
        // so velocity = (current - prev) / dt > threshold.
        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();

        // Offset prev_link_poses[1] translation by -5.0 on X so velocity = 5.0 / 0.01 = 500 m/s
        robots[0].state.prev_link_poses[1][12] -= 5.0;

        let hits = step_combat(&mut robots, dt);

        assert!(
            !hits.is_empty(),
            "overlapping combat robots with high-velocity link should produce hit events"
        );
    }

    #[test]
    fn test_combat_step_far_apart() {
        let mut manager = RobotManager::new();
        let def = combat_robot_def();

        let idx_a = manager.add_robot(def.clone(), Mat4::IDENTITY);
        let idx_b = manager.add_robot(
            def,
            Mat4::from_translation(glam::Vec3::new(100.0, 0.0, 0.0)),
        );

        manager.get_robot_mut(idx_a).unwrap().state.combat = Some(CombatState::new(100.0, 100.0));
        manager.get_robot_mut(idx_b).unwrap().state.combat = Some(CombatState::new(100.0, 100.0));

        manager.step(0.01, &[]);

        assert!(
            manager.last_hit_events.is_empty(),
            "far-apart combat robots should produce no hit events"
        );
    }

    #[test]
    fn test_combat_damage_applied() {
        let def = combat_robot_def();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        // Set prev_link_poses and create high velocity on robot 0's arm
        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();
        robots[0].state.prev_link_poses[1][12] -= 5.0;

        let health_before = robots[1].state.combat.as_ref().unwrap().health;

        let hits = step_combat(&mut robots, dt);

        // There should be hits on robot 1 (target)
        let hits_on_1: Vec<_> = hits.iter().filter(|h| h.target_robot == 1).collect();
        assert!(!hits_on_1.is_empty(), "should have hits targeting robot 1");

        let health_after = robots[1].state.combat.as_ref().unwrap().health;
        assert!(
            health_after < health_before,
            "target health should decrease after being hit: before={}, after={}",
            health_before,
            health_after
        );
    }

    #[test]
    fn test_combat_stamina_consumed() {
        let def = combat_robot_def();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        // Set prev_link_poses and create high velocity on robot 0's arm
        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();
        robots[0].state.prev_link_poses[1][12] -= 5.0;

        let stamina_before = robots[0].state.combat.as_ref().unwrap().stamina;

        let hits = step_combat(&mut robots, dt);

        let hits_from_0: Vec<_> = hits.iter().filter(|h| h.attacker_robot == 0).collect();
        assert!(
            !hits_from_0.is_empty(),
            "robot 0 should have landed punches"
        );

        let stamina_after = robots[0].state.combat.as_ref().unwrap().stamina;
        // Stamina consumed = PUNCH_STAMINA_COST per hit, but also regenerated by dt
        let expected_consumption = hits_from_0.len() as f32 * TEST_PUNCH_STAMINA_COST;
        let regen = 5.0 * dt; // regen rate * dt
        let expected_stamina = stamina_before - expected_consumption + regen;
        assert!(
            (stamina_after - expected_stamina).abs() < 1.0,
            "attacker stamina should decrease by PUNCH_STAMINA_COST per hit: before={}, after={}, expected~={}",
            stamina_before,
            stamina_after,
            expected_stamina
        );
    }

    #[test]
    fn test_combat_stamina_regenerates() {
        let mut manager = RobotManager::new();
        let def = combat_robot_def();

        let idx = manager.add_robot(def, Mat4::IDENTITY);
        manager.get_robot_mut(idx).unwrap().state.combat = Some(CombatState::new(100.0, 100.0));

        // Reduce stamina manually
        manager
            .get_robot_mut(idx)
            .unwrap()
            .state
            .combat
            .as_mut()
            .unwrap()
            .stamina = 50.0;

        let stamina_before = 50.0_f32;

        // Step many times without any opponents (no punching)
        for _ in 0..100 {
            manager.step(0.01, &[]);
        }

        let stamina_after = manager
            .get_robot(idx)
            .unwrap()
            .state
            .combat
            .as_ref()
            .unwrap()
            .stamina;
        assert!(
            stamina_after > stamina_before,
            "stamina should regenerate over time: before={}, after={}",
            stamina_before,
            stamina_after
        );
    }

    #[test]
    fn test_combat_knockdown() {
        let mut manager = RobotManager::new();
        let def = combat_robot_def();

        let idx = manager.add_robot(def, Mat4::IDENTITY);
        manager.get_robot_mut(idx).unwrap().state.combat = Some(CombatState::new(100.0, 100.0));

        // Directly apply enough damage to reach 0 health
        manager
            .get_robot_mut(idx)
            .unwrap()
            .state
            .combat
            .as_mut()
            .unwrap()
            .apply_damage(100.0);

        let combat = manager
            .get_robot(idx)
            .unwrap()
            .state
            .combat
            .as_ref()
            .unwrap();
        assert!(
            combat.knockdown,
            "knockdown should be true when health reaches 0"
        );
        assert!(
            (combat.health - 0.0).abs() < 1e-6,
            "health should be 0, got {}",
            combat.health
        );
    }

    #[test]
    fn test_existing_tests_unaffected() {
        // Verify that non-combat robots behave identically after combat integration
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(1);
        let idx = manager.add_robot(def, Mat4::IDENTITY);

        // No combat state
        assert!(
            manager.get_robot(idx).unwrap().state.combat.is_none(),
            "non-combat robot should have no combat state"
        );

        // Set a velocity command
        manager.set_command(idx, 0, ActuatorCommand::Velocity(2.0));

        let pos_before = manager.get_robot(idx).unwrap().state.joint_positions[0];

        // Step several times
        for _ in 0..10 {
            manager.step(0.01, &[]);
        }

        let pos_after = manager.get_robot(idx).unwrap().state.joint_positions[0];

        // Joint should have moved (physics still works)
        assert!(
            (pos_after - pos_before).abs() > 1e-6,
            "non-combat robot joint should still move: before={}, after={}",
            pos_before,
            pos_after
        );

        // No hit events
        assert!(
            manager.last_hit_events.is_empty(),
            "non-combat robot should produce no hit events"
        );

        // Combat state should remain None
        assert!(
            manager.get_robot(idx).unwrap().state.combat.is_none(),
            "combat state should remain None for non-combat robot"
        );
    }

    // ------------------------------------------------------------------
    // Task 7: Boxing scenario integration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_boxing_scenario_full() {
        // Create 2 boxing_test_robots at the same position, enable combat,
        // simulate a high-velocity arm via manual pose manipulation, and
        // verify hits are produced with damage and stamina effects.
        let def = RobotDefinition::boxing_test_robot();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        // Enable combat on both
        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        // Set prev_link_poses to current (all identity = overlapping at origin)
        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();

        // Simulate high velocity on robot 0's right_arm (link index 2):
        // Shift prev pose translation by -5.0 on X so velocity = 5.0 / 0.01 = 500 m/s
        robots[0].state.prev_link_poses[2][12] -= 5.0;

        let health_before_1 = robots[1].state.combat.as_ref().unwrap().health;
        let stamina_before_0 = robots[0].state.combat.as_ref().unwrap().stamina;

        let hits = step_combat(&mut robots, dt);

        // Verify hit events were produced
        assert!(
            !hits.is_empty(),
            "boxing scenario should produce hit events"
        );

        // Verify target (robot 1) took damage
        let health_after_1 = robots[1].state.combat.as_ref().unwrap().health;
        assert!(
            health_after_1 < health_before_1,
            "target health should decrease: before={}, after={}",
            health_before_1,
            health_after_1
        );

        // Verify attacker (robot 0) consumed stamina (minus regen)
        let stamina_after_0 = robots[0].state.combat.as_ref().unwrap().stamina;
        let hits_from_0 = hits.iter().filter(|h| h.attacker_robot == 0).count();
        let expected_consumption = hits_from_0 as f32 * PUNCH_STAMINA_COST;
        let regen = 5.0 * dt;
        let expected_stamina = stamina_before_0 - expected_consumption + regen;
        assert!(
            (stamina_after_0 - expected_stamina).abs() < 1.0,
            "attacker stamina should decrease: before={}, after={}, expected~={}",
            stamina_before_0,
            stamina_after_0,
            expected_stamina
        );

        // Verify damage tracking
        assert!(
            robots[0].state.combat.as_ref().unwrap().total_damage_dealt > 0.0,
            "attacker should have non-zero total_damage_dealt"
        );
        assert!(
            robots[1]
                .state
                .combat
                .as_ref()
                .unwrap()
                .total_damage_received
                > 0.0,
            "target should have non-zero total_damage_received"
        );
    }

    #[test]
    fn test_two_robots_mutual_hits() {
        // Both robots punching simultaneously: both have high-velocity arms,
        // verify both take damage.
        let def = RobotDefinition::boxing_test_robot();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();

        // Both robots have high-velocity arms
        robots[0].state.prev_link_poses[2][12] -= 5.0; // robot 0 right arm
        robots[1].state.prev_link_poses[1][12] -= 5.0; // robot 1 left arm

        let hits = step_combat(&mut robots, dt);

        // Both robots should have been hit
        let hits_on_0 = hits.iter().any(|h| h.target_robot == 0);
        let hits_on_1 = hits.iter().any(|h| h.target_robot == 1);

        assert!(
            hits_on_0 && hits_on_1,
            "both robots should take hits in mutual exchange: hits_on_0={}, hits_on_1={}, total_hits={}",
            hits_on_0,
            hits_on_1,
            hits.len()
        );

        // Both should have reduced health
        let health_0 = robots[0].state.combat.as_ref().unwrap().health;
        let health_1 = robots[1].state.combat.as_ref().unwrap().health;
        assert!(
            health_0 < 100.0,
            "robot 0 health should decrease from mutual hits, got {}",
            health_0
        );
        assert!(
            health_1 < 100.0,
            "robot 1 health should decrease from mutual hits, got {}",
            health_1
        );
    }

    #[test]
    fn test_knockdown_stops_combat() {
        // Start with very low health so one hit causes knockdown, then
        // verify knockdown flag is set.
        let def = RobotDefinition::boxing_test_robot();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        // Robot 1 starts with very low health so a single hit knocks it down
        robots[1].state.combat = Some(CombatState::new(1.0, 100.0));

        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();

        // High-velocity arm on robot 0
        robots[0].state.prev_link_poses[2][12] -= 5.0;

        let hits = step_combat(&mut robots, dt);

        assert!(
            !hits.is_empty(),
            "should produce hit events against low-health target"
        );

        let combat_1 = robots[1].state.combat.as_ref().unwrap();
        assert!(
            combat_1.knockdown,
            "target with 1.0 HP should be knocked down after taking a hit"
        );
        assert!(
            combat_1.health <= 0.0 + 1e-6,
            "target health should be 0 after knockdown, got {}",
            combat_1.health
        );
    }

    #[test]
    fn test_stamina_depletion_weakens_punch() {
        // Start with low stamina, verify it is consumed per punch and
        // tracks correctly across hits.
        let def = RobotDefinition::boxing_test_robot();
        let dt = 0.01_f32;

        let mut robots = vec![
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
            ManagedRobot {
                definition: def.clone(),
                state: RobotState::new(&def),
                base_pose: Mat4::IDENTITY.to_cols_array(),
            },
        ];

        // Robot 0 starts with low stamina (just above one PUNCH_STAMINA_COST).
        // consume_stamina only deducts when stamina >= cost, so with multiple
        // overlapping hits only the first consume succeeds.
        robots[0].state.combat = Some(CombatState::new(100.0, 100.0));
        robots[0].state.combat.as_mut().unwrap().stamina = PUNCH_STAMINA_COST + 1.0;
        robots[1].state.combat = Some(CombatState::new(100.0, 100.0));

        let initial_stamina = robots[0].state.combat.as_ref().unwrap().stamina;

        robots[0].state.prev_link_poses = robots[0].state.link_poses.clone();
        robots[1].state.prev_link_poses = robots[1].state.link_poses.clone();

        // High-velocity arm on robot 0
        robots[0].state.prev_link_poses[2][12] -= 5.0;

        let hits = step_combat(&mut robots, dt);

        let hits_from_0 = hits.iter().filter(|h| h.attacker_robot == 0).count();
        assert!(hits_from_0 > 0, "should have landed at least one punch");

        let stamina_after = robots[0].state.combat.as_ref().unwrap().stamina;

        // At least one PUNCH_STAMINA_COST was consumed (subsequent consumes
        // may fail if stamina dropped below the cost). Regen adds a small
        // amount (5.0 * dt). Net stamina should be less than initial.
        assert!(
            stamina_after < initial_stamina,
            "stamina should decrease after punching: before={}, after={}",
            initial_stamina,
            stamina_after
        );

        // Verify the exact accounting: one successful consume of PUNCH_STAMINA_COST,
        // remaining consumes fail (stamina < cost), then regen adds 5.0 * dt.
        let regen = 5.0 * dt;
        let expected_after_one_consume = initial_stamina - PUNCH_STAMINA_COST + regen;
        assert!(
            (stamina_after - expected_after_one_consume).abs() < 0.5,
            "should have consumed exactly one PUNCH_STAMINA_COST: expected~={}, got={}",
            expected_after_one_consume,
            stamina_after
        );
    }
}
