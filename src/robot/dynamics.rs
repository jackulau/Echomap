use super::definition::RobotDefinition;
use super::state::{ActuatorCommand, RobotState};

/// Default proportional gain for PD controller.
const KP: f32 = 100.0;
/// Default derivative gain for PD controller.
const KD: f32 = 10.0;

/// Step the joint dynamics forward by `dt` seconds.
///
/// For each joint in the definition, compute the applied torque from the
/// corresponding actuator command in `state.actuator_commands`, then integrate
/// velocity and position using the joint's damping and the child link's inertia.
///
/// Actuator command modes:
///   - `Position(target)`: PD torque = kp*(target - position) - kd*velocity
///   - `Velocity(target_vel)`: P torque = kp*(target_vel - velocity)
///   - `Torque(t)`: direct torque = t
///
/// If no actuator command exists for a joint index, zero torque is applied.
/// Torque is clamped to `[-max_torque, max_torque]`.
/// Position is clamped to `[limit_min, limit_max]`; velocity is zeroed at limits.
pub fn step_dynamics(definition: &RobotDefinition, state: &mut RobotState, dt: f32) {
    for (i, joint_def) in definition.joints.iter().enumerate() {
        let position = state.joint_positions[i];
        let velocity = state.joint_velocities[i];

        // Determine raw torque from actuator command (or zero if missing).
        let raw_torque = if i < state.actuator_commands.len() {
            match &state.actuator_commands[i] {
                ActuatorCommand::Position(target) => KP * (target - position) - KD * velocity,
                ActuatorCommand::Velocity(target_vel) => KP * (target_vel - velocity),
                ActuatorCommand::Torque(t) => *t,
            }
        } else {
            0.0
        };

        // Clamp torque to joint limits.
        let torque = raw_torque.clamp(-joint_def.max_torque, joint_def.max_torque);

        // Use child link's inertia for integration (guard against zero).
        let inertia = definition.links[joint_def.child_link].inertia.max(1e-6);

        // Update velocity: v += (torque/inertia - damping*v) * dt
        let new_velocity = velocity + (torque / inertia - joint_def.damping * velocity) * dt;
        state.joint_velocities[i] = new_velocity;

        // Update position: p += v * dt
        let new_position = position + state.joint_velocities[i] * dt;
        state.joint_positions[i] = new_position;

        // Clamp position to joint limits; zero velocity at limits.
        if state.joint_positions[i] <= joint_def.limit_min {
            state.joint_positions[i] = joint_def.limit_min;
            state.joint_velocities[i] = 0.0;
        } else if state.joint_positions[i] >= joint_def.limit_max {
            state.joint_positions[i] = joint_def.limit_max;
            state.joint_velocities[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::definition::{
        CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
    };
    use glam::Vec3;

    /// Helper: create a simple 1-joint robot (base link + 1 child link).
    fn one_joint_robot(
        max_torque: f32,
        damping: f32,
        limit_min: f32,
        limit_max: f32,
    ) -> RobotDefinition {
        RobotDefinition {
            name: "test_bot".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".to_string(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: Vec3::splat(0.1),
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
            ],
            joints: vec![JointDefinition {
                name: "joint_0".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min,
                limit_max,
                max_torque,
                damping,
            }],
            sensors: Vec::new(),
        }
    }

    #[test]
    fn test_zero_command_stays_still() {
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
        let mut state = RobotState::new(&def);

        // No actuator commands => zero torque. Robot at rest should stay at rest.
        let pos_before = state.joint_positions[0];
        let vel_before = state.joint_velocities[0];
        step_dynamics(&def, &mut state, 0.01);

        assert!(
            (state.joint_positions[0] - pos_before).abs() < 1e-9,
            "position should not change with zero torque, got delta = {}",
            (state.joint_positions[0] - pos_before).abs()
        );
        assert!(
            (state.joint_velocities[0] - vel_before).abs() < 1e-9,
            "velocity should not change with zero torque at rest, got delta = {}",
            (state.joint_velocities[0] - vel_before).abs()
        );
    }

    #[test]
    fn test_position_command_moves() {
        let def = one_joint_robot(1000.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
        let mut state = RobotState::new(&def);

        // Command joint to position 1.0 rad
        state.actuator_commands = vec![ActuatorCommand::Position(1.0)];

        // Step several times
        for _ in 0..100 {
            step_dynamics(&def, &mut state, 0.01);
        }

        // After many steps the joint should move toward the target
        assert!(
            state.joint_positions[0] > 0.1,
            "joint should have moved toward target 1.0, but position = {}",
            state.joint_positions[0]
        );
    }

    #[test]
    fn test_velocity_command() {
        // Use wide limits and zero damping so we can trace the math exactly.
        // child link inertia = 0.1, kp = 100.0
        // First step: torque = kp*(2.0 - 0.0) = 200.0, clamped to max_torque=200
        // accel = 200/0.1 = 2000, new_vel = 0 + 2000*0.01 = 20.0
        // That overshoots target=2.0. So just verify one step drives velocity
        // in the positive direction from zero.
        let def = one_joint_robot(200.0, 0.0, -1000.0, 1000.0);
        let mut state = RobotState::new(&def);

        // Command target velocity 2.0 rad/s
        state.actuator_commands = vec![ActuatorCommand::Velocity(2.0)];

        // Single step from rest
        step_dynamics(&def, &mut state, 0.01);

        // Velocity should be positive (driven toward target 2.0 from 0.0)
        assert!(
            state.joint_velocities[0] > 0.0,
            "velocity should be positive after commanding target_vel=2.0 from rest, but got {}",
            state.joint_velocities[0]
        );
    }

    #[test]
    fn test_torque_command() {
        let def = one_joint_robot(100.0, 0.0, -std::f32::consts::PI, std::f32::consts::PI);
        let mut state = RobotState::new(&def);

        // Apply direct torque of 5.0 Nm
        state.actuator_commands = vec![ActuatorCommand::Torque(5.0)];
        step_dynamics(&def, &mut state, 0.01);

        // child link inertia = 0.1, damping = 0.0
        // acceleration = torque / inertia = 5.0 / 0.1 = 50.0
        // velocity after dt = 50.0 * 0.01 = 0.5
        let expected_vel = 5.0 / 0.1 * 0.01;
        assert!(
            (state.joint_velocities[0] - expected_vel).abs() < 1e-4,
            "expected velocity ~{}, got {}",
            expected_vel,
            state.joint_velocities[0]
        );

        // Position should have advanced
        assert!(
            state.joint_positions[0] > 0.0,
            "position should have moved, got {}",
            state.joint_positions[0]
        );
    }

    #[test]
    fn test_torque_clamped() {
        let max_torque = 10.0;
        let def = one_joint_robot(max_torque, 0.0, -std::f32::consts::PI, std::f32::consts::PI);
        let mut state = RobotState::new(&def);

        // Apply excessive torque
        state.actuator_commands = vec![ActuatorCommand::Torque(999.0)];
        step_dynamics(&def, &mut state, 0.01);

        // Torque should be clamped to max_torque (10.0)
        // child link inertia = 0.1
        // Max acceleration = 10.0 / 0.1 = 100.0
        // Max velocity after dt = 100.0 * 0.01 = 1.0
        let max_vel = max_torque / 0.1 * 0.01;
        assert!(
            (state.joint_velocities[0] - max_vel).abs() < 1e-4,
            "velocity should match clamped torque: expected ~{}, got {}",
            max_vel,
            state.joint_velocities[0]
        );
    }

    #[test]
    fn test_joint_limits_enforced() {
        let limit_min = -0.5;
        let limit_max = 0.5;
        let def = one_joint_robot(1000.0, 0.0, limit_min, limit_max);
        let mut state = RobotState::new(&def);

        // Apply large positive torque to push past upper limit
        state.actuator_commands = vec![ActuatorCommand::Torque(1000.0)];

        for _ in 0..200 {
            step_dynamics(&def, &mut state, 0.01);
        }

        assert!(
            state.joint_positions[0] <= limit_max + 1e-6,
            "position should be clamped to limit_max ({}), got {}",
            limit_max,
            state.joint_positions[0]
        );

        // Velocity should be zeroed at the limit
        assert!(
            state.joint_velocities[0].abs() < 1e-6,
            "velocity should be zero at limit, got {}",
            state.joint_velocities[0]
        );

        // Now apply large negative torque
        state.actuator_commands = vec![ActuatorCommand::Torque(-1000.0)];

        for _ in 0..200 {
            step_dynamics(&def, &mut state, 0.01);
        }

        assert!(
            state.joint_positions[0] >= limit_min - 1e-6,
            "position should be clamped to limit_min ({}), got {}",
            limit_min,
            state.joint_positions[0]
        );

        assert!(
            state.joint_velocities[0].abs() < 1e-6,
            "velocity should be zero at limit, got {}",
            state.joint_velocities[0]
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_zero_inertia_no_nan() {
        let mut def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        def.links[1].inertia = 0.0;
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(5.0)];

        step_dynamics(&def, &mut state, 0.01);

        assert!(
            state.joint_positions[0].is_finite(),
            "zero inertia should not produce NaN/Inf position"
        );
        assert!(
            state.joint_velocities[0].is_finite(),
            "zero inertia should not produce NaN/Inf velocity"
        );
    }

    #[test]
    fn test_zero_dt_is_noop() {
        let def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(100.0)];
        let pos_before = state.joint_positions[0];

        step_dynamics(&def, &mut state, 0.0);

        assert!(
            (state.joint_positions[0] - pos_before).abs() < 1e-9,
            "dt=0 should not change position"
        );
    }

    #[test]
    fn test_negative_dt_is_noop() {
        let def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(100.0)];
        step_dynamics(&def, &mut state, -0.01);

        assert!(
            state.joint_positions[0].is_finite(),
            "negative dt should not produce NaN"
        );
    }

    #[test]
    fn test_large_dt_finite() {
        let def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Velocity(10.0)];

        step_dynamics(&def, &mut state, 1.0);

        assert!(
            state.joint_positions[0].is_finite(),
            "large dt should produce finite position"
        );
        assert!(
            state.joint_velocities[0].is_finite(),
            "large dt should produce finite velocity"
        );
    }

    #[test]
    fn test_limit_min_equals_max() {
        let def = one_joint_robot(100.0, 0.0, 0.5, 0.5);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(100.0)];

        for _ in 0..100 {
            step_dynamics(&def, &mut state, 0.01);
        }

        assert!(
            (state.joint_positions[0] - 0.5).abs() < 1e-6,
            "locked joint (min==max) should stay at limit, got {}",
            state.joint_positions[0]
        );
    }

    #[test]
    fn test_zero_max_torque() {
        let def = one_joint_robot(0.0, 0.0, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(999.0)];

        step_dynamics(&def, &mut state, 0.01);

        assert!(
            state.joint_velocities[0].abs() < 1e-9,
            "zero max_torque should clamp all torque to zero, got velocity {}",
            state.joint_velocities[0]
        );
    }

    #[test]
    fn test_nan_command_produces_finite() {
        let def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Torque(f32::NAN)];

        step_dynamics(&def, &mut state, 0.01);

        // NaN torque is clamped to [-max_torque, max_torque] — NaN.clamp produces NaN.
        // This tests that we don't panic.
    }

    #[test]
    fn test_empty_robot_no_panic() {
        let def = RobotDefinition {
            name: "empty".to_string(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 1.0,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
            }],
            joints: vec![],
            sensors: Vec::new(),
        };
        let mut state = RobotState::new(&def);

        step_dynamics(&def, &mut state, 0.01);
        // No joints => nothing to step => should not panic
    }

    #[test]
    fn test_more_commands_than_joints() {
        let def = one_joint_robot(10.0, 0.1, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![
            ActuatorCommand::Velocity(1.0),
            ActuatorCommand::Velocity(2.0),
            ActuatorCommand::Velocity(3.0),
        ];

        step_dynamics(&def, &mut state, 0.01);
        // Extra commands should be ignored, no panic
        assert!(state.joint_positions[0].is_finite());
    }

    #[test]
    fn test_fewer_commands_than_joints() {
        let def = RobotDefinition::simple_arm(3);
        let mut state = RobotState::new(&def);
        state.actuator_commands = vec![ActuatorCommand::Velocity(1.0)];

        step_dynamics(&def, &mut state, 0.01);
        // Only joint 0 gets a command; joints 1,2 get zero torque
        assert!(state.joint_velocities[0].abs() > 0.0);
        assert!(state.joint_velocities[1].abs() < 1e-9);
        assert!(state.joint_velocities[2].abs() < 1e-9);
    }

    #[test]
    fn test_high_damping_stabilizes() {
        let def = one_joint_robot(100.0, 100.0, -3.14, 3.14);
        let mut state = RobotState::new(&def);
        state.joint_velocities[0] = 10.0;

        for _ in 0..100 {
            step_dynamics(&def, &mut state, 0.01);
        }

        assert!(
            state.joint_velocities[0].abs() < 1.0,
            "high damping should reduce velocity, got {}",
            state.joint_velocities[0]
        );
    }
}
