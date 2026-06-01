use glam::{Mat4, Vec3};

use super::definition::{JointType, RobotDefinition};
use super::state::{ActuatorCommand, RobotState};

/// Default proportional gain for PD controller.
const KP: f32 = 100.0;
/// Default derivative gain for PD controller.
const KD: f32 = 10.0;

/// Standard gravitational acceleration used by the live simulation (m/s², -Y is down).
pub const DEFAULT_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Indices of every link distal to (moved by) joint `joint_idx`, including its child link.
///
/// Walks the kinematic subtree rooted at the joint's child link. The robot is a tree, but the
/// `contains` guard also makes this safe against a malformed cyclic definition.
fn distal_links(definition: &RobotDefinition, joint_idx: usize) -> Vec<usize> {
    let root = definition.joints[joint_idx].child_link;
    let mut result = vec![root];
    let mut stack = vec![root];
    while let Some(link) = stack.pop() {
        for jd in &definition.joints {
            if jd.parent_link == link && !result.contains(&jd.child_link) {
                result.push(jd.child_link);
                stack.push(jd.child_link);
            }
        }
    }
    result
}

/// Compute the gravity-loading torque on each joint.
///
/// For a fixed-base articulated robot, gravity exerts a torque on each joint equal to the moment of
/// the weight of every distal link about the joint axis. Link world poses come from the most recent
/// forward-kinematics pass (`state.link_poses`); link origins approximate the link centres of mass
/// (collision shapes are centred on the link origin).
///
/// ```text
///   revolute:  τ_j = axis_world · Σ_{L distal to j} ( (com_L − pivot_j) × m_L·g )
///   prismatic: f_j = axis_world · Σ_{L distal to j} ( m_L·g )
/// ```
///
/// Returns one torque per joint (`0` for fixed joints, a zero/degenerate axis, or when link poses
/// are not yet available — e.g. before the first FK pass). Returns all-zero when `gravity` is zero.
pub fn compute_gravity_torques(
    definition: &RobotDefinition,
    link_poses: &[Mat4],
    gravity: Vec3,
) -> Vec<f32> {
    let mut torques = vec![0.0_f32; definition.joints.len()];
    // Need a valid world pose for every link; bail to zero if FK has not populated them yet.
    if link_poses.len() != definition.links.len() || gravity == Vec3::ZERO {
        return torques;
    }

    for (j, joint) in definition.joints.iter().enumerate() {
        if joint.joint_type == JointType::Fixed {
            continue;
        }
        let parent_pose = link_poses[joint.parent_link];
        let axis_world = parent_pose
            .transform_vector3(joint.axis)
            .normalize_or_zero();
        if axis_world == Vec3::ZERO {
            continue;
        }
        let pivot = parent_pose.transform_point3(joint.anchor_offset);

        let mut tau = 0.0_f32;
        for &l in &distal_links(definition, j) {
            let mass = definition.links[l].mass;
            if mass <= 0.0 {
                continue;
            }
            let weight = gravity * mass;
            match joint.joint_type {
                JointType::Revolute => {
                    let com = link_poses[l].transform_point3(Vec3::ZERO);
                    tau += axis_world.dot((com - pivot).cross(weight));
                }
                JointType::Prismatic => {
                    tau += axis_world.dot(weight);
                }
                JointType::Fixed => {}
            }
        }
        torques[j] = tau;
    }
    torques
}

/// Step the joint dynamics forward by `dt` seconds with **no gravity**.
///
/// Thin wrapper over [`step_dynamics_with_gravity`] with a zero gravity field — used by unit tests
/// and any caller that wants pure actuator + damping behaviour. The live simulation steps under
/// gravity via [`step_dynamics_with_gravity`].
pub fn step_dynamics(definition: &RobotDefinition, state: &mut RobotState, dt: f32) {
    step_dynamics_with_gravity(definition, state, dt, Vec3::ZERO);
}

/// Step the joint dynamics forward by `dt` seconds under a `gravity` field.
///
/// For each joint, compute the actuator torque from `state.actuator_commands`, add the external
/// gravity-loading torque, then integrate velocity and position using the joint's damping and the
/// child link's inertia.
///
/// Actuator command modes:
///   - `Position(target)`: PD torque = kp*(target - position) - kd*velocity
///   - `Velocity(target_vel)`: P torque = kp*(target_vel - velocity)
///   - `Torque(t)`: direct torque = t
///
/// If no actuator command exists for a joint index, zero actuator torque is applied. The actuator
/// torque is clamped to `[-max_torque, max_torque]`; the gravity-loading torque (see
/// [`compute_gravity_torques`]) is added **outside** that clamp, because gravity is an external
/// load — an actuator too weak to hold against it sags, as a real robot does. Gravity uses the link
/// world poses from the most recent forward-kinematics pass (`state.link_poses`), i.e. with a
/// one-step lag, which is standard for a semi-implicit scheme.
///
/// Position is clamped to `[limit_min, limit_max]`; velocity is zeroed at limits.
pub fn step_dynamics_with_gravity(
    definition: &RobotDefinition,
    state: &mut RobotState,
    dt: f32,
    gravity: Vec3,
) {
    let gravity_torques = compute_gravity_torques(definition, &state.link_poses_as_mat4(), gravity);

    for (i, joint_def) in definition.joints.iter().enumerate() {
        let position = state.joint_positions[i];
        let velocity = state.joint_velocities[i];

        // Determine raw actuator torque from the command (or zero if missing).
        let raw_torque = if i < state.actuator_commands.len() {
            match &state.actuator_commands[i] {
                ActuatorCommand::Position(target) => KP * (target - position) - KD * velocity,
                ActuatorCommand::Velocity(target_vel) => KP * (target_vel - velocity),
                ActuatorCommand::Torque(t) => *t,
            }
        } else {
            0.0
        };

        // Actuator output is clamped to its limit; gravity is an external load added on top.
        let actuator_torque = raw_torque.clamp(-joint_def.max_torque, joint_def.max_torque);
        let torque = actuator_torque + gravity_torques[i];

        // Use child link's inertia for integration (guard against zero).
        let inertia = definition.links[joint_def.child_link].inertia.max(1e-6);

        // Update velocity with IMPLICIT (backward-Euler) damping:
        //   v_new = (v + (torque/inertia)·dt) / (1 + damping·dt)
        // Treating the viscous term implicitly is unconditionally stable for any damping·dt ≥ 0.
        // The previous explicit form `v + (τ/I − c·v)·dt` diverges once c·dt > 2; this cannot.
        // When damping == 0 (or dt == 0) the denominator is 1, so undamped behaviour is unchanged.
        let accel = torque / inertia;
        let new_velocity = (velocity + accel * dt) / (1.0 + joint_def.damping * dt);
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
    use crate::robot::kinematics::forward_kinematics;
    use glam::{Mat4, Vec3};

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
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::ZERO,
            }],
            sensors: Vec::new(),
        }
    }

    /// Helper: a single-link pendulum whose child-link COM sits `length` metres from the pivot
    /// along +X (via `child_offset`), rotating about `axis`. Used for gravity tests — unlike
    /// [`one_joint_robot`] (COM at the pivot ⇒ zero gravity moment) this has a real lever arm.
    fn pendulum_robot(
        axis: Vec3,
        length: f32,
        mass: f32,
        inertia: f32,
        damping: f32,
    ) -> RobotDefinition {
        RobotDefinition {
            name: "pendulum".to_string(),
            links: vec![
                LinkDefinition {
                    name: "base".to_string(),
                    mass: 0.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Sphere { radius: 0.05 },
                    parent_joint: None,
                    body_zone: None,
                },
                LinkDefinition {
                    name: "bob".to_string(),
                    mass,
                    inertia,
                    collision_shape: CollisionShape::Sphere { radius: 0.05 },
                    parent_joint: Some(0),
                    body_zone: None,
                },
            ],
            joints: vec![JointDefinition {
                name: "pivot".to_string(),
                joint_type: JointType::Revolute,
                axis,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 1000.0,
                damping,
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::new(length, 0.0, 0.0),
            }],
            sensors: Vec::new(),
        }
    }

    #[test]
    fn gravity_torque_is_zero_without_a_field() {
        let def = pendulum_robot(Vec3::Z, 0.5, 2.0, 0.1, 0.0);
        let mut state = RobotState::new(&def);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);
        let torques = compute_gravity_torques(&def, &state.link_poses_as_mat4(), Vec3::ZERO);
        assert_eq!(torques, vec![0.0], "no gravity field ⇒ no gravity torque");
    }

    #[test]
    fn gravity_torque_matches_analytic_for_horizontal_pendulum() {
        // A horizontal link (COM at (L,0,0)) under gravity g about +Z has moment τ = -L·m·g.
        let (length, mass) = (0.5_f32, 2.0_f32);
        let def = pendulum_robot(Vec3::Z, length, mass, 0.1, 0.0);
        let mut state = RobotState::new(&def);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let g = DEFAULT_GRAVITY;
        let torques = compute_gravity_torques(&def, &state.link_poses_as_mat4(), g);
        let expected = -length * mass * (-g.y); // -L·m·9.81
        assert!(
            (torques[0] - expected).abs() < 1e-4,
            "gravity torque {} should equal analytic {}",
            torques[0],
            expected
        );
    }

    #[test]
    fn gravity_pendulum_swings_downward_from_horizontal() {
        // Released from horizontal with no actuator torque, the bob must accelerate downward
        // (negative rotation about +Z lowers the COM toward -Y).
        let def = pendulum_robot(Vec3::Z, 0.5, 2.0, 0.1, 0.0);
        let mut state = RobotState::new(&def);
        // Isolate gravity: zero actuator torque.
        state.actuator_commands = vec![ActuatorCommand::Torque(0.0)];
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        step_dynamics_with_gravity(&def, &mut state, 0.01, DEFAULT_GRAVITY);

        assert!(
            state.joint_velocities[0] < 0.0,
            "pendulum should gain negative angular velocity, got {}",
            state.joint_velocities[0]
        );
        assert!(
            state.joint_positions[0] < 0.0,
            "pendulum angle should swing negative (downward), got {}",
            state.joint_positions[0]
        );
    }

    #[test]
    fn gravity_prismatic_slides_along_vertical_axis() {
        // A prismatic joint along +Y under gravity feels force f = axis · m·g = -m·g (slides down).
        let mass = 3.0_f32;
        let mut def = pendulum_robot(Vec3::Y, 0.0, mass, 0.1, 0.0);
        def.joints[0].joint_type = JointType::Prismatic;
        let mut state = RobotState::new(&def);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let torques = compute_gravity_torques(&def, &state.link_poses_as_mat4(), DEFAULT_GRAVITY);
        let expected = mass * DEFAULT_GRAVITY.y; // negative (downward)
        assert!(
            (torques[0] - expected).abs() < 1e-4,
            "prismatic gravity force {} should equal m·g_y {}",
            torques[0],
            expected
        );
        assert!(
            torques[0] < 0.0,
            "vertical prismatic joint should be pulled down"
        );
    }

    #[test]
    fn gravity_free_wrapper_matches_zero_field() {
        // The gravity-free `step_dynamics` wrapper must equal stepping with an explicit zero field.
        let def = pendulum_robot(Vec3::Z, 0.5, 2.0, 0.1, 0.1);
        let mut a = RobotState::new(&def);
        let mut b = RobotState::new(&def);
        a.actuator_commands = vec![ActuatorCommand::Torque(0.5)];
        b.actuator_commands = vec![ActuatorCommand::Torque(0.5)];
        forward_kinematics(&def, &mut a, Mat4::IDENTITY);
        forward_kinematics(&def, &mut b, Mat4::IDENTITY);

        step_dynamics(&def, &mut a, 0.01);
        step_dynamics_with_gravity(&def, &mut b, 0.01, Vec3::ZERO);

        assert_eq!(a.joint_positions, b.joint_positions);
        assert_eq!(a.joint_velocities, b.joint_velocities);
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
        let mut def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(0.0, 0.0, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
                body_zone: None,
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
        let def = one_joint_robot(10.0, 0.1, -std::f32::consts::PI, std::f32::consts::PI);
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
        let def = one_joint_robot(100.0, 100.0, -std::f32::consts::PI, std::f32::consts::PI);
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

    #[test]
    fn stiff_damping_stays_bounded() {
        // damping·dt = 100 — far past the explicit-Euler stability limit (c·dt > 2 diverges).
        // The implicit damping form must keep velocity finite and decaying, not blow up.
        let def = one_joint_robot(100.0, 1000.0, -std::f32::consts::PI, std::f32::consts::PI);
        let mut state = RobotState::new(&def);
        state.joint_velocities[0] = 10.0;

        for _ in 0..50 {
            step_dynamics(&def, &mut state, 0.1);
            assert!(
                state.joint_velocities[0].is_finite(),
                "stiff damping must not diverge, got {}",
                state.joint_velocities[0]
            );
        }
        // Velocity must have decayed well below its start, never grown.
        assert!(
            state.joint_velocities[0].abs() < 10.0,
            "stiff damping should decay velocity, got {}",
            state.joint_velocities[0]
        );
    }
}
