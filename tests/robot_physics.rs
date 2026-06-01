//! Rigorous robot-physics correctness tests.
//!
//! Unlike the "doesn't panic / stays finite" guards in `physics_stability.rs`, these assert real
//! physical properties of the gravity-driven articulated dynamics added in goal 020:
//!   (a) a pendulum's small-angle period matches the analytical `2π√(I/mgL)`,
//!   (b) total mechanical energy (KE + PE) is conserved over a long undamped swing,
//!   (c) a PD controller actively compensates gravity (holds a horizontal pose at a small sag),
//!   (d) the parallel `RobotManager::step` is deterministic.

use echomap::robot::definition::{
    CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
};
use echomap::robot::dynamics::{step_dynamics_with_gravity, DEFAULT_GRAVITY};
use echomap::robot::kinematics::forward_kinematics;
use echomap::robot::state::{ActuatorCommand, RobotState};
use echomap::robot::RobotManager;
use glam::{Mat4, Vec3};

const G: f32 = 9.81;
const PI: f32 = std::f32::consts::PI;

/// Single-link pendulum: the bob COM sits `length` m from the pivot along +X (via `child_offset`),
/// rotating about +Z. Joint position 0 ⇒ horizontal; hanging equilibrium ⇒ θ = -π/2.
fn pendulum(length: f32, mass: f32, inertia: f32, damping: f32) -> RobotDefinition {
    RobotDefinition {
        name: "pendulum".into(),
        links: vec![
            LinkDefinition {
                name: "base".into(),
                mass: 0.0,
                inertia: 1.0,
                collision_shape: CollisionShape::Sphere { radius: 0.05 },
                parent_joint: None,
                body_zone: None,
            },
            LinkDefinition {
                name: "bob".into(),
                mass,
                inertia,
                collision_shape: CollisionShape::Sphere { radius: 0.05 },
                parent_joint: Some(0),
                body_zone: None,
            },
        ],
        joints: vec![JointDefinition {
            name: "pivot".into(),
            joint_type: JointType::Revolute,
            axis: Vec3::Z,
            parent_link: 0,
            child_link: 1,
            limit_min: -PI,
            limit_max: PI,
            max_torque: 1.0e6,
            damping,
            anchor_offset: Vec3::ZERO,
            child_offset: Vec3::new(length, 0.0, 0.0),
        }],
        sensors: Vec::new(),
    }
}

/// COM world height (y) of the bob for joint angle θ: link points along (cosθ, sinθ, 0)·L.
fn bob_height(length: f32, theta: f32) -> f32 {
    length * theta.sin()
}

#[test]
fn pendulum_small_angle_period_matches_analytic() {
    let (length, mass, inertia) = (0.5_f32, 1.0_f32, 0.2_f32);
    let def = pendulum(length, mass, inertia, 0.0);
    let mut state = RobotState::new(&def);
    state.actuator_commands = vec![ActuatorCommand::Torque(0.0)];

    let eq = -PI / 2.0; // hanging equilibrium
    let amp = 0.05_f32; // small displacement ⇒ sinθ ≈ θ
    state.joint_positions[0] = eq + amp;
    state.joint_velocities[0] = 0.0;

    // Seed link poses at the initial configuration so gravity acts from step 1 (no first-frame lag).
    forward_kinematics(&def, &mut state, Mat4::IDENTITY);

    let dt = 0.0002_f32;
    let mut prev_disp = state.joint_positions[0] - eq;
    let mut crossings = Vec::new();
    for k in 1..40_000 {
        step_dynamics_with_gravity(&def, &mut state, dt, DEFAULT_GRAVITY);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);
        let disp = state.joint_positions[0] - eq;
        if prev_disp.signum() != disp.signum() {
            crossings.push(k as f32 * dt);
            if crossings.len() == 3 {
                break;
            }
        }
        prev_disp = disp;
    }

    assert!(
        crossings.len() == 3,
        "pendulum should oscillate (zero-crossings)"
    );
    // Two zero-crossings span half a period; three span a full period.
    let measured = crossings[2] - crossings[0];
    let analytic = 2.0 * PI * (inertia / (mass * G * length)).sqrt();
    let rel_err = (measured - analytic).abs() / analytic;
    assert!(
        rel_err < 0.02,
        "measured period {measured:.4}s vs analytic {analytic:.4}s (rel err {:.3})",
        rel_err
    );
}

#[test]
fn pendulum_conserves_total_energy_undamped() {
    let (length, mass, inertia) = (0.5_f32, 1.0_f32, 0.2_f32);
    let def = pendulum(length, mass, inertia, 0.0);
    let mut state = RobotState::new(&def);
    state.actuator_commands = vec![ActuatorCommand::Torque(0.0)];

    // Release from horizontal (θ = 0): KE = 0, PE = m·G·0 = 0 ⇒ E0 = 0.
    state.joint_positions[0] = 0.0;
    state.joint_velocities[0] = 0.0;
    forward_kinematics(&def, &mut state, Mat4::IDENTITY);

    let dt = 0.0002_f32;
    let energy = |theta: f32, omega: f32| {
        0.5 * inertia * omega * omega + mass * G * bob_height(length, theta)
    };
    let e0 = energy(state.joint_positions[0], state.joint_velocities[0]);
    let scale = mass * G * length; // the PE swing magnitude (≈ 4.9 J)

    let mut max_dev = 0.0_f32;
    for _ in 0..25_000 {
        // 5 s
        step_dynamics_with_gravity(&def, &mut state, dt, DEFAULT_GRAVITY);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);
        let e = energy(state.joint_positions[0], state.joint_velocities[0]);
        max_dev = max_dev.max((e - e0).abs());
    }
    assert!(
        max_dev < 0.01 * scale,
        "energy drift {max_dev:.5} J should be < 1% of the {scale:.3} J swing"
    );
}

#[test]
fn pd_controller_compensates_gravity_holding_horizontal() {
    let (length, mass, inertia) = (0.5_f32, 1.0_f32, 0.2_f32);
    let def = pendulum(length, mass, inertia, 3.0); // damped so it settles
    let mut state = RobotState::new(&def);
    // Command: hold the link horizontal (θ = 0) against gravity.
    state.actuator_commands = vec![ActuatorCommand::Position(0.0)];
    state.joint_positions[0] = 0.0;
    state.joint_velocities[0] = 0.0;
    forward_kinematics(&def, &mut state, Mat4::IDENTITY);

    let dt = 0.001_f32;
    for _ in 0..8000 {
        // 8 s — long enough to settle
        step_dynamics_with_gravity(&def, &mut state, dt, DEFAULT_GRAVITY);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);
    }

    let theta_ss = state.joint_positions[0];
    // It must have settled and must hold NEAR (but not exactly at) horizontal — gravity sags it a
    // few degrees and the PD torque actively balances it. θ_ss ≈ -m·g·L/Kp ≈ -0.049 rad with Kp=100.
    assert!(
        state.joint_velocities[0].abs() < 1e-2,
        "joint should have settled, |ω| = {}",
        state.joint_velocities[0]
    );
    assert!(
        theta_ss < -0.02 && theta_ss > -0.12,
        "PD should hold near horizontal with a small gravity sag, θ_ss = {theta_ss}"
    );
}

/// Build a manager with `n` identical moving arms (so the parallel step path is exercised).
fn moving_arms_manager(n: usize) -> RobotManager {
    let mut m = RobotManager::new();
    for i in 0..n {
        let def = RobotDefinition::simple_arm(3);
        let idx = m.add_robot(
            def,
            Mat4::from_translation(Vec3::new(i as f32 * 2.0, 0.0, 0.0)),
        );
        // Drive each joint toward a distinct target so the state actually evolves.
        m.robots[idx].state.actuator_commands = vec![
            ActuatorCommand::Position(0.7),
            ActuatorCommand::Position(-0.5),
            ActuatorCommand::Velocity(0.3),
        ];
    }
    m
}

#[test]
fn parallel_step_is_deterministic() {
    let mut a = moving_arms_manager(4);
    let mut b = moving_arms_manager(4);
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        a.step(dt, &[]);
        b.step(dt, &[]);
    }
    for (ra, rb) in a.robots.iter().zip(b.robots.iter()) {
        assert_eq!(
            ra.state.joint_positions, rb.state.joint_positions,
            "parallel step must be deterministic (positions)"
        );
        assert_eq!(
            ra.state.joint_velocities, rb.state.joint_velocities,
            "parallel step must be deterministic (velocities)"
        );
        assert_eq!(
            ra.state.link_poses, rb.state.link_poses,
            "parallel step must be deterministic (link poses)"
        );
    }
}
