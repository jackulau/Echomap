//! Numerical stability stress tests for physics solvers.
//!
//! Edge-regime coverage: CFL violations, energy conservation, NaN/Inf guards,
//! near-singular contact geometry. Each test pins a stability invariant that
//! must hold under adversarial parameters.

use echomap::fluids::grid::{CellType, FluidGrid};
use echomap::fluids::solver::{self as fluid_solver, FluidConfig};
use echomap::gas::grid::{GasCellType, GasGrid, GasSpecies};
use echomap::gas::solver::{self as gas_solver, GasConfig};
use echomap::robot::collision::{aabb_overlap, detect_punches, detect_robot_collisions, Aabb};
use echomap::robot::definition::{
    CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
};
use echomap::robot::dynamics::step_dynamics;
use echomap::robot::state::{ActuatorCommand, RobotState};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_fluid_grid(n: usize, dx: f32) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, n, dx, Vec3::ZERO);
    for ct in g.cell_types.iter_mut() {
        *ct = CellType::Fluid;
    }
    g
}

fn make_gas_grid(n: usize, dx: f32) -> GasGrid {
    let species = vec![GasSpecies {
        name: "CO2".to_string(),
        diffusion_coefficient: 0.2,
        molecular_weight: 44.0,
        density_at_stp: 1.977,
        color: [0.5, 0.5, 0.5],
    }];
    let mut g = GasGrid::new(n, n, n, dx, Vec3::ZERO, species);
    for ct in g.cell_types.iter_mut() {
        *ct = GasCellType::Gas;
    }
    g
}

fn fluid_has_nonfinite(g: &FluidGrid) -> bool {
    g.u.iter()
        .chain(g.v.iter())
        .chain(g.w.iter())
        .chain(g.pressure.iter())
        .chain(g.density.iter())
        .any(|v| !v.is_finite())
}

fn gas_has_nonfinite(g: &GasGrid) -> bool {
    let scalar = g
        .temperature
        .iter()
        .chain(g.pressure.iter())
        .chain(g.vel_x.iter())
        .chain(g.vel_y.iter())
        .chain(g.vel_z.iter())
        .any(|v| !v.is_finite());
    let conc = g
        .concentrations
        .iter()
        .flat_map(|c| c.iter())
        .any(|v| !v.is_finite());
    scalar || conc
}

fn single_revolute(damping: f32, inertia: f32) -> RobotDefinition {
    RobotDefinition {
        name: "stability_bot".to_string(),
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
                inertia,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.5,
                },
                parent_joint: Some(0),
                body_zone: None,
            },
        ],
        joints: vec![JointDefinition {
            name: "shoulder".to_string(),
            joint_type: JointType::Revolute,
            axis: Vec3::Y,
            parent_link: 0,
            child_link: 1,
            limit_min: -100.0,
            limit_max: 100.0,
            max_torque: 1000.0,
            damping,
            anchor_offset: Vec3::ZERO,
            child_offset: Vec3::ZERO,
        }],
        sensors: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// CFL violation detection — fluid + gas solvers stay stable
// ---------------------------------------------------------------------------

/// Fluid solver with dt above the CFL limit must remain finite (solver clamps
/// diffusion factor internally).
#[test]
fn fluid_solver_large_dt_no_nan() {
    let mut grid = make_fluid_grid(8, 0.125);
    for u in grid.u.iter_mut() {
        *u = 0.5;
    }

    let huge_dt_config = FluidConfig {
        dt: 1.0,
        viscosity: 0.01,
        density: 1000.0,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        surface_tension: 0.0,
        jacobi_iterations: 10,
    };

    for _ in 0..20 {
        fluid_solver::step(&mut grid, &huge_dt_config);
        assert!(
            !fluid_has_nonfinite(&grid),
            "Fluid solver produced NaN/Inf at large dt"
        );
    }
}

/// Fluid solver with extreme viscosity (factor would exceed stability limit)
/// must clamp internally and stay finite.
#[test]
fn fluid_solver_extreme_viscosity_clamps() {
    let mut grid = make_fluid_grid(6, 0.05);
    grid.u[0] = 1.0;

    let config = FluidConfig {
        dt: 0.1,
        viscosity: 1.0e6,
        density: 1000.0,
        gravity: Vec3::ZERO,
        surface_tension: 0.0,
        jacobi_iterations: 5,
    };

    fluid_solver::diffuse(&mut grid, config.viscosity, config.dt);
    assert!(
        !fluid_has_nonfinite(&grid),
        "Diffuse clamp failed under extreme viscosity"
    );

    let u_max = grid.u.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        u_max <= 1.0 + 1e-3,
        "Diffusion amplified velocity beyond initial magnitude: {u_max}"
    );
}

/// Gas solver under CFL-violating dt produces no NaN/Inf.
#[test]
fn gas_solver_large_dt_no_nan() {
    let mut grid = make_gas_grid(8, 0.1);
    let ci = grid.idx(4, 4, 4);
    grid.concentrations[0][ci] = 100.0;
    for v in grid.vel_x.iter_mut() {
        *v = 0.5;
    }

    let cfg = GasConfig {
        dt: 1.0,
        ambient_temperature: 293.15,
        thermal_diffusivity: 2.2e-5,
        buoyancy_coefficient: 0.01,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    };

    for _ in 0..20 {
        gas_solver::step(&mut grid, &cfg);
        assert!(!gas_has_nonfinite(&grid), "Gas solver produced NaN/Inf");
    }
}

/// Gas diffusion of an extreme coefficient must not blow up concentration.
#[test]
fn gas_diffusion_extreme_coeff_stays_bounded() {
    let species = vec![GasSpecies {
        name: "X".to_string(),
        diffusion_coefficient: 1.0e9,
        molecular_weight: 1.0,
        density_at_stp: 1.0,
        color: [0.0, 0.0, 0.0],
    }];
    let n = 8;
    let mut grid = GasGrid::new(n, n, n, 0.1, Vec3::ZERO, species);
    for ct in grid.cell_types.iter_mut() {
        *ct = GasCellType::Gas;
    }
    let ci = grid.idx(n / 2, n / 2, n / 2);
    grid.concentrations[0][ci] = 1.0;
    let initial: f32 = grid.concentrations[0].iter().sum();

    for _ in 0..20 {
        gas_solver::diffuse_concentrations(&mut grid, 0.1);
    }

    assert!(
        !gas_has_nonfinite(&grid),
        "Extreme diffusion coefficient produced NaN/Inf"
    );

    let total: f32 = grid.concentrations[0].iter().sum();
    let drift = ((total - initial) / initial).abs();
    assert!(
        drift < 0.05,
        "Mass drift {drift} exceeds 5% under extreme diffusion (initial={initial}, total={total})"
    );
}

// ---------------------------------------------------------------------------
// Energy conservation / decay in rigid-body dynamics
// ---------------------------------------------------------------------------

/// 10 seconds of free joint motion: zero actuation, zero damping → joint
/// velocity must remain constant (energy preserved).
#[test]
fn dynamics_energy_conservation_undamped_10s() {
    let def = single_revolute(0.0, 0.1);
    let mut state = RobotState::new(&def);
    state.joint_velocities[0] = 1.5;
    state.actuator_commands = vec![ActuatorCommand::Torque(0.0)];

    let dt: f32 = 0.001;
    let v0 = state.joint_velocities[0];
    let ke0 = 0.5 * def.links[1].inertia * v0 * v0;

    for _ in 0..10_000 {
        step_dynamics(&def, &mut state, dt);
    }

    let v = state.joint_velocities[0];
    let ke = 0.5 * def.links[1].inertia * v * v;
    let drift = ((ke - ke0) / ke0).abs();

    assert!(v.is_finite(), "velocity became non-finite over 10s");
    assert!(
        drift < 1e-4,
        "KE drift {drift} exceeds 1e-4 over 10s (v0={v0}, v={v})"
    );
}

/// Same run but with damping: KE must decrease monotonically each second.
#[test]
fn dynamics_energy_decay_with_damping_10s() {
    let def = single_revolute(0.5, 0.1);
    let mut state = RobotState::new(&def);
    state.joint_velocities[0] = 2.0;
    state.actuator_commands = vec![ActuatorCommand::Torque(0.0)];

    let dt: f32 = 0.001;
    let mut prev_ke = 0.5 * def.links[1].inertia * state.joint_velocities[0].powi(2);

    for second in 1..=10 {
        for _ in 0..1000 {
            step_dynamics(&def, &mut state, dt);
        }
        let v = state.joint_velocities[0];
        assert!(v.is_finite(), "velocity non-finite at t={second}s");
        let ke = 0.5 * def.links[1].inertia * v * v;
        assert!(
            ke <= prev_ke + 1e-9,
            "KE grew under damping at t={second}s: prev={prev_ke}, now={ke}"
        );
        prev_ke = ke;
    }
    assert!(
        prev_ke < 1e-3,
        "KE failed to decay close to zero: {prev_ke}"
    );
}

/// Extreme dt + large torque must not produce NaN/Inf joint state.
#[test]
fn dynamics_extreme_dt_no_nan() {
    let def = single_revolute(0.1, 0.001);
    let mut state = RobotState::new(&def);
    state.actuator_commands = vec![ActuatorCommand::Torque(1000.0)];

    for dt in [1.0_f32, 10.0, 100.0] {
        let mut s = state.clone();
        for _ in 0..50 {
            step_dynamics(&def, &mut s, dt);
        }
        assert!(
            s.joint_positions[0].is_finite() && s.joint_velocities[0].is_finite(),
            "Joint state non-finite at dt={dt}: pos={}, vel={}",
            s.joint_positions[0],
            s.joint_velocities[0]
        );
        assert!(
            s.joint_positions[0] >= def.joints[0].limit_min
                && s.joint_positions[0] <= def.joints[0].limit_max,
            "Joint position escaped limits at dt={dt}: {}",
            s.joint_positions[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Near-singular contact geometry
// ---------------------------------------------------------------------------

/// Two AABBs at identical centers and identical extents (max overlap,
/// degenerate normal). Overlap test returns true, robot collision routine
/// must not panic and must produce finite penetration.
#[test]
fn aabb_deep_penetration_finite_normal() {
    let a = Aabb {
        center: Vec3::new(1.0, 2.0, 3.0),
        half_extents: Vec3::splat(0.5),
    };
    let b = Aabb {
        center: a.center,
        half_extents: a.half_extents,
    };
    assert!(aabb_overlap(&a, &b));

    let def = single_revolute(0.0, 0.1);
    let mut state_a = RobotState::new(&def);
    state_a.link_poses[1] = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)).to_cols_array();
    let mut state_b = RobotState::new(&def);
    state_b.link_poses[1] = glam::Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)).to_cols_array();

    let robots = vec![(0usize, &def, &state_a), (1usize, &def, &state_b)];
    let collisions = detect_robot_collisions(&robots);
    assert!(
        !collisions.is_empty(),
        "Coincident robots should report a collision"
    );
    for c in &collisions {
        assert!(c.penetration.is_finite(), "non-finite penetration");
        assert!(c.contact_normal.is_finite(), "non-finite contact normal");
        assert!(
            (c.contact_normal.length() - 1.0).abs() < 1e-3,
            "contact normal not unit length: {}",
            c.contact_normal.length()
        );
    }
}

/// Zero-velocity links must not generate any punch hits (avoids division by
/// zero in damage scaling and prevents phantom hits at rest).
#[test]
fn punch_detection_zero_velocity_no_hits() {
    let def = single_revolute(0.0, 0.1);
    let mut state_a = RobotState::new(&def);
    state_a.link_poses[1] = glam::Mat4::IDENTITY.to_cols_array();
    let mut state_b = RobotState::new(&def);
    state_b.link_poses[1] = glam::Mat4::IDENTITY.to_cols_array();

    let robots = vec![(0usize, &def, &state_a), (1usize, &def, &state_b)];
    let collisions = detect_robot_collisions(&robots);

    let velocities_a = vec![Vec3::ZERO; def.links.len()];
    let velocities_b = vec![Vec3::ZERO; def.links.len()];
    let robots_with_vel = vec![
        (0usize, &def, &state_a, velocities_a.as_slice()),
        (1usize, &def, &state_b, velocities_b.as_slice()),
    ];
    let hits = detect_punches(&collisions, &robots_with_vel);
    assert!(
        hits.is_empty(),
        "Zero-velocity collision produced {} phantom hits",
        hits.len()
    );
}

/// Punch detection with NaN velocity vector must not panic and must not emit
/// a hit (NaN comparisons always return false → guard at threshold check).
#[test]
fn punch_detection_nan_velocity_no_hits() {
    let def = single_revolute(0.0, 0.1);
    let state_a = RobotState::new(&def);
    let state_b = RobotState::new(&def);

    let robots = vec![(0usize, &def, &state_a), (1usize, &def, &state_b)];
    let collisions = detect_robot_collisions(&robots);

    let bad = vec![Vec3::splat(f32::NAN); def.links.len()];
    let robots_with_vel = vec![
        (0usize, &def, &state_a, bad.as_slice()),
        (1usize, &def, &state_b, bad.as_slice()),
    ];
    let hits = detect_punches(&collisions, &robots_with_vel);
    assert!(
        hits.is_empty(),
        "NaN velocity should not emit punch (threshold guard); got {} hits",
        hits.len()
    );
}

// ---------------------------------------------------------------------------
// Long-run NaN guard — solvers do not propagate spurious NaNs
// ---------------------------------------------------------------------------

/// 100 steps of fluid + gas + dynamics in sequence with non-trivial state
/// must never produce a non-finite value anywhere.
#[test]
fn integrated_long_run_no_nan() {
    let mut fluid = make_fluid_grid(6, 0.1);
    for u in fluid.u.iter_mut() {
        *u = 0.1;
    }
    let fluid_cfg = FluidConfig {
        dt: 0.01,
        viscosity: 0.01,
        density: 1000.0,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        surface_tension: 0.0,
        jacobi_iterations: 5,
    };

    let mut gas = make_gas_grid(6, 0.1);
    let gas_center = gas.idx(3, 3, 3);
    gas.concentrations[0][gas_center] = 50.0;
    let gas_cfg = GasConfig {
        dt: 0.01,
        ambient_temperature: 293.15,
        thermal_diffusivity: 2.2e-5,
        buoyancy_coefficient: 0.01,
        gravity: Vec3::new(0.0, -9.81, 0.0),
    };

    let def = single_revolute(0.1, 0.1);
    let mut state = RobotState::new(&def);
    state.joint_velocities[0] = 1.0;
    state.actuator_commands = vec![ActuatorCommand::Position(0.5)];

    for step in 0..100 {
        fluid_solver::step(&mut fluid, &fluid_cfg);
        gas_solver::step(&mut gas, &gas_cfg);
        step_dynamics(&def, &mut state, fluid_cfg.dt);

        assert!(
            !fluid_has_nonfinite(&fluid),
            "fluid non-finite at step {step}"
        );
        assert!(!gas_has_nonfinite(&gas), "gas non-finite at step {step}");
        assert!(
            state.joint_velocities.iter().all(|v| v.is_finite())
                && state.joint_positions.iter().all(|v| v.is_finite()),
            "joint state non-finite at step {step}"
        );
    }
}
