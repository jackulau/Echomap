//! Criterion microbenchmarks for the physics hot paths.
//!
//! Targets, in order: acoustic ray-triangle / ray-scene cast, fluid solver
//! step (full advect/diffuse/project), gas solver step (advect + diffuse),
//! rigid-body dynamics step, robot collision (broad + narrow phase).
//!
//! Each bench is sized to be representative of in-game workloads, not
//! microbenchmark micro-toys: e.g. fluid is a 16³ grid, gas is 16³ with
//! one species, dynamics is a 5-joint arm. Coefficients of variation are
//! expected <5% on a quiet machine.
//!
//! Run with: `cargo bench --bench physics`
//! Save baseline: `cargo bench --bench physics -- --save-baseline main`
//! Compare: `cargo bench --bench physics -- --baseline main`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use glam::{Mat4, Vec3};

use echomap::acoustics::ray::AcousticRay;
use echomap::fluids::grid::{CellType, FluidGrid};
use echomap::fluids::solver::{self as fluid_solver, FluidConfig};
use echomap::gas::grid::{GasCellType, GasGrid, GasSpecies};
use echomap::gas::solver::{self as gas_solver, GasConfig};
use echomap::robot::collision::{
    aabb_overlap, collect_link_aabbs, detect_robot_collisions, ray_triangle_intersect, Aabb,
};
use echomap::robot::definition::RobotDefinition;
use echomap::robot::dynamics::step_dynamics;
use echomap::robot::state::{ActuatorCommand, RobotState};
use echomap::scene::material::{MediumLibrary, MediumProperties};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const FLUID_N: usize = 16;
const GAS_N: usize = 16;

fn make_fluid_grid(n: usize) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, n, 0.125, Vec3::ZERO);
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let idx = g.idx(i, j, k);
                g.cell_types[idx] = CellType::Fluid;
                g.density[idx] = 1000.0;
                g.level_set[idx] = -1.0;
            }
        }
    }
    // Seed a circulating velocity field so advect actually does work.
    for u in g.u.iter_mut() {
        *u = 0.5;
    }
    for v in g.v.iter_mut() {
        *v = -0.25;
    }
    g
}

fn make_gas_grid(n: usize) -> GasGrid {
    let species = vec![GasSpecies {
        name: "CO2".to_string(),
        diffusion_coefficient: 0.16,
        molecular_weight: 44.0,
        density_at_stp: 1.842,
        color: [1.0, 0.0, 0.0],
    }];
    let mut g = GasGrid::new(n, n, n, 0.125, Vec3::ZERO, species);
    for k in 1..n - 1 {
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let idx = g.idx(i, j, k);
                g.cell_types[idx] = GasCellType::Gas;
                g.temperature[idx] = 293.15;
                g.concentrations[0][idx] = 0.5;
                g.vel_x[idx] = 0.2;
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_fluid_step(c: &mut Criterion) {
    let cfg = FluidConfig::default();
    let mut group = c.benchmark_group("fluid");
    group.throughput(Throughput::Elements((FLUID_N * FLUID_N * FLUID_N) as u64));
    group.bench_function("step_16cubed", |b| {
        b.iter_batched(
            || make_fluid_grid(FLUID_N),
            |mut grid| {
                fluid_solver::step(&mut grid, black_box(&cfg));
                black_box(grid)
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn bench_gas_step(c: &mut Criterion) {
    let cfg = GasConfig::default();
    let mut group = c.benchmark_group("gas");
    group.throughput(Throughput::Elements((GAS_N * GAS_N * GAS_N) as u64));
    group.bench_function("step_16cubed", |b| {
        b.iter_batched(
            || make_gas_grid(GAS_N),
            |mut grid| {
                gas_solver::step(&mut grid, black_box(&cfg));
                black_box(grid)
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("diffuse_concentrations_16cubed", |b| {
        b.iter_batched(
            || make_gas_grid(GAS_N),
            |mut grid| {
                gas_solver::diffuse_concentrations(&mut grid, 0.016);
                black_box(grid)
            },
            criterion::BatchSize::LargeInput,
        )
    });
    group.finish();
}

fn bench_dynamics_step(c: &mut Criterion) {
    let def = RobotDefinition::simple_arm(5);
    let mut group = c.benchmark_group("dynamics");
    group.throughput(Throughput::Elements(def.joints.len() as u64));
    group.bench_function("step_5dof", |b| {
        b.iter_batched(
            || {
                let mut s = RobotState::new(&def);
                s.actuator_commands = def
                    .joints
                    .iter()
                    .map(|_| ActuatorCommand::Position(0.5))
                    .collect();
                s
            },
            |mut state| {
                step_dynamics(black_box(&def), &mut state, 0.016);
                black_box(state)
            },
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_collision(c: &mut Criterion) {
    let def_a = RobotDefinition::boxing_test_robot();
    let def_b = RobotDefinition::boxing_test_robot();
    let mut state_a = RobotState::new(&def_a);
    let mut state_b = RobotState::new(&def_b);
    // Position robot B slightly offset so AABBs overlap.
    state_a.link_poses[0] = Mat4::from_translation(Vec3::ZERO).to_cols_array();
    state_b.link_poses[0] = Mat4::from_translation(Vec3::new(0.1, 0.0, 0.0)).to_cols_array();

    let mut group = c.benchmark_group("collision");

    // Broad phase: AABB collection for one robot.
    group.bench_function("collect_link_aabbs_3links", |b| {
        b.iter(|| {
            let aabbs = collect_link_aabbs(black_box(&def_a), black_box(&state_a));
            black_box(aabbs)
        })
    });

    // Pair-wise overlap test.
    let aabb1 = Aabb {
        center: Vec3::ZERO,
        half_extents: Vec3::splat(0.5),
    };
    let aabb2 = Aabb {
        center: Vec3::new(0.3, 0.0, 0.0),
        half_extents: Vec3::splat(0.5),
    };
    group.bench_function("aabb_overlap", |b| {
        b.iter(|| aabb_overlap(black_box(&aabb1), black_box(&aabb2)))
    });

    // Full robot-vs-robot detection (broad + narrow phase).
    group.bench_function("detect_robot_collisions_2bots", |b| {
        b.iter(|| {
            let robots = vec![(0usize, &def_a, &state_a), (1usize, &def_b, &state_b)];
            let cols = detect_robot_collisions(black_box(&robots));
            black_box(cols)
        })
    });

    group.finish();
}

fn bench_acoustic_ray(c: &mut Criterion) {
    let mut group = c.benchmark_group("acoustics");

    let v0 = Vec3::new(-1.0, 0.0, -1.0);
    let v1 = Vec3::new(1.0, 0.0, -1.0);
    let v2 = Vec3::new(0.0, 0.0, 1.0);
    let origin = Vec3::new(0.0, 1.0, 0.0);
    let dir = Vec3::new(0.0, -1.0, 0.0);

    group.bench_function("ray_triangle_intersect", |b| {
        b.iter(|| {
            ray_triangle_intersect(
                black_box(origin),
                black_box(dir),
                black_box(v0),
                black_box(v1),
                black_box(v2),
            )
        })
    });

    let air = MediumProperties::air();
    let media = MediumLibrary::with_defaults();
    let water = media.get("Water").expect("Water medium present").clone();
    group.bench_function("ray_refract_air_water", |b| {
        b.iter(|| {
            let ray = AcousticRay::new(
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                1.0,
                air.clone(),
            );
            black_box(ray.refract(Vec3::Y, &water))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_fluid_step,
    bench_gas_step,
    bench_dynamics_step,
    bench_collision,
    bench_acoustic_ray
);
criterion_main!(benches);
