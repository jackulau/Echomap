//! Integrated performance budget: a representative end-to-end scene
//! (fluid + gas + robots + acoustics) must hit at least 60 sim steps/sec
//! in release mode. If this drops, a regression in the hot path has landed.

use std::time::Instant;

use echomap::robot::boxing::{BoxingMatchConfig, BoxingScenario};
use echomap::robot::definition::RobotDefinition;
use echomap::robot::RobotManager;
use echomap::scenarios::builders::{FluidRoomScenario, GasLeakScenario, ScenarioConfig};
use glam::Mat4;

const TARGET_FPS: f64 = 60.0;
const WARMUP_STEPS: usize = 10;
const MEASURED_STEPS: usize = 60;

/// Combined fluid + gas + robots end-to-end step throughput, asserted to
/// meet a 60 sim_fps budget in release. Debug builds skip the assert
/// because the cargo `cfg(debug_assertions)` path is 10-30x slower.
///
/// Marked `#[ignore]` because default `cargo test` runs the suite in
/// parallel and CPU contention reliably tanks throughput below budget,
/// producing false alarms. Run explicitly on a quiet machine via:
///     cargo test --release --test integrated_perf -- --ignored
#[test]
#[ignore]
fn integrated_perf_meets_60_steps_per_sec() {
    let config = ScenarioConfig::default();
    let mut fluid = FluidRoomScenario::build(&config);
    let mut gas = GasLeakScenario::build(&config);

    // Robot manager with two simple arms — exercises FK/dynamics each step.
    let mut robots = RobotManager::new();
    robots.add_robot(RobotDefinition::simple_arm(2), Mat4::IDENTITY);
    robots.add_robot(RobotDefinition::simple_arm(2), Mat4::IDENTITY);

    // Warmup: JIT-style first-run caching, ensure measurement is steady-state.
    for _ in 0..WARMUP_STEPS {
        fluid.simulation.step();
        gas.simulation.step();
        robots.step(0.016, &[]);
    }

    let start = Instant::now();
    for _ in 0..MEASURED_STEPS {
        fluid.simulation.step();
        gas.simulation.step();
        robots.step(0.016, &[]);
    }
    let elapsed = start.elapsed();
    let steps_per_sec = MEASURED_STEPS as f64 / elapsed.as_secs_f64();
    let sim_fps = steps_per_sec;

    eprintln!(
        "integrated perf: {} steps in {:?} -> {:.1} sim_fps (target {} sim_fps)",
        MEASURED_STEPS, elapsed, sim_fps, TARGET_FPS
    );

    // Only enforce the budget in release builds — debug is ~10x slower.
    if cfg!(debug_assertions) {
        eprintln!("debug build: skipping hard 60 sim_fps assertion");
        return;
    }
    assert!(
        sim_fps >= TARGET_FPS,
        "integrated sim {:.1} sim_fps below target {}",
        sim_fps,
        TARGET_FPS
    );
}

/// Stand-alone fluid throughput on the default grid — guards against
/// fluid-solver regression independently of gas/robot. Less strict (>=120
/// sim_fps) because fluid alone is faster than the combined budget.
///
/// `#[ignore]` for the same reason as `integrated_perf_meets_60_steps_per_sec`.
#[test]
#[ignore]
fn fluid_only_perf_release() {
    let config = ScenarioConfig::default();
    let mut fluid = FluidRoomScenario::build(&config);
    for _ in 0..WARMUP_STEPS {
        fluid.simulation.step();
    }
    let start = Instant::now();
    for _ in 0..MEASURED_STEPS {
        fluid.simulation.step();
    }
    let elapsed = start.elapsed();
    let sim_fps = MEASURED_STEPS as f64 / elapsed.as_secs_f64();
    eprintln!("fluid_only: {:.1} sim_fps", sim_fps);

    if cfg!(debug_assertions) {
        return;
    }
    assert!(
        sim_fps >= 120.0,
        "fluid-only {:.1} sim_fps below 120 — solver regressed",
        sim_fps
    );
}

/// Stand-alone gas throughput, post-rayon parallelization. Must clear
/// >=120 sim_fps as a regression guard for D2.
///
/// `#[ignore]` for the same reason as `integrated_perf_meets_60_steps_per_sec`.
#[test]
#[ignore]
fn gas_only_perf_release() {
    let config = ScenarioConfig::default();
    let mut gas = GasLeakScenario::build(&config);
    for _ in 0..WARMUP_STEPS {
        gas.simulation.step();
    }
    let start = Instant::now();
    for _ in 0..MEASURED_STEPS {
        gas.simulation.step();
    }
    let elapsed = start.elapsed();
    let sim_fps = MEASURED_STEPS as f64 / elapsed.as_secs_f64();
    eprintln!("gas_only: {:.1} sim_fps", sim_fps);

    if cfg!(debug_assertions) {
        return;
    }
    assert!(
        sim_fps >= 120.0,
        "gas-only {:.1} sim_fps below 120 — solver regressed",
        sim_fps
    );
}

/// 60 Hz frame budget on the actual boxing scenario (D6 of the
/// physics-quality-and-perf goal). Full integrated physics step — fluid grid,
/// gas grid, both boxing humanoids' dynamics + collision — must run in less
/// than 16.67 ms averaged over 1000 measured steps on a release build.
///
/// Marked `#[ignore]` so it does not block ordinary `cargo test`. Run via:
///     cargo test --release --test integrated_perf -- --ignored physics_step_budget
#[test]
#[ignore]
fn physics_step_budget() {
    let frame_budget_ms = 16.67_f64;
    let measured_steps = 1000usize;
    let warmup_steps = 50usize;

    // Build the actual fluid + gas scenarios used in the boxing-match server.
    let config = ScenarioConfig::default();
    let mut fluid = FluidRoomScenario::build(&config);
    let mut gas = GasLeakScenario::build(&config);

    // BoxingScenario owns the ring + boxing match; the returned RobotManager
    // already has both humanoids inserted with combat state enabled.
    let (_scenario, mut robots) = BoxingScenario::new(BoxingMatchConfig::default());

    for _ in 0..warmup_steps {
        fluid.simulation.step();
        gas.simulation.step();
        robots.step(0.016, &[]);
    }

    let start = Instant::now();
    for _ in 0..measured_steps {
        fluid.simulation.step();
        gas.simulation.step();
        robots.step(0.016, &[]);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / measured_steps as f64;
    eprintln!(
        "physics_step_budget: {} steps in {:?} -> avg {:.3} ms/step (budget {:.2} ms)",
        measured_steps, elapsed, avg_ms, frame_budget_ms
    );

    if cfg!(debug_assertions) {
        eprintln!("debug build: skipping hard {frame_budget_ms} ms assertion");
        return;
    }
    assert!(
        avg_ms < frame_budget_ms,
        "integrated physics step avg {avg_ms:.3} ms exceeds 60 Hz budget {frame_budget_ms:.2} ms"
    );
}
