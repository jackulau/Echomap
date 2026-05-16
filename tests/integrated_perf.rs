//! Integrated performance budget: a representative end-to-end scene
//! (fluid + gas + robots + acoustics) must hit at least 60 sim steps/sec
//! in release mode. If this drops, a regression in the hot path has landed.

use std::time::Instant;

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
#[test]
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
#[test]
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
#[test]
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
