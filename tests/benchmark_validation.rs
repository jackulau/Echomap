use echomap::acoustics::ray::AcousticRay;
use echomap::assert_relative_eq;
use echomap::benchmarks::analytical;
use echomap::scenarios::builders::*;
use echomap::scene::material::MediumProperties;
use glam::Vec3;

#[test]
fn test_echomap_public_api_accessible() {
    // Verify that the public API is importable and functional.
    let config = ScenarioConfig::default();
    let scenario = FluidRoomScenario::build(&config);

    // The scenario should produce a valid scene with walls.
    assert!(
        !scenario.scene.meshes.is_empty(),
        "FluidRoomScenario should have scene meshes"
    );

    // Grid should be initialized.
    assert!(
        scenario.simulation.grid.is_some(),
        "FluidRoomScenario should have an initialized grid"
    );
}

#[test]
fn test_full_scenario_underwater() {
    // Build an underwater acoustics scenario.
    let config = ScenarioConfig::default();
    let scenario = UnderwaterAcousticsScenario::build(&config);

    // The background medium should be water.
    assert_eq!(
        scenario.scene.background_medium.name, "Water",
        "Background medium should be Water"
    );

    // Create an acoustic ray in air, refract against the water medium.
    let air = MediumProperties::air();
    let ray = AcousticRay::new(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        1.0,
        air.clone(),
    );

    let normal = Vec3::new(0.0, 1.0, 0.0);
    let result = ray
        .refract(normal, &scenario.scene.background_medium)
        .unwrap();

    // Compare Fresnel reflection to analytical prediction.
    let z_air = air.impedance as f64;
    let z_water = scenario.scene.background_medium.impedance as f64;
    let expected_r = analytical::fresnel_reflection(z_air, z_water);

    assert_relative_eq!(result.reflected_energy[0] as f64, expected_r, 0.01);

    // Energy should be conserved: R + T = 1.0 (per band — bands are uniform
    // at construction, so band 0 is representative)
    let total = result.reflected_energy[0] + result.transmitted_energy[0];
    assert!(
        (total - 1.0).abs() < 1e-4,
        "Energy should be conserved: R + T = {total}"
    );
}

#[test]
fn test_full_scenario_gas_leak() {
    // Build a gas leak scenario with a CO2 source.
    let config = ScenarioConfig::default();
    let mut scenario = GasLeakScenario::build(&config);

    // Step the simulation a few times to let gas diffuse.
    for _ in 0..5 {
        scenario.simulation.step();
    }

    // After stepping, the gas concentration at the source position should
    // be positive (gas has been injected).
    let grid = scenario.simulation.grid.as_ref().unwrap();
    let concentration = grid.concentration_at(0, scenario.source_position);

    assert!(
        concentration > 0.0,
        "Gas concentration at source should be positive after stepping, got {concentration}"
    );
}
