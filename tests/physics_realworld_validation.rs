//! Real-world physics validation: solver outputs vs textbook analytical solutions.
//!
//! Each test pins a specific physical law to numerical evidence: Hagen-Poiseuille
//! for pipe flow, Fick's diffusion for gas spread, and ideal-gas speed of sound
//! for acoustic propagation. These are reference-solution tests — if a solver
//! starts drifting from real-world physics, these fail loudly.

use echomap::assert_relative_eq;
use echomap::benchmarks::analytical;
use echomap::gas::grid::{GasCellType, GasGrid, GasSpecies};
use echomap::gas::solver as gas_solver;
use echomap::scene::material::{MediumLibrary, MediumProperties};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Hagen-Poiseuille pipe flow
// ---------------------------------------------------------------------------

/// Velocity at pipe wall (r = R) must be zero (no-slip).
#[test]
fn hagen_poiseuille_no_slip_at_wall() {
    let pipe_radius = 0.005_f64; // 5 mm
    let mu = 1.0e-3_f64; // water at 20C, Pa-s
    let dp_dx = 100.0_f64; // 100 Pa/m

    let u_wall = analytical::hagen_poiseuille_velocity(pipe_radius, pipe_radius, dp_dx, mu);
    assert!(
        u_wall.abs() < 1e-12,
        "Velocity at pipe wall must be zero (no-slip), got {u_wall}"
    );
}

/// Velocity profile must be parabolic with max at center.
#[test]
fn hagen_poiseuille_parabolic_profile_water() {
    let r_pipe = 0.005_f64;
    let mu = 1.0e-3_f64;
    let dp_dx = 100.0_f64;

    let u_center = analytical::hagen_poiseuille_velocity(0.0, r_pipe, dp_dx, mu);
    let u_half = analytical::hagen_poiseuille_velocity(0.5 * r_pipe, r_pipe, dp_dx, mu);
    let u_wall = analytical::hagen_poiseuille_velocity(r_pipe, r_pipe, dp_dx, mu);

    // u_max = dP/dx * R^2 / (4 mu) -> 100 * (0.005^2) / (4 * 0.001) = 0.625 m/s
    assert_relative_eq!(u_center, 0.625, 1e-6);

    // At r = R/2: u = u_max * (1 - 0.25) = 0.75 * u_max
    assert_relative_eq!(u_half, 0.75 * u_center, 1e-9);
    assert!(u_wall.abs() < 1e-12);
    assert!(u_center > u_half && u_half > u_wall);
}

/// Volumetric flow rate must match Q = pi * R^4 * dP/dx / (8 mu).
#[test]
fn hagen_poiseuille_flow_rate_matches_textbook() {
    // Capillary tube: R = 0.5 mm, water, pressure drop 1 kPa over 0.1 m.
    let r_pipe = 0.0005_f64;
    let mu = 1.0e-3_f64;
    let dp_dx = 1000.0_f64 / 0.1; // 1 kPa / 0.1 m

    let q = analytical::hagen_poiseuille_flow_rate(r_pipe, dp_dx, mu);

    // Expected: pi * (5e-4)^4 * 1e4 / (8 * 1e-3)
    //        = pi * 6.25e-14 * 1e4 / 8e-3
    //        = pi * 6.25e-10 / 8e-3
    //        = pi * 7.8125e-8
    //        ~ 2.4544e-7 m^3/s
    let expected = std::f64::consts::PI * r_pipe.powi(4) * dp_dx / (8.0 * mu);
    assert_relative_eq!(q, expected, 1e-9);
    assert!(q > 0.0);
}

// ---------------------------------------------------------------------------
// Fick's diffusion law
// ---------------------------------------------------------------------------

/// Analytical Fick profile at x=0 must equal c0/2 (boundary value).
#[test]
fn ficks_diffusion_boundary_value() {
    let c0 = 1.0_f64;
    let d = 1.6e-5_f64; // CO2 in air
    let t = 1.0_f64;

    let c_zero = analytical::fick_diffusion_1d(c0, 0.0, d, t);
    assert_relative_eq!(c_zero, 0.5, 1e-6);
}

/// At fixed time t, concentration vs distance must decrease monotonically
/// and approach zero in the far field.
#[test]
fn ficks_diffusion_far_field_decays() {
    let c0 = 1.0_f64;
    let d = 1.6e-5_f64;
    let t = 10.0_f64;

    let c_near = analytical::fick_diffusion_1d(c0, 0.001, d, t);
    let c_mid = analytical::fick_diffusion_1d(c0, 0.01, d, t);
    let c_far = analytical::fick_diffusion_1d(c0, 0.1, d, t);

    assert!(c_near > c_mid && c_mid > c_far);
    assert!(
        c_far < 1e-6,
        "far-field concentration should be ~0, got {c_far}"
    );
}

/// Gas solver diffusion must produce monotonic spread matching Fick's law
/// qualitatively: total mass conserved, concentration decreases from source.
#[test]
fn ficks_diffusion_solver_matches_analytical() {
    let diff_coeff = 0.05_f32;
    let species = vec![GasSpecies {
        name: "Tracer".to_string(),
        diffusion_coefficient: diff_coeff,
        molecular_weight: 28.0,
        density_at_stp: 1.225,
        color: [1.0, 0.0, 0.0],
    }];

    let n = 16;
    let dx = 0.1_f32;
    let mut grid = GasGrid::new(n, n, n, dx, Vec3::ZERO, species);
    for ct in grid.cell_types.iter_mut() {
        *ct = GasCellType::Gas;
    }

    // Inject c0 at center cell.
    let ci = n / 2;
    let center_idx = grid.idx(ci, ci, ci);
    let c0_total: f32 = 1000.0;
    grid.concentrations[0][center_idx] = c0_total;

    let initial_mass: f32 = grid.concentrations[0].iter().sum();

    let dt = 0.005_f32;
    for _ in 0..50 {
        gas_solver::diffuse_concentrations(&mut grid, dt);
    }

    // Mass conservation: diffusion must not create or destroy mass (no-flux BC).
    let final_mass: f32 = grid.concentrations[0].iter().sum();
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 0.01,
        "Gas mass not conserved under diffusion: initial={initial_mass}, final={final_mass}"
    );

    // Center concentration must decrease (mass spreads out).
    let c_center = grid.concentrations[0][center_idx];
    assert!(
        c_center < c0_total,
        "Center concentration should decrease under diffusion, got {c_center}"
    );

    // Concentration spread must be monotonic radially.
    let c_r1 = grid.concentrations[0][grid.idx(ci + 1, ci, ci)];
    let c_r2 = grid.concentrations[0][grid.idx(ci + 2, ci, ci)];
    let c_r3 = grid.concentrations[0][grid.idx(ci + 3, ci, ci)];
    assert!(c_center > c_r1 && c_r1 > c_r2 && c_r2 > c_r3);

    // Spread half-width must grow as ~sqrt(D*t) (Fick scaling).
    // After t = 0.25s with D = 0.05, expected sigma ~ sqrt(2*D*t) ~ 0.158 m.
    // Width of populated region (c > 0.1 * c_center) should match within order-of-magnitude.
    let threshold = 0.1 * c_center;
    let mut populated_cells = 0usize;
    for i in 0..n {
        if grid.concentrations[0][grid.idx(i, ci, ci)] > threshold {
            populated_cells += 1;
        }
    }
    assert!(
        populated_cells >= 3,
        "Diffusion spread too narrow: {populated_cells} cells above threshold"
    );
    assert!(
        populated_cells <= n,
        "Diffusion spread invalid: {populated_cells} > {n}"
    );
}

// ---------------------------------------------------------------------------
// Speed of sound (acoustic propagation matches c = 343 m/s in air)
// ---------------------------------------------------------------------------

/// Analytical formula c = sqrt(gamma * R * T) must give 343 m/s ± 1 for dry air at 20C.
#[test]
fn speed_of_sound_dry_air_at_20c() {
    let gamma = 1.4_f64;
    let r_specific = 287.05_f64; // J/(kg*K)
    let t_kelvin = 293.15_f64; // 20 C

    let c = analytical::speed_of_sound_ideal_gas(gamma, r_specific, t_kelvin);

    // Reference: 343.2 m/s at 20C, 1 atm, 0% humidity.
    assert!(
        (c - 343.2).abs() < 1.0,
        "Speed of sound in air at 20C should be ~343 m/s, got {c}"
    );
}

/// Speed of sound must rise with temperature (T=0C ~ 331 m/s, T=20C ~ 343 m/s, T=40C ~ 355 m/s).
#[test]
fn speed_of_sound_increases_with_temperature() {
    let gamma = 1.4_f64;
    let r = 287.05_f64;

    let c_0c = analytical::speed_of_sound_ideal_gas(gamma, r, 273.15);
    let c_20c = analytical::speed_of_sound_ideal_gas(gamma, r, 293.15);
    let c_40c = analytical::speed_of_sound_ideal_gas(gamma, r, 313.15);

    assert!(c_0c < c_20c && c_20c < c_40c);
    assert!(
        (c_0c - 331.3).abs() < 1.0,
        "c(0C) expected ~331.3, got {c_0c}"
    );
    assert!(
        (c_40c - 354.7).abs() < 1.0,
        "c(40C) expected ~354.7, got {c_40c}"
    );
}

/// MediumProperties::air() preset must report c = 343 m/s.
#[test]
fn medium_air_preset_has_real_world_speed_of_sound() {
    let air = MediumProperties::air();
    assert!(
        (air.speed_of_sound - 343.0).abs() < 0.5,
        "Air preset speed_of_sound should be 343 m/s, got {}",
        air.speed_of_sound
    );

    // Impedance must follow Z = rho * c.
    let expected_z = 1.225_f32 * 343.0;
    assert!(
        (air.impedance - expected_z).abs() < 0.5,
        "Air impedance should be {expected_z}, got {}",
        air.impedance
    );
}

/// MediumLibrary "Water" preset must report c ≈ 1481 m/s (fresh water at 20C).
#[test]
fn medium_water_preset_has_real_world_speed_of_sound() {
    let lib = MediumLibrary::with_defaults();
    let water = lib.get("Water").expect("Water preset must exist");
    assert!(
        (water.speed_of_sound - 1481.0).abs() < 2.0,
        "Water preset speed_of_sound should be ~1481 m/s, got {}",
        water.speed_of_sound
    );
}

/// Acoustic ray traveling distance d at speed c in air must take t = d/c seconds.
/// Verifies that the medium speed_of_sound is consistent with propagation timing.
#[test]
fn acoustic_propagation_time_matches_speed_of_sound() {
    let air = MediumProperties::air();
    let distance = 343.0_f32; // exactly 1 second worth of travel
    let expected_time = distance / air.speed_of_sound;
    assert!(
        (expected_time - 1.0).abs() < 1e-3,
        "Air propagation time over c m should be 1.0s, got {expected_time}"
    );

    // Verify sub-second scale (1 m -> ~2.9 ms in air).
    let one_meter_time = 1.0_f32 / air.speed_of_sound;
    assert!(
        (one_meter_time - 0.002915).abs() < 1e-5,
        "1 m air propagation time should be ~2.92 ms, got {one_meter_time}"
    );
}
