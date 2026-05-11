use crate::benchmarks::analytical;
use crate::scene::material::{AcousticMaterial, MaterialLibrary, MediumLibrary, MediumProperties};
use crate::surface::SurfaceInteraction;
use glam::Vec3;

use crate::acoustics::ray::AcousticRay;

/// Result of a single benchmark comparing simulation output to an analytical
/// solution.
pub struct BenchmarkResult {
    pub name: String,
    pub expected: f64,
    pub actual: f64,
    pub relative_error: f64,
    pub tolerance: f64,
    pub passed: bool,
}

impl BenchmarkResult {
    pub fn new(name: &str, expected: f64, actual: f64, tolerance: f64) -> Self {
        let relative_error = if expected.abs() < 1e-15 {
            actual.abs()
        } else {
            ((actual - expected) / expected).abs()
        };
        Self {
            name: name.to_string(),
            expected,
            actual,
            relative_error,
            tolerance,
            passed: relative_error <= tolerance,
        }
    }
}

/// Run all benchmarks and return their results.
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    vec![
        benchmark_fresnel_reflection(),
        benchmark_coulomb_friction(),
        benchmark_beckmann_scattering(),
        benchmark_stokes_drag(),
        benchmark_darcy_flow(),
    ]
}

/// Benchmark Fresnel reflection at normal incidence for an air-to-water
/// boundary. Uses `AcousticRay::refract()` and compares reflected energy
/// against `analytical::fresnel_reflection(z_air, z_water)`.
pub fn benchmark_fresnel_reflection() -> BenchmarkResult {
    let air = MediumProperties::air();
    let medium_lib = MediumLibrary::with_defaults();
    let water = medium_lib.get("Water").unwrap();

    let z_air = air.impedance as f64;
    let z_water = water.impedance as f64;

    // Analytical prediction
    let expected = analytical::fresnel_reflection(z_air, z_water);

    // Simulation: ray at normal incidence (straight down into water)
    let ray = AcousticRay::new(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        1.0,
        air,
    );
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let result = ray.refract(normal, water).unwrap();
    let actual = result.reflected_energy as f64;

    BenchmarkResult::new("fresnel_reflection", expected, actual, 0.01)
}

/// Benchmark Coulomb kinetic friction. Gets "Concrete" from MaterialLibrary,
/// creates a SurfaceInteraction, and compares the friction force magnitude
/// against `analytical::coulomb_kinetic_friction(mu_k, normal_force)`.
pub fn benchmark_coulomb_friction() -> BenchmarkResult {
    let lib = MaterialLibrary::with_defaults();
    let concrete = lib.materials.get("Concrete").unwrap();
    let si = SurfaceInteraction::from_material(concrete);

    let normal_force = 100.0_f32;
    let mu_k = si.friction_kinetic as f64;

    let expected = analytical::coulomb_kinetic_friction(mu_k, normal_force as f64);

    let force = si.friction(normal_force, Vec3::X);
    let actual = force.length() as f64;

    BenchmarkResult::new("coulomb_friction", expected, actual, 0.01)
}

/// Benchmark Beckmann scattering. Gets roughness from the default material,
/// computes scattering at 1000 Hz / 343 m/s, and compares the specular
/// fraction against `analytical::beckmann_specular_fraction(roughness, wavelength)`.
pub fn benchmark_beckmann_scattering() -> BenchmarkResult {
    let mat = AcousticMaterial::default();
    let si = SurfaceInteraction::from_material(&mat);

    let frequency = 1000.0_f32;
    let speed_of_sound = 343.0_f32;
    let wavelength = (speed_of_sound / frequency) as f64;
    let roughness = si.roughness as f64;

    let expected = analytical::beckmann_specular_fraction(roughness, wavelength);

    let scat = si.scattering_at_frequency(frequency, speed_of_sound);
    let actual = scat.specular_weight as f64;

    BenchmarkResult::new("beckmann_scattering", expected, actual, 0.10)
}

/// Benchmark Stokes drag. Pure analytical comparison:
/// `stokes_drag(0.001, 0.001, 0.01)` should equal `6 * pi * 0.001 * 0.001 * 0.01`.
pub fn benchmark_stokes_drag() -> BenchmarkResult {
    let viscosity = 0.001;
    let radius = 0.001;
    let velocity = 0.01;

    let actual = analytical::stokes_drag(viscosity, radius, velocity);
    let expected = 6.0 * std::f64::consts::PI * viscosity * radius * velocity;

    BenchmarkResult::new("stokes_drag", expected, actual, 0.0001)
}

/// Benchmark Darcy flow rate. Pure analytical comparison using
/// `darcy_flow_rate(1e-15, 1.0, 1.8e-5, 100.0, 0.1)`.
pub fn benchmark_darcy_flow() -> BenchmarkResult {
    let permeability = 1e-15;
    let area = 1.0;
    let viscosity = 1.8e-5;
    let pressure_drop = 100.0;
    let length = 0.1;

    let actual = analytical::darcy_flow_rate(permeability, area, viscosity, pressure_drop, length);
    let expected = (permeability * area / viscosity) * (pressure_drop / length);

    BenchmarkResult::new("darcy_flow", expected, actual, 0.0001)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_fresnel_accuracy() {
        let result = benchmark_fresnel_reflection();
        assert!(
            result.passed,
            "Fresnel benchmark failed: error={:.6} > tol={:.6} (expected={}, actual={})",
            result.relative_error, result.tolerance, result.expected, result.actual
        );
    }

    #[test]
    fn test_benchmark_friction_accuracy() {
        let result = benchmark_coulomb_friction();
        assert!(
            result.passed,
            "Coulomb friction benchmark failed: error={:.6} > tol={:.6} (expected={}, actual={})",
            result.relative_error, result.tolerance, result.expected, result.actual
        );
    }

    #[test]
    fn test_benchmark_scattering_accuracy() {
        let result = benchmark_beckmann_scattering();
        assert!(
            result.passed,
            "Beckmann scattering benchmark failed: error={:.6} > tol={:.6} (expected={}, actual={})",
            result.relative_error, result.tolerance, result.expected, result.actual
        );
    }

    #[test]
    fn test_benchmark_stokes_drag() {
        let result = benchmark_stokes_drag();
        assert!(
            result.passed,
            "Stokes drag benchmark failed: error={:.6} > tol={:.6} (expected={}, actual={})",
            result.relative_error, result.tolerance, result.expected, result.actual
        );
    }

    #[test]
    fn test_benchmark_all_pass() {
        let results = run_all_benchmarks();
        assert!(!results.is_empty(), "Should have benchmarks to run");
        for r in &results {
            assert!(
                r.passed,
                "Benchmark '{}' failed: error={:.6} > tol={:.6} (expected={}, actual={})",
                r.name, r.relative_error, r.tolerance, r.expected, r.actual
            );
        }
    }
}
