/// Result of wetting and capillary computation.
pub struct WettingResult {
    pub surface_energy: f32,
    pub capillary_pressure: f32,
    pub is_hydrophilic: bool,
}

/// Compute wetting properties using Young's equation and Young-Laplace capillary pressure.
///
/// - surface_energy = surface_tension * cos(contact_angle) (Young's equation)
/// - capillary_pressure = 2 * surface_tension * cos(contact_angle) / pore_radius (Young-Laplace)
/// - is_hydrophilic: contact_angle < π/2
///
/// Pore radius is clamped to a minimum of 1e-10 to avoid division by zero.
pub fn compute_wetting(
    contact_angle: f32,
    surface_tension: f32,
    pore_radius: f32,
) -> WettingResult {
    let surface_tension = surface_tension.max(0.0);
    let cos_angle = contact_angle.cos();
    let surface_energy = surface_tension * cos_angle;

    // Clamp pore_radius to avoid division by zero
    let r = pore_radius.max(1e-10);
    let capillary_pressure = 2.0 * surface_tension * cos_angle / r;

    let is_hydrophilic = contact_angle < std::f32::consts::FRAC_PI_2;

    WettingResult {
        surface_energy,
        capillary_pressure,
        is_hydrophilic,
    }
}

/// Compute the spreading coefficient.
///
/// S = surface_tension * (cos(contact_angle) - 1)
/// Positive S means spontaneous spreading (complete wetting).
/// At contact_angle = 0, cos(0) = 1, so S = 0 (boundary of complete wetting).
pub fn spreading_coefficient(contact_angle: f32, surface_tension: f32) -> f32 {
    surface_tension * (contact_angle.cos() - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_hydrophilic_surface() {
        // contact_angle < π/2 → positive surface energy, is_hydrophilic = true
        let contact_angle = PI / 4.0; // 45 degrees
        let surface_tension = 0.072; // water ~72 mN/m
        let pore_radius = 1e-3;
        let result = compute_wetting(contact_angle, surface_tension, pore_radius);
        assert!(
            result.surface_energy > 0.0,
            "Hydrophilic surface should have positive surface energy, got {}",
            result.surface_energy
        );
        assert!(
            result.is_hydrophilic,
            "Contact angle < π/2 should be hydrophilic"
        );
    }

    #[test]
    fn test_hydrophobic_surface() {
        // contact_angle > π/2 → negative surface energy, is_hydrophilic = false
        let contact_angle = 2.0 * PI / 3.0; // 120 degrees
        let surface_tension = 0.072;
        let pore_radius = 1e-3;
        let result = compute_wetting(contact_angle, surface_tension, pore_radius);
        assert!(
            result.surface_energy < 0.0,
            "Hydrophobic surface should have negative surface energy, got {}",
            result.surface_energy
        );
        assert!(
            !result.is_hydrophilic,
            "Contact angle > π/2 should not be hydrophilic"
        );
    }

    #[test]
    fn test_capillary_pressure_positive_hydrophilic() {
        let contact_angle = PI / 4.0; // hydrophilic
        let surface_tension = 0.072;
        let pore_radius = 1e-3;
        let result = compute_wetting(contact_angle, surface_tension, pore_radius);
        assert!(
            result.capillary_pressure > 0.0,
            "Hydrophilic material should have positive capillary pressure, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_capillary_pressure_negative_hydrophobic() {
        let contact_angle = 2.0 * PI / 3.0; // hydrophobic
        let surface_tension = 0.072;
        let pore_radius = 1e-3;
        let result = compute_wetting(contact_angle, surface_tension, pore_radius);
        assert!(
            result.capillary_pressure < 0.0,
            "Hydrophobic material should have negative capillary pressure, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_zero_pore_radius() {
        let contact_angle = PI / 4.0;
        let surface_tension = 0.072;
        let pore_radius = 0.0;
        let result = compute_wetting(contact_angle, surface_tension, pore_radius);
        assert!(
            result.capillary_pressure.is_finite(),
            "Zero pore radius should produce finite capillary pressure (clamped), got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_spreading_coefficient_complete_wetting() {
        // contact_angle = 0 → cos(0) = 1 → S = surface_tension * (1 - 1) = 0
        let s = spreading_coefficient(0.0, 0.072);
        assert!(
            s.abs() < EPSILON,
            "Complete wetting (contact_angle=0) should give S=0, got {}",
            s
        );
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_contact_angle_exactly_pi_over_2() {
        let result = compute_wetting(std::f32::consts::FRAC_PI_2, 0.072, 1e-3);
        // cos(pi/2) in f32 is not exactly 0 (approx -4.37e-8), so allow wider tolerance
        assert!(
            result.surface_energy.abs() < 1e-4,
            "At pi/2, surface_energy should be ~0, got {}",
            result.surface_energy
        );
        assert!(
            result.capillary_pressure.abs() < 1e-2,
            "At pi/2, capillary_pressure should be near zero, got {}",
            result.capillary_pressure
        );
        assert!(
            !result.is_hydrophilic,
            "Exactly pi/2 should NOT be hydrophilic (strict <)"
        );
    }

    #[test]
    fn test_edge_contact_angle_zero_complete_wetting() {
        let result = compute_wetting(0.0, 0.072, 1e-3);
        assert!(
            (result.surface_energy - 0.072).abs() < EPSILON,
            "At angle=0, surface_energy should be 0.072, got {}",
            result.surface_energy
        );
        assert!(result.is_hydrophilic, "angle=0 should be hydrophilic");
        assert!(
            (result.capillary_pressure - 144.0).abs() < 0.1,
            "capillary_pressure at angle=0: expected 144, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_edge_contact_angle_pi_complete_hydrophobic() {
        let result = compute_wetting(PI, 0.072, 1e-3);
        assert!(
            (result.surface_energy - (-0.072)).abs() < EPSILON,
            "At angle=pi, surface_energy should be -0.072, got {}",
            result.surface_energy
        );
        assert!(!result.is_hydrophilic, "angle=pi should be hydrophobic");
        assert!(
            result.capillary_pressure < 0.0,
            "At angle=pi, capillary_pressure should be negative"
        );
    }

    #[test]
    fn test_edge_surface_tension_zero() {
        let result = compute_wetting(PI / 4.0, 0.0, 1e-3);
        assert!(
            result.surface_energy.abs() < EPSILON,
            "Zero surface tension should yield zero surface_energy, got {}",
            result.surface_energy
        );
        assert!(
            result.capillary_pressure.abs() < EPSILON,
            "Zero surface tension should yield zero capillary_pressure, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_edge_pore_radius_negative() {
        let result = compute_wetting(PI / 4.0, 0.072, -1e-3);
        assert!(
            result.capillary_pressure.is_finite(),
            "Negative pore_radius should produce finite result, got {}",
            result.capillary_pressure
        );
        assert!(
            result.capillary_pressure > 1e8,
            "Negative pore_radius clamped to 1e-10 should produce very large pressure"
        );
    }

    #[test]
    fn test_edge_pore_radius_very_small() {
        let result = compute_wetting(PI / 4.0, 0.072, 1e-9);
        assert!(
            result.capillary_pressure.is_finite(),
            "Very small pore radius should produce finite result"
        );
        assert!(
            result.capillary_pressure > 1e5,
            "Very small pore radius should produce very large capillary pressure"
        );
    }

    #[test]
    fn test_edge_pore_radius_very_large() {
        let result = compute_wetting(PI / 4.0, 0.072, 1e6);
        assert!(
            result.capillary_pressure.abs() < 1e-6,
            "Very large pore radius should produce near-zero capillary pressure, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_edge_contact_angle_nan() {
        let result = compute_wetting(f32::NAN, 0.072, 1e-3);
        assert!(
            result.surface_energy.is_nan(),
            "NaN contact angle should produce NaN surface_energy"
        );
        assert!(
            result.capillary_pressure.is_nan(),
            "NaN contact angle should produce NaN capillary_pressure"
        );
    }

    #[test]
    fn test_edge_spreading_coefficient_at_pi() {
        let s = spreading_coefficient(PI, 0.072);
        let expected = -2.0 * 0.072;
        assert!(
            (s - expected).abs() < EPSILON,
            "At pi, spreading coeff should be {}, got {}",
            expected,
            s
        );
    }

    #[test]
    fn test_edge_spreading_coefficient_zero_tension() {
        let s = spreading_coefficient(PI / 4.0, 0.0);
        assert!(s.abs() < EPSILON, "Zero surface tension S=0, got {}", s);
    }

    #[test]
    fn test_edge_spreading_coefficient_always_non_positive() {
        for angle in [0.0_f32, 0.1, 0.5, 1.0, PI / 2.0, PI] {
            let s = spreading_coefficient(angle, 0.072);
            assert!(
                s <= EPSILON,
                "Spreading coefficient should be <= 0 for angle={}, got {}",
                angle,
                s
            );
        }
    }

    #[test]
    fn test_edge_wetting_symmetry_hydrophilic_hydrophobic() {
        let angle_phil = PI / 4.0;
        let angle_phob = 3.0 * PI / 4.0;
        let r_phil = compute_wetting(angle_phil, 0.072, 1e-3);
        let r_phob = compute_wetting(angle_phob, 0.072, 1e-3);

        assert!(r_phil.is_hydrophilic);
        assert!(!r_phob.is_hydrophilic);
        assert!(
            (r_phil.surface_energy + r_phob.surface_energy).abs() < EPSILON,
            "Symmetric angles should have opposite surface energies: {} vs {}",
            r_phil.surface_energy,
            r_phob.surface_energy
        );
    }

    #[test]
    fn test_edge_negative_surface_tension_clamped() {
        // surface_tension is clamped to max(0.0), so -0.072 becomes 0.0
        let result = compute_wetting(PI / 4.0, -0.072, 1e-3);
        assert!(
            result.surface_energy.is_finite(),
            "Negative surface tension should still produce finite result"
        );
        assert!(
            result.surface_energy.abs() < EPSILON,
            "Negative surface tension clamped to 0 should yield zero energy, got {}",
            result.surface_energy
        );
        assert!(
            result.capillary_pressure.abs() < EPSILON,
            "Negative surface tension clamped to 0 should yield zero capillary pressure, got {}",
            result.capillary_pressure
        );
    }

    #[test]
    fn test_edge_contact_angle_beyond_pi() {
        // contact_angle > pi: physically nonsensical but should not panic
        let result = compute_wetting(2.0 * PI, 0.072, 1e-3);
        assert!(
            result.surface_energy.is_finite(),
            "contact_angle=2pi should produce finite surface energy"
        );
        assert!(
            result.capillary_pressure.is_finite(),
            "contact_angle=2pi should produce finite capillary pressure"
        );
    }

    #[test]
    fn test_edge_negative_contact_angle() {
        // cos(-theta) = cos(theta), so surface_energy should match
        let pos = compute_wetting(PI / 4.0, 0.072, 1e-3);
        let neg = compute_wetting(-PI / 4.0, 0.072, 1e-3);
        assert!(
            (pos.surface_energy - neg.surface_energy).abs() < EPSILON,
            "Negative angle should give same surface energy as positive: {} vs {}",
            pos.surface_energy,
            neg.surface_energy
        );
        // But hydrophilicity check: -PI/4 < FRAC_PI_2 is true
        assert!(
            neg.is_hydrophilic,
            "Negative contact angle < pi/2 should be hydrophilic"
        );
    }

    #[test]
    fn test_edge_infinity_surface_tension() {
        let result = compute_wetting(PI / 4.0, f32::INFINITY, 1e-3);
        assert!(
            result.surface_energy.is_infinite(),
            "Infinite surface tension should yield infinite surface energy"
        );
        assert!(
            result.capillary_pressure.is_infinite(),
            "Infinite surface tension should yield infinite capillary pressure"
        );
    }

    #[test]
    fn test_edge_nan_pore_radius() {
        let result = compute_wetting(PI / 4.0, 0.072, f32::NAN);
        // NaN.max(1e-10) = 1e-10 in Rust => division by 1e-10 => finite
        assert!(
            result.capillary_pressure.is_finite() || result.capillary_pressure.is_nan(),
            "NaN pore radius should not panic"
        );
        // surface_energy does not depend on pore_radius
        assert!(
            result.surface_energy.is_finite(),
            "NaN pore radius should not affect surface_energy"
        );
    }

    #[test]
    fn test_edge_nan_surface_tension() {
        let result = compute_wetting(PI / 4.0, f32::NAN, 1e-3);
        // NaN.max(0.0) = 0.0 in Rust; 0.0 * cos(pi/4) = 0.0
        assert!(
            result.surface_energy.abs() < EPSILON || result.surface_energy.is_nan(),
            "NaN surface tension should produce zero or NaN energy"
        );
    }

    #[test]
    fn test_edge_capillary_pressure_inversely_proportional() {
        // Halving pore radius should double capillary pressure
        let r1 = compute_wetting(PI / 4.0, 0.072, 1e-3);
        let r2 = compute_wetting(PI / 4.0, 0.072, 0.5e-3);
        let ratio = r2.capillary_pressure / r1.capillary_pressure;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Halving pore radius should double capillary pressure, ratio={ratio}"
        );
    }

    #[test]
    fn test_edge_spreading_coefficient_nan() {
        let s = spreading_coefficient(f32::NAN, 0.072);
        assert!(
            s.is_nan(),
            "NaN contact angle should produce NaN spreading coefficient"
        );
    }

    #[test]
    fn test_edge_spreading_coefficient_negative_tension() {
        // spreading_coefficient does NOT clamp surface_tension (unlike compute_wetting)
        // so negative tension * (cos - 1) could be positive
        let s = spreading_coefficient(PI / 4.0, -0.072);
        // cos(pi/4) - 1 ~ -0.293, * -0.072 ~ +0.021
        assert!(
            s > 0.0,
            "Negative tension with positive cos-1 should give positive S, got {s}"
        );
    }
}
