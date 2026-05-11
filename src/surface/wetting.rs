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
}
