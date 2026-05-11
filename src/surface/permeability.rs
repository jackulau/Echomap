/// Result of a Darcy permeation computation.
pub struct PermeationResult {
    pub flux: f32,
    pub effective_permeability: f32,
}

/// Compute gas permeation through a porous solid boundary using Darcy's law.
///
/// - flux = permeability * porosity * concentration_gradient / dx
/// - effective_permeability = permeability * porosity
///
/// Zero permeability or zero porosity yields zero flux.
/// dx is clamped to a minimum of 1e-10 to avoid division by zero.
pub fn compute_permeation(
    permeability: f32,
    concentration_gradient: f32,
    porosity: f32,
    dx: f32,
) -> PermeationResult {
    let permeability = permeability.max(0.0);
    let porosity = porosity.clamp(0.0, 1.0);
    let dx = dx.abs().max(1e-10);
    let k_eff = effective_permeability(permeability, porosity);
    let flux = k_eff * concentration_gradient / dx;

    PermeationResult {
        flux,
        effective_permeability: k_eff,
    }
}

/// Compute effective permeability (Kozeny-Carman simplified).
///
/// k_eff = k * porosity
pub fn effective_permeability(permeability: f32, porosity: f32) -> f32 {
    permeability.max(0.0) * porosity.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_impermeable_zero_flux() {
        // permeability=0 → zero flux regardless of other parameters
        let result = compute_permeation(0.0, 100.0, 0.5, 0.01);
        assert!(
            result.flux.abs() < EPSILON,
            "Impermeable material should have zero flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_nonporous_zero_flux() {
        // porosity=0 → zero flux regardless of other parameters
        let result = compute_permeation(1e-12, 100.0, 0.0, 0.01);
        assert!(
            result.flux.abs() < EPSILON,
            "Non-porous material should have zero flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_flux_proportional_to_gradient() {
        // Doubling the concentration gradient should double the flux
        let result1 = compute_permeation(1e-12, 50.0, 0.3, 0.01);
        let result2 = compute_permeation(1e-12, 100.0, 0.3, 0.01);
        let ratio = result2.flux / result1.flux;
        assert!(
            (ratio - 2.0).abs() < EPSILON,
            "Doubling gradient should double flux, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_flux_proportional_to_permeability() {
        // Doubling the permeability should double the flux
        let result1 = compute_permeation(1e-12, 100.0, 0.3, 0.01);
        let result2 = compute_permeation(2e-12, 100.0, 0.3, 0.01);
        let ratio = result2.flux / result1.flux;
        assert!(
            (ratio - 2.0).abs() < EPSILON,
            "Doubling permeability should double flux, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_effective_permeability() {
        // k_eff = k * porosity
        let k = 1e-12_f32;
        let porosity = 0.3_f32;
        let k_eff = effective_permeability(k, porosity);
        let expected = k * porosity;
        assert!(
            (k_eff - expected).abs() < EPSILON,
            "Effective permeability should be k*porosity={expected}, got {k_eff}"
        );
    }

    #[test]
    fn test_negative_gradient_reverses_flux() {
        // Negative gradient should produce negative flux
        let pos = compute_permeation(1e-12, 100.0, 0.3, 0.01);
        let neg = compute_permeation(1e-12, -100.0, 0.3, 0.01);
        assert!(
            pos.flux > 0.0,
            "Positive gradient should yield positive flux, got {}",
            pos.flux
        );
        assert!(
            neg.flux < 0.0,
            "Negative gradient should yield negative flux, got {}",
            neg.flux
        );
        assert!(
            (pos.flux + neg.flux).abs() < EPSILON,
            "Magnitude should be equal for opposite gradients"
        );
    }
}
