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

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_all_params_zero() {
        let result = compute_permeation(0.0, 0.0, 0.0, 0.0);
        // permeability=0, porosity=0 -> k_eff=0 -> flux=0
        // dx=0 clamped to 1e-10
        assert!(
            result.flux.abs() < EPSILON,
            "All zero params should give zero flux, got {}",
            result.flux
        );
        assert!(
            result.effective_permeability.abs() < EPSILON,
            "All zero params should give zero effective_permeability"
        );
    }

    #[test]
    fn test_edge_dx_zero() {
        // dx=0 should be clamped to 1e-10, producing a large but finite flux
        let result = compute_permeation(1e-12, 100.0, 0.3, 0.0);
        assert!(
            result.flux.is_finite(),
            "dx=0 should produce finite flux (clamped), got {}",
            result.flux
        );
        // flux = k_eff * gradient / 1e-10
        let k_eff = 1e-12 * 0.3;
        let expected = k_eff * 100.0 / 1e-10;
        assert!(
            (result.flux - expected).abs() / expected.abs() < 1e-4,
            "dx=0 flux should match clamped calculation: expected {}, got {}",
            expected,
            result.flux
        );
    }

    #[test]
    fn test_edge_dx_negative() {
        // dx negative should produce same result as abs(dx)
        let result_neg = compute_permeation(1e-12, 100.0, 0.3, -0.01);
        let result_pos = compute_permeation(1e-12, 100.0, 0.3, 0.01);
        assert!(
            (result_neg.flux - result_pos.flux).abs() < EPSILON,
            "Negative dx should produce same flux as positive: neg={}, pos={}",
            result_neg.flux,
            result_pos.flux
        );
    }

    #[test]
    fn test_edge_porosity_above_one_clamped() {
        // Porosity > 1.0 should be clamped to 1.0
        let result = compute_permeation(1e-12, 100.0, 1.5, 0.01);
        let result_at_one = compute_permeation(1e-12, 100.0, 1.0, 0.01);
        assert!(
            (result.flux - result_at_one.flux).abs() < EPSILON,
            "Porosity > 1.0 should be clamped to 1.0: got flux={}, expected={}",
            result.flux,
            result_at_one.flux
        );
    }

    #[test]
    fn test_edge_porosity_negative_clamped() {
        // Negative porosity should be clamped to 0
        let result = compute_permeation(1e-12, 100.0, -0.5, 0.01);
        assert!(
            result.flux.abs() < EPSILON,
            "Negative porosity (clamped to 0) should give zero flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_edge_negative_permeability_clamped() {
        let result = compute_permeation(-1e-12, 100.0, 0.3, 0.01);
        assert!(
            result.flux.abs() < EPSILON,
            "Negative permeability (clamped to 0) should give zero flux, got {}",
            result.flux
        );
        assert!(
            result.effective_permeability.abs() < EPSILON,
            "Negative permeability should give zero effective_permeability"
        );
    }

    #[test]
    fn test_edge_extreme_concentration_gradient() {
        // Very large gradient
        let result = compute_permeation(1e-12, 1e12, 0.3, 0.01);
        assert!(
            result.flux.is_finite(),
            "Extreme gradient should produce finite flux, got {}",
            result.flux
        );
        // Very small gradient
        let result_tiny = compute_permeation(1e-12, 1e-12, 0.3, 0.01);
        assert!(
            result_tiny.flux.is_finite(),
            "Tiny gradient should produce finite flux"
        );
        assert!(
            result.flux.abs() > result_tiny.flux.abs(),
            "Larger gradient should produce larger flux"
        );
    }

    #[test]
    fn test_edge_zero_gradient() {
        let result = compute_permeation(1e-12, 0.0, 0.3, 0.01);
        assert!(
            result.flux.abs() < EPSILON,
            "Zero gradient should produce zero flux, got {}",
            result.flux
        );
        // effective_permeability should still be computed
        assert!(
            result.effective_permeability > 0.0,
            "effective_permeability should be nonzero even with zero gradient"
        );
    }

    #[test]
    fn test_edge_full_porosity() {
        // porosity=1.0: fully porous (like empty space)
        let result = compute_permeation(1e-12, 100.0, 1.0, 0.01);
        let k_eff = 1e-12 * 1.0;
        let expected_flux = k_eff * 100.0 / 0.01;
        assert!(
            (result.flux - expected_flux).abs() / expected_flux.abs() < 1e-4,
            "Full porosity flux: expected {}, got {}",
            expected_flux,
            result.flux
        );
    }

    #[test]
    fn test_edge_effective_permeability_zero_porosity() {
        let k_eff = effective_permeability(1e-12, 0.0);
        assert!(
            k_eff.abs() < EPSILON,
            "Zero porosity effective_permeability should be 0"
        );
    }

    #[test]
    fn test_edge_effective_permeability_zero_permeability() {
        let k_eff = effective_permeability(0.0, 0.5);
        assert!(
            k_eff.abs() < EPSILON,
            "Zero permeability effective_permeability should be 0"
        );
    }

    #[test]
    fn test_edge_nan_permeability() {
        let result = compute_permeation(f32::NAN, 100.0, 0.3, 0.01);
        // NaN.max(0.0) returns 0.0 -> flux should be 0
        assert!(
            result.flux.abs() < EPSILON || result.flux.is_nan(),
            "NaN permeability should produce zero or NaN flux"
        );
    }

    #[test]
    fn test_edge_infinity_gradient() {
        let result = compute_permeation(1e-12, f32::INFINITY, 0.3, 0.01);
        // Should produce infinite flux
        assert!(
            result.flux.is_infinite(),
            "Infinite gradient should produce infinite flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_edge_very_small_dx() {
        // dx just above the clamp threshold
        let result = compute_permeation(1e-12, 100.0, 0.3, 1e-9);
        assert!(
            result.flux.is_finite(),
            "Very small dx should produce finite flux"
        );
        // flux = 1e-12 * 0.3 * 100 / 1e-9 = 3e-2
        let expected = 1e-12 * 0.3 * 100.0 / 1e-9;
        assert!(
            (result.flux - expected).abs() / expected.abs() < 1e-4,
            "Very small dx flux: expected {}, got {}",
            expected,
            result.flux
        );
    }

    #[test]
    fn test_edge_nan_porosity() {
        let result = compute_permeation(1e-12, 100.0, f32::NAN, 0.01);
        // NaN.clamp(0.0, 1.0) = NaN in Rust => k_eff = NaN => flux = NaN
        assert!(
            result.flux.is_nan(),
            "NaN porosity should propagate NaN to flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_edge_nan_dx() {
        let result = compute_permeation(1e-12, 100.0, 0.3, f32::NAN);
        // NaN.abs() = NaN; NaN.max(1e-10) = 1e-10 in Rust
        // so dx = 1e-10, flux should be finite (very large)
        assert!(
            result.flux.is_finite() || result.flux.is_nan(),
            "NaN dx should not panic"
        );
    }

    #[test]
    fn test_edge_nan_gradient() {
        let result = compute_permeation(1e-12, f32::NAN, 0.3, 0.01);
        // k_eff * NaN / dx = NaN
        assert!(
            result.flux.is_nan(),
            "NaN gradient should propagate NaN to flux"
        );
        // effective_permeability should still be finite (doesn't depend on gradient)
        assert!(
            result.effective_permeability.is_finite(),
            "NaN gradient should not affect effective_permeability"
        );
    }

    #[test]
    fn test_edge_infinity_permeability() {
        let result = compute_permeation(f32::INFINITY, 100.0, 0.3, 0.01);
        assert!(
            result.flux.is_infinite(),
            "Infinite permeability should produce infinite flux"
        );
        assert!(
            result.effective_permeability.is_infinite(),
            "Infinite permeability should produce infinite k_eff"
        );
    }

    #[test]
    fn test_edge_neg_infinity_permeability_clamped() {
        let result = compute_permeation(f32::NEG_INFINITY, 100.0, 0.3, 0.01);
        // NEG_INFINITY.max(0.0) = 0.0
        assert!(
            result.flux.abs() < EPSILON,
            "Negative infinity permeability clamped to 0 should yield zero flux, got {}",
            result.flux
        );
    }

    #[test]
    fn test_edge_effective_permeability_negative_inputs() {
        let k_eff = effective_permeability(-1e-12, -0.5);
        assert!(
            k_eff.abs() < EPSILON,
            "Negative inputs to effective_permeability should both clamp to 0, got {}",
            k_eff
        );
    }

    #[test]
    fn test_edge_effective_permeability_porosity_above_one() {
        let k_eff = effective_permeability(1e-12, 1.5);
        let k_eff_one = effective_permeability(1e-12, 1.0);
        assert!(
            (k_eff - k_eff_one).abs() < EPSILON,
            "Porosity > 1 should be clamped to 1 in effective_permeability"
        );
    }

    #[test]
    fn test_edge_flux_inversely_proportional_to_dx() {
        // Halving dx should double flux
        let r1 = compute_permeation(1e-12, 100.0, 0.3, 0.02);
        let r2 = compute_permeation(1e-12, 100.0, 0.3, 0.01);
        let ratio = r2.flux / r1.flux;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "Halving dx should double flux, ratio={ratio}"
        );
    }

    #[test]
    fn test_edge_large_overflow_scenario() {
        // Large permeability * large gradient / tiny dx => potential overflow
        let result = compute_permeation(1e20, 1e20, 1.0, 1e-10);
        // 1e20 * 1.0 * 1e20 / 1e-10 = 1e50 which overflows f32 (max ~3.4e38)
        assert!(
            result.flux.is_infinite(),
            "Overflow scenario should produce infinite flux, got {}",
            result.flux
        );
    }
}
