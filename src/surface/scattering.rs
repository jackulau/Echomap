use glam::Vec3;

/// Result of roughness-based acoustic scattering computation.
pub struct ScatteringResult {
    pub specular_weight: f32,
    pub diffuse_weight: f32,
    pub beckmann_width: f32,
}

/// Compute scattering weights from surface roughness and acoustic frequency.
///
/// Beckmann width sigma = roughness. When sigma = 0, fully specular.
/// Frequency dependence: scattering increases when wavelength ~ roughness.
/// When roughness << wavelength, surface appears smooth; when roughness >> wavelength, fully diffuse.
pub fn compute_scattering(
    roughness: f32,
    frequency_hz: f32,
    speed_of_sound: f32,
) -> ScatteringResult {
    let roughness = roughness.max(0.0);

    // Perfect mirror case
    if roughness < 1e-12 {
        return ScatteringResult {
            specular_weight: 1.0,
            diffuse_weight: 0.0,
            beckmann_width: 0.0,
        };
    }

    // Wavelength = speed_of_sound / frequency
    let frequency_hz = frequency_hz.max(1.0);
    let speed_of_sound = speed_of_sound.max(1e-6);
    let wavelength = speed_of_sound / frequency_hz;

    // Ratio of roughness to wavelength determines scattering behavior.
    // Rayleigh roughness parameter: k * sigma = 2*pi*roughness / wavelength
    let k_sigma = 2.0 * std::f32::consts::PI * roughness / wavelength;

    // Specular weight decays exponentially with (k*sigma)^2 (Rayleigh criterion)
    // exp(-(2*k*sigma)^2) is the coherent (specular) reflection coefficient
    let specular_weight = (-4.0 * k_sigma * k_sigma).exp().clamp(0.0, 1.0);
    let diffuse_weight = 1.0 - specular_weight;

    ScatteringResult {
        specular_weight,
        diffuse_weight,
        beckmann_width: roughness,
    }
}

/// Beckmann microfacet distribution probability density for scattered angle.
///
/// Returns the probability density for angle theta deviation from specular.
/// When roughness is zero (or near-zero), returns 0.0 (delta function, handle specially).
///
/// P(theta) = exp(-tan^2(theta) / sigma^2) / (pi * sigma^2 * cos^4(theta))
///
/// Normalized over the hemisphere with the solid angle measure sin(theta) d_theta d_phi:
///   integral_0^{2pi} integral_0^{pi/2} P(theta) * sin(theta) d_theta d_phi ~ 1
pub fn beckmann_pdf(theta: f32, roughness: f32) -> f32 {
    let roughness = roughness.max(0.0);

    // Guard against roughness=0: delta function, handle specially
    if roughness < 1e-12 {
        return 0.0;
    }

    let cos_theta = theta.cos();
    // Avoid division by zero when theta approaches pi/2
    if cos_theta < 1e-8 {
        return 0.0;
    }

    let sigma_sq = roughness * roughness;
    let tan_theta = theta.tan();
    let cos4 = cos_theta * cos_theta * cos_theta * cos_theta;

    // P(theta) = exp(-tan^2(theta)/sigma^2) / (pi * sigma^2 * cos^4(theta))
    let exponent = -(tan_theta * tan_theta) / sigma_sq;
    exponent.exp() / (std::f32::consts::PI * sigma_sq * cos4)
}

/// Sample a scattered direction from the Beckmann distribution in local frame.
///
/// Uses inverse CDF sampling. The local frame has z-up (surface normal).
/// rng_u1 and rng_u2 should be uniform random in [0, 1).
/// Returns a unit vector in the upper hemisphere (z >= 0).
pub fn sample_beckmann(roughness: f32, rng_u1: f32, rng_u2: f32) -> Vec3 {
    let roughness = roughness.max(0.0);

    // Perfect specular: always return surface normal
    if roughness < 1e-12 {
        return Vec3::new(0.0, 0.0, 1.0);
    }

    let sigma_sq = roughness * roughness;

    // Clamp u1 away from 1.0 to avoid ln(0)
    let u1 = rng_u1.clamp(0.0, 1.0 - 1e-7);

    // Inverse CDF for Beckmann: tan^2(theta) = -sigma^2 * ln(1 - u1)
    let tan_theta_sq = -sigma_sq * (1.0 - u1).ln();
    let tan_theta = tan_theta_sq.sqrt();
    let theta = tan_theta.atan();

    // Azimuthal angle uniform in [0, 2*pi)
    let phi = 2.0 * std::f32::consts::PI * rng_u2;

    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let x = sin_theta * phi.cos();
    let y = sin_theta * phi.sin();
    let z = cos_theta;

    let dir = Vec3::new(x, y, z);
    // Normalize to ensure unit length (should already be close)
    dir.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_smooth_surface_specular() {
        let result = compute_scattering(0.0, 1000.0, 343.0);
        assert!(
            (result.specular_weight - 1.0).abs() < EPSILON,
            "Smooth surface should be fully specular, got {}",
            result.specular_weight
        );
        assert!(
            result.diffuse_weight.abs() < EPSILON,
            "Smooth surface should have zero diffuse, got {}",
            result.diffuse_weight
        );
    }

    #[test]
    fn test_rough_surface_diffuse() {
        // Very rough surface relative to wavelength
        let result = compute_scattering(1.0, 1000.0, 343.0);
        assert!(
            result.diffuse_weight > 0.8,
            "Rough surface should be mostly diffuse, got {}",
            result.diffuse_weight
        );
        assert!(
            result.specular_weight < 0.2,
            "Rough surface should have little specular, got {}",
            result.specular_weight
        );
    }

    #[test]
    fn test_frequency_dependence() {
        let roughness = 0.01; // 1cm roughness
                              // Low frequency: long wavelength (343/100 = 3.43m), roughness << wavelength -> smoother
        let low_freq = compute_scattering(roughness, 100.0, 343.0);
        // High frequency: short wavelength (343/10000 = 0.0343m), roughness comparable -> rougher
        let high_freq = compute_scattering(roughness, 10000.0, 343.0);

        assert!(
            low_freq.specular_weight > high_freq.specular_weight,
            "Low frequency should see surface as smoother (more specular): low={} > high={}",
            low_freq.specular_weight,
            high_freq.specular_weight
        );
    }

    #[test]
    fn test_beckmann_pdf_normalized() {
        let roughness = 0.3;
        let n_steps = 1000;
        let d_theta = std::f32::consts::FRAC_PI_2 / n_steps as f32;
        let mut integral = 0.0_f32;

        for i in 0..n_steps {
            let theta = (i as f32 + 0.5) * d_theta;
            let pdf_val = beckmann_pdf(theta, roughness);
            // Integrate over solid angle: pdf * sin(theta) * d_theta * 2*pi
            integral += pdf_val * theta.sin() * d_theta * 2.0 * std::f32::consts::PI;
        }

        assert!(
            (integral - 1.0).abs() < 0.1,
            "Beckmann PDF should integrate to ~1.0 over hemisphere, got {}",
            integral
        );
    }

    #[test]
    fn test_beckmann_pdf_peak_at_zero() {
        let roughness = 0.1;
        let pdf_at_zero = beckmann_pdf(0.01, roughness); // near zero
        let pdf_at_45 = beckmann_pdf(std::f32::consts::FRAC_PI_4, roughness);

        assert!(
            pdf_at_zero > pdf_at_45,
            "Beckmann PDF should peak near theta=0 for low roughness: near_zero={} > at_45={}",
            pdf_at_zero,
            pdf_at_45
        );
    }

    #[test]
    fn test_sample_beckmann_in_hemisphere() {
        let roughness = 0.5;
        // Test multiple samples
        let samples = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8), (0.0, 0.0), (0.99, 0.99)];

        for (u1, u2) in samples {
            let dir = sample_beckmann(roughness, u1, u2);
            assert!(
                dir.z >= 0.0,
                "Sampled direction should be in upper hemisphere: z={} for u1={}, u2={}",
                dir.z,
                u1,
                u2
            );
            let len = dir.length();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "Sampled direction should be unit length: len={} for u1={}, u2={}",
                len,
                u1,
                u2
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_roughness_one_max_diffuse() {
        // roughness = 1.0 (max realistic), should be fully diffuse at typical audio
        let result = compute_scattering(1.0, 1000.0, 343.0);
        assert!(
            result.diffuse_weight > 0.99,
            "roughness=1.0 should be nearly fully diffuse, got diffuse={}",
            result.diffuse_weight
        );
        assert!(
            (result.specular_weight + result.diffuse_weight - 1.0).abs() < EPSILON,
            "Weights should sum to 1.0"
        );
    }

    #[test]
    fn test_edge_frequency_1hz() {
        // Very low frequency: huge wavelength, surface appears smooth
        let result = compute_scattering(0.01, 1.0, 343.0);
        // wavelength = 343.0m, roughness/wavelength = 0.01/343 ~ 3e-5
        assert!(
            result.specular_weight > 0.99,
            "1Hz frequency should make any surface appear specular, got {}",
            result.specular_weight
        );
    }

    #[test]
    fn test_edge_frequency_1mhz() {
        // Very high frequency: tiny wavelength, even smooth surface becomes diffuse
        let result = compute_scattering(0.001, 1_000_000.0, 343.0);
        // wavelength = 343/1e6 = 0.000343m, roughness/wavelength = 0.001/0.000343 ~ 2.9
        assert!(
            result.diffuse_weight > 0.9,
            "1MHz with roughness=0.001 should be mostly diffuse, got diffuse={}",
            result.diffuse_weight
        );
    }

    #[test]
    fn test_edge_speed_of_sound_zero() {
        // speed_of_sound = 0 should be clamped to 1e-6
        let result = compute_scattering(0.01, 1000.0, 0.0);
        assert!(
            result.specular_weight.is_finite(),
            "speed_of_sound=0 should produce finite result, got specular={}",
            result.specular_weight
        );
        assert!(
            result.diffuse_weight.is_finite(),
            "speed_of_sound=0 should produce finite diffuse result"
        );
    }

    #[test]
    fn test_edge_speed_of_sound_negative() {
        // Negative speed_of_sound should be clamped to 1e-6
        let result = compute_scattering(0.01, 1000.0, -343.0);
        assert!(
            result.specular_weight.is_finite(),
            "Negative speed_of_sound should produce finite result"
        );
    }

    #[test]
    fn test_edge_negative_roughness_clamped() {
        let result = compute_scattering(-0.5, 1000.0, 343.0);
        // Negative roughness clamped to 0 -> specular
        assert!(
            (result.specular_weight - 1.0).abs() < EPSILON,
            "Negative roughness should be clamped to 0 (fully specular), got {}",
            result.specular_weight
        );
    }

    #[test]
    fn test_edge_negative_frequency_clamped() {
        // Negative frequency clamped to 1.0
        let result = compute_scattering(0.01, -1000.0, 343.0);
        assert!(
            result.specular_weight.is_finite(),
            "Negative frequency should produce finite result"
        );
    }

    #[test]
    fn test_edge_all_zero_inputs() {
        let result = compute_scattering(0.0, 0.0, 0.0);
        // roughness=0 triggers the early return for perfect mirror
        assert!(
            (result.specular_weight - 1.0).abs() < EPSILON,
            "All-zero inputs should be specular (roughness=0 path)"
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_theta_zero() {
        // theta=0 should give the peak value
        let pdf = beckmann_pdf(0.0, 0.3);
        assert!(
            pdf > 0.0 && pdf.is_finite(),
            "PDF at theta=0 should be positive finite, got {}",
            pdf
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_theta_pi_over_2() {
        // At theta = pi/2, cos(theta) = 0, should return 0.0 (guard clause)
        let pdf = beckmann_pdf(std::f32::consts::FRAC_PI_2, 0.3);
        assert!(
            pdf.abs() < EPSILON,
            "PDF at theta=pi/2 should be 0.0, got {}",
            pdf
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_roughness_1() {
        // High roughness: PDF should be broader and lower at peak
        let pdf_low = beckmann_pdf(0.01, 0.1);
        let pdf_high = beckmann_pdf(0.01, 1.0);
        assert!(
            pdf_low > pdf_high,
            "Low roughness should have higher peak PDF: low={} > high={}",
            pdf_low,
            pdf_high
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_zero_roughness() {
        // roughness=0 -> delta function, returns 0.0
        let pdf = beckmann_pdf(0.1, 0.0);
        assert!(
            pdf.abs() < EPSILON,
            "Zero roughness PDF should return 0.0, got {}",
            pdf
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_negative_roughness() {
        // Negative roughness clamped to 0 -> returns 0.0
        let pdf = beckmann_pdf(0.1, -0.5);
        assert!(
            pdf.abs() < EPSILON,
            "Negative roughness PDF should return 0.0, got {}",
            pdf
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_negative_theta() {
        // Negative theta: cos(-theta)=cos(theta), should still work
        let pdf_pos = beckmann_pdf(0.3, 0.5);
        let pdf_neg = beckmann_pdf(-0.3, 0.5);
        assert!(
            (pdf_pos - pdf_neg).abs() < EPSILON,
            "PDF should be symmetric: pos={}, neg={}",
            pdf_pos,
            pdf_neg
        );
    }

    #[test]
    fn test_edge_sample_beckmann_zero_roughness() {
        let dir = sample_beckmann(0.0, 0.5, 0.5);
        // Should return surface normal (0, 0, 1)
        assert!(
            (dir.z - 1.0).abs() < 1e-4,
            "Zero roughness should sample surface normal, got z={}",
            dir.z
        );
    }

    #[test]
    fn test_edge_sample_beckmann_u1_zero() {
        let dir = sample_beckmann(0.5, 0.0, 0.5);
        // u1=0 -> ln(1-0)=0 -> tan_theta=0 -> theta=0 -> z=1
        assert!(
            dir.z > 0.99,
            "u1=0 should produce near-normal direction, got z={}",
            dir.z
        );
        assert!((dir.length() - 1.0).abs() < 1e-4, "Should be unit length");
    }

    #[test]
    fn test_edge_sample_beckmann_u1_one() {
        // u1=1.0 is clamped to 1.0-1e-7 to avoid ln(0)
        let dir = sample_beckmann(0.5, 1.0, 0.5);
        assert!(
            dir.z >= 0.0,
            "u1=1.0 should still be in upper hemisphere, got z={}",
            dir.z
        );
        assert!(
            dir.length().is_finite(),
            "u1=1.0 should produce finite direction"
        );
    }

    #[test]
    fn test_edge_sample_beckmann_u2_zero_and_one() {
        // u2 controls azimuthal angle
        let dir0 = sample_beckmann(0.5, 0.5, 0.0);
        let dir1 = sample_beckmann(0.5, 0.5, 1.0);
        // phi=0 and phi=2*pi should be nearly the same direction
        assert!(
            (dir0 - dir1).length() < 1e-3,
            "u2=0 and u2=1 should produce same direction (phi=0 vs 2pi)"
        );
    }

    #[test]
    fn test_edge_sample_beckmann_negative_roughness() {
        let dir = sample_beckmann(-1.0, 0.5, 0.5);
        // Negative roughness clamped to 0 -> returns normal
        assert!(
            (dir.z - 1.0).abs() < 1e-4,
            "Negative roughness should clamp to 0 and return normal"
        );
    }

    #[test]
    fn test_edge_scattering_weights_sum_to_one_sweep() {
        // Sweep roughness and frequency, verify weights always sum to 1.0
        for roughness in [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0] {
            for freq in [1.0_f32, 100.0, 1000.0, 10000.0, 100000.0] {
                let result = compute_scattering(roughness, freq, 343.0);
                let sum = result.specular_weight + result.diffuse_weight;
                assert!(
                    (sum - 1.0).abs() < EPSILON,
                    "Weights must sum to 1.0 for roughness={roughness}, freq={freq}: got {sum}"
                );
                assert!(
                    result.specular_weight >= 0.0 && result.specular_weight <= 1.0,
                    "Specular weight out of range [0,1]: {} for roughness={roughness}, freq={freq}",
                    result.specular_weight
                );
            }
        }
    }

    #[test]
    fn test_edge_scattering_nan_roughness() {
        let result = compute_scattering(f32::NAN, 1000.0, 343.0);
        // NaN.max(0.0) returns 0.0 in Rust -> should produce specular
        assert!(
            (result.specular_weight - 1.0).abs() < EPSILON,
            "NaN roughness should be clamped to 0.0 (specular), got {}",
            result.specular_weight
        );
    }

    #[test]
    fn test_edge_scattering_inf_roughness() {
        let result = compute_scattering(f32::INFINITY, 1000.0, 343.0);
        // Inf roughness: k_sigma = Inf, exp(-Inf) = 0 -> fully diffuse
        assert!(
            result.diffuse_weight >= 1.0 - EPSILON,
            "Infinite roughness should be fully diffuse, got {}",
            result.diffuse_weight
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_theta_beyond_pi_over_2() {
        // theta > pi/2: cos(theta) < 0, guard returns 0.0
        let pdf = beckmann_pdf(2.0, 0.5); // ~114 degrees
        assert!(
            pdf.abs() < EPSILON,
            "Beckmann PDF for theta > pi/2 should return 0, got {pdf}"
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_theta_exactly_zero() {
        // theta = 0 exactly: tan(0)=0, cos(0)=1, exp(0)=1
        // P(0) = 1 / (pi * sigma^2)
        let roughness = 0.3;
        let pdf = beckmann_pdf(0.0, roughness);
        let expected = 1.0 / (std::f32::consts::PI * roughness * roughness);
        assert!(
            (pdf - expected).abs() < 0.01,
            "PDF(0) should be 1/(pi*sigma^2)={expected}, got {pdf}"
        );
    }

    #[test]
    fn test_edge_beckmann_pdf_very_large_roughness() {
        // Very large roughness: distribution becomes very flat
        let pdf_center = beckmann_pdf(0.01, 100.0);
        let pdf_off = beckmann_pdf(1.0, 100.0);
        assert!(
            pdf_center.is_finite(),
            "Large roughness center PDF should be finite"
        );
        assert!(
            pdf_off.is_finite(),
            "Large roughness off-axis PDF should be finite"
        );
        // With roughness=100, distribution is very flat so ratio should be near 1
        if pdf_center > EPSILON && pdf_off > EPSILON {
            let ratio = pdf_center / pdf_off;
            assert!(
                ratio < 2.0,
                "Large roughness should have flat distribution, ratio={ratio}"
            );
        }
    }

    #[test]
    fn test_edge_sample_beckmann_high_roughness_spread() {
        // High roughness should produce more off-axis samples
        let roughness = 2.0;
        let dir = sample_beckmann(roughness, 0.9, 0.25);
        assert!(dir.is_finite(), "High roughness sample should be finite");
        assert!(
            dir.z >= 0.0,
            "High roughness sample must still be in upper hemisphere"
        );
        // With high roughness and u1=0.9, theta should be far from normal
        assert!(
            dir.z < 0.9,
            "High roughness + high u1 should produce off-axis direction, z={}",
            dir.z
        );
    }

    #[test]
    fn test_edge_scattering_nan_frequency() {
        let result = compute_scattering(0.01, f32::NAN, 343.0);
        // NaN.max(1.0) = 1.0 in Rust => proceeds with freq=1.0
        assert!(
            result.specular_weight.is_finite() || result.specular_weight.is_nan(),
            "NaN frequency should not panic"
        );
    }

    #[test]
    fn test_edge_scattering_nan_speed_of_sound() {
        let result = compute_scattering(0.01, 1000.0, f32::NAN);
        // NaN.max(1e-6) = 1e-6 in Rust
        assert!(
            result.specular_weight.is_finite() || result.specular_weight.is_nan(),
            "NaN speed_of_sound should not panic"
        );
    }

    #[test]
    fn test_edge_sample_beckmann_nan_roughness() {
        let dir = sample_beckmann(f32::NAN, 0.5, 0.5);
        // NaN.max(0.0) = 0.0 in Rust => returns surface normal
        assert!(
            (dir.z - 1.0).abs() < 1e-4 || dir.z.is_nan(),
            "NaN roughness should clamp to 0 or propagate NaN"
        );
    }
}
