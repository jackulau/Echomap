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
    let frequency_hz = frequency_hz.max(1.0); // guard against zero frequency
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
}
