//! Underwater acoustics scenario tests.
//!
//! Verifies Snell's law refraction, Fresnel reflection coefficients, energy
//! conservation, and total internal reflection at air-water boundaries.

#[cfg(test)]
mod tests {
    use crate::acoustics::ray::{AcousticRay, RayHit, DEFAULT_MAX_PATH_LENGTH};
    use crate::assert_relative_eq;
    use crate::scene::material::{
        AcousticMaterial, FrequencyBands, MediumLibrary, MediumProperties,
    };
    use glam::Vec3;

    fn air() -> MediumProperties {
        MediumProperties::air()
    }

    fn water() -> MediumProperties {
        MediumLibrary::with_defaults().get("Water").unwrap().clone()
    }

    /// Normal-incidence air-to-water: Fresnel R = ((Z_water - Z_air) / (Z_water + Z_air))^2.
    /// With Z_air ~420 and Z_water ~1.48e6, R should be ~0.999.
    #[test]
    fn test_air_water_reflection_coefficient() {
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let ray = AcousticRay {
            origin,
            direction: Vec3::new(0.0, -1.0, 0.0),
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        let z_air = air().impedance;
        let z_water = water_med.impedance;
        let expected_r = ((z_water - z_air) / (z_water + z_air)).powi(2);

        // R should match Fresnel within 1%
        assert_relative_eq!(result.reflected_energy, expected_r, 0.01);

        // R should be approximately 0.999 (massive impedance mismatch)
        assert!(
            result.reflected_energy > 0.99,
            "R should be ~0.999, got {}",
            result.reflected_energy
        );
    }

    /// Ray at 5 degrees from air to water (below critical angle 13.4 degrees).
    /// Snell: sin(theta2) = (c2/c1) * sin(theta1) = (1481/343) * sin(5 deg) ~ 0.376.
    /// Verify the ratio sin(theta2)/sin(theta1) = c2/c1 within 5%.
    #[test]
    fn test_snell_refraction_angle() {
        let angle = 5.0_f32.to_radians();
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin,
            direction: dir,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();
        let transmitted = result
            .transmitted_direction
            .expect("5 deg is below critical angle, should transmit");

        // sin(theta2) from the transmitted direction (x component = sin of angle from normal)
        let sin_theta2 = transmitted.x.abs();
        let sin_theta1 = angle.sin();
        let speed_ratio = 1481.0_f64 / 343.0_f64;
        let actual_ratio = sin_theta2 as f64 / sin_theta1 as f64;

        // Verify sin(theta2)/sin(theta1) = c2/c1 within 5%
        assert_relative_eq!(actual_ratio, speed_ratio, 0.05);

        // Also verify absolute value: sin(theta2) ~ 0.376
        let expected_sin_theta2 = (1481.0_f32 / 343.0) * angle.sin();
        assert!(
            (sin_theta2 - expected_sin_theta2).abs() < 0.02,
            "sin(theta2) expected ~{:.3}, got {:.3}",
            expected_sin_theta2,
            sin_theta2
        );
    }

    /// Ray from air to water at 20 degrees (above critical angle 13.4 degrees).
    /// Should produce total internal reflection: transmitted_direction is None,
    /// reflected_energy equals incident energy.
    #[test]
    fn test_total_internal_reflection() {
        let angle = 20.0_f32.to_radians();
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin,
            direction: dir,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        // sin(theta2) = (1481/343)*sin(20 deg) = 4.316*0.342 = 1.476 > 1 => TIR
        assert!(
            result.transmitted_direction.is_none(),
            "Air to water at 20 deg (beyond critical) should give total internal reflection"
        );
        assert!(
            (result.reflected_energy - 1.0).abs() < 0.001,
            "All energy should be reflected in TIR, got {}",
            result.reflected_energy
        );
        assert!(
            result.transmitted_energy.abs() < 0.001,
            "No energy should be transmitted in TIR, got {}",
            result.transmitted_energy
        );
    }

    /// For angles 2, 5, 8, 10, 12 degrees (all below critical angle 13.4 degrees):
    /// reflected_energy + transmitted_energy should equal incident energy (1.0)
    /// within 0.1%.
    #[test]
    fn test_energy_conservation_at_boundary() {
        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        for angle_deg in [2.0_f32, 5.0, 8.0, 10.0, 12.0] {
            let angle = angle_deg.to_radians();
            let origin = Vec3::new(0.0, 1.0, 0.0);
            let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
            let ray = AcousticRay {
                origin,
                direction: dir,
                energy: 1.0,
                bounces: 0,
                path: vec![origin],
                current_medium: air(),
                frequency_hz: 1000.0,
                max_path_length: DEFAULT_MAX_PATH_LENGTH,
            };

            let result = ray.refract(normal, &water_med).unwrap();
            let total = result.reflected_energy + result.transmitted_energy;

            // Within 0.1% of incident energy
            assert_relative_eq!(total as f64, 1.0_f64, 0.001);
        }
    }

    /// Compare air-to-water vs air-to-glass impedance mismatch.
    /// Glass has higher impedance (~1.3e7) than water (~1.48e6), so
    /// air-to-glass should reflect more energy than air-to-water.
    #[test]
    fn test_impedance_mismatch_strength() {
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let direction = Vec3::new(0.0, -1.0, 0.0);

        // Air to water
        let ray_water = AcousticRay {
            origin,
            direction,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };
        let water_med = water();
        let result_water = ray_water.refract(normal, &water_med).unwrap();

        // Air to glass (create glass medium with Z ~ 1.3e7)
        let glass_med = MediumProperties::new(
            "Glass",
            crate::scene::material::Medium::Solid,
            2500.0, // density kg/m^3
            5200.0, // speed of sound m/s
            3.7e10, // bulk modulus
            FrequencyBands {
                hz_125: 0.00002,
                hz_250: 0.00005,
                hz_500: 0.0002,
                hz_1000: 0.0005,
                hz_2000: 0.001,
                hz_4000: 0.003,
            },
        );
        // Z_glass = 2500 * 5200 = 1.3e7

        let ray_glass = AcousticRay {
            origin,
            direction,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };
        let result_glass = ray_glass.refract(normal, &glass_med).unwrap();

        // Glass impedance is higher than water impedance
        assert!(
            glass_med.impedance > water_med.impedance,
            "Glass impedance ({}) should exceed water impedance ({})",
            glass_med.impedance,
            water_med.impedance
        );

        // Higher impedance mismatch => higher reflection coefficient
        assert!(
            result_glass.reflected_energy >= result_water.reflected_energy,
            "Glass reflection ({}) should be >= water reflection ({})",
            result_glass.reflected_energy,
            result_water.reflected_energy
        );
    }

    /// Create a ray in water, call reflect() multiple times with a material.
    /// Energy should decrease monotonically after each bounce.
    #[test]
    fn test_multi_bounce_underwater() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let mut ray = AcousticRay {
            origin,
            direction: Vec3::new(1.0, 0.0, 0.0),
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: water(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let material = AcousticMaterial {
            name: "Underwater Wall".into(),
            absorption: FrequencyBands {
                hz_125: 0.10,
                hz_250: 0.10,
                hz_500: 0.10,
                hz_1000: 0.10,
                hz_2000: 0.10,
                hz_4000: 0.10,
            },
            scattering: 0.1,
            color: [0.5, 0.5, 0.5],
            ..Default::default()
        };

        let mut prev_energy = ray.energy;

        for i in 0..5 {
            let hit = RayHit {
                point: ray.origin + ray.direction * 5.0,
                normal: -ray.direction,
                distance: 5.0,
                triangle_index: 0,
            };

            ray.reflect(&hit, &material);

            assert!(
                ray.energy < prev_energy,
                "Energy should decrease after bounce {}: {} >= {}",
                i + 1,
                ray.energy,
                prev_energy
            );
            assert!(
                ray.energy > 0.0,
                "Energy should remain positive after bounce {}, got {}",
                i + 1,
                ray.energy
            );

            prev_energy = ray.energy;
        }

        // After 5 bounces with 10% absorption each: energy = (0.9)^5 = 0.59049
        assert_relative_eq!(ray.energy as f64, 0.9_f64.powi(5), 0.01);
    }

    /// Ray at 12.5 degrees (just below critical angle 13.4 degrees).
    /// Reflection coefficient should be high (approaching 1.0 near critical angle).
    #[test]
    fn test_grazing_incidence_reflection() {
        let angle = 12.5_f32.to_radians();
        let origin = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin,
            direction: dir,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        // Should still transmit (below critical angle)
        assert!(
            result.transmitted_direction.is_some(),
            "12.5 deg is below critical angle, should transmit"
        );

        // Near the critical angle, reflection coefficient should be high
        // (approaching 1.0 as we near critical angle of 13.4 deg)
        assert!(
            result.reflected_energy > 0.99,
            "Near-critical angle should have high reflection, got {}",
            result.reflected_energy
        );

        // Compare against a smaller angle (5 degrees) - reflection should be higher near critical
        let small_angle = 5.0_f32.to_radians();
        let small_dir = Vec3::new(small_angle.sin(), -small_angle.cos(), 0.0).normalize();
        let ray_small = AcousticRay {
            origin,
            direction: small_dir,
            energy: 1.0,
            bounces: 0,
            path: vec![origin],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };
        let result_small = ray_small.refract(normal, &water_med).unwrap();

        assert!(
            result.reflected_energy >= result_small.reflected_energy,
            "Reflection at 12.5 deg ({}) should be >= reflection at 5 deg ({})",
            result.reflected_energy,
            result_small.reflected_energy
        );
    }
}
