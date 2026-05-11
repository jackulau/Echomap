use std::f64::consts::PI;

/// Abramowitz and Stegun approximation for the complementary error function.
fn erfc_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        2.0 - result
    }
}

/// Stokes drag on a sphere: F = 6*pi*mu*r*v
pub fn stokes_drag(viscosity: f64, radius: f64, velocity: f64) -> f64 {
    6.0 * PI * viscosity * radius * velocity
}

/// Terminal velocity: v_t = (2/9)*(r^2/mu)*(rho_s - rho_f)*g
pub fn terminal_velocity(
    radius: f64,
    viscosity: f64,
    density_solid: f64,
    density_fluid: f64,
    gravity: f64,
) -> f64 {
    (2.0 / 9.0) * (radius * radius / viscosity) * (density_solid - density_fluid) * gravity
}

/// Fick's 1D diffusion: c(x,t) = (c0/2) * erfc(x / (2*sqrt(D*t)))
pub fn fick_diffusion_1d(c0: f64, x: f64, diffusion_coeff: f64, time: f64) -> f64 {
    let arg = x / (2.0 * (diffusion_coeff * time).sqrt());
    (c0 / 2.0) * erfc_approx(arg)
}

/// Fresnel reflection at normal incidence: R = ((Z2 - Z1) / (Z2 + Z1))^2
pub fn fresnel_reflection(z1: f64, z2: f64) -> f64 {
    let ratio = (z2 - z1) / (z2 + z1);
    ratio * ratio
}

/// Darcy flow rate: Q = (k * A / mu) * (delta_P / L)
pub fn darcy_flow_rate(
    permeability: f64,
    area: f64,
    viscosity: f64,
    pressure_drop: f64,
    length: f64,
) -> f64 {
    (permeability * area / viscosity) * (pressure_drop / length)
}

/// Coulomb kinetic friction: F = mu_k * N
pub fn coulomb_kinetic_friction(friction_coeff: f64, normal_force: f64) -> f64 {
    friction_coeff * normal_force
}

/// Coulomb static friction threshold: F_max = mu_s * N
pub fn coulomb_static_threshold(friction_coeff: f64, normal_force: f64) -> f64 {
    friction_coeff * normal_force
}

/// Beckmann specular fraction from RMS roughness and wavelength.
/// At normal incidence: exp(-(4*pi*sigma/lambda)^2)
pub fn beckmann_specular_fraction(rms_roughness: f64, wavelength: f64) -> f64 {
    let arg = 4.0 * PI * rms_roughness / wavelength;
    (-arg * arg).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_relative_eq;

    #[test]
    fn test_stokes_drag_sphere() {
        // 1mm sphere in water (mu=0.001 Pa-s) at 0.01 m/s
        let f = stokes_drag(0.001, 0.001, 0.01);
        let expected = 6.0 * PI * 0.001 * 0.001 * 0.01; // ~1.885e-7 N
        assert_relative_eq!(f, expected, 1e-6);
    }

    #[test]
    fn test_terminal_velocity_steel_in_water() {
        // steel (rho=7800), water (rho=998), r=0.005m, mu=0.001, g=9.81
        let v = terminal_velocity(0.005, 0.001, 7800.0, 998.0, 9.81);
        let expected = (2.0 / 9.0) * (0.005_f64.powi(2) / 0.001) * (7800.0 - 998.0) * 9.81;
        assert_relative_eq!(v, expected, 1e-6);
    }

    #[test]
    fn test_fick_diffusion_profile() {
        // CO2 in air (D=1.6e-5), c0=1.0, t=1.0s
        let d = 1.6e-5;
        let c0 = 1.0;
        let t = 1.0;

        // At x=0: c = c0/2 * erfc(0) = c0/2 * 1 = 0.5
        let c_at_zero = fick_diffusion_1d(c0, 0.0, d, t);
        assert_relative_eq!(c_at_zero, 0.5, 1e-6);

        // Verify monotonically decreasing with distance
        let c_near = fick_diffusion_1d(c0, 0.001, d, t);
        let c_far = fick_diffusion_1d(c0, 0.01, d, t);
        assert!(
            c_near > c_far,
            "concentration must decrease with distance: c_near={}, c_far={}",
            c_near,
            c_far
        );

        // At large x, concentration approaches 0
        let c_very_far = fick_diffusion_1d(c0, 1.0, d, t);
        assert!(
            c_very_far < 1e-6,
            "concentration should be near zero far from source: {}",
            c_very_far
        );
    }

    #[test]
    fn test_fresnel_air_water() {
        // Z_air = 1.225 * 343 = 420.175, Z_water = 998 * 1481 = 1478038
        let z_air = 1.225 * 343.0;
        let z_water = 998.0 * 1481.0;
        let r = fresnel_reflection(z_air, z_water);
        let expected = ((z_water - z_air) / (z_water + z_air)).powi(2);
        assert_relative_eq!(r, expected, 1e-6);
        // Should be close to 0.9989
        assert!(r > 0.998, "air-water reflection should be very high: {}", r);
    }

    #[test]
    fn test_fresnel_air_glass() {
        // Z_glass ~ 1.3e7 (typical acoustic impedance of glass)
        let z_air = 1.225 * 343.0;
        let z_glass = 1.3e7;
        let r = fresnel_reflection(z_air, z_glass);
        let expected = ((z_glass - z_air) / (z_glass + z_air)).powi(2);
        assert_relative_eq!(r, expected, 1e-6);
        // Should be very close to 1.0 for air-glass
        assert!(r > 0.999, "air-glass reflection should be very high: {}", r);
    }

    #[test]
    fn test_darcy_flow_concrete() {
        // k=1e-15 m^2, A=1 m^2, mu=1.8e-5 Pa-s, delta_P=100 Pa, L=0.1m
        let q = darcy_flow_rate(1e-15, 1.0, 1.8e-5, 100.0, 0.1);
        let expected = (1e-15 * 1.0 / 1.8e-5) * (100.0 / 0.1);
        assert_relative_eq!(q, expected, 1e-6);
    }

    #[test]
    fn test_coulomb_static_threshold() {
        let f_max = coulomb_static_threshold(0.6, 100.0);
        assert_relative_eq!(f_max, 60.0, 1e-6);
    }

    #[test]
    fn test_coulomb_kinetic_force() {
        let f = coulomb_kinetic_friction(0.4, 100.0);
        assert_relative_eq!(f, 40.0, 1e-6);
    }

    #[test]
    fn test_beckmann_smooth_surface() {
        // roughness=0.001, wavelength=1.0 => sigma/lambda << 1 => specular ~ 1.0
        let spec = beckmann_specular_fraction(0.001, 1.0);
        assert!(
            spec > 0.999,
            "smooth surface should have specular fraction near 1.0: {}",
            spec
        );
        let expected = (-(4.0 * PI * 0.001 / 1.0_f64).powi(2)).exp();
        assert_relative_eq!(spec, expected, 1e-6);
    }

    #[test]
    fn test_beckmann_rough_surface() {
        // roughness=1.0, wavelength=0.01 => sigma/lambda >> 1 => specular ~ 0.0
        let spec = beckmann_specular_fraction(1.0, 0.01);
        assert!(
            spec < 1e-10,
            "rough surface should have specular fraction near 0.0: {}",
            spec
        );
    }
}
