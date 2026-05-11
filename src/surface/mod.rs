#[allow(dead_code)]
pub mod friction;
#[allow(dead_code)]
pub mod permeability;
#[allow(dead_code)]
pub mod scattering;
#[allow(dead_code)]
pub mod wetting;

#[allow(unused_imports)]
pub use friction::*;
#[allow(unused_imports)]
pub use permeability::*;
#[allow(unused_imports)]
pub use scattering::*;
#[allow(unused_imports)]
pub use wetting::*;

use crate::scene::material::AcousticMaterial;
use glam::Vec3;

/// Facade aggregating all surface physics computations.
///
/// Constructed from an `AcousticMaterial`, dispatches to the friction,
/// scattering, wetting, and permeability submodules.
#[allow(dead_code)]
pub struct SurfaceInteraction {
    pub friction_static: f32,
    pub friction_kinetic: f32,
    pub roughness: f32,
    pub porosity: f32,
    pub permeability: f32,
    pub contact_angle: f32,
}

#[allow(dead_code)]
impl SurfaceInteraction {
    /// Extract surface properties from an AcousticMaterial.
    pub fn from_material(material: &AcousticMaterial) -> Self {
        Self {
            friction_static: material.friction_static,
            friction_kinetic: material.friction_kinetic,
            roughness: material.roughness,
            porosity: material.porosity,
            permeability: material.permeability,
            contact_angle: material.contact_angle,
        }
    }

    /// Compute roughness-based scattering at a given frequency.
    pub fn scattering_at_frequency(
        &self,
        frequency_hz: f32,
        speed_of_sound: f32,
    ) -> ScatteringResult {
        scattering::compute_scattering(self.roughness, frequency_hz, speed_of_sound)
    }

    /// Compute friction force vector opposing velocity direction.
    pub fn friction(&self, normal_force: f32, velocity: Vec3) -> Vec3 {
        friction::compute_friction_force(
            normal_force,
            velocity,
            self.friction_static,
            self.friction_kinetic,
        )
    }

    /// Compute wetting properties from contact angle and given fluid parameters.
    pub fn wetting(&self, surface_tension: f32, pore_radius: f32) -> WettingResult {
        wetting::compute_wetting(self.contact_angle, surface_tension, pore_radius)
    }

    /// Compute gas permeation through the surface boundary.
    pub fn permeation(&self, concentration_gradient: f32, dx: f32) -> PermeationResult {
        permeability::compute_permeation(
            self.permeability,
            concentration_gradient,
            self.porosity,
            dx,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_surface_interaction_from_material() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        assert!(
            (si.friction_static - mat.friction_static).abs() < EPSILON,
            "friction_static mismatch: {} vs {}",
            si.friction_static,
            mat.friction_static
        );
        assert!(
            (si.friction_kinetic - mat.friction_kinetic).abs() < EPSILON,
            "friction_kinetic mismatch"
        );
        assert!(
            (si.roughness - mat.roughness).abs() < EPSILON,
            "roughness mismatch"
        );
        assert!(
            (si.porosity - mat.porosity).abs() < EPSILON,
            "porosity mismatch"
        );
        assert!(
            (si.permeability - mat.permeability).abs() < EPSILON,
            "permeability mismatch"
        );
        assert!(
            (si.contact_angle - mat.contact_angle).abs() < EPSILON,
            "contact_angle mismatch"
        );
    }

    #[test]
    fn test_surface_interaction_scattering() {
        let mat = AcousticMaterial::default(); // concrete, roughness=0.002
        let si = SurfaceInteraction::from_material(&mat);
        let result = si.scattering_at_frequency(1000.0, 343.0);
        // Scattering result must have valid weights summing to ~1.0
        assert!(
            (result.specular_weight + result.diffuse_weight - 1.0).abs() < EPSILON,
            "Scattering weights should sum to 1.0, got {} + {} = {}",
            result.specular_weight,
            result.diffuse_weight,
            result.specular_weight + result.diffuse_weight
        );
        assert!(
            result.beckmann_width >= 0.0,
            "Beckmann width should be non-negative"
        );
        // Concrete roughness 0.002m at 1000Hz (wavelength ~0.343m): roughness << wavelength,
        // so should be mostly specular
        assert!(
            result.specular_weight > 0.5,
            "Concrete at 1000Hz should be mostly specular, got {}",
            result.specular_weight
        );
    }

    #[test]
    fn test_surface_interaction_friction() {
        let mat = AcousticMaterial::default(); // concrete: mu_s=0.6, mu_k=0.5
        let si = SurfaceInteraction::from_material(&mat);

        // Moving object: should get kinetic friction opposing velocity
        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let normal_force = 10.0;
        let force = si.friction(normal_force, velocity);

        // Force should oppose velocity (negative x)
        assert!(
            force.x < 0.0,
            "Friction should oppose x velocity, got fx={}",
            force.x
        );
        // Magnitude should be mu_k * N = 0.5 * 10 = 5.0
        assert!(
            (force.length() - 5.0).abs() < EPSILON,
            "Friction magnitude should be 5.0, got {}",
            force.length()
        );

        // Stationary object: friction force should be zero (no direction to oppose)
        let force_static = si.friction(normal_force, Vec3::ZERO);
        assert!(
            force_static.length() < EPSILON,
            "Static friction with zero velocity should return zero vector, got length={}",
            force_static.length()
        );
    }

    #[test]
    fn test_surface_interaction_default() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);

        // Scattering should produce finite results
        let scat = si.scattering_at_frequency(1000.0, 343.0);
        assert!(scat.specular_weight.is_finite());
        assert!(scat.diffuse_weight.is_finite());

        // Friction should produce finite results
        let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(fric.is_finite());

        // Wetting should produce finite results
        let wet = si.wetting(0.072, 1e-3);
        assert!(wet.surface_energy.is_finite());
        assert!(wet.capillary_pressure.is_finite());

        // Permeation should produce finite results
        let perm = si.permeation(100.0, 0.01);
        assert!(perm.flux.is_finite());
        assert!(perm.effective_permeability.is_finite());

        // Default material values should be sensible
        assert!(si.friction_static >= si.friction_kinetic);
        assert!(si.roughness >= 0.0);
        assert!(si.porosity >= 0.0 && si.porosity <= 1.0);
        assert!(si.permeability >= 0.0);
        assert!(si.contact_angle >= 0.0 && si.contact_angle <= std::f32::consts::PI);
    }

    // -----------------------------------------------------------------------
    // Task 8: Surface Physics Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_integration_concrete_surface() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        let concrete = lib.materials.get("Concrete").unwrap();
        let si = SurfaceInteraction::from_material(concrete);

        // Scattering: must produce valid weights summing to 1.0
        let scat = si.scattering_at_frequency(1000.0, 343.0);
        assert!(
            scat.specular_weight.is_finite(),
            "specular_weight not finite"
        );
        assert!(scat.diffuse_weight.is_finite(), "diffuse_weight not finite");
        assert!(
            (scat.specular_weight + scat.diffuse_weight - 1.0).abs() < EPSILON,
            "Scattering weights must sum to 1.0, got {}",
            scat.specular_weight + scat.diffuse_weight
        );

        // Friction: moving object produces finite opposing force
        let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(fric.is_finite(), "Friction force not finite");
        assert!(
            fric.length() > 0.0,
            "Concrete friction should produce nonzero force"
        );

        // Wetting: finite surface energy and capillary pressure
        let wet = si.wetting(0.072, 1e-3);
        assert!(wet.surface_energy.is_finite(), "surface_energy not finite");
        assert!(
            wet.capillary_pressure.is_finite(),
            "capillary_pressure not finite"
        );

        // Permeation: concrete has low but nonzero porosity and permeability
        let perm = si.permeation(100.0, 0.01);
        assert!(perm.flux.is_finite(), "flux not finite");
        assert!(
            perm.effective_permeability.is_finite(),
            "effective_permeability not finite"
        );
        // Concrete has porosity=0.15, permeability=1e-15; flux should be nonzero but tiny
        assert!(
            perm.flux.abs() > 0.0,
            "Concrete should have nonzero permeation flux"
        );
    }

    #[test]
    fn test_integration_glass_smooth() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        let glass = lib.materials.get("Glass").unwrap();
        let si = SurfaceInteraction::from_material(glass);

        // Glass roughness=0.0001m is very small. At 1000 Hz (wavelength ~0.343m),
        // roughness << wavelength so scattering should be near-specular.
        let scat = si.scattering_at_frequency(1000.0, 343.0);
        assert!(
            scat.specular_weight > 0.9,
            "Glass should be near-specular at 1000Hz, specular_weight={}",
            scat.specular_weight
        );

        // Glass has porosity=0.0 and permeability=0.0: zero permeation
        let perm = si.permeation(100.0, 0.01);
        assert!(
            perm.flux.abs() < EPSILON,
            "Glass (zero porosity/permeability) should have zero permeation flux, got {}",
            perm.flux
        );
        assert!(
            perm.effective_permeability.abs() < EPSILON,
            "Glass effective_permeability should be zero, got {}",
            perm.effective_permeability
        );
    }

    #[test]
    fn test_integration_foam_porous() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        let foam = lib.materials.get("Acoustic Foam").unwrap();
        let si = SurfaceInteraction::from_material(foam);

        // Acoustic Foam: porosity=0.95, permeability=1e-9
        // Should produce significant permeation flux compared to glass or concrete
        let perm = si.permeation(100.0, 0.01);
        assert!(perm.flux.is_finite(), "Foam flux not finite");
        assert!(
            perm.flux.abs() > 1e-10,
            "Foam should have significant permeation flux, got {}",
            perm.flux
        );

        // Effective permeability should reflect high porosity
        let expected_k_eff = foam.permeability * foam.porosity;
        assert!(
            (perm.effective_permeability - expected_k_eff).abs() < 1e-15,
            "Foam effective_permeability should be k*porosity={}, got {}",
            expected_k_eff,
            perm.effective_permeability
        );

        // Compare to glass: foam flux should be orders of magnitude larger
        let glass = lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);
        let perm_glass = si_glass.permeation(100.0, 0.01);
        assert!(
            perm.flux.abs() > perm_glass.flux.abs() + EPSILON,
            "Foam flux ({}) should exceed glass flux ({})",
            perm.flux,
            perm_glass.flux
        );
    }

    #[test]
    fn test_integration_frequency_sweep() {
        // Same roughness, different frequencies should show frequency dependence.
        // Low frequency (long wavelength) -> more specular.
        // High frequency (short wavelength) -> more diffuse.
        let roughness = 0.002; // concrete-like roughness
        let mat = AcousticMaterial {
            roughness,
            ..AcousticMaterial::default()
        };
        let si = SurfaceInteraction::from_material(&mat);

        let scat_125 = si.scattering_at_frequency(125.0, 343.0);
        let scat_4000 = si.scattering_at_frequency(4000.0, 343.0);

        // At 125 Hz, wavelength = 343/125 = 2.744m; roughness/wavelength tiny -> specular
        // At 4000 Hz, wavelength = 343/4000 = 0.086m; roughness/wavelength larger -> more diffuse
        assert!(
            scat_125.specular_weight > scat_4000.specular_weight,
            "125Hz should be more specular than 4000Hz: {} vs {}",
            scat_125.specular_weight,
            scat_4000.specular_weight
        );
        assert!(
            scat_125.diffuse_weight < scat_4000.diffuse_weight,
            "125Hz should be less diffuse than 4000Hz: {} vs {}",
            scat_125.diffuse_weight,
            scat_4000.diffuse_weight
        );

        // Both must still sum to 1.0
        assert!(
            (scat_125.specular_weight + scat_125.diffuse_weight - 1.0).abs() < EPSILON,
            "125Hz weights must sum to 1.0"
        );
        assert!(
            (scat_4000.specular_weight + scat_4000.diffuse_weight - 1.0).abs() < EPSILON,
            "4000Hz weights must sum to 1.0"
        );
    }

    #[test]
    fn test_integration_friction_transition() {
        // Velocity sweep: zero velocity -> static friction, nonzero -> kinetic friction.
        let mat = AcousticMaterial::default(); // concrete: mu_s=0.6, mu_k=0.5
        let normal_force = 10.0;

        // Use the friction module's compute_friction directly for is_static check
        let result_static = friction::compute_friction(
            normal_force,
            0.0,
            mat.friction_static,
            mat.friction_kinetic,
        );
        assert!(
            result_static.is_static,
            "velocity=0 should give static friction"
        );
        assert!(
            (result_static.force_magnitude - mat.friction_static * normal_force).abs() < EPSILON,
            "Static friction magnitude should be mu_s * N = {}, got {}",
            mat.friction_static * normal_force,
            result_static.force_magnitude
        );

        let result_kinetic = friction::compute_friction(
            normal_force,
            1.0,
            mat.friction_static,
            mat.friction_kinetic,
        );
        assert!(
            !result_kinetic.is_static,
            "velocity=1.0 should give kinetic friction"
        );
        assert!(
            (result_kinetic.force_magnitude - mat.friction_kinetic * normal_force).abs() < EPSILON,
            "Kinetic friction magnitude should be mu_k * N = {}, got {}",
            mat.friction_kinetic * normal_force,
            result_kinetic.force_magnitude
        );

        // Static friction magnitude >= kinetic friction magnitude
        assert!(
            result_static.force_magnitude >= result_kinetic.force_magnitude,
            "Static friction ({}) should be >= kinetic friction ({})",
            result_static.force_magnitude,
            result_kinetic.force_magnitude
        );
    }

    #[test]
    fn test_integration_all_presets_valid() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        assert!(
            !lib.materials.is_empty(),
            "MaterialLibrary should have at least one preset"
        );

        for (name, mat) in &lib.materials {
            let si = SurfaceInteraction::from_material(mat);

            // Scattering at multiple frequencies
            for &freq in &[125.0_f32, 500.0, 1000.0, 4000.0] {
                let scat = si.scattering_at_frequency(freq, 343.0);
                assert!(
                    scat.specular_weight.is_finite() && !scat.specular_weight.is_nan(),
                    "{name} specular_weight not finite at {freq}Hz"
                );
                assert!(
                    scat.diffuse_weight.is_finite() && !scat.diffuse_weight.is_nan(),
                    "{name} diffuse_weight not finite at {freq}Hz"
                );
                assert!(
                    scat.beckmann_width.is_finite() && !scat.beckmann_width.is_nan(),
                    "{name} beckmann_width not finite at {freq}Hz"
                );
                assert!(
                    (scat.specular_weight + scat.diffuse_weight - 1.0).abs() < EPSILON,
                    "{name} scattering weights don't sum to 1.0 at {freq}Hz"
                );
            }

            // Friction: both zero and nonzero velocity
            let fric_moving = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
            assert!(
                fric_moving.is_finite(),
                "{name} friction force not finite for moving object"
            );

            let fric_static = si.friction(10.0, Vec3::ZERO);
            assert!(
                fric_static.is_finite(),
                "{name} friction force not finite for static object"
            );

            // Wetting
            let wet = si.wetting(0.072, 1e-3);
            assert!(
                wet.surface_energy.is_finite() && !wet.surface_energy.is_nan(),
                "{name} surface_energy not finite"
            );
            assert!(
                wet.capillary_pressure.is_finite() && !wet.capillary_pressure.is_nan(),
                "{name} capillary_pressure not finite"
            );

            // Permeation
            let perm = si.permeation(100.0, 0.01);
            assert!(
                perm.flux.is_finite() && !perm.flux.is_nan(),
                "{name} flux not finite"
            );
            assert!(
                perm.effective_permeability.is_finite() && !perm.effective_permeability.is_nan(),
                "{name} effective_permeability not finite"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge case tests for SurfaceInteraction facade
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_surface_interaction_from_each_preset() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        let expected_names = [
            "Concrete",
            "Glass",
            "Carpet",
            "Drywall",
            "Wood Panel",
            "Acoustic Foam",
        ];

        for name in &expected_names {
            let mat = lib
                .materials
                .get(*name)
                .expect(&format!("Missing preset: {name}"));
            let si = SurfaceInteraction::from_material(mat);

            // Verify all fields transferred correctly
            assert!(
                (si.friction_static - mat.friction_static).abs() < EPSILON,
                "{name}: friction_static mismatch"
            );
            assert!(
                (si.friction_kinetic - mat.friction_kinetic).abs() < EPSILON,
                "{name}: friction_kinetic mismatch"
            );
            assert!(
                (si.roughness - mat.roughness).abs() < EPSILON,
                "{name}: roughness mismatch"
            );
            assert!(
                (si.porosity - mat.porosity).abs() < EPSILON,
                "{name}: porosity mismatch"
            );
            assert!(
                (si.permeability - mat.permeability).abs() < 1e-20,
                "{name}: permeability mismatch"
            );
            assert!(
                (si.contact_angle - mat.contact_angle).abs() < EPSILON,
                "{name}: contact_angle mismatch"
            );
        }
    }

    #[test]
    fn test_edge_surface_interaction_scattering_zero_frequency() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        // Zero frequency: clamped to 1.0 in compute_scattering
        let scat = si.scattering_at_frequency(0.0, 343.0);
        assert!(
            scat.specular_weight.is_finite(),
            "Zero frequency scattering should be finite"
        );
        assert!(
            (scat.specular_weight + scat.diffuse_weight - 1.0).abs() < EPSILON,
            "Weights should sum to 1.0"
        );
    }

    #[test]
    fn test_edge_surface_interaction_scattering_negative_speed() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        let scat = si.scattering_at_frequency(1000.0, -343.0);
        assert!(
            scat.specular_weight.is_finite(),
            "Negative speed_of_sound scattering should be finite"
        );
    }

    #[test]
    fn test_edge_surface_interaction_friction_negative_normal() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        let force = si.friction(-10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            force.length() < EPSILON,
            "Negative normal force should give zero friction via facade"
        );
    }

    #[test]
    fn test_edge_surface_interaction_wetting_zero_tension() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        let wet = si.wetting(0.0, 1e-3);
        assert!(
            wet.surface_energy.abs() < EPSILON,
            "Zero tension wetting should give zero surface_energy"
        );
        assert!(
            wet.capillary_pressure.abs() < EPSILON,
            "Zero tension wetting should give zero capillary_pressure"
        );
    }

    #[test]
    fn test_edge_surface_interaction_permeation_zero_dx() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);
        let perm = si.permeation(100.0, 0.0);
        assert!(
            perm.flux.is_finite(),
            "Zero dx permeation should be finite (clamped)"
        );
    }

    #[test]
    fn test_edge_surface_interaction_all_zero_material() {
        // Construct a material with all surface properties at zero
        let mat = AcousticMaterial {
            friction_static: 0.0,
            friction_kinetic: 0.0,
            roughness: 0.0,
            porosity: 0.0,
            permeability: 0.0,
            contact_angle: 0.0,
            ..AcousticMaterial::default()
        };
        let si = SurfaceInteraction::from_material(&mat);

        // Scattering: roughness=0 -> specular
        let scat = si.scattering_at_frequency(1000.0, 343.0);
        assert!(
            (scat.specular_weight - 1.0).abs() < EPSILON,
            "Zero roughness should be fully specular"
        );

        // Friction: zero coefficients -> zero force
        let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            fric.length() < EPSILON,
            "Zero friction coefficients should give zero force"
        );

        // Wetting: angle=0 -> hydrophilic, surface_energy = surface_tension
        let wet = si.wetting(0.072, 1e-3);
        assert!(wet.is_hydrophilic, "contact_angle=0 should be hydrophilic");

        // Permeation: porosity=0 and permeability=0 -> zero flux
        let perm = si.permeation(100.0, 0.01);
        assert!(
            perm.flux.abs() < EPSILON,
            "Zero porosity/permeability should give zero flux"
        );
    }

    #[test]
    fn test_edge_surface_interaction_max_roughness_material() {
        let mat = AcousticMaterial {
            roughness: 1.0,
            porosity: 1.0,
            permeability: 1.0,
            friction_static: 1.0,
            friction_kinetic: 1.0,
            contact_angle: std::f32::consts::PI,
            ..AcousticMaterial::default()
        };
        let si = SurfaceInteraction::from_material(&mat);

        // All outputs should be finite
        let scat = si.scattering_at_frequency(1000.0, 343.0);
        assert!(scat.specular_weight.is_finite());
        assert!(scat.diffuse_weight.is_finite());

        let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(fric.is_finite());

        let wet = si.wetting(0.072, 1e-3);
        assert!(wet.surface_energy.is_finite());
        assert!(wet.capillary_pressure.is_finite());

        let perm = si.permeation(100.0, 0.01);
        assert!(perm.flux.is_finite());
    }

    #[test]
    fn test_edge_carpet_high_friction_and_porosity() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        let carpet = lib.materials.get("Carpet").unwrap();
        let si = SurfaceInteraction::from_material(carpet);

        // Carpet has high friction: mu_s=0.8, mu_k=0.6
        let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (fric.length() - 6.0).abs() < EPSILON,
            "Carpet kinetic friction should be mu_k*N=6.0, got {}",
            fric.length()
        );

        // Carpet has high porosity=0.6 -> significant permeation
        let perm = si.permeation(100.0, 0.01);
        assert!(
            perm.flux > 0.0,
            "Carpet should have positive permeation flux"
        );

        // Carpet is hydrophobic (contact_angle=1.92 > pi/2)
        let wet = si.wetting(0.072, 1e-3);
        assert!(
            !wet.is_hydrophilic,
            "Carpet (synthetic fiber) should be hydrophobic"
        );
    }

    #[test]
    fn test_edge_invariant_friction_static_ge_kinetic_all_presets() {
        use crate::scene::material::MaterialLibrary;

        let lib = MaterialLibrary::with_defaults();
        for (name, mat) in &lib.materials {
            let si = SurfaceInteraction::from_material(mat);
            assert!(
                si.friction_static >= si.friction_kinetic,
                "{name}: friction_static ({}) < friction_kinetic ({}) -- invariant violated",
                si.friction_static,
                si.friction_kinetic
            );
        }
    }
}
