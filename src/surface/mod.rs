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
}
