//! Surface physics validation tests.
//!
//! Verifies Coulomb friction, Beckmann scattering frequency dependence,
//! wetting classification, and permeability contrast across materials.

#[cfg(test)]
mod tests {
    use crate::scene::material::{AcousticMaterial, MaterialLibrary};
    use crate::surface::SurfaceInteraction;
    use glam::Vec3;

    const EPSILON: f32 = 1e-6;

    // -----------------------------------------------------------------
    // 1. Coulomb friction — Concrete
    // -----------------------------------------------------------------

    #[test]
    fn test_coulomb_friction_concrete() {
        let lib = MaterialLibrary::with_defaults();
        let concrete = lib.materials.get("Concrete").unwrap();
        let si = SurfaceInteraction::from_material(concrete);

        // Concrete: friction_kinetic = 0.5
        let normal_force = 100.0_f32;
        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let force = si.friction(normal_force, velocity);

        // Force should oppose velocity (negative x)
        assert!(
            force.x < 0.0,
            "Friction should oppose velocity, got fx={}",
            force.x
        );

        // Magnitude should be mu_k * N = 0.5 * 100 = 50.0
        let expected = concrete.friction_kinetic * normal_force;
        assert!(
            (force.length() - expected).abs() < EPSILON,
            "Concrete kinetic friction magnitude: expected {}, got {}",
            expected,
            force.length()
        );
    }

    // -----------------------------------------------------------------
    // 2. Coulomb friction — Glass
    // -----------------------------------------------------------------

    #[test]
    fn test_coulomb_friction_glass() {
        let lib = MaterialLibrary::with_defaults();
        let glass = lib.materials.get("Glass").unwrap();
        let si = SurfaceInteraction::from_material(glass);

        // Glass: friction_kinetic = 0.3
        let normal_force = 100.0_f32;
        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let force = si.friction(normal_force, velocity);

        // Magnitude should be mu_k * N = 0.3 * 100 = 30.0
        let expected = glass.friction_kinetic * normal_force;
        assert!(
            (force.length() - expected).abs() < EPSILON,
            "Glass kinetic friction magnitude: expected {}, got {}",
            expected,
            force.length()
        );

        // Glass friction should differ from concrete friction
        let concrete = lib.materials.get("Concrete").unwrap();
        assert!(
            (glass.friction_kinetic - concrete.friction_kinetic).abs() > EPSILON,
            "Glass and Concrete should have different friction coefficients"
        );
    }

    // -----------------------------------------------------------------
    // 3. Beckmann frequency dependence
    // -----------------------------------------------------------------

    #[test]
    fn test_beckmann_frequency_dependence() {
        let mat = AcousticMaterial::default(); // roughness = 0.002
        let si = SurfaceInteraction::from_material(&mat);

        let scat_low = si.scattering_at_frequency(125.0, 343.0);
        let scat_high = si.scattering_at_frequency(4000.0, 343.0);

        // Low frequency (long wavelength) -> surface appears smoother -> more specular
        // High frequency (short wavelength) -> more diffuse
        assert!(
            scat_low.specular_weight > scat_high.specular_weight,
            "125Hz should be more specular than 4000Hz: {} vs {}",
            scat_low.specular_weight,
            scat_high.specular_weight
        );

        // Both must still have weights summing to 1.0
        assert!(
            (scat_low.specular_weight + scat_low.diffuse_weight - 1.0).abs() < EPSILON,
            "125Hz weights should sum to 1.0"
        );
        assert!(
            (scat_high.specular_weight + scat_high.diffuse_weight - 1.0).abs() < EPSILON,
            "4000Hz weights should sum to 1.0"
        );
    }

    // -----------------------------------------------------------------
    // 4. Wetting: hydrophilic vs hydrophobic
    // -----------------------------------------------------------------

    #[test]
    fn test_wetting_hydrophilic_vs_hydrophobic() {
        // Glass: contact_angle = 0.52 rad (~30 degrees) — hydrophilic
        let lib = MaterialLibrary::with_defaults();
        let glass = lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);

        // Carpet: contact_angle = 1.92 rad (~110 degrees) — hydrophobic
        let carpet = lib.materials.get("Carpet").unwrap();
        let si_carpet = SurfaceInteraction::from_material(carpet);

        let surface_tension = 0.072_f32; // water
        let pore_radius = 1e-3_f32;

        let wet_glass = si_glass.wetting(surface_tension, pore_radius);
        let wet_carpet = si_carpet.wetting(surface_tension, pore_radius);

        // Hydrophilic (glass) should have positive capillary pressure
        assert!(
            wet_glass.is_hydrophilic,
            "Glass (contact_angle={}) should be hydrophilic",
            glass.contact_angle
        );
        assert!(
            wet_glass.capillary_pressure > 0.0,
            "Hydrophilic material should have positive capillary pressure, got {}",
            wet_glass.capillary_pressure
        );
        assert!(
            wet_glass.surface_energy > 0.0,
            "Hydrophilic material should have positive surface energy, got {}",
            wet_glass.surface_energy
        );

        // Hydrophobic (carpet) should have negative capillary pressure
        assert!(
            !wet_carpet.is_hydrophilic,
            "Carpet (contact_angle={}) should be hydrophobic",
            carpet.contact_angle
        );
        assert!(
            wet_carpet.capillary_pressure < 0.0,
            "Hydrophobic material should have negative capillary pressure, got {}",
            wet_carpet.capillary_pressure
        );
    }

    // -----------------------------------------------------------------
    // 5. Permeability: Carpet vs Glass
    // -----------------------------------------------------------------

    #[test]
    fn test_permeability_carpet_vs_glass() {
        let lib = MaterialLibrary::with_defaults();

        // Carpet: porosity=0.6, permeability=1e-10
        let carpet = lib.materials.get("Carpet").unwrap();
        let si_carpet = SurfaceInteraction::from_material(carpet);

        // Glass: porosity=0.0, permeability=0.0
        let glass = lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);

        let concentration_gradient = 100.0_f32;
        let dx = 0.01_f32;

        let perm_carpet = si_carpet.permeation(concentration_gradient, dx);
        let perm_glass = si_glass.permeation(concentration_gradient, dx);

        // Carpet should have nonzero permeation flux
        assert!(
            perm_carpet.flux > 0.0,
            "Carpet should have positive permeation flux, got {}",
            perm_carpet.flux
        );
        assert!(
            perm_carpet.effective_permeability > 0.0,
            "Carpet should have positive effective_permeability, got {}",
            perm_carpet.effective_permeability
        );

        // Glass should have zero/near-zero permeation
        assert!(
            perm_glass.flux.abs() < EPSILON,
            "Glass should have near-zero permeation flux, got {}",
            perm_glass.flux
        );
        assert!(
            perm_glass.effective_permeability.abs() < EPSILON,
            "Glass should have near-zero effective_permeability, got {}",
            perm_glass.effective_permeability
        );

        // Carpet flux should be strictly greater than glass flux
        assert!(
            perm_carpet.flux > perm_glass.flux,
            "Carpet flux ({}) should exceed glass flux ({})",
            perm_carpet.flux,
            perm_glass.flux
        );
    }

    // -----------------------------------------------------------------
    // 6. Zero normal force friction
    // -----------------------------------------------------------------

    #[test]
    fn test_zero_normal_force_friction() {
        let mat = AcousticMaterial::default();
        let si = SurfaceInteraction::from_material(&mat);

        let velocity = Vec3::new(1.0, 0.0, 0.0);
        let force = si.friction(0.0, velocity);

        // Zero normal force should produce zero friction, no division errors
        assert!(
            force.length() < EPSILON,
            "Zero normal force should produce zero friction vector, got length={}",
            force.length()
        );
        assert!(
            force.is_finite(),
            "Zero normal force friction should be finite, got {:?}",
            force
        );
    }
}
