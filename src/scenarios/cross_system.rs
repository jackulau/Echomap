//! Cross-system integration tests.
//!
//! Verifies that acoustic+surface, fluid+gas, and material library interactions
//! compose correctly without producing NaN/Inf or inconsistent results.

#[cfg(test)]
mod tests {
    use crate::scene::material::MaterialLibrary;
    use crate::surface::SurfaceInteraction;
    use glam::Vec3;

    const EPSILON: f32 = 1e-6;

    // -----------------------------------------------------------------
    // 1. Acoustic + surface roughness
    // -----------------------------------------------------------------

    #[test]
    fn test_acoustic_plus_surface_roughness() {
        let lib = MaterialLibrary::with_defaults();

        // Glass: roughness = 0.0001 (very smooth)
        let glass = lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);

        // Carpet: roughness = 0.005 (rough)
        let carpet = lib.materials.get("Carpet").unwrap();
        let si_carpet = SurfaceInteraction::from_material(carpet);

        let scat_glass = si_glass.scattering_at_frequency(1000.0, 343.0);
        let scat_carpet = si_carpet.scattering_at_frequency(1000.0, 343.0);

        // Rougher surface -> more diffuse (lower specular_fraction)
        assert!(
            scat_glass.specular_weight > scat_carpet.specular_weight,
            "Glass (roughness={}) should be more specular than Carpet (roughness={}) at 1000Hz: {} vs {}",
            glass.roughness,
            carpet.roughness,
            scat_glass.specular_weight,
            scat_carpet.specular_weight
        );

        // Both should have weights summing to 1.0
        assert!(
            (scat_glass.specular_weight + scat_glass.diffuse_weight - 1.0).abs() < EPSILON,
            "Glass scattering weights should sum to 1.0"
        );
        assert!(
            (scat_carpet.specular_weight + scat_carpet.diffuse_weight - 1.0).abs() < EPSILON,
            "Carpet scattering weights should sum to 1.0"
        );
    }

    // -----------------------------------------------------------------
    // 2. Fluid viscosity affects drag (analytical)
    // -----------------------------------------------------------------

    #[test]
    fn test_fluid_viscosity_affects_drag() {
        use crate::benchmarks::analytical::stokes_drag;

        let radius = 0.001_f64; // 1mm sphere
        let velocity = 0.01_f64; // 0.01 m/s

        // Water: mu = 0.001 Pa-s
        let mu_water = 0.001_f64;
        let drag_water = stokes_drag(mu_water, radius, velocity);

        // Honey-like: mu = 10.0 Pa-s
        let mu_honey = 10.0_f64;
        let drag_honey = stokes_drag(mu_honey, radius, velocity);

        // Higher viscosity -> higher drag
        assert!(
            drag_honey > drag_water,
            "Higher viscosity should produce higher drag: honey={} vs water={}",
            drag_honey,
            drag_water
        );

        // Stokes drag is linear in viscosity: ratio should match viscosity ratio
        let drag_ratio = drag_honey / drag_water;
        let mu_ratio = mu_honey / mu_water;
        assert!(
            (drag_ratio - mu_ratio).abs() < EPSILON as f64,
            "Drag ratio ({}) should equal viscosity ratio ({})",
            drag_ratio,
            mu_ratio
        );

        // Verify exact formula: F = 6*pi*mu*r*v
        let expected_water = 6.0 * std::f64::consts::PI * mu_water * radius * velocity;
        assert!(
            (drag_water - expected_water).abs() < 1e-15,
            "Stokes drag for water: expected {}, got {}",
            expected_water,
            drag_water
        );
    }

    // -----------------------------------------------------------------
    // 3. Gas diffusion + permeability
    // -----------------------------------------------------------------

    #[test]
    fn test_gas_diffusion_plus_permeability() {
        let lib = MaterialLibrary::with_defaults();

        // Acoustic Foam: porosity=0.95, permeability=1e-9 (permeable)
        let foam = lib.materials.get("Acoustic Foam").unwrap();
        let si_foam = SurfaceInteraction::from_material(foam);

        // Glass: porosity=0.0, permeability=0.0 (impermeable)
        let glass = lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);

        let concentration_gradient = 100.0_f32; // positive gradient
        let dx = 0.01_f32;

        let perm_foam = si_foam.permeation(concentration_gradient, dx);
        let perm_glass = si_glass.permeation(concentration_gradient, dx);

        // Permeable material: positive gradient -> positive flux
        assert!(
            perm_foam.flux > 0.0,
            "Permeable foam should have positive flux for positive gradient, got {}",
            perm_foam.flux
        );
        assert!(
            perm_foam.flux.is_finite(),
            "Foam flux should be finite, got {}",
            perm_foam.flux
        );

        // Impermeable material: zero flux
        assert!(
            perm_glass.flux.abs() < EPSILON,
            "Impermeable glass should have zero flux, got {}",
            perm_glass.flux
        );
    }

    // -----------------------------------------------------------------
    // 4. All materials produce finite results
    // -----------------------------------------------------------------

    #[test]
    fn test_all_materials_produce_finite_results() {
        let lib = MaterialLibrary::with_defaults();
        assert!(
            !lib.materials.is_empty(),
            "MaterialLibrary should have at least one preset"
        );

        for (name, mat) in &lib.materials {
            let si = SurfaceInteraction::from_material(mat);

            // Friction with nonzero velocity
            let fric = si.friction(10.0, Vec3::new(1.0, 0.0, 0.0));
            assert!(
                fric.is_finite(),
                "{name}: friction force is not finite: {:?}",
                fric
            );
            assert!(
                !fric.x.is_nan() && !fric.y.is_nan() && !fric.z.is_nan(),
                "{name}: friction force contains NaN"
            );

            // Wetting
            let wet = si.wetting(0.072, 1e-3);
            assert!(
                wet.surface_energy.is_finite() && !wet.surface_energy.is_nan(),
                "{name}: surface_energy not finite"
            );
            assert!(
                wet.capillary_pressure.is_finite() && !wet.capillary_pressure.is_nan(),
                "{name}: capillary_pressure not finite"
            );

            // Permeation
            let perm = si.permeation(100.0, 0.01);
            assert!(
                perm.flux.is_finite() && !perm.flux.is_nan(),
                "{name}: permeation flux not finite"
            );
            assert!(
                perm.effective_permeability.is_finite() && !perm.effective_permeability.is_nan(),
                "{name}: effective_permeability not finite"
            );

            // Scattering at multiple frequencies
            for &freq in &[125.0_f32, 500.0, 1000.0, 4000.0] {
                let scat = si.scattering_at_frequency(freq, 343.0);
                assert!(
                    scat.specular_weight.is_finite() && !scat.specular_weight.is_nan(),
                    "{name}: specular_weight not finite at {freq}Hz"
                );
                assert!(
                    scat.diffuse_weight.is_finite() && !scat.diffuse_weight.is_nan(),
                    "{name}: diffuse_weight not finite at {freq}Hz"
                );
                assert!(
                    scat.beckmann_width.is_finite() && !scat.beckmann_width.is_nan(),
                    "{name}: beckmann_width not finite at {freq}Hz"
                );
                assert!(
                    (scat.specular_weight + scat.diffuse_weight - 1.0).abs() < EPSILON,
                    "{name}: scattering weights don't sum to 1.0 at {freq}Hz"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // 5. Compose fluid and gas simulations independently
    // -----------------------------------------------------------------

    #[test]
    fn test_scenario_compose_fluid_and_gas() {
        use crate::fluids::FluidSimulation;
        use crate::gas::grid::GasSpecies;
        use crate::gas::GasSimulation;

        // Create both simulations with default configs
        let mut fluid_sim = FluidSimulation::default();
        let mut gas_sim = GasSimulation::default();

        let bounds = (Vec3::ZERO, Vec3::new(2.0, 2.0, 2.0));
        let resolution = 1.0;

        // Initialize fluid
        fluid_sim.initialize(bounds, resolution, &[]);

        // Initialize gas with a test species
        let species = vec![GasSpecies {
            name: "CO2".to_string(),
            diffusion_coefficient: 0.16,
            molecular_weight: 44.0,
            density_at_stp: 1.842,
            color: [1.0, 0.0, 0.0],
        }];
        gas_sim.initialize(bounds, resolution, species, &[]);

        // Step each independently
        fluid_sim.step();
        gas_sim.step();

        // Both should advance frame
        assert_eq!(
            fluid_sim.frame, 1,
            "Fluid simulation should advance to frame 1"
        );
        assert_eq!(gas_sim.frame, 1, "Gas simulation should advance to frame 1");

        // Fluid grid values should be finite (no NaN)
        let fluid_grid = fluid_sim.grid.as_ref().unwrap();
        assert!(
            fluid_grid.pressure.iter().all(|v| v.is_finite()),
            "Fluid pressure should be finite after step"
        );
        assert!(
            fluid_grid.u.iter().all(|v| v.is_finite()),
            "Fluid velocity u should be finite after step"
        );
        assert!(
            fluid_grid.v.iter().all(|v| v.is_finite()),
            "Fluid velocity v should be finite after step"
        );

        // Gas grid values should be finite (no NaN)
        let gas_grid = gas_sim.grid.as_ref().unwrap();
        assert!(
            gas_grid.temperature.iter().all(|v| v.is_finite()),
            "Gas temperature should be finite after step"
        );
        for (i, conc) in gas_grid.concentrations.iter().enumerate() {
            assert!(
                conc.iter().all(|v| v.is_finite()),
                "Gas concentration[{}] should be finite after step",
                i
            );
        }

        // Step a few more times to check stability
        for _ in 0..5 {
            fluid_sim.step();
            gas_sim.step();
        }

        assert_eq!(fluid_sim.frame, 6);
        assert_eq!(gas_sim.frame, 6);

        // Still finite after multiple steps
        let fluid_grid = fluid_sim.grid.as_ref().unwrap();
        assert!(
            fluid_grid.pressure.iter().all(|v| v.is_finite()),
            "Fluid pressure should remain finite after 6 steps"
        );

        let gas_grid = gas_sim.grid.as_ref().unwrap();
        assert!(
            gas_grid.temperature.iter().all(|v| v.is_finite()),
            "Gas temperature should remain finite after 6 steps"
        );
    }
}
