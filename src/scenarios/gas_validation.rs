//! Gas diffusion and permeability validation tests.
//!
//! Validates point source diffusion profiles, rate scaling, concentration
//! conservation, boundary conditions, multi-species independence,
//! temperature-driven convection, and Darcy permeability effects.

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::gas::grid::{GasCellType, GasGrid, GasSpecies};
    use crate::gas::solver::{self, GasConfig};
    use crate::scene::material::MaterialLibrary;
    use crate::surface::SurfaceInteraction;

    const EPSILON: f32 = 1e-6;

    /// Helper: create a gas species with the given name and diffusion coefficient.
    fn make_species(name: &str, diff_coeff: f32) -> GasSpecies {
        GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: diff_coeff,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

    /// Helper: create a gas grid with all cells marked as Gas.
    fn make_gas_grid(n: usize, dx: f32, species: Vec<GasSpecies>) -> GasGrid {
        let mut g = GasGrid::new(n, n, n, dx, Vec3::ZERO, species);
        for ct in g.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }
        g
    }

    // -----------------------------------------------------------------------
    // Test 1: Point source diffusion profile
    // -----------------------------------------------------------------------

    /// Inject gas at center cell, step N times. Concentration should decrease
    /// monotonically with distance from center.
    #[test]
    fn test_point_source_diffusion_profile() {
        let diff_coeff = 0.05;
        let species = vec![make_species("CO2", diff_coeff)];
        let n: usize = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Inject gas at center cell.
        let ci = n / 2;
        let center_idx = grid.idx(ci, ci, ci);
        grid.concentrations[0][center_idx] = 1000.0;

        // Run pure diffusion for many steps.
        let dt = 0.005;
        let num_steps = 100;
        for _ in 0..num_steps {
            solver::diffuse_concentrations(&mut grid, dt);
        }

        // Verify monotonic decrease with distance from center.
        let c_center = grid.concentrations[0][center_idx];
        let c_r1 = grid.concentrations[0][grid.idx(ci + 1, ci, ci)];
        let c_r2 = grid.concentrations[0][grid.idx(ci + 2, ci, ci)];
        let c_r3 = grid.concentrations[0][grid.idx(ci + 3, ci, ci)];

        assert!(
            c_center > c_r1,
            "Center ({c_center}) should be greater than r=1 ({c_r1})"
        );
        assert!(
            c_r1 > c_r2,
            "r=1 ({c_r1}) should be greater than r=2 ({c_r2})"
        );
        assert!(
            c_r2 > c_r3,
            "r=2 ({c_r2}) should be greater than r=3 ({c_r3})"
        );

        // All sampled concentrations should be positive (diffusion spreads,
        // does not create negative values).
        assert!(c_center > 0.0, "Center concentration should be positive");
        assert!(c_r3 > 0.0, "r=3 concentration should be positive");

        // Verify isotropy: concentration at same distance in different
        // directions should be approximately equal.
        let c_r2_y = grid.concentrations[0][grid.idx(ci, ci + 2, ci)];
        let c_r2_z = grid.concentrations[0][grid.idx(ci, ci, ci + 2)];
        let iso_err_y = (c_r2 - c_r2_y).abs() / c_r2.max(EPSILON);
        let iso_err_z = (c_r2 - c_r2_z).abs() / c_r2.max(EPSILON);
        assert!(
            iso_err_y < 0.05,
            "Diffusion should be isotropic in Y: c_r2_x={c_r2:.4}, c_r2_y={c_r2_y:.4}"
        );
        assert!(
            iso_err_z < 0.05,
            "Diffusion should be isotropic in Z: c_r2_x={c_r2:.4}, c_r2_z={c_r2_z:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Diffusion rate scaling
    // -----------------------------------------------------------------------

    /// Two species with 2x diffusion_coefficient difference. The faster
    /// species should spread further (higher concentration at a fixed
    /// distance from source).
    #[test]
    fn test_diffusion_rate_scaling() {
        // Use small enough dt and large enough dx so that both species
        // stay below the solver stability clamp (D*dt/dx^2 <= 1/6).
        // slow: 0.01 * 0.001 / 0.25^2 = 0.00016 (well below 1/6)
        // fast: 0.02 * 0.001 / 0.25^2 = 0.00032 (well below 1/6)
        let slow_coeff = 0.01;
        let fast_coeff = 0.02; // 2x faster
        let species = vec![
            make_species("Slow", slow_coeff),
            make_species("Fast", fast_coeff),
        ];
        let n: usize = 16;
        let dx = 0.25;
        let mut grid = make_gas_grid(n, dx, species);

        // Inject both species at center with equal initial concentration.
        let ci = n / 2;
        let center_idx = grid.idx(ci, ci, ci);
        grid.concentrations[0][center_idx] = 1000.0;
        grid.concentrations[1][center_idx] = 1000.0;

        // Run pure diffusion for many steps with a small dt.
        let dt = 0.001;
        for _ in 0..500 {
            solver::diffuse_concentrations(&mut grid, dt);
        }

        // At center, faster species should have spread out more, so its
        // center concentration should be lower.
        let c_slow_center = grid.concentrations[0][center_idx];
        let c_fast_center = grid.concentrations[1][center_idx];
        assert!(
            c_fast_center < c_slow_center,
            "Faster species should have lower center concentration: \
             fast={c_fast_center:.4}, slow={c_slow_center:.4}"
        );

        // At a fixed distance from center, the faster species should have
        // higher concentration (it spreads further).
        let probe_dist = 2; // 2 cells away
        let probe_idx = grid.idx(ci + probe_dist, ci, ci);
        let c_slow = grid.concentrations[0][probe_idx];
        let c_fast = grid.concentrations[1][probe_idx];

        assert!(
            c_fast > c_slow,
            "Faster species should have higher concentration at distance {probe_dist}: \
             fast={c_fast:.6}, slow={c_slow:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: Concentration conservation
    // -----------------------------------------------------------------------

    /// Inject a known amount of gas. After N steps, total concentration
    /// (sum over all cells) should be approximately conserved (within 5%).
    #[test]
    fn test_concentration_conservation() {
        let species = vec![make_species("CO2", 0.05)];
        let n: usize = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Inject gas at center.
        let ci = n / 2;
        let center_idx = grid.idx(ci, ci, ci);
        grid.concentrations[0][center_idx] = 500.0;

        let mass_before: f64 = grid.concentrations[0].iter().map(|&v| v as f64).sum();

        // Use a config with no buoyancy/thermal effects so mass is truly
        // conserved (only diffusion active, no advection without velocity).
        let config = GasConfig {
            dt: 0.001,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.0,
            gravity: Vec3::ZERO,
        };

        for _ in 0..200 {
            solver::step(&mut grid, &config);
        }

        let mass_after: f64 = grid.concentrations[0].iter().map(|&v| v as f64).sum();

        let rel_change = if mass_before > 0.0 {
            ((mass_after - mass_before) / mass_before).abs()
        } else {
            0.0
        };

        assert!(
            rel_change < 0.05,
            "Total concentration should be conserved within 5%: \
             before={mass_before:.4}, after={mass_after:.4}, rel_change={rel_change:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: No flux boundary (solid wall blocks gas)
    // -----------------------------------------------------------------------

    /// Gas should not leak through solid walls. Inject gas on one side of a
    /// solid wall, step, verify no concentration passes through.
    #[test]
    fn test_no_flux_boundary() {
        use crate::gas::boundary::enforce_boundary_conditions;

        let species = vec![make_species("CO2", 0.05)];
        let n: usize = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Place a 2-cell-thick solid wall at i = n/2 and i = n/2 + 1
        // spanning the full y-z plane. Two cells thick ensures the
        // diffusion stencil cannot bridge across.
        let wall_i0 = n / 2;
        let wall_i1 = wall_i0 + 1;
        for k in 0..n {
            for j in 0..n {
                let idx0 = grid.idx(wall_i0, j, k);
                let idx1 = grid.idx(wall_i1, j, k);
                grid.cell_types[idx0] = GasCellType::Solid;
                grid.cell_types[idx1] = GasCellType::Solid;
            }
        }

        // Fill the left side with high concentration.
        for k in 0..n {
            for j in 0..n {
                for i in 0..wall_i0 {
                    let idx = grid.idx(i, j, k);
                    grid.concentrations[0][idx] = 100.0;
                }
            }
        }

        // Run diffusion + boundary enforcement.
        let dt = 0.005;
        for _ in 0..200 {
            solver::diffuse_concentrations(&mut grid, dt);
            enforce_boundary_conditions(&mut grid);
        }

        // All Gas cells to the right of the wall should have negligible
        // concentration.
        let mut max_right = 0.0_f32;
        for k in 0..n {
            for j in 0..n {
                for i in (wall_i1 + 1)..n {
                    let idx = grid.idx(i, j, k);
                    if grid.cell_types[idx] == GasCellType::Gas {
                        max_right = max_right.max(grid.concentrations[0][idx]);
                    }
                }
            }
        }

        assert!(
            max_right < 1.0,
            "Concentration should not pass through solid wall: max on right side={max_right:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: Multi-species independence
    // -----------------------------------------------------------------------

    /// Inject two different species at different locations. Verify they
    /// diffuse independently without cross-contamination.
    #[test]
    fn test_multi_species_independence() {
        let species = vec![
            make_species("SpeciesA", 0.05),
            make_species("SpeciesB", 0.05),
        ];
        let n: usize = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Inject species A at one corner, species B at the opposite corner.
        let pos_a = (2, 2, 2);
        let pos_b = (n - 3, n - 3, n - 3);
        let idx_a = grid.idx(pos_a.0, pos_a.1, pos_a.2);
        let idx_b = grid.idx(pos_b.0, pos_b.1, pos_b.2);

        grid.concentrations[0][idx_a] = 500.0; // species A
        grid.concentrations[1][idx_b] = 500.0; // species B

        // Run pure diffusion for many steps.
        let dt = 0.005;
        for _ in 0..100 {
            solver::diffuse_concentrations(&mut grid, dt);
        }

        // Species A concentration at species B source should be ~0.
        let a_at_b = grid.concentrations[0][idx_b];
        assert!(
            a_at_b < 1.0,
            "Species A at species B source should be near zero, got {a_at_b:.4}"
        );

        // Species B concentration at species A source should be ~0.
        let b_at_a = grid.concentrations[1][idx_a];
        assert!(
            b_at_a < 1.0,
            "Species B at species A source should be near zero, got {b_at_a:.4}"
        );

        // Each species should have diffused around its own source.
        let a_near_source = grid.concentrations[0][grid.idx(pos_a.0 + 1, pos_a.1, pos_a.2)];
        let b_near_source = grid.concentrations[1][grid.idx(pos_b.0 - 1, pos_b.1, pos_b.2)];
        assert!(
            a_near_source > 0.0,
            "Species A should have diffused near its source: {a_near_source:.4}"
        );
        assert!(
            b_near_source > 0.0,
            "Species B should have diffused near its source: {b_near_source:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Temperature-driven convection
    // -----------------------------------------------------------------------

    /// Set a temperature perturbation (hot region). After stepping, verify
    /// that gas concentration is influenced by temperature-driven convection
    /// (concentration moves upward in hot region).
    #[test]
    fn test_temperature_driven_convection() {
        let species = vec![make_species("CO2", 0.01)];
        let n: usize = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        let config = GasConfig {
            dt: 0.005,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.01,
            buoyancy_coefficient: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        // Set all cells to ambient temperature.
        for t in grid.temperature.iter_mut() {
            *t = config.ambient_temperature;
        }

        // Create a hot spot at the bottom (j=0..2) in the center region.
        for k in 4..12 {
            for i in 4..12 {
                for j in 0..2 {
                    let idx = grid.idx(i, j, k);
                    grid.temperature[idx] = 500.0;
                }
            }
        }

        // Place gas concentration at the bottom where it is hot.
        for k in 4..12 {
            for i in 4..12 {
                for j in 0..3 {
                    let idx = grid.idx(i, j, k);
                    grid.concentrations[0][idx] = 100.0;
                }
            }
        }

        // Snapshot concentration profile: measure concentration above the
        // hot spot before stepping.
        let mut upper_conc_before = 0.0_f64;
        for k in 6..10 {
            for i in 6..10 {
                for j in 6..10 {
                    let idx = grid.idx(i, j, k);
                    upper_conc_before += grid.concentrations[0][idx] as f64;
                }
            }
        }

        // Run full simulation steps (buoyancy drives hot gas upward).
        for _ in 0..50 {
            solver::step(&mut grid, &config);
        }

        // Measure concentration in the upper region after stepping.
        let mut upper_conc_after = 0.0_f64;
        for k in 6..10 {
            for i in 6..10 {
                for j in 6..10 {
                    let idx = grid.idx(i, j, k);
                    upper_conc_after += grid.concentrations[0][idx] as f64;
                }
            }
        }

        // Buoyancy should have transported gas upward, increasing
        // concentration in the upper region.
        assert!(
            upper_conc_after > upper_conc_before,
            "Temperature-driven convection should move gas upward: \
             before={upper_conc_before:.4}, after={upper_conc_after:.4}"
        );

        // Also verify upward velocity developed above the hot spot.
        let mut total_vy = 0.0_f64;
        let mut count = 0;
        for k in 6..10 {
            for i in 6..10 {
                for j in 4..8 {
                    let idx = grid.idx(i, j, k);
                    total_vy += grid.vel_y[idx] as f64;
                    count += 1;
                }
            }
        }
        let avg_vy = total_vy / count as f64;
        assert!(
            avg_vy > 0.0,
            "Hot bottom should produce upward velocity: avg_vy={avg_vy:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Darcy permeability effect
    // -----------------------------------------------------------------------

    /// Test gas permeation through permeable surfaces. Permeable materials
    /// should allow gas flow while impermeable ones block it.
    #[test]
    fn test_darcy_permeability_effect() {
        let mat_lib = MaterialLibrary::with_defaults();

        // Glass: porosity=0.0, permeability=0.0 -> impermeable
        let glass = mat_lib.materials.get("Glass").unwrap();
        let si_glass = SurfaceInteraction::from_material(glass);

        // Acoustic Foam: porosity=0.95, permeability=1e-9 -> highly permeable
        let foam = mat_lib.materials.get("Acoustic Foam").unwrap();
        let si_foam = SurfaceInteraction::from_material(foam);

        // Concrete: porosity=0.15, permeability=1e-15 -> low permeability
        let concrete = mat_lib.materials.get("Concrete").unwrap();
        let si_concrete = SurfaceInteraction::from_material(concrete);

        let concentration_gradient = 100.0;
        let dx = 0.01;

        // Compute permeation for each material.
        let perm_glass = si_glass.permeation(concentration_gradient, dx);
        let perm_foam = si_foam.permeation(concentration_gradient, dx);
        let perm_concrete = si_concrete.permeation(concentration_gradient, dx);

        // Glass (impermeable) should have zero flux.
        assert!(
            perm_glass.flux.abs() < EPSILON,
            "Glass should have zero permeation flux, got {:.6e}",
            perm_glass.flux
        );
        assert!(
            perm_glass.effective_permeability.abs() < EPSILON,
            "Glass should have zero effective permeability, got {:.6e}",
            perm_glass.effective_permeability
        );

        // Foam (highly permeable) should have significant flux.
        assert!(
            perm_foam.flux > 0.0,
            "Acoustic Foam should have positive permeation flux, got {:.6e}",
            perm_foam.flux
        );
        assert!(
            perm_foam.effective_permeability > 0.0,
            "Acoustic Foam should have positive effective permeability, got {:.6e}",
            perm_foam.effective_permeability
        );

        // Concrete (low permeability) should have nonzero but small flux.
        assert!(
            perm_concrete.flux > 0.0,
            "Concrete should have positive permeation flux, got {:.6e}",
            perm_concrete.flux
        );

        // Foam flux should be much larger than concrete flux.
        assert!(
            perm_foam.flux > perm_concrete.flux * 1000.0,
            "Foam flux ({:.6e}) should be >> concrete flux ({:.6e})",
            perm_foam.flux,
            perm_concrete.flux
        );

        // Verify ordering: foam > concrete > glass.
        assert!(
            perm_foam.effective_permeability > perm_concrete.effective_permeability,
            "Foam effective_k ({:.6e}) should be > concrete ({:.6e})",
            perm_foam.effective_permeability,
            perm_concrete.effective_permeability
        );
        assert!(
            perm_concrete.effective_permeability > perm_glass.effective_permeability,
            "Concrete effective_k ({:.6e}) should be > glass ({:.6e})",
            perm_concrete.effective_permeability,
            perm_glass.effective_permeability
        );
    }
}
