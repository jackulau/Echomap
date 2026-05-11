//! Fluid dynamics validation tests.
//!
//! Exercises the fluid simulation through scenario builders and direct
//! `FluidSimulation` construction, verifying physical invariants such as
//! rest-state stability, gravity-driven settling, boundary conditions,
//! pressure convergence, mass conservation, and viscous damping.

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use crate::fluids::grid::CellType;
    use crate::fluids::solver::FluidConfig;
    use crate::fluids::FluidSimulation;
    use crate::scenarios::builders::{FluidRoomScenario, ScenarioConfig};

    const EPSILON: f32 = 1e-6;

    // -----------------------------------------------------------------------
    // 1. Fluid at rest
    // -----------------------------------------------------------------------

    /// A freshly built fluid room (no stepping) should have zero velocity
    /// everywhere since the grid is initialised with all-zero face values.
    #[test]
    fn test_fluid_at_rest() {
        let scenario = FluidRoomScenario::build(&ScenarioConfig::default());
        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should be initialised");

        let u_max: f32 = grid.u.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let v_max: f32 = grid.v.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let w_max: f32 = grid.w.iter().map(|v| v.abs()).fold(0.0, f32::max);

        assert!(
            u_max < EPSILON,
            "u-velocity should be zero at rest, max = {u_max}"
        );
        assert!(
            v_max < EPSILON,
            "v-velocity should be zero at rest, max = {v_max}"
        );
        assert!(
            w_max < EPSILON,
            "w-velocity should be zero at rest, max = {w_max}"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Gravity-driven settling
    // -----------------------------------------------------------------------

    /// With gravity pointing downward, stepping the simulation should produce
    /// negative y-velocity at interior fluid cells.
    #[test]
    fn test_gravity_driven_settling() {
        let config = ScenarioConfig {
            grid_size: 8,
            resolution: 0.5,
            dt: 0.005,
            ..ScenarioConfig::default()
        };
        let mut scenario = FluidRoomScenario::build(&config);

        // The default FluidConfig already has gravity = (0, -9.81, 0).
        // Step 10 times to let gravity accelerate the fluid.
        for _ in 0..10 {
            scenario.simulation.step();
        }

        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should be initialised");

        // Check that interior v-faces (not on the domain boundary j=0 or j=ny)
        // have acquired negative (downward) velocity.
        let mut any_negative = false;
        for k in 1..grid.nz - 1 {
            for j in 1..grid.ny {
                // interior v-faces
                for i in 1..grid.nx - 1 {
                    let vidx = grid.idx_v(i, j, k);
                    if grid.v[vidx] < -EPSILON {
                        any_negative = true;
                    }
                }
            }
        }

        assert!(
            any_negative,
            "After 10 steps with gravity, at least some interior v-faces \
             should have negative (downward) velocity"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Buoyancy / heated-column analogue
    // -----------------------------------------------------------------------

    /// The solver applies buoyancy: `buoyancy = -g * (rho - rho_ref) / rho_ref`.
    /// When a column has *higher* density than the reference, the buoyancy
    /// term becomes positive (upward push on the v-faces above), which should
    /// alter the velocity field compared to a uniform-density baseline.
    ///
    /// We verify that a density perturbation produces a measurably different
    /// velocity field than the uniform case.
    #[test]
    fn test_buoyancy_heated_column() {
        let sc = ScenarioConfig {
            grid_size: 8,
            resolution: 0.5,
            dt: 0.005,
            ..ScenarioConfig::default()
        };

        // Baseline: uniform density, run 10 steps.
        let mut baseline = FluidRoomScenario::build(&sc);
        for _ in 0..10 {
            baseline.simulation.step();
        }
        let baseline_v: Vec<f32> = baseline.simulation.grid.as_ref().unwrap().v.clone();

        // Perturbed: heavier density in a column, run 10 steps.
        let mut perturbed = FluidRoomScenario::build(&sc);
        {
            let grid = perturbed
                .simulation
                .grid
                .as_mut()
                .expect("grid should be initialised");

            let ci = grid.nx / 2;
            let ck = grid.nz / 2;
            for j in 1..grid.ny - 1 {
                let idx = grid.idx(ci, j, ck);
                if grid.cell_types[idx] == CellType::Fluid {
                    grid.density[idx] = 1500.0; // heavier than reference 1000
                }
            }
        }
        for _ in 0..10 {
            perturbed.simulation.step();
        }
        let perturbed_v: Vec<f32> = perturbed.simulation.grid.as_ref().unwrap().v.clone();

        // The velocity fields should differ due to buoyancy.
        let max_diff: f32 = baseline_v
            .iter()
            .zip(perturbed_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_diff > EPSILON,
            "Buoyancy from density perturbation should produce a measurably \
             different velocity field, got max_diff = {max_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. No-slip boundary
    // -----------------------------------------------------------------------

    /// After stepping, domain-boundary velocity faces (i=0, i=nx for u;
    /// j=0, j=ny for v; k=0, k=nz for w) should be zero, enforced by the
    /// solver's `enforce_boundary_velocities`.
    #[test]
    fn test_no_slip_boundary() {
        let config = ScenarioConfig {
            grid_size: 8,
            resolution: 0.5,
            dt: 0.005,
            ..ScenarioConfig::default()
        };
        let mut scenario = FluidRoomScenario::build(&config);

        // Step a few times so the solver enforces boundary conditions.
        for _ in 0..5 {
            scenario.simulation.step();
        }

        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should be initialised");

        // u-faces at i=0 and i=nx
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                let u0 = grid.u[grid.idx_u(0, j, k)];
                let un = grid.u[grid.idx_u(grid.nx, j, k)];
                assert!(
                    u0.abs() < EPSILON,
                    "u at i=0 boundary should be ~0, got {u0} at j={j},k={k}"
                );
                assert!(
                    un.abs() < EPSILON,
                    "u at i=nx boundary should be ~0, got {un} at j={j},k={k}"
                );
            }
        }

        // v-faces at j=0 and j=ny
        for k in 0..grid.nz {
            for i in 0..grid.nx {
                let v0 = grid.v[grid.idx_v(i, 0, k)];
                let vn = grid.v[grid.idx_v(i, grid.ny, k)];
                assert!(
                    v0.abs() < EPSILON,
                    "v at j=0 boundary should be ~0, got {v0} at i={i},k={k}"
                );
                assert!(
                    vn.abs() < EPSILON,
                    "v at j=ny boundary should be ~0, got {vn} at i={i},k={k}"
                );
            }
        }

        // w-faces at k=0 and k=nz
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let w0 = grid.w[grid.idx_w(i, j, 0)];
                let wn = grid.w[grid.idx_w(i, j, grid.nz)];
                assert!(
                    w0.abs() < EPSILON,
                    "w at k=0 boundary should be ~0, got {w0} at i={i},j={j}"
                );
                assert!(
                    wn.abs() < EPSILON,
                    "w at k=nz boundary should be ~0, got {wn} at i={i},j={j}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5. Pressure convergence
    // -----------------------------------------------------------------------

    /// After stepping (which runs the Jacobi pressure solver), all pressure
    /// values should be finite (no NaN or Inf).
    #[test]
    fn test_pressure_convergence() {
        let config = ScenarioConfig {
            grid_size: 8,
            resolution: 0.5,
            dt: 0.005,
            ..ScenarioConfig::default()
        };
        let mut scenario = FluidRoomScenario::build(&config);

        for _ in 0..10 {
            scenario.simulation.step();
        }

        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should be initialised");

        assert!(
            grid.pressure.iter().all(|p| p.is_finite()),
            "All pressure values should be finite after stepping"
        );
        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "All u-velocities should be finite after stepping"
        );
        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "All v-velocities should be finite after stepping"
        );
        assert!(
            grid.w.iter().all(|v| v.is_finite()),
            "All w-velocities should be finite after stepping"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Mass conservation
    // -----------------------------------------------------------------------

    /// Total density summed over all cells should be approximately conserved
    /// (within 5%) after N simulation steps, since the solver does not
    /// advect density (it remains cell-fixed).
    #[test]
    fn test_mass_conservation() {
        let config = ScenarioConfig {
            grid_size: 8,
            resolution: 0.5,
            dt: 0.005,
            ..ScenarioConfig::default()
        };
        let mut scenario = FluidRoomScenario::build(&config);

        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should be initialised");
        let mass_before: f32 = grid.density.iter().sum();
        assert!(
            mass_before > 0.0,
            "Initial total density should be positive"
        );

        for _ in 0..20 {
            scenario.simulation.step();
        }

        let grid = scenario
            .simulation
            .grid
            .as_ref()
            .expect("grid should still be initialised");
        let mass_after: f32 = grid.density.iter().sum();

        let rel_change = ((mass_after - mass_before) / mass_before).abs();
        assert!(
            rel_change < 0.05,
            "Total density should be conserved within 5%: \
             before={mass_before}, after={mass_after}, rel_change={rel_change}"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Viscous damping
    // -----------------------------------------------------------------------

    /// Set an initial velocity perturbation, then step multiple times with
    /// viscosity. Kinetic energy (sum of v^2) should decrease due to
    /// viscous dissipation.
    #[test]
    fn test_viscous_damping() {
        // Use a custom FluidConfig with non-zero viscosity and zero gravity
        // to isolate viscous effects.
        let fluid_config = FluidConfig {
            dt: 0.005,
            viscosity: 0.1,
            density: 1000.0,
            gravity: Vec3::ZERO,
            surface_tension: 0.0,
            jacobi_iterations: 80,
        };

        let mut sim = FluidSimulation::new(fluid_config);
        let bounds = (Vec3::ZERO, Vec3::splat(4.0));
        sim.initialize(bounds, 0.5, &[]);

        // Inject velocity perturbation at interior faces.
        {
            let grid = sim.grid.as_mut().expect("grid should be initialised");

            // Mark all cells as Fluid and set reference density.
            for ct in grid.cell_types.iter_mut() {
                *ct = CellType::Fluid;
            }
            for d in grid.density.iter_mut() {
                *d = 1000.0;
            }

            // Set a spike of u-velocity in the center of the grid.
            let ci = grid.nx / 2;
            let cj = grid.ny / 2;
            let ck = grid.nz / 2;
            for di in 0..2 {
                for dj in 0..2 {
                    for dk in 0..2 {
                        let uidx = grid.idx_u(ci + di, cj + dj, ck + dk);
                        grid.u[uidx] = 5.0;
                    }
                }
            }
        }

        // Compute initial kinetic energy proxy (sum of u^2 + v^2 + w^2).
        let ke_before = {
            let grid = sim.grid.as_ref().unwrap();
            let u_ke: f32 = grid.u.iter().map(|v| v * v).sum();
            let v_ke: f32 = grid.v.iter().map(|v| v * v).sum();
            let w_ke: f32 = grid.w.iter().map(|v| v * v).sum();
            u_ke + v_ke + w_ke
        };
        assert!(ke_before > 0.0, "Initial kinetic energy should be positive");

        // Step multiple times.
        for _ in 0..50 {
            sim.step();
        }

        let ke_after = {
            let grid = sim.grid.as_ref().unwrap();
            let u_ke: f32 = grid.u.iter().map(|v| v * v).sum();
            let v_ke: f32 = grid.v.iter().map(|v| v * v).sum();
            let w_ke: f32 = grid.w.iter().map(|v| v * v).sum();
            u_ke + v_ke + w_ke
        };

        assert!(
            ke_after < ke_before,
            "Kinetic energy should decrease due to viscous damping: \
             before={ke_before}, after={ke_after}"
        );
    }

    // -----------------------------------------------------------------------
    // 8. Zero-dt no-op
    // -----------------------------------------------------------------------

    /// A FluidSimulation with dt=0 should not modify velocities when
    /// stepped, since all force/advection/diffusion terms scale with dt.
    #[test]
    fn test_zero_dt_noop() {
        // Construct config directly (bypassing FluidConfig::new which asserts dt > 0).
        let fluid_config = FluidConfig {
            dt: 0.0,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 80,
        };

        let mut sim = FluidSimulation::new(fluid_config);
        let bounds = (Vec3::ZERO, Vec3::splat(4.0));
        sim.initialize(bounds, 0.5, &[]);

        let _grid_before = sim.grid.as_ref().unwrap().clone();

        sim.step();

        let grid_after = sim.grid.as_ref().unwrap();

        // All velocity components should remain zero (the initial state).
        let u_max: f32 = grid_after.u.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let v_max: f32 = grid_after.v.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let w_max: f32 = grid_after.w.iter().map(|v| v.abs()).fold(0.0, f32::max);

        assert!(
            u_max < EPSILON,
            "u should remain zero with dt=0, max = {u_max}"
        );
        assert!(
            v_max < EPSILON,
            "v should remain zero with dt=0, max = {v_max}"
        );
        assert!(
            w_max < EPSILON,
            "w should remain zero with dt=0, max = {w_max}"
        );

        // Pressure should also remain zero.
        let p_max: f32 = grid_after
            .pressure
            .iter()
            .map(|p| p.abs())
            .fold(0.0, f32::max);
        assert!(
            p_max < EPSILON,
            "pressure should remain zero with dt=0, max = {p_max}"
        );
    }
}
