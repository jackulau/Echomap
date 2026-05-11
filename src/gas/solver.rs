use glam::Vec3;

use super::grid::{GasCellType, GasGrid, GasSpecies};

/// Configuration for the gas advection-diffusion solver.
#[derive(Clone, Debug)]
pub struct GasConfig {
    /// Simulation timestep (seconds).
    pub dt: f32,
    /// Ambient / background temperature (K).
    pub ambient_temperature: f32,
    /// Thermal diffusivity (m²/s).
    pub thermal_diffusivity: f32,
    /// Buoyancy coefficient (dimensionless scaling factor).
    pub buoyancy_coefficient: f32,
    /// Gravitational acceleration vector.
    pub gravity: Vec3,
}

impl GasConfig {
    /// Create a new gas config.
    ///
    /// # Panics
    /// - `dt` is zero or negative.
    pub fn new(
        dt: f32,
        ambient_temperature: f32,
        thermal_diffusivity: f32,
        buoyancy_coefficient: f32,
        gravity: Vec3,
    ) -> Self {
        assert!(dt > 0.0, "Timestep dt must be positive, got {dt}");
        Self {
            dt,
            ambient_temperature,
            thermal_diffusivity,
            buoyancy_coefficient,
            gravity,
        }
    }
}

impl Default for GasConfig {
    fn default() -> Self {
        Self {
            dt: 0.016,
            ambient_temperature: 293.15,
            thermal_diffusivity: 2.2e-5,
            buoyancy_coefficient: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Semi-Lagrangian advection (cell-centered)
// ---------------------------------------------------------------------------

/// Semi-Lagrangian advection for all species concentration fields.
///
/// For each Gas cell, back-trace through the velocity field and sample the
/// concentration from the previous timestep via trilinear interpolation.
pub fn advect_concentrations(grid: &mut GasGrid, dt: f32) {
    let num_species = grid.species.len();
    if num_species == 0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // Snapshot old concentration arrays.
    let old_concentrations = grid.concentrations.clone();

    // Build a temporary grid that holds the old data for interpolation.
    // We reuse the existing grid's interpolation but need old values, so we
    // swap in old data, sample, then write results.
    let old_grid = {
        let mut g = grid.clone();
        g.concentrations = old_concentrations;
        g
    };

    for s in 0..num_species {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);
                    if grid.cell_types[idx] != GasCellType::Gas {
                        continue;
                    }
                    let pos = grid.cell_center(i, j, k);
                    let vel = grid.velocity_at(pos);
                    let back_pos = pos - vel * dt;
                    grid.concentrations[s][idx] = old_grid.concentration_at(s, back_pos);
                }
            }
        }
    }
}

/// Semi-Lagrangian advection for the temperature field.
pub fn advect_temperature(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    let old_grid = grid.clone();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Gas {
                    continue;
                }
                let pos = grid.cell_center(i, j, k);
                let vel = grid.velocity_at(pos);
                let back_pos = pos - vel * dt;
                grid.temperature[idx] = old_grid.temperature_at(back_pos);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fickian diffusion (explicit forward-Euler)
// ---------------------------------------------------------------------------

/// Explicit Fickian diffusion for all species concentration fields.
///
/// Each species uses its own `diffusion_coefficient`. The stability factor
/// `D * dt / dx²` is clamped to `1/6` to prevent numerical blowup in 3D.
pub fn diffuse_concentrations(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let num_species = grid.species.len();

    for s in 0..num_species {
        let diff_coeff = grid.species[s].diffusion_coefficient;
        if diff_coeff <= 0.0 {
            continue;
        }

        let factor = (diff_coeff * dt / (dx * dx)).min(1.0 / 6.0);
        let old = grid.concentrations[s].clone();

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);
                    if grid.cell_types[idx] != GasCellType::Gas {
                        continue;
                    }

                    let c = old[idx];

                    // 6-connected neighbours with Neumann BC (replicate boundary).
                    let xm = if i > 0 { old[grid.idx(i - 1, j, k)] } else { c };
                    let xp = if i < nx - 1 {
                        old[grid.idx(i + 1, j, k)]
                    } else {
                        c
                    };
                    let ym = if j > 0 { old[grid.idx(i, j - 1, k)] } else { c };
                    let yp = if j < ny - 1 {
                        old[grid.idx(i, j + 1, k)]
                    } else {
                        c
                    };
                    let zm = if k > 0 { old[grid.idx(i, j, k - 1)] } else { c };
                    let zp = if k < nz - 1 {
                        old[grid.idx(i, j, k + 1)]
                    } else {
                        c
                    };

                    let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
                    grid.concentrations[s][idx] = c + factor * laplacian;
                }
            }
        }
    }
}

/// Explicit thermal diffusion for the temperature field.
pub fn diffuse_temperature(grid: &mut GasGrid, thermal_diffusivity: f32, dt: f32) {
    if thermal_diffusivity <= 0.0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let factor = (thermal_diffusivity * dt / (dx * dx)).min(1.0 / 6.0);

    let old = grid.temperature.clone();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Gas {
                    continue;
                }

                let c = old[idx];
                let xm = if i > 0 { old[grid.idx(i - 1, j, k)] } else { c };
                let xp = if i < nx - 1 {
                    old[grid.idx(i + 1, j, k)]
                } else {
                    c
                };
                let ym = if j > 0 { old[grid.idx(i, j - 1, k)] } else { c };
                let yp = if j < ny - 1 {
                    old[grid.idx(i, j + 1, k)]
                } else {
                    c
                };
                let zm = if k > 0 { old[grid.idx(i, j, k - 1)] } else { c };
                let zp = if k < nz - 1 {
                    old[grid.idx(i, j, k + 1)]
                } else {
                    c
                };

                let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
                grid.temperature[idx] = c + factor * laplacian;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Buoyancy
// ---------------------------------------------------------------------------

/// Apply temperature-driven buoyancy to `vel_y`.
///
/// Hot gas rises: cells hotter than ambient get an upward acceleration
/// proportional to `buoyancy_coefficient * (T - T_ambient)`.
pub fn apply_buoyancy(grid: &mut GasGrid, config: &GasConfig, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    let buoy = config.buoyancy_coefficient;
    let t_ambient = config.ambient_temperature;
    let g_y = config.gravity.y;

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Gas {
                    continue;
                }
                let delta_t = grid.temperature[idx] - t_ambient;
                // Buoyancy: force opposes gravity direction proportional to temperature excess.
                // Hot gas: delta_t > 0 => upward force => subtract g_y * buoy * delta_t * dt
                // (g_y is negative for downward gravity, so subtracting a negative pushes up)
                grid.vel_y[idx] -= g_y * buoy * delta_t * dt;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pressure gradient
// ---------------------------------------------------------------------------

/// Update velocity from pressure differences between adjacent cells.
pub fn apply_pressure_gradient(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let scale = dt / dx;

    // Snapshot pressure (read-only).
    let p = grid.pressure.clone();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Gas {
                    continue;
                }

                // x-gradient
                if i > 0 && grid.cell_types[grid.idx(i - 1, j, k)] == GasCellType::Gas {
                    let dp = p[idx] - p[grid.idx(i - 1, j, k)];
                    grid.vel_x[idx] -= scale * dp * 0.5;
                }
                if i < nx - 1 && grid.cell_types[grid.idx(i + 1, j, k)] == GasCellType::Gas {
                    let dp = p[grid.idx(i + 1, j, k)] - p[idx];
                    grid.vel_x[idx] -= scale * dp * 0.5;
                }

                // y-gradient
                if j > 0 && grid.cell_types[grid.idx(i, j - 1, k)] == GasCellType::Gas {
                    let dp = p[idx] - p[grid.idx(i, j - 1, k)];
                    grid.vel_y[idx] -= scale * dp * 0.5;
                }
                if j < ny - 1 && grid.cell_types[grid.idx(i, j + 1, k)] == GasCellType::Gas {
                    let dp = p[grid.idx(i, j + 1, k)] - p[idx];
                    grid.vel_y[idx] -= scale * dp * 0.5;
                }

                // z-gradient
                if k > 0 && grid.cell_types[grid.idx(i, j, k - 1)] == GasCellType::Gas {
                    let dp = p[idx] - p[grid.idx(i, j, k - 1)];
                    grid.vel_z[idx] -= scale * dp * 0.5;
                }
                if k < nz - 1 && grid.cell_types[grid.idx(i, j, k + 1)] == GasCellType::Gas {
                    let dp = p[grid.idx(i, j, k + 1)] - p[idx];
                    grid.vel_z[idx] -= scale * dp * 0.5;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full timestep
// ---------------------------------------------------------------------------

/// Execute a full gas simulation timestep:
/// advect -> diffuse -> buoyancy -> pressure gradient.
pub fn step(grid: &mut GasGrid, config: &GasConfig) {
    let dt = config.dt;

    // 1. Advection (semi-Lagrangian)
    advect_concentrations(grid, dt);
    advect_temperature(grid, dt);

    // 2. Diffusion
    diffuse_concentrations(grid, dt);
    diffuse_temperature(grid, config.thermal_diffusivity, dt);

    // 3. Buoyancy forces
    apply_buoyancy(grid, config, dt);

    // 4. Pressure gradient
    apply_pressure_gradient(grid, dt);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_species(name: &str, diff_coeff: f32) -> GasSpecies {
        GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: diff_coeff,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

    /// Create a gas grid with all cells marked as Gas.
    fn make_gas_grid(n: usize, dx: f32, species: Vec<GasSpecies>) -> GasGrid {
        let mut g = GasGrid::new(n, n, n, dx, Vec3::ZERO, species);
        for ct in g.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }
        g
    }

    fn default_config() -> GasConfig {
        GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.02,
            buoyancy_coefficient: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }

    // ----- 8 required tests -----

    #[test]
    fn test_gas_solver_zero_concentration_stays_zero() {
        let species = vec![make_species("CO2", 0.2)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let config = default_config();

        // All concentrations, temperature, pressure, velocity are zero.
        step(&mut grid, &config);

        let max_conc: f32 = grid.concentrations[0]
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max);
        assert!(
            max_conc < 1e-10,
            "Zero concentration should stay zero after step, got max={max_conc}"
        );
    }

    #[test]
    fn test_gas_solver_diffusion_spreads_concentration() {
        let species = vec![make_species("CO2", 0.2)];
        let mut grid = make_gas_grid(16, 0.1, species);

        // Place a point source at the center.
        let ci = 8;
        let center_idx = grid.idx(ci, ci, ci);
        grid.concentrations[0][center_idx] = 100.0;

        let initial_center = grid.concentrations[0][center_idx];

        // Run several diffusion steps.
        for _ in 0..20 {
            diffuse_concentrations(&mut grid, 0.01);
        }

        // Center value should have decreased (spread out).
        let final_center = grid.concentrations[0][center_idx];
        assert!(
            final_center < initial_center,
            "Diffusion should spread the point source: initial={initial_center}, final={final_center}"
        );

        // A direct neighbour should have gained concentration.
        let neighbour_idx = grid.idx(ci + 1, ci, ci);
        let neighbour_val = grid.concentrations[0][neighbour_idx];
        assert!(
            neighbour_val > 0.0,
            "Neighbour should have nonzero concentration after diffusion, got {neighbour_val}"
        );
    }

    #[test]
    fn test_gas_solver_diffusion_conserves_mass() {
        let species = vec![make_species("CO2", 0.2)];
        let mut grid = make_gas_grid(16, 0.1, species);

        // Place a point source.
        let center_idx = grid.idx(8, 8, 8);
        grid.concentrations[0][center_idx] = 100.0;

        let mass_before: f32 = grid.concentrations[0].iter().sum();

        for _ in 0..50 {
            diffuse_concentrations(&mut grid, 0.01);
        }

        let mass_after: f32 = grid.concentrations[0].iter().sum();

        let rel_change = if mass_before > 0.0 {
            ((mass_after - mass_before) / mass_before).abs()
        } else {
            0.0
        };
        assert!(
            rel_change < 0.01,
            "Diffusion should conserve mass within 1%: before={mass_before}, after={mass_after}, change={rel_change}"
        );
    }

    #[test]
    fn test_gas_solver_advection_uniform_field() {
        let species = vec![make_species("CO2", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);

        // Set uniform concentration = 5.0.
        for val in grid.concentrations[0].iter_mut() {
            *val = 5.0;
        }

        // Set a uniform velocity field.
        for val in grid.vel_x.iter_mut() {
            *val = 1.0;
        }

        advect_concentrations(&mut grid, 0.01);

        // Interior cells should remain close to 5.0.
        for k in 1..7 {
            for j in 1..7 {
                for i in 1..7 {
                    let idx = grid.idx(i, j, k);
                    let c = grid.concentrations[0][idx];
                    assert!(
                        (c - 5.0).abs() < 0.1,
                        "Uniform concentration should stay ~5.0 after advection at ({i},{j},{k}), got {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gas_solver_buoyancy_hot_rises() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);

        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        // Set all cells to ambient temperature first.
        for val in grid.temperature.iter_mut() {
            *val = config.ambient_temperature;
        }

        // Heat up the bottom layer well above ambient.
        for k in 0..8 {
            for i in 0..8 {
                let idx = grid.idx(i, 0, k);
                grid.temperature[idx] = 500.0; // much hotter than ambient
            }
        }

        apply_buoyancy(&mut grid, &config, config.dt);

        // The bottom row should have gained upward velocity.
        // gravity.y = -9.81, delta_t > 0, buoy > 0 =>
        // vel_y -= g_y * buoy * delta_t * dt  =>  vel_y -= (-9.81) * 0.5 * 206.85 * 0.01
        // => vel_y += positive number => upward
        let mut bottom_vy: f32 = 0.0;
        for k in 0..8 {
            for i in 0..8 {
                bottom_vy += grid.vel_y[grid.idx(i, 0, k)];
            }
        }
        assert!(
            bottom_vy > 0.0,
            "Hot region should gain upward velocity, got sum={bottom_vy}"
        );
    }

    #[test]
    fn test_gas_solver_pressure_gradient_drives_flow() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);

        // Set up a pressure gradient: high on the left, low on the right.
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    let idx = grid.idx(i, j, k);
                    grid.pressure[idx] = (8 - i) as f32 * 10.0;
                }
            }
        }

        apply_pressure_gradient(&mut grid, 0.01);

        // Pressure decreases in +x direction, so flow should be in +x direction
        // (velocity driven from high to low pressure).
        // Interior cells away from boundaries should have positive vel_x.
        let mut positive_count = 0;
        let mut total_count = 0;
        for k in 1..7 {
            for j in 1..7 {
                for i in 1..7 {
                    let idx = grid.idx(i, j, k);
                    total_count += 1;
                    if grid.vel_x[idx] > 0.0 {
                        positive_count += 1;
                    }
                }
            }
        }

        let ratio = positive_count as f32 / total_count as f32;
        assert!(
            ratio > 0.5,
            "Pressure gradient should drive flow from high to low pressure. \
             {positive_count}/{total_count} cells have positive vel_x (ratio={ratio})"
        );
    }

    #[test]
    fn test_gas_solver_step_all_finite() {
        let species = vec![make_species("CO2", 0.2), make_species("CH4", 0.15)];
        let mut grid = make_gas_grid(8, 0.125, species);

        // Set some initial conditions to make things interesting.
        let center = grid.idx(4, 4, 4);
        grid.concentrations[0][center] = 50.0;
        grid.concentrations[1][center] = 30.0;
        grid.temperature[center] = 400.0;
        grid.pressure[center] = 10.0;

        let config = default_config();

        for iteration in 0..100 {
            step(&mut grid, &config);

            // Check all fields remain finite.
            assert!(
                grid.vel_x.iter().all(|v| v.is_finite()),
                "vel_x has NaN/Inf at step {iteration}"
            );
            assert!(
                grid.vel_y.iter().all(|v| v.is_finite()),
                "vel_y has NaN/Inf at step {iteration}"
            );
            assert!(
                grid.vel_z.iter().all(|v| v.is_finite()),
                "vel_z has NaN/Inf at step {iteration}"
            );
            assert!(
                grid.temperature.iter().all(|v| v.is_finite()),
                "temperature has NaN/Inf at step {iteration}"
            );
            assert!(
                grid.pressure.iter().all(|v| v.is_finite()),
                "pressure has NaN/Inf at step {iteration}"
            );
            for (s, conc) in grid.concentrations.iter().enumerate() {
                assert!(
                    conc.iter().all(|v| v.is_finite()),
                    "concentration[{s}] has NaN/Inf at step {iteration}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Timestep dt must be positive")]
    fn test_gas_solver_config_validation() {
        GasConfig::new(0.0, 293.15, 0.02, 0.01, Vec3::new(0.0, -9.81, 0.0));
    }
}
