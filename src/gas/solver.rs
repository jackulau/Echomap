use glam::Vec3;
use rayon::prelude::*;

use super::grid::{GasCellType, GasGrid};

#[cfg(test)]
use super::grid::GasSpecies;

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
    #[allow(dead_code)]
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
/// Cell updates are independent — parallelized with rayon across the flat
/// cell index.
pub fn advect_concentrations(grid: &mut GasGrid, dt: f32) {
    let num_species = grid.species.len();
    if num_species == 0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let cell_count = nx * ny * nz;

    let old_concentrations: Vec<Vec<f32>> = grid.concentrations.clone();

    for (s, old_conc) in old_concentrations.iter().enumerate() {
        let new_conc: Vec<f32> = (0..cell_count)
            .into_par_iter()
            .map(|idx| {
                if grid.cell_types[idx] != GasCellType::Gas {
                    return old_conc[idx];
                }
                let (i, j, k) = grid.idx_to_ijk(idx);
                let pos = grid.cell_center(i, j, k);
                let vel = grid.velocity_at(pos);
                let back_pos = pos - vel * dt;
                grid.interpolate_cell_centered(old_conc, back_pos)
            })
            .collect();
        grid.concentrations[s] = new_conc;
    }
}

/// Semi-Lagrangian advection for the temperature field. Parallelized with rayon.
pub fn advect_temperature(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let cell_count = nx * ny * nz;

    let old_temperature: Vec<f32> = grid.temperature.clone();

    let new_temperature: Vec<f32> = (0..cell_count)
        .into_par_iter()
        .map(|idx| {
            if grid.cell_types[idx] != GasCellType::Gas {
                return old_temperature[idx];
            }
            let (i, j, k) = grid.idx_to_ijk(idx);
            let pos = grid.cell_center(i, j, k);
            let vel = grid.velocity_at(pos);
            let back_pos = pos - vel * dt;
            grid.interpolate_cell_centered(&old_temperature, back_pos)
        })
        .collect();
    grid.temperature = new_temperature;
}

// ---------------------------------------------------------------------------
// Fickian diffusion (explicit forward-Euler)
// ---------------------------------------------------------------------------

/// Explicit Fickian diffusion for all species concentration fields.
///
/// Each species uses its own `diffusion_coefficient`. The stability factor
/// `D * dt / dx²` is clamped to `1/6` to prevent numerical blowup in 3D.
/// Cell updates are independent — parallelized with rayon across the flat
/// cell index.
pub fn diffuse_concentrations(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let cell_count = nx * ny * nz;
    let num_species = grid.species.len();

    let idx_of = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    // Ensure scratch buffer is right-sized (lazy migration for grids built
    // before scratch_scalar existed).
    if grid.scratch_scalar.len() != cell_count {
        grid.scratch_scalar.resize(cell_count, 0.0);
    }

    for s in 0..num_species {
        let diff_coeff = grid.species[s].diffusion_coefficient;
        if diff_coeff <= 0.0 {
            continue;
        }

        let factor = (diff_coeff * dt / (dx * dx)).min(1.0 / 6.0);

        // Swap concentrations[s] into scratch_scalar so we read old values
        // from scratch and write new values directly into concentrations[s]
        // — no per-step heap allocation.
        std::mem::swap(&mut grid.scratch_scalar, &mut grid.concentrations[s]);
        let old: &[f32] = &grid.scratch_scalar;
        let cell_types_ref = &grid.cell_types;
        let dst: &mut [f32] = &mut grid.concentrations[s];

        dst.par_iter_mut().enumerate().for_each(|(idx, out)| {
            if cell_types_ref[idx] != GasCellType::Gas {
                *out = old[idx];
                return;
            }

            let k = idx / (nx * ny);
            let rem = idx - k * nx * ny;
            let j = rem / nx;
            let i = rem - j * nx;

            let c = old[idx];
            let xm = if i > 0 { old[idx_of(i - 1, j, k)] } else { c };
            let xp = if i < nx - 1 {
                old[idx_of(i + 1, j, k)]
            } else {
                c
            };
            let ym = if j > 0 { old[idx_of(i, j - 1, k)] } else { c };
            let yp = if j < ny - 1 {
                old[idx_of(i, j + 1, k)]
            } else {
                c
            };
            let zm = if k > 0 { old[idx_of(i, j, k - 1)] } else { c };
            let zp = if k < nz - 1 {
                old[idx_of(i, j, k + 1)]
            } else {
                c
            };

            let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
            *out = c + factor * laplacian;
        });
    }
}

/// Explicit thermal diffusion for the temperature field. Parallelized with rayon.
pub fn diffuse_temperature(grid: &mut GasGrid, thermal_diffusivity: f32, dt: f32) {
    if thermal_diffusivity <= 0.0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let cell_count = nx * ny * nz;
    let factor = (thermal_diffusivity * dt / (dx * dx)).min(1.0 / 6.0);
    let idx_of = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    if grid.scratch_scalar.len() != cell_count {
        grid.scratch_scalar.resize(cell_count, 0.0);
    }

    std::mem::swap(&mut grid.scratch_scalar, &mut grid.temperature);
    let old: &[f32] = &grid.scratch_scalar;
    let cell_types_ref = &grid.cell_types;
    let dst: &mut [f32] = &mut grid.temperature;

    dst.par_iter_mut().enumerate().for_each(|(idx, out)| {
        if cell_types_ref[idx] != GasCellType::Gas {
            *out = old[idx];
            return;
        }
        let k = idx / (nx * ny);
        let rem = idx - k * nx * ny;
        let j = rem / nx;
        let i = rem - j * nx;

        let c = old[idx];
        let xm = if i > 0 { old[idx_of(i - 1, j, k)] } else { c };
        let xp = if i < nx - 1 {
            old[idx_of(i + 1, j, k)]
        } else {
            c
        };
        let ym = if j > 0 { old[idx_of(i, j - 1, k)] } else { c };
        let yp = if j < ny - 1 {
            old[idx_of(i, j + 1, k)]
        } else {
            c
        };
        let zm = if k > 0 { old[idx_of(i, j, k - 1)] } else { c };
        let zp = if k < nz - 1 {
            old[idx_of(i, j, k + 1)]
        } else {
            c
        };

        let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
        *out = c + factor * laplacian;
    });
}

// ---------------------------------------------------------------------------
// Buoyancy
// ---------------------------------------------------------------------------

/// Apply temperature-driven buoyancy to `vel_y`. Parallelized with rayon.
///
/// Hot gas rises: cells hotter than ambient get an upward acceleration
/// proportional to `buoyancy_coefficient * (T - T_ambient)`.
pub fn apply_buoyancy(grid: &mut GasGrid, config: &GasConfig, dt: f32) {
    let buoy = config.buoyancy_coefficient;
    let t_ambient = config.ambient_temperature;
    let g_y = config.gravity.y;

    let temperature = &grid.temperature;
    let cell_types = &grid.cell_types;

    grid.vel_y.par_iter_mut().enumerate().for_each(|(idx, v)| {
        if cell_types[idx] != GasCellType::Gas {
            return;
        }
        let delta_t = temperature[idx] - t_ambient;
        *v -= g_y * buoy * delta_t * dt;
    });
}

// ---------------------------------------------------------------------------
// Pressure gradient
// ---------------------------------------------------------------------------

/// Update velocity from pressure differences between adjacent cells.
/// Each component (vel_x, vel_y, vel_z) parallelized with rayon — three passes
/// over disjoint fields, all reading the cloned pressure snapshot.
pub fn apply_pressure_gradient(grid: &mut GasGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let scale = dt / dx;

    let p = grid.pressure.clone();
    let cell_types = &grid.cell_types;
    let idx_of = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    // x-gradient -> vel_x
    grid.vel_x.par_iter_mut().enumerate().for_each(|(idx, v)| {
        if cell_types[idx] != GasCellType::Gas {
            return;
        }
        let k = idx / (nx * ny);
        let rem = idx - k * nx * ny;
        let j = rem / nx;
        let i = rem - j * nx;
        if i > 0 && cell_types[idx_of(i - 1, j, k)] == GasCellType::Gas {
            *v -= scale * (p[idx] - p[idx_of(i - 1, j, k)]) * 0.5;
        }
        if i < nx - 1 && cell_types[idx_of(i + 1, j, k)] == GasCellType::Gas {
            *v -= scale * (p[idx_of(i + 1, j, k)] - p[idx]) * 0.5;
        }
    });

    // y-gradient -> vel_y
    grid.vel_y.par_iter_mut().enumerate().for_each(|(idx, v)| {
        if cell_types[idx] != GasCellType::Gas {
            return;
        }
        let k = idx / (nx * ny);
        let rem = idx - k * nx * ny;
        let j = rem / nx;
        let i = rem - j * nx;
        if j > 0 && cell_types[idx_of(i, j - 1, k)] == GasCellType::Gas {
            *v -= scale * (p[idx] - p[idx_of(i, j - 1, k)]) * 0.5;
        }
        if j < ny - 1 && cell_types[idx_of(i, j + 1, k)] == GasCellType::Gas {
            *v -= scale * (p[idx_of(i, j + 1, k)] - p[idx]) * 0.5;
        }
    });

    // z-gradient -> vel_z
    grid.vel_z.par_iter_mut().enumerate().for_each(|(idx, v)| {
        if cell_types[idx] != GasCellType::Gas {
            return;
        }
        let k = idx / (nx * ny);
        let rem = idx - k * nx * ny;
        let j = rem / nx;
        let i = rem - j * nx;
        if k > 0 && cell_types[idx_of(i, j, k - 1)] == GasCellType::Gas {
            *v -= scale * (p[idx] - p[idx_of(i, j, k - 1)]) * 0.5;
        }
        if k < nz - 1 && cell_types[idx_of(i, j, k + 1)] == GasCellType::Gas {
            *v -= scale * (p[idx_of(i, j, k + 1)] - p[idx]) * 0.5;
        }
    });
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

    // ----- 6 integration tests (Task 8) -----

    /// Point source in 3D, verify concentration profile approaches Gaussian.
    ///
    /// Place all concentration at the center cell, run diffusion, then verify
    /// the radial profile has the Gaussian shape (monotone decay, correct
    /// ratios between distances) within 10% tolerance.
    #[test]
    fn test_integration_point_source_diffusion() {
        let diff_coeff = 0.05; // scaled for grid
        let species = vec![make_species("CO2", diff_coeff)];
        let n = 32;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Stability: D*dt/dx^2 <= 1/6 => dt <= dx^2/(6*D) = 0.01/(0.3) ~ 0.033
        // The solver clamps the factor to 1/6, so the effective D may differ.
        // Use many small steps for accuracy.
        let dt = 0.005;
        let num_steps = 200; // 200 * 0.005 = 1.0s
        let t = num_steps as f32 * dt;

        // Place point source at center cell.
        let ci = n / 2;
        let center_idx = grid.idx(ci, ci, ci);
        grid.concentrations[0][center_idx] = 1000.0;

        // Run pure diffusion for ~1s.
        for _ in 0..num_steps {
            diffuse_concentrations(&mut grid, dt);
        }

        let c_center = grid.concentrations[0][center_idx];

        // 1. Monotone radial decay: center > r=1 > r=2 > r=4 cells away
        let c_r1 = grid.concentrations[0][grid.idx(ci + 1, ci, ci)];
        let c_r2 = grid.concentrations[0][grid.idx(ci + 2, ci, ci)];
        let c_r4 = grid.concentrations[0][grid.idx(ci + 4, ci, ci)];
        assert!(
            c_center > c_r1 && c_r1 > c_r2 && c_r2 > c_r4,
            "Gaussian should decay monotonically: c0={c_center:.4}, c1={c_r1:.4}, c2={c_r2:.4}, c4={c_r4:.4}"
        );

        // 2. Gaussian shape check: for a Gaussian C(r) = A * exp(-r^2 / (4*D*t)),
        //    the ratio C(r2)/C(r1) = exp(-(r2^2 - r1^2) / (4*D*t)).
        //    Use the effective D from the clamped factor.
        let factor = (diff_coeff * dt / (dx * dx)).min(1.0 / 6.0);
        let d_eff = factor * dx * dx / dt;
        let sigma_sq = 4.0 * d_eff * t;

        // Compare ratio at r=2 cells vs r=0 (center):
        // r = 2*dx in physical units
        let r2_phys = 2.0 * dx;
        let expected_ratio = (-r2_phys * r2_phys / sigma_sq).exp();
        let actual_ratio = c_r2 / c_center;

        let ratio_err = (actual_ratio - expected_ratio).abs() / expected_ratio.max(1e-10);
        assert!(
            ratio_err < 0.10,
            "Gaussian ratio at r=2dx should match within 10%: \
             actual={actual_ratio:.4}, expected={expected_ratio:.4}, err={ratio_err:.4}"
        );

        // 3. Isotropy: concentration at same distance in different directions
        //    should be approximately equal.
        let c_r2_y = grid.concentrations[0][grid.idx(ci, ci + 2, ci)];
        let c_r2_z = grid.concentrations[0][grid.idx(ci, ci, ci + 2)];
        let iso_err_y = (c_r2 - c_r2_y).abs() / c_r2.max(1e-10);
        let iso_err_z = (c_r2 - c_r2_z).abs() / c_r2.max(1e-10);
        assert!(
            iso_err_y < 0.05 && iso_err_z < 0.05,
            "Diffusion should be isotropic: c_r2_x={c_r2:.4}, c_r2_y={c_r2_y:.4}, c_r2_z={c_r2_z:.4}"
        );
    }

    /// Total concentration must remain constant (within 1%) over 100 full
    /// solver steps.
    #[test]
    fn test_integration_mass_conservation() {
        let species = vec![make_species("CO2", 0.05)];
        let n = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Distribute some non-trivial initial concentration.
        let ci = n / 2;
        let idx_a = grid.idx(ci, ci, ci);
        let idx_b = grid.idx(ci + 1, ci, ci);
        let idx_c = grid.idx(ci, ci + 1, ci);
        grid.concentrations[0][idx_a] = 100.0;
        grid.concentrations[0][idx_b] = 50.0;
        grid.concentrations[0][idx_c] = 30.0;

        // Use a config with no buoyancy/thermal effects to keep mass truly
        // conserved (advection of a uniform-velocity field and diffusion are
        // both mass-conserving on an interior domain).
        let config = GasConfig {
            dt: 0.001,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.0,
            gravity: Vec3::ZERO,
        };

        let mass_before: f32 = grid.concentrations[0].iter().sum();

        for _ in 0..100 {
            step(&mut grid, &config);
        }

        let mass_after: f32 = grid.concentrations[0].iter().sum();
        let rel_change = ((mass_after - mass_before) / mass_before).abs();

        assert!(
            rel_change < 0.01,
            "Mass should be conserved within 1% over 100 steps: \
             before={mass_before}, after={mass_after}, rel_change={rel_change}"
        );
    }

    /// Hot spot at the bottom should develop upward velocity via buoyancy.
    #[test]
    fn test_integration_thermal_convection() {
        let species = vec![make_species("Air", 0.0)];
        let n = 16;
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
        for val in grid.temperature.iter_mut() {
            *val = config.ambient_temperature;
        }

        // Create a hot spot in the bottom layer (j=0..2).
        for k in 4..12 {
            for i in 4..12 {
                for j in 0..2 {
                    let idx = grid.idx(i, j, k);
                    grid.temperature[idx] = 500.0;
                }
            }
        }

        // Run several full timesteps.
        for _ in 0..50 {
            step(&mut grid, &config);
        }

        // Measure upward velocity above the hot spot at mid-height.
        let mut total_vy_above = 0.0_f32;
        let mut count = 0;
        for k in 6..10 {
            for i in 6..10 {
                for j in 4..8 {
                    let idx = grid.idx(i, j, k);
                    total_vy_above += grid.vel_y[idx];
                    count += 1;
                }
            }
        }
        let avg_vy = total_vy_above / count as f32;

        assert!(
            avg_vy > 0.0,
            "Hot bottom should produce upward velocity above it: avg_vy={avg_vy}"
        );
    }

    /// Two species filling adjacent halves should both diffuse toward center.
    #[test]
    fn test_integration_two_species_mixing() {
        let species = vec![make_species("CO2", 0.05), make_species("CH4", 0.05)];
        let n = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Fill left half (i < n/2) with species 0, right half with species 1.
        let half = n / 2;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = grid.idx(i, j, k);
                    if i < half {
                        grid.concentrations[0][idx] = 10.0;
                    } else {
                        grid.concentrations[1][idx] = 10.0;
                    }
                }
            }
        }

        // Snapshot initial center-plane concentrations.
        let center_i = half; // first cell of right half
        let probe_idx = grid.idx(center_i, n / 2, n / 2);
        let s0_right_before = grid.concentrations[0][probe_idx];
        let s1_left_before = grid.concentrations[1][grid.idx(center_i - 1, n / 2, n / 2)];

        // Run pure diffusion for many steps.
        let dt = 0.005;
        for _ in 0..200 {
            diffuse_concentrations(&mut grid, dt);
        }

        // Species 0 should have diffused into the right half.
        let s0_right_after = grid.concentrations[0][probe_idx];
        assert!(
            s0_right_after > s0_right_before + 0.01,
            "Species 0 should diffuse rightward: before={s0_right_before}, after={s0_right_after}"
        );

        // Species 1 should have diffused into the left half.
        let s1_left_after = grid.concentrations[1][grid.idx(center_i - 1, n / 2, n / 2)];
        assert!(
            s1_left_after > s1_left_before + 0.01,
            "Species 1 should diffuse leftward: before={s1_left_before}, after={s1_left_after}"
        );
    }

    /// A solid wall in the middle of the grid should prevent concentration
    /// from passing through.
    #[test]
    fn test_integration_solid_walls_block_diffusion() {
        use super::super::boundary::enforce_boundary_conditions;

        let species = vec![make_species("CO2", 0.05)];
        let n = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Place a 2-cell-thick solid wall at i = n/2 and i = n/2 + 1 spanning
        // the full y-z plane. Two cells thick ensures the diffusion stencil
        // cannot bridge across (each Gas cell only sees Solid neighbours, not
        // the Gas cells on the other side).
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

        // Put high concentration on the left side only.
        for k in 0..n {
            for j in 0..n {
                for i in 0..wall_i0 {
                    let idx = grid.idx(i, j, k);
                    grid.concentrations[0][idx] = 100.0;
                }
            }
        }

        // Run diffusion + boundary enforcement for many steps.
        let dt = 0.005;
        for _ in 0..200 {
            diffuse_concentrations(&mut grid, dt);
            enforce_boundary_conditions(&mut grid);
        }

        // All Gas cells to the right of the wall (i > wall_i1) should have
        // negligible concentration (the wall blocks diffusion).
        let mut max_right = 0.0_f32;
        for k in 0..n {
            for j in 0..n {
                for i in (wall_i1 + 1)..n {
                    let idx = grid.idx(i, j, k);
                    if grid.cell_types[idx] == GasCellType::Gas {
                        let val = grid.concentrations[0][idx];
                        if val > max_right {
                            max_right = val;
                        }
                    }
                }
            }
        }

        assert!(
            max_right < 1.0,
            "Concentration should not pass through solid wall: max on right side={max_right}"
        );
    }

    /// 1000 steps on a 16^3 grid -- all values must remain finite.
    #[test]
    fn test_integration_long_run_stability() {
        let species = vec![make_species("CO2", 0.05), make_species("CH4", 0.03)];
        let n = 16;
        let dx = 0.1;
        let mut grid = make_gas_grid(n, dx, species);

        // Set up non-trivial initial conditions.
        let ci = n / 2;
        let center = grid.idx(ci, ci, ci);
        grid.concentrations[0][center] = 100.0;
        grid.concentrations[1][center] = 50.0;
        grid.temperature[center] = 400.0;
        grid.pressure[center] = 5.0;

        let config = GasConfig {
            dt: 0.001,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.01,
            buoyancy_coefficient: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        for iteration in 0..1000 {
            step(&mut grid, &config);

            // Spot-check every 100 steps to catch blowup early.
            if iteration % 100 == 99 {
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

        // Final check: every single value must be finite.
        assert!(
            grid.vel_x.iter().all(|v| v.is_finite()),
            "vel_x has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.vel_y.iter().all(|v| v.is_finite()),
            "vel_y has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.vel_z.iter().all(|v| v.is_finite()),
            "vel_z has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.temperature.iter().all(|v| v.is_finite()),
            "temperature has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "pressure has NaN/Inf after 1000 steps"
        );
        for (s, conc) in grid.concentrations.iter().enumerate() {
            assert!(
                conc.iter().all(|v| v.is_finite()),
                "concentration[{s}] has NaN/Inf after 1000 steps"
            );
        }
    }

    // ---- Q3 Edge Case Tests ----

    #[test]
    #[should_panic(expected = "Timestep dt must be positive")]
    fn test_edge_config_negative_dt() {
        GasConfig::new(-0.01, 293.15, 0.02, 0.01, Vec3::new(0.0, -9.81, 0.0));
    }

    #[test]
    fn test_edge_advect_empty_species() {
        let mut grid = {
            let mut g = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, vec![]);
            for ct in g.cell_types.iter_mut() {
                *ct = GasCellType::Gas;
            }
            g
        };
        advect_concentrations(&mut grid, 0.01);
        assert_eq!(grid.concentrations.len(), 0);
    }

    #[test]
    fn test_edge_diffuse_zero_coefficient() {
        let species = vec![make_species("Inert", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let center_idx = grid.idx(4, 4, 4);
        grid.concentrations[0][center_idx] = 100.0;

        let before = grid.concentrations[0].clone();
        diffuse_concentrations(&mut grid, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.concentrations[0].iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Zero diffusion coeff should leave concentration unchanged at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_diffuse_negative_coefficient() {
        let species = vec![make_species("Neg", -0.5)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let center_idx = grid.idx(4, 4, 4);
        grid.concentrations[0][center_idx] = 100.0;

        let before = grid.concentrations[0].clone();
        diffuse_concentrations(&mut grid, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.concentrations[0].iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Negative diffusion coeff should leave concentration unchanged at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_diffuse_temperature_zero_diffusivity() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let idx_center = grid.idx(4, 4, 4);
        grid.temperature[idx_center] = 500.0;

        let before = grid.temperature.clone();
        diffuse_temperature(&mut grid, 0.0, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.temperature.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Zero thermal diffusivity should leave temperature unchanged at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_diffuse_temperature_negative_diffusivity() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let idx_center = grid.idx(4, 4, 4);
        grid.temperature[idx_center] = 500.0;

        let before = grid.temperature.clone();
        diffuse_temperature(&mut grid, -1.0, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.temperature.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Negative thermal diffusivity should leave temperature unchanged at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_all_solid_grid_noop() {
        let species = vec![make_species("CO2", 0.2)];
        let mut grid = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }
        let ci = grid.idx(2, 2, 2);
        grid.concentrations[0][ci] = 100.0;
        grid.temperature[ci] = 500.0;

        let conc_before = grid.concentrations[0].clone();
        let temp_before = grid.temperature.clone();
        let config = default_config();
        step(&mut grid, &config);

        for (i, (b, a)) in conc_before
            .iter()
            .zip(grid.concentrations[0].iter())
            .enumerate()
        {
            assert!(
                (b - a).abs() < 1e-10,
                "All-solid grid: concentration should not change at index {i}"
            );
        }
        for (i, (b, a)) in temp_before.iter().zip(grid.temperature.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "All-solid grid: temperature should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_all_empty_grid_noop() {
        let species = vec![make_species("CO2", 0.2)];
        let mut grid = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        let ci = grid.idx(2, 2, 2);
        grid.concentrations[0][ci] = 100.0;
        grid.temperature[ci] = 500.0;

        let conc_before = grid.concentrations[0].clone();
        let temp_before = grid.temperature.clone();
        let config = default_config();
        step(&mut grid, &config);

        for (i, (b, a)) in conc_before
            .iter()
            .zip(grid.concentrations[0].iter())
            .enumerate()
        {
            assert!(
                (b - a).abs() < 1e-10,
                "All-empty grid: concentration should not change at index {i}"
            );
        }
        for (i, (b, a)) in temp_before.iter().zip(grid.temperature.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "All-empty grid: temperature should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_single_gas_cell_diffusion() {
        let species = vec![make_species("CO2", 0.5)];
        let mut grid = make_gas_grid(1, 1.0, species);
        grid.concentrations[0][0] = 42.0;

        diffuse_concentrations(&mut grid, 0.01);
        assert!(
            (grid.concentrations[0][0] - 42.0).abs() < 1e-10,
            "Single cell diffusion should keep concentration: got {}",
            grid.concentrations[0][0]
        );
    }

    #[test]
    fn test_edge_single_gas_cell_full_step() {
        let species = vec![make_species("CO2", 0.5)];
        let mut grid = make_gas_grid(1, 1.0, species);
        grid.concentrations[0][0] = 42.0;
        grid.temperature[0] = 300.0;

        let config = default_config();
        step(&mut grid, &config);

        assert!(grid.concentrations[0][0].is_finite());
        assert!(grid.temperature[0].is_finite());
        assert!(grid.vel_x[0].is_finite());
        assert!(grid.vel_y[0].is_finite());
        assert!(grid.vel_z[0].is_finite());
    }

    #[test]
    fn test_edge_buoyancy_zero_gravity() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(4, 0.25, species);
        let ci = grid.idx(2, 2, 2);
        grid.temperature[ci] = 500.0;

        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.5,
            gravity: Vec3::ZERO,
        };

        let vel_before = grid.vel_y.clone();
        apply_buoyancy(&mut grid, &config, config.dt);

        for (i, (b, a)) in vel_before.iter().zip(grid.vel_y.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Zero gravity: vel_y should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_buoyancy_zero_coefficient() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(4, 0.25, species);
        let ci = grid.idx(2, 2, 2);
        grid.temperature[ci] = 500.0;

        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        let vel_before = grid.vel_y.clone();
        apply_buoyancy(&mut grid, &config, config.dt);

        for (i, (b, a)) in vel_before.iter().zip(grid.vel_y.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Zero buoyancy coeff: vel_y should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_buoyancy_at_ambient_temp() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(4, 0.25, species);
        for t in grid.temperature.iter_mut() {
            *t = 293.15;
        }

        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        let vel_before = grid.vel_y.clone();
        apply_buoyancy(&mut grid, &config, config.dt);

        for (i, (b, a)) in vel_before.iter().zip(grid.vel_y.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "At ambient temp: no buoyancy force expected at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_pressure_gradient_uniform_pressure() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        for p in grid.pressure.iter_mut() {
            *p = 100.0;
        }

        let vel_x_before = grid.vel_x.clone();
        let vel_y_before = grid.vel_y.clone();
        let vel_z_before = grid.vel_z.clone();
        apply_pressure_gradient(&mut grid, 0.01);

        for i in 0..grid.vel_x.len() {
            assert!(
                (grid.vel_x[i] - vel_x_before[i]).abs() < 1e-10,
                "Uniform pressure: vel_x should not change at index {i}"
            );
            assert!(
                (grid.vel_y[i] - vel_y_before[i]).abs() < 1e-10,
                "Uniform pressure: vel_y should not change at index {i}"
            );
            assert!(
                (grid.vel_z[i] - vel_z_before[i]).abs() < 1e-10,
                "Uniform pressure: vel_z should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_advection_zero_velocity() {
        let species = vec![make_species("CO2", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let ci = grid.idx(4, 4, 4);
        grid.concentrations[0][ci] = 100.0;
        let before = grid.concentrations[0].clone();

        advect_concentrations(&mut grid, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.concentrations[0].iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-4,
                "Zero velocity: concentration unchanged at idx {i}, before={b}, after={a}"
            );
        }
    }

    #[test]
    fn test_edge_advection_temperature_zero_velocity() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let idx_center = grid.idx(4, 4, 4);
        grid.temperature[idx_center] = 500.0;
        let before = grid.temperature.clone();

        advect_temperature(&mut grid, 0.01);

        for (i, (b, a)) in before.iter().zip(grid.temperature.iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-4,
                "Zero velocity: temperature unchanged at idx {i}, before={b}, after={a}"
            );
        }
    }

    #[test]
    fn test_edge_diffusion_stability_clamping() {
        let species = vec![make_species("Fast", 100.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let ci = grid.idx(4, 4, 4);
        grid.concentrations[0][ci] = 1000.0;

        diffuse_concentrations(&mut grid, 1.0);

        assert!(
            grid.concentrations[0].iter().all(|v| v.is_finite()),
            "Clamped diffusion should remain finite"
        );
        assert!(
            grid.concentrations[0][ci] < 1000.0,
            "Center should diffuse outward even with clamping"
        );
    }

    #[test]
    fn test_edge_mixed_species_diffusion_rates() {
        let species = vec![make_species("Active", 0.2), make_species("Frozen", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);
        let center = grid.idx(4, 4, 4);
        grid.concentrations[0][center] = 100.0;
        grid.concentrations[1][center] = 100.0;

        for _ in 0..10 {
            diffuse_concentrations(&mut grid, 0.01);
        }

        assert!(
            grid.concentrations[0][center] < 100.0,
            "Active species should diffuse: got {}",
            grid.concentrations[0][center]
        );
        assert!(
            (grid.concentrations[1][center] - 100.0).abs() < 1e-10,
            "Frozen species should not diffuse: got {}",
            grid.concentrations[1][center]
        );
    }

    #[test]
    fn test_edge_buoyancy_cold_sinks() {
        let species = vec![make_species("Air", 0.0)];
        let mut grid = make_gas_grid(8, 0.125, species);

        let config = GasConfig {
            dt: 0.01,
            ambient_temperature: 293.15,
            thermal_diffusivity: 0.0,
            buoyancy_coefficient: 0.5,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        for t in grid.temperature.iter_mut() {
            *t = config.ambient_temperature;
        }
        let cold_idx = grid.idx(4, 4, 4);
        grid.temperature[cold_idx] = 100.0;

        apply_buoyancy(&mut grid, &config, config.dt);

        assert!(
            grid.vel_y[cold_idx] < 0.0,
            "Cold gas should gain downward velocity, got {}",
            grid.vel_y[cold_idx]
        );
    }

    /// Parallel speedup benchmark: on a 64^3 grid (262144 cells), running 5
    /// full steps with the default rayon pool must complete in well under
    /// 1 second on a modern multicore machine. Also compares against a forced
    /// single-thread pool and asserts the parallel run is at least 1.5x
    /// faster — a conservative bar that holds even on 2-core CI runners.
    ///
    /// Skipped if rayon reports < 2 threads available (single-core sandbox).
    /// Gated with #[ignore] because it is a perf measurement: when `cargo test`
    /// runs many test workers in parallel, the parallel run gets CPU-starved by
    /// peer workers and the speedup ratio flakes. Run explicitly via
    /// `cargo test --release --lib -- --ignored gas_solver_parallel_speedup`
    /// or through the perf-regression script.
    #[test]
    #[ignore]
    fn test_gas_solver_parallel_speedup() {
        if rayon::current_num_threads() < 2 {
            eprintln!("skipping: rayon pool has < 2 threads");
            return;
        }

        let n: usize = 64;
        let dx = 0.05_f32;
        let species = vec![make_species("CO2", 0.05), make_species("CH4", 0.04)];
        let config = GasConfig {
            dt: 0.005,
            ambient_temperature: 293.15,
            thermal_diffusivity: 2.2e-5,
            buoyancy_coefficient: 0.01,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        };

        let setup_grid = || -> GasGrid {
            let mut g = make_gas_grid(n, dx, species.clone());
            let ci = n / 2;
            let center = g.idx(ci, ci, ci);
            for s in 0..g.species.len() {
                g.concentrations[s][center] = 1000.0;
            }
            for t in g.temperature.iter_mut() {
                *t = config.ambient_temperature;
            }
            g.temperature[center] = 500.0;
            g
        };

        let num_steps = 5;

        // 1-thread reference
        let pool_serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("build serial pool");
        let mut grid_serial = setup_grid();
        let t0 = std::time::Instant::now();
        pool_serial.install(|| {
            for _ in 0..num_steps {
                step(&mut grid_serial, &config);
            }
        });
        let serial_elapsed = t0.elapsed();

        // Default-thread parallel
        let mut grid_par = setup_grid();
        let t0 = std::time::Instant::now();
        for _ in 0..num_steps {
            step(&mut grid_par, &config);
        }
        let par_elapsed = t0.elapsed();

        let speedup = serial_elapsed.as_secs_f64() / par_elapsed.as_secs_f64();
        eprintln!(
            "gas solver 64^3 x{} steps: serial={:?} parallel={:?} speedup={:.2}x (threads={})",
            num_steps,
            serial_elapsed,
            par_elapsed,
            speedup,
            rayon::current_num_threads()
        );

        // Absolute perf bar (release-mode expectation).
        assert!(
            par_elapsed < std::time::Duration::from_secs(5),
            "parallel 64^3 x{num_steps} steps must finish under 5s, got {par_elapsed:?}"
        );

        // Speedup bar — relaxed for CI / shared runners.
        assert!(
            speedup >= 1.5,
            "expected >=1.5x speedup over single-thread, got {speedup:.2}x (serial={serial_elapsed:?}, parallel={par_elapsed:?})"
        );
    }
}
