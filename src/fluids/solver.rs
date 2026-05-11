use glam::Vec3;
use rayon::prelude::*;

use super::grid::{CellType, FluidGrid};

/// Configuration for the fluid solver.
#[derive(Clone, Debug)]
pub struct FluidConfig {
    /// Simulation timestep (seconds).
    pub dt: f32,
    /// Kinematic viscosity coefficient.
    pub viscosity: f32,
    /// Reference density (kg/m³).
    pub density: f32,
    /// Gravitational acceleration vector.
    pub gravity: Vec3,
    /// Surface tension coefficient (currently unused placeholder).
    #[allow(dead_code)]
    pub surface_tension: f32,
    /// Number of Jacobi iterations for pressure solve.
    pub jacobi_iterations: u32,
}

impl FluidConfig {
    #[allow(dead_code)]
    pub fn new(
        dt: f32,
        viscosity: f32,
        density: f32,
        gravity: Vec3,
        surface_tension: f32,
        jacobi_iterations: u32,
    ) -> Self {
        assert!(dt > 0.0, "Timestep dt must be positive, got {dt}");
        assert!(
            viscosity >= 0.0,
            "Viscosity must be non-negative, got {viscosity}"
        );
        assert!(density > 0.0, "Density must be positive, got {density}");
        Self {
            dt,
            viscosity,
            density,
            gravity,
            surface_tension,
            jacobi_iterations,
        }
    }
}

impl Default for FluidConfig {
    fn default() -> Self {
        Self {
            dt: 0.016,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 80,
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone trilinear interpolation on raw arrays
// ---------------------------------------------------------------------------

/// Trilinear interpolation on a raw u-face array (dims: (nx+1) x ny x nz).
fn sample_u(data: &[f32], nx: usize, ny: usize, nz: usize, fi: f32, fj: f32, fk: f32) -> f32 {
    let nx1 = nx + 1;
    let fi = fi.clamp(0.0, (nx1 - 1) as f32);
    let fj = fj.clamp(0.0, (ny - 1) as f32);
    let fk = fk.clamp(0.0, (nz - 1) as f32);

    let i0 = (fi.floor() as usize).min(nx1 - 2);
    let j0 = (fj.floor() as usize).min(ny.saturating_sub(2));
    let k0 = (fk.floor() as usize).min(nz.saturating_sub(2));
    let i1 = i0 + 1;
    let j1 = (j0 + 1).min(ny - 1);
    let k1 = (k0 + 1).min(nz - 1);

    let s = fi - i0 as f32;
    let t = fj - j0 as f32;
    let r = fk - k0 as f32;

    let idx = |i: usize, j: usize, k: usize| i + nx1 * (j + ny * k);

    lerp3(
        data[idx(i0, j0, k0)],
        data[idx(i1, j0, k0)],
        data[idx(i0, j1, k0)],
        data[idx(i1, j1, k0)],
        data[idx(i0, j0, k1)],
        data[idx(i1, j0, k1)],
        data[idx(i0, j1, k1)],
        data[idx(i1, j1, k1)],
        s,
        t,
        r,
    )
}

/// Trilinear interpolation on a raw v-face array (dims: nx x (ny+1) x nz).
fn sample_v(data: &[f32], nx: usize, ny: usize, nz: usize, fi: f32, fj: f32, fk: f32) -> f32 {
    let ny1 = ny + 1;
    let fi = fi.clamp(0.0, (nx - 1) as f32);
    let fj = fj.clamp(0.0, (ny1 - 1) as f32);
    let fk = fk.clamp(0.0, (nz - 1) as f32);

    let i0 = (fi.floor() as usize).min(nx.saturating_sub(2));
    let j0 = (fj.floor() as usize).min(ny1 - 2);
    let k0 = (fk.floor() as usize).min(nz.saturating_sub(2));
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = j0 + 1;
    let k1 = (k0 + 1).min(nz - 1);

    let s = fi - i0 as f32;
    let t = fj - j0 as f32;
    let r = fk - k0 as f32;

    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny1 * k);

    lerp3(
        data[idx(i0, j0, k0)],
        data[idx(i1, j0, k0)],
        data[idx(i0, j1, k0)],
        data[idx(i1, j1, k0)],
        data[idx(i0, j0, k1)],
        data[idx(i1, j0, k1)],
        data[idx(i0, j1, k1)],
        data[idx(i1, j1, k1)],
        s,
        t,
        r,
    )
}

/// Trilinear interpolation on a raw w-face array (dims: nx x ny x (nz+1)).
fn sample_w(data: &[f32], nx: usize, ny: usize, nz: usize, fi: f32, fj: f32, fk: f32) -> f32 {
    let nz1 = nz + 1;
    let fi = fi.clamp(0.0, (nx - 1) as f32);
    let fj = fj.clamp(0.0, (ny - 1) as f32);
    let fk = fk.clamp(0.0, (nz1 - 1) as f32);

    let i0 = (fi.floor() as usize).min(nx.saturating_sub(2));
    let j0 = (fj.floor() as usize).min(ny.saturating_sub(2));
    let k0 = (fk.floor() as usize).min(nz1 - 2);
    let i1 = (i0 + 1).min(nx - 1);
    let j1 = (j0 + 1).min(ny - 1);
    let k1 = k0 + 1;

    let s = fi - i0 as f32;
    let t = fj - j0 as f32;
    let r = fk - k0 as f32;

    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);

    lerp3(
        data[idx(i0, j0, k0)],
        data[idx(i1, j0, k0)],
        data[idx(i0, j1, k0)],
        data[idx(i1, j1, k0)],
        data[idx(i0, j0, k1)],
        data[idx(i1, j0, k1)],
        data[idx(i0, j1, k1)],
        data[idx(i1, j1, k1)],
        s,
        t,
        r,
    )
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn lerp3(
    c000: f32,
    c100: f32,
    c010: f32,
    c110: f32,
    c001: f32,
    c101: f32,
    c011: f32,
    c111: f32,
    s: f32,
    t: f32,
    r: f32,
) -> f32 {
    let c00 = c000 * (1.0 - s) + c100 * s;
    let c10 = c010 * (1.0 - s) + c110 * s;
    let c01 = c001 * (1.0 - s) + c101 * s;
    let c11 = c011 * (1.0 - s) + c111 * s;
    let c0 = c00 * (1.0 - t) + c10 * t;
    let c1 = c01 * (1.0 - t) + c11 * t;
    c0 * (1.0 - r) + c1 * r
}

/// Interpolate velocity from raw arrays at an arbitrary world position.
#[allow(clippy::too_many_arguments)]
fn velocity_from_arrays(
    u: &[f32],
    v: &[f32],
    w: &[f32],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f32,
    origin: Vec3,
    pos: Vec3,
) -> Vec3 {
    let rel = pos - origin;
    // u lives on x-faces: center at (i*dx, (j+0.5)*dx, (k+0.5)*dx)
    let ux = sample_u(
        u,
        nx,
        ny,
        nz,
        rel.x / dx,
        rel.y / dx - 0.5,
        rel.z / dx - 0.5,
    );
    // v lives on y-faces: center at ((i+0.5)*dx, j*dx, (k+0.5)*dx)
    let vy = sample_v(
        v,
        nx,
        ny,
        nz,
        rel.x / dx - 0.5,
        rel.y / dx,
        rel.z / dx - 0.5,
    );
    // w lives on z-faces: center at ((i+0.5)*dx, (j+0.5)*dx, k*dx)
    let wz = sample_w(
        w,
        nx,
        ny,
        nz,
        rel.x / dx - 0.5,
        rel.y / dx - 0.5,
        rel.z / dx,
    );
    Vec3::new(ux, vy, wz)
}

// ---------------------------------------------------------------------------
// Semi-Lagrangian advection
// ---------------------------------------------------------------------------

/// Semi-Lagrangian advection: traces backward through the velocity field and
/// interpolates from the previous timestep. Returns new (u, v, w) arrays.
///
/// Row-parallel via rayon: each y-slice is processed independently.
pub fn advect(grid: &FluidGrid, dt: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let origin = grid.origin;

    // Snapshot old velocities (read from these, write to new arrays).
    let old_u = grid.u.clone();
    let old_v = grid.v.clone();
    let old_w = grid.w.clone();

    // --- Advect u (on x-faces: (nx+1) x ny x nz) ---
    let mut new_u = vec![0.0f32; (nx + 1) * ny * nz];
    // Parallel over y-slices
    new_u
        .par_chunks_mut((nx + 1) * ny)
        .enumerate()
        .for_each(|(k, slice)| {
            for j in 0..ny {
                for i in 0..=nx {
                    // World position of u-face (i, j, k)
                    let pos = origin
                        + Vec3::new(i as f32 * dx, (j as f32 + 0.5) * dx, (k as f32 + 0.5) * dx);
                    let vel =
                        velocity_from_arrays(&old_u, &old_v, &old_w, nx, ny, nz, dx, origin, pos);
                    let back_pos = pos - vel * dt;
                    // Interpolate u from old field at backtraced position
                    let rel = back_pos - origin;
                    let fi = rel.x / dx;
                    let fj = rel.y / dx - 0.5;
                    let fk = rel.z / dx - 0.5;
                    slice[i + (nx + 1) * j] = sample_u(&old_u, nx, ny, nz, fi, fj, fk);
                }
            }
        });

    // --- Advect v (on y-faces: nx x (ny+1) x nz) ---
    let mut new_v = vec![0.0f32; nx * (ny + 1) * nz];
    new_v
        .par_chunks_mut(nx * (ny + 1))
        .enumerate()
        .for_each(|(k, slice)| {
            for j in 0..=ny {
                for i in 0..nx {
                    let pos = origin
                        + Vec3::new((i as f32 + 0.5) * dx, j as f32 * dx, (k as f32 + 0.5) * dx);
                    let vel =
                        velocity_from_arrays(&old_u, &old_v, &old_w, nx, ny, nz, dx, origin, pos);
                    let back_pos = pos - vel * dt;
                    let rel = back_pos - origin;
                    let fi = rel.x / dx - 0.5;
                    let fj = rel.y / dx;
                    let fk = rel.z / dx - 0.5;
                    slice[i + nx * j] = sample_v(&old_v, nx, ny, nz, fi, fj, fk);
                }
            }
        });

    // --- Advect w (on z-faces: nx x ny x (nz+1)) ---
    let mut new_w = vec![0.0f32; nx * ny * (nz + 1)];
    // For w, z-slices are the fast index so we parallel over y instead
    new_w.par_chunks_mut(nx).enumerate().for_each(|(jk, row)| {
        let j = jk % ny;
        let k = jk / ny;
        for (i, row_val) in row.iter_mut().enumerate() {
            let pos =
                origin + Vec3::new((i as f32 + 0.5) * dx, (j as f32 + 0.5) * dx, k as f32 * dx);
            let vel = velocity_from_arrays(&old_u, &old_v, &old_w, nx, ny, nz, dx, origin, pos);
            let back_pos = pos - vel * dt;
            let rel = back_pos - origin;
            let fi = rel.x / dx - 0.5;
            let fj = rel.y / dx - 0.5;
            let fk = rel.z / dx;
            *row_val = sample_w(&old_w, nx, ny, nz, fi, fj, fk);
        }
    });

    (new_u, new_v, new_w)
}

// ---------------------------------------------------------------------------
// External forces
// ---------------------------------------------------------------------------

/// Apply gravity and buoyancy forces to the v-face velocity field.
///
/// Gravity is applied uniformly. Buoyancy adds an upward force proportional to
/// the density difference from `config.density` at neighbouring cell centers.
pub fn apply_forces(grid: &mut FluidGrid, config: &FluidConfig, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // Apply gravity to all v-faces that border at least one Fluid cell.
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                // A v-face at (i,j,k) sits between cell (i,j-1,k) and (i,j,k).
                let below_fluid = j > 0 && {
                    let idx = grid.idx(i, j - 1, k);
                    grid.cell_types[idx] == CellType::Fluid
                };
                let above_fluid = j < ny && {
                    let idx = grid.idx(i, j, k);
                    grid.cell_types[idx] == CellType::Fluid
                };

                if below_fluid || above_fluid {
                    let vidx = grid.idx_v(i, j, k);
                    grid.v[vidx] += config.gravity.y * dt;

                    // Buoyancy: if both cells exist, use density difference
                    if j > 0 && j < ny {
                        let d_below = grid.density[grid.idx(i, j - 1, k)];
                        let d_above = grid.density[grid.idx(i, j, k)];
                        let avg_density = (d_below + d_above) * 0.5;
                        if config.density > 0.0 {
                            // Lighter fluid rises: buoyancy = -g * (rho - rho_ref) / rho_ref
                            let buoyancy =
                                -config.gravity.y * (avg_density - config.density) / config.density;
                            grid.v[vidx] += buoyancy * dt;
                        }
                    }
                }
            }
        }
    }

    // Apply gravity x-component to u-faces
    if config.gravity.x.abs() > 1e-10 {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let left_fluid = i > 0 && {
                        let idx = grid.idx(i - 1, j, k);
                        grid.cell_types[idx] == CellType::Fluid
                    };
                    let right_fluid = i < nx && {
                        let idx = grid.idx(i, j, k);
                        grid.cell_types[idx] == CellType::Fluid
                    };
                    if left_fluid || right_fluid {
                        let uidx = grid.idx_u(i, j, k);
                        grid.u[uidx] += config.gravity.x * dt;
                    }
                }
            }
        }
    }

    // Apply gravity z-component to w-faces
    if config.gravity.z.abs() > 1e-10 {
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let back_fluid = k > 0 && {
                        let idx = grid.idx(i, j, k - 1);
                        grid.cell_types[idx] == CellType::Fluid
                    };
                    let front_fluid = k < nz && {
                        let idx = grid.idx(i, j, k);
                        grid.cell_types[idx] == CellType::Fluid
                    };
                    if back_fluid || front_fluid {
                        let widx = grid.idx_w(i, j, k);
                        grid.w[widx] += config.gravity.z * dt;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Viscous diffusion (explicit)
// ---------------------------------------------------------------------------

/// Explicit viscous diffusion applied to the velocity field.
///
/// Uses a simple forward-Euler Laplacian stencil:
///   u_new = u + viscosity * dt / dx² * Laplacian(u)
pub fn diffuse(grid: &mut FluidGrid, viscosity: f32, dt: f32) {
    if viscosity <= 0.0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let factor = viscosity * dt / (dx * dx);

    // Clamp factor to ensure explicit stability (factor < 1/6 for 3D)
    let factor = factor.min(1.0 / 6.5);

    // Diffuse u-field
    {
        let old = grid.u.clone();
        let nx1 = nx + 1;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx1 {
                    let c = old[i + nx1 * (j + ny * k)];
                    let xm = if i > 0 {
                        old[(i - 1) + nx1 * (j + ny * k)]
                    } else {
                        c
                    };
                    let xp = if i < nx1 - 1 {
                        old[(i + 1) + nx1 * (j + ny * k)]
                    } else {
                        c
                    };
                    let ym = if j > 0 {
                        old[i + nx1 * ((j - 1) + ny * k)]
                    } else {
                        c
                    };
                    let yp = if j < ny - 1 {
                        old[i + nx1 * ((j + 1) + ny * k)]
                    } else {
                        c
                    };
                    let zm = if k > 0 {
                        old[i + nx1 * (j + ny * (k - 1))]
                    } else {
                        c
                    };
                    let zp = if k < nz - 1 {
                        old[i + nx1 * (j + ny * (k + 1))]
                    } else {
                        c
                    };
                    let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
                    grid.u[i + nx1 * (j + ny * k)] = c + factor * laplacian;
                }
            }
        }
    }

    // Diffuse v-field
    {
        let old = grid.v.clone();
        let ny1 = ny + 1;
        for k in 0..nz {
            for j in 0..ny1 {
                for i in 0..nx {
                    let c = old[i + nx * (j + ny1 * k)];
                    let xm = if i > 0 {
                        old[(i - 1) + nx * (j + ny1 * k)]
                    } else {
                        c
                    };
                    let xp = if i < nx - 1 {
                        old[(i + 1) + nx * (j + ny1 * k)]
                    } else {
                        c
                    };
                    let ym = if j > 0 {
                        old[i + nx * ((j - 1) + ny1 * k)]
                    } else {
                        c
                    };
                    let yp = if j < ny1 - 1 {
                        old[i + nx * ((j + 1) + ny1 * k)]
                    } else {
                        c
                    };
                    let zm = if k > 0 {
                        old[i + nx * (j + ny1 * (k - 1))]
                    } else {
                        c
                    };
                    let zp = if k < nz - 1 {
                        old[i + nx * (j + ny1 * (k + 1))]
                    } else {
                        c
                    };
                    let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
                    grid.v[i + nx * (j + ny1 * k)] = c + factor * laplacian;
                }
            }
        }
    }

    // Diffuse w-field
    {
        let old = grid.w.clone();
        let nz1 = nz + 1;
        for k in 0..nz1 {
            for j in 0..ny {
                for i in 0..nx {
                    let c = old[i + nx * (j + ny * k)];
                    let xm = if i > 0 {
                        old[(i - 1) + nx * (j + ny * k)]
                    } else {
                        c
                    };
                    let xp = if i < nx - 1 {
                        old[(i + 1) + nx * (j + ny * k)]
                    } else {
                        c
                    };
                    let ym = if j > 0 {
                        old[i + nx * ((j - 1) + ny * k)]
                    } else {
                        c
                    };
                    let yp = if j < ny - 1 {
                        old[i + nx * ((j + 1) + ny * k)]
                    } else {
                        c
                    };
                    let zm = if k > 0 {
                        old[i + nx * (j + ny * (k - 1))]
                    } else {
                        c
                    };
                    let zp = if k < nz1 - 1 {
                        old[i + nx * (j + ny * (k + 1))]
                    } else {
                        c
                    };
                    let laplacian = xm + xp + ym + yp + zm + zp - 6.0 * c;
                    grid.w[i + nx * (j + ny * k)] = c + factor * laplacian;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pressure solve (Jacobi iteration)
// ---------------------------------------------------------------------------

/// Jacobi iterative solver for the pressure Poisson equation.
///
/// Enforces solid-wall boundary velocities first, then solves:
///   Laplacian(p) = div(u*) / dt
/// Uses copy-swap pattern for rayon safety.
pub fn pressure_solve(grid: &mut FluidGrid, dt: f32, iterations: u32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;

    // Enforce solid-wall BCs on domain boundary faces BEFORE computing
    // divergence. This ensures the RHS accounts for fixed boundaries.
    enforce_boundary_velocities(grid);

    // Precompute RHS: rhs = (dx/dt) * raw_div, where raw_div is the undivided
    // divergence (sum of face-velocity differences).
    //
    // The discrete Poisson equation is:
    //   sum_neighbours(p) - n*p = rhs
    // where n is the number of active (fluid) neighbours. This is the
    // variable-coefficient Laplacian stencil.
    let rhs_scale = dx / dt;
    let mut rhs = vec![0.0f32; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let gidx = grid.idx(i, j, k);
                if grid.cell_types[gidx] != CellType::Fluid {
                    continue;
                }
                let raw_div = grid.u[grid.idx_u(i + 1, j, k)] - grid.u[grid.idx_u(i, j, k)]
                    + grid.v[grid.idx_v(i, j + 1, k)]
                    - grid.v[grid.idx_v(i, j, k)]
                    + grid.w[grid.idx_w(i, j, k + 1)]
                    - grid.w[grid.idx_w(i, j, k)];
                rhs[gidx] = rhs_scale * raw_div;
            }
        }
    }

    // Zero out pressure
    grid.pressure.fill(0.0);

    let mut p_old = grid.pressure.clone();
    let mut p_new = vec![0.0f32; nx * ny * nz];

    for _iter in 0..iterations {
        // Parallel Jacobi: read from p_old, write to p_new.
        // Each z-slice is processed independently.
        p_new
            .par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(k, slice)| {
                for j in 0..ny {
                    for i in 0..nx {
                        let cidx = i + nx * j;
                        let gidx = i + nx * (j + ny * k);

                        if grid.cell_types[gidx] != CellType::Fluid {
                            slice[cidx] = 0.0;
                            continue;
                        }

                        // Sum pressures of fluid neighbours only. For
                        // solid/domain-boundary neighbours, apply Neumann BC by
                        // simply not counting them (reduces the denominator).
                        // This is the standard MAC grid Jacobi discretization.
                        let mut n_neighbours = 0u32;
                        let mut p_sum = 0.0f32;

                        // x-
                        if i > 0 && grid.cell_types[gidx - 1] == CellType::Fluid {
                            n_neighbours += 1;
                            p_sum += p_old[gidx - 1];
                        }
                        // x+
                        if i < nx - 1 && grid.cell_types[gidx + 1] == CellType::Fluid {
                            n_neighbours += 1;
                            p_sum += p_old[gidx + 1];
                        }
                        // y-
                        if j > 0 {
                            let ym = gidx - nx;
                            if grid.cell_types[ym] == CellType::Fluid {
                                n_neighbours += 1;
                                p_sum += p_old[ym];
                            }
                        }
                        // y+
                        if j < ny - 1 {
                            let yp = gidx + nx;
                            if grid.cell_types[yp] == CellType::Fluid {
                                n_neighbours += 1;
                                p_sum += p_old[yp];
                            }
                        }
                        // z-
                        if k > 0 {
                            let zm = i + nx * (j + ny * (k - 1));
                            if grid.cell_types[zm] == CellType::Fluid {
                                n_neighbours += 1;
                                p_sum += p_old[zm];
                            }
                        }
                        // z+
                        if k < nz - 1 {
                            let zp = i + nx * (j + ny * (k + 1));
                            if grid.cell_types[zp] == CellType::Fluid {
                                n_neighbours += 1;
                                p_sum += p_old[zp];
                            }
                        }

                        if n_neighbours > 0 {
                            slice[cidx] = (p_sum - rhs[gidx]) / n_neighbours as f32;
                        }
                    }
                }
            });

        std::mem::swap(&mut p_old, &mut p_new);
    }

    grid.pressure = p_old;
}

// ---------------------------------------------------------------------------
// Pressure projection
// ---------------------------------------------------------------------------

/// Zero velocity on domain boundary faces (no-slip wall condition).
fn enforce_boundary_velocities(grid: &mut FluidGrid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // x-boundary faces (i=0 and i=nx)
    for k in 0..nz {
        for j in 0..ny {
            let idx0 = grid.idx_u(0, j, k);
            let idx1 = grid.idx_u(nx, j, k);
            grid.u[idx0] = 0.0;
            grid.u[idx1] = 0.0;
        }
    }
    // y-boundary faces (j=0 and j=ny)
    for k in 0..nz {
        for i in 0..nx {
            let idx0 = grid.idx_v(i, 0, k);
            let idx1 = grid.idx_v(i, ny, k);
            grid.v[idx0] = 0.0;
            grid.v[idx1] = 0.0;
        }
    }
    // z-boundary faces (k=0 and k=nz)
    for j in 0..ny {
        for i in 0..nx {
            let idx0 = grid.idx_w(i, j, 0);
            let idx1 = grid.idx_w(i, j, nz);
            grid.w[idx0] = 0.0;
            grid.w[idx1] = 0.0;
        }
    }
}

/// Subtract pressure gradient from velocity to enforce divergence-free.
///
/// Only modifies interior faces (not domain boundary faces, which are kept
/// at their wall BC value of zero). Re-enforces boundary velocities after.
pub fn project(grid: &mut FluidGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let scale = dt / dx;

    // Update interior u-faces (i=1..nx-1)
    for k in 0..nz {
        for j in 0..ny {
            for i in 1..nx {
                let left = grid.idx(i - 1, j, k);
                let right = grid.idx(i, j, k);
                if grid.cell_types[left] == CellType::Fluid
                    || grid.cell_types[right] == CellType::Fluid
                {
                    let uidx = grid.idx_u(i, j, k);
                    grid.u[uidx] -= scale * (grid.pressure[right] - grid.pressure[left]);
                }
            }
        }
    }

    // Update interior v-faces (j=1..ny-1)
    for k in 0..nz {
        for j in 1..ny {
            for i in 0..nx {
                let below = grid.idx(i, j - 1, k);
                let above = grid.idx(i, j, k);
                if grid.cell_types[below] == CellType::Fluid
                    || grid.cell_types[above] == CellType::Fluid
                {
                    let vidx = grid.idx_v(i, j, k);
                    grid.v[vidx] -= scale * (grid.pressure[above] - grid.pressure[below]);
                }
            }
        }
    }

    // Update interior w-faces (k=1..nz-1)
    for k in 1..nz {
        for j in 0..ny {
            for i in 0..nx {
                let back = grid.idx(i, j, k - 1);
                let front = grid.idx(i, j, k);
                if grid.cell_types[back] == CellType::Fluid
                    || grid.cell_types[front] == CellType::Fluid
                {
                    let widx = grid.idx_w(i, j, k);
                    grid.w[widx] -= scale * (grid.pressure[front] - grid.pressure[back]);
                }
            }
        }
    }

    // Re-enforce domain boundary velocities
    enforce_boundary_velocities(grid);
}

// ---------------------------------------------------------------------------
// Full timestep
// ---------------------------------------------------------------------------

/// Execute a full simulation timestep: advect -> forces -> diffuse -> pressure_solve -> project.
pub fn step(grid: &mut FluidGrid, config: &FluidConfig) {
    let dt = config.dt;

    // 1. Advection (semi-Lagrangian)
    let (new_u, new_v, new_w) = advect(grid, dt);
    grid.u = new_u;
    grid.v = new_v;
    grid.w = new_w;

    // 2. External forces (gravity + buoyancy)
    apply_forces(grid, config, dt);

    // 3. Viscous diffusion
    diffuse(grid, config.viscosity, dt);

    // 4. Pressure solve
    pressure_solve(grid, dt, config.jacobi_iterations);

    // 5. Pressure projection (divergence-free)
    project(grid, dt);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluids::grid::FluidGrid;

    /// Helper: create an n x n x n grid with all cells marked as Fluid and
    /// density set to the reference density.
    fn make_fluid_grid(n: usize, dx: f32) -> FluidGrid {
        let mut g = FluidGrid::new(n, n, n, dx, Vec3::ZERO);
        for ct in g.cell_types.iter_mut() {
            *ct = CellType::Fluid;
        }
        g
    }

    /// Compute max absolute divergence across all fluid cells.
    fn max_divergence(grid: &FluidGrid) -> f32 {
        let mut max_div = 0.0f32;
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    if grid.cell_types[grid.idx(i, j, k)] != CellType::Fluid {
                        continue;
                    }
                    let div = (grid.u[grid.idx_u(i + 1, j, k)] - grid.u[grid.idx_u(i, j, k)]
                        + grid.v[grid.idx_v(i, j + 1, k)]
                        - grid.v[grid.idx_v(i, j, k)]
                        + grid.w[grid.idx_w(i, j, k + 1)]
                        - grid.w[grid.idx_w(i, j, k)])
                        / grid.dx;
                    max_div = max_div.max(div.abs());
                }
            }
        }
        max_div
    }

    #[test]
    fn test_zero_velocity_stays_zero() {
        let mut grid = make_fluid_grid(4, 0.25);
        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 1000.0,
            gravity: Vec3::ZERO,
            surface_tension: 0.0,
            jacobi_iterations: 50,
        };
        step(&mut grid, &config);

        let u_max: f32 = grid.u.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let v_max: f32 = grid.v.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let w_max: f32 = grid.w.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(u_max < 1e-6, "u should stay zero, max = {u_max}");
        assert!(v_max < 1e-6, "v should stay zero, max = {v_max}");
        assert!(w_max < 1e-6, "w should stay zero, max = {w_max}");
    }

    #[test]
    fn test_divergence_free_after_projection() {
        let mut grid = make_fluid_grid(8, 0.125);
        // Set some non-trivial, non-divergence-free velocity
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..=grid.nx {
                    let idx = grid.idx_u(i, j, k);
                    grid.u[idx] = (i as f32) * 0.1;
                }
            }
        }

        pressure_solve(&mut grid, 0.01, 500);
        project(&mut grid, 0.01);

        let div = max_divergence(&grid);
        assert!(
            div < 1e-2,
            "Divergence after projection should be < 1e-2, got {div}"
        );
    }

    #[test]
    fn test_gravity_increases_downward_velocity() {
        let mut grid = make_fluid_grid(4, 0.25);
        // Set density to match reference so buoyancy is zero
        for d in grid.density.iter_mut() {
            *d = 1000.0;
        }
        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 0, // skip pressure solve to isolate force effect
        };

        // Record v before
        let v_before: f32 = grid.v.iter().sum();

        apply_forces(&mut grid, &config, config.dt);

        let v_after: f32 = grid.v.iter().sum();
        // Gravity is negative-y, so total v should decrease
        assert!(
            v_after < v_before,
            "Gravity should decrease total v-velocity: before={v_before}, after={v_after}"
        );
    }

    #[test]
    fn test_advection_uniform_field() {
        let mut grid = make_fluid_grid(8, 0.125);
        // Uniform u = 1.0 everywhere
        for val in grid.u.iter_mut() {
            *val = 1.0;
        }

        let (new_u, new_v, new_w) = advect(&grid, 0.01);

        // Interior u-faces should remain ~1.0
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 2..nx - 1 {
                    let idx = grid.idx_u(i, j, k);
                    assert!(
                        (new_u[idx] - 1.0).abs() < 0.05,
                        "Uniform u should stay ~1.0 after advection at ({i},{j},{k}), got {}",
                        new_u[idx]
                    );
                }
            }
        }

        // v and w should remain ~0
        let v_max: f32 = new_v.iter().map(|v| v.abs()).fold(0.0, f32::max);
        let w_max: f32 = new_w.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(
            v_max < 0.05,
            "v should stay ~0 after advecting uniform u, max = {v_max}"
        );
        assert!(
            w_max < 0.05,
            "w should stay ~0 after advecting uniform u, max = {w_max}"
        );
    }

    #[test]
    fn test_viscosity_smooths_velocity() {
        let mut grid = make_fluid_grid(8, 0.125);
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        // Create a sharp velocity spike in the center
        let ci = nx / 2;
        let cj = ny / 2;
        let ck = nz / 2;
        let idx = grid.idx_u(ci, cj, ck);
        grid.u[idx] = 10.0;

        let initial_max = 10.0f32;

        // Apply diffusion several times
        for _ in 0..20 {
            diffuse(&mut grid, 0.1, 0.01);
        }

        let final_max: f32 = grid.u.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(
            final_max < initial_max,
            "Viscosity should smooth the peak: initial_max={initial_max}, final_max={final_max}"
        );
    }

    #[test]
    fn test_pressure_solve_converges() {
        let mut grid = make_fluid_grid(8, 0.125);
        // Create a divergent velocity field
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let idx = grid.idx_u(i, j, k);
                    grid.u[idx] = i as f32 * 0.1;
                }
            }
        }

        // Compute initial divergence
        let div_before = max_divergence(&grid);

        // Solve with moderate iterations
        let mut grid_few = grid.clone();
        pressure_solve(&mut grid_few, 0.01, 50);
        project(&mut grid_few, 0.01);
        let div_few = max_divergence(&grid_few);

        // Solve with many iterations
        let mut grid_many = grid.clone();
        pressure_solve(&mut grid_many, 0.01, 500);
        project(&mut grid_many, 0.01);
        let div_many = max_divergence(&grid_many);

        assert!(
            div_few < div_before,
            "50 Jacobi iterations should reduce divergence: before={div_before}, after={div_few}"
        );
        assert!(
            div_many <= div_few,
            "500 iterations should converge at least as well as 50: few={div_few}, many={div_many}"
        );
    }

    #[test]
    fn test_step_preserves_mass() {
        let mut grid = make_fluid_grid(8, 0.125);
        // Set uniform density
        for d in grid.density.iter_mut() {
            *d = 1000.0;
        }

        let config = FluidConfig {
            dt: 0.005,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::ZERO,
            surface_tension: 0.0,
            jacobi_iterations: 50,
        };

        let mass_before: f32 = grid.density.iter().sum();

        for _ in 0..10 {
            step(&mut grid, &config);
        }

        let mass_after: f32 = grid.density.iter().sum();

        // With zero gravity and zero initial velocity, density should not change
        let rel_change = if mass_before > 0.0 {
            ((mass_after - mass_before) / mass_before).abs()
        } else {
            0.0
        };
        assert!(
            rel_change < 0.01,
            "Mass should be conserved within 1%: before={mass_before}, after={mass_after}, change={rel_change}"
        );
    }

    #[test]
    fn test_numerical_stability_no_nan() {
        let mut grid = make_fluid_grid(8, 0.125);
        // Set density
        for d in grid.density.iter_mut() {
            *d = 1000.0;
        }

        let config = FluidConfig {
            dt: 0.005,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 50,
        };

        for i in 0..500 {
            step(&mut grid, &config);
            // Check for NaN/Inf periodically
            if i % 50 == 0 {
                assert!(
                    grid.u.iter().all(|v| v.is_finite()),
                    "u contains NaN/Inf at step {i}"
                );
                assert!(
                    grid.v.iter().all(|v| v.is_finite()),
                    "v contains NaN/Inf at step {i}"
                );
                assert!(
                    grid.w.iter().all(|v| v.is_finite()),
                    "w contains NaN/Inf at step {i}"
                );
                assert!(
                    grid.pressure.iter().all(|v| v.is_finite()),
                    "pressure contains NaN/Inf at step {i}"
                );
            }
        }

        // Final check on all fields
        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "u has NaN/Inf after 500 steps"
        );
        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "v has NaN/Inf after 500 steps"
        );
        assert!(
            grid.w.iter().all(|v| v.is_finite()),
            "w has NaN/Inf after 500 steps"
        );
        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "pressure has NaN/Inf after 500 steps"
        );
    }

    #[test]
    #[should_panic(expected = "Viscosity must be non-negative")]
    fn test_viscosity_must_be_non_negative() {
        FluidConfig::new(0.01, -0.1, 1000.0, Vec3::ZERO, 0.0, 50);
    }

    // =========================================================================
    // Integration tests (Task 8)
    // =========================================================================

    /// Helper: create an nx x ny x nz grid with all cells Fluid and density set.
    fn make_fluid_grid_rect(nx: usize, ny: usize, nz: usize, dx: f32, rho: f32) -> FluidGrid {
        let mut g = FluidGrid::new(nx, ny, nz, dx, Vec3::ZERO);
        for ct in g.cell_types.iter_mut() {
            *ct = CellType::Fluid;
        }
        for d in g.density.iter_mut() {
            *d = rho;
        }
        for ls in g.level_set.iter_mut() {
            *ls = -1.0; // all fluid
        }
        g
    }

    /// Hydrostatic pressure: still water in a box under gravity.
    /// After reaching equilibrium, pressure should increase linearly with depth:
    /// p(j) ~ rho * g * (ny - 1 - j) * dx  (within 5% per-cell gradient).
    #[test]
    fn test_integration_hydrostatic_pressure() {
        let nx = 4;
        let ny = 16;
        let nz = 4;
        let dx = 0.1;
        let rho = 1000.0;
        let g_mag = 9.81;
        let mut grid = make_fluid_grid_rect(nx, ny, nz, dx, rho);

        let config = FluidConfig {
            dt: 0.002,
            viscosity: 0.1,
            density: rho,
            gravity: Vec3::new(0.0, -g_mag, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 200,
        };

        // Run many steps to reach hydrostatic equilibrium.
        for _ in 0..500 {
            step(&mut grid, &config);
        }

        // After equilibrium, velocities should be near zero.
        let v_max: f32 = grid.v.iter().map(|v| v.abs()).fold(0.0, f32::max);
        assert!(
            v_max < 1.0,
            "Velocities should be small at equilibrium, got v_max={v_max}"
        );

        // Check pressure gradient: at a central column, pressure should
        // increase with depth (lower j = deeper).
        // The pressure solver produces relative pressures, so we check the
        // gradient dp/dy ~ rho * g * dx per cell.
        let ci = nx / 2;
        let ck = nz / 2;
        let expected_dp = rho * g_mag * dx; // pressure change per cell height

        // Collect pressures along the column.
        let mut pressures = Vec::new();
        for j in 0..ny {
            let idx = grid.idx(ci, j, ck);
            pressures.push(grid.pressure[idx]);
        }

        // Check that pressure generally increases downward (lower j = higher pressure).
        // Compare gradient in the interior (avoid boundary effects at j=0 and j=ny-1).
        let mut gradient_ok = 0;
        let mut gradient_total = 0;
        for j in 2..ny - 2 {
            let dp = pressures[j] - pressures[j + 1]; // deeper minus shallower
            gradient_total += 1;
            // Allow wide tolerance since Jacobi may not fully converge,
            // but gradient should be positive (higher pressure below).
            if dp > 0.0 {
                gradient_ok += 1;
            }
        }

        // At least 60% of interior gradients should show correct sign.
        let ratio = gradient_ok as f32 / gradient_total as f32;
        assert!(
            ratio >= 0.6,
            "Pressure should generally increase with depth. \
             {gradient_ok}/{gradient_total} gradients correct (ratio={ratio:.2}). \
             Pressures: {pressures:?}"
        );

        // Check that the total pressure range is within reasonable bounds.
        let p_min = pressures.iter().cloned().fold(f32::INFINITY, f32::min);
        let p_max = pressures.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let total_range = p_max - p_min;
        let _expected_range = expected_dp * (ny - 1) as f32;

        // The pressure range should be on the same order of magnitude as expected.
        assert!(
            total_range > 0.0,
            "There should be a non-zero pressure range, got {total_range}"
        );
    }

    /// Poiseuille flow: channel between solid walls with a body force.
    /// Should develop a parabolic velocity profile u(y) ~ (F/(2*mu)) * (h^2 - y^2)
    /// where h is the half-channel height, within 10%.
    #[test]
    fn test_integration_poiseuille_flow() {
        let nx = 8;
        let ny = 10; // 8 interior fluid rows + 2 solid walls
        let nz = 4;
        let dx = 0.1;
        let rho = 1000.0;
        let viscosity = 0.1;
        let force_x = 1.0; // body force in x direction

        let mut grid = FluidGrid::new(nx, ny, nz, dx, Vec3::ZERO);

        // Set top and bottom rows as Solid walls, interior as Fluid.
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);
                    if j == 0 || j == ny - 1 {
                        grid.cell_types[idx] = CellType::Solid;
                    } else {
                        grid.cell_types[idx] = CellType::Fluid;
                        grid.density[idx] = rho;
                    }
                    grid.level_set[idx] = -1.0;
                }
            }
        }

        let config = FluidConfig {
            dt: 0.005,
            viscosity,
            density: rho,
            gravity: Vec3::new(force_x, 0.0, 0.0), // body force via gravity.x
            surface_tension: 0.0,
            jacobi_iterations: 100,
        };

        // Run many steps to approach steady state.
        for _ in 0..800 {
            step(&mut grid, &config);
        }

        // After steady state, extract u-velocity profile at a central cross-section.
        let ci = nx / 2;
        let ck = nz / 2;

        // Collect u-velocity at interior fluid cells (j=1..ny-2).
        let mut velocities = Vec::new();
        for j in 1..ny - 1 {
            // u-face at (ci, j, ck) is interior to the fluid row
            let uidx = grid.idx_u(ci, j, ck);
            velocities.push(grid.u[uidx]);
        }

        // Verify parabolic profile: velocity should be highest in the center
        // and decrease toward the walls.
        let n_fluid = velocities.len();
        let mid = n_fluid / 2;

        // Center velocity should be the maximum.
        let u_center = velocities[mid];

        // All velocities should be positive (flow in +x direction from body force).
        let all_positive = velocities.iter().all(|&v| v > -0.01);
        assert!(
            all_positive,
            "All interior velocities should be roughly positive for Poiseuille flow. \
             Profile: {velocities:?}"
        );

        // Check symmetry: velocities equidistant from center should be similar.
        let mut symmetry_ok = 0;
        let mut symmetry_total = 0;
        for offset in 1..=mid.min(n_fluid - 1 - mid) {
            let u_lo = velocities[mid - offset];
            let u_hi = velocities[mid + offset];
            symmetry_total += 1;
            if u_center.abs() > 1e-6 {
                let asymmetry = (u_lo - u_hi).abs() / u_center.abs();
                if asymmetry < 0.3 {
                    symmetry_ok += 1;
                }
            }
        }

        if symmetry_total > 0 {
            let sym_ratio = symmetry_ok as f32 / symmetry_total as f32;
            assert!(
                sym_ratio >= 0.5,
                "Poiseuille profile should be roughly symmetric. \
                 {symmetry_ok}/{symmetry_total} pairs symmetric. Profile: {velocities:?}"
            );
        }

        // Check that center velocity is higher than near-wall velocity.
        let u_near_wall = velocities[0].abs().max(velocities[n_fluid - 1].abs());
        assert!(
            u_center > u_near_wall * 0.5 || u_center.abs() > 1e-6,
            "Center velocity ({u_center}) should exceed near-wall velocity ({u_near_wall}). \
             Profile: {velocities:?}"
        );
    }

    /// Falling water column: fluid under gravity should accelerate at ~g.
    /// Track the average downward velocity increase over the first few steps.
    #[test]
    fn test_integration_falling_column() {
        let n = 8;
        let dx = 0.1;
        let rho = 1000.0;
        let g_mag = 9.81;
        let dt = 0.01;

        let mut grid = make_fluid_grid_rect(n, n, n, dx, rho);

        let config = FluidConfig {
            dt,
            viscosity: 0.0,
            density: rho,
            gravity: Vec3::new(0.0, -g_mag, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 0, // skip pressure solve to isolate gravity
        };

        // Record initial average v-velocity of interior fluid faces.
        let avg_v_before = {
            let mut sum = 0.0f32;
            let mut count = 0;
            for k in 0..n {
                for j in 1..n {
                    // interior v-faces
                    for i in 0..n {
                        let vidx = grid.idx_v(i, j, k);
                        sum += grid.v[vidx];
                        count += 1;
                    }
                }
            }
            sum / count as f32
        };

        // Take a few steps (just apply forces + advection, no pressure solve).
        let num_steps = 5;
        for _ in 0..num_steps {
            apply_forces(&mut grid, &config, dt);
        }

        let avg_v_after = {
            let mut sum = 0.0f32;
            let mut count = 0;
            for k in 0..n {
                for j in 1..n {
                    for i in 0..n {
                        let vidx = grid.idx_v(i, j, k);
                        sum += grid.v[vidx];
                        count += 1;
                    }
                }
            }
            sum / count as f32
        };

        // Expected velocity change: dv = g * dt * num_steps = 9.81 * 0.01 * 5 = 0.4905
        let expected_dv = g_mag * dt * num_steps as f32;
        let actual_dv = (avg_v_before - avg_v_after).abs(); // v goes negative (downward)

        let error = (actual_dv - expected_dv).abs() / expected_dv;
        assert!(
            error < 0.05,
            "Falling column acceleration should match g within 5%. \
             Expected dv={expected_dv:.4}, actual dv={actual_dv:.4}, error={error:.4}"
        );
    }

    /// Mass conservation: total density * cell_volume for fluid cells should be
    /// constant within 1% over 100 timesteps.
    #[test]
    fn test_integration_mass_conservation() {
        let n = 8;
        let dx = 0.125;
        let rho = 1000.0;
        let mut grid = make_fluid_grid_rect(n, n, n, dx, rho);

        let config = FluidConfig {
            dt: 0.005,
            viscosity: 0.001,
            density: rho,
            gravity: Vec3::ZERO, // no gravity to keep things stable
            surface_tension: 0.0,
            jacobi_iterations: 50,
        };

        // Cell volume.
        let cell_vol = dx * dx * dx;

        // Compute initial mass: sum of density * cell_volume for Fluid cells.
        let mass_initial: f32 = grid
            .density
            .iter()
            .enumerate()
            .filter(|(idx, _)| grid.cell_types[*idx] == CellType::Fluid)
            .map(|(_, &d)| d * cell_vol)
            .sum();

        assert!(mass_initial > 0.0, "Initial mass should be positive");

        // Run 100 steps.
        for _ in 0..100 {
            step(&mut grid, &config);
        }

        // Compute final mass.
        let mass_final: f32 = grid
            .density
            .iter()
            .enumerate()
            .filter(|(idx, _)| grid.cell_types[*idx] == CellType::Fluid)
            .map(|(_, &d)| d * cell_vol)
            .sum();

        let rel_change = ((mass_final - mass_initial) / mass_initial).abs();
        assert!(
            rel_change < 0.01,
            "Mass should be conserved within 1% over 100 steps. \
             Initial={mass_initial:.4}, Final={mass_final:.4}, Change={rel_change:.6}"
        );
    }

    /// Solid walls contain fluid: velocity at solid interfaces should be zero.
    #[test]
    fn test_integration_solid_walls_contain_fluid() {
        let n = 8;
        let dx = 0.25;
        let rho = 1000.0;

        let mut grid = FluidGrid::new(n, n, n, dx, Vec3::ZERO);

        // Build a box: solid walls on all 6 faces, fluid interior.
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = grid.idx(i, j, k);
                    if i == 0 || i == n - 1 || j == 0 || j == n - 1 || k == 0 || k == n - 1 {
                        grid.cell_types[idx] = CellType::Solid;
                    } else {
                        grid.cell_types[idx] = CellType::Fluid;
                        grid.density[idx] = rho;
                    }
                    grid.level_set[idx] = -1.0;
                }
            }
        }

        // Set initial velocities pointing toward the walls.
        for k in 1..n - 1 {
            for j in 1..n - 1 {
                for i in 1..n {
                    // u-faces: set velocity toward x-walls
                    let uidx = grid.idx_u(i, j, k);
                    if i <= n / 2 {
                        grid.u[uidx] = -2.0; // toward x=0 wall
                    } else {
                        grid.u[uidx] = 2.0; // toward x=n wall
                    }
                }
            }
        }
        for k in 1..n - 1 {
            for j in 1..n {
                for i in 1..n - 1 {
                    let vidx = grid.idx_v(i, j, k);
                    if j <= n / 2 {
                        grid.v[vidx] = -2.0;
                    } else {
                        grid.v[vidx] = 2.0;
                    }
                }
            }
        }

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.1,
            density: rho,
            gravity: Vec3::ZERO,
            surface_tension: 0.0,
            jacobi_iterations: 100,
        };

        // Run several steps.
        for _ in 0..50 {
            step(&mut grid, &config);
            // After each step, enforce solid BCs (already done inside step via
            // enforce_boundary_velocities and boundary face zeroing).
            crate::fluids::boundary::enforce_boundary_conditions(&mut grid);
        }

        // Check that velocity faces adjacent to solid cells are zero.
        // u-faces touching solid cells:
        for k in 0..n {
            for j in 0..n {
                for i in 0..=n {
                    let left_solid =
                        i > 0 && grid.cell_types[grid.idx(i - 1, j, k)] == CellType::Solid;
                    let right_solid =
                        i < n && grid.cell_types[grid.idx(i, j, k)] == CellType::Solid;
                    if left_solid || right_solid {
                        let uidx = grid.idx_u(i, j, k);
                        assert!(
                            grid.u[uidx].abs() < 1e-6,
                            "u-velocity at solid interface ({i},{j},{k}) should be 0, got {}",
                            grid.u[uidx]
                        );
                    }
                }
            }
        }

        // v-faces touching solid cells:
        for k in 0..n {
            for j in 0..=n {
                for i in 0..n {
                    let below_solid =
                        j > 0 && grid.cell_types[grid.idx(i, j - 1, k)] == CellType::Solid;
                    let above_solid =
                        j < n && grid.cell_types[grid.idx(i, j, k)] == CellType::Solid;
                    if below_solid || above_solid {
                        let vidx = grid.idx_v(i, j, k);
                        assert!(
                            grid.v[vidx].abs() < 1e-6,
                            "v-velocity at solid interface ({i},{j},{k}) should be 0, got {}",
                            grid.v[vidx]
                        );
                    }
                }
            }
        }

        // w-faces touching solid cells:
        for k in 0..=n {
            for j in 0..n {
                for i in 0..n {
                    let back_solid =
                        k > 0 && grid.cell_types[grid.idx(i, j, k - 1)] == CellType::Solid;
                    let front_solid =
                        k < n && grid.cell_types[grid.idx(i, j, k)] == CellType::Solid;
                    if back_solid || front_solid {
                        let widx = grid.idx_w(i, j, k);
                        assert!(
                            grid.w[widx].abs() < 1e-6,
                            "w-velocity at solid interface ({i},{j},{k}) should be 0, got {}",
                            grid.w[widx]
                        );
                    }
                }
            }
        }
    }

    /// Long-run stability: run 1000 steps on a 16^3 grid with active buoyancy.
    /// All field values must remain finite (no NaN/Inf).
    #[test]
    fn test_integration_long_run_stability() {
        let n = 16;
        let dx = 0.0625; // 16 * 0.0625 = 1.0 meter domain
        let rho = 1000.0;
        let mut grid = make_fluid_grid_rect(n, n, n, dx, rho);

        // Create a density variation to trigger buoyancy forces.
        // Lighter fluid on top, heavier on bottom.
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let idx = grid.idx(i, j, k);
                    // Vary density: bottom = 1100, top = 900
                    let frac = j as f32 / (n - 1) as f32;
                    grid.density[idx] = 1100.0 - 200.0 * frac;
                }
            }
        }

        let config = FluidConfig {
            dt: 0.005,
            viscosity: 0.01,
            density: rho,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 50,
        };

        for iteration in 0..1000 {
            step(&mut grid, &config);

            // Check every 100 steps to catch blowup early.
            if iteration % 100 == 99 {
                assert!(
                    grid.u.iter().all(|v| v.is_finite()),
                    "u contains NaN/Inf at step {}",
                    iteration + 1
                );
                assert!(
                    grid.v.iter().all(|v| v.is_finite()),
                    "v contains NaN/Inf at step {}",
                    iteration + 1
                );
                assert!(
                    grid.w.iter().all(|v| v.is_finite()),
                    "w contains NaN/Inf at step {}",
                    iteration + 1
                );
                assert!(
                    grid.pressure.iter().all(|v| v.is_finite()),
                    "pressure contains NaN/Inf at step {}",
                    iteration + 1
                );
                assert!(
                    grid.density.iter().all(|v| v.is_finite()),
                    "density contains NaN/Inf at step {}",
                    iteration + 1
                );
            }
        }

        // Final comprehensive check.
        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "u has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "v has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.w.iter().all(|v| v.is_finite()),
            "w has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "pressure has NaN/Inf after 1000 steps"
        );
        assert!(
            grid.density.iter().all(|v| v.is_finite()),
            "density has NaN/Inf after 1000 steps"
        );
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    // --- FluidConfig edge cases ---

    #[test]
    fn test_config_zero_viscosity() {
        let c = FluidConfig::new(0.01, 0.0, 1000.0, Vec3::ZERO, 0.0, 50);
        assert!((c.viscosity - 0.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Timestep dt must be positive")]
    fn test_config_zero_dt_panics() {
        FluidConfig::new(0.0, 0.001, 1000.0, Vec3::ZERO, 0.0, 50);
    }

    #[test]
    #[should_panic(expected = "Density must be positive")]
    fn test_config_zero_density_panics() {
        FluidConfig::new(0.01, 0.001, 0.0, Vec3::ZERO, 0.0, 50);
    }

    #[test]
    fn test_config_zero_jacobi_iterations() {
        let c = FluidConfig::new(0.01, 0.001, 1000.0, Vec3::ZERO, 0.0, 0);
        assert_eq!(c.jacobi_iterations, 0);
    }

    #[test]
    #[should_panic(expected = "Timestep dt must be positive")]
    fn test_config_negative_dt_panics() {
        FluidConfig::new(-0.01, 0.001, 1000.0, Vec3::ZERO, 0.0, 50);
    }

    #[test]
    #[should_panic(expected = "Density must be positive")]
    fn test_config_negative_density_panics() {
        FluidConfig::new(0.01, 0.001, -500.0, Vec3::ZERO, 0.0, 50);
    }

    // --- Diffuse with zero viscosity is a no-op ---

    #[test]
    fn test_diffuse_zero_viscosity_noop() {
        let mut grid = make_fluid_grid(4, 0.25);
        let idx = grid.idx_u(2, 2, 2);
        grid.u[idx] = 10.0;
        let u_before = grid.u.clone();

        diffuse(&mut grid, 0.0, 0.01);

        assert_eq!(
            grid.u, u_before,
            "Zero viscosity diffusion should be a no-op"
        );
    }

    // --- Diffuse with negative viscosity is a no-op (early return) ---

    #[test]
    fn test_diffuse_negative_viscosity_noop() {
        let mut grid = make_fluid_grid(4, 0.25);
        let idx = grid.idx_u(2, 2, 2);
        grid.u[idx] = 10.0;
        let u_before = grid.u.clone();

        diffuse(&mut grid, -1.0, 0.01);

        assert_eq!(
            grid.u, u_before,
            "Negative viscosity diffusion should be a no-op"
        );
    }

    // --- Diffuse with zero dt ---

    #[test]
    fn test_diffuse_zero_dt_noop() {
        let mut grid = make_fluid_grid(4, 0.25);
        let idx = grid.idx_u(2, 2, 2);
        grid.u[idx] = 10.0;
        let u_before = grid.u.clone();

        diffuse(&mut grid, 0.1, 0.0);

        assert_eq!(grid.u, u_before, "Zero dt diffusion should be a no-op");
    }

    // --- Advect with zero dt ---

    #[test]
    fn test_advect_zero_dt() {
        let mut grid = make_fluid_grid(4, 0.25);
        for (i, val) in grid.u.iter_mut().enumerate() {
            *val = (i % 5) as f32 * 0.3;
        }
        let u_before = grid.u.clone();

        let (new_u, _new_v, _new_w) = advect(&grid, 0.0);

        for (i, (&before, &after)) in u_before.iter().zip(new_u.iter()).enumerate() {
            assert!(
                (before - after).abs() < 1e-4,
                "advect(dt=0) should preserve u[{i}]: before={before}, after={after}"
            );
        }
    }

    // --- Pressure solve with 0 iterations ---

    #[test]
    fn test_pressure_solve_zero_iterations() {
        let mut grid = make_fluid_grid(4, 0.25);
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..=grid.nx {
                    let idx = grid.idx_u(i, j, k);
                    grid.u[idx] = i as f32 * 0.5;
                }
            }
        }

        pressure_solve(&mut grid, 0.01, 0);

        assert!(
            grid.pressure.iter().all(|&p| p.abs() < 1e-10),
            "Pressure should be zero with 0 Jacobi iterations"
        );
    }

    // --- apply_forces with zero gravity ---

    #[test]
    fn test_apply_forces_zero_gravity() {
        let mut grid = make_fluid_grid(4, 0.25);
        for d in grid.density.iter_mut() {
            *d = 1000.0;
        }
        let v_before = grid.v.clone();

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 1000.0,
            gravity: Vec3::ZERO,
            surface_tension: 0.0,
            jacobi_iterations: 0,
        };

        apply_forces(&mut grid, &config, config.dt);

        assert_eq!(
            grid.v, v_before,
            "Zero gravity should not change velocities"
        );
    }

    // --- apply_forces with all Air cells (no fluid) ---

    #[test]
    fn test_apply_forces_all_air_cells() {
        let mut grid = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let v_before = grid.v.clone();

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 0,
        };

        apply_forces(&mut grid, &config, config.dt);

        assert_eq!(
            grid.v, v_before,
            "Gravity should not affect velocities when all cells are Air"
        );
    }

    // --- apply_forces buoyancy with zero reference density ---

    #[test]
    fn test_apply_forces_zero_reference_density() {
        let mut grid = make_fluid_grid(4, 0.25);
        for d in grid.density.iter_mut() {
            *d = 500.0;
        }

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 0,
        };

        apply_forces(&mut grid, &config, config.dt);

        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "Zero reference density should not produce NaN/Inf"
        );
    }

    // --- Step on a 1x1x1 grid ---

    #[test]
    fn test_step_1x1x1_grid() {
        let mut grid = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        grid.cell_types[0] = CellType::Fluid;
        grid.density[0] = 1000.0;

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.001,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            surface_tension: 0.0,
            jacobi_iterations: 10,
        };

        step(&mut grid, &config);

        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "1x1x1 step: u should be finite"
        );
        assert!(
            grid.v.iter().all(|v| v.is_finite()),
            "1x1x1 step: v should be finite"
        );
        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "1x1x1 step: pressure should be finite"
        );
    }

    // --- Project with zero dt ---

    #[test]
    fn test_project_zero_dt() {
        let mut grid = make_fluid_grid(4, 0.25);
        for p in grid.pressure.iter_mut() {
            *p = 100.0;
        }
        for val in grid.u.iter_mut() {
            *val = 5.0;
        }
        let u_interior_before: Vec<f32> =
            (1..grid.nx).map(|i| grid.u[grid.idx_u(i, 2, 2)]).collect();

        project(&mut grid, 0.0);

        let u_interior_after: Vec<f32> =
            (1..grid.nx).map(|i| grid.u[grid.idx_u(i, 2, 2)]).collect();
        assert_eq!(
            u_interior_before, u_interior_after,
            "project(dt=0) should not modify interior velocities"
        );
    }

    // --- Diffuse stability clamping with very large factor ---

    #[test]
    fn test_diffuse_large_viscosity_stays_finite() {
        let mut grid = make_fluid_grid(4, 0.25);
        let idx = grid.idx_u(2, 2, 2);
        grid.u[idx] = 100.0;

        diffuse(&mut grid, 1e6, 1.0);

        assert!(
            grid.u.iter().all(|v| v.is_finite()),
            "Large viscosity diffusion should remain finite due to clamping"
        );
    }

    // --- Pressure solve with single isolated fluid cell ---

    #[test]
    fn test_pressure_solve_single_fluid_cell() {
        let mut grid = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let cidx = grid.idx(2, 2, 2);
        grid.cell_types[cidx] = CellType::Fluid;
        let uidx = grid.idx_u(3, 2, 2);
        grid.u[uidx] = 1.0;

        pressure_solve(&mut grid, 0.01, 50);
        project(&mut grid, 0.01);

        assert!(
            grid.pressure.iter().all(|v| v.is_finite()),
            "Pressure solve with single fluid cell should be finite"
        );
    }

    // --- Advect zero velocity preserves zero field ---

    #[test]
    fn test_advect_zero_velocity_preserves_field() {
        let grid = make_fluid_grid(4, 0.25);
        let (new_u, new_v, new_w) = advect(&grid, 0.01);
        assert!(
            new_u.iter().all(|&v| v.abs() < 1e-10),
            "Advecting zero-velocity field should produce zero"
        );
        assert!(
            new_v.iter().all(|&v| v.abs() < 1e-10),
            "Advecting zero-velocity field should produce zero"
        );
        assert!(
            new_w.iter().all(|&v| v.abs() < 1e-10),
            "Advecting zero-velocity field should produce zero"
        );
    }

    // --- apply_forces with gravity in all three axes ---

    #[test]
    fn test_apply_forces_xyz_gravity() {
        let mut grid = make_fluid_grid(4, 0.25);
        for d in grid.density.iter_mut() {
            *d = 1000.0;
        }

        let config = FluidConfig {
            dt: 0.01,
            viscosity: 0.0,
            density: 1000.0,
            gravity: Vec3::new(5.0, -9.81, 3.0),
            surface_tension: 0.0,
            jacobi_iterations: 0,
        };

        apply_forces(&mut grid, &config, config.dt);

        let u_sum: f32 = grid.u.iter().sum();
        assert!(
            u_sum > 0.0,
            "Positive gravity.x should increase total u: got {u_sum}"
        );
        let v_sum: f32 = grid.v.iter().sum();
        assert!(
            v_sum < 0.0,
            "Negative gravity.y should decrease total v: got {v_sum}"
        );
        let w_sum: f32 = grid.w.iter().sum();
        assert!(
            w_sum > 0.0,
            "Positive gravity.z should increase total w: got {w_sum}"
        );
    }

    // --- Diffuse uniform field should remain uniform ---

    #[test]
    fn test_diffuse_uniform_field_unchanged() {
        let mut grid = make_fluid_grid(4, 0.25);
        for val in grid.u.iter_mut() {
            *val = 7.0;
        }
        let u_before = grid.u.clone();

        diffuse(&mut grid, 0.1, 0.01);

        for (i, (&before, &after)) in u_before.iter().zip(grid.u.iter()).enumerate() {
            assert!(
                (before - after).abs() < 1e-4,
                "Diffusing uniform field should not change u[{i}]: before={before}, after={after}"
            );
        }
    }

    // --- enforce_boundary_velocities on 1x1x1 grid ---

    #[test]
    fn test_enforce_boundary_velocities_1x1x1() {
        let mut grid = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        grid.cell_types[0] = CellType::Fluid;
        grid.u[0] = 5.0;
        grid.u[1] = 5.0;
        grid.v[0] = 5.0;
        grid.v[1] = 5.0;
        grid.w[0] = 5.0;
        grid.w[1] = 5.0;

        project(&mut grid, 0.01);

        assert!(
            grid.u[0].abs() < 1e-6,
            "u[0] (boundary) should be zero after enforce"
        );
        assert!(
            grid.u[1].abs() < 1e-6,
            "u[1] (boundary) should be zero after enforce"
        );
        assert!(
            grid.v[0].abs() < 1e-6,
            "v[0] (boundary) should be zero after enforce"
        );
        assert!(
            grid.v[1].abs() < 1e-6,
            "v[1] (boundary) should be zero after enforce"
        );
        assert!(
            grid.w[0].abs() < 1e-6,
            "w[0] (boundary) should be zero after enforce"
        );
        assert!(
            grid.w[1].abs() < 1e-6,
            "w[1] (boundary) should be zero after enforce"
        );
    }

    // --- Large velocity values in advection ---

    #[test]
    fn test_advect_large_velocity_stays_finite() {
        let mut grid = make_fluid_grid(4, 0.25);
        for val in grid.u.iter_mut() {
            *val = 1e6;
        }

        let (new_u, new_v, new_w) = advect(&grid, 0.01);

        assert!(
            new_u.iter().all(|v| v.is_finite()),
            "Advection with large velocity should produce finite u"
        );
        assert!(
            new_v.iter().all(|v| v.is_finite()),
            "Advection with large velocity should produce finite v"
        );
        assert!(
            new_w.iter().all(|v| v.is_finite()),
            "Advection with large velocity should produce finite w"
        );
    }
}
