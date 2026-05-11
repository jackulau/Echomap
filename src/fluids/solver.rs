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
    pub surface_tension: f32,
    /// Number of Jacobi iterations for pressure solve.
    pub jacobi_iterations: u32,
}

impl FluidConfig {
    /// Create a new config, panicking if viscosity is negative.
    pub fn new(
        dt: f32,
        viscosity: f32,
        density: f32,
        gravity: Vec3,
        surface_tension: f32,
        jacobi_iterations: u32,
    ) -> Self {
        assert!(
            viscosity >= 0.0,
            "Viscosity must be non-negative, got {viscosity}"
        );
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
        for i in 0..nx {
            let pos =
                origin + Vec3::new((i as f32 + 0.5) * dx, (j as f32 + 0.5) * dx, k as f32 * dx);
            let vel = velocity_from_arrays(&old_u, &old_v, &old_w, nx, ny, nz, dx, origin, pos);
            let back_pos = pos - vel * dt;
            let rel = back_pos - origin;
            let fi = rel.x / dx - 0.5;
            let fj = rel.y / dx - 0.5;
            let fk = rel.z / dx;
            row[i] = sample_w(&old_w, nx, ny, nz, fi, fj, fk);
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
}
