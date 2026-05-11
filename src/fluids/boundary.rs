use glam::Vec3;

use super::grid::{CellType, FluidGrid};
use crate::scene::SceneObject;

/// Mark grid cells overlapping solid scene meshes as `CellType::Solid` using
/// AABB intersection between each mesh's bounding box and each grid cell.
pub fn voxelize_scene(grid: &mut FluidGrid, meshes: &[SceneObject]) {
    for obj in meshes {
        if obj.mesh.triangles.is_empty() {
            continue;
        }
        let (mesh_min, mesh_max) = obj.mesh.bounds();

        // Determine the range of grid cells that could overlap the mesh AABB.
        let rel_min = mesh_min - grid.origin;
        let rel_max = mesh_max - grid.origin;

        let i_start = ((rel_min.x / grid.dx).floor() as i32).max(0) as usize;
        let j_start = ((rel_min.y / grid.dx).floor() as i32).max(0) as usize;
        let k_start = ((rel_min.z / grid.dx).floor() as i32).max(0) as usize;

        let i_end = ((rel_max.x / grid.dx).ceil() as usize).min(grid.nx);
        let j_end = ((rel_max.y / grid.dx).ceil() as usize).min(grid.ny);
        let k_end = ((rel_max.z / grid.dx).ceil() as usize).min(grid.nz);

        for k in k_start..k_end {
            for j in j_start..j_end {
                for i in i_start..i_end {
                    // Cell AABB
                    let cell_min = grid.origin
                        + Vec3::new(i as f32 * grid.dx, j as f32 * grid.dx, k as f32 * grid.dx);
                    let cell_max = cell_min + Vec3::splat(grid.dx);

                    // AABB-AABB overlap test
                    if cell_min.x < mesh_max.x
                        && cell_max.x > mesh_min.x
                        && cell_min.y < mesh_max.y
                        && cell_max.y > mesh_min.y
                        && cell_min.z < mesh_max.z
                        && cell_max.z > mesh_min.z
                    {
                        let idx = grid.idx(i, j, k);
                        grid.cell_types[idx] = CellType::Solid;
                    }
                }
            }
        }
    }
}

/// Enforce no-slip boundary conditions at solid boundaries.
///
/// - Velocity faces adjacent to solid cells are set to zero.
/// - Pressure gradient is zero at solid walls (Neumann BC), enforced by
///   copying pressure from the fluid neighbour into the solid cell.
pub fn enforce_boundary_conditions(grid: &mut FluidGrid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // --- No-slip: zero velocity on faces touching solid cells ---

    // u-faces: face (i, j, k) sits between cell (i-1, j, k) and (i, j, k).
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let left_solid = i > 0 && {
                    let idx = grid.idx(i - 1, j, k);
                    grid.cell_types[idx] == CellType::Solid
                };
                let right_solid = i < nx && {
                    let idx = grid.idx(i, j, k);
                    grid.cell_types[idx] == CellType::Solid
                };
                if left_solid || right_solid {
                    let uidx = grid.idx_u(i, j, k);
                    grid.u[uidx] = 0.0;
                }
            }
        }
    }

    // v-faces: face (i, j, k) sits between cell (i, j-1, k) and (i, j, k).
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                let below_solid = j > 0 && {
                    let idx = grid.idx(i, j - 1, k);
                    grid.cell_types[idx] == CellType::Solid
                };
                let above_solid = j < ny && {
                    let idx = grid.idx(i, j, k);
                    grid.cell_types[idx] == CellType::Solid
                };
                if below_solid || above_solid {
                    let vidx = grid.idx_v(i, j, k);
                    grid.v[vidx] = 0.0;
                }
            }
        }
    }

    // w-faces: face (i, j, k) sits between cell (i, j, k-1) and (i, j, k).
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                let back_solid = k > 0 && {
                    let idx = grid.idx(i, j, k - 1);
                    grid.cell_types[idx] == CellType::Solid
                };
                let front_solid = k < nz && {
                    let idx = grid.idx(i, j, k);
                    grid.cell_types[idx] == CellType::Solid
                };
                if back_solid || front_solid {
                    let widx = grid.idx_w(i, j, k);
                    grid.w[widx] = 0.0;
                }
            }
        }
    }

    // --- Neumann BC for pressure: copy pressure from fluid neighbour ---
    // This effectively sets the pressure gradient to zero at solid walls.
    let pressure_copy = grid.pressure.clone();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != CellType::Solid {
                    continue;
                }
                // Average pressure from non-solid neighbours.
                let mut sum = 0.0f32;
                let mut count = 0u32;
                if i > 0 {
                    let n = grid.idx(i - 1, j, k);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if i + 1 < nx {
                    let n = grid.idx(i + 1, j, k);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if j > 0 {
                    let n = grid.idx(i, j - 1, k);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if j + 1 < ny {
                    let n = grid.idx(i, j + 1, k);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if k > 0 {
                    let n = grid.idx(i, j, k - 1);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if k + 1 < nz {
                    let n = grid.idx(i, j, k + 1);
                    if grid.cell_types[n] == CellType::Fluid {
                        sum += pressure_copy[n];
                        count += 1;
                    }
                }
                if count > 0 {
                    grid.pressure[idx] = sum / count as f32;
                }
            }
        }
    }
}

/// Classify cells based on the level set field, preserving Solid cells.
///
/// - `level_set < 0` => Fluid
/// - `level_set > 0` => Air
/// - Cells already marked Solid remain Solid.
pub fn classify_cells(grid: &mut FluidGrid) {
    let n = grid.cell_types.len();
    for idx in 0..n {
        if grid.cell_types[idx] == CellType::Solid {
            continue;
        }
        if grid.level_set[idx] < 0.0 {
            grid.cell_types[idx] = CellType::Fluid;
        } else {
            grid.cell_types[idx] = CellType::Air;
        }
    }
}

/// Semi-Lagrangian advection of the level set field.
///
/// Traces particles backward through the velocity field and interpolates
/// the old level set value at the departure point.
pub fn advect_level_set(grid: &mut FluidGrid, dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let origin = grid.origin;

    let old_ls = grid.level_set.clone();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let pos = grid.cell_center(i, j, k);
                let vel = grid.velocity_at(pos);
                let back_pos = pos - vel * dt;

                // Trilinear interpolation of old level set at back_pos.
                let rel = back_pos - origin;
                let fi = (rel.x / dx - 0.5).clamp(0.0, (nx - 1) as f32);
                let fj = (rel.y / dx - 0.5).clamp(0.0, (ny - 1) as f32);
                let fk = (rel.z / dx - 0.5).clamp(0.0, (nz - 1) as f32);

                let i0 = (fi.floor() as usize).min(nx.saturating_sub(2));
                let j0 = (fj.floor() as usize).min(ny.saturating_sub(2));
                let k0 = (fk.floor() as usize).min(nz.saturating_sub(2));
                let i1 = (i0 + 1).min(nx - 1);
                let j1 = (j0 + 1).min(ny - 1);
                let k1 = (k0 + 1).min(nz - 1);

                let s = fi - i0 as f32;
                let t = fj - j0 as f32;
                let r = fk - k0 as f32;

                let idx = |ii: usize, jj: usize, kk: usize| ii + nx * (jj + ny * kk);

                let c000 = old_ls[idx(i0, j0, k0)];
                let c100 = old_ls[idx(i1, j0, k0)];
                let c010 = old_ls[idx(i0, j1, k0)];
                let c110 = old_ls[idx(i1, j1, k0)];
                let c001 = old_ls[idx(i0, j0, k1)];
                let c101 = old_ls[idx(i1, j0, k1)];
                let c011 = old_ls[idx(i0, j1, k1)];
                let c111 = old_ls[idx(i1, j1, k1)];

                let c00 = c000 * (1.0 - s) + c100 * s;
                let c10 = c010 * (1.0 - s) + c110 * s;
                let c01 = c001 * (1.0 - s) + c101 * s;
                let c11 = c011 * (1.0 - s) + c111 * s;

                let c0 = c00 * (1.0 - t) + c10 * t;
                let c1 = c01 * (1.0 - t) + c11 * t;

                let val = c0 * (1.0 - r) + c1 * r;

                let ls_idx = grid.idx(i, j, k);
                grid.level_set[ls_idx] = val;
            }
        }
    }
}

/// Iterative reinitialization of the level set to maintain the signed
/// distance property (|grad(phi)| = 1).
///
/// Uses the PDE-based approach:
///   phi_t + sign(phi0) * (|grad(phi)| - 1) = 0
/// with forward Euler time stepping.
pub fn reinitialize_level_set(grid: &mut FluidGrid, iterations: u32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let dx = grid.dx;
    let dt_reinit = 0.5 * dx; // CFL-limited pseudo-timestep

    let phi0 = grid.level_set.clone();

    for _iter in 0..iterations {
        let old = grid.level_set.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);

                    // Smoothed sign function: sign(phi0) = phi0 / sqrt(phi0^2 + dx^2)
                    let p0 = phi0[idx];
                    let sign_phi = p0 / (p0 * p0 + dx * dx).sqrt();

                    let c = old[idx];

                    // Upwind finite differences for gradient magnitude
                    let dxm = if i > 0 {
                        c - old[grid.idx(i - 1, j, k)]
                    } else {
                        0.0
                    };
                    let dxp = if i + 1 < nx {
                        old[grid.idx(i + 1, j, k)] - c
                    } else {
                        0.0
                    };
                    let dym = if j > 0 {
                        c - old[grid.idx(i, j - 1, k)]
                    } else {
                        0.0
                    };
                    let dyp = if j + 1 < ny {
                        old[grid.idx(i, j + 1, k)] - c
                    } else {
                        0.0
                    };
                    let dzm = if k > 0 {
                        c - old[grid.idx(i, j, k - 1)]
                    } else {
                        0.0
                    };
                    let dzp = if k + 1 < nz {
                        old[grid.idx(i, j, k + 1)] - c
                    } else {
                        0.0
                    };

                    // Godunov upwind scheme
                    let grad_mag_sq = if sign_phi > 0.0 {
                        let ax = dxm.max(0.0).powi(2) + dxp.min(0.0).powi(2);
                        let ay = dym.max(0.0).powi(2) + dyp.min(0.0).powi(2);
                        let az = dzm.max(0.0).powi(2) + dzp.min(0.0).powi(2);
                        ax + ay + az
                    } else {
                        let ax = dxm.min(0.0).powi(2) + dxp.max(0.0).powi(2);
                        let ay = dym.min(0.0).powi(2) + dyp.max(0.0).powi(2);
                        let az = dzm.min(0.0).powi(2) + dzp.max(0.0).powi(2);
                        ax + ay + az
                    };

                    let grad_mag = (grad_mag_sq / (dx * dx)).sqrt();

                    grid.level_set[idx] = c - dt_reinit * sign_phi * (grad_mag - 1.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::AcousticMaterial;
    use crate::scene::SceneObject;
    use crate::scene::{Mesh, Triangle, Vertex};

    /// Create a box mesh from (min) to (max) as 12 triangles.
    fn box_mesh(min: Vec3, max: Vec3) -> Mesh {
        let p = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];

        let v = |pos: Vec3| Vertex {
            position: pos,
            normal: Vec3::Y,
        };

        let quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3| -> [Triangle; 2] {
            [
                Triangle {
                    vertices: [v(a), v(b), v(c)],
                },
                Triangle {
                    vertices: [v(a), v(c), v(d)],
                },
            ]
        };

        let mut triangles = Vec::with_capacity(12);
        // bottom
        triangles.extend(quad(p[0], p[1], p[2], p[3]));
        // top
        triangles.extend(quad(p[4], p[7], p[6], p[5]));
        // front
        triangles.extend(quad(p[0], p[4], p[5], p[1]));
        // back
        triangles.extend(quad(p[2], p[6], p[7], p[3]));
        // left
        triangles.extend(quad(p[3], p[7], p[4], p[0]));
        // right
        triangles.extend(quad(p[1], p[5], p[6], p[2]));

        Mesh { triangles }
    }

    fn make_scene_object(mesh: Mesh) -> SceneObject {
        SceneObject {
            name: "Test".into(),
            mesh,
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }
    }

    #[test]
    fn test_voxelize_box_marks_interior_solid() {
        // 8x8x8 grid, dx=1.0, origin at (0,0,0). Grid spans [0,8]^3.
        let mut grid = FluidGrid::new(8, 8, 8, 1.0, Vec3::ZERO);
        // Place a box from (2,2,2) to (5,5,5) => cells (2..5, 2..5, 2..5) should be Solid.
        let mesh = box_mesh(Vec3::new(2.0, 2.0, 2.0), Vec3::new(5.0, 5.0, 5.0));
        let obj = make_scene_object(mesh);

        voxelize_scene(&mut grid, &[obj]);

        // Interior cells should be Solid
        let mut solid_count = 0;
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    let idx = grid.idx(i, j, k);
                    if i >= 2 && i < 5 && j >= 2 && j < 5 && k >= 2 && k < 5 {
                        assert_eq!(
                            grid.cell_types[idx],
                            CellType::Solid,
                            "Cell ({i},{j},{k}) inside box should be Solid"
                        );
                        solid_count += 1;
                    }
                }
            }
        }
        assert_eq!(solid_count, 3 * 3 * 3, "Should have 27 solid cells");

        // Cells outside the box should remain Air (the default)
        let idx_outside = grid.idx(0, 0, 0);
        assert_eq!(
            grid.cell_types[idx_outside],
            CellType::Air,
            "Cell (0,0,0) outside box should be Air"
        );
    }

    #[test]
    fn test_no_slip_zeroes_velocity_at_walls() {
        let mut grid = FluidGrid::new(8, 8, 8, 1.0, Vec3::ZERO);

        // Set all velocities to a non-zero value
        for val in grid.u.iter_mut() {
            *val = 5.0;
        }
        for val in grid.v.iter_mut() {
            *val = 5.0;
        }
        for val in grid.w.iter_mut() {
            *val = 5.0;
        }

        // Mark cell (4,4,4) as Solid
        let solid_idx = grid.idx(4, 4, 4);
        grid.cell_types[solid_idx] = CellType::Solid;

        enforce_boundary_conditions(&mut grid);

        // u-faces adjacent to the solid cell should be zero.
        // Face u(4,4,4) is on the left of cell (4,4,4).
        // Face u(5,4,4) is on the right of cell (4,4,4).
        let u_left = grid.u[grid.idx_u(4, 4, 4)];
        let u_right = grid.u[grid.idx_u(5, 4, 4)];
        assert!(
            u_left.abs() < 1e-6,
            "u-face at left of solid cell should be zero, got {u_left}"
        );
        assert!(
            u_right.abs() < 1e-6,
            "u-face at right of solid cell should be zero, got {u_right}"
        );

        // v-faces
        let v_below = grid.v[grid.idx_v(4, 4, 4)];
        let v_above = grid.v[grid.idx_v(4, 5, 4)];
        assert!(
            v_below.abs() < 1e-6,
            "v-face below solid cell should be zero, got {v_below}"
        );
        assert!(
            v_above.abs() < 1e-6,
            "v-face above solid cell should be zero, got {v_above}"
        );

        // w-faces
        let w_back = grid.w[grid.idx_w(4, 4, 4)];
        let w_front = grid.w[grid.idx_w(4, 4, 5)];
        assert!(
            w_back.abs() < 1e-6,
            "w-face behind solid cell should be zero, got {w_back}"
        );
        assert!(
            w_front.abs() < 1e-6,
            "w-face in front of solid cell should be zero, got {w_front}"
        );
    }

    #[test]
    fn test_classify_cells_from_level_set() {
        let mut grid = FluidGrid::new(4, 4, 4, 1.0, Vec3::ZERO);

        // Set level_set: negative for some cells (Fluid), positive for others (Air)
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let idx = grid.idx(i, j, k);
                    // Lower half is fluid (negative level set)
                    if j < 2 {
                        grid.level_set[idx] = -1.0;
                    } else {
                        grid.level_set[idx] = 1.0;
                    }
                }
            }
        }

        // Mark one cell as Solid (should be preserved)
        let solid_idx = grid.idx(0, 0, 0);
        grid.cell_types[solid_idx] = CellType::Solid;

        classify_cells(&mut grid);

        // Check that negative level_set => Fluid (except solid)
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let idx = grid.idx(i, j, k);
                    if i == 0 && j == 0 && k == 0 {
                        assert_eq!(
                            grid.cell_types[idx],
                            CellType::Solid,
                            "Solid cell should remain Solid"
                        );
                    } else if j < 2 {
                        assert_eq!(
                            grid.cell_types[idx],
                            CellType::Fluid,
                            "Cell ({i},{j},{k}) with negative level_set should be Fluid"
                        );
                    } else {
                        assert_eq!(
                            grid.cell_types[idx],
                            CellType::Air,
                            "Cell ({i},{j},{k}) with positive level_set should be Air"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_level_set_advection_conserves_interface() {
        // A uniform velocity field (1, 0, 0) should shift the interface position
        // by velocity * dt in the x-direction.
        let nx = 16;
        let dx = 1.0;
        let mut grid = FluidGrid::new(nx, 4, 4, dx, Vec3::ZERO);

        // Set uniform u-velocity = 1.0
        for val in grid.u.iter_mut() {
            *val = 1.0;
        }

        // Initialize level set: interface at x = 8 (negative to the left = fluid)
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);
                    let x = grid.cell_center(i, j, k).x;
                    grid.level_set[idx] = x - 8.0; // negative left of x=8, positive right
                }
            }
        }

        // Find the zero-crossing position before advection (along center row)
        let j_mid = grid.ny / 2;
        let k_mid = grid.nz / 2;
        let interface_before = find_zero_crossing(&grid, j_mid, k_mid);

        // Advect with dt = 1.0 (interface should move right by ~1.0)
        let dt = 1.0;
        advect_level_set(&mut grid, dt);

        let interface_after = find_zero_crossing(&grid, j_mid, k_mid);

        // The interface should have moved approximately 1.0 in x
        let shift = interface_after - interface_before;
        assert!(
            (shift - 1.0).abs() < 1.5,
            "Interface should shift by ~1.0 with u=1 and dt=1, got shift={shift:.3} \
             (before={interface_before:.3}, after={interface_after:.3})"
        );
    }

    /// Find approximate zero-crossing of level_set along x at given j,k
    fn find_zero_crossing(grid: &FluidGrid, j: usize, k: usize) -> f32 {
        let nx = grid.nx;
        for i in 0..nx - 1 {
            let ls0 = grid.level_set[grid.idx(i, j, k)];
            let ls1 = grid.level_set[grid.idx(i + 1, j, k)];
            if ls0 * ls1 <= 0.0 && (ls1 - ls0).abs() > 1e-10 {
                // Linear interpolation for zero crossing
                let t = -ls0 / (ls1 - ls0);
                let x0 = grid.cell_center(i, j, k).x;
                let x1 = grid.cell_center(i + 1, j, k).x;
                return x0 + t * (x1 - x0);
            }
        }
        // Fallback: return midpoint
        grid.cell_center(nx / 2, j, k).x
    }

    #[test]
    fn test_boundary_conditions_preserve_interior() {
        let mut grid = FluidGrid::new(8, 8, 8, 1.0, Vec3::ZERO);

        // Mark all cells as Fluid
        for ct in grid.cell_types.iter_mut() {
            *ct = CellType::Fluid;
        }

        // Mark edges as Solid (single layer border)
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    if i == 0 || i == 7 || j == 0 || j == 7 || k == 0 || k == 7 {
                        let idx = grid.idx(i, j, k);
                        grid.cell_types[idx] = CellType::Solid;
                    }
                }
            }
        }

        // Set interior velocities to a known pattern
        // Interior u-faces are between cells (1..6, 1..6, 1..6)
        // i.e. u-face indices 2..6 for interior faces not touching solid
        for k in 1..7 {
            for j in 1..7 {
                for i in 2..7 {
                    let uidx = grid.idx_u(i, j, k);
                    grid.u[uidx] = 3.14;
                }
            }
        }

        // Record interior velocity values before enforcing BC
        let mut interior_u_before = Vec::new();
        for k in 2..6 {
            for j in 2..6 {
                for i in 3..6 {
                    let uidx = grid.idx_u(i, j, k);
                    interior_u_before.push((i, j, k, grid.u[uidx]));
                }
            }
        }

        enforce_boundary_conditions(&mut grid);

        // Interior velocities far from solid cells should be unchanged
        for (i, j, k, val_before) in &interior_u_before {
            let uidx = grid.idx_u(*i, *j, *k);
            let val_after = grid.u[uidx];
            assert!(
                (val_after - val_before).abs() < 1e-6,
                "Interior u-face ({},{},{}) should be preserved: before={}, after={}",
                i,
                j,
                k,
                val_before,
                val_after
            );
        }
    }
}
