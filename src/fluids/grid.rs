use glam::Vec3;

/// Cell classification for boundary handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellType {
    Fluid,
    Solid,
    Air,
}

/// 3D MAC (Marker-And-Cell) staggered grid.
///
/// Velocities live on cell faces:
///   - `u` on x-faces: (nx+1) x ny x nz
///   - `v` on y-faces: nx x (ny+1) x nz
///   - `w` on z-faces: nx x ny x (nz+1)
///
/// Scalars (pressure, density, level_set, cell_types) are cell-centered:
///   nx x ny x nz
#[derive(Clone)]
pub struct FluidGrid {
    /// Number of cells along each axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Cell size (uniform in all directions).
    pub dx: f32,
    /// World-space origin of the grid (min corner).
    pub origin: Vec3,

    // --- Face-centered velocity arrays (MAC staggered) ---
    /// x-velocity on x-faces: (nx+1) * ny * nz
    pub u: Vec<f32>,
    /// y-velocity on y-faces: nx * (ny+1) * nz
    pub v: Vec<f32>,
    /// z-velocity on z-faces: nx * ny * (nz+1)
    pub w: Vec<f32>,

    // --- Cell-centered scalar arrays ---
    pub pressure: Vec<f32>,
    pub density: Vec<f32>,
    pub level_set: Vec<f32>,
    pub cell_types: Vec<CellType>,
}

/// Maximum allowed dimension per axis (prevents OOM).
const MAX_DIM: usize = 1024;

impl FluidGrid {
    /// Create a new grid with all fields zeroed.
    ///
    /// # Panics
    /// - Any dimension is 0.
    /// - Any dimension exceeds `MAX_DIM` (1024).
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f32, origin: Vec3) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0, "Grid dimensions must be > 0");
        assert!(
            nx <= MAX_DIM && ny <= MAX_DIM && nz <= MAX_DIM,
            "Grid dimensions must be <= {MAX_DIM}"
        );

        let cell_count = nx * ny * nz;

        Self {
            nx,
            ny,
            nz,
            dx,
            origin,
            u: vec![0.0; (nx + 1) * ny * nz],
            v: vec![0.0; nx * (ny + 1) * nz],
            w: vec![0.0; nx * ny * (nz + 1)],
            pressure: vec![0.0; cell_count],
            density: vec![0.0; cell_count],
            level_set: vec![0.0; cell_count],
            cell_types: vec![CellType::Air; cell_count],
        }
    }

    // ----- Index helpers -----

    /// Cell-centered linear index.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// x-face linear index (grid is (nx+1) x ny x nz).
    #[inline]
    pub fn idx_u(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }

    /// y-face linear index (grid is nx x (ny+1) x nz).
    #[inline]
    pub fn idx_v(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }

    /// z-face linear index (grid is nx x ny x (nz+1)).
    #[inline]
    pub fn idx_w(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    /// Decompose a cell-centered linear index back to (i, j, k).
    #[inline]
    pub fn idx_to_ijk(&self, index: usize) -> (usize, usize, usize) {
        let i = index % self.nx;
        let jk = index / self.nx;
        let j = jk % self.ny;
        let k = jk / self.ny;
        (i, j, k)
    }

    // ----- Spatial queries -----

    /// World-space position of the center of cell (i, j, k).
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                (i as f32 + 0.5) * self.dx,
                (j as f32 + 0.5) * self.dx,
                (k as f32 + 0.5) * self.dx,
            )
    }

    /// Check whether the cell indices are within bounds.
    pub fn in_bounds(&self, i: i32, j: i32, k: i32) -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && (i as usize) < self.nx
            && (j as usize) < self.ny
            && (k as usize) < self.nz
    }

    /// Trilinear interpolation of the staggered velocity field at an
    /// arbitrary world-space position.
    pub fn velocity_at(&self, pos: Vec3) -> Vec3 {
        let u = self.interpolate_u(pos);
        let v = self.interpolate_v(pos);
        let w = self.interpolate_w(pos);
        Vec3::new(u, v, w)
    }

    // ----- Private interpolation helpers -----

    /// Interpolate x-velocity. u lives on x-faces, so it is centered at
    /// (i*dx, (j+0.5)*dx, (k+0.5)*dx) relative to origin.
    fn interpolate_u(&self, pos: Vec3) -> f32 {
        let rel = pos - self.origin;
        let fi = rel.x / self.dx;
        let fj = rel.y / self.dx - 0.5;
        let fk = rel.z / self.dx - 0.5;
        self.trilinear_sample_u(fi, fj, fk)
    }

    /// Interpolate y-velocity. v lives on y-faces, centered at
    /// ((i+0.5)*dx, j*dx, (k+0.5)*dx) relative to origin.
    fn interpolate_v(&self, pos: Vec3) -> f32 {
        let rel = pos - self.origin;
        let fi = rel.x / self.dx - 0.5;
        let fj = rel.y / self.dx;
        let fk = rel.z / self.dx - 0.5;
        self.trilinear_sample_v(fi, fj, fk)
    }

    /// Interpolate z-velocity. w lives on z-faces, centered at
    /// ((i+0.5)*dx, (j+0.5)*dx, k*dx) relative to origin.
    fn interpolate_w(&self, pos: Vec3) -> f32 {
        let rel = pos - self.origin;
        let fi = rel.x / self.dx - 0.5;
        let fj = rel.y / self.dx - 0.5;
        let fk = rel.z / self.dx;
        self.trilinear_sample_w(fi, fj, fk)
    }

    fn trilinear_sample_u(&self, fi: f32, fj: f32, fk: f32) -> f32 {
        let nx1 = self.nx + 1;
        let ny = self.ny;
        let nz = self.nz;

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

        let idx = |i: usize, j: usize, k: usize| -> usize { i + nx1 * (j + ny * k) };

        let c000 = self.u[idx(i0, j0, k0)];
        let c100 = self.u[idx(i1, j0, k0)];
        let c010 = self.u[idx(i0, j1, k0)];
        let c110 = self.u[idx(i1, j1, k0)];
        let c001 = self.u[idx(i0, j0, k1)];
        let c101 = self.u[idx(i1, j0, k1)];
        let c011 = self.u[idx(i0, j1, k1)];
        let c111 = self.u[idx(i1, j1, k1)];

        let c00 = c000 * (1.0 - s) + c100 * s;
        let c10 = c010 * (1.0 - s) + c110 * s;
        let c01 = c001 * (1.0 - s) + c101 * s;
        let c11 = c011 * (1.0 - s) + c111 * s;

        let c0 = c00 * (1.0 - t) + c10 * t;
        let c1 = c01 * (1.0 - t) + c11 * t;

        c0 * (1.0 - r) + c1 * r
    }

    fn trilinear_sample_v(&self, fi: f32, fj: f32, fk: f32) -> f32 {
        let nx = self.nx;
        let ny1 = self.ny + 1;
        let nz = self.nz;

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

        let idx = |i: usize, j: usize, k: usize| -> usize { i + nx * (j + ny1 * k) };

        let c000 = self.v[idx(i0, j0, k0)];
        let c100 = self.v[idx(i1, j0, k0)];
        let c010 = self.v[idx(i0, j1, k0)];
        let c110 = self.v[idx(i1, j1, k0)];
        let c001 = self.v[idx(i0, j0, k1)];
        let c101 = self.v[idx(i1, j0, k1)];
        let c011 = self.v[idx(i0, j1, k1)];
        let c111 = self.v[idx(i1, j1, k1)];

        let c00 = c000 * (1.0 - s) + c100 * s;
        let c10 = c010 * (1.0 - s) + c110 * s;
        let c01 = c001 * (1.0 - s) + c101 * s;
        let c11 = c011 * (1.0 - s) + c111 * s;

        let c0 = c00 * (1.0 - t) + c10 * t;
        let c1 = c01 * (1.0 - t) + c11 * t;

        c0 * (1.0 - r) + c1 * r
    }

    fn trilinear_sample_w(&self, fi: f32, fj: f32, fk: f32) -> f32 {
        let nx = self.nx;
        let ny = self.ny;
        let nz1 = self.nz + 1;

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

        let idx = |i: usize, j: usize, k: usize| -> usize { i + nx * (j + ny * k) };

        let c000 = self.w[idx(i0, j0, k0)];
        let c100 = self.w[idx(i1, j0, k0)];
        let c010 = self.w[idx(i0, j1, k0)];
        let c110 = self.w[idx(i1, j1, k0)];
        let c001 = self.w[idx(i0, j0, k1)];
        let c101 = self.w[idx(i1, j0, k1)];
        let c011 = self.w[idx(i0, j1, k1)];
        let c111 = self.w[idx(i1, j1, k1)];

        let c00 = c000 * (1.0 - s) + c100 * s;
        let c10 = c010 * (1.0 - s) + c110 * s;
        let c01 = c001 * (1.0 - s) + c101 * s;
        let c11 = c011 * (1.0 - s) + c111 * s;

        let c0 = c00 * (1.0 - t) + c10 * t;
        let c1 = c01 * (1.0 - t) + c11 * t;

        c0 * (1.0 - r) + c1 * r
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation_dimensions() {
        let g = FluidGrid::new(4, 5, 6, 0.1, Vec3::ZERO);
        // Cell-centered arrays
        assert_eq!(g.pressure.len(), 4 * 5 * 6);
        assert_eq!(g.density.len(), 4 * 5 * 6);
        assert_eq!(g.level_set.len(), 4 * 5 * 6);
        assert_eq!(g.cell_types.len(), 4 * 5 * 6);
        // Face-centered velocity arrays (MAC staggered)
        assert_eq!(g.u.len(), 5 * 5 * 6); // (nx+1)*ny*nz
        assert_eq!(g.v.len(), 4 * 6 * 6); // nx*(ny+1)*nz
        assert_eq!(g.w.len(), 4 * 5 * 7); // nx*ny*(nz+1)
    }

    #[test]
    fn test_cell_center_positions() {
        let origin = Vec3::new(1.0, 2.0, 3.0);
        let dx = 0.5;
        let g = FluidGrid::new(4, 4, 4, dx, origin);
        let c = g.cell_center(0, 0, 0);
        let expected = origin + Vec3::splat(dx / 2.0);
        assert!(
            (c - expected).length() < 1e-6,
            "cell(0,0,0) center should be origin + dx/2, got {c:?} expected {expected:?}"
        );

        let c2 = g.cell_center(1, 2, 3);
        let expected2 = origin + Vec3::new(1.5 * dx, 2.5 * dx, 3.5 * dx);
        assert!(
            (c2 - expected2).length() < 1e-6,
            "cell(1,2,3) center wrong: got {c2:?} expected {expected2:?}"
        );
    }

    #[test]
    fn test_velocity_at_zero_field() {
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let center = g.cell_center(2, 2, 2);
        let vel = g.velocity_at(center);
        assert!(
            vel.length() < 1e-6,
            "Velocity in zero field should be zero, got {vel:?}"
        );
    }

    #[test]
    fn test_velocity_at_uniform_field() {
        let mut g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        // Set all u-face velocities to 1.0
        for val in g.u.iter_mut() {
            *val = 1.0;
        }
        // Sample at several interior cell centers
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let center = g.cell_center(i, j, k);
                    let vel = g.velocity_at(center);
                    assert!(
                        (vel.x - 1.0).abs() < 1e-4,
                        "Uniform u=1 should give vx=1 at ({i},{j},{k}), got {:.6}",
                        vel.x
                    );
                    assert!(
                        vel.y.abs() < 1e-4,
                        "vy should be 0 at ({i},{j},{k}), got {:.6}",
                        vel.y
                    );
                    assert!(
                        vel.z.abs() < 1e-4,
                        "vz should be 0 at ({i},{j},{k}), got {:.6}",
                        vel.z
                    );
                }
            }
        }
    }

    #[test]
    fn test_velocity_at_interpolation() {
        // Linear gradient in u: u(i,j,k) = i as f32
        // At x-face i, the value is i. Between faces 1 and 2 (midpoint)
        // the interpolated value should be 1.5.
        let nx = 4;
        let ny = 4;
        let nz = 4;
        let dx = 1.0;
        let mut g = FluidGrid::new(nx, ny, nz, dx, Vec3::ZERO);

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let idx = g.idx_u(i, j, k);
                    g.u[idx] = i as f32;
                }
            }
        }

        // Cell center of cell (1,1,1) is at (1.5, 1.5, 1.5).
        // u lives on x-faces at x=0,1,2,3,4. Midpoint 1.5 is between
        // face i=1 (val=1) and face i=2 (val=2), so interpolated u = 1.5.
        let pos = g.cell_center(1, 1, 1);
        let vel = g.velocity_at(pos);
        assert!(
            (vel.x - 1.5).abs() < 1e-4,
            "Interpolated u at cell center (1,1,1) should be 1.5, got {:.6}",
            vel.x
        );
    }

    #[test]
    fn test_idx_roundtrip() {
        let g = FluidGrid::new(7, 5, 3, 0.1, Vec3::ZERO);
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let linear = g.idx(i, j, k);
                    let (ri, rj, rk) = g.idx_to_ijk(linear);
                    assert_eq!(
                        (ri, rj, rk),
                        (i, j, k),
                        "Roundtrip failed for ({i},{j},{k}): got ({ri},{rj},{rk})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_in_bounds_edges() {
        let g = FluidGrid::new(4, 5, 6, 0.1, Vec3::ZERO);
        assert!(g.in_bounds(0, 0, 0), "(0,0,0) should be in bounds");
        assert!(g.in_bounds(3, 4, 5), "(3,4,5) should be in bounds");
        assert!(!g.in_bounds(-1, 0, 0), "(-1,0,0) should be out of bounds");
        assert!(!g.in_bounds(0, -1, 0), "(0,-1,0) should be out of bounds");
        assert!(!g.in_bounds(0, 0, -1), "(0,0,-1) should be out of bounds");
        assert!(!g.in_bounds(4, 0, 0), "(nx,0,0) should be out of bounds");
        assert!(!g.in_bounds(0, 5, 0), "(0,ny,0) should be out of bounds");
        assert!(!g.in_bounds(0, 0, 6), "(0,0,nz) should be out of bounds");
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be > 0")]
    fn test_grid_dimension_validation() {
        FluidGrid::new(0, 4, 4, 0.1, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_grid_max_size_guard() {
        FluidGrid::new(1025, 4, 4, 0.1, Vec3::ZERO);
    }
}
