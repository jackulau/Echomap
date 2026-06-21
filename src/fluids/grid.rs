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

    /// Scratch buffers for the advection step — swapped with `u/v/w` to avoid
    /// per-step heap allocation. Sized to match `u/v/w` respectively.
    #[doc(hidden)]
    pub scratch_u: Vec<f32>,
    #[doc(hidden)]
    pub scratch_v: Vec<f32>,
    #[doc(hidden)]
    pub scratch_w: Vec<f32>,
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
        // dx <= 0 (or non-finite) floods every field with NaN, and the CFL stability clamp can't
        // catch it because `NaN.min(x) == x`. Reject it at construction.
        assert!(
            dx.is_finite() && dx > 0.0,
            "grid dx must be finite and > 0, got {dx}"
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
            scratch_u: vec![0.0; (nx + 1) * ny * nz],
            scratch_v: vec![0.0; nx * (ny + 1) * nz],
            scratch_w: vec![0.0; nx * ny * (nz + 1)],
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
    #[allow(dead_code)]
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
    #[cfg(test)]
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

    // =========================================================================
    // Edge case tests
    // =========================================================================

    // --- Minimum grid (1x1x1) ---

    #[test]
    fn test_grid_1x1x1_creation() {
        let g = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        assert_eq!(g.pressure.len(), 1);
        assert_eq!(g.u.len(), 2); // (1+1)*1*1
        assert_eq!(g.v.len(), 2); // 1*(1+1)*1
        assert_eq!(g.w.len(), 2); // 1*1*(1+1)
    }

    #[test]
    fn test_grid_1x1x1_idx_roundtrip() {
        let g = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        let linear = g.idx(0, 0, 0);
        assert_eq!(linear, 0);
        let (ri, rj, rk) = g.idx_to_ijk(linear);
        assert_eq!((ri, rj, rk), (0, 0, 0));
    }

    #[test]
    fn test_grid_1x1x1_velocity_at() {
        let mut g = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        g.u[0] = 2.0;
        g.u[1] = 4.0;
        // velocity_at at cell center (0.5, 0.5, 0.5) should interpolate u
        let vel = g.velocity_at(g.cell_center(0, 0, 0));
        assert!(vel.x.is_finite(), "1x1x1 velocity_at.x should be finite");
        assert!(vel.y.is_finite(), "1x1x1 velocity_at.y should be finite");
        assert!(vel.z.is_finite(), "1x1x1 velocity_at.z should be finite");
    }

    #[test]
    fn test_grid_1x1x1_in_bounds() {
        let g = FluidGrid::new(1, 1, 1, 1.0, Vec3::ZERO);
        assert!(g.in_bounds(0, 0, 0));
        assert!(!g.in_bounds(1, 0, 0));
        assert!(!g.in_bounds(0, 1, 0));
        assert!(!g.in_bounds(0, 0, 1));
        assert!(!g.in_bounds(-1, 0, 0));
    }

    // --- Zero/panic dimension tests for y and z axes ---

    #[test]
    #[should_panic(expected = "Grid dimensions must be > 0")]
    fn test_grid_zero_ny() {
        FluidGrid::new(4, 0, 4, 0.1, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be > 0")]
    fn test_grid_zero_nz() {
        FluidGrid::new(4, 4, 0, 0.1, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_grid_max_size_ny() {
        FluidGrid::new(4, 1025, 4, 0.1, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_grid_max_size_nz() {
        FluidGrid::new(4, 4, 1025, 0.1, Vec3::ZERO);
    }

    // --- Asymmetric grid dimensions ---

    #[test]
    fn test_grid_asymmetric_dimensions() {
        // nx=1, ny=2, nz=3 -- very non-cubic
        let g = FluidGrid::new(1, 2, 3, 0.5, Vec3::ZERO);
        assert_eq!(g.u.len(), 2 * 2 * 3); // (1+1)*2*3
        assert_eq!(g.v.len(), 3 * 3); // 1*(2+1)*3
        assert_eq!(g.w.len(), 2 * 4); // 1*2*(3+1)
        assert_eq!(g.pressure.len(), 2 * 3);
    }

    #[test]
    fn test_idx_roundtrip_asymmetric() {
        let g = FluidGrid::new(3, 5, 7, 0.1, Vec3::ZERO);
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let linear = g.idx(i, j, k);
                    assert!(
                        linear < g.pressure.len(),
                        "idx out of bounds at ({i},{j},{k})"
                    );
                    let (ri, rj, rk) = g.idx_to_ijk(linear);
                    assert_eq!((ri, rj, rk), (i, j, k));
                }
            }
        }
    }

    // --- Face index boundary checks ---

    #[test]
    fn test_face_indices_within_bounds() {
        let g = FluidGrid::new(3, 4, 5, 0.1, Vec3::ZERO);
        // u-faces: (nx+1) x ny x nz
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..=g.nx {
                    let idx = g.idx_u(i, j, k);
                    assert!(
                        idx < g.u.len(),
                        "idx_u({i},{j},{k})={idx} out of bounds (len={})",
                        g.u.len()
                    );
                }
            }
        }
        // v-faces: nx x (ny+1) x nz
        for k in 0..g.nz {
            for j in 0..=g.ny {
                for i in 0..g.nx {
                    let idx = g.idx_v(i, j, k);
                    assert!(
                        idx < g.v.len(),
                        "idx_v({i},{j},{k})={idx} out of bounds (len={})",
                        g.v.len()
                    );
                }
            }
        }
        // w-faces: nx x ny x (nz+1)
        for k in 0..=g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let idx = g.idx_w(i, j, k);
                    assert!(
                        idx < g.w.len(),
                        "idx_w({i},{j},{k})={idx} out of bounds (len={})",
                        g.w.len()
                    );
                }
            }
        }
    }

    // --- Velocity interpolation at domain boundaries ---

    #[test]
    fn test_velocity_at_origin() {
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let vel = g.velocity_at(Vec3::ZERO);
        assert!(vel.x.is_finite(), "velocity at origin x should be finite");
        assert!(vel.y.is_finite(), "velocity at origin y should be finite");
        assert!(vel.z.is_finite(), "velocity at origin z should be finite");
    }

    #[test]
    fn test_velocity_at_far_corner() {
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        // Far corner of domain: (4*0.25, 4*0.25, 4*0.25) = (1.0, 1.0, 1.0)
        let vel = g.velocity_at(Vec3::new(1.0, 1.0, 1.0));
        assert!(
            vel.x.is_finite(),
            "velocity at far corner x should be finite"
        );
        assert!(
            vel.y.is_finite(),
            "velocity at far corner y should be finite"
        );
        assert!(
            vel.z.is_finite(),
            "velocity at far corner z should be finite"
        );
    }

    #[test]
    fn test_velocity_at_outside_domain() {
        // Position well outside the grid -- clamping should keep it finite
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let vel = g.velocity_at(Vec3::new(-10.0, -10.0, -10.0));
        assert!(
            vel.x.is_finite(),
            "velocity far outside should clamp, not crash"
        );
        let vel2 = g.velocity_at(Vec3::new(100.0, 100.0, 100.0));
        assert!(
            vel2.x.is_finite(),
            "velocity far outside should clamp, not crash"
        );
    }

    // --- Negative origin ---

    #[test]
    fn test_grid_negative_origin() {
        let origin = Vec3::new(-5.0, -5.0, -5.0);
        let g = FluidGrid::new(4, 4, 4, 1.0, origin);
        let c = g.cell_center(0, 0, 0);
        let expected = origin + Vec3::splat(0.5);
        assert!(
            (c - expected).length() < 1e-6,
            "cell_center with negative origin: got {c:?} expected {expected:?}"
        );
    }

    // --- Very small dx ---

    #[test]
    fn test_grid_tiny_dx() {
        let g = FluidGrid::new(2, 2, 2, 1e-6, Vec3::ZERO);
        let c = g.cell_center(0, 0, 0);
        assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
        let vel = g.velocity_at(c);
        assert!(vel.x.is_finite());
    }

    // --- Very large dx ---

    #[test]
    fn test_grid_large_dx() {
        let g = FluidGrid::new(2, 2, 2, 1e6, Vec3::ZERO);
        let c = g.cell_center(1, 1, 1);
        assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
    }

    // --- dx = 0.0 (degenerate cell size) is rejected at construction ---

    #[test]
    #[should_panic(expected = "grid dx must be finite and > 0")]
    fn test_grid_zero_dx_rejected() {
        // dx=0 would flood every field with NaN (and the CFL clamp can't catch it because
        // NaN.min(x) == x), so construction must reject it up front.
        FluidGrid::new(2, 2, 2, 0.0, Vec3::ZERO);
    }

    #[test]
    #[should_panic(expected = "grid dx must be finite and > 0")]
    fn test_grid_negative_dx_rejected() {
        FluidGrid::new(2, 2, 2, -0.1, Vec3::ZERO);
    }

    // --- in_bounds extreme negative values ---

    #[test]
    fn test_in_bounds_i32_min() {
        let g = FluidGrid::new(4, 4, 4, 1.0, Vec3::ZERO);
        assert!(!g.in_bounds(i32::MIN, 0, 0));
        assert!(!g.in_bounds(0, i32::MIN, 0));
        assert!(!g.in_bounds(0, 0, i32::MIN));
        assert!(!g.in_bounds(i32::MAX, 0, 0));
        assert!(!g.in_bounds(0, i32::MAX, 0));
        assert!(!g.in_bounds(0, 0, i32::MAX));
    }

    // --- cell_center at maximum valid indices ---

    #[test]
    fn test_cell_center_max_indices() {
        let g = FluidGrid::new(4, 5, 6, 0.1, Vec3::ZERO);
        // Last cell: (3, 4, 5)
        let c = g.cell_center(3, 4, 5);
        let expected = Vec3::new(3.5 * 0.1, 4.5 * 0.1, 5.5 * 0.1);
        assert!(
            (c - expected).length() < 1e-5,
            "cell_center at max indices: got {c:?} expected {expected:?}"
        );
    }

    // --- NaN position in velocity_at ---

    #[test]
    fn test_velocity_at_nan_position() {
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let vel = g.velocity_at(Vec3::new(f32::NAN, 0.5, 0.5));
        // NaN input should propagate through but not panic
        let _ = vel;
    }

    // --- Infinity position in velocity_at ---

    #[test]
    fn test_velocity_at_inf_position() {
        let g = FluidGrid::new(4, 4, 4, 0.25, Vec3::ZERO);
        let vel = g.velocity_at(Vec3::new(f32::INFINITY, 0.5, 0.5));
        // Should clamp or produce some finite result, not panic
        let _ = vel;
    }
}
