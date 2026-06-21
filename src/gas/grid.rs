use glam::Vec3;

/// Gas species definition with physical properties and visualization color.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct GasSpecies {
    pub name: String,
    pub diffusion_coefficient: f32,
    pub molecular_weight: f32,
    pub density_at_stp: f32,
    pub color: [f32; 3],
}

/// Cell classification for gas boundary handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum GasCellType {
    Gas,
    Solid,
    Empty,
}

/// Maximum allowed dimension per axis (prevents OOM).
const MAX_DIM: usize = 1024;

/// 3D cell-centered gas grid.
///
/// Unlike the MAC staggered grid in `fluids`, all fields live at cell centers:
/// concentration (one Vec<f32> per species), temperature, pressure, and
/// velocity (stored as three separate arrays vel_x, vel_y, vel_z).
#[derive(Clone)]
pub struct GasGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f32,
    pub origin: Vec3,
    pub species: Vec<GasSpecies>,
    /// One concentration array per species, each of length nx*ny*nz.
    pub concentrations: Vec<Vec<f32>>,
    pub temperature: Vec<f32>,
    pub pressure: Vec<f32>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,
    pub cell_types: Vec<GasCellType>,
    /// Per-step scratch buffer (length nx*ny*nz). Owned by the grid so solvers
    /// can swap with `concentrations[s]` / `temperature` instead of allocating
    /// a fresh Vec each call.
    #[doc(hidden)]
    pub scratch_scalar: Vec<f32>,
}

impl GasGrid {
    /// Create a new gas grid with all fields zeroed.
    ///
    /// # Panics
    /// - Any dimension is 0.
    /// - Any dimension exceeds `MAX_DIM` (1024).
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f32,
        origin: Vec3,
        species: Vec<GasSpecies>,
    ) -> Self {
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
        let num_species = species.len();
        let concentrations = vec![vec![0.0f32; cell_count]; num_species];

        Self {
            nx,
            ny,
            nz,
            dx,
            origin,
            species,
            concentrations,
            temperature: vec![0.0; cell_count],
            pressure: vec![0.0; cell_count],
            vel_x: vec![0.0; cell_count],
            vel_y: vec![0.0; cell_count],
            vel_z: vec![0.0; cell_count],
            cell_types: vec![GasCellType::Empty; cell_count],
            scratch_scalar: vec![0.0; cell_count],
        }
    }

    /// Cell-centered linear index.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
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
    #[allow(dead_code)]
    pub fn in_bounds(&self, i: i32, j: i32, k: i32) -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && (i as usize) < self.nx
            && (j as usize) < self.ny
            && (k as usize) < self.nz
    }

    /// Trilinear interpolation of the cell-centered velocity field at an
    /// arbitrary world-space position.
    pub fn velocity_at(&self, pos: Vec3) -> Vec3 {
        let vx = self.interpolate_cell_centered(&self.vel_x, pos);
        let vy = self.interpolate_cell_centered(&self.vel_y, pos);
        let vz = self.interpolate_cell_centered(&self.vel_z, pos);
        Vec3::new(vx, vy, vz)
    }

    /// Trilinear interpolation of concentration for a given species at a
    /// world-space position.
    #[allow(dead_code)]
    pub fn concentration_at(&self, species_idx: usize, pos: Vec3) -> f32 {
        self.interpolate_cell_centered(&self.concentrations[species_idx], pos)
    }

    /// Trilinear interpolation of the temperature field at a world-space position.
    #[allow(dead_code)]
    pub fn temperature_at(&self, pos: Vec3) -> f32 {
        self.interpolate_cell_centered(&self.temperature, pos)
    }

    // ----- Interpolation helpers -----

    /// Trilinear interpolation for a cell-centered scalar field.
    ///
    /// Cell centers are at (i+0.5)*dx relative to origin, so we shift by -0.5
    /// to get fractional cell coordinates before interpolating.
    pub fn interpolate_cell_centered(&self, field: &[f32], pos: Vec3) -> f32 {
        let rel = pos - self.origin;
        let fi = rel.x / self.dx - 0.5;
        let fj = rel.y / self.dx - 0.5;
        let fk = rel.z / self.dx - 0.5;

        // Clamp to valid range
        let fi = fi.clamp(0.0, (self.nx - 1) as f32);
        let fj = fj.clamp(0.0, (self.ny - 1) as f32);
        let fk = fk.clamp(0.0, (self.nz - 1) as f32);

        let i0 = (fi.floor() as usize).min(self.nx.saturating_sub(2));
        let j0 = (fj.floor() as usize).min(self.ny.saturating_sub(2));
        let k0 = (fk.floor() as usize).min(self.nz.saturating_sub(2));
        let i1 = (i0 + 1).min(self.nx - 1);
        let j1 = (j0 + 1).min(self.ny - 1);
        let k1 = (k0 + 1).min(self.nz - 1);

        let s = fi - i0 as f32;
        let t = fj - j0 as f32;
        let r = fk - k0 as f32;

        let c000 = field[self.idx(i0, j0, k0)];
        let c100 = field[self.idx(i1, j0, k0)];
        let c010 = field[self.idx(i0, j1, k0)];
        let c110 = field[self.idx(i1, j1, k0)];
        let c001 = field[self.idx(i0, j0, k1)];
        let c101 = field[self.idx(i1, j0, k1)];
        let c011 = field[self.idx(i0, j1, k1)];
        let c111 = field[self.idx(i1, j1, k1)];

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

    fn make_species(name: &str) -> GasSpecies {
        GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: 0.2,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn test_gas_grid_creation() {
        let species = vec![make_species("CO2"), make_species("CH4")];
        let g = GasGrid::new(4, 5, 6, 0.1, Vec3::ZERO, species);
        let cell_count = 4 * 5 * 6;

        // Cell-centered arrays
        assert_eq!(g.temperature.len(), cell_count);
        assert_eq!(g.pressure.len(), cell_count);
        assert_eq!(g.vel_x.len(), cell_count);
        assert_eq!(g.vel_y.len(), cell_count);
        assert_eq!(g.vel_z.len(), cell_count);
        assert_eq!(g.cell_types.len(), cell_count);

        // Concentration arrays: one per species, each of length cell_count
        assert_eq!(g.concentrations.len(), 2);
        assert_eq!(g.concentrations[0].len(), cell_count);
        assert_eq!(g.concentrations[1].len(), cell_count);
    }

    #[test]
    fn test_gas_grid_cell_center() {
        let origin = Vec3::new(1.0, 2.0, 3.0);
        let dx = 0.5;
        let species = vec![make_species("Air")];
        let g = GasGrid::new(4, 4, 4, dx, origin, species);
        let c = g.cell_center(0, 0, 0);
        let expected = origin + Vec3::splat(dx / 2.0);
        assert!(
            (c - expected).length() < 1e-6,
            "cell(0,0,0) center should be origin + dx/2, got {c:?} expected {expected:?}"
        );
    }

    #[test]
    fn test_gas_grid_velocity_at_uniform() {
        let species = vec![make_species("Air")];
        let mut g = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        // Set uniform velocity field: vx=1, vy=2, vz=3
        for val in g.vel_x.iter_mut() {
            *val = 1.0;
        }
        for val in g.vel_y.iter_mut() {
            *val = 2.0;
        }
        for val in g.vel_z.iter_mut() {
            *val = 3.0;
        }

        // Sample at several interior cell centers
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let center = g.cell_center(i, j, k);
                    let vel = g.velocity_at(center);
                    assert!(
                        (vel.x - 1.0).abs() < 1e-4,
                        "Uniform vx=1 should give vx=1 at ({i},{j},{k}), got {:.6}",
                        vel.x
                    );
                    assert!(
                        (vel.y - 2.0).abs() < 1e-4,
                        "Uniform vy=2 should give vy=2 at ({i},{j},{k}), got {:.6}",
                        vel.y
                    );
                    assert!(
                        (vel.z - 3.0).abs() < 1e-4,
                        "Uniform vz=3 should give vz=3 at ({i},{j},{k}), got {:.6}",
                        vel.z
                    );
                }
            }
        }
    }

    #[test]
    fn test_gas_grid_concentration_at_uniform() {
        let species = vec![make_species("CO2")];
        let mut g = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        // Set uniform concentration of 5.0
        for val in g.concentrations[0].iter_mut() {
            *val = 5.0;
        }

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let center = g.cell_center(i, j, k);
                    let c = g.concentration_at(0, center);
                    assert!(
                        (c - 5.0).abs() < 1e-4,
                        "Uniform concentration=5 should interpolate to 5 at ({i},{j},{k}), got {c:.6}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gas_grid_idx_roundtrip() {
        let species = vec![make_species("Air")];
        let g = GasGrid::new(7, 5, 3, 0.1, Vec3::ZERO, species);
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
    fn test_gas_grid_in_bounds() {
        let species = vec![make_species("Air")];
        let g = GasGrid::new(4, 5, 6, 0.1, Vec3::ZERO, species);
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
    fn test_gas_grid_dimension_validation() {
        let species = vec![make_species("Air")];
        GasGrid::new(0, 4, 4, 0.1, Vec3::ZERO, species);
    }

    #[test]
    fn test_gas_grid_multi_species() {
        let species = vec![
            make_species("CO2"),
            make_species("CH4"),
            make_species("N2O"),
        ];
        let mut g = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        let cell_count = 4 * 4 * 4;

        // Verify 3 independent concentration arrays
        assert_eq!(g.concentrations.len(), 3);
        for arr in &g.concentrations {
            assert_eq!(arr.len(), cell_count);
        }

        // Set different uniform values for each species
        for val in g.concentrations[0].iter_mut() {
            *val = 1.0;
        }
        for val in g.concentrations[1].iter_mut() {
            *val = 2.0;
        }
        for val in g.concentrations[2].iter_mut() {
            *val = 3.0;
        }

        // Verify independence: modifying one species doesn't affect others
        let center = g.cell_center(2, 2, 2);
        let c0 = g.concentration_at(0, center);
        let c1 = g.concentration_at(1, center);
        let c2 = g.concentration_at(2, center);
        assert!((c0 - 1.0).abs() < 1e-4, "Species 0 should be 1.0, got {c0}");
        assert!((c1 - 2.0).abs() < 1e-4, "Species 1 should be 2.0, got {c1}");
        assert!((c2 - 3.0).abs() < 1e-4, "Species 2 should be 3.0, got {c2}");
    }

    // ---- Q3 Edge Case Tests ----

    #[test]
    #[should_panic(expected = "Grid dimensions must be > 0")]
    fn test_edge_zero_ny() {
        let species = vec![make_species("Air")];
        GasGrid::new(4, 0, 4, 0.1, Vec3::ZERO, species);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be > 0")]
    fn test_edge_zero_nz() {
        let species = vec![make_species("Air")];
        GasGrid::new(4, 4, 0, 0.1, Vec3::ZERO, species);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_edge_exceed_max_dim_nx() {
        let species = vec![make_species("Air")];
        GasGrid::new(1025, 1, 1, 0.1, Vec3::ZERO, species);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_edge_exceed_max_dim_ny() {
        let species = vec![make_species("Air")];
        GasGrid::new(1, 1025, 1, 0.1, Vec3::ZERO, species);
    }

    #[test]
    #[should_panic(expected = "Grid dimensions must be <= 1024")]
    fn test_edge_exceed_max_dim_nz() {
        let species = vec![make_species("Air")];
        GasGrid::new(1, 1, 1025, 0.1, Vec3::ZERO, species);
    }

    #[test]
    fn test_edge_empty_species_list() {
        let g = GasGrid::new(4, 4, 4, 0.1, Vec3::ZERO, vec![]);
        assert_eq!(g.species.len(), 0);
        assert_eq!(g.concentrations.len(), 0);
        // Other fields should still be allocated
        assert_eq!(g.temperature.len(), 64);
        assert_eq!(g.vel_x.len(), 64);
    }

    #[test]
    fn test_edge_single_cell_grid() {
        let species = vec![make_species("Air")];
        let g = GasGrid::new(1, 1, 1, 1.0, Vec3::ZERO, species);
        assert_eq!(g.idx(0, 0, 0), 0);
        let (i, j, k) = g.idx_to_ijk(0);
        assert_eq!((i, j, k), (0, 0, 0));

        let center = g.cell_center(0, 0, 0);
        assert!((center.x - 0.5).abs() < 1e-6);
        assert!((center.y - 0.5).abs() < 1e-6);
        assert!((center.z - 0.5).abs() < 1e-6);

        assert!(g.in_bounds(0, 0, 0));
        assert!(!g.in_bounds(1, 0, 0));
        assert!(!g.in_bounds(0, 1, 0));
        assert!(!g.in_bounds(0, 0, 1));
    }

    #[test]
    fn test_edge_single_cell_interpolation() {
        // Test that interpolation works on 1x1x1 grid (saturating_sub edge)
        let species = vec![make_species("Air")];
        let mut g = GasGrid::new(1, 1, 1, 1.0, Vec3::ZERO, species);
        g.vel_x[0] = 7.0;
        g.vel_y[0] = 8.0;
        g.vel_z[0] = 9.0;
        g.concentrations[0][0] = 42.0;
        g.temperature[0] = 300.0;

        let center = g.cell_center(0, 0, 0);
        let vel = g.velocity_at(center);
        assert!(
            (vel.x - 7.0).abs() < 1e-4,
            "Single cell vel_x should be 7.0, got {}",
            vel.x
        );
        assert!(
            (vel.y - 8.0).abs() < 1e-4,
            "Single cell vel_y should be 8.0, got {}",
            vel.y
        );
        assert!(
            (vel.z - 9.0).abs() < 1e-4,
            "Single cell vel_z should be 9.0, got {}",
            vel.z
        );

        let c = g.concentration_at(0, center);
        assert!(
            (c - 42.0).abs() < 1e-4,
            "Single cell concentration should be 42.0, got {c}"
        );

        let t = g.temperature_at(center);
        assert!(
            (t - 300.0).abs() < 1e-4,
            "Single cell temperature should be 300.0, got {t}"
        );
    }

    #[test]
    fn test_edge_interpolation_outside_grid_clamps() {
        // Position far outside the grid should clamp to boundary values
        let species = vec![make_species("Air")];
        let mut g = GasGrid::new(4, 4, 4, 0.25, Vec3::ZERO, species);
        // Set a known gradient: concentration increases with i
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let idx = g.idx(i, j, k);
                    g.concentrations[0][idx] = i as f32 * 10.0;
                }
            }
        }

        // Position way below origin should clamp to i=0 face
        let below = Vec3::new(-100.0, -100.0, -100.0);
        let c_below = g.concentration_at(0, below);
        assert!(
            (c_below - 0.0).abs() < 1e-3,
            "Below grid should clamp to min, got {c_below}"
        );

        // Position way above grid should clamp to i=3 face
        let above = Vec3::new(100.0, 100.0, 100.0);
        let c_above = g.concentration_at(0, above);
        assert!(
            (c_above - 30.0).abs() < 1e-3,
            "Above grid should clamp to max, got {c_above}"
        );
    }

    #[test]
    fn test_edge_non_uniform_dimensions_idx_roundtrip() {
        // Highly non-square grid
        let species = vec![make_species("Air")];
        let g = GasGrid::new(2, 3, 5, 0.1, Vec3::ZERO, species);
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let idx = g.idx(i, j, k);
                    let (ri, rj, rk) = g.idx_to_ijk(idx);
                    assert_eq!(
                        (ri, rj, rk),
                        (i, j, k),
                        "Non-uniform roundtrip failed for ({i},{j},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_edge_cell_center_at_max_indices() {
        let species = vec![make_species("Air")];
        let g = GasGrid::new(4, 5, 6, 0.5, Vec3::new(1.0, 2.0, 3.0), species);
        let c = g.cell_center(3, 4, 5);
        let expected = Vec3::new(
            1.0 + (3.0 + 0.5) * 0.5,
            2.0 + (4.0 + 0.5) * 0.5,
            3.0 + (5.0 + 0.5) * 0.5,
        );
        assert!(
            (c - expected).length() < 1e-6,
            "cell_center at max should be correct: got {c:?}, expected {expected:?}"
        );
    }

    #[test]
    fn test_edge_in_bounds_i32_extremes() {
        let species = vec![make_species("Air")];
        let g = GasGrid::new(4, 4, 4, 0.1, Vec3::ZERO, species);
        assert!(
            !g.in_bounds(i32::MIN, 0, 0),
            "i32::MIN should be out of bounds"
        );
        assert!(
            !g.in_bounds(0, i32::MIN, 0),
            "i32::MIN j should be out of bounds"
        );
        assert!(
            !g.in_bounds(0, 0, i32::MIN),
            "i32::MIN k should be out of bounds"
        );
        assert!(
            !g.in_bounds(i32::MAX, 0, 0),
            "i32::MAX should be out of bounds"
        );
        assert!(
            !g.in_bounds(0, i32::MAX, 0),
            "i32::MAX j should be out of bounds"
        );
        assert!(
            !g.in_bounds(0, 0, i32::MAX),
            "i32::MAX k should be out of bounds"
        );
    }

    #[test]
    fn test_edge_max_dim_boundary_grid() {
        // Grid at MAX_DIM should succeed
        let species = vec![make_species("Air")];
        let g = GasGrid::new(1024, 1, 1, 0.01, Vec3::ZERO, species);
        assert_eq!(g.nx, 1024);
        assert_eq!(g.temperature.len(), 1024);
    }

    #[test]
    fn test_edge_temperature_at_interpolation() {
        let species = vec![make_species("Air")];
        let mut g = GasGrid::new(2, 2, 2, 1.0, Vec3::ZERO, species);
        // Set gradient: temperature = i*100
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    let idx = g.idx(i, j, k);
                    g.temperature[idx] = i as f32 * 100.0;
                }
            }
        }
        // Midpoint between cell centers should interpolate
        let mid = Vec3::new(1.0, 0.5, 0.5); // between i=0 center (0.5) and i=1 center (1.5)
        let t = g.temperature_at(mid);
        assert!(
            (t - 50.0).abs() < 1e-2,
            "Temperature midpoint should interpolate to 50, got {t}"
        );
    }

    #[test]
    #[should_panic(expected = "grid dx must be finite and > 0")]
    fn test_gas_grid_zero_dx_rejected() {
        // dx=0 would flood every field with NaN; construction must reject it up front.
        let species = vec![make_species("Air")];
        GasGrid::new(2, 2, 2, 0.0, Vec3::ZERO, species);
    }

    #[test]
    #[should_panic(expected = "grid dx must be finite and > 0")]
    fn test_gas_grid_negative_dx_rejected() {
        let species = vec![make_species("Air")];
        GasGrid::new(2, 2, 2, -1.0, Vec3::ZERO, species);
    }
}
