use glam::Vec3;

use super::grid::{GasCellType, GasGrid};
use crate::scene::SceneObject;

/// A gas source that injects concentration of a specific species into the grid.
#[derive(Clone, Debug)]
pub struct GasSource {
    /// World-space position of the source center.
    pub position: Vec3,
    /// Index into `GasGrid::species` for the species to inject.
    pub species_index: usize,
    /// Concentration injection rate (concentration/second).
    pub rate: f32,
    /// Radius of influence (world units). Cells within this radius receive concentration.
    pub radius: f32,
}

/// Mark grid cells overlapping solid scene meshes as `GasCellType::Solid` using
/// AABB intersection between each mesh's bounding box and each grid cell.
///
/// Follows the same AABB overlap pattern as `fluids::boundary::voxelize_scene`.
pub fn voxelize_scene(grid: &mut GasGrid, meshes: &[SceneObject]) {
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
                        grid.cell_types[idx] = GasCellType::Solid;
                    }
                }
            }
        }
    }
}

/// Enforce boundary conditions for the gas grid.
///
/// - Zero velocity at solid boundaries: if a cell is Solid, its velocity is
///   zeroed. Additionally, Gas cells adjacent to Solid cells have the velocity
///   component pointing toward the solid face zeroed (no-penetration).
/// - Zero-gradient concentration at walls (Neumann BC): solid cells copy
///   concentration from the average of their Gas neighbours.
#[allow(dead_code)]
pub fn enforce_boundary_conditions(grid: &mut GasGrid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    // --- Zero velocity at solid cells ---
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] == GasCellType::Solid {
                    grid.vel_x[idx] = 0.0;
                    grid.vel_y[idx] = 0.0;
                    grid.vel_z[idx] = 0.0;
                }
            }
        }
    }

    // --- No-penetration: zero velocity component on Gas cells facing Solid ---
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Gas {
                    continue;
                }

                // If neighbour in -x is Solid and vel_x < 0, zero it (flowing into wall)
                if i > 0
                    && grid.cell_types[grid.idx(i - 1, j, k)] == GasCellType::Solid
                    && grid.vel_x[idx] < 0.0
                {
                    grid.vel_x[idx] = 0.0;
                }
                // If neighbour in +x is Solid and vel_x > 0, zero it
                if i + 1 < nx
                    && grid.cell_types[grid.idx(i + 1, j, k)] == GasCellType::Solid
                    && grid.vel_x[idx] > 0.0
                {
                    grid.vel_x[idx] = 0.0;
                }
                // -y
                if j > 0
                    && grid.cell_types[grid.idx(i, j - 1, k)] == GasCellType::Solid
                    && grid.vel_y[idx] < 0.0
                {
                    grid.vel_y[idx] = 0.0;
                }
                // +y
                if j + 1 < ny
                    && grid.cell_types[grid.idx(i, j + 1, k)] == GasCellType::Solid
                    && grid.vel_y[idx] > 0.0
                {
                    grid.vel_y[idx] = 0.0;
                }
                // -z
                if k > 0
                    && grid.cell_types[grid.idx(i, j, k - 1)] == GasCellType::Solid
                    && grid.vel_z[idx] < 0.0
                {
                    grid.vel_z[idx] = 0.0;
                }
                // +z
                if k + 1 < nz
                    && grid.cell_types[grid.idx(i, j, k + 1)] == GasCellType::Solid
                    && grid.vel_z[idx] > 0.0
                {
                    grid.vel_z[idx] = 0.0;
                }
            }
        }
    }

    // --- Zero-gradient concentration (Neumann BC) at solid walls ---
    // Copy concentration from average of Gas neighbours into Solid cells.
    let conc_snapshot: Vec<Vec<f32>> = grid.concentrations.clone();

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = grid.idx(i, j, k);
                if grid.cell_types[idx] != GasCellType::Solid {
                    continue;
                }

                // Collect Gas neighbour indices.
                let mut gas_neighbours = Vec::new();
                if i > 0 && grid.cell_types[grid.idx(i - 1, j, k)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i - 1, j, k));
                }
                if i + 1 < nx && grid.cell_types[grid.idx(i + 1, j, k)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i + 1, j, k));
                }
                if j > 0 && grid.cell_types[grid.idx(i, j - 1, k)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i, j - 1, k));
                }
                if j + 1 < ny && grid.cell_types[grid.idx(i, j + 1, k)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i, j + 1, k));
                }
                if k > 0 && grid.cell_types[grid.idx(i, j, k - 1)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i, j, k - 1));
                }
                if k + 1 < nz && grid.cell_types[grid.idx(i, j, k + 1)] == GasCellType::Gas {
                    gas_neighbours.push(grid.idx(i, j, k + 1));
                }

                if gas_neighbours.is_empty() {
                    continue;
                }

                let count = gas_neighbours.len() as f32;
                for (s, conc_snap) in conc_snapshot.iter().enumerate() {
                    let avg: f32 =
                        gas_neighbours.iter().map(|&n| conc_snap[n]).sum::<f32>() / count;
                    grid.concentrations[s][idx] = avg;
                }
            }
        }
    }
}

/// Classify cells as Gas, Solid, or Empty.
///
/// After voxelization marks Solid cells, the remaining cells are classified:
/// - Cells already `Solid` stay `Solid`.
/// - All other cells become `Gas` (in the gas context, non-solid cells within
///   the simulation domain are gas cells, unlike fluids which use a level set
///   to distinguish Fluid from Air).
#[allow(dead_code)]
pub fn classify_cells(grid: &mut GasGrid) {
    let n = grid.cell_types.len();
    for idx in 0..n {
        if grid.cell_types[idx] == GasCellType::Solid {
            continue;
        }
        // Non-solid cells are Gas (the simulation domain).
        grid.cell_types[idx] = GasCellType::Gas;
    }
}

/// Inject concentration at source locations.
///
/// For each source, all Gas cells whose center falls within the source radius
/// receive `source.rate * dt` concentration of the specified species.
pub fn apply_sources(grid: &mut GasGrid, sources: &[GasSource], dt: f32) {
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;

    for source in sources {
        if source.species_index >= grid.concentrations.len() {
            continue;
        }
        let r_sq = source.radius * source.radius;

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = grid.idx(i, j, k);
                    if grid.cell_types[idx] != GasCellType::Gas {
                        continue;
                    }
                    let center = grid.cell_center(i, j, k);
                    let dist_sq = (center - source.position).length_squared();
                    if dist_sq <= r_sq {
                        grid.concentrations[source.species_index][idx] += source.rate * dt;
                    }
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
    use crate::scene::{Mesh, SceneObject, Triangle, Vertex};

    use super::super::grid::GasSpecies;

    fn make_species(name: &str) -> GasSpecies {
        GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: 0.2,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

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
        triangles.extend(quad(p[0], p[1], p[2], p[3])); // bottom
        triangles.extend(quad(p[4], p[7], p[6], p[5])); // top
        triangles.extend(quad(p[0], p[4], p[5], p[1])); // front
        triangles.extend(quad(p[2], p[6], p[7], p[3])); // back
        triangles.extend(quad(p[3], p[7], p[4], p[0])); // left
        triangles.extend(quad(p[1], p[5], p[6], p[2])); // right

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

    // ----- 5 required tests -----

    #[test]
    fn test_gas_boundary_voxelize_marks_solid() {
        // 8x8x8 grid, dx=1.0, origin at (0,0,0). Grid spans [0,8]^3.
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Place a box from (2,2,2) to (5,5,5) => cells (2..5, 2..5, 2..5) should be Solid.
        let mesh = box_mesh(Vec3::new(2.0, 2.0, 2.0), Vec3::new(5.0, 5.0, 5.0));
        let obj = make_scene_object(mesh);

        voxelize_scene(&mut grid, &[obj]);

        let mut solid_count = 0;
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    let idx = grid.idx(i, j, k);
                    if i >= 2 && i < 5 && j >= 2 && j < 5 && k >= 2 && k < 5 {
                        assert_eq!(
                            grid.cell_types[idx],
                            GasCellType::Solid,
                            "Cell ({i},{j},{k}) inside box should be Solid"
                        );
                        solid_count += 1;
                    }
                }
            }
        }
        assert_eq!(solid_count, 3 * 3 * 3, "Should have 27 solid cells");

        // Cells outside the box should remain Empty (the default).
        let idx_outside = grid.idx(0, 0, 0);
        assert_eq!(
            grid.cell_types[idx_outside],
            GasCellType::Empty,
            "Cell (0,0,0) outside box should remain Empty"
        );
    }

    #[test]
    fn test_gas_boundary_zeroes_velocity() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Mark all cells as Gas, then mark one cell as Solid.
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }
        let solid_idx = grid.idx(4, 4, 4);
        grid.cell_types[solid_idx] = GasCellType::Solid;

        // Set non-zero velocity everywhere.
        for val in grid.vel_x.iter_mut() {
            *val = 5.0;
        }
        for val in grid.vel_y.iter_mut() {
            *val = 5.0;
        }
        for val in grid.vel_z.iter_mut() {
            *val = 5.0;
        }

        enforce_boundary_conditions(&mut grid);

        // The solid cell itself should have zero velocity.
        assert!(
            grid.vel_x[solid_idx].abs() < 1e-6,
            "Solid cell vel_x should be zero, got {}",
            grid.vel_x[solid_idx]
        );
        assert!(
            grid.vel_y[solid_idx].abs() < 1e-6,
            "Solid cell vel_y should be zero, got {}",
            grid.vel_y[solid_idx]
        );
        assert!(
            grid.vel_z[solid_idx].abs() < 1e-6,
            "Solid cell vel_z should be zero, got {}",
            grid.vel_z[solid_idx]
        );

        // Gas cell at (5,4,4) has Solid neighbour in -x, and vel_x was positive (5.0),
        // which is flowing away from the wall, so it should be preserved.
        // Gas cell at (3,4,4) has Solid neighbour in +x, and vel_x was positive (5.0),
        // which flows toward the wall, so vel_x should be zeroed.
        let idx_3 = grid.idx(3, 4, 4);
        assert!(
            grid.vel_x[idx_3].abs() < 1e-6,
            "Gas cell (3,4,4) vel_x toward solid should be zero, got {}",
            grid.vel_x[idx_3]
        );
    }

    #[test]
    fn test_gas_boundary_preserves_interior() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Mark all as Gas, with solid border.
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }
        for k in 0..8 {
            for j in 0..8 {
                for i in 0..8 {
                    if i == 0 || i == 7 || j == 0 || j == 7 || k == 0 || k == 7 {
                        let idx = grid.idx(i, j, k);
                        grid.cell_types[idx] = GasCellType::Solid;
                    }
                }
            }
        }

        // Set known velocity on all interior Gas cells.
        for k in 2..6 {
            for j in 2..6 {
                for i in 2..6 {
                    let idx = grid.idx(i, j, k);
                    grid.vel_x[idx] = 3.14;
                    grid.vel_y[idx] = 2.72;
                    grid.vel_z[idx] = 1.41;
                }
            }
        }

        // Snapshot interior velocities.
        let mut interior_before = Vec::new();
        for k in 2..6 {
            for j in 2..6 {
                for i in 2..6 {
                    let idx = grid.idx(i, j, k);
                    interior_before.push((
                        i,
                        j,
                        k,
                        grid.vel_x[idx],
                        grid.vel_y[idx],
                        grid.vel_z[idx],
                    ));
                }
            }
        }

        enforce_boundary_conditions(&mut grid);

        // Interior cells far from solid walls should be unchanged.
        for (i, j, k, vx, vy, vz) in &interior_before {
            let idx = grid.idx(*i, *j, *k);
            assert!(
                (grid.vel_x[idx] - vx).abs() < 1e-6,
                "Interior vel_x at ({i},{j},{k}) should be preserved: before={vx}, after={}",
                grid.vel_x[idx]
            );
            assert!(
                (grid.vel_y[idx] - vy).abs() < 1e-6,
                "Interior vel_y at ({i},{j},{k}) should be preserved: before={vy}, after={}",
                grid.vel_y[idx]
            );
            assert!(
                (grid.vel_z[idx] - vz).abs() < 1e-6,
                "Interior vel_z at ({i},{j},{k}) should be preserved: before={vz}, after={}",
                grid.vel_z[idx]
            );
        }
    }

    #[test]
    fn test_gas_boundary_source_injection() {
        let species = vec![make_species("CO2"), make_species("CH4")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Mark all cells as Gas.
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }

        // Place a source at the center, species 0, rate=10.0, radius=1.5.
        let source = GasSource {
            position: grid.cell_center(4, 4, 4),
            species_index: 0,
            rate: 10.0,
            radius: 1.5,
        };

        // All concentrations start at zero.
        assert!(
            grid.concentrations[0].iter().all(|&v| v.abs() < 1e-10),
            "Concentrations should start at zero"
        );

        apply_sources(&mut grid, &[source], 1.0);

        // The center cell should have received concentration.
        let center_idx = grid.idx(4, 4, 4);
        assert!(
            grid.concentrations[0][center_idx] > 0.0,
            "Source should inject concentration at center, got {}",
            grid.concentrations[0][center_idx]
        );
        assert!(
            (grid.concentrations[0][center_idx] - 10.0).abs() < 1e-6,
            "Center cell should have rate=10.0 injected, got {}",
            grid.concentrations[0][center_idx]
        );

        // Species 1 should be unaffected.
        assert!(
            grid.concentrations[1].iter().all(|&v| v.abs() < 1e-10),
            "Species 1 should remain at zero"
        );

        // A far-away cell should be unaffected.
        let far_idx = grid.idx(0, 0, 0);
        assert!(
            grid.concentrations[0][far_idx].abs() < 1e-10,
            "Cell far from source should remain at zero"
        );
    }

    #[test]
    fn test_gas_boundary_classify_cells() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Mark a few cells as Solid (simulating voxelization).
        let solid_idx = grid.idx(3, 3, 3);
        grid.cell_types[solid_idx] = GasCellType::Solid;
        let solid_idx2 = grid.idx(4, 4, 4);
        grid.cell_types[solid_idx2] = GasCellType::Solid;

        // All others should be Empty (default from GasGrid::new).
        let empty_count = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Empty)
            .count();
        assert_eq!(
            empty_count,
            8 * 8 * 8 - 2,
            "Before classify, non-solid cells should be Empty"
        );

        classify_cells(&mut grid);

        // Solid cells stay Solid.
        assert_eq!(
            grid.cell_types[solid_idx],
            GasCellType::Solid,
            "Solid cell (3,3,3) should remain Solid"
        );
        assert_eq!(
            grid.cell_types[solid_idx2],
            GasCellType::Solid,
            "Solid cell (4,4,4) should remain Solid"
        );

        // All other cells should now be Gas.
        let gas_count = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Gas)
            .count();
        assert_eq!(
            gas_count,
            8 * 8 * 8 - 2,
            "Non-solid cells should be classified as Gas"
        );

        // No Empty cells should remain.
        let remaining_empty = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Empty)
            .count();
        assert_eq!(
            remaining_empty, 0,
            "classify_cells should convert all Empty to Gas"
        );
    }

    // ---- Q3 Edge Case Tests ----

    #[test]
    fn test_edge_voxelize_empty_meshes() {
        let species = vec![make_species("Air")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        voxelize_scene(&mut grid, &[]);

        // No cells should have been marked as Solid
        let solid_count = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Solid)
            .count();
        assert_eq!(solid_count, 0, "Empty mesh list should mark no cells solid");
    }

    #[test]
    fn test_edge_voxelize_mesh_outside_grid() {
        let species = vec![make_species("Air")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);

        // Mesh completely outside the grid bounds [0..8]
        let mesh = box_mesh(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(110.0, 110.0, 110.0),
        );
        let obj = make_scene_object(mesh);

        voxelize_scene(&mut grid, &[obj]);

        let solid_count = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Solid)
            .count();
        assert_eq!(
            solid_count, 0,
            "Mesh outside grid should mark no cells solid"
        );
    }

    #[test]
    fn test_edge_apply_sources_out_of_range_species() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }

        // species_index = 5, but only 1 species exists
        let source = GasSource {
            position: grid.cell_center(4, 4, 4),
            species_index: 5,
            rate: 10.0,
            radius: 2.0,
        };

        // Should not panic, should skip the source
        apply_sources(&mut grid, &[source], 1.0);

        let max_conc: f32 = grid.concentrations[0]
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max);
        assert!(
            max_conc < 1e-10,
            "Out-of-range species source should inject nothing, got max={max_conc}"
        );
    }

    #[test]
    fn test_edge_apply_sources_zero_radius() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }

        let source = GasSource {
            position: grid.cell_center(4, 4, 4),
            species_index: 0,
            rate: 10.0,
            radius: 0.0, // zero radius
        };

        apply_sources(&mut grid, &[source], 1.0);

        // With radius=0, dist_sq <= 0 requires dist_sq == 0, i.e. exact center match.
        // The cell center is exactly at the source position, so dist_sq = 0.0 <= 0.0 is true.
        let center_idx = grid.idx(4, 4, 4);
        let center_val = grid.concentrations[0][center_idx];

        // Exactly at center should still get injected (0 <= 0)
        assert!(
            (center_val - 10.0).abs() < 1e-6,
            "Zero radius: center cell should get injection (dist=0), got {center_val}"
        );

        // But no other cell should be affected
        let mut other_max = 0.0_f32;
        for (i, &v) in grid.concentrations[0].iter().enumerate() {
            if i != center_idx {
                other_max = other_max.max(v.abs());
            }
        }
        assert!(
            other_max < 1e-10,
            "Zero radius: non-center cells should have no injection, got max={other_max}"
        );
    }

    #[test]
    fn test_edge_apply_sources_empty_list() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(4, 4, 4, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Gas;
        }

        let before = grid.concentrations[0].clone();
        apply_sources(&mut grid, &[], 1.0);

        for (i, (b, a)) in before.iter().zip(grid.concentrations[0].iter()).enumerate() {
            assert!(
                (b - a).abs() < 1e-10,
                "Empty sources: concentration should not change at index {i}"
            );
        }
    }

    #[test]
    fn test_edge_apply_sources_on_solid_cell() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(8, 8, 8, 1.0, Vec3::ZERO, species);
        // Mark all as Solid
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }

        let source = GasSource {
            position: grid.cell_center(4, 4, 4),
            species_index: 0,
            rate: 10.0,
            radius: 3.0,
        };

        apply_sources(&mut grid, &[source], 1.0);

        // No cell should have received concentration (all Solid, source skips non-Gas)
        let max_conc: f32 = grid.concentrations[0]
            .iter()
            .map(|v| v.abs())
            .fold(0.0, f32::max);
        assert!(
            max_conc < 1e-10,
            "Source on solid cells should inject nothing, got max={max_conc}"
        );
    }

    #[test]
    fn test_edge_enforce_bc_all_solid() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(4, 4, 4, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }
        // Set non-zero velocity
        for v in grid.vel_x.iter_mut() {
            *v = 5.0;
        }
        for v in grid.vel_y.iter_mut() {
            *v = 3.0;
        }
        for v in grid.vel_z.iter_mut() {
            *v = 7.0;
        }

        enforce_boundary_conditions(&mut grid);

        // All velocity should be zeroed (all cells solid)
        assert!(
            grid.vel_x.iter().all(|&v| v.abs() < 1e-10),
            "All-solid: vel_x should be zeroed"
        );
        assert!(
            grid.vel_y.iter().all(|&v| v.abs() < 1e-10),
            "All-solid: vel_y should be zeroed"
        );
        assert!(
            grid.vel_z.iter().all(|&v| v.abs() < 1e-10),
            "All-solid: vel_z should be zeroed"
        );
    }

    #[test]
    fn test_edge_enforce_bc_single_cell_gas() {
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(1, 1, 1, 1.0, Vec3::ZERO, species);
        grid.cell_types[0] = GasCellType::Gas;
        grid.vel_x[0] = 5.0;
        grid.vel_y[0] = -3.0;
        grid.vel_z[0] = 2.0;
        grid.concentrations[0][0] = 42.0;

        enforce_boundary_conditions(&mut grid);

        // Single Gas cell with no neighbours -- velocity should be preserved
        // (no solid neighbours to enforce no-penetration)
        assert!(
            (grid.vel_x[0] - 5.0).abs() < 1e-6,
            "Single gas cell vel_x should be preserved"
        );
        assert!(
            (grid.vel_y[0] - (-3.0)).abs() < 1e-6,
            "Single gas cell vel_y should be preserved"
        );
        assert!(
            (grid.vel_z[0] - 2.0).abs() < 1e-6,
            "Single gas cell vel_z should be preserved"
        );
        assert!(
            (grid.concentrations[0][0] - 42.0).abs() < 1e-6,
            "Single gas cell concentration should be preserved"
        );
    }

    #[test]
    fn test_edge_neumann_bc_solid_no_gas_neighbours() {
        // Solid cell surrounded by other Solid cells -- no Gas neighbours
        let species = vec![make_species("CO2")];
        let mut grid = GasGrid::new(4, 4, 4, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }
        // Set some concentration on the center solid cell
        let center = grid.idx(2, 2, 2);
        grid.concentrations[0][center] = 99.0;

        enforce_boundary_conditions(&mut grid);

        // Solid with no Gas neighbours should keep its concentration (skip branch)
        assert!(
            (grid.concentrations[0][center] - 99.0).abs() < 1e-6,
            "Solid cell with no gas neighbours should keep concentration"
        );
    }

    #[test]
    fn test_edge_classify_cells_already_all_gas() {
        let species = vec![make_species("Air")];
        let mut grid = GasGrid::new(4, 4, 4, 1.0, Vec3::ZERO, species);
        // First classify: Empty -> Gas
        classify_cells(&mut grid);

        let gas_count_before = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Gas)
            .count();
        assert_eq!(gas_count_before, 64);

        // Classify again -- should be idempotent
        classify_cells(&mut grid);

        let gas_count_after = grid
            .cell_types
            .iter()
            .filter(|&&ct| ct == GasCellType::Gas)
            .count();
        assert_eq!(
            gas_count_after, 64,
            "classify_cells should be idempotent on all-Gas grid"
        );
    }

    #[test]
    fn test_edge_no_penetration_all_directions() {
        // Gas cell surrounded by Solid on all 6 faces
        let species = vec![make_species("Air")];
        let mut grid = GasGrid::new(3, 3, 3, 1.0, Vec3::ZERO, species);
        // All solid except center
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }
        let center = grid.idx(1, 1, 1);
        grid.cell_types[center] = GasCellType::Gas;

        // Set velocity pointing outward in all directions
        grid.vel_x[center] = 5.0; // +x toward solid
        grid.vel_y[center] = 5.0; // +y toward solid
        grid.vel_z[center] = 5.0; // +z toward solid

        enforce_boundary_conditions(&mut grid);

        // All positive velocity components should be zeroed (pointing into solid)
        assert!(
            grid.vel_x[center].abs() < 1e-6,
            "vel_x toward +x solid should be zeroed, got {}",
            grid.vel_x[center]
        );
        assert!(
            grid.vel_y[center].abs() < 1e-6,
            "vel_y toward +y solid should be zeroed, got {}",
            grid.vel_y[center]
        );
        assert!(
            grid.vel_z[center].abs() < 1e-6,
            "vel_z toward +z solid should be zeroed, got {}",
            grid.vel_z[center]
        );
    }

    #[test]
    fn test_edge_no_penetration_negative_directions() {
        // Gas cell surrounded by Solid on all 6 faces, velocity pointing in -x,-y,-z
        let species = vec![make_species("Air")];
        let mut grid = GasGrid::new(3, 3, 3, 1.0, Vec3::ZERO, species);
        for ct in grid.cell_types.iter_mut() {
            *ct = GasCellType::Solid;
        }
        let center = grid.idx(1, 1, 1);
        grid.cell_types[center] = GasCellType::Gas;

        grid.vel_x[center] = -5.0; // -x toward solid
        grid.vel_y[center] = -5.0; // -y toward solid
        grid.vel_z[center] = -5.0; // -z toward solid

        enforce_boundary_conditions(&mut grid);

        assert!(
            grid.vel_x[center].abs() < 1e-6,
            "vel_x toward -x solid should be zeroed, got {}",
            grid.vel_x[center]
        );
        assert!(
            grid.vel_y[center].abs() < 1e-6,
            "vel_y toward -y solid should be zeroed, got {}",
            grid.vel_y[center]
        );
        assert!(
            grid.vel_z[center].abs() < 1e-6,
            "vel_z toward -z solid should be zeroed, got {}",
            grid.vel_z[center]
        );
    }
}
