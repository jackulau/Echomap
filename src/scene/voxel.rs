use glam::Vec3;

use super::SceneObject;

/// Shared scene voxelization: for each solid scene mesh, mark every grid cell
/// whose AABB overlaps the mesh's bounding box.
///
/// This is the geometry-only core shared by `fluids::boundary::voxelize_scene`
/// and `gas::boundary::voxelize_scene`. The caller supplies the grid geometry
/// (`origin`, `dx`, `nx`/`ny`/`nz`) and a `mark_solid` callback that receives
/// the flat cell index (`i + nx * (j + ny * k)`) of each overlapping cell. The
/// callback writes the module-specific `Solid` cell type into its own grid.
///
/// The flat-index convention matches `FluidGrid::idx` and `GasGrid::idx`
/// exactly, so wrapping callers observe bit-identical cell selection.
pub fn voxelize_meshes(
    origin: Vec3,
    dx: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    meshes: &[SceneObject],
    mut mark_solid: impl FnMut(usize),
) {
    for obj in meshes {
        if obj.mesh.triangles.is_empty() {
            continue;
        }
        let (mesh_min, mesh_max) = obj.mesh.bounds();

        // Determine the range of grid cells that could overlap the mesh AABB.
        let rel_min = mesh_min - origin;
        let rel_max = mesh_max - origin;

        let i_start = ((rel_min.x / dx).floor() as i32).max(0) as usize;
        let j_start = ((rel_min.y / dx).floor() as i32).max(0) as usize;
        let k_start = ((rel_min.z / dx).floor() as i32).max(0) as usize;

        let i_end = ((rel_max.x / dx).ceil() as usize).min(nx);
        let j_end = ((rel_max.y / dx).ceil() as usize).min(ny);
        let k_end = ((rel_max.z / dx).ceil() as usize).min(nz);

        for k in k_start..k_end {
            for j in j_start..j_end {
                for i in i_start..i_end {
                    // Cell AABB
                    let cell_min = origin + Vec3::new(i as f32 * dx, j as f32 * dx, k as f32 * dx);
                    let cell_max = cell_min + Vec3::splat(dx);

                    // AABB-AABB overlap test
                    if cell_min.x < mesh_max.x
                        && cell_max.x > mesh_min.x
                        && cell_min.y < mesh_max.y
                        && cell_max.y > mesh_min.y
                        && cell_min.z < mesh_max.z
                        && cell_max.z > mesh_min.z
                    {
                        let idx = i + nx * (j + ny * k);
                        mark_solid(idx);
                    }
                }
            }
        }
    }
}
