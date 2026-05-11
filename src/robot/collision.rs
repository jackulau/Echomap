use glam::{Mat3, Quat, Vec3};

use crate::scene::SceneObject;

/// Result of a ray intersection test.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RayHit {
    pub distance: f32,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Axis-aligned bounding box.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Aabb {
    pub center: Vec3,
    pub half_extents: Vec3,
}

/// Möller-Trumbore ray-triangle intersection.
///
/// Returns `None` for zero-length direction, degenerate triangles, parallel
/// rays, and rays that miss the triangle. Hits both front and back faces.
#[allow(dead_code)]
pub fn ray_triangle_intersect(
    origin: Vec3,
    direction: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> Option<RayHit> {
    // Guard: zero-length direction
    if direction.length_squared() < f32::EPSILON * f32::EPSILON {
        return None;
    }

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;

    let h = direction.cross(edge2);
    let det = edge1.dot(h);

    // If determinant is near zero the ray is parallel to the triangle (or
    // the triangle is degenerate).
    if det.abs() < f32::EPSILON {
        return None;
    }

    let inv_det = 1.0 / det;
    let s = origin - v0;
    let u = s.dot(h) * inv_det;

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(edge1);
    let v = direction.dot(q) * inv_det;

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = edge2.dot(q) * inv_det;

    // Intersection must be in front of the ray origin.
    if t < f32::EPSILON {
        return None;
    }

    let point = origin + direction * t;
    let normal = edge1.cross(edge2).normalize();

    Some(RayHit {
        distance: t,
        point,
        normal,
    })
}

/// Cast a ray against all scene mesh triangles, returning the nearest hit
/// within `max_distance`.
#[allow(dead_code)]
pub fn ray_scene_cast(
    origin: Vec3,
    direction: Vec3,
    meshes: &[SceneObject],
    max_distance: f32,
) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;

    for obj in meshes {
        for tri in &obj.mesh.triangles {
            let v0 = tri.vertices[0].position;
            let v1 = tri.vertices[1].position;
            let v2 = tri.vertices[2].position;

            if let Some(hit) = ray_triangle_intersect(origin, direction, v0, v1, v2) {
                if hit.distance <= max_distance {
                    let dominated = best.as_ref().is_some_and(|b| b.distance <= hit.distance);
                    if !dominated {
                        best = Some(hit);
                    }
                }
            }
        }
    }

    best
}

/// Axis-aligned overlap test for two AABBs.
///
/// Two AABBs overlap when their projections overlap on all three axes.
/// Touching edges (distance == 0) counts as overlapping.
#[allow(dead_code)]
pub fn aabb_overlap(a: &Aabb, b: &Aabb) -> bool {
    let diff = (a.center - b.center).abs();
    let sum = a.half_extents + b.half_extents;

    diff.x <= sum.x && diff.y <= sum.y && diff.z <= sum.z
}

/// Compute a world-space AABB from a link's world position, rotation, and
/// local half-extents. Rotation expands the AABB to remain axis-aligned.
///
/// The expanded half-extents are computed by taking the absolute values of each
/// column of the 3x3 rotation matrix, scaled by the local half-extents.
#[allow(dead_code)]
pub fn aabb_from_link(link_world_pos: Vec3, link_world_rot: Quat, half_extents: Vec3) -> Aabb {
    let rot = Mat3::from_quat(link_world_rot);

    // Each column of the rotation matrix tells us how the local axes map to
    // world axes. Taking absolute values and dotting with half_extents gives
    // the world-space extent on each world axis.
    let abs_col0 = rot.x_axis.abs();
    let abs_col1 = rot.y_axis.abs();
    let abs_col2 = rot.z_axis.abs();

    let world_half = Vec3::new(
        abs_col0.x * half_extents.x + abs_col1.x * half_extents.y + abs_col2.x * half_extents.z,
        abs_col0.y * half_extents.x + abs_col1.y * half_extents.y + abs_col2.y * half_extents.z,
        abs_col0.z * half_extents.x + abs_col1.z * half_extents.y + abs_col2.z * half_extents.z,
    );

    Aabb {
        center: link_world_pos,
        half_extents: world_half,
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
    use std::f32::consts::FRAC_PI_4;

    const EPSILON: f32 = 1e-5;

    /// Helper: build a triangle from three positions (normals set to zero).
    fn tri(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
        Triangle {
            vertices: [
                Vertex {
                    position: a,
                    normal: Vec3::ZERO,
                },
                Vertex {
                    position: b,
                    normal: Vec3::ZERO,
                },
                Vertex {
                    position: c,
                    normal: Vec3::ZERO,
                },
            ],
        }
    }

    /// Helper: build a SceneObject from triangles.
    fn scene_obj(tris: Vec<Triangle>) -> SceneObject {
        SceneObject {
            name: "test".into(),
            mesh: Mesh { triangles: tris },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }
    }

    // -----------------------------------------------------------------------
    // Ray-triangle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ray_triangle_hit() {
        // Triangle in the XY plane at z=0, winding CCW when viewed from +Z.
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Ray from z=5 shooting toward -Z.
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);

        let hit =
            ray_triangle_intersect(origin, direction, v0, v1, v2).expect("should hit the triangle");

        assert!(
            (hit.distance - 5.0).abs() < EPSILON,
            "distance should be 5, got {}",
            hit.distance
        );
        assert!(
            (hit.point - Vec3::new(0.0, 0.0, 0.0)).length() < EPSILON,
            "point should be at origin, got {:?}",
            hit.point
        );
        // Normal should face +Z (CCW winding from front).
        assert!(
            hit.normal.dot(Vec3::Z).abs() > 0.9,
            "normal should be roughly +Z or -Z, got {:?}",
            hit.normal
        );
    }

    #[test]
    fn test_ray_triangle_miss() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Ray aimed away from the triangle.
        let origin = Vec3::new(10.0, 10.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);

        assert!(
            ray_triangle_intersect(origin, direction, v0, v1, v2).is_none(),
            "ray should miss the triangle"
        );
    }

    #[test]
    fn test_ray_triangle_parallel() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Ray parallel to the triangle plane.
        let origin = Vec3::new(0.0, 0.0, 1.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        assert!(
            ray_triangle_intersect(origin, direction, v0, v1, v2).is_none(),
            "parallel ray should return None"
        );
    }

    #[test]
    fn test_ray_triangle_backface() {
        // Triangle in XY plane, CCW when viewed from +Z.
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        // Ray from -Z shooting toward +Z (hitting backface).
        let origin = Vec3::new(0.0, 0.0, -5.0);
        let direction = Vec3::new(0.0, 0.0, 1.0);

        let hit =
            ray_triangle_intersect(origin, direction, v0, v1, v2).expect("should hit the backface");

        assert!(
            (hit.distance - 5.0).abs() < EPSILON,
            "distance should be 5, got {}",
            hit.distance
        );
    }

    // -----------------------------------------------------------------------
    // AABB tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_aabb_overlap_true() {
        let a = Aabb {
            center: Vec3::ZERO,
            half_extents: Vec3::splat(1.0),
        };
        let b = Aabb {
            center: Vec3::new(1.5, 0.0, 0.0),
            half_extents: Vec3::splat(1.0),
        };
        assert!(aabb_overlap(&a, &b), "AABBs should overlap on X axis");
    }

    #[test]
    fn test_aabb_overlap_false() {
        let a = Aabb {
            center: Vec3::ZERO,
            half_extents: Vec3::splat(1.0),
        };
        let b = Aabb {
            center: Vec3::new(5.0, 0.0, 0.0),
            half_extents: Vec3::splat(1.0),
        };
        assert!(!aabb_overlap(&a, &b), "AABBs should NOT overlap");
    }

    #[test]
    fn test_aabb_from_link_identity() {
        let half = Vec3::new(0.5, 0.25, 0.1);
        let aabb = aabb_from_link(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, half);

        assert!(
            (aabb.center - Vec3::new(1.0, 2.0, 3.0)).length() < EPSILON,
            "center should match link position"
        );
        assert!(
            (aabb.half_extents - half).length() < EPSILON,
            "identity rotation should preserve half_extents, got {:?}",
            aabb.half_extents
        );
    }

    // -----------------------------------------------------------------------
    // Edge-case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ray_zero_direction() {
        let v0 = Vec3::new(-1.0, -1.0, 0.0);
        let v1 = Vec3::new(1.0, -1.0, 0.0);
        let v2 = Vec3::new(0.0, 1.0, 0.0);

        assert!(
            ray_triangle_intersect(Vec3::ZERO, Vec3::ZERO, v0, v1, v2).is_none(),
            "zero-length direction should return None"
        );
    }

    #[test]
    fn test_ray_degenerate_triangle() {
        // Degenerate triangle (three collinear points => zero area).
        let v0 = Vec3::new(0.0, 0.0, 0.0);
        let v1 = Vec3::new(1.0, 0.0, 0.0);
        let v2 = Vec3::new(2.0, 0.0, 0.0);

        let origin = Vec3::new(0.5, 0.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);

        assert!(
            ray_triangle_intersect(origin, direction, v0, v1, v2).is_none(),
            "degenerate (zero-area) triangle should return None"
        );
    }

    #[test]
    fn test_ray_scene_cast_empty() {
        let meshes: Vec<SceneObject> = vec![];
        let result = ray_scene_cast(Vec3::ZERO, Vec3::Z, &meshes, 100.0);
        assert!(result.is_none(), "empty scene should return None");
    }

    #[test]
    fn test_aabb_zero_extents() {
        // A flat (zero-volume) AABB should still work for overlap checks.
        let a = Aabb {
            center: Vec3::ZERO,
            half_extents: Vec3::new(1.0, 0.0, 1.0), // flat on Y
        };
        let b = Aabb {
            center: Vec3::new(0.5, 0.0, 0.5),
            half_extents: Vec3::new(0.1, 0.0, 0.1),
        };
        // They overlap on X and Z, touching on Y.
        assert!(
            aabb_overlap(&a, &b),
            "flat AABBs touching on Y should count as overlapping"
        );

        // Now with 45-degree rotation the AABB expands.
        let aabb = aabb_from_link(
            Vec3::ZERO,
            Quat::from_rotation_y(FRAC_PI_4),
            Vec3::new(1.0, 0.0, 1.0),
        );
        // After 45-degree Y rotation of a box with half_extents (1,0,1),
        // the world AABB half_extents on X and Z should both be sqrt(2).
        let expected_xz = (2.0_f32).sqrt();
        assert!(
            (aabb.half_extents.x - expected_xz).abs() < 0.01,
            "rotated half_extent X should be ~{}, got {}",
            expected_xz,
            aabb.half_extents.x
        );
        assert!(
            (aabb.half_extents.z - expected_xz).abs() < 0.01,
            "rotated half_extent Z should be ~{}, got {}",
            expected_xz,
            aabb.half_extents.z
        );
    }
}
