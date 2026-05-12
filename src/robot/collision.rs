use glam::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::definition::{BodyZone, CollisionShape, RobotDefinition};
use super::state::RobotState;
use crate::scene::SceneObject;

/// A combat hit event capturing the full details of a strike between two robots.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HitEvent {
    pub attacker_robot: usize,
    pub target_robot: usize,
    pub attacker_link: usize,
    pub target_link: usize,
    pub zone: BodyZone,
    pub impact_force: f32,
    pub damage: f32,
    pub contact_point: Vec3,
    pub contact_normal: Vec3,
}

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
// Robot-robot collision detection
// ---------------------------------------------------------------------------

/// Convert a CollisionShape to AABB half-extents.
#[allow(dead_code)]
pub fn collision_shape_to_half_extents(shape: &CollisionShape) -> Vec3 {
    match shape {
        CollisionShape::Sphere { radius } => Vec3::splat(*radius),
        CollisionShape::Cuboid { half_extents } => *half_extents,
        CollisionShape::Cylinder { radius, height } => Vec3::new(*radius, *height / 2.0, *radius),
    }
}

/// A collision contact between two links on different robots.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LinkCollision {
    pub robot_a: usize,
    pub link_a: usize,
    pub robot_b: usize,
    pub link_b: usize,
    pub contact_point: Vec3,
    pub contact_normal: Vec3,
    pub penetration: f32,
}

/// Collect world-space AABBs for every link in a robot.
///
/// Returns a Vec of `(link_index, Aabb)` pairs.
#[allow(dead_code)]
pub fn collect_link_aabbs(definition: &RobotDefinition, state: &RobotState) -> Vec<(usize, Aabb)> {
    definition
        .links
        .iter()
        .enumerate()
        .map(|(i, link)| {
            let mat = Mat4::from_cols_array(&state.link_poses[i]);
            let pos = mat.w_axis.truncate();
            let rot = Quat::from_mat4(&mat);
            let half_extents = collision_shape_to_half_extents(&link.collision_shape);
            let aabb = aabb_from_link(pos, rot, half_extents);
            (i, aabb)
        })
        .collect()
}

/// Detect collisions between all pairs of distinct robots.
///
/// For each pair of robots (i < j), checks every link-link pair for AABB
/// overlap and produces a `LinkCollision` for each overlapping pair.
#[allow(dead_code)]
pub fn detect_robot_collisions(
    robots: &[(usize, &RobotDefinition, &RobotState)],
) -> Vec<LinkCollision> {
    let mut collisions = Vec::new();

    for i in 0..robots.len() {
        let (id_a, def_a, state_a) = &robots[i];
        let aabbs_a = collect_link_aabbs(def_a, state_a);

        for j in (i + 1)..robots.len() {
            let (id_b, def_b, state_b) = &robots[j];
            let aabbs_b = collect_link_aabbs(def_b, state_b);

            for &(link_a_idx, ref aabb_a) in &aabbs_a {
                for &(link_b_idx, ref aabb_b) in &aabbs_b {
                    if aabb_overlap(aabb_a, aabb_b) {
                        let contact_point = (aabb_a.center + aabb_b.center) * 0.5;

                        let diff = aabb_b.center - aabb_a.center;
                        let contact_normal = if diff.length_squared() < f32::EPSILON {
                            Vec3::X
                        } else {
                            diff.normalize()
                        };

                        // Penetration: min positive overlap across all axes
                        let mut penetration = f32::MAX;
                        for axis in 0..3 {
                            let overlap = (aabb_a.half_extents[axis] + aabb_b.half_extents[axis])
                                - (aabb_a.center[axis] - aabb_b.center[axis]).abs();
                            if overlap < penetration {
                                penetration = overlap;
                            }
                        }

                        collisions.push(LinkCollision {
                            robot_a: *id_a,
                            link_a: link_a_idx,
                            robot_b: *id_b,
                            link_b: link_b_idx,
                            contact_point,
                            contact_normal,
                            penetration,
                        });
                    }
                }
            }
        }
    }

    collisions
}

// ---------------------------------------------------------------------------
// Punch detection
// ---------------------------------------------------------------------------

/// Minimum link velocity magnitude (m/s) to count as a punch.
pub const PUNCH_VELOCITY_THRESHOLD: f32 = 2.0;

/// Stamina cost per punch.
pub const PUNCH_STAMINA_COST: f32 = 10.0;

/// Detect punches from a set of link-link collisions by checking link velocities.
///
/// Each element of `robots` is `(robot_id, definition, state, link_velocities)`.
/// For every collision where the attacker link's velocity exceeds
/// [`PUNCH_VELOCITY_THRESHOLD`], a [`HitEvent`] is emitted provided the target
/// link has a `body_zone` assigned (links without a zone produce no damage).
#[allow(dead_code)]
pub fn detect_punches(
    collisions: &[LinkCollision],
    robots: &[(usize, &RobotDefinition, &RobotState, &[Vec3])],
) -> Vec<HitEvent> {
    let mut hits = Vec::new();

    for collision in collisions {
        // Find robot entries matching collision ids
        let robot_a = robots.iter().find(|(id, ..)| *id == collision.robot_a);
        let robot_b = robots.iter().find(|(id, ..)| *id == collision.robot_b);

        let (robot_a, robot_b) = match (robot_a, robot_b) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };

        // Check if robot_a's link is punching robot_b
        if let Some(vel_a) = robot_a.3.get(collision.link_a) {
            let speed_a = vel_a.length();
            if speed_a > PUNCH_VELOCITY_THRESHOLD {
                // Target is robot_b's link
                if let Some(zone) = &robot_b.1.links[collision.link_b].body_zone {
                    let mass_a = robot_a.1.links[collision.link_a].mass;
                    let impact_force = mass_a * speed_a;
                    let damage = impact_force * zone.damage_multiplier();
                    hits.push(HitEvent {
                        attacker_robot: collision.robot_a,
                        target_robot: collision.robot_b,
                        attacker_link: collision.link_a,
                        target_link: collision.link_b,
                        zone: zone.clone(),
                        impact_force,
                        damage,
                        contact_point: collision.contact_point,
                        contact_normal: collision.contact_normal,
                    });
                }
            }
        }

        // Check if robot_b's link is punching robot_a
        if let Some(vel_b) = robot_b.3.get(collision.link_b) {
            let speed_b = vel_b.length();
            if speed_b > PUNCH_VELOCITY_THRESHOLD {
                // Target is robot_a's link
                if let Some(zone) = &robot_a.1.links[collision.link_a].body_zone {
                    let mass_b = robot_b.1.links[collision.link_b].mass;
                    let impact_force = mass_b * speed_b;
                    let damage = impact_force * zone.damage_multiplier();
                    hits.push(HitEvent {
                        attacker_robot: collision.robot_b,
                        target_robot: collision.robot_a,
                        attacker_link: collision.link_b,
                        target_link: collision.link_a,
                        zone: zone.clone(),
                        impact_force,
                        damage,
                        contact_point: collision.contact_point,
                        contact_normal: collision.contact_normal,
                    });
                }
            }
        }
    }

    hits
}

// ---------------------------------------------------------------------------
// BVH — Bounding Volume Hierarchy for accelerated ray-casting
// ---------------------------------------------------------------------------

struct StoredTriangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
}

enum BvhNode {
    Leaf {
        aabb: Aabb,
        triangles: Vec<StoredTriangle>,
    },
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

/// Acceleration structure for ray-casting against static scene geometry.
pub struct SceneBvh {
    root: Option<BvhNode>,
}

impl SceneBvh {
    /// Build a BVH from scene mesh triangles.
    pub fn build(meshes: &[SceneObject]) -> Self {
        let mut tris: Vec<StoredTriangle> = Vec::new();
        for obj in meshes {
            for tri in &obj.mesh.triangles {
                tris.push(StoredTriangle {
                    v0: tri.vertices[0].position,
                    v1: tri.vertices[1].position,
                    v2: tri.vertices[2].position,
                });
            }
        }

        if tris.is_empty() {
            return Self { root: None };
        }

        let root = Self::build_recursive(tris);
        Self { root: Some(root) }
    }

    fn build_recursive(mut tris: Vec<StoredTriangle>) -> BvhNode {
        let aabb = Self::compute_aabb(&tris);

        if tris.len() <= 4 {
            return BvhNode::Leaf {
                aabb,
                triangles: tris,
            };
        }

        let extent = aabb.half_extents;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        tris.sort_by(|a, b| {
            let ca = (a.v0[axis] + a.v1[axis] + a.v2[axis]) / 3.0;
            let cb = (b.v0[axis] + b.v1[axis] + b.v2[axis]) / 3.0;
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = tris.len() / 2;
        let right_tris = tris.split_off(mid);

        BvhNode::Internal {
            aabb,
            left: Box::new(Self::build_recursive(tris)),
            right: Box::new(Self::build_recursive(right_tris)),
        }
    }

    fn compute_aabb(tris: &[StoredTriangle]) -> Aabb {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for tri in tris {
            for v in [tri.v0, tri.v1, tri.v2] {
                min = min.min(v);
                max = max.max(v);
            }
        }
        let center = (min + max) * 0.5;
        let half_extents = (max - min) * 0.5;
        Aabb {
            center,
            half_extents,
        }
    }

    /// Cast a ray against the BVH, returning the nearest hit within `max_distance`.
    pub fn ray_cast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RayHit> {
        match &self.root {
            None => None,
            Some(node) => Self::ray_cast_node(node, origin, direction, max_distance),
        }
    }

    fn ray_cast_node(
        node: &BvhNode,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Option<RayHit> {
        match node {
            BvhNode::Leaf { aabb, triangles } => {
                if !Self::ray_aabb_test(origin, direction, aabb, max_distance) {
                    return None;
                }
                let mut best: Option<RayHit> = None;
                for tri in triangles {
                    if let Some(hit) =
                        ray_triangle_intersect(origin, direction, tri.v0, tri.v1, tri.v2)
                    {
                        let limit = best.as_ref().map_or(max_distance, |b| b.distance);
                        if hit.distance <= limit {
                            best = Some(hit);
                        }
                    }
                }
                best
            }
            BvhNode::Internal { aabb, left, right } => {
                if !Self::ray_aabb_test(origin, direction, aabb, max_distance) {
                    return None;
                }
                let hit_left = Self::ray_cast_node(left, origin, direction, max_distance);
                let limit = hit_left.as_ref().map_or(max_distance, |h| h.distance);
                let hit_right = Self::ray_cast_node(right, origin, direction, limit);
                match (hit_left, hit_right) {
                    (None, r) => r,
                    (l, None) => l,
                    (Some(l), Some(r)) => {
                        if l.distance <= r.distance {
                            Some(l)
                        } else {
                            Some(r)
                        }
                    }
                }
            }
        }
    }

    fn ray_aabb_test(origin: Vec3, direction: Vec3, aabb: &Aabb, max_distance: f32) -> bool {
        let min = aabb.center - aabb.half_extents;
        let max = aabb.center + aabb.half_extents;

        let mut tmin = 0.0_f32;
        let mut tmax = max_distance;

        for i in 0..3 {
            let d = direction[i];
            let o = origin[i];
            if d.abs() < f32::EPSILON {
                if o < min[i] || o > max[i] {
                    return false;
                }
            } else {
                let inv_d = 1.0 / d;
                let mut t0 = (min[i] - o) * inv_d;
                let mut t1 = (max[i] - o) * inv_d;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                tmin = tmin.max(t0);
                tmax = tmax.min(t1);
                if tmin > tmax {
                    return false;
                }
            }
        }
        true
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

    // -----------------------------------------------------------------------
    // BVH tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bvh_build_empty_scene() {
        let meshes: Vec<SceneObject> = vec![];
        let bvh = SceneBvh::build(&meshes);
        let result = bvh.ray_cast(Vec3::ZERO, Vec3::Z, 100.0);
        assert!(result.is_none(), "empty BVH should return None");
    }

    #[test]
    fn test_bvh_build_single_triangle() {
        let t = tri(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let meshes = vec![scene_obj(vec![t])];
        let bvh = SceneBvh::build(&meshes);

        let brute = ray_scene_cast(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, &meshes, 100.0);
        let accel = bvh.ray_cast(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, 100.0);

        assert!(brute.is_some() && accel.is_some(), "both should hit");
        assert!(
            (brute.unwrap().distance - accel.unwrap().distance).abs() < EPSILON,
            "BVH distance should match brute-force"
        );
    }

    #[test]
    fn test_bvh_cast_matches_brute_force() {
        let tris = vec![
            tri(
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.5, 1.0, 0.0),
            ),
            tri(
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
                Vec3::new(2.5, 1.0, 0.0),
            ),
            tri(
                Vec3::new(-2.0, -2.0, -1.0),
                Vec3::new(-1.0, -2.0, -1.0),
                Vec3::new(-1.5, -1.0, -1.0),
            ),
            tri(
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::new(1.0, 0.0, 3.0),
                Vec3::new(0.5, 1.0, 3.0),
            ),
            tri(
                Vec3::new(-1.0, -1.0, 5.0),
                Vec3::new(1.0, -1.0, 5.0),
                Vec3::new(0.0, 1.0, 5.0),
            ),
        ];
        let meshes = vec![scene_obj(tris)];
        let bvh = SceneBvh::build(&meshes);

        let rays = [
            (Vec3::new(0.5, 0.3, 10.0), Vec3::NEG_Z),
            (Vec3::new(2.5, 0.3, 10.0), Vec3::NEG_Z),
            (Vec3::new(-1.5, -1.5, 10.0), Vec3::NEG_Z),
            (Vec3::new(10.0, 10.0, 10.0), Vec3::NEG_Z),
            (Vec3::new(0.5, 0.3, -10.0), Vec3::Z),
        ];

        for (origin, dir) in &rays {
            let brute = ray_scene_cast(*origin, *dir, &meshes, 100.0);
            let accel = bvh.ray_cast(*origin, *dir, 100.0);
            match (&brute, &accel) {
                (None, None) => {}
                (Some(b), Some(a)) => {
                    assert!(
                        (b.distance - a.distance).abs() < EPSILON,
                        "BVH mismatch at {:?}: brute={} accel={}",
                        origin,
                        b.distance,
                        a.distance
                    );
                }
                _ => panic!(
                    "BVH/brute mismatch at {:?}: brute={:?} accel={:?}",
                    origin,
                    brute.as_ref().map(|h| h.distance),
                    accel.as_ref().map(|h| h.distance)
                ),
            }
        }
    }

    #[test]
    fn test_bvh_cast_miss() {
        let t = tri(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let meshes = vec![scene_obj(vec![t])];
        let bvh = SceneBvh::build(&meshes);

        let result = bvh.ray_cast(Vec3::new(10.0, 10.0, 5.0), Vec3::NEG_Z, 100.0);
        assert!(result.is_none(), "ray should miss");
    }

    #[test]
    fn test_bvh_cast_nearest_hit() {
        let tris = vec![
            tri(
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ),
            tri(
                Vec3::new(-1.0, -1.0, 3.0),
                Vec3::new(1.0, -1.0, 3.0),
                Vec3::new(0.0, 1.0, 3.0),
            ),
        ];
        let meshes = vec![scene_obj(tris)];
        let bvh = SceneBvh::build(&meshes);

        let hit = bvh
            .ray_cast(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, 100.0)
            .unwrap();
        assert!(
            (hit.distance - 2.0).abs() < EPSILON,
            "should hit nearest triangle at z=3, distance=2, got {}",
            hit.distance
        );
    }

    #[test]
    fn test_bvh_max_distance() {
        let t = tri(
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let meshes = vec![scene_obj(vec![t])];
        let bvh = SceneBvh::build(&meshes);

        let result = bvh.ray_cast(Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, 1.0);
        assert!(
            result.is_none(),
            "hit at distance 5 should be beyond max_distance 1"
        );
    }

    // ---- Task 2: HitEvent serialization test ----

    #[test]
    fn test_hit_event_serialization() {
        use crate::robot::definition::BodyZone;

        let hit = HitEvent {
            attacker_robot: 0,
            target_robot: 1,
            attacker_link: 2,
            target_link: 3,
            zone: BodyZone::Head,
            impact_force: 15.0,
            damage: 45.0,
            contact_point: Vec3::new(1.0, 2.0, 3.0),
            contact_normal: Vec3::new(0.0, 1.0, 0.0),
        };

        let json = serde_json::to_string(&hit).expect("serialize HitEvent");
        let deser: HitEvent = serde_json::from_str(&json).expect("deserialize HitEvent");

        assert_eq!(deser.attacker_robot, 0);
        assert_eq!(deser.target_robot, 1);
        assert_eq!(deser.attacker_link, 2);
        assert_eq!(deser.target_link, 3);
        assert_eq!(deser.zone, BodyZone::Head);
        assert!((deser.impact_force - 15.0).abs() < 1e-6);
        assert!((deser.damage - 45.0).abs() < 1e-6);
        assert!((deser.contact_point - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
        assert!((deser.contact_normal - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Task 3: Robot-robot collision detection tests
    // -----------------------------------------------------------------------

    use crate::robot::definition::{CollisionShape, LinkDefinition, RobotDefinition};
    use crate::robot::state::RobotState;
    use glam::Mat4;

    fn test_robot_def(num_links: usize, shape: CollisionShape) -> RobotDefinition {
        let links = (0..num_links)
            .map(|i| LinkDefinition {
                name: format!("link_{}", i),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: shape.clone(),
                parent_joint: if i == 0 { None } else { Some(i - 1) },
                body_zone: None,
            })
            .collect();
        RobotDefinition {
            name: "test_robot".to_string(),
            links,
            joints: vec![],
            sensors: vec![],
        }
    }

    fn test_robot_state(_num_links: usize, link_poses: Vec<Mat4>) -> RobotState {
        RobotState {
            joint_positions: vec![],
            joint_velocities: vec![],
            link_poses: link_poses.iter().map(|m| m.to_cols_array()).collect(),
            prev_link_poses: vec![],
            sensor_readings: vec![],
            actuator_commands: vec![],
            timestamp: 0.0,
            combat: None,
        }
    }

    #[test]
    fn test_collision_shape_to_half_extents() {
        let sphere = CollisionShape::Sphere { radius: 0.5 };
        assert!((collision_shape_to_half_extents(&sphere) - Vec3::splat(0.5)).length() < EPSILON);

        let cuboid = CollisionShape::Cuboid {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        assert!(
            (collision_shape_to_half_extents(&cuboid) - Vec3::new(1.0, 2.0, 3.0)).length()
                < EPSILON
        );

        let cylinder = CollisionShape::Cylinder {
            radius: 0.3,
            height: 1.0,
        };
        assert!(
            (collision_shape_to_half_extents(&cylinder) - Vec3::new(0.3, 0.5, 0.3)).length()
                < EPSILON
        );
    }

    #[test]
    fn test_collect_link_aabbs_simple() {
        let def = test_robot_def(
            1,
            CollisionShape::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
        );
        let state = test_robot_state(1, vec![Mat4::IDENTITY]);
        let aabbs = collect_link_aabbs(&def, &state);

        assert_eq!(aabbs.len(), 1);
        assert_eq!(aabbs[0].0, 0);
        assert!(
            aabbs[0].1.center.length() < EPSILON,
            "center should be at origin"
        );
        assert!(
            (aabbs[0].1.half_extents - Vec3::splat(0.5)).length() < EPSILON,
            "half_extents should be (0.5, 0.5, 0.5)"
        );
    }

    #[test]
    fn test_collect_link_aabbs_rotated() {
        let def = test_robot_def(
            1,
            CollisionShape::Cuboid {
                half_extents: Vec3::new(1.0, 0.5, 0.1),
            },
        );
        let rot_mat = Mat4::from_rotation_y(FRAC_PI_4);
        let state = test_robot_state(1, vec![rot_mat]);
        let aabbs = collect_link_aabbs(&def, &state);

        assert_eq!(aabbs.len(), 1);
        // After 45-degree rotation, the AABB should expand on X and Z
        let identity_def = test_robot_def(
            1,
            CollisionShape::Cuboid {
                half_extents: Vec3::new(1.0, 0.5, 0.1),
            },
        );
        let identity_state = test_robot_state(1, vec![Mat4::IDENTITY]);
        let identity_aabbs = collect_link_aabbs(&identity_def, &identity_state);

        assert!(
            aabbs[0].1.half_extents.x > identity_aabbs[0].1.half_extents.x
                || aabbs[0].1.half_extents.z > identity_aabbs[0].1.half_extents.z,
            "rotated AABB should expand on at least one axis"
        );
    }

    #[test]
    fn test_detect_no_collision() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let def_a = test_robot_def(1, shape.clone());
        let def_b = test_robot_def(1, shape);

        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0))]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> =
            vec![(0, &def_a, &state_a), (1, &def_b, &state_b)];
        let collisions = detect_robot_collisions(&robots);

        assert!(collisions.is_empty(), "far-apart robots should not collide");
    }

    #[test]
    fn test_detect_overlapping_robots() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let def_a = test_robot_def(1, shape.clone());
        let def_b = test_robot_def(1, shape);

        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> =
            vec![(0, &def_a, &state_a), (1, &def_b, &state_b)];
        let collisions = detect_robot_collisions(&robots);

        assert!(
            !collisions.is_empty(),
            "overlapping robots should produce collisions"
        );
        assert_eq!(collisions[0].robot_a, 0);
        assert_eq!(collisions[0].robot_b, 1);
    }

    #[test]
    fn test_same_robot_links_skipped() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let def = test_robot_def(3, shape);
        let state = test_robot_state(3, vec![Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> = vec![(0, &def, &state)];
        let collisions = detect_robot_collisions(&robots);

        assert!(
            collisions.is_empty(),
            "single robot should have no collisions (no self-collision)"
        );
    }

    #[test]
    fn test_contact_normal_direction() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(1.0),
        };
        let def_a = test_robot_def(1, shape.clone());
        let def_b = test_robot_def(1, shape);

        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> =
            vec![(0, &def_a, &state_a), (1, &def_b, &state_b)];
        let collisions = detect_robot_collisions(&robots);

        assert!(!collisions.is_empty(), "should detect collision");
        // Normal should point from A toward B, i.e. in +X direction
        assert!(
            collisions[0].contact_normal.x > 0.9,
            "normal should point in +X, got {:?}",
            collisions[0].contact_normal
        );
    }

    #[test]
    fn test_penetration_depth_positive() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(1.0),
        };
        let def_a = test_robot_def(1, shape.clone());
        let def_b = test_robot_def(1, shape);

        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0))]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> =
            vec![(0, &def_a, &state_a), (1, &def_b, &state_b)];
        let collisions = detect_robot_collisions(&robots);

        assert!(!collisions.is_empty(), "should detect collision");
        assert!(
            collisions[0].penetration > 0.0,
            "penetration should be positive, got {}",
            collisions[0].penetration
        );
        // With half_extents 1.0 each, centers 0.5 apart, overlap on X = 1.0+1.0-0.5 = 1.5
        // overlap on Y = 1.0+1.0-0.0 = 2.0, overlap on Z = 2.0
        // min = 1.5
        assert!(
            (collisions[0].penetration - 1.5).abs() < EPSILON,
            "penetration should be 1.5, got {}",
            collisions[0].penetration
        );
    }

    #[test]
    fn test_multiple_link_collisions() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let def_a = test_robot_def(2, shape.clone());
        let def_b = test_robot_def(2, shape);

        // Both robots have 2 links at the same position => all 4 pairs overlap
        let state_a = test_robot_state(2, vec![Mat4::IDENTITY, Mat4::IDENTITY]);
        let state_b = test_robot_state(2, vec![Mat4::IDENTITY, Mat4::IDENTITY]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> =
            vec![(0, &def_a, &state_a), (1, &def_b, &state_b)];
        let collisions = detect_robot_collisions(&robots);

        assert_eq!(
            collisions.len(),
            4,
            "2 links x 2 links = 4 collisions, got {}",
            collisions.len()
        );
    }

    #[test]
    fn test_empty_robots_no_panic() {
        let robots: Vec<(usize, &RobotDefinition, &RobotState)> = vec![];
        let collisions = detect_robot_collisions(&robots);
        assert!(
            collisions.is_empty(),
            "empty input should produce empty output"
        );
    }

    #[test]
    fn test_single_robot_no_collisions() {
        let shape = CollisionShape::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let def = test_robot_def(1, shape);
        let state = test_robot_state(1, vec![Mat4::IDENTITY]);

        let robots: Vec<(usize, &RobotDefinition, &RobotState)> = vec![(0, &def, &state)];
        let collisions = detect_robot_collisions(&robots);

        assert!(
            collisions.is_empty(),
            "single robot should produce no collisions"
        );
    }

    // -----------------------------------------------------------------------
    // Task 4: Punch detection tests
    // -----------------------------------------------------------------------

    use crate::robot::definition::BodyZone;

    /// Helper: build a robot definition with body zones assigned to links.
    fn punch_robot_def(
        num_links: usize,
        mass: f32,
        zones: Vec<Option<BodyZone>>,
    ) -> RobotDefinition {
        let links = (0..num_links)
            .map(|i| LinkDefinition {
                name: format!("link_{}", i),
                mass,
                inertia: 0.1,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: Vec3::splat(0.5),
                },
                parent_joint: if i == 0 { None } else { Some(i - 1) },
                body_zone: zones.get(i).cloned().flatten(),
            })
            .collect();
        RobotDefinition {
            name: "punch_robot".to_string(),
            links,
            joints: vec![],
            sensors: vec![],
        }
    }

    #[test]
    fn test_punch_detected_high_velocity() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::Body)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        // Robot A's link moving at 3.0 m/s (above threshold of 2.0)
        let vels_a = vec![Vec3::new(3.0, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert_eq!(hits.len(), 1, "should detect one punch");
        assert_eq!(hits[0].attacker_robot, 0);
        assert_eq!(hits[0].target_robot, 1);
    }

    #[test]
    fn test_no_punch_low_velocity() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::Body)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        // Robot A's link moving at 1.5 m/s (below threshold of 2.0)
        let vels_a = vec![Vec3::new(1.5, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert!(hits.is_empty(), "low velocity should not produce a punch");
    }

    #[test]
    fn test_punch_damage_head_zone() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::Head)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        let speed = 5.0_f32;
        let vels_a = vec![Vec3::new(speed, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert_eq!(hits.len(), 1);
        let expected_force = 2.0 * speed; // mass * speed
        let expected_damage = expected_force * 3.0; // Head multiplier = 3.0
        assert!(
            (hits[0].impact_force - expected_force).abs() < 1e-6,
            "impact_force should be {}, got {}",
            expected_force,
            hits[0].impact_force
        );
        assert!(
            (hits[0].damage - expected_damage).abs() < 1e-6,
            "damage should be {} (head zone 3x), got {}",
            expected_damage,
            hits[0].damage
        );
    }

    #[test]
    fn test_punch_damage_body_zone() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::Body)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        let speed = 5.0_f32;
        let vels_a = vec![Vec3::new(speed, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert_eq!(hits.len(), 1);
        let expected_force = 2.0 * speed;
        let expected_damage = expected_force * 1.0; // Body multiplier = 1.0
        assert!(
            (hits[0].damage - expected_damage).abs() < 1e-6,
            "damage should be {} (body zone 1x), got {}",
            expected_damage,
            hits[0].damage
        );
    }

    #[test]
    fn test_punch_damage_arm_zone() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::LeftArm)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        let speed = 5.0_f32;
        let vels_a = vec![Vec3::new(speed, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert_eq!(hits.len(), 1);
        let expected_force = 2.0 * speed;
        let expected_damage = expected_force * 0.5; // LeftArm multiplier = 0.5
        assert!(
            (hits[0].damage - expected_damage).abs() < 1e-6,
            "damage should be {} (arm zone 0.5x), got {}",
            expected_damage,
            hits[0].damage
        );
    }

    #[test]
    fn test_punch_no_zone_no_damage() {
        let def_a = punch_robot_def(1, 2.0, vec![None]);
        let def_b = punch_robot_def(1, 2.0, vec![None]); // No body zone on target
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        let vels_a = vec![Vec3::new(5.0, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert!(
            hits.is_empty(),
            "no body zone on target should produce no HitEvent"
        );
    }

    #[test]
    fn test_zero_mass_link_no_panic() {
        let def_a = punch_robot_def(1, 0.0, vec![None]); // Zero mass attacker
        let def_b = punch_robot_def(1, 2.0, vec![Some(BodyZone::Body)]);
        let state_a = test_robot_state(1, vec![Mat4::IDENTITY]);
        let state_b = test_robot_state(1, vec![Mat4::IDENTITY]);

        let collision = LinkCollision {
            robot_a: 0,
            link_a: 0,
            robot_b: 1,
            link_b: 0,
            contact_point: Vec3::ZERO,
            contact_normal: Vec3::X,
            penetration: 0.1,
        };

        let vels_a = vec![Vec3::new(5.0, 0.0, 0.0)];
        let vels_b = vec![Vec3::ZERO];

        let robots: Vec<(usize, &RobotDefinition, &RobotState, &[Vec3])> = vec![
            (0, &def_a, &state_a, &vels_a),
            (1, &def_b, &state_b, &vels_b),
        ];

        let hits = detect_punches(&[collision], &robots);
        assert_eq!(hits.len(), 1, "zero mass should still produce a HitEvent");
        assert!(
            (hits[0].impact_force - 0.0).abs() < 1e-6,
            "impact_force should be 0 for zero mass"
        );
        assert!(
            (hits[0].damage - 0.0).abs() < 1e-6,
            "damage should be 0 for zero mass"
        );
        // Verify no NaN or Inf
        assert!(
            hits[0].damage.is_finite(),
            "damage should be finite, not NaN/Inf"
        );
    }
}
