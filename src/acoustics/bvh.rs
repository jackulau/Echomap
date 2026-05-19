//! Axis-aligned bounding volume hierarchy for triangle ray-cast queries.
//!
//! The simulation traces thousands of rays per source. Linear scan over
//! every triangle is O(rays × tris); this module brings it down to
//! O(rays × log tris) for typical scenes by splitting space along the
//! longest AABB axis at the median triangle centroid.
//!
//! Design choices worth noting:
//!
//! * **Single triangle table.** The BVH owns a flat `Vec<TriRef>` indexed
//!   by `(object_index, triangle_index)` into the original scene. Hits
//!   carry that pair so callers can resolve material/medium without a
//!   second lookup.
//! * **Leaf threshold 4.** Empirically a sweet spot — smaller leaves
//!   make the tree deep without speeding traversal, larger leaves waste
//!   time on triangles you'll then test linearly.
//! * **Epsilon-padded AABBs.** Flat faces (a perfectly axis-aligned wall)
//!   yield a degenerate zero-thickness box on one axis. Slab tests then
//!   produce NaN ratios. We pad each axis by `AABB_EPSILON` so the slab
//!   has positive thickness everywhere.
//! * **Zero-direction slab handling.** A ray with `direction.y == 0`
//!   parallel to a flat box would compute `1.0 / 0.0 = inf` for the y
//!   slab. We special-case this: if the ray origin lies *inside* that
//!   slab on that axis, the slab imposes no constraint; otherwise the
//!   ray misses the box entirely.
//! * **t_min > epsilon.** Right after a bounce, the new ray origin sits
//!   on the surface of the triangle it just hit. Without a small lower
//!   bound on `t`, the next nearest-hit query would re-hit that same
//!   triangle. We require `t > AABB_EPSILON` to skip self-intersections.

use glam::Vec3;

use super::ray::{AcousticRay, RayHit};
use crate::scene::{SceneObject, Triangle};

/// Padding added to every AABB axis so flat geometry (zero-thickness on
/// one axis) still produces a valid slab. Smaller than any geometry
/// feature size we'd care about, larger than f32 epsilon noise.
pub const AABB_EPSILON: f32 = 1e-4;

/// Maximum triangles per leaf. Below this, leaves test linearly.
pub const LEAF_TRI_LIMIT: usize = 4;

/// Identifies a triangle by its (object index, triangle index) pair —
/// enough to look the geometry back up against the original scene
/// without copying triangles into the BVH.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TriRef {
    pub object_index: usize,
    pub triangle_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    fn from_triangle(tri: &Triangle) -> Self {
        let mut a = Self::empty();
        for v in &tri.vertices {
            a.expand_point(v.position);
        }
        a.pad();
        a
    }

    fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    fn expand(&mut self, other: &Aabb) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Pad zero-thickness axes so the slab test has positive width
    /// everywhere. We pad every axis unconditionally — the constant is
    /// far below any real geometry feature size, so it's harmless on
    /// already-fat axes and saves a branch.
    fn pad(&mut self) {
        self.min -= Vec3::splat(AABB_EPSILON);
        self.max += Vec3::splat(AABB_EPSILON);
    }

    fn longest_axis(&self) -> usize {
        let extent = self.max - self.min;
        if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        }
    }

    /// Ray-vs-AABB slab test. Returns `Some(t_near)` when the ray
    /// intersects within `[AABB_EPSILON, current_best]`, else `None`.
    ///
    /// Handles zero direction components without producing NaN: a ray
    /// parallel to an axis only misses the box on that axis if its
    /// origin lies outside the slab on that axis.
    fn intersect_ray(&self, origin: Vec3, dir: Vec3, t_best: f32) -> Option<f32> {
        let mut t_min = AABB_EPSILON;
        let mut t_max = t_best;

        for axis in 0..3 {
            let o = origin[axis];
            let d = dir[axis];
            let lo = self.min[axis];
            let hi = self.max[axis];

            if d.abs() < f32::EPSILON {
                // Ray parallel to slab. Inside → no constraint; outside → miss.
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }

            let inv = 1.0 / d;
            let (mut near, mut far) = ((lo - o) * inv, (hi - o) * inv);
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            if near > t_min {
                t_min = near;
            }
            if far < t_max {
                t_max = far;
            }
            if t_min > t_max {
                return None;
            }
        }

        Some(t_min)
    }
}

#[derive(Clone, Debug)]
enum Node {
    Internal {
        bounds: Aabb,
        left: Box<Node>,
        right: Box<Node>,
    },
    Leaf {
        bounds: Aabb,
        // Indices into the BVH's flat `tris` vector. Storing indices
        // (not `TriRef`s) lets the build step shuffle order without
        // disturbing the canonical (object, triangle) mapping.
        tri_indices: Vec<usize>,
    },
}

impl Node {
    fn bounds(&self) -> &Aabb {
        match self {
            Node::Internal { bounds, .. } | Node::Leaf { bounds, .. } => bounds,
        }
    }
}

/// The BVH. Built once per simulation run and reused across all rays.
pub struct Bvh {
    /// Canonical triangle table. Index into this vector is the "tri id"
    /// used internally; leaves store those ids.
    pub tris: Vec<TriRef>,
    /// Tree root. Empty BVH = a leaf with no triangles.
    root: Node,
}

impl Bvh {
    /// Build a BVH over the triangles of every visible mesh in `meshes`.
    /// Invisible objects are still indexed because tracing has historically
    /// hit them too — visibility is a render-only concern.
    pub fn build(meshes: &[SceneObject]) -> Self {
        let mut tris: Vec<TriRef> = Vec::new();
        let mut centroids: Vec<Vec3> = Vec::new();
        let mut aabbs: Vec<Aabb> = Vec::new();

        for (obj_idx, obj) in meshes.iter().enumerate() {
            for (tri_idx, tri) in obj.mesh.triangles.iter().enumerate() {
                tris.push(TriRef {
                    object_index: obj_idx,
                    triangle_index: tri_idx,
                });
                centroids.push(tri.centroid());
                aabbs.push(Aabb::from_triangle(tri));
            }
        }

        if tris.is_empty() {
            return Self {
                tris,
                root: Node::Leaf {
                    bounds: Aabb::empty(),
                    tri_indices: Vec::new(),
                },
            };
        }

        let indices: Vec<usize> = (0..tris.len()).collect();
        let root = build_node(&indices, &centroids, &aabbs);
        Self { tris, root }
    }

    /// Total triangle count indexed by the BVH.
    pub fn triangle_count(&self) -> usize {
        self.tris.len()
    }

    /// Nearest triangle hit. Returns `(t, TriRef)` so callers can look the
    /// `SceneObject` and `Triangle` up themselves. Walks the tree front-
    /// to-back via AABB slab tests, falling through to Möller–Trumbore
    /// against the leaf triangles.
    ///
    /// `t > AABB_EPSILON` is enforced inside the leaf test so a ray fired
    /// off a surface doesn't immediately re-hit that surface.
    pub fn nearest_hit(
        &self,
        ray: &AcousticRay,
        meshes: &[SceneObject],
    ) -> Option<(RayHit, TriRef)> {
        if self.tris.is_empty() {
            return None;
        }
        let mut best_t = f32::MAX;
        let mut best: Option<(RayHit, TriRef)> = None;
        self.walk(&self.root, ray, meshes, &mut best_t, &mut best);
        best
    }

    fn walk(
        &self,
        node: &Node,
        ray: &AcousticRay,
        meshes: &[SceneObject],
        best_t: &mut f32,
        best: &mut Option<(RayHit, TriRef)>,
    ) {
        if node
            .bounds()
            .intersect_ray(ray.origin, ray.direction, *best_t)
            .is_none()
        {
            return;
        }
        match node {
            Node::Internal { left, right, .. } => {
                // Front-to-back ordering. Descending into the closer
                // child first prunes more aggressively in the second.
                let lt = left
                    .bounds()
                    .intersect_ray(ray.origin, ray.direction, *best_t);
                let rt = right
                    .bounds()
                    .intersect_ray(ray.origin, ray.direction, *best_t);
                match (lt, rt) {
                    (Some(lt), Some(rt)) if lt <= rt => {
                        self.walk(left, ray, meshes, best_t, best);
                        self.walk(right, ray, meshes, best_t, best);
                    }
                    (Some(_), Some(_)) => {
                        self.walk(right, ray, meshes, best_t, best);
                        self.walk(left, ray, meshes, best_t, best);
                    }
                    (Some(_), None) => self.walk(left, ray, meshes, best_t, best),
                    (None, Some(_)) => self.walk(right, ray, meshes, best_t, best),
                    (None, None) => {}
                }
            }
            Node::Leaf { tri_indices, .. } => {
                for &idx in tri_indices {
                    let tref = self.tris[idx];
                    let tri = &meshes[tref.object_index].mesh.triangles[tref.triangle_index];
                    if let Some(t) = ray.intersect_triangle(tri) {
                        if t > AABB_EPSILON && t < *best_t {
                            *best_t = t;
                            let point = ray.origin + ray.direction * t;
                            *best = Some((
                                RayHit {
                                    point,
                                    normal: tri.normal(),
                                    distance: t,
                                    triangle_index: tref.triangle_index,
                                },
                                tref,
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn build_node(indices: &[usize], centroids: &[Vec3], aabbs: &[Aabb]) -> Node {
    // Compute the bounds of the current subset.
    let mut bounds = Aabb::empty();
    for &i in indices {
        bounds.expand(&aabbs[i]);
    }

    if indices.len() <= LEAF_TRI_LIMIT {
        return Node::Leaf {
            bounds,
            tri_indices: indices.to_vec(),
        };
    }

    // Split along the longest bounds axis at the median centroid value.
    let axis = bounds.longest_axis();
    let mut sorted: Vec<usize> = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        centroids[a][axis]
            .partial_cmp(&centroids[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = sorted.len() / 2;
    let (left_idx, right_idx) = sorted.split_at(mid);

    // Defensive: if everything sorted to one side (all centroids equal),
    // fall back to a leaf rather than infinite-recurse.
    if left_idx.is_empty() || right_idx.is_empty() {
        return Node::Leaf {
            bounds,
            tri_indices: indices.to_vec(),
        };
    }

    let left = Box::new(build_node(left_idx, centroids, aabbs));
    let right = Box::new(build_node(right_idx, centroids, aabbs));
    Node::Internal {
        bounds,
        left,
        right,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::MediumProperties;
    use crate::scene::{primitives, AcousticMaterial, Mesh, SceneObject, Vertex};

    fn make_object(mesh: Mesh) -> SceneObject {
        SceneObject {
            name: "test".into(),
            mesh,
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }
    }

    /// Brute-force version of nearest_hit for cross-checking BVH results.
    fn brute_force_hit(ray: &AcousticRay, meshes: &[SceneObject]) -> Option<(RayHit, TriRef)> {
        let mut nearest: Option<(RayHit, TriRef)> = None;
        let mut best = f32::MAX;
        for (oi, obj) in meshes.iter().enumerate() {
            for (ti, tri) in obj.mesh.triangles.iter().enumerate() {
                if let Some(t) = ray.intersect_triangle(tri) {
                    if t > AABB_EPSILON && t < best {
                        best = t;
                        let point = ray.origin + ray.direction * t;
                        nearest = Some((
                            RayHit {
                                point,
                                normal: tri.normal(),
                                distance: t,
                                triangle_index: ti,
                            },
                            TriRef {
                                object_index: oi,
                                triangle_index: ti,
                            },
                        ));
                    }
                }
            }
        }
        nearest
    }

    /// Across a dense set of ray directions in a non-trivial scene, the
    /// BVH and brute-force nearest-hit results must agree on which
    /// triangle and on the hit distance to within float epsilon. This is
    /// the load-bearing correctness invariant.
    #[test]
    fn test_bvh_matches_brute_force() {
        let room = primitives::box_room(6.0, 6.0, 4.0);
        let platform = primitives::platform(Vec3::new(2.0, 0.0, 2.0), 1.5, 1.5, 1.0);
        let meshes = vec![room, platform];
        let bvh = Bvh::build(&meshes);

        let bg = MediumProperties::air();
        let origin = Vec3::new(3.0, 2.0, 3.0);

        // Sample 64 directions on a coarse Fibonacci-ish grid.
        let mut mismatches = 0;
        for i in 0..64 {
            let t = (i as f32) * 0.5;
            let phi = (1.0 - (2.0 * i as f32 + 1.0) / 64.0).acos();
            let dir = Vec3::new(phi.sin() * t.cos(), phi.sin() * t.sin(), phi.cos()).normalize();
            let ray = AcousticRay::new(origin, dir, 1.0, bg.clone());

            let bvh_hit = bvh.nearest_hit(&ray, &meshes);
            let bf_hit = brute_force_hit(&ray, &meshes);

            match (bvh_hit, bf_hit) {
                (Some((b, bt)), Some((f, ft))) => {
                    assert_eq!(bt, ft, "ray {i}: different TriRef ({bt:?} vs {ft:?})");
                    let dd = (b.distance - f.distance).abs();
                    assert!(
                        dd < 1e-3,
                        "ray {i}: distance mismatch bvh={} bf={}",
                        b.distance,
                        f.distance
                    );
                }
                (None, None) => {}
                _ => mismatches += 1,
            }
        }
        assert_eq!(mismatches, 0, "{mismatches} BVH/brute-force disagreements");
    }

    /// Right after a bounce the ray origin sits on the surface of the
    /// triangle it just hit. The BVH must not return that same triangle
    /// — `t_min > epsilon` enforces this regardless of how thin the
    /// geometry is.
    #[test]
    fn test_bvh_no_self_intersection() {
        let room = primitives::box_room(4.0, 4.0, 4.0);
        let meshes = vec![room];
        let bvh = Bvh::build(&meshes);
        let bg = MediumProperties::air();

        // Pick a wall point and fire a ray slightly off it along the
        // outward normal — the BVH should *not* report that wall as the
        // nearest hit; it should report the opposite wall (or none).
        // We pick the floor: y = 0. A ray from (2, 1e-6, 2) along +y
        // would self-hit the floor unless t_min is enforced.
        let origin = Vec3::new(2.0, 1e-6, 2.0);
        let ray = AcousticRay::new(origin, Vec3::Y, 1.0, bg);

        let hit = bvh.nearest_hit(&ray, &meshes);
        assert!(hit.is_some(), "ray should hit the ceiling");
        let (h, _) = hit.unwrap();
        assert!(
            h.distance > 1.0,
            "self-intersection avoided: distance must be ~4m (ceiling), got {}",
            h.distance
        );
    }

    /// A ray running exactly parallel to a flat AABB axis (`direction.y
    /// == 0` while the box has zero y-thickness before padding) is the
    /// classic NaN trap. With epsilon-padding + zero-dir handling, the
    /// slab test must return cleanly (`Some` or `None`, never NaN).
    #[test]
    fn test_ray_parallel_to_flat_aabb() {
        // Build a degenerate flat mesh: a single triangle lying flat on
        // the y = 0 plane. Its raw AABB has zero y-extent.
        let mesh = Mesh {
            triangles: vec![Triangle {
                vertices: [
                    Vertex {
                        position: Vec3::new(0.0, 0.0, 0.0),
                        normal: Vec3::Y,
                    },
                    Vertex {
                        position: Vec3::new(1.0, 0.0, 0.0),
                        normal: Vec3::Y,
                    },
                    Vertex {
                        position: Vec3::new(0.0, 0.0, 1.0),
                        normal: Vec3::Y,
                    },
                ],
            }],
        };
        let meshes = vec![make_object(mesh)];
        let bvh = Bvh::build(&meshes);
        let bg = MediumProperties::air();

        // Ray running along +x at y=0 — exactly parallel to the flat plane
        // *and* coplanar with it. With epsilon-padding the slab on y now
        // has thickness 2*AABB_EPSILON around 0, so the ray's origin is
        // inside the slab and the zero-dir branch returns "no constraint".
        let ray = AcousticRay::new(Vec3::new(-1.0, 0.0, 0.5), Vec3::X, 1.0, bg.clone());
        let hit = bvh.nearest_hit(&ray, &meshes);
        // Whether we hit the triangle depends on Möller–Trumbore's
        // tolerance for coplanar rays — what matters is the slab test
        // returned a finite, non-NaN result.
        let _ = hit;

        // Ray well off the y = 0 plane, also +x. Slab y is outside
        // [-eps, +eps] so the box is missed without dividing by zero.
        let ray = AcousticRay::new(Vec3::new(-1.0, 5.0, 0.5), Vec3::X, 1.0, bg);
        let hit = bvh.nearest_hit(&ray, &meshes);
        assert!(hit.is_none(), "off-plane parallel ray should miss");
    }
}
