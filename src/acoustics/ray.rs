use crate::scene::{AcousticMaterial, Triangle};
use glam::Vec3;

pub struct AcousticRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub energy: f32,
    pub bounces: u32,
    pub path: Vec<Vec3>,
}

pub struct RayHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub triangle_index: usize,
}

impl AcousticRay {
    pub fn new(origin: Vec3, direction: Vec3, energy: f32) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            energy,
            bounces: 0,
            path: vec![origin],
        }
    }

    pub fn intersect_triangle(&self, tri: &Triangle) -> Option<f32> {
        // Möller–Trumbore intersection
        let edge1 = tri.vertices[1].position - tri.vertices[0].position;
        let edge2 = tri.vertices[2].position - tri.vertices[0].position;
        let h = self.direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < 1e-7 {
            return None;
        }

        let f = 1.0 / a;
        let s = self.origin - tri.vertices[0].position;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * self.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);

        if t > 1e-5 {
            Some(t)
        } else {
            None
        }
    }

    pub fn reflect(&mut self, hit: &RayHit, material: &AcousticMaterial) {
        let absorption = material.absorption.average();
        self.energy *= 1.0 - absorption;
        self.origin = hit.point + hit.normal * 1e-4;
        self.direction = self.direction - 2.0 * self.direction.dot(hit.normal) * hit.normal;
        self.direction = self.direction.normalize();
        self.bounces += 1;
        self.path.push(hit.point);
    }
}
