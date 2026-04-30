use glam::Vec3;
use rayon::prelude::*;
use std::f32::consts::PI;

use super::ray::{AcousticRay, RayHit};
use crate::scene::{AcousticMaterial, Scene};

#[derive(Clone)]
pub struct SimulationConfig {
    pub ray_count: u32,
    pub max_bounces: u32,
    pub energy_threshold: f32,
    pub grid_resolution: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            ray_count: 10_000,
            max_bounces: 50,
            energy_threshold: 0.001,
            grid_resolution: 0.25,
        }
    }
}

#[derive(Clone, Default)]
pub struct SimulationResult {
    pub energy_grid: Vec<GridPoint>,
    pub ray_paths: Vec<Vec<Vec3>>,
    pub max_energy: f32,
}

#[derive(Clone)]
pub struct GridPoint {
    pub position: Vec3,
    pub energy: f32,
}

#[derive(Default)]
pub struct SimulationState {
    pub config: SimulationConfig,
    pub result: Option<SimulationResult>,
    pub running: bool,
    pub progress: f32,
}

impl SimulationState {
    pub fn run(&mut self, scene: &Scene) {
        if scene.sound_sources.is_empty() || scene.meshes.is_empty() {
            return;
        }

        self.running = true;
        self.progress = 0.0;

        let mut result = SimulationResult::default();

        for source in &scene.sound_sources {
            if !source.enabled {
                continue;
            }

            let rays = generate_sphere_rays(source.position, self.config.ray_count);

            let paths: Vec<Vec<Vec3>> = rays
                .into_par_iter()
                .map(|dir| trace_ray(source.position, dir, source.power_db, &self.config, scene))
                .collect();

            result.ray_paths.extend(paths);
        }

        let (min, max) = scene_bounds(scene);
        result.energy_grid =
            build_energy_grid(min, max, self.config.grid_resolution, &result.ray_paths);

        result.max_energy = result
            .energy_grid
            .iter()
            .map(|p| p.energy)
            .fold(0.0_f32, f32::max);

        self.result = Some(result);
        self.running = false;
        self.progress = 1.0;
    }
}

fn generate_sphere_rays(origin: Vec3, count: u32) -> Vec<Vec3> {
    // Fibonacci sphere for uniform distribution
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut directions = Vec::with_capacity(count as usize);

    for i in 0..count {
        let theta = 2.0 * PI * (i as f32) / golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / count as f32).acos();

        directions.push(Vec3::new(
            phi.sin() * theta.cos(),
            phi.sin() * theta.sin(),
            phi.cos(),
        ));
    }

    let _ = origin;
    directions
}

fn trace_ray(
    origin: Vec3,
    direction: Vec3,
    power_db: f32,
    config: &SimulationConfig,
    scene: &Scene,
) -> Vec<Vec3> {
    let initial_energy = db_to_linear(power_db);
    let mut ray = AcousticRay::new(origin, direction, initial_energy);

    while ray.bounces < config.max_bounces && ray.energy > config.energy_threshold {
        if let Some((hit, material)) = find_nearest_hit(&ray, scene) {
            ray.reflect(&hit, material);
        } else {
            break;
        }
    }

    ray.path
}

fn find_nearest_hit<'a>(
    ray: &AcousticRay,
    scene: &'a Scene,
) -> Option<(RayHit, &'a AcousticMaterial)> {
    let mut nearest: Option<(RayHit, &AcousticMaterial)> = None;
    let mut nearest_dist = f32::MAX;

    for obj in &scene.meshes {
        for (idx, tri) in obj.mesh.triangles.iter().enumerate() {
            if let Some(t) = ray.intersect_triangle(tri) {
                if t < nearest_dist {
                    nearest_dist = t;
                    let point = ray.origin + ray.direction * t;
                    nearest = Some((
                        RayHit {
                            point,
                            normal: tri.normal(),
                            distance: t,
                            triangle_index: idx,
                        },
                        &obj.material,
                    ));
                }
            }
        }
    }

    nearest
}

fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 10.0)
}

fn scene_bounds(scene: &Scene) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for obj in &scene.meshes {
        let (obj_min, obj_max) = obj.mesh.bounds();
        min = min.min(obj_min);
        max = max.max(obj_max);
    }

    (min, max)
}

fn build_energy_grid(
    min: Vec3,
    max: Vec3,
    resolution: f32,
    ray_paths: &[Vec<Vec3>],
) -> Vec<GridPoint> {
    let size = max - min;
    let nx = (size.x / resolution).ceil() as usize;
    let ny = (size.y / resolution).ceil() as usize;
    let nz = (size.z / resolution).ceil() as usize;

    let mut grid = Vec::with_capacity(nx * ny * nz);

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let pos = min
                    + Vec3::new(
                        (ix as f32 + 0.5) * resolution,
                        (iy as f32 + 0.5) * resolution,
                        (iz as f32 + 0.5) * resolution,
                    );

                let energy = compute_point_energy(pos, resolution, ray_paths);
                grid.push(GridPoint {
                    position: pos,
                    energy,
                });
            }
        }
    }

    grid
}

fn compute_point_energy(point: Vec3, radius: f32, ray_paths: &[Vec<Vec3>]) -> f32 {
    let r2 = radius * radius;
    let mut energy = 0.0_f32;

    for path in ray_paths {
        for segment in path.windows(2) {
            let closest = closest_point_on_segment(segment[0], segment[1], point);
            let dist2 = (closest - point).length_squared();
            if dist2 < r2 {
                energy += 1.0 - (dist2 / r2).sqrt();
            }
        }
    }

    energy
}

fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    a + ab * t
}
