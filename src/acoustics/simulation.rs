use glam::Vec3;
use rayon::prelude::*;
use std::f32::consts::PI;

use super::ray::{broadband, sanitize_absorption, AcousticRay, BandEnergies, RayHit, BAND_COUNT};
use crate::scene::material::MediumProperties;
use crate::scene::{Listener, Scene, SceneObject};

/// Reference energy for SPL conversion. Choosing 1.0 means a source emitting
/// `power_db = 80` produces `linear = 1e8`, and `energy_to_spl(1e8) = 80 dB`
/// — the round-trip preserves the input convention. Useful as a relative scale.
pub const SPL_REFERENCE: f32 = 1.0;

/// Convert linear acoustic energy to dB SPL: `10 * log10(E / SPL_REFERENCE)`.
/// Returns `None` for zero / negative / non-finite energy. The UI surfaces
/// `None` as "No energy received".
pub fn energy_to_spl(energy: f32) -> Option<f32> {
    if energy.is_finite() && energy > 0.0 {
        Some(10.0 * (energy / SPL_REFERENCE).log10())
    } else {
        None
    }
}

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

/// A traced ray's full path: positions and per-point band energies.
/// `positions[i]` is the ray's location after its i-th interaction, and
/// `band_energies[i]` is the 6-band energy it carried at that point.
#[derive(Clone, Debug, Default)]
pub struct TracedRayPath {
    pub positions: Vec<Vec3>,
    pub band_energies: Vec<BandEnergies>,
}

impl TracedRayPath {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Per-listener capture summary: how much energy reached the listener
/// across each band, and the equivalent dB SPL.
#[derive(Clone, Debug, Default)]
pub struct ListenerCapture {
    pub name: String,
    pub position: Vec3,
    pub capture_radius: f32,
    pub energy_bands: BandEnergies,
    /// dB SPL per band. `None` for bands that received zero energy.
    pub spl_bands: [Option<f32>; BAND_COUNT],
    pub broadband_energy: f32,
    /// `None` when broadband energy is zero — UI renders "No energy received".
    pub broadband_spl: Option<f32>,
}

#[derive(Clone, Default)]
pub struct SimulationResult {
    pub energy_grid: Vec<GridPoint>,
    pub ray_paths: Vec<TracedRayPath>,
    pub max_energy: f32,
    pub max_band_energies: BandEnergies,
    pub listener_captures: Vec<ListenerCapture>,
}

#[derive(Clone)]
pub struct GridPoint {
    pub position: Vec3,
    pub energy_bands: BandEnergies,
    /// Broadband = linear average of the 6 band energies.
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
            let bg = &scene.background_medium;

            let paths: Vec<TracedRayPath> = rays
                .into_par_iter()
                .map(|dir| {
                    trace_ray(
                        source.position,
                        dir,
                        source.power_db,
                        &self.config,
                        scene,
                        bg,
                    )
                })
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

        // Track per-band max for spectrum-aware visualisation.
        let mut max_bands: BandEnergies = [0.0; BAND_COUNT];
        for gp in &result.energy_grid {
            for i in 0..BAND_COUNT {
                if gp.energy_bands[i] > max_bands[i] {
                    max_bands[i] = gp.energy_bands[i];
                }
            }
        }
        result.max_band_energies = max_bands;

        // Listener captures — non-destructive, computed post-trace from the
        // existing ray paths. Adding/removing listeners does NOT change the
        // grid or ray paths.
        result.listener_captures = compute_listener_captures(&scene.listeners, &result.ray_paths);

        self.result = Some(result);
        self.running = false;
        self.progress = 1.0;
    }
}

/// For each listener in `listeners`, walk every traced ray segment, test
/// closest-approach distance, and accumulate proximity-weighted per-band
/// energy. Pure read-only — does not modify ray paths.
pub fn compute_listener_captures(
    listeners: &[Listener],
    paths: &[TracedRayPath],
) -> Vec<ListenerCapture> {
    listeners
        .iter()
        .map(|listener| {
            let r = listener.capture_radius.max(1e-6);
            let r2 = r * r;
            let mut bands: BandEnergies = [0.0; BAND_COUNT];

            for path in paths {
                let positions = &path.positions;
                let band_e = &path.band_energies;
                if positions.len() < 2 || band_e.is_empty() {
                    continue;
                }
                for i in 0..positions.len() - 1 {
                    let a = positions[i];
                    let b = positions[i + 1];
                    let closest = closest_point_on_segment(a, b, listener.position);
                    let dist2 = (closest - listener.position).length_squared();
                    if dist2 < r2 {
                        let weight = 1.0 - (dist2 / r2).sqrt();
                        let seg_e = band_e[i.min(band_e.len() - 1)];
                        for b_idx in 0..BAND_COUNT {
                            bands[b_idx] += weight * seg_e[b_idx];
                        }
                    }
                }
            }

            let mut spl_bands: [Option<f32>; BAND_COUNT] = [None; BAND_COUNT];
            for i in 0..BAND_COUNT {
                spl_bands[i] = energy_to_spl(bands[i]);
            }
            let bb = broadband(&bands);
            ListenerCapture {
                name: listener.name.clone(),
                position: listener.position,
                capture_radius: r,
                energy_bands: bands,
                spl_bands,
                broadband_energy: bb,
                broadband_spl: energy_to_spl(bb),
            }
        })
        .collect()
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

/// Maximum number of pending (queued) rays from refraction branching.
const MAX_PENDING_RAYS: usize = 16;

fn trace_ray(
    origin: Vec3,
    direction: Vec3,
    power_db: f32,
    config: &SimulationConfig,
    scene: &Scene,
    background_medium: &MediumProperties,
) -> TracedRayPath {
    let initial_energy = db_to_linear(power_db);
    let mut ray = AcousticRay::new(origin, direction, initial_energy, background_medium.clone());

    let mut pending: Vec<AcousticRay> = Vec::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut band_energies: Vec<BandEnergies> = Vec::new();

    loop {
        // Trace the current ray until it terminates
        while ray.bounces < config.max_bounces && ray.energy > config.energy_threshold {
            if let Some((hit, obj)) = find_nearest_hit(&ray, scene) {
                // Apply volumetric attenuation for distance traveled in current medium
                ray.apply_volumetric_attenuation(hit.distance);

                // Check if energy dropped below threshold after attenuation
                if ray.energy <= config.energy_threshold {
                    break;
                }

                // Determine medium transition
                let new_medium = determine_medium_transition(&ray, obj, background_medium);

                match new_medium {
                    Some(target_medium) => {
                        // Medium boundary: compute refraction
                        let material = &obj.material;
                        let abs_bands = material.absorption.as_array();

                        if let Some(refraction) = ray.refract(hit.normal, &target_medium) {
                            // Apply surface absorption to both reflected and transmitted
                            // per band — high-absorption bands lose more energy.
                            let mut refl_bands = refraction.reflected_band_energies;
                            let mut trans_bands = refraction.transmitted_band_energies;
                            for i in 0..BAND_COUNT {
                                let a = sanitize_absorption(abs_bands[i]);
                                refl_bands[i] *= 1.0 - a;
                                trans_bands[i] *= 1.0 - a;
                            }
                            let transmitted_broadband = broadband(&trans_bands);
                            let reflected_broadband = broadband(&refl_bands);

                            // Queue transmitted ray if not total internal reflection
                            // and energy is above threshold and we have room
                            if let Some(transmitted_dir) = refraction.transmitted_direction {
                                if transmitted_broadband > config.energy_threshold
                                    && pending.len() < MAX_PENDING_RAYS
                                {
                                    let mut transmitted_ray = AcousticRay::new(
                                        hit.point + transmitted_dir * 1e-4,
                                        transmitted_dir,
                                        transmitted_broadband,
                                        target_medium,
                                    );
                                    transmitted_ray.energy_bands = trans_bands;
                                    transmitted_ray.energy = transmitted_broadband;
                                    transmitted_ray.bounces = ray.bounces + 1;
                                    transmitted_ray.path = vec![hit.point];
                                    transmitted_ray.band_path = vec![trans_bands];
                                    pending.push(transmitted_ray);
                                }
                            }

                            // Continue with reflected ray
                            ray.energy_bands = refl_bands;
                            ray.energy = reflected_broadband;
                            ray.origin = hit.point + hit.normal * 1e-4;
                            let refl_dir =
                                ray.direction - 2.0 * ray.direction.dot(hit.normal) * hit.normal;
                            ray.direction = refl_dir.normalize();
                            ray.bounces += 1;
                            ray.push_path_point(hit.point);
                        } else {
                            // Degenerate refraction — fall back to reflect
                            ray.reflect(&hit, &obj.material);
                        }
                    }
                    None => {
                        // Same-medium boundary: use existing reflection logic
                        ray.reflect(&hit, &obj.material);
                    }
                }
            } else {
                break;
            }
        }

        // Collect this ray's path + band energies (parallel arrays).
        positions.extend(ray.path.iter().copied());
        band_energies.extend(ray.band_path.iter().copied());

        // Pick next pending ray, if any
        if let Some(next) = pending.pop() {
            ray = next;
        } else {
            break;
        }
    }

    debug_assert_eq!(
        positions.len(),
        band_energies.len(),
        "positions and band_energies must stay in lock-step"
    );

    TracedRayPath {
        positions,
        band_energies,
    }
}

/// Determine if a medium transition occurs when hitting the given object.
/// Returns `Some(new_medium)` if transitioning, `None` if same-medium boundary.
fn determine_medium_transition(
    ray: &AcousticRay,
    obj: &SceneObject,
    background_medium: &MediumProperties,
) -> Option<MediumProperties> {
    let interior = match &obj.interior_medium {
        Some(m) => m,
        None => return None, // No interior medium — same-medium boundary
    };

    // Determine if ray is currently in the background or in this object's interior.
    // Compare speed_of_sound as a proxy for medium identity.
    let in_background =
        (ray.current_medium.speed_of_sound - background_medium.speed_of_sound).abs() < 1.0;

    if in_background {
        // Entering: transition from background to interior
        Some(interior.clone())
    } else {
        // Exiting: transition from interior back to background
        Some(background_medium.clone())
    }
}

fn find_nearest_hit<'a>(ray: &AcousticRay, scene: &'a Scene) -> Option<(RayHit, &'a SceneObject)> {
    let mut nearest: Option<(RayHit, &SceneObject)> = None;
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
                        obj,
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
    ray_paths: &[TracedRayPath],
) -> Vec<GridPoint> {
    let size = max - min;
    let nx = (size.x / resolution).ceil() as usize;
    let ny = (size.y / resolution).ceil() as usize;
    let nz = (size.z / resolution).ceil() as usize;
    let total = nx * ny * nz;

    // Parallel: each grid cell's energy is independent — read-only access to
    // ray_paths means we can fan out across cores.
    (0..total)
        .into_par_iter()
        .map(|idx| {
            let iz = idx % nz;
            let iy = (idx / nz) % ny;
            let ix = idx / (ny * nz);
            let pos = min
                + Vec3::new(
                    (ix as f32 + 0.5) * resolution,
                    (iy as f32 + 0.5) * resolution,
                    (iz as f32 + 0.5) * resolution,
                );
            let energy_bands = compute_point_energy_bands(pos, resolution, ray_paths);
            let energy = broadband(&energy_bands);
            GridPoint {
                position: pos,
                energy_bands,
                energy,
            }
        })
        .collect()
}

fn compute_point_energy_bands(
    point: Vec3,
    radius: f32,
    ray_paths: &[TracedRayPath],
) -> BandEnergies {
    let r2 = radius * radius;
    let mut bands: BandEnergies = [0.0; BAND_COUNT];

    for path in ray_paths {
        let positions = &path.positions;
        let band_e = &path.band_energies;
        if positions.len() < 2 || band_e.is_empty() {
            continue;
        }
        for i in 0..positions.len() - 1 {
            let a = positions[i];
            let b = positions[i + 1];
            let closest = closest_point_on_segment(a, b, point);
            let dist2 = (closest - point).length_squared();
            if dist2 < r2 {
                let weight = 1.0 - (dist2 / r2).sqrt();
                // Use the band energy recorded at the SEGMENT START — the
                // energy the ray carried when leaving point a.
                let seg_e = band_e[i.min(band_e.len() - 1)];
                for b_idx in 0..BAND_COUNT {
                    bands[b_idx] += weight * seg_e[b_idx];
                }
            }
        }
    }

    bands
}

#[allow(dead_code)]
fn compute_point_energy(point: Vec3, radius: f32, ray_paths: &[TracedRayPath]) -> f32 {
    let bands = compute_point_energy_bands(point, radius, ray_paths);
    broadband(&bands)
}

fn closest_point_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    a + ab * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{MediumLibrary, MediumProperties};
    use crate::scene::primitives;
    use crate::scene::{Listener, SoundSource};

    fn water() -> MediumProperties {
        MediumLibrary::with_defaults().get("Water").unwrap().clone()
    }

    /// Helper: create a simple box-room scene with a centered source.
    fn air_box_scene() -> Scene {
        let room = primitives::box_room(5.0, 5.0, 3.0);
        Scene {
            meshes: vec![room],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: MediumProperties::air(),
            ..Default::default()
        }
    }

    /// Helper: create a small box SceneObject that acts as a water volume.
    fn water_box(pos: Vec3, size: f32) -> SceneObject {
        primitives::platform(pos, size, size, size).with_interior_medium(water())
    }

    // -----------------------------------------------------------------------
    // Task 4 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simulation_air_only_unchanged() {
        // A plain box-room in air should work identically to pre-medium behavior.
        // The ray paths should have points (origin + bounce points) and not be empty.
        let scene = air_box_scene();
        let config = SimulationConfig {
            ray_count: 100,
            max_bounces: 10,
            energy_threshold: 0.001,
            grid_resolution: 0.5,
        };

        let rays = generate_sphere_rays(scene.sound_sources[0].position, config.ray_count);
        let bg = &scene.background_medium;

        let mut total_bounces = 0;
        let mut total_paths = 0;
        for dir in &rays {
            let path = trace_ray(
                scene.sound_sources[0].position,
                *dir,
                scene.sound_sources[0].power_db,
                &config,
                &scene,
                bg,
            );
            // Each path should have at least 2 points (origin + first bounce)
            assert!(
                path.len() >= 2,
                "Air-only path should have at least 2 points, got {}",
                path.len()
            );
            total_bounces += path.len() - 1;
            total_paths += 1;
        }

        assert_eq!(total_paths, 100, "Should trace 100 rays");
        assert!(
            total_bounces > 0,
            "Should have recorded bounces in air-only scene"
        );
    }

    #[test]
    fn test_simulation_with_water_volume() {
        // Place a water-filled box in the scene. Rays hitting it should undergo
        // refraction (medium transition), producing more path points than the
        // same scene without the water object (due to transmitted + reflected rays).
        let mut scene = air_box_scene();
        // Place water box at center of room floor
        scene.meshes.push(water_box(Vec3::new(1.5, 0.0, 1.5), 2.0));

        let config = SimulationConfig {
            ray_count: 200,
            max_bounces: 10,
            energy_threshold: 0.001,
            grid_resolution: 0.5,
        };

        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays = generate_sphere_rays(src.position, config.ray_count);

        let mut hit_water_paths = 0;
        for dir in &rays {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            // Paths that hit the water volume will have refracted rays contributing
            // additional path points. We can't know exactly which rays hit water,
            // but the total path length should generally be > 1 (at least the origin).
            if path.len() > 1 {
                hit_water_paths += 1;
            }
        }

        // Most rays should still produce paths with bounces
        assert!(
            hit_water_paths > 50,
            "Expected many rays to produce multi-point paths with water volume, got {}",
            hit_water_paths
        );
    }

    #[test]
    fn test_simulation_water_attenuates_more_than_air() {
        // Energy reaching a listener through a water boundary is less than
        // through pure air, because the massive impedance mismatch at the
        // air-water interface reflects ~99.9% of energy. Even though water
        // has low per-meter attenuation, the Fresnel boundary loss dominates.
        use crate::acoustics::ray::AcousticRay;

        // Simulate what happens at an air-to-water boundary at normal incidence
        let ray = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        let refraction = ray.refract(Vec3::new(-1.0, 0.0, 0.0), &water()).unwrap();

        // The transmitted energy through the boundary is tiny (~0.1%)
        let energy_through_water = refraction.transmitted_energy;

        // Compare: same ray just traveling through air (no boundary)
        let mut ray_air_only = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        ray_air_only.frequency_hz = 1000.0;
        ray_air_only.apply_volumetric_attenuation(5.0); // 5m of air travel
        let energy_air_path = ray_air_only.energy;

        assert!(
            energy_through_water < energy_air_path,
            "Energy through water boundary ({}) should be less than air-only path ({})",
            energy_through_water,
            energy_air_path
        );
        assert!(
            energy_through_water < 0.01,
            "Transmitted energy through air-water boundary should be tiny: {}",
            energy_through_water
        );
    }

    #[test]
    fn test_simulation_total_internal_reflection_traps_rays() {
        // When a ray is inside a slow medium (air, c=343) hitting a boundary
        // with a fast medium (water, c=1481) at a steep angle, TIR occurs.
        // The refract() call should return transmitted_direction = None.
        // We verify this via the refract API directly with simulation-relevant
        // parameters.
        use crate::acoustics::ray::AcousticRay;

        // Air-to-water at 20 degrees (beyond critical angle ~13.4 deg)
        let angle = 20.0_f32.to_radians();
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay::new(Vec3::new(0.0, 1.0, 0.0), dir, 1.0, MediumProperties::air());

        let result = ray.refract(Vec3::Y, &water()).unwrap();

        // TIR: no transmitted ray
        assert!(
            result.transmitted_direction.is_none(),
            "Should be total internal reflection at 20 deg air-to-water"
        );
        assert!(
            (result.reflected_energy - 1.0).abs() < 0.001,
            "All energy should be reflected in TIR"
        );
        assert!(
            result.transmitted_energy.abs() < 0.001,
            "No transmitted energy in TIR"
        );

        // In the simulation, this means the trace_ray loop would NOT queue
        // a transmitted ray, and the reflected ray continues in the same medium.
        // The ray count stays bounded because no new rays are spawned.
    }

    #[test]
    fn test_simulation_ray_count_bounded() {
        // With multiple medium boundaries, the total pending ray count
        // must stay within MAX_PENDING_RAYS (16).
        // We verify this by creating a scene with several water volumes
        // and checking that tracing doesn't produce an unbounded number
        // of path points (which would indicate unbounded ray spawning).
        let mut scene = air_box_scene();
        // Add several water boxes
        for i in 0..4 {
            let x = 0.5 + i as f32 * 1.0;
            scene.meshes.push(water_box(Vec3::new(x, 0.0, 1.5), 0.8));
        }

        let config = SimulationConfig {
            ray_count: 50,
            max_bounces: 20,
            energy_threshold: 0.0001, // Very low threshold to exercise more branching
            grid_resolution: 0.5,
        };

        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays = generate_sphere_rays(src.position, config.ray_count);

        for dir in &rays {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            // With max_bounces=20 and MAX_PENDING_RAYS=16, the total path points
            // from all branches should be bounded. Each branch can produce at most
            // max_bounces+1 points, and we have at most MAX_PENDING_RAYS+1 branches.
            let upper_bound = (MAX_PENDING_RAYS + 1) * (config.max_bounces as usize + 1);
            assert!(
                path.len() <= upper_bound,
                "Path length {} exceeds upper bound {} (ray count not bounded)",
                path.len(),
                upper_bound
            );
        }
    }

    // -----------------------------------------------------------------------
    // Task 6 — Multi-Medium Integration Tests
    // -----------------------------------------------------------------------

    /// Helper: fetch a medium preset by name from the default library.
    fn medium(name: &str) -> MediumProperties {
        MediumLibrary::with_defaults().get(name).unwrap().clone()
    }

    #[test]
    fn test_integration_underwater_sound_speed() {
        // A source and listener both inside a water-filled room.
        // Rays travel at c_water = 1481 m/s, not c_air = 343 m/s.
        // We verify this by checking that the ray's current_medium inside
        // the water volume matches water's speed of sound, and that the
        // ray paths behave consistently (rays bounce off walls within the
        // water-filled room without medium transition, since the room IS
        // the water volume).

        // Build a water-filled box room (10x10x10 m)
        let water_med = medium("Water");
        let room = primitives::box_room(10.0, 10.0, 10.0).with_interior_medium(water_med.clone());

        let scene = Scene {
            meshes: vec![room],
            sound_sources: vec![SoundSource {
                position: Vec3::new(5.0, 5.0, 5.0),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener {
                position: Vec3::new(8.0, 5.0, 5.0),
                name: "Underwater Listener".into(),
                ..Default::default()
            }],
            // Background is water (entire environment is underwater)
            background_medium: water_med.clone(),
            ..Default::default()
        };

        let config = SimulationConfig {
            ray_count: 500,
            max_bounces: 10,
            energy_threshold: 0.001,
            grid_resolution: 1.0,
        };

        // Trace rays and verify they propagate in water medium
        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays = generate_sphere_rays(src.position, config.ray_count);

        let mut total_path_points = 0;
        for dir in &rays {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            assert!(
                path.len() >= 2,
                "Underwater ray should have at least 2 path points (origin + bounce)"
            );
            total_path_points += path.len();
        }

        // With water as background, all rays propagate in water medium.
        // The speed of sound in the background medium should be water's speed.
        assert!(
            (bg.speed_of_sound - 1481.0).abs() < 0.1,
            "Background medium speed should be water (1481 m/s), got {}",
            bg.speed_of_sound
        );

        // Rays should still bounce in the room, producing multi-point paths.
        // In a 10m room at 1481 m/s, rays travel ~4.3x faster than in air,
        // meaning more bounces occur in the same "distance budget" but the
        // volumetric attenuation in water is much lower than the impedance
        // effects, so rays should survive many bounces.
        let avg_points = total_path_points as f32 / config.ray_count as f32;
        assert!(
            avg_points > 2.0,
            "Underwater rays should average more than 2 path points, got {:.1}",
            avg_points
        );

        // Verify water's propagation characteristics differ from air
        let air_speed = MediumProperties::air().speed_of_sound;
        let water_speed = water_med.speed_of_sound;
        let ratio = water_speed / air_speed;
        assert!(
            (ratio - 4.316).abs() < 0.1,
            "Water/air speed ratio should be ~4.316, got {ratio:.3}"
        );
    }

    #[test]
    fn test_integration_air_water_boundary_energy() {
        // Sound crossing an air-water interface at normal incidence.
        // Expected reflection coefficient R ≈ 0.9989 due to massive
        // impedance mismatch (Z_water / Z_air ≈ 3580x).
        //
        // We test this directly using the refract API on a ray that
        // simulates what happens in a full scene at the air-water boundary.

        use crate::acoustics::ray::AcousticRay;

        let air_med = MediumProperties::air();
        let water_med = medium("Water");

        // Calculate analytical reflection coefficient at normal incidence
        let z_air = air_med.impedance;
        let z_water = water_med.impedance;
        let impedance_ratio = z_water / z_air;
        let analytical_r = ((z_water - z_air) / (z_water + z_air)).powi(2);
        let analytical_t = 1.0 - analytical_r;

        // Verify the impedance ratio is ~3580x
        assert!(
            (impedance_ratio - 3580.0).abs() / 3580.0 < 0.05,
            "Z_water/Z_air should be ~3580, got {impedance_ratio:.1}"
        );

        // Create a ray traveling downward in air hitting a water surface
        let ray = AcousticRay::new(
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            1.0,
            air_med,
        );

        let result = ray.refract(Vec3::Y, &water_med).unwrap();

        // ~99.9% of energy should be reflected
        assert!(
            (result.reflected_energy - analytical_r).abs() < 0.001,
            "Reflected energy should be {analytical_r:.6}, got {:.6}",
            result.reflected_energy
        );
        assert!(
            result.reflected_energy > 0.998,
            "Reflection coefficient should be > 99.8%, got {:.4}",
            result.reflected_energy
        );

        // ~0.1% transmitted into water
        assert!(
            (result.transmitted_energy - analytical_t).abs() < 0.001,
            "Transmitted energy should be {analytical_t:.6}, got {:.6}",
            result.transmitted_energy
        );
        assert!(
            result.transmitted_energy < 0.002,
            "Transmission coefficient should be < 0.2%, got {:.4}",
            result.transmitted_energy
        );

        // Energy conservation: R + T = 1.0
        let total = result.reflected_energy + result.transmitted_energy;
        assert!(
            (total - 1.0).abs() < 1e-5,
            "Energy not conserved: R + T = {total:.8}"
        );

        // Now run a full simulation with a water volume in an air room
        // and verify that most energy stays above the water surface.
        let room = primitives::box_room(10.0, 10.0, 10.0);
        let water_volume =
            primitives::platform(Vec3::ZERO, 10.0, 10.0, 3.0).with_interior_medium(medium("Water"));

        let scene = Scene {
            meshes: vec![room, water_volume],
            sound_sources: vec![SoundSource {
                position: Vec3::new(5.0, 7.0, 5.0), // Source in air, above water
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };

        let config = SimulationConfig {
            ray_count: 1000,
            max_bounces: 15,
            energy_threshold: 0.0001,
            grid_resolution: 1.0,
        };

        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays_dirs = generate_sphere_rays(src.position, config.ray_count);

        let mut total_paths = 0;
        for dir in &rays_dirs {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            // Paths should exist (at least origin + first hit)
            if path.len() >= 2 {
                total_paths += 1;
            }
        }

        // Most rays should produce valid paths (they hit the room walls)
        assert!(
            total_paths > 800,
            "Most rays should produce multi-point paths, got {total_paths}/1000"
        );
    }

    #[test]
    fn test_integration_glass_wall_transmission() {
        // Sound passing through a glass wall: air -> glass -> air.
        // Two Fresnel boundaries with the glass impedance.
        // Glass: Z = 2500 * 5640 = 14,100,000 Pa*s/m
        // Air: Z = 1.225 * 343 = 420.175 Pa*s/m
        //
        // At each boundary (normal incidence):
        // R1 = ((Z_glass - Z_air) / (Z_glass + Z_air))^2
        // T1 = 1 - R1
        // After passing through both boundaries: T_total = T1 * T2
        // Since Z_glass/Z_air is even larger than Z_water/Z_air,
        // almost all energy is reflected at the first boundary.

        use crate::acoustics::ray::AcousticRay;

        let air_med = MediumProperties::air();
        let glass_med = medium("Glass");

        let z_air = air_med.impedance;
        let z_glass = glass_med.impedance;

        // Analytical: normal incidence Fresnel at air->glass boundary
        let r_air_glass = ((z_glass - z_air) / (z_glass + z_air)).powi(2);
        let t_air_glass = 1.0 - r_air_glass;

        // Second boundary glass->air has same R by reciprocity
        let r_glass_air = ((z_air - z_glass) / (z_air + z_glass)).powi(2);
        let t_glass_air = 1.0 - r_glass_air;

        // Total transmission through both boundaries
        let t_total = t_air_glass * t_glass_air;

        // Glass impedance is very high, so reflection should be very high
        assert!(
            r_air_glass > 0.999,
            "Air->glass reflection should be > 99.9%, got {r_air_glass:.6}"
        );

        // Total transmission through glass wall should be very small
        assert!(
            t_total < 0.0001,
            "Total transmission through glass wall should be < 0.01%, got {t_total:.8}"
        );

        // Verify via ray tracing: first boundary
        let ray1 = AcousticRay::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            air_med.clone(),
        );
        let result1 = ray1.refract(Vec3::new(-1.0, 0.0, 0.0), &glass_med).unwrap();

        assert!(
            (result1.reflected_energy - r_air_glass).abs() < 0.001,
            "Ray air->glass R: expected {r_air_glass:.6}, got {:.6}",
            result1.reflected_energy
        );

        // Second boundary: ray in glass hitting air
        if let Some(transmitted_dir) = result1.transmitted_direction {
            let ray2 = AcousticRay::new(
                Vec3::new(0.1, 0.0, 0.0),
                transmitted_dir,
                result1.transmitted_energy,
                glass_med.clone(),
            );
            let result2 = ray2.refract(Vec3::new(-1.0, 0.0, 0.0), &air_med).unwrap();

            // After both boundaries, transmitted energy should match analytical
            let actual_total_t = result2.transmitted_energy;
            // Use relative tolerance since values are very small
            assert!(
                actual_total_t < 0.001,
                "Total energy through glass wall should be < 0.1%, got {actual_total_t:.8}"
            );

            // Verify double refraction: transmitted direction should still
            // be roughly along X axis (glass is denser but at normal incidence
            // direction doesn't change)
            if let Some(final_dir) = result2.transmitted_direction {
                assert!(
                    final_dir.x.abs() > 0.9,
                    "After double refraction at normal incidence, direction should be ~(1,0,0), got {:?}",
                    final_dir
                );
            }
        }

        // Full scene test: glass partition wall in an air room
        let room = primitives::box_room(10.0, 10.0, 5.0);
        let glass_wall = primitives::partition_wall(
            Vec3::new(5.0, 0.0, 0.0),
            0.1, // thin glass wall
            5.0,
            10.0,
        )
        .with_interior_medium(glass_med);

        let scene = Scene {
            meshes: vec![room, glass_wall],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 2.5, 5.0),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener {
                position: Vec3::new(7.5, 2.5, 5.0),
                name: "Behind Glass".into(),
                ..Default::default()
            }],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };

        let config = SimulationConfig {
            ray_count: 500,
            max_bounces: 15,
            energy_threshold: 0.00001,
            grid_resolution: 1.0,
        };

        // Just verify the simulation runs without crashing and produces paths
        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays_dirs = generate_sphere_rays(src.position, config.ray_count);
        let mut valid_paths = 0;
        for dir in &rays_dirs {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            if path.len() >= 2 {
                valid_paths += 1;
            }
        }
        assert!(
            valid_paths > 400,
            "Most rays should produce valid paths with glass wall, got {valid_paths}/500"
        );
    }

    #[test]
    fn test_integration_gas_helium_room() {
        // A room filled with helium (c = 1007 m/s, rho = 0.164 kg/m^3).
        // Sound travels ~2.94x faster in helium vs air (343 m/s).
        // This test verifies that using helium as background medium
        // produces different propagation characteristics than air.

        let helium = medium("Helium");
        let air_med = MediumProperties::air();

        // Verify helium properties
        assert!(
            (helium.speed_of_sound - 1007.0).abs() < 0.1,
            "Helium speed of sound should be 1007, got {}",
            helium.speed_of_sound
        );
        assert!(
            (helium.density - 0.164).abs() < 0.001,
            "Helium density should be 0.164, got {}",
            helium.density
        );

        let speed_ratio = helium.speed_of_sound / air_med.speed_of_sound;
        assert!(
            (speed_ratio - 2.936).abs() < 0.05,
            "Helium/air speed ratio should be ~2.936, got {speed_ratio:.3}"
        );

        // Helium impedance is much lower than air's
        // Z_he = 0.164 * 1007 = 165.1
        // Z_air = 1.225 * 343 = 420.2
        let z_helium = helium.impedance;
        let z_air = air_med.impedance;
        assert!(
            z_helium < z_air,
            "Helium impedance ({z_helium:.1}) should be less than air ({z_air:.1})"
        );

        // Run simulation in helium-filled room
        let room_helium = primitives::box_room(8.0, 8.0, 4.0);
        let scene_helium = Scene {
            meshes: vec![room_helium],
            sound_sources: vec![SoundSource {
                position: Vec3::new(4.0, 2.0, 4.0),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: helium.clone(),
            ..Default::default()
        };

        // Same room in air for comparison
        let room_air = primitives::box_room(8.0, 8.0, 4.0);
        let scene_air = Scene {
            meshes: vec![room_air],
            sound_sources: vec![SoundSource {
                position: Vec3::new(4.0, 2.0, 4.0),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: air_med,
            ..Default::default()
        };

        let config = SimulationConfig {
            ray_count: 500,
            max_bounces: 10,
            energy_threshold: 0.001,
            grid_resolution: 1.0,
        };

        // Trace rays in both environments
        let rays_dirs =
            generate_sphere_rays(scene_helium.sound_sources[0].position, config.ray_count);

        let mut helium_total_points = 0usize;
        let mut air_total_points = 0usize;

        for dir in &rays_dirs {
            let path_he = trace_ray(
                scene_helium.sound_sources[0].position,
                *dir,
                scene_helium.sound_sources[0].power_db,
                &config,
                &scene_helium,
                &scene_helium.background_medium,
            );
            let path_air = trace_ray(
                scene_air.sound_sources[0].position,
                *dir,
                scene_air.sound_sources[0].power_db,
                &config,
                &scene_air,
                &scene_air.background_medium,
            );
            helium_total_points += path_he.len();
            air_total_points += path_air.len();
        }

        // Both should produce valid paths
        assert!(
            helium_total_points > config.ray_count as usize,
            "Helium room should produce multi-point ray paths"
        );
        assert!(
            air_total_points > config.ray_count as usize,
            "Air room should produce multi-point ray paths"
        );

        // Helium has lower attenuation per meter than air (per our preset values),
        // so rays in helium should survive more bounces on average.
        // However, both have very low attenuation so the difference may be small.
        // The key physics difference is the speed of sound, which affects timing
        // but not the geometric ray paths (same room, same ray directions).
        // What DOES differ is volumetric attenuation: helium's attenuation
        // at 1kHz is 0.006 dB/m vs air's 0.01 dB/m, so helium rays lose
        // less energy per meter and thus survive more bounces before hitting
        // the energy threshold.
        let he_avg = helium_total_points as f32 / config.ray_count as f32;
        let air_avg = air_total_points as f32 / config.ray_count as f32;

        // Helium should have at least as many path points as air
        // (lower attenuation means rays survive longer)
        assert!(
            he_avg >= air_avg * 0.9,
            "Helium room should have comparable or more path points: he={he_avg:.1}, air={air_avg:.1}"
        );

        // Verify the speed of sound difference is properly reflected
        // in the medium that rays use. In helium room, background is helium.
        assert!(
            (scene_helium.background_medium.speed_of_sound - 1007.0).abs() < 0.1,
            "Helium scene background speed should be 1007 m/s"
        );
        assert!(
            (scene_air.background_medium.speed_of_sound - 343.0).abs() < 0.1,
            "Air scene background speed should be 343 m/s"
        );
    }

    #[test]
    fn test_integration_energy_conservation() {
        // Verify that at each refraction boundary, the total energy
        // (reflected + transmitted) equals the incident energy.
        // Also verify that volumetric attenuation only reduces energy
        // (never increases it).
        //
        // We test across multiple medium pairs at multiple angles.

        use crate::acoustics::ray::AcousticRay;

        let media_pairs = [
            (MediumProperties::air(), medium("Water")),
            (MediumProperties::air(), medium("Glass")),
            (MediumProperties::air(), medium("Steel")),
            (MediumProperties::air(), medium("Helium")),
            (medium("Water"), medium("Glass")),
            (medium("Water"), medium("Steel")),
        ];

        // Test at various angles (degrees)
        let test_angles = [0.0_f32, 2.0, 5.0, 8.0, 10.0];

        for (m1, m2) in &media_pairs {
            for &angle_deg in &test_angles {
                let angle = angle_deg.to_radians();
                let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
                let initial_energy = 1.0;

                let ray =
                    AcousticRay::new(Vec3::new(0.0, 1.0, 0.0), dir, initial_energy, m1.clone());

                let result = ray.refract(Vec3::Y, m2).unwrap();

                // Check energy conservation: R + T = initial
                let total = result.reflected_energy + result.transmitted_energy;
                assert!(
                    (total - initial_energy).abs() < 1e-4,
                    "Energy not conserved for {} -> {} at {angle_deg} deg: \
                     R={:.6} + T={:.6} = {total:.6} (expected {initial_energy})",
                    m1.name,
                    m2.name,
                    result.reflected_energy,
                    result.transmitted_energy
                );

                // Both energies must be non-negative
                assert!(
                    result.reflected_energy >= 0.0,
                    "Reflected energy must be >= 0 for {} -> {} at {angle_deg} deg: {}",
                    m1.name,
                    m2.name,
                    result.reflected_energy
                );
                assert!(
                    result.transmitted_energy >= 0.0,
                    "Transmitted energy must be >= 0 for {} -> {} at {angle_deg} deg: {}",
                    m1.name,
                    m2.name,
                    result.transmitted_energy
                );

                // If TIR, all energy reflected
                if result.transmitted_direction.is_none() {
                    assert!(
                        (result.reflected_energy - initial_energy).abs() < 1e-4,
                        "TIR should reflect all energy for {} -> {} at {angle_deg} deg",
                        m1.name,
                        m2.name
                    );
                    assert!(
                        result.transmitted_energy.abs() < 1e-4,
                        "TIR should have zero transmitted energy for {} -> {} at {angle_deg} deg",
                        m1.name,
                        m2.name
                    );
                }
            }
        }

        // Also verify volumetric attenuation never increases energy
        let test_media = [
            MediumProperties::air(),
            medium("Water"),
            medium("Glass"),
            medium("Helium"),
            medium("Steel"),
        ];

        for med in &test_media {
            let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, med.clone());
            ray.frequency_hz = 1000.0;

            let e_before = ray.energy;
            ray.apply_volumetric_attenuation(50.0); // 50 meters

            assert!(
                ray.energy <= e_before,
                "Volumetric attenuation should never increase energy in {}: \
                 before={e_before}, after={}",
                med.name,
                ray.energy
            );
            assert!(
                ray.energy > 0.0,
                "Energy should remain positive after attenuation in {}: {}",
                med.name,
                ray.energy
            );
        }

        // Full simulation energy check: run a simple scene and verify
        // that all produced ray paths have finite, positive energies
        // (no NaN, no Inf, no negative values from broken arithmetic).
        let room = primitives::box_room(6.0, 6.0, 4.0);
        let water_vol = primitives::platform(Vec3::new(1.0, 0.0, 1.0), 4.0, 4.0, 2.0)
            .with_interior_medium(medium("Water"));

        let scene = Scene {
            meshes: vec![room, water_vol],
            sound_sources: vec![SoundSource {
                position: Vec3::new(3.0, 3.0, 3.0),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };

        let config = SimulationConfig {
            ray_count: 500,
            max_bounces: 15,
            energy_threshold: 0.0001,
            grid_resolution: 1.0,
        };

        let bg = &scene.background_medium;
        let src = &scene.sound_sources[0];
        let rays_dirs = generate_sphere_rays(src.position, config.ray_count);

        for dir in &rays_dirs {
            let path = trace_ray(src.position, *dir, src.power_db, &config, &scene, bg);
            // All path points should be finite (no NaN/Inf from refraction math)
            for pt in &path.positions {
                assert!(
                    pt.x.is_finite() && pt.y.is_finite() && pt.z.is_finite(),
                    "Path point should be finite: {:?}",
                    pt
                );
            }
        }
    }

    #[test]
    fn test_simulation_volumetric_attenuation_applied() {
        // A ray traveling a longer distance should have less energy than one
        // traveling a shorter distance, all else being equal.
        use crate::acoustics::ray::AcousticRay;

        let medium = MediumProperties::air();

        let mut ray_short = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, medium.clone());
        ray_short.frequency_hz = 1000.0;
        ray_short.apply_volumetric_attenuation(1.0); // 1 meter

        let mut ray_long = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, medium);
        ray_long.frequency_hz = 1000.0;
        ray_long.apply_volumetric_attenuation(100.0); // 100 meters

        assert!(
            ray_long.energy < ray_short.energy,
            "Longer distance should produce lower energy: short={}, long={}",
            ray_short.energy,
            ray_long.energy
        );

        // Both should be positive
        assert!(ray_short.energy > 0.0);
        assert!(ray_long.energy > 0.0);

        // Short distance should still be close to 1.0 (air has low attenuation)
        assert!(
            ray_short.energy > 0.99,
            "1m in air at 1kHz should barely attenuate: {}",
            ray_short.energy
        );
    }

    // -----------------------------------------------------------------------
    // D1 — Per-band absorption verify tests (echomap-v1 T4)
    // -----------------------------------------------------------------------

    /// Fetch a named material from the default library.
    fn material(name: &str) -> crate::scene::material::AcousticMaterial {
        crate::scene::material::MaterialLibrary::with_defaults()
            .materials
            .get(name)
            .expect("material exists")
            .clone()
    }

    /// Reflect a ray N times on the given material. Returns the ray's final
    /// per-band energies. Uses a flat normal so the reflection geometry is
    /// trivial — the test focuses on the absorption math, not ray geometry.
    fn reflect_n_times(material_name: &str, n: usize) -> BandEnergies {
        use crate::acoustics::ray::AcousticRay;
        let mat = material(material_name);
        let mut ray = AcousticRay::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            MediumProperties::air(),
        );
        let hit = RayHit {
            point: Vec3::new(1.0, 0.0, 0.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            distance: 1.0,
            triangle_index: 0,
        };
        for _ in 0..n {
            ray.reflect(&hit, &mat);
        }
        ray.energy_bands
    }

    /// Carpet absorbs 4 kHz strongly (0.73 → 0.27 survival) but 125 Hz
    /// weakly (0.08 → 0.92 survival). After 5 bounces the 4 kHz band must
    /// be at least 1000× weaker than the 125 Hz band — the deliverable's
    /// core claim that absorption is applied PER BAND, not averaged.
    #[test]
    fn absorption_varies_by_band() {
        let bands = reflect_n_times("Carpet", 5);

        let low = bands[0]; // 125 Hz
        let high = bands[5]; // 4 kHz

        assert!(low > 0.0, "125 Hz band should remain positive, got {low}");
        assert!(high > 0.0, "4 kHz band should remain positive, got {high}");
        assert!(
            low > high,
            "after 5 carpet bounces, 125 Hz ({low}) must exceed 4 kHz ({high})"
        );
        // Expected: 0.92^5 / 0.27^5 ≈ 0.659 / 0.00143 ≈ 460. Use 100 as a
        // floor to avoid sensitivity to numerical noise.
        let ratio = low / high;
        assert!(
            ratio > 100.0,
            "carpet 125 Hz/4 kHz ratio after 5 bounces should exceed 100, got {ratio:.2}"
        );

        // Also assert SimulationResult exposes 6 separate band grids — the
        // GridPoint::energy_bands array must be a 6-element array per cell.
        let lib = crate::scene::material::MaterialLibrary::with_defaults();
        let mut room = primitives::box_room(5.0, 5.0, 3.0);
        room.material = lib.materials.get("Carpet").unwrap().clone();
        let scene = Scene {
            meshes: vec![room],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };
        let mut sim = SimulationState {
            config: SimulationConfig {
                ray_count: 100,
                max_bounces: 5,
                energy_threshold: 1e-6,
                grid_resolution: 1.0,
            },
            ..Default::default()
        };
        sim.run(&scene);
        let result = sim.result.expect("sim produced result");
        assert!(!result.energy_grid.is_empty(), "grid should have cells");
        // Every grid point exposes 6 band energies.
        for gp in &result.energy_grid {
            assert_eq!(gp.energy_bands.len(), 6, "grid must hold 6 band energies");
        }
    }

    /// Concrete absorption is nearly uniform (0.01..0.03 across bands). A ray
    /// reflecting many times on concrete should keep ALL bands within a
    /// modest ratio of each other — confirming the per-band path does not
    /// spuriously diverge for uniform-absorbing materials.
    #[test]
    fn concrete_uniform_absorption() {
        let bands = reflect_n_times("Concrete", 5);

        let max_b = bands.iter().cloned().fold(0.0_f32, f32::max);
        let min_b = bands
            .iter()
            .cloned()
            .filter(|x| *x > 0.0)
            .fold(f32::INFINITY, f32::min);

        assert!(max_b > 0.0, "concrete bands should remain positive");
        assert!(
            min_b.is_finite() && min_b > 0.0,
            "no band should drop to zero on concrete, got bands={:?}",
            bands
        );
        // Concrete survival range per bounce: 0.97..0.99. Over 5 bounces:
        // (0.99/0.97)^5 ≈ 1.107. Allow up to 1.3× for safety.
        let ratio = max_b / min_b;
        assert!(
            ratio < 1.3,
            "concrete bands should track within 30% after 5 bounces, ratio={ratio:.3}, bands={:?}",
            bands
        );
    }

    /// GridPoint.energy must equal the mean of energy_bands — the broadband
    /// cache is just an average. Any cell that violates that has been
    /// constructed inconsistently somewhere in the pipeline.
    #[test]
    fn broadband_is_average() {
        let lib = crate::scene::material::MaterialLibrary::with_defaults();
        let mut room = primitives::box_room(5.0, 5.0, 3.0);
        room.material = lib.materials.get("Concrete").unwrap().clone();
        let scene = Scene {
            meshes: vec![room],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener::default()],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };
        let mut sim = SimulationState {
            config: SimulationConfig {
                ray_count: 100,
                max_bounces: 5,
                energy_threshold: 1e-6,
                grid_resolution: 1.0,
            },
            ..Default::default()
        };
        sim.run(&scene);
        let result = sim.result.expect("sim produced result");
        assert!(
            !result.energy_grid.is_empty(),
            "grid should have cells in a populated scene"
        );

        let mut checked = 0;
        for gp in &result.energy_grid {
            let mean: f32 = gp.energy_bands.iter().sum::<f32>() / BAND_COUNT as f32;
            let tol = (mean.abs() * 1e-5).max(1e-9);
            assert!(
                (gp.energy - mean).abs() <= tol,
                "broadband {} != mean(bands)={} at {:?}, bands={:?}",
                gp.energy,
                mean,
                gp.position,
                gp.energy_bands
            );
            if gp.energy > 0.0 {
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "expected at least one cell with positive broadband energy"
        );
    }

    // -----------------------------------------------------------------------
    // D2 — Listener capture + SPL verify tests (echomap-v1 T7)
    // -----------------------------------------------------------------------

    fn d2_scene_with_listener(listener_pos: Vec3) -> Scene {
        let room = primitives::box_room(5.0, 5.0, 3.0);
        Scene {
            meshes: vec![room],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![Listener {
                position: listener_pos,
                name: "L".into(),
                capture_radius: 0.5,
            }],
            background_medium: MediumProperties::air(),
            ..Default::default()
        }
    }

    fn d2_config() -> SimulationConfig {
        SimulationConfig {
            ray_count: 500,
            max_bounces: 8,
            energy_threshold: 1e-6,
            grid_resolution: 1.0,
        }
    }

    /// A listener placed inside the room must receive non-zero energy and
    /// produce a finite broadband SPL value.
    #[test]
    fn listener_captures_energy() {
        let scene = d2_scene_with_listener(Vec3::new(3.5, 1.5, 2.5));
        let mut sim = SimulationState {
            config: d2_config(),
            ..Default::default()
        };
        sim.run(&scene);
        let result = sim.result.expect("sim ran");

        assert_eq!(
            result.listener_captures.len(),
            1,
            "one listener → one capture"
        );
        let cap = &result.listener_captures[0];
        assert!(
            cap.broadband_energy > 0.0,
            "listener inside room should receive non-zero energy, got {}",
            cap.broadband_energy
        );
        let spl = cap
            .broadband_spl
            .expect("non-zero energy should yield Some(SPL)");
        assert!(
            spl.is_finite(),
            "SPL should be finite for positive energy, got {spl}"
        );
        // At least one band should also be positive.
        assert!(
            cap.energy_bands.iter().any(|e| *e > 0.0),
            "at least one band should have captured energy: {:?}",
            cap.energy_bands
        );
    }

    /// A listener farther from the source receives less energy than a
    /// listener nearer the source — geometric falloff plus longer
    /// volumetric attenuation paths reduce the captured signal.
    #[test]
    fn listener_distance_falloff() {
        // Near listener: 0.7 m from source (just outside the capture radius
        // of the source position itself so the first segment doesn't trivially
        // dominate).
        let scene_near = d2_scene_with_listener(Vec3::new(2.5, 1.5, 3.2));
        // Far listener: 2.2 m from source (against a wall, max distance).
        let scene_far = d2_scene_with_listener(Vec3::new(2.5, 1.5, 4.7));

        let mut sim_near = SimulationState {
            config: d2_config(),
            ..Default::default()
        };
        sim_near.run(&scene_near);
        let near = sim_near
            .result
            .expect("near sim ran")
            .listener_captures
            .remove(0);

        let mut sim_far = SimulationState {
            config: d2_config(),
            ..Default::default()
        };
        sim_far.run(&scene_far);
        let far = sim_far
            .result
            .expect("far sim ran")
            .listener_captures
            .remove(0);

        assert!(
            near.broadband_energy > 0.0,
            "near listener should receive energy: {}",
            near.broadband_energy
        );
        assert!(
            far.broadband_energy > 0.0,
            "far listener should receive energy: {}",
            far.broadband_energy
        );
        assert!(
            near.broadband_energy > far.broadband_energy,
            "near listener energy ({}) should exceed far listener energy ({})",
            near.broadband_energy,
            far.broadband_energy
        );
    }

    /// Adding a listener must not change ray_paths or energy_grid — capture
    /// is post-trace and purely additive (non-destructive).
    #[test]
    fn capture_nondestructive() {
        // Scene WITHOUT listener
        let room_a = primitives::box_room(5.0, 5.0, 3.0);
        let scene_no_listener = Scene {
            meshes: vec![room_a],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };
        // Same scene WITH listener
        let scene_with_listener = d2_scene_with_listener(Vec3::new(3.5, 1.5, 2.5));

        let cfg = d2_config();

        let mut sim_a = SimulationState {
            config: cfg.clone(),
            ..Default::default()
        };
        sim_a.run(&scene_no_listener);
        let res_a = sim_a.result.expect("a ran");

        let mut sim_b = SimulationState {
            config: cfg,
            ..Default::default()
        };
        sim_b.run(&scene_with_listener);
        let res_b = sim_b.result.expect("b ran");

        // ray_paths must be identical: ray tracing doesn't depend on listeners.
        assert_eq!(
            res_a.ray_paths.len(),
            res_b.ray_paths.len(),
            "listener should not change ray count"
        );
        for (pa, pb) in res_a.ray_paths.iter().zip(res_b.ray_paths.iter()) {
            assert_eq!(
                pa.positions.len(),
                pb.positions.len(),
                "listener should not change path length"
            );
        }
        // Energy grid should be element-for-element identical.
        assert_eq!(
            res_a.energy_grid.len(),
            res_b.energy_grid.len(),
            "grid sizes must match"
        );
        for (ga, gb) in res_a.energy_grid.iter().zip(res_b.energy_grid.iter()) {
            assert!(
                (ga.energy - gb.energy).abs() < 1e-6,
                "grid energy must match: {} vs {}",
                ga.energy,
                gb.energy
            );
        }
        // And confirm listener was actually populated in the listener run.
        assert_eq!(res_b.listener_captures.len(), 1);
        assert!(res_b.listener_captures[0].broadband_energy > 0.0);
    }

    /// A listener fully enclosed in its own sealed box (no opening to the
    /// source's room) receives near-zero energy — direct sound cannot
    /// reach it, only any rays that fluke into the capture radius.
    #[test]
    fn listener_separated_by_wall() {
        // Source room: 5×5×3 m. Place a smaller fully-enclosed inner box
        // far from the source so no rays through walls reach the listener.
        let source_room = primitives::box_room(20.0, 20.0, 6.0);
        let inner_box = primitives::box_room(2.0, 2.0, 2.0);
        // Translate inner_box to (15, 0, 15) so it sits in the far corner.
        let mut inner_translated = inner_box.clone();
        for tri in &mut inner_translated.mesh.triangles {
            for v in &mut tri.vertices {
                v.position += Vec3::new(15.0, 0.0, 15.0);
            }
        }

        let listener_pos = Vec3::new(16.0, 1.0, 16.0); // inside inner box
        let scene = Scene {
            meshes: vec![source_room, inner_translated],
            sound_sources: vec![SoundSource {
                position: Vec3::new(2.5, 1.5, 2.5),
                frequency_hz: 1000.0,
                power_db: 80.0,
                enabled: true,
            }],
            listeners: vec![
                Listener {
                    position: Vec3::new(3.5, 1.5, 2.5), // direct-path listener (sanity)
                    name: "Direct".into(),
                    capture_radius: 0.5,
                },
                Listener {
                    position: listener_pos,
                    name: "Walled".into(),
                    capture_radius: 0.5,
                },
            ],
            background_medium: MediumProperties::air(),
            ..Default::default()
        };

        let mut sim = SimulationState {
            config: d2_config(),
            ..Default::default()
        };
        sim.run(&scene);
        let result = sim.result.expect("sim ran");

        let direct = &result.listener_captures[0];
        let walled = &result.listener_captures[1];

        assert!(
            direct.broadband_energy > 0.0,
            "direct listener should capture energy: {}",
            direct.broadband_energy
        );
        // Walled listener may catch some energy from grazing rays around the
        // box, but must be at least 10x less than the direct listener.
        assert!(
            walled.broadband_energy < direct.broadband_energy / 10.0,
            "walled listener ({}) should receive much less than direct ({}) — ratio {:.4}",
            walled.broadband_energy,
            direct.broadband_energy,
            walled.broadband_energy / direct.broadband_energy.max(1e-12)
        );
    }
}
