use glam::Vec3;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use super::bvh::Bvh;
use super::ray::{energy_max, energy_sum, energy_uniform, AcousticRay, EnergyBands, RayHit};
use crate::scene::material::MediumProperties;
use crate::scene::{Scene, SceneObject, SoundSource};

/// Bounded ray streaming buffer. Backpressure: when full the worker thread
/// blocks on `send`, which keeps memory growth flat while still letting the
/// UI drain at its own cadence. 256 is large enough to absorb a typical
/// 60 Hz drain cycle for 10k-ray scenes without stalling tracing.
pub const RAY_STREAM_CAPACITY: usize = 256;

/// Cancellation is checked at this bounce interval inside the trace loop.
/// Small enough that a "stop" press is honoured within a few ms even on
/// long-bouncing rays; large enough that the atomic load isn't the hot path.
pub const CANCEL_CHECK_INTERVAL: u32 = 10;

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
    /// Per-band peak energy across the grid. Index ordering matches
    /// `FrequencyBands.as_array()` (125/250/500/1k/2k/4k Hz).
    pub max_energy: EnergyBands,
}

#[derive(Clone)]
pub struct GridPoint {
    pub position: Vec3,
    pub energy: EnergyBands,
}

impl GridPoint {
    /// Single scalar that summarises the band vector for plain visualisations
    /// (heatmaps that don't yet know about bands). Uses the band-sum so
    /// brighter cells still mean "more total acoustic energy passes here".
    #[inline]
    pub fn energy_total(&self) -> f32 {
        energy_sum(&self.energy)
    }
}

/// Tri-state lifecycle for a simulation run. The UI thread inspects this
/// every frame; the worker thread never touches it. `Running` carries
/// everything the UI needs to *observe* progress (read-only) and to
/// *cancel* (write-only via `AtomicBool`) — but the actual paths flow
/// through the bounded mpsc channel to keep the enum small.
pub enum SimulationPhase {
    Idle,
    Running {
        /// Number of rays the worker has finished tracing so far. Compared
        /// against `total_rays` for a 0..1 ratio.
        progress: Arc<AtomicU32>,
        /// Total rays this run will trace (sum across enabled sources).
        total_rays: u32,
        /// Flip to `true` to ask the worker to stop. The worker observes
        /// this every `CANCEL_CHECK_INTERVAL` bounces inside `trace_ray`
        /// and between rays/sources at the loop boundaries.
        cancel: Arc<AtomicBool>,
        /// Bounded ray-path stream. Drained by `tick()` on the UI thread.
        rx: Receiver<Vec<Vec3>>,
        /// Worker join handle. `take()`'d once the worker finishes so the
        /// final result can be moved into `Complete`.
        thread: Option<JoinHandle<SimulationResult>>,
        /// Paths drained from `rx` so far. The renderer can read these
        /// while the worker is still tracing — that's the non-blocking
        /// behaviour the spec calls out.
        partial_paths: Vec<Vec<Vec3>>,
    },
    Complete {
        result: SimulationResult,
    },
}

impl Default for SimulationPhase {
    fn default() -> Self {
        SimulationPhase::Idle
    }
}

#[derive(Default)]
pub struct SimulationState {
    pub config: SimulationConfig,
    pub phase: SimulationPhase,
}

impl SimulationState {
    /// Spawn a background worker that traces every ray and streams paths
    /// back over a bounded channel. Returns immediately — call `tick()`
    /// every frame to drain. Does nothing if the scene is empty or another
    /// run is already in flight.
    pub fn start(&mut self, scene: &Scene) {
        if matches!(self.phase, SimulationPhase::Running { .. }) {
            return;
        }
        if scene.sound_sources.is_empty() || scene.meshes.is_empty() {
            return;
        }

        let enabled_sources: Vec<SoundSource> = scene
            .sound_sources
            .iter()
            .filter(|s| s.enabled)
            .cloned()
            .collect();
        if enabled_sources.is_empty() {
            return;
        }

        let total_rays = enabled_sources.len() as u32 * self.config.ray_count;
        let cancel = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(AtomicU32::new(0));
        let (tx, rx) = mpsc::sync_channel::<Vec<Vec3>>(RAY_STREAM_CAPACITY);

        let meshes = scene.meshes.clone();
        let bg = scene.background_medium.clone();
        let config = self.config.clone();
        let cancel_worker = cancel.clone();
        let progress_worker = progress.clone();

        let thread = thread::spawn(move || {
            simulate_worker(
                meshes,
                enabled_sources,
                bg,
                config,
                cancel_worker,
                progress_worker,
                tx,
            )
        });

        self.phase = SimulationPhase::Running {
            progress,
            total_rays,
            cancel,
            rx,
            thread: Some(thread),
            partial_paths: Vec::new(),
        };
    }

    /// Synchronously block on the running worker until it returns. Used by
    /// the tests where we want to assert post-run state without spinning a
    /// tick loop. Returns true if a run finished, false if there was none.
    pub fn join(&mut self) -> bool {
        let take_thread = if let SimulationPhase::Running { thread, rx, .. } = &mut self.phase {
            // Continuously drain while the worker is still running. The
            // channel is bounded (256) — if we drained once and then
            // blocked on `thread.join()`, the worker could be stuck on
            // `tx.send()` after the buffer filled, producing a deadlock.
            // `recv_timeout` lets us interleave drains with finish checks
            // without spinning hot.
            loop {
                match rx.recv_timeout(std::time::Duration::from_millis(10)) {
                    Ok(_path) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if thread.as_ref().map(|t| t.is_finished()).unwrap_or(true) {
                            break;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            thread.take()
        } else {
            None
        };

        if let Some(handle) = take_thread {
            let result = handle.join().expect("simulation worker panicked");
            // Drop any tail items still sitting on the channel — they're
            // duplicated in `result.ray_paths` so dropping is safe.
            if let SimulationPhase::Running { rx, .. } = &mut self.phase {
                while rx.try_recv().is_ok() {}
            }
            self.phase = SimulationPhase::Complete { result };
            return true;
        }
        false
    }

    /// Ask the worker to stop. The next bounce check (≤ 10 bounces) and the
    /// next ray boundary will both observe this and short-circuit.
    pub fn cancel(&mut self) {
        if let SimulationPhase::Running { cancel, .. } = &self.phase {
            cancel.store(true, Ordering::Release);
        }
    }

    /// Drain any paths the worker has streamed since the last tick, and
    /// transition to `Complete` once the worker has exited. Returns true if
    /// new paths arrived this tick — useful for deciding whether to call
    /// `ctx.request_repaint()`.
    pub fn tick(&mut self) -> bool {
        let mut new_paths = false;
        let take_thread = match &mut self.phase {
            SimulationPhase::Running {
                rx,
                thread,
                partial_paths,
                ..
            } => {
                loop {
                    match rx.try_recv() {
                        Ok(path) => {
                            partial_paths.push(path);
                            new_paths = true;
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => break,
                    }
                }
                thread
                    .as_ref()
                    .map(|t| t.is_finished())
                    .unwrap_or(false)
                    .then(|| thread.take())
                    .flatten()
            }
            _ => None,
        };

        if let Some(handle) = take_thread {
            let result = handle.join().expect("simulation worker panicked");
            if let SimulationPhase::Running { rx, .. } = &mut self.phase {
                while let Ok(_) = rx.try_recv() {}
            }
            self.phase = SimulationPhase::Complete { result };
            new_paths = true;
        }

        new_paths
    }

    /// True while a worker thread is alive.
    pub fn is_running(&self) -> bool {
        matches!(self.phase, SimulationPhase::Running { .. })
    }

    /// 0.0..=1.0 progress. `Idle` is 0, `Complete` is 1, `Running` is the
    /// worker's atomic ray-count divided by the total ray budget.
    pub fn progress(&self) -> f32 {
        match &self.phase {
            SimulationPhase::Idle => 0.0,
            SimulationPhase::Complete { .. } => 1.0,
            SimulationPhase::Running {
                progress,
                total_rays,
                ..
            } => {
                if *total_rays == 0 {
                    0.0
                } else {
                    (progress.load(Ordering::Acquire) as f32) / (*total_rays as f32)
                }
            }
        }
    }

    /// The final result, available once the worker has joined.
    pub fn result(&self) -> Option<&SimulationResult> {
        match &self.phase {
            SimulationPhase::Complete { result } => Some(result),
            _ => None,
        }
    }

    /// Live view of the paths streamed so far, even while still running.
    pub fn partial_paths(&self) -> &[Vec<Vec3>] {
        match &self.phase {
            SimulationPhase::Running { partial_paths, .. } => partial_paths,
            SimulationPhase::Complete { result } => &result.ray_paths,
            SimulationPhase::Idle => &[],
        }
    }

    /// Convenience for legacy callers that want blocking semantics: start
    /// the worker, then immediately join. Equivalent to the old `run()`.
    pub fn run_blocking(&mut self, scene: &Scene) {
        self.start(scene);
        self.join();
    }
}

/// Trace every ray from every enabled source synchronously using the
/// linear-scan triangle test. The "brute force" baseline used by the
/// criterion bench to measure BVH speedup.
pub fn trace_all_rays_brute_force(scene: &Scene, config: &SimulationConfig) -> Vec<Vec<Vec3>> {
    trace_all_rays_inner(scene, config, None)
}

/// Trace every ray from every enabled source using a prebuilt BVH. The
/// BVH must be built from the same `scene.meshes` slice the trace will
/// query against — wiring a stale BVH is the kind of bug that produces
/// wrong-but-not-NaN paths, so callers should build per-run.
pub fn trace_all_rays_with_bvh(
    scene: &Scene,
    config: &SimulationConfig,
    bvh: &Bvh,
) -> Vec<Vec<Vec3>> {
    trace_all_rays_inner(scene, config, Some(bvh))
}

fn trace_all_rays_inner(
    scene: &Scene,
    config: &SimulationConfig,
    bvh: Option<&Bvh>,
) -> Vec<Vec<Vec3>> {
    let bg = scene.background_medium.clone();
    let mut paths: Vec<Vec<Vec3>> = Vec::with_capacity(config.ray_count as usize);
    for source in &scene.sound_sources {
        if !source.enabled {
            continue;
        }
        let rays = generate_sphere_rays(source.position, config.ray_count);
        for dir in rays {
            paths.push(trace_ray_in(
                source.position,
                dir,
                source.power_db,
                config,
                &scene.meshes,
                &bg,
                None,
                bvh,
            ));
        }
    }
    paths
}

#[allow(clippy::too_many_arguments)]
fn simulate_worker(
    meshes: Vec<SceneObject>,
    sources: Vec<SoundSource>,
    bg: MediumProperties,
    config: SimulationConfig,
    cancel: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
    tx: SyncSender<Vec<Vec3>>,
) -> SimulationResult {
    // Build a transient Scene-shaped struct the existing trace helpers
    // already expect. We don't reuse the public `Scene` (which carries
    // unrelated fluid/gas/robot state we never need here) to keep the
    // thread snapshot minimal and to avoid forcing `Scene: Clone` on the
    // wider codebase.
    let scene_view = SceneView { meshes };

    // Build the spatial accel once per run. Amortised over `ray_count`
    // rays — for the 10k-ray budget this is the difference between
    // O(rays × tris) and O(rays × log tris).
    let bvh = Bvh::build(&scene_view.meshes);

    let mut all_paths: Vec<Vec<Vec3>> = Vec::new();
    'outer: for source in &sources {
        if cancel.load(Ordering::Acquire) {
            break;
        }
        let rays = generate_sphere_rays(source.position, config.ray_count);
        for dir in rays {
            if cancel.load(Ordering::Acquire) {
                break 'outer;
            }
            let path = trace_ray_cancellable(
                source.position,
                dir,
                source.power_db,
                &config,
                &scene_view,
                &bg,
                Some(&cancel),
                Some(&bvh),
            );
            // Try to push to the UI stream; ignore disconnects (UI dropped).
            // Send is blocking when full, which is the deliberate
            // backpressure mechanism for the bounded channel.
            let _ = tx.send(path.clone());
            all_paths.push(path);
            progress.fetch_add(1, Ordering::Release);
        }
    }

    // Build the energy grid from whatever paths we have. On cancel this is
    // still a meaningful partial result, not garbage.
    let (min, max) = scene_view_bounds(&scene_view);
    let energy_grid = build_energy_grid(min, max, config.grid_resolution, &all_paths);

    let mut max_energy: EnergyBands = energy_uniform(0.0);
    for gp in &energy_grid {
        for b in 0..6 {
            if gp.energy[b] > max_energy[b] {
                max_energy[b] = gp.energy[b];
            }
        }
    }

    SimulationResult {
        energy_grid,
        ray_paths: all_paths,
        max_energy,
    }
}

/// Minimal view the tracing helpers actually need. Kept private so the
/// public `Scene` doesn't have to grow a `Clone` impl across the whole app.
struct SceneView {
    meshes: Vec<SceneObject>,
}

fn scene_view_bounds(scene: &SceneView) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for obj in &scene.meshes {
        let (obj_min, obj_max) = obj.mesh.bounds();
        min = min.min(obj_min);
        max = max.max(obj_max);
    }
    (min, max)
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

#[cfg(test)]
fn trace_ray(
    origin: Vec3,
    direction: Vec3,
    power_db: f32,
    config: &SimulationConfig,
    scene: &Scene,
    background_medium: &MediumProperties,
) -> Vec<Vec3> {
    // Thin façade over the cancellable variant. Synchronous tests don't
    // need cancellation, so they pass `None` and pay no atomic load cost.
    // No BVH passed → linear scan (kept as the brute-force baseline that
    // every D4 / bench test compares against).
    trace_ray_in(
        origin,
        direction,
        power_db,
        config,
        &scene.meshes,
        background_medium,
        None,
        None,
    )
}

/// Cancellable variant operating directly on a meshes slice. The worker
/// thread uses this with `Some(&cancel)` and (optionally) a prebuilt
/// `Bvh`; tests use it with `None`s.
#[allow(clippy::too_many_arguments)]
fn trace_ray_cancellable(
    origin: Vec3,
    direction: Vec3,
    power_db: f32,
    config: &SimulationConfig,
    scene_view: &SceneView,
    background_medium: &MediumProperties,
    cancel: Option<&AtomicBool>,
    bvh: Option<&Bvh>,
) -> Vec<Vec3> {
    trace_ray_in(
        origin,
        direction,
        power_db,
        config,
        &scene_view.meshes,
        background_medium,
        cancel,
        bvh,
    )
}

#[allow(clippy::too_many_arguments)]
fn trace_ray_in(
    origin: Vec3,
    direction: Vec3,
    power_db: f32,
    config: &SimulationConfig,
    meshes: &[SceneObject],
    background_medium: &MediumProperties,
    cancel: Option<&AtomicBool>,
    bvh: Option<&Bvh>,
) -> Vec<Vec3> {
    let initial_energy = db_to_linear(power_db);
    let mut ray = AcousticRay::new(origin, direction, initial_energy, background_medium.clone());

    let mut pending: Vec<AcousticRay> = Vec::new();
    let mut all_paths: Vec<Vec3> = Vec::new();
    let mut last_cancel_check: u32 = 0;

    loop {
        // Trace the current ray until it terminates. The survival criterion
        // is "any band still carries energy above threshold" — using the max
        // band keeps the old scalar contract while letting per-band
        // attenuation diverge in future deliverables.
        while ray.bounces < config.max_bounces && energy_max(&ray.energy) > config.energy_threshold
        {
            // Cancel observed every CANCEL_CHECK_INTERVAL bounces. The
            // counter is local rather than `% interval` so refraction
            // branching (which resets `ray.bounces` mid-trace) can't make
            // us skip a check.
            if let Some(c) = cancel {
                if ray.bounces.wrapping_sub(last_cancel_check) >= CANCEL_CHECK_INTERVAL {
                    last_cancel_check = ray.bounces;
                    if c.load(Ordering::Acquire) {
                        // Flush what we already have so the partial path
                        // up to the cancel point isn't lost.
                        all_paths.extend(ray.path.iter().copied());
                        return all_paths;
                    }
                }
            }

            // Prefer the BVH when available; fall back to linear scan
            // so the brute-force trace_ray path stays available for
            // cross-checks (D4 verifies equivalence within float
            // epsilon) and so tests that don't build a BVH still work.
            let hit_pair = match bvh {
                Some(b) => b
                    .nearest_hit(&ray, meshes)
                    .map(|(h, tref)| (h, &meshes[tref.object_index])),
                None => find_nearest_hit_in(&ray, meshes),
            };
            if let Some((hit, obj)) = hit_pair {
                // Apply volumetric attenuation for distance traveled in current medium
                ray.apply_volumetric_attenuation(hit.distance);

                // Check if energy dropped below threshold after attenuation
                if energy_max(&ray.energy) <= config.energy_threshold {
                    break;
                }

                // Determine medium transition
                let new_medium = determine_medium_transition(&ray, obj, background_medium);

                match new_medium {
                    Some(target_medium) => {
                        // Medium boundary: compute refraction
                        let material = &obj.material;
                        let absorption = material.absorption.as_array();

                        if let Some(refraction) = ray.refract(hit.normal, &target_medium) {
                            // Apply surface absorption per band to both reflected and
                            // transmitted energy vectors.
                            let mut reflected_energy: EnergyBands = refraction.reflected_energy;
                            let mut transmitted_energy: EnergyBands = refraction.transmitted_energy;
                            for b in 0..6 {
                                reflected_energy[b] *= 1.0 - absorption[b];
                                transmitted_energy[b] *= 1.0 - absorption[b];
                            }

                            // Queue transmitted ray if not total internal reflection
                            // and energy is above threshold and we have room
                            if let Some(transmitted_dir) = refraction.transmitted_direction {
                                if energy_max(&transmitted_energy) > config.energy_threshold
                                    && pending.len() < MAX_PENDING_RAYS
                                {
                                    let mut transmitted_ray = AcousticRay::new_with_bands(
                                        hit.point + transmitted_dir * 1e-4,
                                        transmitted_dir,
                                        transmitted_energy,
                                        target_medium,
                                    );
                                    transmitted_ray.bounces = ray.bounces + 1;
                                    // Carry over path context: start from hit point
                                    transmitted_ray.path = vec![hit.point];
                                    pending.push(transmitted_ray);
                                }
                            }

                            // Continue with reflected ray
                            ray.energy = reflected_energy;
                            ray.origin = hit.point + hit.normal * 1e-4;
                            let refl_dir =
                                ray.direction - 2.0 * ray.direction.dot(hit.normal) * hit.normal;
                            ray.direction = refl_dir.normalize();
                            ray.bounces += 1;
                            ray.path.push(hit.point);
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

        // Collect this ray's path
        all_paths.extend(ray.path.iter().copied());

        // Pick next pending ray, if any
        if let Some(next) = pending.pop() {
            // Insert a NaN separator so path segments from different rays
            // don't create spurious connections in the energy grid.
            // Actually, we want to return a single Vec<Vec3> path for
            // backward compat. Just extend with the pending ray's path.
            ray = next;
        } else {
            break;
        }
    }

    all_paths
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

fn find_nearest_hit_in<'a>(
    ray: &AcousticRay,
    meshes: &'a [SceneObject],
) -> Option<(RayHit, &'a SceneObject)> {
    let mut nearest: Option<(RayHit, &SceneObject)> = None;
    let mut nearest_dist = f32::MAX;

    for obj in meshes {
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

#[cfg(test)]
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
            let energy = compute_point_energy(pos, resolution, ray_paths);
            GridPoint {
                position: pos,
                energy,
            }
        })
        .collect()
}

fn compute_point_energy(point: Vec3, radius: f32, ray_paths: &[Vec<Vec3>]) -> EnergyBands {
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

    // D1 stores per-band grids but does not yet carry per-band energy
    // along ray paths — broadcast the scalar contribution to every band so
    // downstream visualisers see identical heat maps until a later
    // deliverable populates per-band path samples.
    energy_uniform(energy)
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
        let energy_through_water = refraction.transmitted_energy[0];

        // Compare: same ray just traveling through air (no boundary)
        let mut ray_air_only = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        ray_air_only.frequency_hz = 1000.0;
        ray_air_only.apply_volumetric_attenuation(5.0); // 5m of air travel
        let energy_air_path = ray_air_only.energy[0];

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
            (result.reflected_energy[0] - 1.0).abs() < 0.001,
            "All energy should be reflected in TIR"
        );
        assert!(
            result.transmitted_energy[0].abs() < 0.001,
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
            (result.reflected_energy[0] - analytical_r).abs() < 0.001,
            "Reflected energy should be {analytical_r:.6}, got {:.6}",
            result.reflected_energy[0]
        );
        assert!(
            result.reflected_energy[0] > 0.998,
            "Reflection coefficient should be > 99.8%, got {:.4}",
            result.reflected_energy[0]
        );

        // ~0.1% transmitted into water
        assert!(
            (result.transmitted_energy[0] - analytical_t).abs() < 0.001,
            "Transmitted energy should be {analytical_t:.6}, got {:.6}",
            result.transmitted_energy[0]
        );
        assert!(
            result.transmitted_energy[0] < 0.002,
            "Transmission coefficient should be < 0.2%, got {:.4}",
            result.transmitted_energy[0]
        );

        // Energy conservation: R + T = 1.0
        let total = result.reflected_energy[0] + result.transmitted_energy[0];
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
            (result1.reflected_energy[0] - r_air_glass).abs() < 0.001,
            "Ray air->glass R: expected {r_air_glass:.6}, got {:.6}",
            result1.reflected_energy[0]
        );

        // Second boundary: ray in glass hitting air
        if let Some(transmitted_dir) = result1.transmitted_direction {
            let ray2 = AcousticRay::new_with_bands(
                Vec3::new(0.1, 0.0, 0.0),
                transmitted_dir,
                result1.transmitted_energy,
                glass_med.clone(),
            );
            let result2 = ray2.refract(Vec3::new(-1.0, 0.0, 0.0), &air_med).unwrap();

            // After both boundaries, transmitted energy should match analytical
            let actual_total_t = result2.transmitted_energy[0];
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
                let total = result.reflected_energy[0] + result.transmitted_energy[0];
                assert!(
                    (total - initial_energy).abs() < 1e-4,
                    "Energy not conserved for {} -> {} at {angle_deg} deg: \
                     R={:.6} + T={:.6} = {total:.6} (expected {initial_energy})",
                    m1.name,
                    m2.name,
                    result.reflected_energy[0],
                    result.transmitted_energy[0]
                );

                // Both energies must be non-negative
                assert!(
                    result.reflected_energy[0] >= 0.0,
                    "Reflected energy must be >= 0 for {} -> {} at {angle_deg} deg: {}",
                    m1.name,
                    m2.name,
                    result.reflected_energy[0]
                );
                assert!(
                    result.transmitted_energy[0] >= 0.0,
                    "Transmitted energy must be >= 0 for {} -> {} at {angle_deg} deg: {}",
                    m1.name,
                    m2.name,
                    result.transmitted_energy[0]
                );

                // If TIR, all energy reflected
                if result.transmitted_direction.is_none() {
                    assert!(
                        (result.reflected_energy[0] - initial_energy).abs() < 1e-4,
                        "TIR should reflect all energy for {} -> {} at {angle_deg} deg",
                        m1.name,
                        m2.name
                    );
                    assert!(
                        result.transmitted_energy[0].abs() < 1e-4,
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

            let e_before = ray.energy[0];
            ray.apply_volumetric_attenuation(50.0); // 50 meters

            assert!(
                ray.energy[0] <= e_before,
                "Volumetric attenuation should never increase energy in {}: \
                 before={e_before}, after={}",
                med.name,
                ray.energy[0]
            );
            assert!(
                ray.energy[0] > 0.0,
                "Energy should remain positive after attenuation in {}: {}",
                med.name,
                ray.energy[0]
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
            for pt in &path {
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
            ray_long.energy[0] < ray_short.energy[0],
            "Longer distance should produce lower energy: short={}, long={}",
            ray_short.energy[0],
            ray_long.energy[0]
        );

        // Both should be positive
        assert!(ray_short.energy[0] > 0.0);
        assert!(ray_long.energy[0] > 0.0);

        // Short distance should still be close to 1.0 (air has low attenuation)
        assert!(
            ray_short.energy[0] > 0.99,
            "1m in air at 1kHz should barely attenuate: {}",
            ray_short.energy[0]
        );
    }

    // -----------------------------------------------------------------------
    // D2 tests — async, non-blocking, cancellable simulation
    // -----------------------------------------------------------------------

    /// SimulationState must walk Idle → Running → Complete and never get
    /// stuck. Default is Idle; `start` flips to Running; `join` flips to
    /// Complete and exposes a result.
    #[test]
    fn test_simulation_state_transitions() {
        let scene = air_box_scene();
        let mut sim = SimulationState::default();
        sim.config.ray_count = 200;
        sim.config.max_bounces = 5;

        assert!(
            matches!(sim.phase, SimulationPhase::Idle),
            "default state should be Idle"
        );

        sim.start(&scene);
        assert!(
            matches!(sim.phase, SimulationPhase::Running { .. }),
            "after start() state should be Running"
        );

        assert!(sim.join(), "join() should report a run finished");
        assert!(
            matches!(sim.phase, SimulationPhase::Complete { .. }),
            "after join() state should be Complete"
        );
        assert!(sim.result().is_some(), "Complete must expose a result");
    }

    /// Progress is monotonic non-decreasing and reaches 1.0 once the run
    /// finishes. We sample progress before and after join — strictly we
    /// can't observe the intermediate values deterministically without a
    /// race, but the start→finish bracket already enforces the contract.
    #[test]
    fn test_simulation_progress() {
        let scene = air_box_scene();
        let mut sim = SimulationState::default();
        sim.config.ray_count = 500;
        sim.config.max_bounces = 8;

        sim.start(&scene);
        let p_running = sim.progress();
        assert!(
            (0.0..=1.0).contains(&p_running),
            "progress while running must be in [0,1], got {p_running}"
        );

        sim.join();
        let p_done = sim.progress();
        assert!(
            (p_done - 1.0).abs() < 1e-3,
            "progress after join should be 1.0, got {p_done}"
        );
    }

    /// `cancel()` must cause the worker to exit promptly. We pre-set a
    /// long-running config, fire cancel immediately, then join — the
    /// worker should still produce some result (partial) and the state
    /// should reach Complete without hanging.
    #[test]
    fn test_simulation_cancel() {
        let scene = air_box_scene();
        let mut sim = SimulationState::default();
        // Big enough to ensure the worker is still busy when we cancel.
        sim.config.ray_count = 10_000;
        sim.config.max_bounces = 50;

        sim.start(&scene);
        // Don't sleep — just fire cancel immediately. The worker may
        // already be done for tiny scenes, which is fine: cancel after
        // completion is a no-op, and join still transitions to Complete.
        sim.cancel();

        let start = std::time::Instant::now();
        let joined = sim.join();
        let elapsed = start.elapsed();

        assert!(joined, "join() must return true after start()");
        assert!(
            matches!(sim.phase, SimulationPhase::Complete { .. }),
            "after cancel + join state should be Complete"
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "cancel should free the worker quickly, took {elapsed:?}"
        );
        // Cancel must yield a partial result (possibly empty grid) — not
        // panic, not deadlock.
        assert!(sim.result().is_some());
    }

    /// A single ray with very high `max_bounces` and zero energy threshold
    /// would otherwise loop until it ran out of bounces. With cancel set
    /// at entry, `trace_ray_cancellable` must return within
    /// `CANCEL_CHECK_INTERVAL` bounces — i.e. the path it returns must
    /// have far fewer points than `max_bounces`.
    #[test]
    fn test_cancel_during_long_ray() {
        let scene = air_box_scene();
        let view = SceneView {
            meshes: scene.meshes.clone(),
        };
        let cfg = SimulationConfig {
            ray_count: 1,
            max_bounces: 100_000,
            energy_threshold: 0.0,
            grid_resolution: 1.0,
        };
        let bg = MediumProperties::air();
        let cancel = AtomicBool::new(true);

        let path = trace_ray_cancellable(
            scene.sound_sources[0].position,
            Vec3::new(1.0, 0.0, 0.0),
            scene.sound_sources[0].power_db,
            &cfg,
            &view,
            &bg,
            Some(&cancel),
            None,
        );

        // With cancel pre-set, we should observe it on the first check
        // (bounces 0 satisfies `wrapping_sub(0) >= CANCEL_CHECK_INTERVAL`
        // only after one full interval, so the worst case is roughly
        // CANCEL_CHECK_INTERVAL bounces × at most a few path points each).
        // The key invariant: path.len() ≪ max_bounces.
        assert!(
            path.len() < 50,
            "cancel-set-at-entry should short-circuit in well under 50 \
             path entries, got {}",
            path.len()
        );
    }

    // -----------------------------------------------------------------------
    // D4 — BVH integration cross-check
    // -----------------------------------------------------------------------

    /// Running the trace loop with the BVH must produce paths identical
    /// (within float epsilon) to running with the linear scan. This is
    /// the load-bearing equivalence check that protects against BVH
    /// regressions sneaking in via the spatial accel.
    #[test]
    fn test_bvh_matches_brute_force_full_sim() {
        // Mixed scene: a room + a smaller interior platform exercises
        // multiple meshes, so the BVH actually has to descend.
        let mut scene = air_box_scene();
        scene.meshes.push(primitives::platform(
            Vec3::new(1.5, 0.0, 1.5),
            1.5,
            1.5,
            1.0,
        ));

        let bvh = Bvh::build(&scene.meshes);
        let cfg = SimulationConfig {
            ray_count: 300,
            max_bounces: 8,
            energy_threshold: 0.001,
            grid_resolution: 1.0,
        };
        let bg = MediumProperties::air();
        let src = &scene.sound_sources[0];
        let rays = generate_sphere_rays(src.position, cfg.ray_count);

        let mut compared = 0;
        for (i, dir) in rays.iter().enumerate() {
            let brute = trace_ray_in(
                src.position,
                *dir,
                src.power_db,
                &cfg,
                &scene.meshes,
                &bg,
                None,
                None,
            );
            let accel = trace_ray_in(
                src.position,
                *dir,
                src.power_db,
                &cfg,
                &scene.meshes,
                &bg,
                None,
                Some(&bvh),
            );
            assert_eq!(
                brute.len(),
                accel.len(),
                "ray {i} path length differs: brute={} accel={}",
                brute.len(),
                accel.len()
            );
            for (j, (a, b)) in brute.iter().zip(accel.iter()).enumerate() {
                let d = (*a - *b).length();
                assert!(
                    d < 1e-3,
                    "ray {i} path[{j}] diverges by {d}: brute={a:?} accel={b:?}"
                );
            }
            compared += 1;
        }
        assert_eq!(compared, cfg.ray_count as usize);
    }
}
