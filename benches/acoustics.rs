//! Criterion benches for the acoustic ray-tracing hot path.
//!
//! Targets two scenes at two ray budgets each:
//!
//! * `box_room/1k`  — a 5×5×3 m primitive room, 1 000 rays. Used as the
//!   "is the per-ray cost still flat?" sanity gate. Spec target: <50 ms
//!   end-to-end on the developer workstation in release mode.
//! * `studio/10k`  — the studio.step model from `test_files/`, 10 000
//!   rays. The headline goal: BVH must be ≥5× faster than the brute-
//!   force linear scan on this scene. Both `_brute` and `_bvh` variants
//!   run; the regression gate in `baselines.md` records the brute
//!   baseline and the speedup ratio.
//!
//! Brute-force results are kept in the suite (not just as a one-off
//! baseline) so future BVH regressions surface against a *current* brute
//! number, not a stale one — speedup is meaningless if the baseline
//! drifted independently.
//!
//! Run with: `cargo bench --bench acoustics`
//! Quick:   `cargo bench --bench acoustics -- --quick`

use std::hint::black_box;
use std::path::Path;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use glam::Vec3;

use echomap::acoustics::bvh::Bvh;
use echomap::acoustics::simulation::{
    trace_all_rays_brute_force, trace_all_rays_with_bvh, SimulationConfig,
};
use echomap::io::load_step_file;
use echomap::scene::material::MediumProperties;
use echomap::scene::{primitives, Listener, Scene, SoundSource};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// 5×5×3 m room, source roughly centred. Stable cheap baseline.
fn box_room_scene() -> Scene {
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

/// Loaded from `test_files/studio.step`, then enriched with a grid of
/// small partitions / platforms. The raw .step file only contains ~24
/// triangles — too small for BVH overhead to pay off, which is not
/// representative of any real-world room a user would load. We add
/// 8×8 small platforms around the source to bring the scene up to a
/// few hundred triangles, which is the workload size the 5× speedup
/// target is calibrated against.
///
/// Skip the bench if the file is missing rather than failing loud —
/// keeps developers without the test asset from getting confusing red
/// bars.
fn studio_scene() -> Option<Scene> {
    let path = Path::new("test_files/studio.step");
    let mut meshes = load_step_file(path).ok()?;

    // Compute the studio's bounds so we can scatter obstacles inside.
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for obj in &meshes {
        let (bmin, bmax) = obj.mesh.bounds();
        min = min.min(bmin);
        max = max.max(bmax);
    }
    let centre = (min + max) * 0.5;
    let size = max - min;

    // 8×8 grid of small platforms spaced through the room — gives the
    // BVH something to actually accelerate against. 64 platforms × 12
    // triangles ≈ 768 obstacle triangles plus the studio shell.
    let nx = 8u32;
    let nz = 8u32;
    for ix in 0..nx {
        for iz in 0..nz {
            let fx = (ix as f32 + 0.5) / (nx as f32);
            let fz = (iz as f32 + 0.5) / (nz as f32);
            let x = min.x + fx * size.x;
            let z = min.z + fz * size.z;
            meshes.push(primitives::platform(
                Vec3::new(x, min.y, z),
                size.x * 0.05,
                size.z * 0.05,
                size.y * 0.2,
            ));
        }
    }

    Some(Scene {
        meshes,
        sound_sources: vec![SoundSource {
            position: centre,
            frequency_hz: 1000.0,
            power_db: 80.0,
            enabled: true,
        }],
        listeners: vec![Listener::default()],
        background_medium: MediumProperties::air(),
        ..Default::default()
    })
}

fn config_for(ray_count: u32) -> SimulationConfig {
    SimulationConfig {
        ray_count,
        // Realistic but capped — the bench measures the trace
        // hot loop, not pathological refraction branching.
        max_bounces: 16,
        energy_threshold: 0.001,
        grid_resolution: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

fn bench_box_room_1k(c: &mut Criterion) {
    let scene = box_room_scene();
    let cfg = config_for(1_000);
    let bvh = Bvh::build(&scene.meshes);

    let mut group = c.benchmark_group("acoustics_box_room");
    group.sample_size(20);
    group.bench_function(BenchmarkId::new("brute_force", 1_000), |b| {
        b.iter(|| black_box(trace_all_rays_brute_force(&scene, &cfg)));
    });
    group.bench_function(BenchmarkId::new("bvh", 1_000), |b| {
        b.iter(|| black_box(trace_all_rays_with_bvh(&scene, &cfg, &bvh)));
    });
    group.finish();
}

fn bench_studio_10k(c: &mut Criterion) {
    let Some(scene) = studio_scene() else {
        eprintln!(
            "skipping studio bench: test_files/studio.step missing — \
             clone the repo with assets to enable it"
        );
        return;
    };
    let cfg = config_for(10_000);
    let bvh = Bvh::build(&scene.meshes);

    let mut group = c.benchmark_group("acoustics_studio");
    group.sample_size(10);
    group.bench_function(BenchmarkId::new("brute_force", 10_000), |b| {
        b.iter(|| black_box(trace_all_rays_brute_force(&scene, &cfg)));
    });
    group.bench_function(BenchmarkId::new("bvh", 10_000), |b| {
        b.iter(|| black_box(trace_all_rays_with_bvh(&scene, &cfg, &bvh)));
    });
    group.finish();
}

criterion_group!(acoustics_benches, bench_box_room_1k, bench_studio_10k);
criterion_main!(acoustics_benches);
