//! Criterion microbenchmarks for the render hot path (painter-free).
//!
//! Targets the per-element operations the 2D-viewport batching work
//! (wireframe / slice / ray-overlay / energy-grid) invokes once per vertex or
//! per cell every frame:
//!   * `project_3d`      — 3D world -> 2D screen projection (every draw loop)
//!   * `energy_to_color` — acoustic-energy heatmap colour map (energy grid)
//!
//! Both are pure, deterministic, and allocation-free in the measured closure,
//! so they make a stable regression *measurement* without an egui context.
//!
//! NOTE: this is a developer measurement harness. It is intentionally NOT
//! wired into `scripts/check_perf_regression.sh` (which gates only
//! `--bench physics`) — render correctness/throughput is already guarded by
//! the `renderer_screenshots` / `renderer_smoke` visual-identity tests, and
//! adding a gated microbench would only widen the perf gate's false-positive
//! surface. Reference medians live in `benches/baselines.md`.
//!
//! Run with:        cargo bench --bench render
//! Save baseline:   cargo bench --bench render -- --save-baseline main
//! Compare:         cargo bench --bench render -- --baseline main

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use egui::Pos2;
use glam::Vec3;

use echomap::renderer::{energy_to_color, project_3d, Camera};

/// Element count for both benches — `32³` matches a dense voxel/grid pass, the
/// heaviest per-frame vertex/cell set the batched draw loops project & colour.
const N_SIDE: usize = 32;
const N_ELEMS: usize = N_SIDE * N_SIDE * N_SIDE; // 32_768

/// Build a deterministic `N_SIDE³`-point lattice spanning an 8m cube around the
/// origin — representative of the grid/wireframe vertex set fed to `project_3d`.
fn lattice() -> Vec<Vec3> {
    let mut pts = Vec::with_capacity(N_ELEMS);
    let step = 1.0 / N_SIDE as f32;
    for i in 0..N_SIDE {
        for j in 0..N_SIDE {
            for k in 0..N_SIDE {
                pts.push(Vec3::new(
                    (i as f32 * step - 0.5) * 8.0,
                    (j as f32 * step - 0.5) * 8.0,
                    (k as f32 * step - 0.5) * 8.0,
                ));
            }
        }
    }
    pts
}

/// Projection sweep: project a full dense vertex set once, as a draw loop does.
fn bench_project_3d(c: &mut Criterion) {
    let camera = Camera::default();
    let center = Pos2::new(640.0, 360.0);
    let scale = 50.0;
    let pts = lattice();

    let mut group = c.benchmark_group("render");
    group.throughput(Throughput::Elements(pts.len() as u64));
    group.bench_function("project_3d_32cubed", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &p in &pts {
                let s = project_3d(black_box(p), &camera, center, scale);
                acc += s.x + s.y;
            }
            black_box(acc)
        })
    });
    group.finish();
}

/// Heatmap colour sweep: map a populated energy grid's worth of values, as the
/// (batched) energy-grid overlay does each frame it is shown.
fn bench_energy_to_color(c: &mut Criterion) {
    let max_energy = 10.0f32;
    let energies: Vec<f32> = (0..N_ELEMS)
        .map(|i| (i as f32) / (N_ELEMS as f32) * max_energy)
        .collect();

    let mut group = c.benchmark_group("render");
    group.throughput(Throughput::Elements(energies.len() as u64));
    group.bench_function("energy_to_color_32cubed", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &e in &energies {
                let col = energy_to_color(black_box(e), max_energy);
                acc = acc.wrapping_add(col.r() as u32 + col.g() as u32 + col.b() as u32);
            }
            black_box(acc)
        })
    });
    group.finish();
}

criterion_group!(render_benches, bench_project_3d, bench_energy_to_color);
criterion_main!(render_benches);
