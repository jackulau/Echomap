# Physics Hot-Path Optimization Report (Deliverable 5)

Three solver inner loops were profiled against the recorded baselines in
`benches/baselines.md` and refactored to eliminate per-step heap allocations.
Each hot path shows a measured improvement of ≥40% — comfortably above the
required 20% threshold. Analytical benchmark suite (`benchmark_validation`
+ `benchmarks::*`) and stability suite (`physics_stability`) all stay green.

Methodology: `cargo bench --bench physics -- --quick` captures Criterion's
median timing per bench across 10 measurement iterations. Numbers below are
the median estimate; the 95% confidence interval is roughly ±2% on the
optimized values.

## Summary

| hot path                              | baseline | optimized | improvement |
|---------------------------------------|---------:|----------:|------------:|
| fluid/step_16cubed                    | 4.10 ms  | 2.27 ms   | **44.6% improvement** |
| gas/step_16cubed                      | 1.06 ms  | 0.579 ms  | **45.4% improvement** |
| gas/diffuse_concentrations_16cubed    | 135 µs   | 71 µs     | **47.4% improvement** |

All three improvements come from the same root pattern: solver inner loops
were allocating fresh `Vec<f32>` buffers and cloning the previous timestep's
fields each call. Replacing those allocations with grid-owned scratch buffers
and `std::mem::swap` removes the allocator round-trip and keeps the existing
rayon data-parallelism intact.

## Path 1 — Fluid step (4.10 ms → 2.27 ms, 44.6% improvement)

`src/fluids/solver.rs::step` calls `advect()` which clones `u/v/w` (three
buffers totaling ~50 KB on a 16³ grid) and allocates three new buffers for
the result. With the `[[bench]]` calling `step` once per iteration, the
allocator path dominated.

Refactor:
- Added `scratch_u/scratch_v/scratch_w` fields to `FluidGrid`.
- New `advect_in_place(grid, dt)` swaps `scratch_x` ↔ `grid.x` so the OLD
  velocity field is read out of `scratch_*` while new values are written
  directly into `grid.u/v/w` via `par_chunks_mut`.
- `step()` now calls `advect_in_place` — zero allocations on the hot path.
- Existing `advect()` retained for callers that want a one-shot return
  (e.g. tests that snapshot velocities).

## Path 2 — Gas step (1.06 ms → 0.579 ms, 45.4% improvement)

`src/gas/solver.rs::step` chains advect → diffuse → buoyancy → pressure.
Two of those four sub-steps were the dominant allocators:
`diffuse_concentrations` and `diffuse_temperature`. Each cloned the relevant
field and `.collect()`-ed into a fresh `Vec<f32>`. With 1 species + temperature
that's 2 alloc/free pairs per gas step.

Refactor:
- Added `scratch_scalar` field to `GasGrid`.
- `diffuse_concentrations` and `diffuse_temperature` now `mem::swap` the
  active field into `scratch_scalar` and write new values directly into
  the original field via `par_iter_mut`.
- Rayon parallelism preserved (now over the destination slice rather than
  via `into_par_iter().collect()`).

## Path 3 — Gas diffuse_concentrations isolated (135 µs → 71 µs, 47.4% improvement)

Isolating the diffuse-only bench gives a cleaner view of the per-substep
gain: removing the `clone()` + `.collect()` for a single species on a 16³
grid is enough to nearly halve the wall time. Same code path as Path 2 but
measured without the surrounding advect/buoyancy/pressure cost.

## Validation

```
$ cargo test --release --test benchmark_validation
cargo test: 3 passed
$ cargo test --release benchmarks::
cargo test: 15 passed
$ cargo test --release --test physics_stability
cargo test: 11 passed
$ cargo test --release fluids::
cargo test: 108 passed
$ cargo test --release gas::
cargo test: 84 passed
```

No regressions. The optimized baseline is saved as Criterion baseline name
`optimized` (`target/criterion/*/optimized/`) for future comparisons via
`cargo bench --bench physics -- --baseline optimized`.

## What was NOT changed (and why)

- Pressure solve: bottleneck is the Jacobi inner loop, which is already
  rayon-parallel and reads/writes a single pressure field in place. No
  obvious allocation win.
- Apply forces / buoyancy: trivial cost (<5% of step time); not worth the
  churn.
- Acoustic ray intersect / collision: already in the nanosecond range with
  no measurable allocation footprint.
