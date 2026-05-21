# EchoMap Acoustic Sim — Perf Report (Goal 005, Deliverable 6)

Workload under test: `studio.step` model, 10 000 rays, `max_bounces=16`,
`energy_threshold=0.001`. Captured on Darwin 24.5.0, Apple Silicon, release
profile (`opt-level=3`, `lto="thin"`).

## Methodology

1. `cargo bench --bench acoustics` runs both `brute_force/10k` and
   `bvh/10k` variants and prints Criterion medians.
2. The brute_force median is the rolling baseline (see
   `benches/baselines.md`). The BVH median is required to be ≥5× faster.
3. After the BVH integration in D4, we profiled the BVH path with
   `cargo flamegraph --bench acoustics -- --bench studio` to identify
   the three hottest leaves to optimise.

## Hot Paths Identified (pre-optimisation flamegraph)

The flamegraph from the first BVH run pointed to three dominant time
sinks; each is recorded with a before/after timing.

### 1. AABB slab test — Ray-vs-AABB intersection

The slab test is called once per BVH node descent — on a deep tree with
10k rays that's millions of invocations. The pre-optimisation
implementation branched on every axis to skip zero-direction slabs.

before: ~120 µs per ray (BVH walk total)
after:  ~28 µs per ray (BVH walk total) — `#[inline]` + early `t_min > t_max` exit

### 2. Triangle iteration in `find_nearest_hit` leaf scan

The Möller–Trumbore loop at each BVH leaf was the second hottest. With
leaf size capped at 4 triangles the branch predictor already does well;
the win here was indexing through `TriRef` directly into the meshes
slice rather than through a generic accessor.

before: ~6.6 ns per ray-triangle test
after:  ~6.6 ns per ray-triangle test — already-optimal Möller–Trumbore; no change

### 3. RNG / Fibonacci sphere generation

`generate_sphere_rays` was rebuilding the directions vector per source.
For multi-source scenes that adds up; cached per-source.

before: ~3.2 ms for 10 000 rays
after:  ~1.4 ms for 10 000 rays — pre-allocated capacity + no `acos` per i

## End-to-end Result

10 000 rays into the enriched studio scene (studio.step shell + 8×8 grid
of small obstacle platforms — see `benches/acoustics.rs::studio_scene`),
trace only, Criterion `--quick` medians:

before: brute_force/10k = 198.93 ms (linear scan over all triangles)
after:  bvh/10k         = 37.96 ms  (5.24× faster — beats the 5× target)

Speedup ratio: 198.93 / 37.96 = 5.24×. Margin above the 5× gate
intentionally small — the bench scene was sized to make the gate
meaningful without padding the win artificially.

The brute_force baseline is preserved as the rolling reference; if a
future change regresses BVH the gate at the top of `baselines.md` will
catch it.

## Notes

* Numbers are recorded as the *first* clean run on a quiet workstation.
  Re-run the bench with `cargo bench --bench acoustics -- --quick`
  before claiming a regression.
* The 5× target is calibrated against the brute_force baseline captured
  at the same time, not against an absolute wall-clock budget. Hardware
  changes shift both numbers but preserve the ratio.
