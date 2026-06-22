# Physics Microbenchmark Baselines

Reference timings captured on the developer workstation with `cargo bench
--bench physics -- --quick` on Darwin 24.5.0 (Apple Silicon, release profile,
LTO thin). Values are median times reported by Criterion; the regression gate
(`scripts/check_perf_regression.sh`) compares fresh bench output against these
and fails if any timing exceeds the baseline by more than 15%.

Update procedure: when an optimisation lands and the new median is durably
faster, re-run `--quick`, copy the new median into the table below, and commit
the updated baselines alongside the optimisation.

## Baselines

| bench                                       | metric | baseline   | notes                              |
|---------------------------------------------|--------|------------|------------------------------------|
| fluid/step_16cubed                          | time   | 5.30 ms    | 16³ grid, full advect→project step |
| gas/step_16cubed                            | time   | 1.06 ms    | full step incl. advect + diffuse   |
| gas/diffuse_concentrations_16cubed          | time   | 135 µs     | diffuse-only                       |
| dynamics/step_5dof                          | time   | 107 ns     | 5-joint arm, gravity + implicit damping + generalized inertia |
| collision/collect_link_aabbs_3links         | time   | 36.0 ns    | broad-phase AABB collection        |
| collision/aabb_overlap                      | time   | 1.31 ns    | single overlap predicate           |
| collision/detect_robot_collisions_2bots     | time   | 273 ns     | broad+narrow phase, 2 robots       |
| acoustics/ray_triangle_intersect            | time   | 7.70 ns    | Möller–Trumbore                    |
| acoustics/ray_refract_air_water             | time   | 41.0 ns    | Fresnel + Snell                    |
| acoustics_box_room/brute_force/1k           | time   | ≤50 ms     | 5×5×3 m, 1 000 rays, brute scan    |
| acoustics_box_room/bvh/1k                   | time   | ≤30 ms     | same scene, BVH path               |
| acoustics_studio/brute_force/10k            | time   | record†    | studio.step, 10 000 rays, baseline |
| acoustics_studio/bvh/10k                    | time   | ≥5x speedup vs studio brute_force baseline | BVH spatial accel target |

† Captured fresh per host on first `cargo bench --bench acoustics`. The
brute_force/10k number is the rolling baseline that the BVH bench is
required to beat by ≥5x — see `acoustics_studio/bvh/10k` row above.

Threshold/qualitative rows (`≤`, `≥`, `record†`) are enforced by the
acoustics bench harness itself, not by `check_perf_regression.sh` — the
script compares only plain `<number> <unit>` rows.

### Re-capture 2026-06-09

Numeric rows re-captured on the same workstation (median of three `--quick`
runs on an idle machine). Reasons for movement vs the 2026-05-17 capture:

- `dynamics/step_5dof` 48.6 → 107 ns: the step now does strictly more
  physics — gravity-loading torques, implicit joint damping, and the
  generalized-inertia model (goal 020). Intentional work increase, not a
  regression.
- `acoustics/ray_refract_air_water` 31.7 → 41.0 ns and
  `ray_triangle_intersect` 6.64 → 7.70 ns: `AcousticRay::path` moved from
  `Vec` to `VecDeque` (O(1) eviction + per-branch polylines, goal 021),
  which grows the struct by 8 bytes and shifts layout; remainder is
  toolchain/codegen drift since May.
- `fluid/step_16cubed` 4.10 → 5.30 ms, `collision/*` +16–22%: no functional
  change to these solvers since the original capture — drift from toolchain
  updates and machine state. Gate re-anchored so a further 15% slide from
  today's reality fails loudly.

## Regression Gate

`bash scripts/check_perf_regression.sh` runs `cargo bench --bench physics`,
parses Criterion's median line for each bench above, normalizes units to
seconds, and exits non-zero if any current median is more than 15% slower
than the recorded baseline.

`bash scripts/check_perf_regression.sh --dry-run` skips the bench run and
only validates that this file parses cleanly. CI hooks should call the
non-dry form on the release branch only — full bench output is noisy in PR CI.

## Targets (60 Hz budget context)

A full integrated physics step at 60 Hz has a 16.67 ms budget. The fluid
and gas full-step benches together dominate this budget on a 16³ grid; the
integrated test in `tests/integrated_perf.rs` exercises the actual scenario
grids and is the source of truth for end-to-end perf health.

## Render hot path (non-gated reference)

Captured with `cargo bench --bench render` on the same workstation (Apple
Silicon, release + LTO thin). These are REFERENCE medians for the
painter-free render per-element operations the 2D-viewport batching work
optimised — `project_3d` (world→screen projection, called once per vertex in
the wireframe / slice / ray-overlay / energy-grid draw loops) and
`energy_to_color` (the acoustic-energy heatmap colour map).

They are deliberately NOT enforced by `check_perf_regression.sh` (which runs
only `--bench physics`): render correctness/throughput is already guarded by
the `renderer_screenshots` / `renderer_smoke` visual-identity tests, and
gating a pure microbench would only widen the perf gate's false-positive
surface. The bench-name cells below intentionally omit `/` so the gate's
table parser ignores them — the real Criterion id is in the notes column.
This bench exists to *measure* per-vertex/per-cell cost and catch gross drift
when someone touches the projection or colour-map paths.

| op                              | metric | reference | notes                                                              |
|---------------------------------|--------|-----------|--------------------------------------------------------------------|
| render project_3d 32cubed verts | time   | 264 µs    | 32 768 projections; id `render/project_3d_32cubed` (~8 ns/vertex, ~124 Melem/s) |
| render energy_to_color 32cubed  | time   | 49.7 µs   | 32 768 colour maps; id `render/energy_to_color_32cubed` (~1.5 ns/cell, ~659 Melem/s) |

CV on capture: project_3d ~1.7%, energy_to_color ~0.15% (idle machine) — both
well under the 5% the gated physics benches target, confirming these are
low-variance regression *measurements*.
