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
| fluid/step_16cubed                          | time   | 4.10 ms    | 16³ grid, full advect→project step |
| gas/step_16cubed                            | time   | 1.06 ms    | full step incl. advect + diffuse   |
| gas/diffuse_concentrations_16cubed          | time   | 135 µs     | diffuse-only                       |
| dynamics/step_5dof                          | time   | 48.6 ns    | 5-joint simple arm                 |
| collision/collect_link_aabbs_3links         | time   | 30.0 ns    | broad-phase AABB collection        |
| collision/aabb_overlap                      | time   | 1.12 ns    | single overlap predicate           |
| collision/detect_robot_collisions_2bots     | time   | 223 ns     | broad+narrow phase, 2 robots       |
| acoustics/ray_triangle_intersect            | time   | 6.64 ns    | Möller–Trumbore                    |
| acoustics/ray_refract_air_water             | time   | 31.7 ns    | Fresnel + Snell                    |
| acoustics_box_room/brute_force/1k           | time   | ≤50 ms     | 5×5×3 m, 1 000 rays, brute scan    |
| acoustics_box_room/bvh/1k                   | time   | ≤30 ms     | same scene, BVH path               |
| acoustics_studio/brute_force/10k            | time   | record†    | studio.step, 10 000 rays, baseline |
| acoustics_studio/bvh/10k                    | time   | ≥5x speedup vs studio brute_force baseline | BVH spatial accel target |

† Captured fresh per host on first `cargo bench --bench acoustics`. The
brute_force/10k number is the rolling baseline that the BVH bench is
required to beat by ≥5x — see `acoustics_studio/bvh/10k` row above.

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
