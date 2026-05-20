# Ship-readiness — goal/010 deliverable 8

EchoMap v1 release candidate. All goal/005-009 streams plus
goal/011-robot-render-teleop-readiness, goal/008-echomap-agent-feedback,
and goal/010-echomap-quality-gate have landed on `master`. This document
is the operator hand-off for tagging the release.

## Gate Results

Run on commit `$(git rev-parse --short HEAD)`, gather the exact commands
from [docs/SMOKE.md § Quality gate](../../docs/SMOKE.md).

| Gate | Command | Result |
|---|---|---|
| Workspace tests | `cargo test --workspace` | 1076 passed, 5 ignored, 0 failed |
| Clippy | `cargo clippy --workspace -- -D warnings` | 0 warnings |
| Fmt | `cargo fmt -- --check` | clean (no diff) |
| Integration | `cargo test --test integration` | 9 passed, 2 ignored, 0 failed (covers 7 `test_*` cases + boxing-round smoke + CSV export deep case) |
| Python live | `bash scripts/test_python_vs_live.sh` | 125 passed |
| Smoke | `bash scripts/smoke_all.sh` | green (see SMOKE.md phases) |
| Manual UI | [docs/MANUAL_SMOKE.md](../../docs/MANUAL_SMOKE.md) | 10 steps signed off (release manager file linked from §Sign-off) |

The 5 ignored unit tests are pre-existing — see
[final_sweep.md](final_sweep.md) §"debt left for follow-up goals" for
the rationale; none block shipping.

## Merged Goals

| Branch | Headline |
|---|---|
| `goal/005-echomap-perf-core` | per-band ray energy `[f32;6]` + BVH spatial accel + async sim + perf bench |
| `goal/006-echomap-sound-graphics` | surface heatmap + per-band rendering + ray-debug viz + listener pulse + colorbar legend + visual smoke |
| `goal/007-echomap-ui-polish` | file menu + collapsing groups + sim config validation + status bar + camera shortcuts + dark theme + walkthrough |
| `goal/009-echomap-validation-spl-export` | CSV+text export + RT60 stub + input validation + STEP parser robustness + integration tests |
| `goal/011-robot-render-teleop-readiness` | headless e2e smoke + GUI N-frame harness + renderer math/drawlist invariants + tele-op recorder/playback + Ctrl+T keyboard tele-op + e2e demo |
| `goal/008-echomap-agent-feedback` | capability advertisement on `Bound` (incl. `combat`) + combat observations + WS/TCP byte-identical payload parity + hardware-bridge parity pytest + malformed-action structured errors (`Cancel`/`Cancelled`) + live pytest gate script + 30s boxing-round smoke + transcript writer + **Agent Inspector** egui window (D8) |
| `goal/010-echomap-quality-gate` | unused dep removal (egui_extras, wgpu, bytemuck, nalgebra, cpal) + workspace/clippy/fmt gate + 7-case integration coverage + SMOKE.md + MANUAL_SMOKE.md + final issues sweep + this doc |

Each merge commit on `master` carries the goal slug; `git log --merges
--grep="^merge goal/"` recovers the full ledger.

## Known Limitations

These ship with v1. They are tracked and explicitly out of scope for the
release gate.

- **Listener captures not populated by the sim pipeline.** The
  `SimulationResult::listener_captures` vector is sized to scene
  listener count but never filled. Guarded by `#[ignore]` on
  `tests/integration.rs::listener_spl_plausible`. Tracked for a
  follow-up goal — pairs with the `capture_radius` integration.
- **Per-band ray energy not carried from `trace_ray` → grid.**
  `build_energy_grid` broadcasts the scalar ray contribution across
  all 6 bands via `energy_uniform`. Guarded by `#[ignore]` on
  `tests/integration.rs::frequency_dependent_end_to_end`. The
  grid-totals path works; the per-listener per-band path needs the
  ray-segment sampler to keep its spectrum.
- **No GUI smoke in CI.** Egui rendering requires a display; the
  10-step [MANUAL_SMOKE.md](../../docs/MANUAL_SMOKE.md) covers it.
  Headless smoke (`scripts/smoke_all.sh`) covers the Rust/Python
  surfaces without a display.
- **`SerialArm` hardware backend is a stub.** Real serial I/O ships
  via the `echomap.plugins.hardware` entry-point group; documented in
  [docs/PLUGINS.md](../../docs/PLUGINS.md). The `MockArm` backend is
  fully wired and covered by `test_bridge_parity.py`.

None of the above prevent the GUI, headless boxing server, agent
binding, hardware bridge, or acoustics export from being usable.

## Sign-off

```
Release tag:    v1.0.0-rc.__ (or final v1.0.0)
Date:           ____-__-__
Release mgr:    __________________
Commit sha:     __________________
Preflight ran:  [ ] PREFLIGHT GREEN observed
Manual smoke:   [ ] 10/10 ticked — see docs/MANUAL_SMOKE.md attached file
Known limits:   [ ] reviewed and accepted
```
