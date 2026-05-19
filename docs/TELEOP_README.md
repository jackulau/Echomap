# Tele-op: Record and Replay

EchoMap can capture an external agent's drive of a robot to a JSONL trace and
deterministically replay it later. This is the primitive used to bootstrap
imitation-learning datasets, regression tests for the sim, and demo scripts.

## Quick start

End-to-end demo (record 100 sinusoidal steps, then replay against a fresh
robot and assert determinism):

```bash
bash scripts/demo_teleop_e2e.sh
# → writes tasks/011-robot-render-teleop-readiness/teleop_trace.jsonl
```

Headless smoke (sanity check the server and WS client without recording):

```bash
bash scripts/smoke_headless_e2e.sh
```

GUI smoke (`ECHOMAP_TEST_FRAMES=N` ticks N frames then exits):

```bash
ECHOMAP_TEST_FRAMES=60 cargo run --release --bin echomap
```

## Recording from your own client

The recorder is a pure client primitive in `src/teleop/recorder.rs`. Trace
lines are JSONL where each line is a serialized `TraceFrame`:

```json
{"ts": 0.05, "step": 1, "obs": <GymRobotState>, "act": <RobotAction>}
```

Driver helper:

```rust
use echomap::teleop::run_session;

let frames = run_session(
    "ws://127.0.0.1:9002",
    /* robot_id */ 0,
    /* steps    */ 200,
    |step, obs| my_policy(step, obs),  // returns RobotAction
    "trace.jsonl",
)
.await?;
```

`run_session` handles the handshake (Connect, initial Observe), drives the
step loop, records each (observation, action) pair, and cleanly closes the
session. The recorded trace can be loaded back via
`echomap::teleop::recorder::read_trace`.

## Replaying

```rust
use echomap::teleop::playback::{Player, DEFAULT_TOLERANCE};

let report = Player::replay("ws://127.0.0.1:9002", 0, "trace.jsonl", DEFAULT_TOLERANCE).await?;
assert!(report.passed(), "drift {} at frame {:?}", report.max_drift, report.diverged_at);
```

`Player::replay` issues `Reset` to put the robot back to its initial state,
replays every recorded action in order, and reports the maximum joint-position
drift plus the first frame (if any) where drift exceeded the tolerance.

## In-GUI tele-op (Ctrl+T)

Open `cargo run --release --bin echomap` and press **Ctrl+T**. The viewport
shows a red `TELE-OP` banner across the top and consumes WASD/QE:

| Key | Effect on `robot/0` |
| --- | --- |
| `W` / `S` | joint 0 motor velocity `+1.0` / `-1.0` |
| `A` / `D` | joint 1 motor velocity `-1.0` / `+1.0` |
| `Q` / `E` | joint 2 motor velocity `-1.0` / `+1.0` |
| (none)    | every motor returns to `0.0` |

Pressing Ctrl+T again exits tele-op and emits a final zero-velocity action so
motors stop on the next frame. The fly-camera handler is gated off while
tele-op is active so WASD goes to the robot, not the camera.

The in-GUI path applies actions directly through `RobotManager` so they show
up on screen instantly. The recorder hook lives on the *external* WS client
side (`run_session` above) and captures the format an external agent would
send — keep using `scripts/demo_teleop_e2e.sh` to produce the canonical
record-then-replay artifact.

## Mock-arm bridge

The Python side has a mock backend exercised via:

```bash
python3 demos/connect_real_arm.py --backend mock
```

It speaks the same WS protocol as `run_session`, so the recorder hook is
agnostic to whether the data came from a sim agent, a real arm, or the mock
shim. The `--backend serial` path remains a stub — only `--backend mock`
is wired end-to-end as of goal 011.

## Determinism notes

* The sim resets to a clean `RobotState::new(&definition)` on every `Reset`,
  so two runs with the same actions reproduce the same joint trajectories
  within numerical noise (the default tolerance is `1e-2`).
* The trace records what the **client** sent + received. If you need
  server-side ground truth (e.g. to debug a divergent step), compare
  `TraceFrame::obs.joint_positions` against `Player::replay`'s ReplayReport
  — `diverged_at` will pinpoint the offending step.
* Trace files are JSON-lines: trivially diffable, grep-able, and you can
  inspect frame N with `sed -n "${N}p" trace.jsonl | jq .`.
