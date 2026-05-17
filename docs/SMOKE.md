# Smoke test — `scripts/smoke_all.sh`

End-to-end sanity that the whole EchoMap stack (Rust release server +
Python client + plugin loader + hardware bridge + boxing agent loop) is
alive after a change. Use it as the pre-ship gate; per-subsystem suites
(`cargo test`, `pytest python/tests/`, `cargo bench`) cover correctness
in detail.

## Run

```bash
bash scripts/smoke_all.sh
```

Optional overrides:

| env var                 | default | meaning                                          |
|-------------------------|---------|--------------------------------------------------|
| `SMOKE_PORT`            | `9117`  | WS port the headless server binds to             |
| `SMOKE_ROUND_DURATION`  | `8`     | seconds per round (1 round) — short for liveness |

Exit code: `0` on success, non-zero on the first failing phase. Per-phase
logs are written to a temp dir whose path is printed on exit; inspect them
when a phase fails.

## Phases

1. **Build server.** Skips the build if `target/release/echomap_server`
   already exists; otherwise runs `cargo build --release --bin echomap_server`.
2. **Plugin loader.** `python3 -m echomap_client.cli list-plugins` —
   prints registered entry-point groups and the example plugin (if
   `pip install -e python/examples/echomap_plugin_example` ran).
3. **Hardware bridge.** `demos/connect_real_arm.py --backend mock --steps 20`
   drives a 6-DOF `MockArm` through the sinusoidal agent loop.
4. **Live boxing.** Boots the release server with `NUM_ROUNDS=1` and
   `ROUND_DURATION=8` on `SMOKE_PORT`, runs
   `demos/connect_boxing_agents.py --mode heuristic` against it, and
   asserts the demo prints a `Final Score:` line. A 45s `timeout` is the
   ceiling. The server is killed on success, failure, or interrupt.

## What is *not* covered

- Egui GUI rendering (no display available in CI).
- Ollama / Claude LLM agent paths (require external API keys and a running
  model server). Heuristic agent stands in for liveness.
- Real serial hardware (the `SerialArm` backend frames packets but does
  not open the port — vendor drivers ship as plugins).

## When the smoke fails

Open the printed log directory:

```
==> logs left in /tmp/echomap-smoke-XXXXXX
```

Files: `plugins.log`, `hardware.log`, `server.log`, `boxing.log`. The
failing phase prints its tail to stdout before exit; the full log lives
in that directory.
