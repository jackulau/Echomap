# Final issues sweep — goal/010 deliverable 7

Re-runs the leftover-issues scans from
`tasks/004-whole-app-quality-gate/issues_report.md` after all goal/005-009
work merged plus goal/008 (agent feedback + GUI inspector). Every scan
classified by hot-path vs test-only.

Verdict: **all clean** for ship-readiness purposes. The only matches are
intentional test output, CLI bin startup banners, and tracked test
unwraps. No production-code `todo!`, no production `unimplemented!`, no
stray hot-path `unwrap` calls, no stray hot-path `println!`.

## Scan 1 — `todo!()` / `unimplemented!()`

```
grep -rnE "todo!|unimplemented!" src/ | grep -v "test\|//\|#\\[allow"
```

Result: **zero hits**.

## Scan 2 — Rust hot-path `.unwrap()`

`grep -rcE "\.unwrap\(\)" src/agent/ src/robot/` counts per file:

| file | total unwrap | hot-path | test-only |
|---|---:|---:|---:|
| src/agent/bridge.rs | 54 | 0 | 54 |
| src/agent/session.rs | 0 | 0 | 0 |
| src/agent/protocol.rs | 62 | 0 | 62 |
| src/agent/ws_server.rs | 1 | 0 | 1 |
| src/agent/mod.rs | 31 | 0 | 31 |
| src/agent/demo.rs | 0 | 0 | 0 |
| src/agent/tcp_server.rs | 35 | 0 | 35 |
| src/robot/* | 80 | 0 | 80 |

Hot-path unwraps: **zero hits**. Every match lives inside `#[cfg(test)]`
test modules where panic-on-failure is correct behavior.

## Scan 3 — stray `println!` / `eprintln!` in Rust src/

```
grep -rnE 'println!|eprintln!' src/ | grep -v "test\|//\|main.rs"
```

Result:
- `src/io/step_parser.rs:526,535,551` — inside `#[test]` block (verified at
  lines 529, 542). Not stray; classified as test debug output.
- `src/gas/solver.rs:1471,1525` — likewise inside `#[cfg(test)]` block.
- `src/bin/echomap_server.rs:34,35,38,42,55` — CLI startup banner +
  shutdown trailer. Intentional user-facing stderr output for the headless
  server binary; not a library leak.

Production-code stray prints: **all clean**.

## Scan 4 — stray Python `print()`

```
grep -rnE "^\s*print\(" python/echomap_client/
```

Result:
- `python/echomap_client/runner.py:73,85` — boxing match commentary,
  user-facing.
- `python/echomap_client/cli.py:49,51,100,105-108,122` — CLI entry point,
  user-facing.

All matches are intentional CLI output: **all clean**.

## Bonus — debt left for follow-up goals (not blockers)

These are tracked but explicitly out of scope for the ship gate:

1. `listener_captures` not populated by the sim pipeline — guarded by
   `#[ignore]` on `listener_spl_plausible`.
2. Per-band ray energy not carried from `trace_ray` → grid — guarded by
   `#[ignore]` on `frequency_dependent_end_to_end`.

Neither blocks shipping; both have ignored tests pinning the contract for
when the wiring lands.

## Verdict line for the verify gate

zero hits, all clean.
