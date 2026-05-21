# Agent Control Interface

TCP/WebSocket server exposing robot sensor readings and accepting motor commands, with a gym-compatible Python SDK for ML agent integration.

## Slug

`agent-control-interface`

## Context

This is deliverable 6 of the EchoMap multi-physics simulation platform. The robot entity system (D5) provides articulated rigid-body robots with joints, sensors, actuators, and gym-compatible state types (GymRobotState, ObservationSpace, ActionSpace, RobotAction) — all already JSON-serializable via serde. This deliverable adds the network transport layer so external agents (Python ML scripts, RL frameworks) can observe and control robots over TCP/WebSocket.

**Prerequisite**: The `bouncy/spec-robot-entity-system` branch must be merged to main before execution, or the worktree must branch from that branch.

## Test Infrastructure

- **Framework**: Rust built-in `#[test]` with `#[cfg(test)] mod tests`
- **Test command**: `cargo test`
- **Lint**: `cargo clippy --all-targets`
- **Type check**: `cargo check`
- **Format**: `cargo fmt --check`
- **Convention**: Inline test modules at bottom of each source file
- **Python tests**: `python3 -m py_compile` for syntax validation

## Requirements

1. TCP server accepting line-delimited JSON connections on a configurable port (default 9001)
2. WebSocket server on a separate port (default 9002) using the same protocol
3. JSON protocol with message types: Connect, Reset, Step, Observe, Close, Error
4. Step-locked execution model: agent sends action → server steps simulation → server returns observation
5. Multiple concurrent agent connections, each assigned to a different robot
6. Observation responses include GymRobotState (joint positions/velocities, sensor readings, gripper states)
7. Action requests accept RobotAction (motor velocities, gripper commands)
8. Server runs on a background tokio thread, communicating with the eframe main loop via channels
9. UI display of server status (running/stopped, connected agents, port numbers)
10. Python SDK client with gym-compatible interface (reset, step, close, observation_space, action_space)

## Design Decisions

### 1. Separate tokio thread for server
**Rationale**: eframe::run_native() blocks the main thread with the GUI event loop. The server needs non-blocking async I/O. Running tokio on a dedicated thread with mpsc channels to the main loop is the standard pattern for eframe + async integration. The alternative (integrating tokio into the eframe loop) is fragile and non-standard.

### 2. Step-locked protocol (not free-running)
**Rationale**: Gym-compatible RL environments use a step-locked model: the agent calls step(action) and blocks until the next observation is ready. This is simpler, deterministic, and compatible with all major RL frameworks. A free-running server (where the sim runs at its own rate) would require complex synchronization and produce non-deterministic training.

### 3. TCP + WebSocket with shared protocol
**Rationale**: TCP with line-delimited JSON is lowest-latency for Python/C++ agents. WebSocket adds browser compatibility and better framing. Both use identical message types from protocol.rs, differing only in transport framing. TCP uses `\n` delimiters; WebSocket uses native message boundaries.

### 4. Channel-based bridge (not Arc<Mutex<RobotManager>>)
**Rationale**: Direct mutex sharing between the server thread and eframe thread risks deadlocks and priority inversion. Instead, use mpsc channels: server sends SimCommand to main loop, main loop processes commands during update(), sends SimResponse back. This keeps RobotManager ownership on the main thread where it already lives.

### 5. One robot per agent connection
**Rationale**: Each WebSocket/TCP connection controls exactly one robot. Multi-robot control requires multiple connections. This simplifies the protocol, avoids multiplexing complexity, and matches the gym single-environment paradigm. Multi-env vectorized training uses multiple connections naturally.

### 6. Conflict resolution: last-write-wins
**Rationale**: If two agents send commands for the same robot in the same tick (shouldn't happen with 1:1 assignment, but possible via race), the last command received overwrites previous. No queuing or merging — simple and predictable.

### 7. Graceful shutdown on app exit
**Rationale**: When eframe window closes, the tokio runtime must shut down cleanly. CancellationToken signals all server tasks to stop. In-flight WebSocket writes are cancelled — clients should handle disconnection. Server logs disconnection events for observability.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Protocol design | HIGH | GymRobotState/RobotAction already serialize to JSON (src/robot/state.rs) |
| Server architecture | HIGH | tokio + channels is standard eframe async pattern |
| Robot API integration | HIGH | RobotManager.step/set_command/get_robot already exist (src/robot/mod.rs) |
| Test approach | HIGH | Inline #[cfg(test)] matches all existing robot tests |
| WebSocket crate choice | HIGH | tokio-tungstenite is the de facto async WebSocket for tokio |
| Python SDK | MEDIUM | Gym interface is well-defined; websocket-client is standard Python lib |
| Multi-agent concurrency | MEDIUM | Channel-based design avoids races, but multi-step coordination untested |
| Performance | MEDIUM | JSON serialization overhead should be negligible at 60Hz; no benchmarks yet |

## Tasks

### Task 1: Protocol Types and Dependencies

**Files**: `src/agent/protocol.rs` (new), `Cargo.toml`
**Test Files**: `src/agent/protocol.rs` (inline tests)

**Description**: Define all JSON message types for the agent protocol and add networking dependencies to Cargo.toml. Message types: ClientMessage (Connect, Reset, Step, Observe, Close) and ServerMessage (Connected, Observation, Error, Closed). Each message is a tagged enum serialized with serde.

**Acceptance Criteria**:
- [ ] ClientMessage enum with Connect { robot_id }, Reset, Step { action: RobotAction }, Observe, Close variants
- [ ] ServerMessage enum with Connected { observation_space, action_space }, Observation { state: GymRobotState, reward: f32, done: bool, step_count: u64 }, Error { message: String }, Closed variants
- [ ] All types derive Serialize, Deserialize, Debug, Clone
- [ ] tokio, tokio-tungstenite added to Cargo.toml
- [ ] Round-trip JSON serialization tests pass for all message variants

**Tests to Write**:
- `test_client_message_connect_roundtrip` — serialize Connect{robot_id:0} to JSON and back
- `test_client_message_step_roundtrip` — serialize Step with motor_velocities and gripper_commands
- `test_server_message_connected_roundtrip` — serialize Connected with observation_space/action_space
- `test_server_message_observation_roundtrip` — serialize Observation with full GymRobotState
- `test_server_message_error_roundtrip` — serialize Error{message}
- `test_client_message_all_variants` — parse each variant from JSON string literals
- `test_server_message_all_variants` — parse each variant from JSON string literals

**Verification**:
```bash
cargo test --lib -- agent::protocol && cargo check
```

---

### Task 2: Simulation Bridge

**Files**: `src/agent/bridge.rs` (new)
**Test Files**: `src/agent/bridge.rs` (inline tests)

**Description**: Create the bridge layer that connects the network server to the simulation. Define SimCommand and SimResponse enums for channel communication. SimBridge holds the sender/receiver pair and provides async methods to send commands and await responses.

**Acceptance Criteria**:
- [ ] SimCommand enum: AddRobot { definition, base_pose }, Step { robot_id, action }, GetObservation { robot_id }, Reset { robot_id }, RemoveRobot { robot_id }
- [ ] SimResponse enum: RobotAdded { robot_id }, Stepped { state: GymRobotState, step_count: u64 }, Observation { state: GymRobotState }, Reset { state: GymRobotState }, Removed, Error { message }
- [ ] create_bridge() function returns (SimBridgeServer, SimBridgeClient) pair — server-side for the agent server, client-side for the main loop
- [ ] SimBridgeClient.process_pending() — non-blocking drain of command channel, processes each via RobotManager, sends responses back
- [ ] SimBridgeServer.send_command() — async send command and await response

**Tests to Write**:
- `test_bridge_creation` — create_bridge returns valid pair
- `test_bridge_step_command` — send Step, receive Stepped with valid state
- `test_bridge_get_observation` — send GetObservation, receive Observation
- `test_bridge_reset` — send Reset, receive Reset response with initial state
- `test_bridge_error_invalid_robot` — send command for nonexistent robot_id, receive Error
- `test_bridge_multiple_commands` — send multiple commands sequentially, all processed

**Verification**:
```bash
cargo test --lib -- agent::bridge && cargo check
```

---

### Task 3: Agent Session Handler

**Files**: `src/agent/session.rs` (new)
**Test Files**: `src/agent/session.rs` (inline tests)

**Description**: Implement the per-connection session handler. AgentSession manages a single agent's lifecycle: connect (assign robot), reset, step loop, close. It translates between protocol messages and bridge commands. Tracks step count per session.

**Acceptance Criteria**:
- [ ] AgentSession struct with robot_id, step_count, bridge handle
- [ ] handle_message(ClientMessage) -> ServerMessage method
- [ ] Connect: validates robot_id available, returns Connected with obs/action spaces
- [ ] Reset: resets robot to initial state, returns initial observation
- [ ] Step: applies action, steps simulation, returns new observation with step_count
- [ ] Observe: returns current observation without stepping
- [ ] Close: releases robot assignment, returns Closed
- [ ] Error handling: invalid robot_id, step before connect, double connect

**Tests to Write**:
- `test_session_connect` — connect to robot 0, receive Connected
- `test_session_step_increments_count` — step 3 times, verify step_count=3
- `test_session_reset_clears_count` — step, reset, verify step_count=0
- `test_session_observe_no_step` — observe returns state without incrementing step_count
- `test_session_close` — close session, verify Closed response
- `test_session_step_before_connect` — step without connecting returns Error
- `test_session_double_connect` — connect twice returns Error
- `test_session_invalid_robot` — connect to nonexistent robot returns Error

**Verification**:
```bash
cargo test --lib -- agent::session && cargo check
```

---

### Task 4: TCP Server

**Files**: `src/agent/tcp_server.rs` (new)
**Test Files**: `src/agent/tcp_server.rs` (inline tests)

**Description**: TCP server accepting line-delimited JSON connections. Each connection spawns an AgentSession. Reads newline-delimited JSON messages, dispatches to session handler, writes JSON responses followed by newline. Supports multiple concurrent connections via tokio tasks.

**Acceptance Criteria**:
- [ ] TcpAgentServer struct with port, bridge handle, active connections count (Arc<AtomicUsize>)
- [ ] start() method: bind to port, accept connections in loop
- [ ] Per-connection task: read line → parse ClientMessage → handle → serialize ServerMessage → write line
- [ ] Graceful shutdown via CancellationToken
- [ ] Connection cleanup on disconnect (release robot assignment)
- [ ] Max connections limit (default 16)

**Tests to Write**:
- `test_tcp_server_binds` — server starts and binds to port 0 (OS-assigned)
- `test_tcp_connect_and_message` — connect via TcpStream, send Connect JSON, receive Connected JSON
- `test_tcp_step_roundtrip` — full connect→reset→step→observe→close cycle over TCP
- `test_tcp_malformed_json` — send invalid JSON, receive Error response
- `test_tcp_connection_cleanup` — disconnect abruptly, verify robot released
- `test_tcp_multiple_connections` — two clients connect to different robots simultaneously

**Verification**:
```bash
cargo test --lib -- agent::tcp_server && cargo check
```

---

### Task 5: WebSocket Server

**Files**: `src/agent/ws_server.rs` (new)
**Test Files**: `src/agent/ws_server.rs` (inline tests)

**Description**: WebSocket server using tokio-tungstenite. Same protocol as TCP but using WebSocket message framing instead of newline delimiters. Shares AgentSession logic with TCP server.

**Acceptance Criteria**:
- [ ] WsAgentServer struct with port, bridge handle, active connections count (Arc<AtomicUsize>)
- [ ] Binary WebSocket messages rejected with Error response
- [ ] start() method: bind, accept, upgrade to WebSocket
- [ ] Per-connection task: receive WS text message → parse → handle → serialize → send WS text message
- [ ] Graceful shutdown via shared CancellationToken
- [ ] Connection cleanup on WebSocket close frame
- [ ] Ping/pong keepalive handling

**Tests to Write**:
- `test_ws_server_binds` — server starts and binds to port 0 (OS-assigned)
- `test_ws_connect_and_message` — connect via WebSocket, send Connect, receive Connected
- `test_ws_step_roundtrip` — full cycle over WebSocket
- `test_ws_close_frame` — send close frame, verify graceful shutdown
- `test_ws_binary_message_rejected` — send binary message, receive Error
- `test_ws_multiple_connections` — two WS clients simultaneously

**Verification**:
```bash
cargo test --lib -- agent::ws_server && cargo check
```

---

### Task 6: Server Orchestrator and Main Integration

**Files**: `src/agent/mod.rs` (new), `src/main.rs`, `src/ui/mod.rs`, `Cargo.toml`
**Test Files**: `src/agent/mod.rs` (inline tests)

**Description**: Create the agent module's public API (AgentServerConfig, start/stop functions). Integrate into main.rs: spawn tokio runtime on background thread, create bridge, wire SimBridgeClient into the eframe update loop. Add UI elements showing server status.

**Acceptance Criteria**:
- [ ] AgentServerConfig struct: tcp_port (default 9001), ws_port (default 9002), max_connections (default 16), enabled (default false)
- [ ] start_agent_server(config, bridge) -> AgentServerHandle function
- [ ] AgentServerHandle with stop() method and status() query
- [ ] AgentServerStatus: port numbers, connected agent count, running state
- [ ] main.rs spawns server thread when enabled, processes bridge commands in update()
- [ ] Connection/disconnection events logged via log::info!
- [ ] UI panel: server on/off toggle, port display, connected agents list
- [ ] `mod agent;` declaration in main.rs

**Tests to Write**:
- `test_config_defaults` — AgentServerConfig::default() has correct port values
- `test_server_handle_status` — start server, check status shows running
- `test_server_handle_stop` — start then stop, verify clean shutdown
- `test_bridge_process_in_update` — simulate main loop calling process_pending

**Verification**:
```bash
cargo test --lib -- agent && cargo check
```

---

### Task 7: Python SDK Client

**Files**: `python/echomap_client/__init__.py` (new), `python/echomap_client/env.py` (new), `python/setup.py` (new)
**Test Files**: `python/tests/test_client.py` (new)

**Description**: Python client SDK with gym-compatible interface. EchoMapEnv class connecting via WebSocket, providing reset(), step(action), close() methods. Supports observation_space and action_space properties. Uses websocket-client library.

**Acceptance Criteria**:
- [ ] EchoMapEnv(host, port, robot_id) constructor — connects via WebSocket
- [ ] reset() -> observation — sends Reset, returns initial GymRobotState dict
- [ ] step(action) -> (observation, reward, done, info) — sends Step with action, returns gym tuple
- [ ] close() — sends Close, disconnects
- [ ] observation_space property — returns ObservationSpace dict from server
- [ ] action_space property — returns ActionSpace dict from server
- [ ] EchoMapEnv usable as context manager (with statement)
- [ ] setup.py with websocket-client dependency

**Tests to Write**:
- `test_import` — import echomap_client succeeds
- `test_env_class_exists` — EchoMapEnv is importable and callable
- `test_env_has_gym_interface` — EchoMapEnv has reset, step, close, observation_space, action_space

**Verification**:
```bash
python3 -m py_compile python/echomap_client/__init__.py && python3 -m py_compile python/echomap_client/env.py
```

---

### Task 8: End-to-End Integration Tests

**Files**: `src/agent/mod.rs` (append integration tests)
**Test Files**: `src/agent/mod.rs` (inline tests)

**Description**: Full round-trip integration tests that start the server, connect a client, perform a complete agent lifecycle, and verify correctness. Tests both TCP and WebSocket transports. Tests multi-agent scenarios.

**Acceptance Criteria**:
- [ ] Single-agent TCP test: start server → connect → reset → step 10 times → observe → close → verify observations change
- [ ] Single-agent WebSocket test: same lifecycle over WebSocket
- [ ] Multi-agent test: two agents connect to different robots, both step independently
- [ ] Reconnection test: agent disconnects, reconnects to same robot
- [ ] Stress test: rapid step commands (100 steps), verify all complete without error

**Tests to Write**:
- `test_integration_tcp_full_lifecycle` — complete connect→reset→step→close over TCP
- `test_integration_ws_full_lifecycle` — complete lifecycle over WebSocket
- `test_integration_multi_agent` — two simultaneous agents on different robots
- `test_integration_reconnect` — disconnect and reconnect to same robot
- `test_integration_rapid_steps` — 100 rapid step commands complete correctly
- `test_integration_observation_changes` — observations differ after steps with non-zero actions

**Verification**:
```bash
cargo test --lib -- agent && cargo check
```

## Integration Tests

The Task 8 integration tests serve as full-feature end-to-end tests. They exercise:
- Server startup and shutdown
- Client connection over both transports
- Protocol message round-trips
- Simulation stepping via agent commands
- Multi-agent concurrent operation
- State observation correctness

## Verification Gate

```bash
cargo check
cargo test
cargo clippy --all-targets
cargo fmt --check
python3 -m py_compile python/echomap_client/__init__.py
python3 -m py_compile python/echomap_client/env.py
```

## Open Questions

None — all questions resolved from codebase research.

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO | 8.4/10 | None |
| Design/Architecture | 5.6/10 | None (soft concern: async integration pattern — addressed in Design Decision #1, #4, #7) |
| Engineering | 4.8/10 | 2 flagged but invalid: (1) types exist in D5 branch per prerequisite, (2) connection count addressed by adding Arc<AtomicUsize> |

**Review feedback applied**: Added conflict resolution design decision (#6), graceful shutdown (#7), connection count tracking (AtomicUsize), binary WS frame rejection, port 0 in tests, observability logging.
