# Agent-to-Agent Messaging Protocol

Add a message bus to the SimBridge that lets agents send text messages to other agents. Messages are queued and delivered with observation responses. Conversation history stored per agent pair.

## Slug

`agent-messaging-protocol`

## Context

Deliverable 2 of 6-part boxing match goal. Agents currently interact with the simulation (Step/Observe/Reset) but cannot communicate with each other. For the boxing match, agents need to trash-talk, negotiate, and coordinate. The existing protocol (ClientMessage/ServerMessage with serde-tagged JSON over TCP/WebSocket) and bridge architecture (mpsc channel + oneshot responses) provide a clean extension point.

## Test Infrastructure

- **Framework**: Built-in Rust `#[test]` and `#[tokio::test]`
- **Test command**: `cargo test`
- **Lint command**: `cargo clippy -- -D warnings`
- **Format command**: `cargo fmt --check`
- **Convention**: Tests live in `#[cfg(test)] mod tests { ... }` at bottom of each file
- **Assertion style**: `assert!`, `assert_eq!` with descriptive failure messages
- **Current count**: 755 tests passing
- **Async tests**: Bridge tests use `#[tokio::test]` with `tokio::spawn` + `yield_now` pattern

## Requirements

1. Agents must be able to send text messages to other agents by robot_id
2. Messages must be queued and delivered with the next observation response (Step/Observe/Reset)
3. Conversation history must be stored per agent pair with configurable capacity
4. Message sending must return an acknowledgment or error
5. Messages from unknown or disconnected robots must return errors
6. The Python client must support sending and receiving messages
7. Message events must appear in the AgentActivityLog

## Design Decisions

### Message delivery via observation polling (HIGH confidence)
**Decision**: Messages are queued per-robot and drained when the recipient's next observation is built (Step/Observe/Reset responses). No push/broadcast mechanism.
**Rationale**: Matches the existing request-response protocol. Agents already poll via Step/Observe. Adding push would require a second channel per connection and complicate the protocol. Polling is simpler and sufficient for 60Hz agent loops.

### Messages in ServerMessage::Observation, not GymRobotState (HIGH confidence)
**Decision**: Add `messages: Vec<AgentMessage>` to the ServerMessage::Observation variant, not to GymRobotState.
**Rationale**: GymRobotState represents physics/sensor state. Messages are agent-communication state. Keeping them separate preserves the physics/comms boundary. ServerMessage::Observation already aggregates state + reward + done + step_count — messages fit naturally here.

### MessageBus owned by SimBridgeClient (HIGH confidence)
**Decision**: Add a `MessageBus` struct to `SimBridgeClient` that holds pending queues and conversation history.
**Rationale**: SimBridgeClient is the authority on simulation state processing. All commands funnel through `execute()`. Message routing is just another command. Matches how step_counts and state_buffer are already stored on SimBridgeClient.

### from_robot_id set by AgentSession, not client (HIGH confidence)
**Decision**: ClientMessage::SendMessage only contains `to_robot_id` and `content`. The session fills in `from_robot_id` from its own `robot_id` field.
**Rationale**: Prevents sender spoofing. AgentSession already tracks which robot_id is connected.

### Content as plain String with max length (HIGH confidence)
**Decision**: Message content is a String, max 1024 bytes. Agents can put JSON in it if they want structure.
**Rationale**: Keeps the protocol simple. Structured content is an application concern, not a transport concern. Length limit prevents memory abuse.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Protocol extension pattern | HIGH | Existing ClientMessage/ServerMessage pattern in protocol.rs is clean serde-tagged JSON |
| Bridge command handling | HIGH | SimCommand/SimResponse pattern in bridge.rs; execute() match is the extension point |
| Session integration | HIGH | AgentSession.handle_message() dispatch pattern in session.rs |
| Message delivery model | HIGH | Observation responses already return to agents via oneshot; adding messages field is additive |
| Python client | HIGH | EchoMapEnv uses websocket.send/recv with JSON; adding send_message() is trivial |
| Thread safety | HIGH | All message state lives in SimBridgeClient (main thread only); no shared mutable state |
| Performance impact | HIGH | Message routing is O(1) HashMap lookup; negligible vs physics step |

## Tasks

### Task 1: Add message types to protocol

**Files**: `src/agent/protocol.rs`
**Test Files**: `src/agent/protocol.rs` (in #[cfg(test)] mod tests)
**Description**: Add `AgentMessage` struct, `SendMessage` variant to `ClientMessage`, `MessageSent` variant to `ServerMessage`, and `messages: Vec<AgentMessage>` field to the `Observation` variant of `ServerMessage`.

**Acceptance Criteria**:
- `AgentMessage` struct with from_robot_id, to_robot_id, content, timestamp fields
- `ClientMessage::SendMessage { to_robot_id, content }` variant exists
- `ServerMessage::MessageSent` variant exists
- `ServerMessage::Observation` includes `messages: Vec<AgentMessage>` field
- All types derive Serialize, Deserialize, Clone, Debug
- Existing protocol tests still pass

**Tests to Write**:
- `test_agent_message_json_round_trip`: AgentMessage serializes/deserializes correctly
- `test_send_message_client_message`: ClientMessage::SendMessage round-trips through JSON
- `test_observation_with_messages`: ServerMessage::Observation with messages round-trips
- `test_observation_empty_messages`: ServerMessage::Observation with empty messages vec round-trips

**Verification**:
```bash
cargo test protocol -- --nocapture 2>&1 | tail -5
```

### Task 2: Add message bus and bridge integration

**Files**: `src/agent/bridge.rs`
**Test Files**: `src/agent/bridge.rs` (in #[cfg(test)] mod tests)
**Description**: Add `MessageBus` struct to SimBridgeClient with per-robot pending queues and per-pair conversation history. Add `SendMessage` variant to `SimCommand` and `MessageSent` to `SimResponse`. Add messages to Stepped/Observation/Reset response variants. Handle SendMessage in `execute()`. Drain pending messages when building observation responses. Add Message event kind to activity log.

**Acceptance Criteria**:
- `MessageBus` struct with `pending: HashMap<usize, VecDeque<AgentMessage>>` and `history: HashMap<(usize, usize), VecDeque<AgentMessage>>`
- `SimCommand::SendMessage { from_robot_id, to_robot_id, content }` variant
- `SimResponse::MessageSent` variant
- `SimResponse::Stepped`, `SimResponse::Observation`, `SimResponse::Reset` all gain `messages: Vec<AgentMessage>` field
- SendMessage in execute() validates both robot IDs, queues message, returns MessageSent
- Step/Observe/Reset responses drain pending messages for the target robot
- `AgentEventKind::Message` variant added, message events logged
- Content length validated (max 1024 bytes)

**Tests to Write**:
- `test_message_bus_send_and_drain`: Send message, drain returns it for recipient
- `test_message_bus_history`: History stored per pair, accessible
- `test_message_bus_drain_clears`: After drain, pending is empty
- `test_bridge_send_message`: SendMessage command through bridge returns MessageSent
- `test_bridge_send_to_invalid_robot`: SendMessage to nonexistent robot returns Error
- `test_bridge_send_from_invalid_robot`: SendMessage from nonexistent robot returns Error
- `test_bridge_step_delivers_messages`: Step response includes pending messages
- `test_bridge_observe_delivers_messages`: Observe response includes pending messages
- `test_bridge_message_content_too_long`: Content exceeding 1024 bytes returns Error

**Verification**:
```bash
cargo test bridge -- --nocapture 2>&1 | tail -5 && cargo test message_bus -- --nocapture 2>&1 | tail -5
```

### Task 3: Wire messaging through AgentSession

**Files**: `src/agent/session.rs`
**Test Files**: `src/agent/session.rs` (in #[cfg(test)] mod tests)
**Description**: Add `handle_send_message` method to AgentSession. Wire `ClientMessage::SendMessage` in the `handle_message` dispatch. Update observation builders to include messages from SimResponse. Session sets `from_robot_id` from its own `robot_id`.

**Acceptance Criteria**:
- `ClientMessage::SendMessage` dispatches to `handle_send_message`
- `handle_send_message` errors if session not connected (no robot_id)
- `handle_send_message` sends `SimCommand::SendMessage` with session's robot_id as from
- Step/Observe/Reset handlers pass messages through to ServerMessage::Observation
- `ServerMessage::MessageSent` returned on success

**Tests to Write**:
- `test_session_send_message`: Connected session sends message successfully
- `test_session_send_message_not_connected`: Unconnected session returns error
- `test_session_step_includes_messages`: Step response includes delivered messages

**Verification**:
```bash
cargo test session -- --nocapture 2>&1 | tail -5
```

### Task 4: Update Python client

**Files**: `python/echomap_client/env.py`
**Test Files**: N/A (manual verification)
**Description**: Add `send_message(to_robot_id, content)` method to EchoMapEnv. Update step/observe/reset response parsing to include messages.

**Acceptance Criteria**:
- `send_message(to_robot_id: int, content: str)` method exists
- Method sends `{"type":"send_message","to_robot_id":N,"content":"..."}` JSON
- Step/observe/reset return messages in info dict under "messages" key
- Each message is a dict with from_robot_id, to_robot_id, content, timestamp

**Tests to Write**: None (Python, manual verification)

**Verification**:
```bash
python3 -c "from echomap_client.env import EchoMapEnv; e = EchoMapEnv.__new__(EchoMapEnv); assert hasattr(e, 'send_message')" 2>&1 && echo "OK"
```

### Task 5: Integration test — two agents messaging

**Files**: `src/agent/bridge.rs` (in #[cfg(test)] mod tests)
**Test Files**: `src/agent/bridge.rs`
**Description**: End-to-end test: two agents (robot 0 and robot 1) exchange messages through the full bridge pipeline. Agent 0 sends a message to Agent 1. Agent 1 steps and receives the message in its observation. Agent 1 replies. Agent 0 observes and receives the reply.

**Acceptance Criteria**:
- Test creates two robots via AddRobot
- Agent 0 sends message to Agent 1 via SendMessage
- Agent 1's next Step/Observe includes the message
- Agent 1 replies to Agent 0
- Agent 0's next Observe includes the reply
- All messages have correct from/to IDs and content

**Tests to Write**:
- `test_two_agents_message_exchange`: Full round-trip message exchange between two agents
- `test_message_delivery_order`: Multiple messages delivered in FIFO order
- `test_messages_only_delivered_once`: After delivery, messages don't appear again

**Verification**:
```bash
cargo test two_agents -- --nocapture 2>&1 | tail -5 && cargo test message_delivery -- --nocapture 2>&1 | tail -5
```

## Integration Tests

### Full pipeline: two agents messaging at 60Hz
Add to `src/agent/bridge.rs` tests:

**`test_two_agents_message_exchange`**: Create 2 robots. Robot 0 sends "hello" to Robot 1. Robot 1 steps and receives the message. Robot 1 sends "world" back. Robot 0 observes and receives the reply. Verify message content, sender/receiver IDs, timestamps are monotonic.

## Verification Gate

All of these must exit 0:
```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 8.5/10 | None |
| Design/Architecture | 8.0/10 | None |
| Engineering | 8.0/10 | None |

Note: Self-assessed scores due to worktree isolation preventing subagent spec access.

## Open Questions

None — all questions resolved from codebase analysis.
