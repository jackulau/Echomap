# Boxing Arena Scenario

Create a boxing ring scenario with match state machine, scoring system, and extended observations for AI agent boxing matches.

## Slug

`boxing-arena-scenario`

## Context

Deliverable 4 of the AI boxing match goal. Deliverables 1-3 are complete: performance optimization (755→826 tests), agent-to-agent messaging, and robot combat physics (BodyZone, CombatState, HitEvent, step_combat, GymCombatState in observations). This deliverable creates the boxing ring environment, match rules, and round management that the LLM agents (deliverable 5) will use to compete.

## Test Infrastructure

- **Framework**: Rust inline `#[cfg(test)] mod tests` per file
- **Runner**: `cargo test`
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt --check`
- **Baseline**: 826 tests passing

## Requirements

1. Boxing ring scene geometry (floor + 4 wall boundaries)
2. Enhanced humanoid robot definition with head, torso, and two arms with body zones
3. BoxingMatch state machine: WaitingForAgents → Countdown → Fighting → RoundEnd → MatchEnd
4. Round timer (configurable duration, default 180s)
5. Scoring system tracking clean hits, knockdowns per round, and judges' scorecard
6. Match state exposed in agent observations (round number, time remaining, opponent health, scores)
7. Two robots spawned facing each other at configurable distance
8. Match auto-advances through states based on agent connections and timer
9. Backward compatible — non-boxing scenarios unaffected

## Design Decisions

1. **BoxingMatch as a standalone struct in a new `src/robot/boxing.rs` module** — The match logic (state machine, scoring, round timing) is distinct from robot physics and agent protocol. A dedicated module keeps concerns separated. It lives under `src/robot/` because it orchestrates `RobotManager` and `CombatState`.

2. **BoxingMatch owns references to robot IDs, not robots** — The match tracks which two robots are fighting by their IDs (indices into RobotManager). It doesn't own robots — RobotManager continues to own them. Match reads CombatState/HitEvents from RobotManager after each step.

3. **Match state flows through a new `match_state` field on ServerMessage::Observation** — Rather than extending GymRobotState (which is per-robot), match state is global (round number, scores, timer). Add `#[serde(default)] pub match_state: Option<BoxingMatchState>` to the Observation variant.

4. **Humanoid robot extends boxing_test_robot()** — The existing `boxing_test_robot()` factory (3 links: torso + 2 arms) gets a `boxing_humanoid()` variant adding a head link (BodyZone::Head, 3x damage). 4 links, 3 joints.

5. **ScenarioPreset not extended** — The existing ScenarioPreset bundles a single RobotDefinition. Boxing needs two robots + match logic. Instead of contorting the preset system, add a `BoxingScenario` struct with its own `setup()` that creates the full environment.

6. **Scoring: 10-point-must system per round** — Each round, the fighter who lands more clean hits gets 10, the other gets 9 (or 8 with knockdown). This is the standard boxing scoring convention. Knockdown = automatic 10-8 round.

7. **Match updates in bridge processing** — `SimBridgeClient::process_pending()` already calls `manager.step()`. After stepping, if a BoxingMatch is active, call `boxing_match.update(manager, dt)` to advance the state machine.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| State machine design | HIGH | Standard enum-based FSM, matches Rust idioms |
| Robot definition | HIGH | Extends existing boxing_test_robot() pattern (definition.rs:109) |
| Observation extension | HIGH | Follows hit_events/combat field pattern (protocol.rs:40) |
| Bridge integration | HIGH | SimBridgeClient::execute() pattern is clear (bridge.rs:334) |
| Scoring system | MEDIUM | 10-point-must is standard boxing but implementation is new territory |
| Scene geometry | HIGH | Follows make_test_room() pattern (scenarios/mod.rs:47) |
| Backward compatibility | HIGH | All new fields use #[serde(default)], Option types |

## Tasks

### Task 1: Boxing ring scene geometry

**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline tests)

Add a boxing ring factory function.

**Changes:**
- Add `pub fn make_boxing_ring(size: f32) -> Scene` — creates a flat floor with 4 wall boundaries forming a square ring. Uses existing `make_wall` helper pattern from `make_test_room`. Default size 6.0 (meters).
- The ring floor is at y=0, walls are low (1.0m high) to represent ropes/boundaries.

**Acceptance Criteria:**
- [ ] make_boxing_ring returns a Scene with floor + 4 walls
- [ ] Ring is centered at origin with configurable size
- [ ] Walls are low (rope height, ~1.0m)

**Tests to Write:**
- `test_boxing_ring_has_floor_and_walls` — verify 5 scene objects (floor + 4 walls)
- `test_boxing_ring_dimensions` — verify ring spans correct size

**Verification:** `cargo test -p echomap --lib scenarios`

---

### Task 2: Enhanced humanoid robot definition

**Files**: `src/robot/definition.rs`
**Test Files**: `src/robot/definition.rs` (inline tests)

Add a proper humanoid boxing robot with head zone.

**Changes:**
- Add `RobotDefinition::boxing_humanoid() -> Self` — 4 links (torso/Body, head/Head, left_arm/LeftArm, right_arm/RightArm), 3 revolute joints (neck, left_shoulder, right_shoulder). Head has mass 3.0, Sphere(0.1). Arms have mass 2.0, Cylinder(0.05, 0.4). Torso has mass 10.0, Cuboid(0.2, 0.3, 0.15).

**Acceptance Criteria:**
- [ ] boxing_humanoid has 4 links with correct body zones
- [ ] Head link has BodyZone::Head
- [ ] All links have collision shapes
- [ ] 3 revolute joints connecting head and arms to torso

**Tests to Write:**
- `test_boxing_humanoid_link_count` — 4 links
- `test_boxing_humanoid_zones` — correct zone assignment per link
- `test_boxing_humanoid_head_has_sphere` — head uses Sphere collision shape

**Verification:** `cargo test -p echomap --lib definition`

---

### Task 3: BoxingMatch state machine and scoring

**Files**: `src/robot/boxing.rs` (NEW), `src/robot/mod.rs`
**Test Files**: `src/robot/boxing.rs` (inline tests)

Core match logic: state machine, round timer, scoring.

**Changes in new `src/robot/boxing.rs`:**
- Add `MatchPhase` enum: `WaitingForAgents`, `Countdown { remaining: f32 }`, `Fighting`, `RoundEnd { remaining: f32 }`, `MatchEnd`
- Add `RoundScore` struct: `hits_a: u32`, `hits_b: u32`, `knockdowns_a: u32`, `knockdowns_b: u32`, `score_a: u8`, `score_b: u8`
- Add `BoxingMatchConfig` struct: `round_duration: f32` (default 180.0), `num_rounds: u8` (default 3), `countdown_duration: f32` (default 3.0), `round_break_duration: f32` (default 5.0)
- Add `BoxingMatch` struct:
  - `config: BoxingMatchConfig`
  - `phase: MatchPhase`
  - `robot_a: usize` (robot index)
  - `robot_b: usize` (robot index)
  - `current_round: u8`
  - `round_time: f32`
  - `rounds: Vec<RoundScore>`
  - `agents_connected: [bool; 2]`
- Add `BoxingMatch::new(robot_a, robot_b, config) -> Self`
- Add `BoxingMatch::update(&mut self, hit_events: &[HitEvent], combat_states: &[(usize, &CombatState)], dt: f32)` — advances state machine:
  - WaitingForAgents: transitions to Countdown when both `agents_connected` are true
  - Countdown: decrements remaining, transitions to Fighting at 0
  - Fighting: increments round_time, tracks hits/knockdowns from HitEvents, transitions to RoundEnd when round_time >= round_duration OR knockdown
  - RoundEnd: scores round (10-point-must), decrements remaining, transitions to Fighting (next round) or MatchEnd
  - MatchEnd: no-op
- Add `BoxingMatch::connect_agent(&mut self, robot_id: usize)` — sets agents_connected flag
- Add `BoxingMatch::winner(&self) -> Option<usize>` — returns robot with more round wins, None if tie
- Add `BoxingMatch::current_scores(&self) -> (u32, u32)` — total scorecard points

**Changes in `src/robot/mod.rs`:**
- Add `pub mod boxing;`

**Acceptance Criteria:**
- [ ] State machine transitions correctly through all phases
- [ ] Round timer counts up during Fighting phase
- [ ] Hits from HitEvents are counted per robot per round
- [ ] Knockdown detected from CombatState
- [ ] 10-point-must scoring: winner gets 10, loser gets 9, knockdown = 10-8
- [ ] Match ends after configured number of rounds

**Tests to Write:**
- `test_match_initial_state` — new match starts in WaitingForAgents
- `test_match_waiting_to_countdown` — connecting both agents triggers Countdown
- `test_countdown_to_fighting` — countdown ticks down to 0, transitions to Fighting
- `test_fighting_round_timer` — round_time increases during Fighting
- `test_fighting_tracks_hits` — HitEvents increment hit counters
- `test_round_end_on_timer` — round ends when time exceeds duration
- `test_round_scoring_10_9` — more hits = 10, fewer = 9
- `test_round_scoring_knockdown_10_8` — knockdown = 10-8
- `test_match_end_after_all_rounds` — match ends after num_rounds
- `test_winner_determination` — winner has more total points
- `test_match_end_on_knockout` — health 0 = immediate MatchEnd

**Verification:** `cargo test -p echomap --lib boxing`

---

### Task 4: Match state in observations

**Files**: `src/robot/boxing.rs`, `src/agent/protocol.rs`
**Test Files**: `src/robot/boxing.rs`, `src/agent/protocol.rs` (inline tests)

Expose match state through the agent protocol.

**Changes in `src/robot/boxing.rs`:**
- Add `BoxingMatchState` struct (serializable snapshot for protocol):
  - `phase: String` (serialized phase name)
  - `current_round: u8`
  - `round_time: f32`
  - `round_duration: f32`
  - `scores: Vec<[u8; 2]>` (per-round scores)
  - `total_score_a: u32`
  - `total_score_b: u32`
  - `your_robot: usize` (which robot this agent controls)
  - `opponent_health: f32`
  - `opponent_stamina: f32`
- Add `BoxingMatch::snapshot(&self, for_robot: usize, opponent_combat: Option<&CombatState>) -> BoxingMatchState`
- Derive Serialize, Deserialize, Clone, Debug on BoxingMatchState

**Changes in `src/agent/protocol.rs`:**
- Add `#[serde(default)] pub match_state: Option<crate::robot::boxing::BoxingMatchState>` to `ServerMessage::Observation`

**Acceptance Criteria:**
- [ ] BoxingMatchState captures all match info needed by agents
- [ ] Opponent health/stamina visible in match state
- [ ] Phase serialized as string for JSON readability
- [ ] Existing observation tests pass with match_state: None

**Tests to Write:**
- `test_match_snapshot_fighting` — snapshot during Fighting has correct round/timer
- `test_match_snapshot_includes_opponent` — opponent health/stamina populated
- `test_match_state_serialization` — round-trip JSON
- `test_observation_with_match_state` — ServerMessage::Observation with match_state field

**Verification:** `cargo test -p echomap --lib boxing && cargo test -p echomap --lib protocol`

---

### Task 5: BoxingScenario setup and bridge integration

**Files**: `src/robot/boxing.rs`, `src/agent/bridge.rs`, `src/agent/session.rs`
**Test Files**: `src/robot/boxing.rs` (inline tests)

Wire up the boxing match with the simulation bridge.

**Changes in `src/robot/boxing.rs`:**
- Add `BoxingScenario` struct:
  - `ring: Scene`
  - `boxing_match: BoxingMatch`
  - `robot_a_id: usize`
  - `robot_b_id: usize`
- Add `BoxingScenario::new(config: BoxingMatchConfig) -> (Self, RobotManager)`:
  - Creates boxing ring via make_boxing_ring()
  - Creates RobotManager with two boxing_humanoid robots facing each other
  - Robot A at (-1.5, 0, 0), Robot B at (1.5, 0, 0)
  - Enables CombatState on both robots (100 health, 100 stamina)
  - Returns the scenario and the pre-configured RobotManager

**Changes in `src/agent/bridge.rs`:**
- Add `pub boxing_match: Option<BoxingMatch>` to `SimBridgeClient`
- In `SimBridgeClient::execute()` for `SimCommand::Step`:
  - After `manager.step(dt, scene_meshes)`, if `self.boxing_match` is Some, call `boxing_match.update(...)` with manager's hit_events and combat_states
  - When building SimResponse::Stepped, populate match_state if boxing_match exists
- In `SimBridgeClient::execute()` for `SimCommand::GetSpaces` (connect):
  - If boxing_match exists, call `boxing_match.connect_agent(robot_id)`

**Changes in `src/agent/session.rs`:**
- In `handle_step`, `handle_observe`, `handle_reset`: pass match_state from SimResponse through to ServerMessage::Observation

**Note:** SimResponse::Stepped/Observation/Reset will need a new `match_state: Option<BoxingMatchState>` field.

**Acceptance Criteria:**
- [ ] BoxingScenario creates ring + 2 humanoid robots
- [ ] Robots positioned facing each other
- [ ] Both robots have CombatState enabled
- [ ] Bridge updates boxing match on each step
- [ ] Agent connect triggers match agent_connected
- [ ] Match state included in observations when boxing match active
- [ ] Non-boxing bridge operations unaffected (boxing_match = None)

**Tests to Write:**
- `test_boxing_scenario_creates_two_robots` — RobotManager has 2 robots
- `test_boxing_scenario_robots_have_combat` — both robots have CombatState
- `test_boxing_scenario_robots_positioned` — robots at expected positions
- `test_boxing_match_update_from_step` — stepping advances match state

**Verification:** `cargo test -p echomap --lib boxing && cargo test -p echomap --lib bridge`

---

### Task 6: Python client boxing support

**Files**: `python/echomap_client/env.py`
**Test Files**: N/A (manual verification)

Extend Python client to expose match state.

**Changes:**
- In `step()`: extract `match_state` from response, add to info dict
- In `observe()`: extract `match_state` from response, add to info dict
- In `reset()`: extract `match_state` from response, add to info dict

**Acceptance Criteria:**
- [ ] step() returns match_state in info dict when present
- [ ] observe() returns match_state in info dict when present
- [ ] Backward compatible when match_state absent (defaults to None)

**Tests to Write:**
- `test_step_returns_match_state` — verify info["match_state"] exists
- `test_step_no_match_state` — verify info["match_state"] is None when absent

**Verification:** `cargo test -p echomap`

---

### Task 7: Integration tests — full boxing match

**Files**: `src/robot/boxing.rs`
**Test Files**: `src/robot/boxing.rs` (inline tests)

End-to-end test of a complete boxing match flow.

**Changes:**
- Add integration tests that exercise the full flow:
  1. Create BoxingScenario
  2. Connect two agents
  3. Step through countdown
  4. Simulate fighting with hits
  5. Verify round scoring
  6. Complete all rounds
  7. Verify match winner

**Acceptance Criteria:**
- [ ] Full match flows from WaitingForAgents to MatchEnd
- [ ] Scoring accumulates correctly across rounds
- [ ] Winner determined correctly
- [ ] Match state snapshots correct at each phase

**Tests to Write:**
- `test_full_boxing_match_flow` — complete 3-round match
- `test_knockout_ends_match_early` — KO in round 1 goes straight to MatchEnd
- `test_boxing_scenario_with_combat_step` — combat physics produce hits during match

**Verification:** `cargo test -p echomap --lib boxing`

## Integration Tests

1. **test_full_boxing_match_flow** — Two boxing humanoids, full 3-round match, scoring verified.
2. **test_knockout_ends_match_early** — One robot KO'd in round 1, match ends immediately.
3. **test_boxing_scenario_with_combat_step** — Full combat step produces hits that score correctly.

## Verification Gate

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test -p echomap
```

All three must exit 0. Test count should increase from 826 to ~860+.

## Open Questions

None — all questions self-resolved from codebase analysis.

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 8.0/10 | None |
| Design/Architecture | 7.5/10 | None |
| Engineering | 8.0/10 | None |
