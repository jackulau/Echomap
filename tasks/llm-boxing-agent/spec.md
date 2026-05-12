# LLM Boxing Agent Integration

Build Python boxing agents (heuristic + LLM-powered) that connect to the EchoMap simulation, fight autonomously, trash-talk, and receive match commentary.

## Slug

`llm-boxing-agent`

## Context

Deliverable 5 of 6 in the AI boxing match goal. Deliverables 1-4 built: optimized physics (22.9µs/step), agent-to-agent messaging, robot combat physics with punch detection, and a boxing arena with state machine + scoring. Now need Python agents that actually fight.

The simulation exposes everything via WebSocket JSON protocol through `EchoMapEnv`:
- `step(action)` → observation with joint positions, combat state (health/stamina/hits), match_state (phase/round/scores)
- `send_message(to_robot_id, content)` → trash-talk delivery
- `info["messages"]` → incoming messages from opponent
- `info["match_state"]` → BoxingMatchState dict

Boxing humanoid has 3 joints: neck (±π/4), left_shoulder (±π), right_shoulder (±π). Motor velocities = 3 floats.

## Test Infrastructure

- **Framework**: unittest (pytest-compatible)
- **Run command**: `python3 -m pytest python/tests/ -v`
- **Convention**: `python/tests/test_*.py`, class-based with `unittest.TestCase`
- **Pattern**: `sys.path.insert(0, ...)` for imports, no server needed for unit tests
- **Rust tests**: `cargo test` (854 passing baseline)

## Requirements

1. Base agent class with `decide(observation, info) -> (action, Optional[str])` interface
2. Heuristic boxing agent that fights using rule-based logic (no API keys needed)
3. LLM-powered agent using Claude API that outputs motor commands + trash-talk
4. Match runner that connects two agents and runs a full match
5. Match commentary generator that narrates the fight from match_state
6. All agents testable without a running simulation server

## Design Decisions

1. **Single `decide()` return tuple** — Returns `(action_dict, optional_message)` instead of separate methods. Simpler interface, LLM can produce both in one call. Rationale: keeps the agent loop tight and lets the LLM reason about actions and words together.

2. **Heuristic agent as default, LLM as optional** — `anthropic` package in `extras_require["llm"]`, not `install_requires`. Rationale: must work without API keys for testing and demos.

3. **Synchronous Claude API calls** — Use `anthropic.Anthropic()` sync client, not async. Rationale: matches the synchronous `EchoMapEnv` step loop. Async would require restructuring the entire client.

4. **Commentary as post-round processor** — Commentary generates text after each round/match end, not every step. Rationale: LLM calls are expensive; per-step commentary would be 60 calls/second.

5. **Agent loop in runner, not in agent** — Runner owns the connect/step loop, agents are pure decision functions. Rationale: separation of concerns, easier testing, agents don't need WebSocket knowledge.

6. **Mock observations for testing** — Tests construct observation dicts directly, no sim server needed. Rationale: fast unit tests, CI-friendly, same pattern as existing test_client.py.

## Confidence

| Area | Level | Evidence |
|------|-------|---------|
| Agent interface design | HIGH | Follows gym agent pattern, EchoMapEnv API confirmed in env.py |
| Observation structure | HIGH | Built the protocol in deliverable 2-4, match_state in boxing.rs:264-276 |
| Motor velocities shape | HIGH | boxing_humanoid has 3 joints (definition.rs), confirmed in bridge.rs |
| Claude API usage | HIGH | Standard anthropic SDK pattern, well-documented |
| Test approach | HIGH | Matches existing test_client.py pattern exactly |
| Commentary design | MEDIUM | No prior art in codebase; post-round granularity is a judgment call |

## Tasks

### Task 1: Base Agent Class and Heuristic Agent

**Files**: `python/echomap_client/agents.py`
**Test Files**: `python/tests/test_agents.py`

**Description**: Create `BoxingAgent` base class with abstract `decide()` method. Implement `HeuristicBoxingAgent` that uses rule-based logic:
- If opponent health < own health: attack aggressively (high shoulder velocities toward opponent)
- If own health low: defensive (retract arms, protect head)
- If match phase is countdown/round_end: idle (zero velocities)
- Random jab patterns with varying intensity
- Occasional trash-talk from a preset list

**Acceptance Criteria**:
- `BoxingAgent` is abstract, cannot be instantiated directly
- `HeuristicBoxingAgent.decide(obs, info)` returns `(action_dict, Optional[str])`
- `action_dict` has `"motor_velocities"` key with list of 3 floats
- Motor velocities stay within joint limits (neck ±0.785, shoulders ±3.14)
- Returns `None` message most steps, trash-talk occasionally
- Returns zero velocities during non-fighting phases

**Tests to Write**:
- `test_base_agent_not_instantiable`: BoxingAgent() raises TypeError
- `test_heuristic_returns_valid_action`: action has motor_velocities with 3 floats
- `test_heuristic_velocities_in_range`: all velocities within joint limits
- `test_heuristic_idle_during_countdown`: zero velocities when phase != "fighting"
- `test_heuristic_attacks_when_healthy`: non-zero arm velocities when health advantage
- `test_heuristic_defends_when_low_health`: different behavior at low health
- `test_heuristic_trash_talk_is_string_or_none`: message type check

**Verification**: `python3 -m pytest python/tests/test_agents.py -v -k "heuristic or base_agent" && echo PASS`

### Task 2: LLM Boxing Agent

**Files**: `python/echomap_client/llm_agent.py`
**Test Files**: `python/tests/test_agents.py` (append)

**Description**: Create `LLMBoxingAgent` that uses Claude API as the brain. Constructs a prompt from observations (own pose, opponent state, health, stamina, match score, incoming messages) and parses the response into motor commands + trash-talk. Uses `ANTHROPIC_API_KEY` env var. Falls back to heuristic if API unavailable.

Prompt structure:
- System: "You are a boxing robot. Output JSON with motor_velocities [neck, left_arm, right_arm] and optional trash_talk string."
- User: formatted observation summary (health, opponent health, round, score, recent messages)

**Acceptance Criteria**:
- `LLMBoxingAgent(model="claude-haiku-4-5-20251001")` accepts model parameter
- Uses `anthropic.Anthropic()` client with env var API key
- `decide()` constructs prompt from observation and calls Claude API
- Parses JSON from Claude response into action + message
- Falls back to heuristic on API error or missing key
- Respects motor velocity limits in parsed output (clamps if needed)

**Tests to Write**:
- `test_llm_agent_instantiable`: can create without API key (lazy init)
- `test_llm_agent_builds_prompt`: verify prompt contains health/score/phase info
- `test_llm_agent_parses_valid_response`: mock API, verify action extraction
- `test_llm_agent_fallback_on_error`: mock API error, verify heuristic fallback
- `test_llm_agent_clamps_velocities`: out-of-range values get clamped
- `test_llm_agent_fallback_without_key`: no ANTHROPIC_API_KEY → heuristic

**Verification**: `python3 -m pytest python/tests/test_agents.py -v -k "llm" && echo PASS`

### Task 3: Match Commentary Generator

**Files**: `python/echomap_client/commentary.py`
**Test Files**: `python/tests/test_commentary.py`

**Description**: Create `MatchCommentary` class that generates fight narration. Two modes:
- `generate_round_summary(match_state, events)` → text summary of a round
- `generate_match_summary(match_state)` → final match result narrative

Built-in template mode (no API needed): uses format strings with stats. Optional LLM mode: uses Claude to generate colorful commentary.

**Acceptance Criteria**:
- `MatchCommentary(use_llm=False)` works without API key
- Template mode produces readable round summaries with scores and hit counts
- Template mode produces match result with winner announcement
- LLM mode (when available) generates more colorful commentary
- Falls back to template mode if LLM unavailable

**Tests to Write**:
- `test_round_summary_template`: verify template output contains round number and scores
- `test_match_summary_template_with_winner`: verify winner announcement
- `test_match_summary_template_draw`: verify draw announcement
- `test_commentary_without_api_key`: template mode works
- `test_round_summary_includes_stats`: hit counts appear in output

**Verification**: `python3 -m pytest python/tests/test_commentary.py -v && echo PASS`

### Task 4: Match Runner

**Files**: `python/echomap_client/runner.py`
**Test Files**: `python/tests/test_runner.py`

**Description**: Create `BoxingMatchRunner` that orchestrates a full match:
1. Connects two agents to the sim (robot 0 and robot 1)
2. Runs the step loop: observe → decide → step → send messages
3. Tracks match_state transitions (countdown → fighting → round_end → match_end)
4. Triggers commentary at round ends and match end
5. Collects match statistics (total hits, messages sent, rounds)
6. Returns match result dict when done

**Acceptance Criteria**:
- `BoxingMatchRunner(agent_a, agent_b, host, port)` takes two agents and server info
- `run()` returns dict with winner, scores, stats, commentary
- Handles match_state phase transitions correctly
- Sends agent trash-talk messages via `env.send_message()`
- Generates commentary at phase transitions
- Stops when match_state phase is "match_end"

**Tests to Write**:
- `test_runner_instantiation`: can create with two heuristic agents
- `test_runner_builds_result_dict`: mock envs, verify result structure
- `test_runner_sends_messages`: verify send_message called when agent returns message
- `test_runner_stops_at_match_end`: verify loop exits on match_end phase

**Verification**: `python3 -m pytest python/tests/test_runner.py -v && echo PASS`

### Task 5: Package Integration and Exports

**Files**: `python/echomap_client/__init__.py`, `python/setup.py`
**Test Files**: `python/tests/test_agents.py` (append)

**Description**: Add new modules to package exports. Add `anthropic` to `extras_require["llm"]` in setup.py. Update `__init__.py` to export agent classes.

**Acceptance Criteria**:
- `from echomap_client import HeuristicBoxingAgent, BoxingMatchRunner` works
- `pip install echomap-client[llm]` would install anthropic
- All existing tests still pass

**Tests to Write**:
- `test_import_agents`: verify HeuristicBoxingAgent importable from package
- `test_import_runner`: verify BoxingMatchRunner importable
- `test_import_commentary`: verify MatchCommentary importable

**Verification**: `python3 -m pytest python/tests/ -v && echo PASS`

### Task 6: CLI Entry Point

**Files**: `python/echomap_client/cli.py`, `python/setup.py` (entry_points)

**Description**: Add `echomap-boxing` CLI command that runs a match:
- `echomap-boxing --mode heuristic` — two heuristic agents fight
- `echomap-boxing --mode llm` — two LLM agents fight (needs API key)
- `echomap-boxing --mode mixed` — one LLM vs one heuristic
- Options: `--host`, `--port`, `--rounds`, `--verbose`
- Prints match progress, trash-talk, and final results

**Acceptance Criteria**:
- `python -m echomap_client.cli --help` shows usage
- `--mode heuristic` creates two HeuristicBoxingAgents
- `--mode llm` creates two LLMBoxingAgents
- `--mode mixed` creates one of each
- Prints commentary and match result at end

**Tests to Write**:
- `test_cli_help`: verify --help doesn't crash
- `test_cli_parse_args_heuristic`: verify arg parsing for heuristic mode
- `test_cli_parse_args_llm`: verify arg parsing for llm mode

**Verification**: `python3 -m echomap_client.cli --help && python3 -m pytest python/tests/ -v && echo PASS`

## Integration Tests

### Full Heuristic Match (offline)
In `python/tests/test_agents.py`:
- `test_heuristic_match_sequence`: Create two HeuristicBoxingAgents, feed them a sequence of mock observations simulating a 3-round match (fighting → round_end → fighting → match_end). Verify both agents produce valid actions throughout, trash-talk appears, and neither crashes.

### LLM Agent with Mock API
In `python/tests/test_agents.py`:
- `test_llm_full_sequence_mocked`: Create LLMBoxingAgent with mocked anthropic client. Feed 10 steps of observations. Verify valid actions returned, trash-talk generated, no exceptions.

## Verification Gate

```bash
python3 -m pytest python/tests/test_agents.py -v
python3 -m pytest python/tests/test_commentary.py -v
python3 -m pytest python/tests/test_runner.py -v
python3 -m pytest python/tests/ -v
cargo test
```

All must exit 0.

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 8/10 | None |
| Design/Architecture | 7.2/10 | None |
| Engineering | 7/10 | None (adjusted — several findings based on wrong codebase) |

## Review Feedback Applied

1. **Runner uses EchoMapEnv** — clarified: runner drives env.reset()/step()/send_message(), no raw sockets
2. **LLM fallback by composition** — LLMBoxingAgent holds a HeuristicBoxingAgent instance for fallback
3. **Defensive dict access** — all observation/info access uses .get() with defaults
4. **Edge case tests added** — missing keys, float boundaries, CLI arg validation
5. **Match log output** — runner.run() returns structured dict usable by deliverable 6
6. **decide() string return** — this IS the trash-talk message, core to the spec requirement (not commentary)

## Open Questions

None — all questions self-resolved from codebase evidence.
