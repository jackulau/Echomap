# Agent Platform Audit — Goal 003

Scope: enumerate current agent surface, identify holes that block arbitrary-object binding, and sketch the target API for `connect_agent(target_id, agent)`. Derived from reading `src/agent/`, `python/echomap_client/`, and `src/robot/`.

## Current Protocol (src/agent/protocol.rs)

`ClientMessage` variants:
- `Connect { robot_id: usize }` — claims a robot by index. Hardcoded assumption: targets live in a single `RobotManager::robots` Vec.
- `Reset` — reset bound robot.
- `Step { action: RobotAction }` — motor velocities + gripper commands.
- `Observe` — read observation without stepping sim.
- `Close` — disconnect.
- `SendMessage { to_robot_id: usize, content }` — inter-robot chat.

`ServerMessage` variants:
- `Connected { observation_space, action_space }`
- `Observation { state, reward, done, step_count, messages, hit_events, match_state }`
- `MessageSent`, `Error { message }`, `Closed`

Hardcodings:
- `usize` robot IDs everywhere — index shifts when robots removed.
- `match_state` + `hit_events` are boxing-domain fields living on the generic Observation envelope.

## Current Python API (python/echomap_client/)

`EchoMapEnv.__init__(host, port, robot_id=0, read_timeout=30.0)` — robot_id is an int, frozen at construction (env.py:27-43). No rebind.

Agents inherit `BoxingAgent` ABC and implement `decide(observation, info) -> (action_dict, optional_message)`. All three (`HeuristicBoxingAgent`, `LLMBoxingAgent`, `OllamaBoxingAgent`) share the same shape — Python side is symmetric. The asymmetry is below, at the Rust protocol layer, which only speaks boxing.

No `connect_agent()` wrapper exists. Caller does:
```
env = EchoMapEnv(robot_id=0)
env.connect()
agent = OllamaBoxingAgent(...)
while not done:
    obs, info = env.observe()
    action, msg = agent.decide(obs, info)
    env.step(action)
```

## Bindable Targets (src/robot/, src/scenarios/)

- `RobotDefinition::boxing_humanoid()` — dual-arm torso, combat state, body zones.
- `RobotDefinition::boxing_test_robot()` — 3-link test variant.
- `RobotDefinition::simple_arm(n)` — generic n-DOF revolute arm (the target for D2's "generic arm" path).
- Any robot can mount `SensorMount` (distance, lidar, contact, IMU). `env.py` does not expose sensor-only entities.
- Non-robot dynamic objects (props, balls, flags) are not currently agent-bindable.

## Holes Blocking Goal

1. **Opaque target_id missing.** `Connect { robot_id: usize }` cannot address sensors, props, or stably-named robots. Need `target_id: String` (e.g., `"robot/0"`, `"robot/boxer_a"`, `"sensor/camera_1"`).
2. **Python `EchoMapEnv` couples target to construction.** Should accept `target_id` string and store it for routing. `connect_agent(target_id, agent)` becomes the one-call front door.
3. **Protocol carries domain fields on generic envelope.** `match_state`, `hit_events` should move to an opaque `domain_data: Option<Value>` or be sent only when the bound target's domain matches.
4. **No capability negotiation at bind time.** A sensor-only entity has 0 motors — current `ActionSpace` cannot represent that cleanly. Bind response should advertise capabilities.
5. **No session multiplexing.** One socket = one target. Multi-target single-agent (e.g., dual-arm whole-body control) requires multiple connections.
6. **Observation schema mismatch.** `state` dict assumes joints + gripper + combat. Pure-sensor or non-robot targets need sparse/optional fields.

## API Sketch for `connect_agent`

Python front door:
```python
def connect_agent(
    self,
    target_id: str,                 # "robot/0", "sensor/camera_1", "entity/ball"
    agent: Optional[Agent] = None,
    agent_type: str = "default",    # hint: "boxer", "manipulator", "observer"
    domain: Optional[str] = None,   # "boxing", "manipulation", "acoustic"
    observe_only: bool = False,
) -> tuple[ObservationSpace, ActionSpace]:
    ...
```

Rust protocol addition:
```rust
pub enum ClientMessage {
    BindTarget {
        target_id: String,
        agent_type: String,
        domain: Option<String>,
        observe_only: bool,
    },
    // existing Connect kept for back-compat, deprecated
}

pub enum ServerMessage {
    Bound {
        target_id: String,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        capabilities: Vec<String>,    // e.g. ["motors", "sensors", "messaging"]
        domain_schema: Option<Value>, // domain-specific keys advertised for `state.domain_data`
    },
}
```

Session-side (session.rs): replace `robot_id: usize` field on `AgentSession` with `target_id: Option<String>` and a resolver that maps the string to whatever it points to (robot index, sensor handle, prop id). Boxing slot mapping ("robot/boxer_a" → robot 0) lives in a single `TargetResolver` so future scenarios can register new prefixes.

Bridge: route observations via `TargetResolver::observe(target_id)` instead of `robots[i]`. Domain-specific fields (boxing match_state) populated only when the bound target's resolved domain matches.

## Non-Goals (this audit)

- Implementing the resolver — that's D2.
- Backward-incompatible removal of `Connect { robot_id }` — keep as deprecated alias for one cycle.
- Plugin system — separate deliverable (D4).
