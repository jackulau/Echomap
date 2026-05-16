# Agents — Bind Any Agent to Any Object

EchoMap exposes a uniform binding surface so the *same* agent class drives:

1. A boxing humanoid in the Rust simulator.
2. A generic n-DOF arm in the same simulator.
3. A real-hardware robot arm via the `echomap_client.hardware` bridge.

All three share the same observation / action schema and the same one-call
front door: `connect_agent(target_id, agent)`.

## 1. Bind an agent to a boxing robot

Start a server (`cargo build --release --bin echomap_server && ROUND_DURATION=30 NUM_ROUNDS=1 ./target/release/echomap_server`), then:

```python
from echomap_client.env import connect_agent
from echomap_client.agents import HeuristicBoxingAgent

agent = HeuristicBoxingAgent(name="Slugger")
env = connect_agent(target_id="robot/0", agent=agent, host="localhost", port=9002)

state, info = env.reset()
for _ in range(200):
    action, msg = agent.decide(state, info)
    state, reward, done, info = env.step(action)
    if done:
        break
env.close()
```

`env.capabilities` advertises what the bound target supports
(`["observe", "step", "motors", "sensors", "messaging"]` for a boxing
humanoid).

## 2. Bind the same agent to a generic arm

Any robot exposed by the running scenario can be bound by its `target_id`.
The boxing scenario currently ships two slots (`robot/0`, `robot/1`).
Plugins can register additional scenarios — see [PLUGINS.md](PLUGINS.md).

```python
from echomap_client.env import connect_agent

env = connect_agent(target_id="robot/1", host="localhost", port=9002)
print(env.observation_space)  # advertises joint counts, sensor counts, etc.
print(env.action_space)       # num_motors, num_grippers
```

Same `step()` / `reset()` / `observe()` API regardless of what `target_id`
resolves to.

## 3. Point the same agent at real hardware

The `hardware` package implements the bridge with the same surface:

```python
from echomap_client.hardware import MockArm, RobotArmBridge

agent = MyAgent()  # uses the same decide(observation, info) protocol
arm = MockArm(num_joints=6)            # swap for SerialArm("/dev/tty.usbserial-...")
bridge = RobotArmBridge(backend=arm)

obs, info = bridge.reset()
for _ in range(100):
    action, _ = agent.decide(obs, info)
    obs, reward, done, info = bridge.step(action)
```

The `SerialArm` backend documents a Dynamixel-style framing format but
ships as a stub — install a vendor driver via the
`echomap.plugins.hardware` entry-point group (see PLUGINS.md) for actual
serial I/O.

## Observation schema

All bindings (sim or hardware) return observations shaped like:

```python
{
    "joint_positions": [float, ...],    # rad, one per joint
    "joint_velocities": [float, ...],   # rad/s
    "sensor_readings": {...},           # optional, varies by target
    # boxing-domain only:
    "combat": {...},                    # health, stamina, recent hits
}
```

Action shape:

```python
{
    "motor_velocities": [float, ...],   # rad/s, one per joint
    "gripper_commands": [bool, ...],    # optional
    "base_velocity": [float, float],    # optional, planar humanoid only
}
```

Capability strings on `env.capabilities` tell you what's actually wired:
- `motors`, `grippers`, `sensors`, `observe`, `step`, `messaging`

## Writing a new agent

Inherit one of the existing agent base classes (`BoxingAgent` for boxing,
or just supply any object with a `decide(observation, info) -> (action, msg)`
method for the hardware bridge or a generic binding):

```python
class MyAgent:
    def decide(self, observation, info):
        joints = observation.get("joint_positions") or []
        vels = [0.05 * len(joints)] * len(joints)
        return {"motor_velocities": vels, "gripper_commands": []}, None
```

Register it as a plugin so others can use it — see
[PLUGINS.md](PLUGINS.md).

## Migration from the legacy `Connect` API

Older callers used `EchoMapEnv(robot_id=0)` which sends a `Connect`
message. That path still works (back-compat). New code should use
`connect_agent(target_id="robot/0")` — it goes through the `BindTarget`
protocol, returns `Bound` with capabilities, and is forward-compatible
with non-robot targets (sensors, props, future scenarios).
