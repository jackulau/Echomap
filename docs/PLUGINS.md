# Plugins — Extend EchoMap Without Forking

## Setup

The `echomap_client` package isn't published — install it from the
repo before any of the commands below work:

```
pip install -e python/
# optional: also install the reference example plugin
pip install -e python/examples/echomap_plugin_example
```

The bundled smoke scripts set `PYTHONPATH=python` instead of installing,
so they work in a fresh clone without a `pip install` step.

## Plugin groups

EchoMap discovers extensions via Python entry points. Five plugin groups
are supported today:

| Group | What it registers | Required interface |
|---|---|---|
| `echomap.plugins.agents` | Agent classes/instances | `.decide(observation, info) -> (action, msg)` |
| `echomap.plugins.sensors` | Sensor backends | `.read() -> dict` |
| `echomap.plugins.scenarios` | Scenario factory functions | Callable returning a scenario descriptor |
| `echomap.plugins.visualizations` | Custom renderers/overlays | Callable accepting scene state |
| `echomap.plugins.hardware` | Real-arm backends | `.reset()`, `.read_state()`, `.apply_action()` |

## Author a plugin (10 minutes)

1. Create a Python package:

   ```
   mypackage/
     pyproject.toml
     src/mypackage/__init__.py
   ```

2. Declare entry points in `pyproject.toml`:

   ```toml
   [project]
   name = "mypackage"
   version = "0.1.0"

   [project.entry-points."echomap.plugins.agents"]
   my_agent = "mypackage:MyAgent"

   [project.entry-points."echomap.plugins.scenarios"]
   my_scenario = "mypackage:make_scenario"
   ```

3. Implement the entry-point targets in `src/mypackage/__init__.py`:

   ```python
   class MyAgent:
       def decide(self, observation, info):
           return {"motor_velocities": [0.0]}, None

   def make_scenario(**kwargs):
       return {"name": "my_scenario", "kwargs": kwargs}
   ```

4. Install editable:

   ```
   pip install -e ./mypackage
   ```

5. Confirm discovery:

   ```
   python3 -m echomap_client.cli list-plugins
   ```

   The output lists your plugin under the right group. JSON form:
   `python3 -m echomap_client.cli list-plugins --json`.

## Reference example

See `python/examples/echomap_plugin_example/` — minimal plugin package
that registers `NoOpAgent` and `make_example_scenario`. Install it with:

```
pip install -e python/examples/echomap_plugin_example
```

## Loader semantics

`echomap_client.plugins.load_all()`:
- Walks every known entry-point group.
- Loads each registration in isolation — a single broken plugin does NOT
  abort the whole load. Errors land in `registry.errors`.
- Validates the bare minimum schema per group (e.g. agents need
  `decide`, hardware backends need `reset`/`read_state`/`apply_action`).

In-process registration (handy for tests):

```python
from echomap_client.plugins import register_in_process, GROUP_AGENTS
register_in_process(GROUP_AGENTS, "scratch", MyAgent)
```

## Hardware backend plugins

A vendor driver for, say, a Dynamixel arm should:

```toml
[project.entry-points."echomap.plugins.hardware"]
dynamixel = "echomap_driver_dynamixel:DynamixelArm"
```

The class must implement the same `reset` / `read_state` / `apply_action`
surface as `MockArm` / `SerialArm`. The bridge can then accept it
unmodified:

```python
from echomap_client.plugins import load_all
from echomap_client.hardware import RobotArmBridge

reg = load_all()
arm_cls = reg.hardware["dynamixel"]
arm = arm_cls(port="/dev/tty.usbserial-A1")
bridge = RobotArmBridge(backend=arm)
```

## Don't yet

- Plugin marketplace / signing — out of scope for v1.
- Per-plugin permission scopes — plugins run with full user privileges;
  install only trusted packages.
- Hot reload — install/restart cycle only.
