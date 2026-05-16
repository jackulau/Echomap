"""Reference EchoMap plugin.

Demonstrates the minimum surface third parties need to ship a plugin:
- An agent class with a `decide()` method (registered under
  `echomap.plugins.agents`).
- A scenario factory function (registered under
  `echomap.plugins.scenarios`).

See `docs/PLUGINS.md` for the authoring guide.
"""

from __future__ import annotations


class NoOpAgent:
    """Trivial agent that returns zero motor velocities.

    Plugin authors typically wrap an LLM, a controller, or a learned policy.
    This is the minimum shape the loader accepts.
    """

    def __init__(self, name: str = "example"):
        self.name = name

    def decide(self, observation, info):
        n = 0
        if isinstance(observation, dict):
            joints = observation.get("joint_positions") or []
            n = len(joints)
        return {"motor_velocities": [0.0] * n, "gripper_commands": []}, None


def make_example_scenario(**kwargs):
    """Return a tiny scenario descriptor.

    Real scenarios construct a Rust-side scene via the protocol or
    Python-side simulation harness. This stub returns a plain dict so
    the plugin loader has something validated to expose.
    """
    return {
        "name": "example_scenario",
        "description": "Reference scenario used by the EchoMap plugin example.",
        "kwargs": dict(kwargs),
    }


__all__ = ["NoOpAgent", "make_example_scenario"]
