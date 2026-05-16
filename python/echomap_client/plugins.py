"""EchoMap plugin system.

Third parties can extend EchoMap without forking by registering Python
entry points under four groups:

    [project.entry-points."echomap.plugins.agents"]
    my_agent = "mypkg:MyAgent"

    [project.entry-points."echomap.plugins.sensors"]
    my_sensor = "mypkg:MySensor"

    [project.entry-points."echomap.plugins.scenarios"]
    my_scenario = "mypkg:make_scenario"

    [project.entry-points."echomap.plugins.visualizations"]
    my_viz = "mypkg:render"

`load_all()` walks all four groups, validates each entry, and returns a
``PluginRegistry``. Validation is light — each loaded object must be
callable or have the documented attribute for its kind. Schema errors are
reported but never abort the whole load.

Usage:

    from echomap_client.plugins import load_all
    reg = load_all()
    for name, agent_cls in reg.agents.items():
        ...
"""

from __future__ import annotations

import importlib.metadata as _metadata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


GROUP_AGENTS = "echomap.plugins.agents"
GROUP_SENSORS = "echomap.plugins.sensors"
GROUP_SCENARIOS = "echomap.plugins.scenarios"
GROUP_VISUALIZATIONS = "echomap.plugins.visualizations"
GROUP_HARDWARE = "echomap.plugins.hardware"

KNOWN_GROUPS = [
    GROUP_AGENTS,
    GROUP_SENSORS,
    GROUP_SCENARIOS,
    GROUP_VISUALIZATIONS,
    GROUP_HARDWARE,
]


@dataclass
class PluginRegistry:
    """Holds all discovered plugins keyed by group then by entry-point name."""

    agents: Dict[str, Any] = field(default_factory=dict)
    sensors: Dict[str, Any] = field(default_factory=dict)
    scenarios: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def get(self, group: str) -> Dict[str, Any]:
        return {
            GROUP_AGENTS: self.agents,
            GROUP_SENSORS: self.sensors,
            GROUP_SCENARIOS: self.scenarios,
            GROUP_VISUALIZATIONS: self.visualizations,
            GROUP_HARDWARE: self.hardware,
        }[group]

    def is_empty(self) -> bool:
        return not any(
            [
                self.agents,
                self.sensors,
                self.scenarios,
                self.visualizations,
                self.hardware,
            ]
        )

    def summary(self) -> str:
        lines = ["EchoMap plugins:"]
        groups = [
            ("agents", self.agents),
            ("sensors", self.sensors),
            ("scenarios", self.scenarios),
            ("visualizations", self.visualizations),
            ("hardware", self.hardware),
        ]
        for label, bucket in groups:
            if bucket:
                lines.append(f"  [{label}]")
                for name, target in bucket.items():
                    mod = getattr(target, "__module__", "<unknown>")
                    qual = getattr(target, "__qualname__", repr(target))
                    lines.append(f"    {name} -> {mod}:{qual}")
            else:
                lines.append(f"  [{label}] (none)")
        if self.errors:
            lines.append("  [errors]")
            for err in self.errors:
                lines.append(f"    ! {err}")
        return "\n".join(lines)


def _validate(group: str, name: str, obj: Any) -> Optional[str]:
    """Light schema check per group. Returns None if OK, error string otherwise."""
    if group == GROUP_AGENTS:
        # Either a class with `decide` method or an instance with `.decide` is OK.
        target = obj if isinstance(obj, type) else type(obj)
        if not hasattr(target, "decide"):
            return f"agent '{name}' missing required `decide(observation, info)` method"
    elif group == GROUP_SCENARIOS:
        if not callable(obj):
            return f"scenario '{name}' must be callable (factory function)"
    elif group == GROUP_VISUALIZATIONS:
        if not callable(obj):
            return f"visualization '{name}' must be callable"
    elif group == GROUP_SENSORS:
        target = obj if isinstance(obj, type) else type(obj)
        if not hasattr(target, "read"):
            return f"sensor '{name}' missing required `read()` method"
    elif group == GROUP_HARDWARE:
        target = obj if isinstance(obj, type) else type(obj)
        for attr in ("reset", "read_state", "apply_action"):
            if not hasattr(target, attr):
                return f"hardware backend '{name}' missing required `{attr}` method"
    # Unknown group → no validation
    return None


def _iter_entry_points(group: str):
    """Compat wrapper across importlib.metadata API drift (3.8 → 3.12)."""
    eps = _metadata.entry_points()
    select = getattr(eps, "select", None)
    if select is not None:
        return list(eps.select(group=group))
    # Pre-3.10 returned a dict.
    return list(eps.get(group, []))  # type: ignore[attr-defined]


def load_all(groups: Optional[List[str]] = None) -> PluginRegistry:
    """Discover and load every entry point under the known groups."""
    reg = PluginRegistry()
    targets = groups or KNOWN_GROUPS
    for group in targets:
        bucket = reg.get(group)
        for ep in _iter_entry_points(group):
            name = ep.name
            try:
                obj = ep.load()
            except Exception as exc:  # noqa: BLE001 — defensive: never abort loader
                reg.errors.append(f"{group}/{name}: load failed: {exc!r}")
                continue
            err = _validate(group, name, obj)
            if err:
                reg.errors.append(err)
                continue
            bucket[name] = obj
    return reg


def list_plugins() -> str:
    """Convenience for the CLI — load and return the summary string."""
    return load_all().summary()


def register_in_process(
    group: str,
    name: str,
    obj: Any,
    *,
    registry: Optional[PluginRegistry] = None,
) -> PluginRegistry:
    """In-process registration shortcut for tests + advanced embedding.

    Skips entry-point discovery — useful when you want to wire a plugin
    without installing a package.
    """
    if group not in KNOWN_GROUPS:
        raise ValueError(f"unknown plugin group: {group}")
    reg = registry or PluginRegistry()
    err = _validate(group, name, obj)
    if err:
        raise ValueError(err)
    reg.get(group)[name] = obj
    return reg
