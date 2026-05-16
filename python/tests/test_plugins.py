"""Tests for the EchoMap plugin system (goal 003 D4)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPluginRegistry(unittest.TestCase):
    def test_load_all_returns_registry(self):
        from echomap_client.plugins import PluginRegistry, load_all
        reg = load_all()
        self.assertIsInstance(reg, PluginRegistry)

    def test_known_groups_exposed(self):
        from echomap_client.plugins import KNOWN_GROUPS, PluginRegistry
        reg = PluginRegistry()
        for group in KNOWN_GROUPS:
            self.assertIsInstance(reg.get(group), dict)

    def test_register_in_process_agent(self):
        from echomap_client.plugins import (
            GROUP_AGENTS,
            PluginRegistry,
            register_in_process,
        )

        class FakeAgent:
            def decide(self, observation, info):
                return {}, None

        reg = register_in_process(GROUP_AGENTS, "fake", FakeAgent)
        self.assertIn("fake", reg.agents)

    def test_register_invalid_agent_rejected(self):
        from echomap_client.plugins import GROUP_AGENTS, register_in_process

        class NoDecide:
            pass

        with self.assertRaises(ValueError):
            register_in_process(GROUP_AGENTS, "broken", NoDecide)

    def test_register_invalid_scenario_rejected(self):
        from echomap_client.plugins import GROUP_SCENARIOS, register_in_process
        with self.assertRaises(ValueError):
            register_in_process(GROUP_SCENARIOS, "not_callable", 42)

    def test_register_unknown_group_rejected(self):
        from echomap_client.plugins import register_in_process
        with self.assertRaises(ValueError):
            register_in_process("echomap.plugins.bogus", "foo", lambda: None)

    def test_summary_includes_all_groups(self):
        from echomap_client.plugins import PluginRegistry
        reg = PluginRegistry()
        summary = reg.summary()
        for label in ("agents", "sensors", "scenarios", "visualizations", "hardware"):
            self.assertIn(f"[{label}]", summary)

    def test_hardware_group_validation(self):
        from echomap_client.plugins import GROUP_HARDWARE, register_in_process

        class IncompleteArm:
            def reset(self):
                pass

        with self.assertRaises(ValueError):
            register_in_process(GROUP_HARDWARE, "incomplete", IncompleteArm)


class TestExamplePluginDiscovery(unittest.TestCase):
    """Skipped unless the example package is pip-installed."""

    def setUp(self):
        try:
            import echomap_plugin_example  # noqa: F401
        except ImportError:
            self.skipTest("echomap-plugin-example not installed")

    def test_example_agent_discovered(self):
        from echomap_client.plugins import load_all
        reg = load_all()
        self.assertIn("example_agent", reg.agents)

    def test_example_scenario_discovered(self):
        from echomap_client.plugins import load_all
        reg = load_all()
        self.assertIn("example_scenario", reg.scenarios)


if __name__ == "__main__":
    unittest.main()
