"""Tests for boxing agents.

Works with both pytest and unittest. Run with:
    python3 -m pytest python/tests/test_agents.py -v
"""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from echomap_client.agents import (
    BoxingAgent,
    HeuristicBoxingAgent,
    clamp_velocities,
    NECK_LIMIT,
    SHOULDER_LIMIT,
)


def _make_obs(health=100.0):
    return {"combat": {"health": health, "stamina": 100.0}}


def _make_info(phase="fighting", opponent_health=100.0):
    return {
        "messages": [],
        "match_state": {
            "phase": phase,
            "current_round": 1,
            "round_time": 10.0,
            "round_duration": 180.0,
            "scores": [],
            "total_score_a": 0,
            "total_score_b": 0,
            "your_robot": 0,
            "opponent_health": opponent_health,
            "opponent_stamina": 100.0,
        },
    }


class TestBoxingAgentBase(unittest.TestCase):
    def test_base_agent_not_instantiable(self):
        with self.assertRaises(TypeError):
            BoxingAgent()


class TestHeuristicAgent(unittest.TestCase):
    def setUp(self):
        self.agent = HeuristicBoxingAgent(trash_talk_chance=0.0)

    def test_heuristic_returns_valid_action(self):
        obs = _make_obs()
        info = _make_info()
        action, msg = self.agent.decide(obs, info)
        self.assertIn("motor_velocities", action)
        self.assertEqual(len(action["motor_velocities"]), 3)
        for v in action["motor_velocities"]:
            self.assertIsInstance(v, float)

    def test_heuristic_velocities_in_range(self):
        obs = _make_obs()
        info = _make_info()
        for _ in range(50):
            action, _ = self.agent.decide(obs, info)
            vels = action["motor_velocities"]
            self.assertGreaterEqual(vels[0], -NECK_LIMIT)
            self.assertLessEqual(vels[0], NECK_LIMIT)
            self.assertGreaterEqual(vels[1], -SHOULDER_LIMIT)
            self.assertLessEqual(vels[1], SHOULDER_LIMIT)
            self.assertGreaterEqual(vels[2], -SHOULDER_LIMIT)
            self.assertLessEqual(vels[2], SHOULDER_LIMIT)

    def test_heuristic_idle_during_countdown(self):
        obs = _make_obs()
        info = _make_info(phase="countdown_3.0")
        action, msg = self.agent.decide(obs, info)
        self.assertEqual(action["motor_velocities"], [0.0, 0.0, 0.0])

    def test_heuristic_idle_during_round_end(self):
        obs = _make_obs()
        info = _make_info(phase="round_end_5.0")
        action, msg = self.agent.decide(obs, info)
        self.assertEqual(action["motor_velocities"], [0.0, 0.0, 0.0])

    def test_heuristic_idle_during_match_end(self):
        obs = _make_obs()
        info = _make_info(phase="match_end")
        action, msg = self.agent.decide(obs, info)
        self.assertEqual(action["motor_velocities"], [0.0, 0.0, 0.0])

    def test_heuristic_attacks_when_healthy(self):
        obs = _make_obs(health=90.0)
        info = _make_info(opponent_health=50.0)
        nonzero_found = False
        for _ in range(10):
            action, _ = self.agent.decide(obs, info)
            vels = action["motor_velocities"]
            if any(abs(v) > 0.5 for v in vels[1:]):
                nonzero_found = True
                break
        self.assertTrue(nonzero_found, "Agent should attack with arm velocities")

    def test_heuristic_defends_when_low_health(self):
        obs = _make_obs(health=15.0)
        info = _make_info(opponent_health=90.0)
        max_arm_vel = 0.0
        for _ in range(20):
            action, _ = self.agent.decide(obs, info)
            vels = action["motor_velocities"]
            max_arm_vel = max(max_arm_vel, abs(vels[1]), abs(vels[2]))
        self.assertLess(max_arm_vel, SHOULDER_LIMIT * 0.8, "Defensive mode should use smaller arm movements")

    def test_heuristic_trash_talk_is_string_or_none(self):
        agent = HeuristicBoxingAgent(trash_talk_chance=1.0)
        obs = _make_obs()
        info = _make_info()
        _, msg = agent.decide(obs, info)
        self.assertIsInstance(msg, str)

        agent_quiet = HeuristicBoxingAgent(trash_talk_chance=0.0)
        _, msg2 = agent_quiet.decide(obs, info)
        self.assertIsNone(msg2)

    def test_heuristic_handles_missing_observation(self):
        info = _make_info()
        action, _ = self.agent.decide(None, info)
        self.assertEqual(len(action["motor_velocities"]), 3)

    def test_heuristic_handles_missing_info(self):
        obs = _make_obs()
        action, _ = self.agent.decide(obs, None)
        self.assertEqual(action["motor_velocities"], [0.0, 0.0, 0.0])

    def test_heuristic_handles_empty_match_state(self):
        obs = _make_obs()
        info = {"messages": [], "match_state": None}
        action, _ = self.agent.decide(obs, info)
        self.assertEqual(action["motor_velocities"], [0.0, 0.0, 0.0])


class TestClampVelocities(unittest.TestCase):
    def test_within_range_unchanged(self):
        vels = [0.5, 1.0, -1.0]
        result = clamp_velocities(vels)
        self.assertEqual(result, vels)

    def test_over_limit_clamped(self):
        vels = [5.0, 10.0, -10.0]
        result = clamp_velocities(vels)
        self.assertAlmostEqual(result[0], NECK_LIMIT)
        self.assertAlmostEqual(result[1], SHOULDER_LIMIT)
        self.assertAlmostEqual(result[2], -SHOULDER_LIMIT)

    def test_exact_boundary_passes(self):
        vels = [NECK_LIMIT, SHOULDER_LIMIT, -SHOULDER_LIMIT]
        result = clamp_velocities(vels)
        self.assertEqual(result[0], NECK_LIMIT)
        self.assertEqual(result[1], SHOULDER_LIMIT)
        self.assertEqual(result[2], -SHOULDER_LIMIT)


class TestHeuristicMatchSequence(unittest.TestCase):
    """Integration test: simulate a 3-round match with two heuristic agents."""

    def test_heuristic_match_sequence(self):
        agent_a = HeuristicBoxingAgent(name="AgentA", trash_talk_chance=0.2)
        agent_b = HeuristicBoxingAgent(name="AgentB", trash_talk_chance=0.2)

        phases = (
            [("fighting", 20)] +
            [("round_end_5.0", 3)] +
            [("fighting", 20)] +
            [("round_end_5.0", 3)] +
            [("fighting", 20)] +
            [("match_end", 2)]
        )

        trash_talk_count = 0
        for phase, steps in phases:
            for _ in range(steps):
                obs = _make_obs(health=80.0)
                info = _make_info(phase=phase, opponent_health=75.0)
                action_a, msg_a = agent_a.decide(obs, info)
                action_b, msg_b = agent_b.decide(obs, info)

                self.assertEqual(len(action_a["motor_velocities"]), 3)
                self.assertEqual(len(action_b["motor_velocities"]), 3)

                if msg_a:
                    self.assertIsInstance(msg_a, str)
                    trash_talk_count += 1
                if msg_b:
                    self.assertIsInstance(msg_b, str)
                    trash_talk_count += 1


class TestPackageExports(unittest.TestCase):
    def test_import_agents(self):
        from echomap_client import HeuristicBoxingAgent
        self.assertIsNotNone(HeuristicBoxingAgent)

    def test_import_runner(self):
        from echomap_client import BoxingMatchRunner
        self.assertIsNotNone(BoxingMatchRunner)

    def test_import_commentary(self):
        from echomap_client import MatchCommentary
        self.assertIsNotNone(MatchCommentary)


class TestLLMAgent(unittest.TestCase):
    def setUp(self):
        from echomap_client.llm_agent import LLMBoxingAgent
        self.LLMBoxingAgent = LLMBoxingAgent

    def test_llm_agent_instantiable(self):
        agent = self.LLMBoxingAgent()
        self.assertIsNotNone(agent)
        self.assertIsNone(agent._client)

    def test_llm_agent_builds_prompt(self):
        agent = self.LLMBoxingAgent()
        obs = _make_obs(health=75.0)
        info = _make_info(phase="fighting", opponent_health=60.0)
        info["messages"] = [{"content": "You're going down!"}]
        prompt = agent._build_prompt(obs, info)
        self.assertIn("fighting", prompt)
        self.assertIn("75.0", prompt)
        self.assertIn("60.0", prompt)
        self.assertIn("going down", prompt)

    def test_llm_agent_parses_valid_response(self):
        agent = self.LLMBoxingAgent()
        text = '{"motor_velocities": [0.1, 1.5, -1.5], "trash_talk": "Take that!"}'
        action, msg = agent._parse_response(text)
        self.assertEqual(action["motor_velocities"], [0.1, 1.5, -1.5])
        self.assertEqual(msg, "Take that!")

    def test_llm_agent_parses_markdown_wrapped(self):
        agent = self.LLMBoxingAgent()
        text = '```json\n{"motor_velocities": [0.1, 1.0, -1.0]}\n```'
        action, msg = agent._parse_response(text)
        self.assertEqual(len(action["motor_velocities"]), 3)

    def test_llm_agent_fallback_on_bad_json(self):
        agent = self.LLMBoxingAgent()
        action, msg = agent._parse_response("not json at all")
        self.assertEqual(len(action["motor_velocities"]), 3)

    def test_llm_agent_fallback_on_missing_keys(self):
        agent = self.LLMBoxingAgent()
        action, msg = agent._parse_response('{"foo": "bar"}')
        self.assertEqual(len(action["motor_velocities"]), 3)

    def test_llm_agent_fallback_on_wrong_length(self):
        agent = self.LLMBoxingAgent()
        action, msg = agent._parse_response('{"motor_velocities": [1.0, 2.0]}')
        self.assertEqual(len(action["motor_velocities"]), 3)

    def test_llm_agent_clamps_velocities(self):
        agent = self.LLMBoxingAgent()
        text = '{"motor_velocities": [5.0, 10.0, -10.0]}'
        action, _ = agent._parse_response(text)
        self.assertAlmostEqual(action["motor_velocities"][0], NECK_LIMIT)
        self.assertAlmostEqual(action["motor_velocities"][1], SHOULDER_LIMIT)
        self.assertAlmostEqual(action["motor_velocities"][2], -SHOULDER_LIMIT)

    def test_llm_agent_fallback_without_key(self):
        agent = self.LLMBoxingAgent()
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            obs = _make_obs()
            info = _make_info()
            action, msg = agent.decide(obs, info)
            self.assertEqual(len(action["motor_velocities"]), 3)
            self.assertFalse(agent._api_available)
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_llm_agent_handles_non_string_trash_talk(self):
        agent = self.LLMBoxingAgent()
        text = '{"motor_velocities": [0.1, 0.2, 0.3], "trash_talk": 42}'
        action, msg = agent._parse_response(text)
        self.assertIsNone(msg)


class TestLLMFullSequenceMocked(unittest.TestCase):
    """Integration test: LLM agent with mocked API over 10 steps."""

    def test_llm_full_sequence_mocked(self):
        from unittest.mock import MagicMock, patch
        from echomap_client.llm_agent import LLMBoxingAgent

        agent = LLMBoxingAgent()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"motor_velocities": [0.1, 2.0, -1.5], "trash_talk": "Eat this!"}')
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        agent._client = mock_client
        agent._api_available = True

        for step in range(10):
            obs = _make_obs(health=100.0 - step * 3)
            info = _make_info(phase="fighting", opponent_health=90.0)
            action, msg = agent.decide(obs, info)
            self.assertEqual(len(action["motor_velocities"]), 3)
            for v in action["motor_velocities"]:
                self.assertGreaterEqual(v, -SHOULDER_LIMIT)
                self.assertLessEqual(v, SHOULDER_LIMIT)

        self.assertEqual(mock_client.messages.create.call_count, 10)


if __name__ == "__main__":
    unittest.main()
