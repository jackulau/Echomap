"""Tests for boxing match runner.

Run with: python3 -m pytest python/tests/test_runner.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from echomap_client.agents import HeuristicBoxingAgent
from echomap_client.runner import BoxingMatchRunner
from echomap_client.commentary import MatchCommentary


def _mock_env(phase_sequence):
    """Create a mock EchoMapEnv that cycles through phases."""
    env = MagicMock()
    env.connect = MagicMock()
    env.close = MagicMock()
    env.send_message = MagicMock()

    step_idx = [0]

    def mock_reset():
        return {}, {"messages": [], "match_state": {"phase": "countdown_3.0"}}

    def mock_step(action):
        idx = min(step_idx[0], len(phase_sequence) - 1)
        phase = phase_sequence[idx]
        step_idx[0] += 1
        ms = {
            "phase": phase,
            "current_round": 1,
            "round_time": 10.0,
            "round_duration": 180.0,
            "scores": [[10, 9]],
            "total_score_a": 10,
            "total_score_b": 9,
            "your_robot": 0,
            "opponent_health": 80.0,
            "opponent_stamina": 90.0,
        }
        done = phase == "match_end"
        return {}, 0.0, done, {"messages": [], "match_state": ms}

    env.reset = MagicMock(side_effect=mock_reset)
    env.step = MagicMock(side_effect=mock_step)
    return env


class TestRunnerInstantiation(unittest.TestCase):
    def test_runner_instantiation(self):
        a = HeuristicBoxingAgent(name="A")
        b = HeuristicBoxingAgent(name="B")
        runner = BoxingMatchRunner(a, b)
        self.assertIsNotNone(runner)
        self.assertEqual(runner.host, "localhost")
        self.assertEqual(runner.port, 9002)


class TestRunnerBehavior(unittest.TestCase):
    @patch("echomap_client.runner.EchoMapEnv")
    def test_runner_builds_result_dict(self, MockEnv):
        phases = ["fighting"] * 5 + ["round_end_5.0"] * 2 + ["match_end"]
        mock_a = _mock_env(phases)
        mock_b = _mock_env(phases)
        MockEnv.side_effect = [mock_a, mock_b]

        a = HeuristicBoxingAgent(name="A", trash_talk_chance=0.0)
        b = HeuristicBoxingAgent(name="B", trash_talk_chance=0.0)
        runner = BoxingMatchRunner(a, b)
        result = runner.run(max_steps=100)

        self.assertIn("winner", result)
        self.assertIn("scores", result)
        self.assertIn("stats", result)
        self.assertIn("commentary", result)
        self.assertIn("steps", result["stats"])

    @patch("echomap_client.runner.EchoMapEnv")
    def test_runner_sends_messages(self, MockEnv):
        phases = ["fighting"] * 3 + ["match_end"]
        mock_a = _mock_env(phases)
        mock_b = _mock_env(phases)
        MockEnv.side_effect = [mock_a, mock_b]

        a = HeuristicBoxingAgent(name="A", trash_talk_chance=1.0)
        b = HeuristicBoxingAgent(name="B", trash_talk_chance=1.0)
        runner = BoxingMatchRunner(a, b)
        result = runner.run(max_steps=100)

        self.assertGreater(mock_a.send_message.call_count, 0)
        self.assertGreater(mock_b.send_message.call_count, 0)

    @patch("echomap_client.runner.EchoMapEnv")
    def test_runner_stops_at_match_end(self, MockEnv):
        phases = ["fighting"] * 2 + ["match_end"]
        mock_a = _mock_env(phases)
        mock_b = _mock_env(phases)
        MockEnv.side_effect = [mock_a, mock_b]

        a = HeuristicBoxingAgent(name="A", trash_talk_chance=0.0)
        b = HeuristicBoxingAgent(name="B", trash_talk_chance=0.0)
        runner = BoxingMatchRunner(a, b)
        result = runner.run(max_steps=1000)

        self.assertLessEqual(result["stats"]["steps"], 5)

    @patch("echomap_client.runner.EchoMapEnv")
    def test_runner_generates_commentary(self, MockEnv):
        phases = ["fighting"] * 3 + ["round_end_5.0"] * 2 + ["match_end"]
        mock_a = _mock_env(phases)
        mock_b = _mock_env(phases)
        MockEnv.side_effect = [mock_a, mock_b]

        a = HeuristicBoxingAgent(name="A", trash_talk_chance=0.0)
        b = HeuristicBoxingAgent(name="B", trash_talk_chance=0.0)
        runner = BoxingMatchRunner(a, b)
        result = runner.run(max_steps=100)

        self.assertGreater(len(result["commentary"]), 0)


class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        from echomap_client.cli import parse_args
        import io
        from contextlib import redirect_stdout, redirect_stderr
        with self.assertRaises(SystemExit) as ctx:
            parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_parse_args_heuristic(self):
        from echomap_client.cli import parse_args
        args = parse_args(["--mode", "heuristic", "--port", "9999"])
        self.assertEqual(args.mode, "heuristic")
        self.assertEqual(args.port, 9999)

    def test_cli_parse_args_llm(self):
        from echomap_client.cli import parse_args
        args = parse_args(["--mode", "llm", "--verbose"])
        self.assertEqual(args.mode, "llm")
        self.assertTrue(args.verbose)

    def test_cli_parse_args_mixed(self):
        from echomap_client.cli import parse_args
        args = parse_args(["--mode", "mixed"])
        self.assertEqual(args.mode, "mixed")

    def test_cli_create_agents_heuristic(self):
        from echomap_client.cli import create_agents
        a, b = create_agents("heuristic")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_cli_create_agents_llm(self):
        from echomap_client.cli import create_agents
        a, b = create_agents("llm")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_cli_create_agents_mixed(self):
        from echomap_client.cli import create_agents
        a, b = create_agents("mixed")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)


if __name__ == "__main__":
    unittest.main()
