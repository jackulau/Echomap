"""Tests for match commentary generator.

Run with: python3 -m pytest python/tests/test_commentary.py -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from echomap_client.commentary import MatchCommentary


def _make_match_state(current_round=2, total_a=20, total_b=19, opp_health=65.0):
    return {
        "phase": "round_end_5.0",
        "current_round": current_round,
        "round_time": 180.0,
        "round_duration": 180.0,
        "scores": [[10, 9], [10, 10]],
        "total_score_a": total_a,
        "total_score_b": total_b,
        "your_robot": 0,
        "opponent_health": opp_health,
        "opponent_stamina": 80.0,
    }


class TestRoundSummary(unittest.TestCase):
    def setUp(self):
        self.commentary = MatchCommentary(use_llm=False)

    def test_round_summary_template(self):
        ms = _make_match_state(current_round=1)
        result = self.commentary.generate_round_summary(ms)
        self.assertIn("Round 1", result)
        self.assertIn("10-9", result)

    def test_round_summary_includes_stats(self):
        ms = _make_match_state(opp_health=45.0)
        events = [{"type": "hit"}, {"type": "hit"}, {"type": "miss"}]
        result = self.commentary.generate_round_summary(ms, events)
        self.assertIn("45", result)
        self.assertIn("2", result)

    def test_round_summary_no_match_state(self):
        result = self.commentary.generate_round_summary(None)
        self.assertEqual(result, "Round complete.")

    def test_round_summary_empty_scores(self):
        ms = _make_match_state()
        ms["scores"] = []
        result = self.commentary.generate_round_summary(ms)
        self.assertIn("Round", result)


class TestMatchSummary(unittest.TestCase):
    def setUp(self):
        self.commentary = MatchCommentary(use_llm=False)

    def test_match_summary_template_with_winner(self):
        ms = _make_match_state(total_a=30, total_b=27)
        result = self.commentary.generate_match_summary(ms)
        self.assertIn("Robot A wins", result)
        self.assertIn("30-27", result)

    def test_match_summary_template_b_wins(self):
        ms = _make_match_state(total_a=27, total_b=30)
        result = self.commentary.generate_match_summary(ms)
        self.assertIn("Robot B wins", result)

    def test_match_summary_template_draw(self):
        ms = _make_match_state(total_a=30, total_b=30)
        result = self.commentary.generate_match_summary(ms)
        self.assertIn("draw", result)

    def test_match_summary_no_match_state(self):
        result = self.commentary.generate_match_summary(None)
        self.assertEqual(result, "Match complete.")

    def test_commentary_without_api_key(self):
        commentary = MatchCommentary(use_llm=True)
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            ms = _make_match_state()
            result = commentary.generate_round_summary(ms)
            self.assertIn("Round", result)
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
