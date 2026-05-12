"""Boxing match runner that orchestrates agents and the simulation."""

from .commentary import MatchCommentary
from .env import EchoMapEnv


class BoxingMatchRunner:
    """Orchestrates a boxing match between two agents via EchoMapEnv."""

    def __init__(self, agent_a, agent_b, host="localhost", port=9002,
                 commentary=None, verbose=False):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.host = host
        self.port = port
        self.commentary = commentary or MatchCommentary(use_llm=False)
        self.verbose = verbose

    def run(self, max_steps=50000):
        """Run a full boxing match.

        Returns:
            dict with match results: winner, scores, stats, commentary.
        """
        env_a = EchoMapEnv(host=self.host, port=self.port, robot_id=0)
        env_b = EchoMapEnv(host=self.host, port=self.port, robot_id=1)

        stats = {
            "steps": 0,
            "messages_a": 0,
            "messages_b": 0,
            "rounds_completed": 0,
        }
        commentary_log = []
        last_phase = ""

        try:
            env_a.connect()
            env_b.connect()

            obs_a, info_a = env_a.reset()
            obs_b, info_b = env_b.reset()

            for step in range(max_steps):
                action_a, msg_a = self.agent_a.decide(obs_a, info_a)
                action_b, msg_b = self.agent_b.decide(obs_b, info_b)

                if msg_a:
                    try:
                        env_a.send_message(1, msg_a)
                        stats["messages_a"] += 1
                    except Exception:
                        pass
                if msg_b:
                    try:
                        env_b.send_message(0, msg_b)
                        stats["messages_b"] += 1
                    except Exception:
                        pass

                obs_a, _, done_a, info_a = env_a.step(action_a)
                obs_b, _, done_b, info_b = env_b.step(action_b)
                stats["steps"] += 1

                match_state = info_a.get("match_state") if info_a else None
                current_phase = match_state.get("phase", "") if match_state else ""

                if self._is_round_end(last_phase, current_phase):
                    stats["rounds_completed"] += 1
                    summary = self.commentary.generate_round_summary(match_state)
                    commentary_log.append(summary)
                    if self.verbose:
                        print(f"[Commentary] {summary}")

                last_phase = current_phase

                if current_phase == "match_end" or done_a or done_b:
                    break

            final_state = info_a.get("match_state") if info_a else None
            if final_state:
                match_summary = self.commentary.generate_match_summary(final_state)
                commentary_log.append(match_summary)
                if self.verbose:
                    print(f"[Commentary] {match_summary}")

            return self._build_result(final_state, stats, commentary_log)

        finally:
            env_a.close()
            env_b.close()

    def _is_round_end(self, old_phase, new_phase):
        if not old_phase or not new_phase:
            return False
        return old_phase.startswith("fighting") and new_phase.startswith("round_end")

    def _build_result(self, match_state, stats, commentary_log):
        result = {
            "winner": None,
            "scores": {"a": 0, "b": 0},
            "stats": stats,
            "commentary": commentary_log,
        }

        if match_state:
            total_a = match_state.get("total_score_a", 0)
            total_b = match_state.get("total_score_b", 0)
            result["scores"] = {"a": total_a, "b": total_b}
            if total_a > total_b:
                result["winner"] = "a"
            elif total_b > total_a:
                result["winner"] = "b"

        return result
