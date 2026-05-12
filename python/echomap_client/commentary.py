"""Match commentary generator for boxing matches."""

import os


class MatchCommentary:
    """Generates fight narration from match state."""

    def __init__(self, use_llm=False, model="claude-haiku-4-5-20251001"):
        self.use_llm = use_llm
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.use_llm:
            return None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client
        except Exception:
            return None

    def generate_round_summary(self, match_state, events=None):
        """Generate a summary for a completed round.

        Args:
            match_state: BoxingMatchState dict from the simulation.
            events: Optional list of notable events during the round.

        Returns:
            String narration of the round.
        """
        if not match_state:
            return "Round complete."

        client = self._get_client() if self.use_llm else None
        if client:
            return self._llm_round_summary(client, match_state, events)
        return self._template_round_summary(match_state, events)

    def generate_match_summary(self, match_state):
        """Generate a final match result narrative.

        Args:
            match_state: BoxingMatchState dict from the simulation.

        Returns:
            String narration of the match result.
        """
        if not match_state:
            return "Match complete."

        client = self._get_client() if self.use_llm else None
        if client:
            return self._llm_match_summary(client, match_state)
        return self._template_match_summary(match_state)

    def _template_round_summary(self, match_state, events=None):
        round_num = match_state.get("current_round", "?")
        scores = match_state.get("scores", [])

        parts = [f"Round {round_num} complete!"]

        if scores and len(scores) >= round_num:
            idx = round_num - 1
            if idx < len(scores):
                score = scores[idx]
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    parts.append(f"Score: {score[0]}-{score[1]}")

        opp_health = match_state.get("opponent_health", 100)
        parts.append(f"Opponent health: {opp_health:.0f}")

        if events:
            hit_count = sum(1 for e in events if isinstance(e, dict) and e.get("type") == "hit")
            if hit_count > 0:
                parts.append(f"Hits landed: {hit_count}")

        return " ".join(parts)

    def _template_match_summary(self, match_state):
        total_a = match_state.get("total_score_a", 0)
        total_b = match_state.get("total_score_b", 0)

        if total_a > total_b:
            result = "Robot A wins!"
        elif total_b > total_a:
            result = "Robot B wins!"
        else:
            result = "It's a draw!"

        rounds = match_state.get("current_round", "?")
        return f"Match over after {rounds} rounds. Final score: {total_a}-{total_b}. {result}"

    def _llm_round_summary(self, client, match_state, events):
        try:
            prompt = f"Boxing round {match_state.get('current_round', '?')} just ended. "
            prompt += f"Scores: {match_state.get('scores', [])}. "
            prompt += f"Opponent health: {match_state.get('opponent_health', 100):.0f}. "
            prompt += "Write one exciting sentence of boxing commentary."

            response = client.messages.create(
                model=self.model,
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return self._template_round_summary(match_state, events)

    def _llm_match_summary(self, client, match_state):
        try:
            total_a = match_state.get("total_score_a", 0)
            total_b = match_state.get("total_score_b", 0)
            prompt = f"Boxing match over! Score: {total_a}-{total_b}. "
            prompt += f"Rounds: {match_state.get('current_round', '?')}. "
            prompt += "Write 2 sentences of dramatic boxing match result commentary."

            response = client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return self._template_match_summary(match_state)
