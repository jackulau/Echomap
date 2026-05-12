"""LLM-powered boxing agent using Claude API."""

import json
import os

from .agents import BoxingAgent, HeuristicBoxingAgent, clamp_velocities

SYSTEM_PROMPT = """You are a boxing robot in a fighting simulation. You receive observation data about
the current fight state and must output a JSON object with your next action.

Output format (strict JSON, no markdown):
{
  "motor_velocities": [neck, left_arm, right_arm],
  "trash_talk": "optional message to opponent"
}

Joint limits:
- neck: -0.785 to 0.785 (dodge left/right)
- left_arm: -3.14 to 3.14 (negative=retract, positive=extend/punch)
- right_arm: -3.14 to 3.14 (same)

Strategy tips:
- Extend arms fast (high positive values) to punch
- Retract arms (negative values) to guard
- Move neck to dodge incoming punches
- When health is low, prioritize defense
- Trash talk to intimidate your opponent"""


class LLMBoxingAgent(BoxingAgent):
    """Boxing agent powered by Claude API."""

    def __init__(self, model="claude-haiku-4-5-20251001", name="LLM"):
        self.model = model
        self.name = name
        self._client = None
        self._fallback = HeuristicBoxingAgent(name=f"{name}-fallback")
        self._api_available = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            self._api_available = False
            return None
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._api_available = True
            return self._client
        except Exception:
            self._api_available = False
            return None

    def decide(self, observation, info):
        client = self._get_client()
        if client is None:
            return self._fallback.decide(observation, info)

        try:
            prompt = self._build_prompt(observation, info)
            response = client.messages.create(
                model=self.model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_response(response.content[0].text)
        except Exception:
            return self._fallback.decide(observation, info)

    def _build_prompt(self, observation, info):
        parts = []
        match_state = info.get("match_state") if info else None
        if match_state:
            parts.append(f"Phase: {match_state.get('phase', 'unknown')}")
            parts.append(f"Round: {match_state.get('current_round', '?')}")
            parts.append(f"Round time: {match_state.get('round_time', 0):.1f}s / {match_state.get('round_duration', 180):.0f}s")
            parts.append(f"Opponent health: {match_state.get('opponent_health', 100):.1f}")
            parts.append(f"Opponent stamina: {match_state.get('opponent_stamina', 100):.1f}")
            scores = match_state.get("scores", [])
            if scores:
                parts.append(f"Round scores: {scores}")
            parts.append(f"Total score: you={match_state.get('total_score_a', 0)} vs opponent={match_state.get('total_score_b', 0)}")

        if observation and isinstance(observation, dict):
            combat = observation.get("combat")
            if combat and isinstance(combat, dict):
                parts.append(f"Your health: {combat.get('health', 100):.1f}")
                parts.append(f"Your stamina: {combat.get('stamina', 100):.1f}")

        messages = info.get("messages", []) if info else []
        if messages:
            recent = messages[-3:]
            for m in recent:
                if isinstance(m, dict):
                    parts.append(f"Opponent said: \"{m.get('content', '')}\"")
                elif isinstance(m, str):
                    parts.append(f"Opponent said: \"{m}\"")

        return "\n".join(parts) if parts else "No observation data available."

    def _parse_response(self, text):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._fallback.decide(None, None)

        vels = data.get("motor_velocities")
        if not isinstance(vels, list) or len(vels) != 3:
            return self._fallback.decide(None, None)

        try:
            vels = [float(v) for v in vels]
        except (TypeError, ValueError):
            return self._fallback.decide(None, None)

        vels = clamp_velocities(vels)
        message = data.get("trash_talk")
        if message is not None and not isinstance(message, str):
            message = None

        return {"motor_velocities": vels}, message
