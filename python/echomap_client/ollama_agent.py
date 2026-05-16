"""LLM-powered boxing agent using local Ollama models."""

import json
import math

from .agents import (
    BASE_SPEED_LIMIT,
    BoxingAgent,
    HeuristicBoxingAgent,
    clamp_base_velocity,
    clamp_velocities,
    relative_opponent,
)

SYSTEM_PROMPT = """You are a boxing humanoid in a 6m square ring. You receive observations and must respond with ONLY a JSON object (no markdown, no explanation).

Output format:
{"motor_velocities": [neck, left_shoulder, right_shoulder], "base_velocity": [vx, vz], "trash_talk": "your taunt"}

Joint ranges: neck [-0.785, 0.785] rad/s, shoulders [-3.14, 3.14] rad/s.
Base velocity: planar movement [vx, vz] in m/s, each clamped to [-2.0, 2.0].
Positive shoulder values swing arm forward (punch). Negative retract (guard).

Tactical guide:
- distance > 0.7m: walk toward opponent — set vx,vz pointing toward them.
- 0.4m < distance <= 0.7m: PUNCH HARD. Alternate left/right shoulder near +/-3.0. Move feet only to track.
- distance < 0.4m: back off — set vx,vz pointing away — and keep punching.
- Low health (<30): guard (negative shoulder values), sidestep, retreat.

Be aggressive and decisive — high shoulder magnitudes land hits."""


class OllamaBoxingAgent(BoxingAgent):
    """Boxing agent powered by a local Ollama model with locomotion + spatial perception.

    Calls the LLM every `decide_interval` steps. Between calls, a HeuristicBoxingAgent
    drives the body so the fighter keeps moving + swinging while inference is pending.
    """

    def __init__(
        self,
        model="llama3.2",
        name="Ollama",
        base_url="http://localhost:11434",
        decide_interval=10,
    ):
        self.model = model
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.decide_interval = decide_interval
        self._fallback = HeuristicBoxingAgent(name=f"{name}-fallback", trash_talk_chance=0.0)
        self._available = None
        self._step = 0
        self._last_action = None
        self._last_message = None

    def _check_available(self):
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def decide(self, observation, info):
        self._step += 1

        if not self._check_available():
            return self._fallback.decide(observation, info)

        # LLM call cadence — between calls hand fully over to heuristic so the fighter
        # keeps swinging + navigating while inference is pending. LLM-driven motor
        # velocities only apply on the LLM-tick frame itself; if the LLM returns
        # weak values we do not let them freeze the body for the next 9 frames.
        if self._step % self.decide_interval != 1:
            fb_action, _ = self._fallback.decide(observation, info)
            return fb_action, None

        try:
            prompt = self._build_prompt(observation, info)
            text = self._generate(prompt)
            action, message = self._parse_response(text, observation, info)
            self._last_action = action
            return action, message
        except Exception:
            return self._fallback.decide(observation, info)

    def _generate(self, prompt):
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 96},
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    def _build_prompt(self, observation, info):
        parts = []
        match_state = info.get("match_state") if info else None
        phase = match_state.get("phase", "unknown") if match_state else "unknown"
        parts.append(f"phase={phase}")
        if match_state:
            parts.append(f"round={match_state.get('current_round', '?')}")
            parts.append(f"time={match_state.get('round_time', 0):.1f}s")
            parts.append(f"score you={match_state.get('total_score_a', 0) if match_state.get('your_robot', 0) == 0 else match_state.get('total_score_b', 0)} opp={match_state.get('total_score_b', 0) if match_state.get('your_robot', 0) == 0 else match_state.get('total_score_a', 0)}")
            parts.append(f"opp_hp={match_state.get('opponent_health', 100):.0f}")

        if observation and isinstance(observation, dict):
            combat = observation.get("combat")
            if combat and isinstance(combat, dict):
                parts.append(f"your_hp={combat.get('health', 100):.0f} stamina={combat.get('stamina', 100):.0f}")
            jp = observation.get("joint_positions")
            if isinstance(jp, list) and len(jp) >= 3:
                parts.append(f"joints neck={jp[0]:.2f} left_shoulder={jp[1]:.2f} right_shoulder={jp[2]:.2f}")

        rel = relative_opponent(observation, info)
        if rel is not None:
            dx, dz, dist = rel
            parts.append(f"Opponent at dx={dx:.2f} dz={dz:.2f} distance={dist:.2f}m")
            if dist > 0.7:
                parts.append("APPROACH: opponent out of range, walk toward them.")
            elif dist < 0.4:
                parts.append("TOO CLOSE: retreat slightly.")
            else:
                parts.append("IN RANGE: PUNCH HARD with high shoulder velocities.")

        if info:
            hit_events = info.get("hit_events", []) or []
            for h in hit_events[-3:]:
                if not isinstance(h, dict):
                    continue
                attacker = h.get("attacker_robot_id", "?")
                victim = h.get("victim_robot_id", "?")
                zone = h.get("body_zone", "?")
                dmg = h.get("damage", 0.0)
                parts.append(f"recent hit: robot{attacker}->robot{victim} {zone} dmg={dmg:.1f}")

            messages = info.get("messages", [])
            for m in messages[-2:]:
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                if content:
                    parts.append(f'opponent said: "{content}"')

        return "\n".join(parts) if parts else "Fight started. Make your move."

    def _parse_response(self, text, observation=None, info=None):
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._fallback.decide(observation, info)

        vels = data.get("motor_velocities")
        if not isinstance(vels, list) or len(vels) != 3:
            return self._fallback.decide(observation, info)
        try:
            vels = [float(v) for v in vels]
        except (TypeError, ValueError):
            return self._fallback.decide(observation, info)
        vels = clamp_velocities(vels)

        bv = data.get("base_velocity")
        if isinstance(bv, list) and len(bv) == 2:
            try:
                bv = clamp_base_velocity([float(bv[0]), float(bv[1])])
            except (TypeError, ValueError):
                bv = [0.0, 0.0]
        else:
            bv = [0.0, 0.0]

        message = data.get("trash_talk")
        if message is not None and not isinstance(message, str):
            message = None

        return {"motor_velocities": vels, "base_velocity": bv}, message
