"""Boxing agent base class and heuristic agent implementation."""

import math
import random
from abc import ABC, abstractmethod


NECK_LIMIT = math.pi / 4
SHOULDER_LIMIT = math.pi

TRASH_TALK = [
    "Is that all you got?",
    "My servos aren't even warm yet!",
    "You call that a punch?",
    "I've seen better moves from a Roomba!",
    "Recalculating... nope, still winning.",
    "Your algorithm needs an update!",
    "Watch this!",
    "Too slow, tin can!",
    "I can predict your every move!",
    "Error 404: your fighting skills not found.",
]


def clamp_velocities(velocities):
    """Clamp motor velocities to joint limits: [neck, left_shoulder, right_shoulder]."""
    limits = [NECK_LIMIT, SHOULDER_LIMIT, SHOULDER_LIMIT]
    return [max(-lim, min(lim, v)) for v, lim in zip(velocities, limits)]


class BoxingAgent(ABC):
    """Abstract base class for boxing agents."""

    @abstractmethod
    def decide(self, observation, info):
        """Decide action and optional message from current observation.

        Args:
            observation: State dict from EchoMapEnv.step() or .reset().
            info: Info dict containing messages, match_state, etc.

        Returns:
            Tuple of (action_dict, optional_message) where:
                action_dict has "motor_velocities" (list of 3 floats)
                optional_message is a string or None
        """


class HeuristicBoxingAgent(BoxingAgent):
    """Rule-based boxing agent that fights without any API keys."""

    def __init__(self, name="Heuristic", trash_talk_chance=0.05):
        self.name = name
        self.trash_talk_chance = trash_talk_chance
        self._step = 0
        self._rng = random.Random()

    def decide(self, observation, info):
        self._step += 1
        match_state = info.get("match_state") if info else None

        phase = ""
        if match_state:
            phase = match_state.get("phase", "")

        if not phase.startswith("fighting"):
            return {"motor_velocities": [0.0, 0.0, 0.0]}, None

        own_health = self._get_own_health(observation)
        opp_health = self._get_opponent_health(match_state)

        if own_health < 30.0:
            velocities = self._defensive_stance()
        elif own_health > opp_health:
            velocities = self._aggressive_attack()
        else:
            velocities = self._balanced_fight()

        velocities = clamp_velocities(velocities)
        message = self._maybe_trash_talk()

        return {"motor_velocities": velocities}, message

    def _get_own_health(self, observation):
        if not observation:
            return 100.0
        combat = observation.get("combat") if isinstance(observation, dict) else None
        if combat and isinstance(combat, dict):
            return combat.get("health", 100.0)
        return 100.0

    def _get_opponent_health(self, match_state):
        if not match_state:
            return 100.0
        return match_state.get("opponent_health", 100.0)

    def _aggressive_attack(self):
        neck = self._rng.uniform(-0.3, 0.3)
        left = self._rng.uniform(1.5, SHOULDER_LIMIT)
        right = self._rng.uniform(1.5, SHOULDER_LIMIT)
        if self._step % 3 == 0:
            left = -left
        if self._step % 4 == 0:
            right = -right
        return [neck, left, right]

    def _defensive_stance(self):
        neck = self._rng.uniform(-NECK_LIMIT, NECK_LIMIT)
        left = self._rng.uniform(-0.5, 0.5)
        right = self._rng.uniform(-0.5, 0.5)
        return [neck, left, right]

    def _balanced_fight(self):
        neck = self._rng.uniform(-0.2, 0.2)
        left = self._rng.uniform(-SHOULDER_LIMIT * 0.7, SHOULDER_LIMIT * 0.7)
        right = self._rng.uniform(-SHOULDER_LIMIT * 0.7, SHOULDER_LIMIT * 0.7)
        return [neck, left, right]

    def _maybe_trash_talk(self):
        if self._rng.random() < self.trash_talk_chance:
            return self._rng.choice(TRASH_TALK)
        return None
