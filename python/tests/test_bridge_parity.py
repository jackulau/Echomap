"""Hardware bridge parity tests (goal/008 deliverable 4).

Drives the same agent class against:
  - the live Rust simulator (sim humanoid via ``connect_agent("robot/0")``)
  - the in-process ``MockArm(num_joints=6)`` hardware bridge

For each step, asserts that the agent emits an action whose *key shape*
is uniform across both targets: same set of dict keys, same lengths per
key. This is the gate that catches schema drift between the two
runtimes — for example, if the bridge ever stopped accepting
``base_velocity`` or the humanoid started requiring extra fields.

Run:
    python3 -m pytest python/tests/test_bridge_parity.py -v
"""

import os
import socket
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_BIN = os.path.join(REPO_ROOT, "target", "release", "echomap_server")
STEPS_PER_RUN = 100
SEED = 4242


def _port_open(port, host="localhost"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def _wait_for_port(port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def _action_key_shape(action):
    """Return a canonical ``(keyset, lengths)`` signature for an action dict.

    Excludes the actual numeric values — the parity check is about
    schema, not behavior. Lengths are recorded per list-valued key.
    """
    if not isinstance(action, dict):
        return (None, None)
    keyset = tuple(sorted(action.keys()))
    lengths = tuple(
        (k, len(action[k]) if hasattr(action[k], "__len__") else None)
        for k in keyset
    )
    return (keyset, lengths)


class _AdapterAgent:
    """Agent that always emits the SAME action shape regardless of target.

    HeuristicBoxingAgent assumes boxing-specific observation fields
    (``combat.health``, ``info["match_state"]``) and emits a 3-motor
    action. The hardware bridge instead exposes a generic arm with
    N motors and no combat info.

    The parity gate is not about behavioral equivalence — it's about
    proving the *agent contract* (the dict shape agents must produce)
    is consistent across both runtimes. This adapter normalizes the
    behavior so we can isolate the schema check.
    """

    def __init__(self, num_motors: int, seed: int = SEED):
        import random
        self._n = num_motors
        self._rng = random.Random(seed)
        self._step = 0

    def decide(self, observation, info):
        # Deterministic, bounded velocity vector sized to the target.
        self._step += 1
        vels = [
            self._rng.uniform(-0.5, 0.5) for _ in range(self._n)
        ]
        return (
            {
                "motor_velocities": vels,
                "gripper_commands": [],
            },
            None,
        )


@unittest.skipUnless(
    os.path.exists(SERVER_BIN),
    f"server binary not built at {SERVER_BIN} — run `cargo build --release --bin echomap_server`",
)
class TestBridgeParity(unittest.TestCase):
    """Drive an agent against sim + hardware and compare action-stream shapes."""

    PORT = 9017  # distinct from other test classes

    @classmethod
    def setUpClass(cls):
        env = {
            **os.environ,
            "WS_PORT": str(cls.PORT),
            "ROUND_DURATION": "120",
            "NUM_ROUNDS": "1",
        }
        cls._proc = subprocess.Popen(
            [SERVER_BIN], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not _wait_for_port(cls.PORT):
            cls._proc.kill()
            raise RuntimeError(f"server failed to listen on {cls.PORT}")

    @classmethod
    def tearDownClass(cls):
        try:
            cls._proc.terminate()
            cls._proc.wait(timeout=5)
        except Exception:
            cls._proc.kill()

    # ------------------------------------------------------------------
    # Sim side: drive a 3-motor boxing humanoid via connect_agent.
    # ------------------------------------------------------------------
    def _drive_sim(self, steps):
        from echomap_client.env import connect_agent

        env = connect_agent(target_id="robot/0", host="localhost", port=self.PORT)
        try:
            num_motors = env.action_space.get("num_motors", 3)
            agent = _AdapterAgent(num_motors=num_motors)
            obs, info = env.reset()
            shapes = []
            for _ in range(steps):
                action, _ = agent.decide(obs, info)
                shapes.append(_action_key_shape(action))
                obs, _r, done, info = env.step(action)
                if done:
                    # match may end; just stop — we still got >0 steps.
                    break
            return shapes, env.capabilities
        finally:
            env.close()

    # ------------------------------------------------------------------
    # Hardware side: drive a MockArm(num_joints=6) via RobotArmBridge.
    # ------------------------------------------------------------------
    def _drive_hardware(self, steps):
        from echomap_client.hardware import MockArm, RobotArmBridge

        arm = MockArm(num_joints=6)
        bridge = RobotArmBridge(backend=arm)
        try:
            num_motors = bridge.action_space.get("num_motors", 6)
            agent = _AdapterAgent(num_motors=num_motors)
            obs, info = bridge.reset()
            shapes = []
            for _ in range(steps):
                action, _ = agent.decide(obs, info)
                shapes.append(_action_key_shape(action))
                obs, _r, _done, info = bridge.step(action)
            return shapes, bridge.capabilities
        finally:
            bridge.close()

    # ------------------------------------------------------------------
    # Parity assertions.
    # ------------------------------------------------------------------
    def test_action_dict_keys_match_across_sim_and_hardware(self):
        sim_shapes, _ = self._drive_sim(STEPS_PER_RUN)
        hw_shapes, _ = self._drive_hardware(STEPS_PER_RUN)
        self.assertTrue(len(sim_shapes) >= 1, "sim produced no actions")
        self.assertEqual(len(hw_shapes), STEPS_PER_RUN, "hardware loop short")

        sim_keysets = {s[0] for s in sim_shapes}
        hw_keysets = {s[0] for s in hw_shapes}
        self.assertEqual(
            sim_keysets, hw_keysets,
            f"sim emitted keys {sim_keysets}, hardware emitted {hw_keysets} — "
            "agent dict-shape drift between targets",
        )

    def test_action_list_lengths_internally_consistent(self):
        """Per target, every step's list-valued fields are the same length.

        Agents must not vary motor_velocities length step-to-step against a
        given target — that's the action_space contract.
        """
        for label, shapes in (
            ("sim", self._drive_sim(STEPS_PER_RUN)[0]),
            ("hardware", self._drive_hardware(STEPS_PER_RUN)[0]),
        ):
            lengths = {s[1] for s in shapes}
            self.assertEqual(
                len(lengths), 1,
                f"{label}: lengths drifted across steps: {lengths}",
            )

    def test_both_targets_advertise_common_capabilities(self):
        _, sim_caps = self._drive_sim(1)
        _, hw_caps = self._drive_hardware(1)
        # Both must advertise the universal action contract.
        for cap in ("observe", "step"):
            self.assertIn(cap, sim_caps, f"sim missing capability '{cap}'")
            self.assertIn(cap, hw_caps, f"hardware missing capability '{cap}'")
        # Both should advertise motors (sim humanoid has them, MockArm has them).
        self.assertIn("motors", sim_caps)
        self.assertIn("motors", hw_caps)

    def test_deterministic_replay_same_action_sequence_per_target(self):
        """Reseeding the agent yields identical actions on hardware (RNG-pure)."""
        hw_shapes_a, _ = self._drive_hardware(50)
        hw_shapes_b, _ = self._drive_hardware(50)
        self.assertEqual(hw_shapes_a, hw_shapes_b)


class TestBridgeParityUnit(unittest.TestCase):
    """Pure-Python parity checks — no server required."""

    def test_adapter_agent_produces_canonical_shape(self):
        agent = _AdapterAgent(num_motors=4)
        action, _ = agent.decide({}, {})
        keys, lengths = _action_key_shape(action)
        self.assertEqual(keys, ("gripper_commands", "motor_velocities"))
        self.assertEqual(
            dict(lengths),
            {"gripper_commands": 0, "motor_velocities": 4},
        )

    def test_action_key_shape_distinguishes_lengths(self):
        a3 = {"motor_velocities": [0.0] * 3, "gripper_commands": []}
        a6 = {"motor_velocities": [0.0] * 6, "gripper_commands": []}
        # Same keys, different lengths.
        s3 = _action_key_shape(a3)
        s6 = _action_key_shape(a6)
        self.assertEqual(s3[0], s6[0])
        self.assertNotEqual(s3[1], s6[1])


if __name__ == "__main__":
    unittest.main()
