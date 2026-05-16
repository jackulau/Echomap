"""Tests for generic agent-to-target binding (goal 003 D2).

Exercises the `connect_agent(target_id, agent)` ergonomics and the
`EchoMapEnv(target_id=...)` BindTarget path against a live headless
server. Skipped if the server binary isn't built.

Run: python3 -m pytest python/tests/test_agent_bind.py -v
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


@unittest.skipUnless(
    os.path.exists(SERVER_BIN),
    f"server binary not built at {SERVER_BIN} — run `cargo build --release --bin echomap_server`",
)
class TestAgentBind(unittest.TestCase):
    """Live binding tests against a short-lived headless boxing server."""

    PORT = 9015  # avoid colliding with 9002 GUI default

    @classmethod
    def setUpClass(cls):
        env = {**os.environ, "WS_PORT": str(cls.PORT), "ROUND_DURATION": "30", "NUM_ROUNDS": "1"}
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

    def test_connect_agent_exists(self):
        from echomap_client.env import connect_agent
        self.assertTrue(callable(connect_agent))

    def test_connect_agent_binds_to_robot_0(self):
        from echomap_client.env import connect_agent
        env = connect_agent(target_id="robot/0", host="localhost", port=self.PORT)
        try:
            self.assertIsNotNone(env.observation_space)
            self.assertIsNotNone(env.action_space)
            self.assertIn("step", env.capabilities)
            self.assertIn("observe", env.capabilities)
        finally:
            env.close()

    def test_connect_agent_binds_to_robot_1(self):
        from echomap_client.env import connect_agent
        env = connect_agent(target_id="robot/1", host="localhost", port=self.PORT)
        try:
            self.assertIsNotNone(env.observation_space)
        finally:
            env.close()

    def test_bare_numeric_target_id(self):
        """`"0"` should resolve the same as `"robot/0"`."""
        from echomap_client.env import connect_agent
        env = connect_agent(target_id="0", host="localhost", port=self.PORT)
        try:
            self.assertIsNotNone(env.action_space)
        finally:
            env.close()

    def test_bind_invalid_target_raises(self):
        from echomap_client.env import connect_agent
        with self.assertRaises(ConnectionError):
            connect_agent(target_id="robot/banana", host="localhost", port=self.PORT)

    def test_bind_step_observation_roundtrip(self):
        """End-to-end: bind, reset, step, get observation back."""
        from echomap_client.env import connect_agent
        env = connect_agent(target_id="robot/0", host="localhost", port=self.PORT)
        try:
            state, _info = env.reset()
            self.assertIsNotNone(state)
            obs, _reward, _done, info = env.step({
                "motor_velocities": [0.1, 0.0, 0.0],
                "gripper_commands": [],
            })
            self.assertIsNotNone(obs)
            self.assertIn("step_count", info)
        finally:
            env.close()


class TestBindTargetUnit(unittest.TestCase):
    """Pure-Python unit tests — no server required."""

    def test_env_accepts_target_id_param(self):
        from echomap_client import EchoMapEnv
        env = EchoMapEnv(target_id="robot/0")
        self.assertEqual(env._target_id, "robot/0")

    def test_env_capabilities_default_empty(self):
        from echomap_client import EchoMapEnv
        env = EchoMapEnv()
        self.assertEqual(env.capabilities, [])


if __name__ == "__main__":
    unittest.main()
