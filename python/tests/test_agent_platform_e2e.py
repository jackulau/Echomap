"""End-to-end smoke for the agent platform (goal 003 D5).

Three paths exercised, end-to-end:
  1. Bind an agent to a boxing robot via `connect_agent("robot/0")`.
  2. Bind the same shape to a second robot (`robot/1`).
  3. Drive a `MockArm` hardware backend with an identical agent class.

Skipped when the Rust release server binary isn't available.
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


class TinyAgent:
    """Same agent class used against sim and hardware."""

    def decide(self, observation, info):
        joints = []
        if isinstance(observation, dict):
            joints = observation.get("joint_positions") or []
        vels = [0.1] * max(1, len(joints))
        return {"motor_velocities": vels, "gripper_commands": []}, None


class TestAgentPlatformE2E(unittest.TestCase):
    """Smoke test for the three documented binding paths."""

    PORT = 9016

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SERVER_BIN):
            raise unittest.SkipTest(f"server binary missing at {SERVER_BIN}")
        env = {**os.environ, "WS_PORT": str(cls.PORT), "ROUND_DURATION": "30", "NUM_ROUNDS": "1"}
        cls._proc = subprocess.Popen(
            [SERVER_BIN], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not _wait_for_port(cls.PORT):
            cls._proc.kill()
            raise RuntimeError(f"server failed to listen on {cls.PORT}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_proc"):
            try:
                cls._proc.terminate()
                cls._proc.wait(timeout=5)
            except Exception:
                cls._proc.kill()

    def test_path_a_bind_to_boxing_robot(self):
        from echomap_client.env import connect_agent
        agent = TinyAgent()
        env = connect_agent(target_id="robot/0", agent=agent, port=self.PORT)
        try:
            state, info = env.reset()
            self.assertIsNotNone(state)
            action, _msg = agent.decide(state, info)
            obs, _r, _d, info2 = env.step(action)
            self.assertIsNotNone(obs)
            self.assertIn("step_count", info2)
        finally:
            env.close()

    def test_path_b_bind_to_second_robot_same_shape(self):
        from echomap_client.env import connect_agent
        env = connect_agent(target_id="robot/1", agent=TinyAgent(), port=self.PORT)
        try:
            self.assertIsNotNone(env.observation_space)
            self.assertIn("step", env.capabilities)
        finally:
            env.close()

    def test_path_c_same_agent_against_hardware_bridge(self):
        from echomap_client.hardware import MockArm, RobotArmBridge

        agent = TinyAgent()
        with RobotArmBridge(backend=MockArm(num_joints=6)) as bridge:
            obs, info = bridge.reset()
            for _ in range(10):
                action, _ = agent.decide(obs, info)
                obs, _r, _d, info = bridge.step(action)
            # joints must have moved (action was non-zero)
            self.assertGreater(abs(obs["joint_positions"][0]), 0.0)

    def test_plugin_loader_runs_clean(self):
        from echomap_client.plugins import load_all
        reg = load_all()
        # whether or not the example plugin is installed, load_all should not
        # raise — it accumulates errors in reg.errors instead.
        self.assertIsNotNone(reg)


if __name__ == "__main__":
    unittest.main()
