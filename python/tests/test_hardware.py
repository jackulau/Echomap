"""Tests for the hardware bridge (goal 003 D3).

Pure-Python — no Rust server needed. Verifies that MockArm + SerialArm
share the same surface as EchoMapEnv (reset / step / observe).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMockArm(unittest.TestCase):
    def test_default_joint_count(self):
        from echomap_client.hardware import MockArm
        arm = MockArm()
        self.assertEqual(arm.num_joints, 6)

    def test_invalid_joint_count_raises(self):
        from echomap_client.hardware import MockArm
        with self.assertRaises(ValueError):
            MockArm(num_joints=0)

    def test_reset_returns_zero_state(self):
        from echomap_client.hardware import MockArm
        arm = MockArm(num_joints=4)
        state = arm.reset()
        self.assertEqual(state["joint_positions"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(state["joint_velocities"], [0.0, 0.0, 0.0, 0.0])

    def test_apply_action_moves_joints(self):
        from echomap_client.hardware import MockArm
        arm = MockArm(num_joints=3, dt=0.1, damping=1.0)
        arm.reset()
        arm.apply_action({"motor_velocities": [1.0, -0.5, 0.0]})
        state = arm.read_state()
        # one tick at dt=0.1 → first joint should advance by ~0.1 rad
        self.assertAlmostEqual(state["joint_positions"][0], 0.1, places=5)
        self.assertAlmostEqual(state["joint_positions"][1], -0.05, places=5)

    def test_joint_limit_clamps(self):
        from echomap_client.hardware import MockArm
        arm = MockArm(num_joints=2, dt=10.0, joint_limit=math.pi, damping=1.0)
        arm.reset()
        # huge step: position must clamp to ±pi
        arm.apply_action({"motor_velocities": [10.0, -10.0]})
        state = arm.read_state()
        self.assertAlmostEqual(state["joint_positions"][0], math.pi, places=4)
        self.assertAlmostEqual(state["joint_positions"][1], -math.pi, places=4)

    def test_gripper_state(self):
        from echomap_client.hardware import MockArm
        arm = MockArm(num_joints=2, has_gripper=True)
        arm.reset()
        arm.apply_action({"motor_velocities": [0.0, 0.0], "gripper_commands": [False]})
        state = arm.read_state()
        self.assertEqual(state["gripper_open"], False)


class TestSerialArmStub(unittest.TestCase):
    def test_open_raises_not_implemented(self):
        from echomap_client.hardware import SerialArm
        arm = SerialArm(port="/dev/null", num_joints=6)
        with self.assertRaises(NotImplementedError):
            arm.open()

    def test_apply_action_frames_packets(self):
        from echomap_client.hardware import SerialArm
        arm = SerialArm(port="/dev/null", num_joints=2)
        # should not raise — stub frames internally and updates local state
        arm.apply_action({"motor_velocities": [0.5, -0.5]})
        state = arm.read_state()
        self.assertEqual(state["joint_velocities"], [0.5, -0.5])

    def test_frame_starts_with_sync_bytes(self):
        from echomap_client.hardware.bridge import SerialArm
        frame = SerialArm._frame(joint_id=3, cmd=0x01, payload=b"\x10\x00")
        self.assertEqual(frame[:2], b"\xff\xff")
        self.assertEqual(frame[2], 3)
        self.assertEqual(frame[3], 0x01)


class TestBridgeSchema(unittest.TestCase):
    """Bridge must expose the same shape as EchoMapEnv."""

    def test_capabilities_advertised(self):
        from echomap_client.hardware import MockArm, RobotArmBridge
        bridge = RobotArmBridge(backend=MockArm(num_joints=4))
        self.assertIn("motors", bridge.capabilities)
        self.assertIn("observe", bridge.capabilities)
        self.assertIn("step", bridge.capabilities)
        self.assertNotIn("grippers", bridge.capabilities)

    def test_gripper_capability(self):
        from echomap_client.hardware import MockArm, RobotArmBridge
        bridge = RobotArmBridge(backend=MockArm(num_joints=4, has_gripper=True))
        self.assertIn("grippers", bridge.capabilities)

    def test_reset_step_roundtrip(self):
        from echomap_client.hardware import MockArm, RobotArmBridge
        bridge = RobotArmBridge(backend=MockArm(num_joints=3))
        obs, info = bridge.reset()
        self.assertEqual(len(obs["joint_positions"]), 3)
        self.assertEqual(info["step_count"], 0)
        obs2, reward, done, info2 = bridge.step({"motor_velocities": [0.1, 0.0, -0.1]})
        self.assertEqual(info2["step_count"], 1)
        self.assertEqual(done, False)
        self.assertEqual(reward, 0.0)

    def test_observation_space_shape(self):
        from echomap_client.hardware import MockArm, RobotArmBridge
        bridge = RobotArmBridge(backend=MockArm(num_joints=6))
        self.assertEqual(bridge.observation_space["num_joints"], 6)
        self.assertEqual(bridge.action_space["num_motors"], 6)

    def test_context_manager(self):
        from echomap_client.hardware import MockArm, RobotArmBridge
        with RobotArmBridge(backend=MockArm()) as bridge:
            obs, _ = bridge.reset()
            self.assertIsNotNone(obs)

    def test_same_agent_drives_bridge(self):
        """An agent built for sim should work against the bridge unchanged."""
        from echomap_client.hardware import MockArm, RobotArmBridge

        class TinyAgent:
            def decide(self, observation, info):
                vels = [0.05] * len(observation["joint_positions"])
                return {"motor_velocities": vels}, None

        bridge = RobotArmBridge(backend=MockArm(num_joints=4))
        obs, info = bridge.reset()
        agent = TinyAgent()
        for _ in range(5):
            action, _ = agent.decide(obs, info)
            obs, _r, _d, info = bridge.step(action)
        # joints should have moved
        self.assertGreater(obs["joint_positions"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
