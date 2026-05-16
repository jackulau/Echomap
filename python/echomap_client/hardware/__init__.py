"""Hardware bridge — drive a real robot arm with the same agent code as the sim.

The bridge exposes the same observation/action schema as `EchoMapEnv`:

    observation = {
        "joint_positions": [float, ...],   # one per joint (rad)
        "joint_velocities": [float, ...],  # one per joint (rad/s)
        "sensor_readings": {...},          # optional, device-specific
    }
    action = {
        "motor_velocities": [float, ...],  # one per joint (rad/s)
        "gripper_commands": [bool, ...],   # optional
    }

Two backends ship:

- ``MockArm`` — pure Python, always available. Simulates a damped 6-DOF
  arm with bounded integration. Use in CI and for local agent dev.
- ``SerialArm`` — stub backend that documents the wire format for a real
  Dynamixel-style arm. Sends framed packets over a serial port. Throws
  ``NotImplementedError`` on actual write until a vendor driver is plugged
  in via the plugin system (goal 003 D4).

Usage:

    from echomap_client.hardware import MockArm, RobotArmBridge

    arm = MockArm(num_joints=6)
    bridge = RobotArmBridge(backend=arm)
    obs = bridge.reset()
    obs, _ = bridge.step({"motor_velocities": [0.1, 0.0, -0.2, 0.0, 0.1, 0.0]})
"""

from .bridge import MockArm, RobotArmBridge, SerialArm

__all__ = ["MockArm", "RobotArmBridge", "SerialArm"]
