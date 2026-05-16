"""RobotArmBridge — uniform interface over real + mock arms.

The bridge presents the same observation/action surface as ``EchoMapEnv`` so
the *same* agent class can drive both the simulator and a physical arm.
Backends implement the ``ArmBackend`` protocol; ``MockArm`` ships for tests
and ``SerialArm`` is a documented stub for a vendor-specific serial driver.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional


class ArmBackend:
    """Protocol every arm backend implements.

    Subclasses override these methods. The bridge calls them in lockstep
    with the agent decide loop.
    """

    num_joints: int = 0
    has_gripper: bool = False

    def reset(self) -> dict:
        raise NotImplementedError

    def read_state(self) -> dict:
        """Return current observation (joint positions, velocities, sensors)."""
        raise NotImplementedError

    def apply_action(self, action: dict) -> None:
        """Command motor velocities (and optionally gripper) on the device."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover — backends override as needed
        pass


class MockArm(ArmBackend):
    """In-process arm — first-order joint dynamics with damping and limits.

    Joint positions integrate from velocity commands; an exponential damping
    term keeps the arm settling toward rest when commands stop. Joint limits
    clamp positions so the arm stays in a realistic envelope.
    """

    def __init__(
        self,
        num_joints: int = 6,
        dt: float = 0.05,
        damping: float = 0.9,
        joint_limit: float = math.pi,
        has_gripper: bool = False,
    ):
        if num_joints < 1:
            raise ValueError("num_joints must be >= 1")
        self.num_joints = num_joints
        self.has_gripper = has_gripper
        self._dt = dt
        self._damping = damping
        self._joint_limit = joint_limit
        self._positions = [0.0] * num_joints
        self._velocities = [0.0] * num_joints
        self._gripper_open = True
        self._t0 = time.monotonic()

    def reset(self) -> dict:
        self._positions = [0.0] * self.num_joints
        self._velocities = [0.0] * self.num_joints
        self._gripper_open = True
        self._t0 = time.monotonic()
        return self.read_state()

    def read_state(self) -> dict:
        state = {
            "joint_positions": list(self._positions),
            "joint_velocities": list(self._velocities),
            "sensor_readings": {
                "wall_time_s": time.monotonic() - self._t0,
            },
        }
        if self.has_gripper:
            state["gripper_open"] = self._gripper_open
        return state

    def apply_action(self, action: dict) -> None:
        vels = action.get("motor_velocities") or []
        if not isinstance(vels, (list, tuple)):
            raise TypeError("motor_velocities must be a list")
        n = min(len(vels), self.num_joints)
        for i in range(n):
            self._velocities[i] = float(vels[i])
        # Integrate one tick. Apply damping so commands decay if not refreshed.
        for i in range(self.num_joints):
            self._positions[i] += self._velocities[i] * self._dt
            # Clamp to joint limits.
            if self._positions[i] > self._joint_limit:
                self._positions[i] = self._joint_limit
                self._velocities[i] = 0.0
            elif self._positions[i] < -self._joint_limit:
                self._positions[i] = -self._joint_limit
                self._velocities[i] = 0.0
            self._velocities[i] *= self._damping

        if self.has_gripper:
            cmds = action.get("gripper_commands") or []
            if cmds:
                self._gripper_open = bool(cmds[0])


class SerialArm(ArmBackend):
    """Stub backend for a real serial-controlled arm (Dynamixel-style).

    Wire format (one frame per joint command, little-endian):

        [0xFF][0xFF][joint_id u8][cmd u8][payload_len u8][payload ...][crc16 u16]

    Commands implemented in this stub:

        0x01 SET_VELOCITY    payload: int16 deci-rad/s (velocity * 100)
        0x02 SET_POSITION    payload: int16 deci-rad
        0x03 READ_STATE      payload: empty -> response carries pos+vel

    The stub validates inputs and computes framing but does NOT open a
    serial port. Plug a vendor driver in via the echomap.plugins.hardware
    entry-point group (see goal 003 D4) to enable real I/O.
    """

    SYNC = bytes([0xFF, 0xFF])
    CMD_SET_VELOCITY = 0x01
    CMD_SET_POSITION = 0x02
    CMD_READ_STATE = 0x03

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        num_joints: int = 6,
        has_gripper: bool = False,
    ):
        if num_joints < 1:
            raise ValueError("num_joints must be >= 1")
        self.port = port
        self.baud = baud
        self.num_joints = num_joints
        self.has_gripper = has_gripper
        self._opened = False
        self._positions = [0.0] * num_joints
        self._velocities = [0.0] * num_joints

    def open(self) -> None:
        """Open the serial port. NotImplemented in the stub."""
        raise NotImplementedError(
            "SerialArm is a stub. Install a vendor driver plugin (e.g. "
            "echomap-driver-dynamixel) and register it under the "
            "`echomap.plugins.hardware` entry-point group."
        )

    def reset(self) -> dict:
        # Reset is a no-op for a stub; on a real arm this would home the joints.
        self._positions = [0.0] * self.num_joints
        self._velocities = [0.0] * self.num_joints
        return self.read_state()

    def read_state(self) -> dict:
        return {
            "joint_positions": list(self._positions),
            "joint_velocities": list(self._velocities),
            "sensor_readings": {},
        }

    def apply_action(self, action: dict) -> None:
        vels = action.get("motor_velocities") or []
        if not isinstance(vels, (list, tuple)):
            raise TypeError("motor_velocities must be a list")
        n = min(len(vels), self.num_joints)
        # Frame each command per the wire spec — even though we can't
        # actually transmit, this exercises the encoder.
        for i in range(n):
            v = float(vels[i])
            frame = self._frame(i, self.CMD_SET_VELOCITY, _i16(int(v * 100)))
            assert frame[0:2] == self.SYNC, "frame must start with sync bytes"
            self._velocities[i] = v

    @classmethod
    def _frame(cls, joint_id: int, cmd: int, payload: bytes) -> bytes:
        if not (0 <= joint_id < 256):
            raise ValueError("joint_id out of range")
        if not (0 <= cmd < 256):
            raise ValueError("cmd out of range")
        header = bytes([0xFF, 0xFF, joint_id & 0xFF, cmd & 0xFF, len(payload)])
        crc = _crc16(header[2:] + payload)
        return header + payload + crc.to_bytes(2, "little")


def _i16(v: int) -> bytes:
    """Clamp to signed 16-bit range and pack little-endian."""
    if v > 32767:
        v = 32767
    elif v < -32768:
        v = -32768
    return v.to_bytes(2, "little", signed=True)


def _crc16(data: bytes) -> int:
    """Tiny CRC-16/IBM. Sufficient for framing demo — replace if vendor differs."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


class RobotArmBridge:
    """Same shape as ``EchoMapEnv`` so identical agent code drives both."""

    def __init__(self, backend: ArmBackend):
        if not isinstance(backend, ArmBackend):
            raise TypeError("backend must be an ArmBackend instance")
        self._backend = backend
        self._step_count = 0

    @property
    def backend(self) -> ArmBackend:
        return self._backend

    @property
    def observation_space(self) -> dict:
        return {
            "num_joints": self._backend.num_joints,
            "has_gripper": self._backend.has_gripper,
        }

    @property
    def action_space(self) -> dict:
        return {
            "num_motors": self._backend.num_joints,
            "num_grippers": 1 if self._backend.has_gripper else 0,
        }

    @property
    def capabilities(self) -> list:
        caps = ["observe", "step", "motors"]
        if self._backend.has_gripper:
            caps.append("grippers")
        return caps

    def reset(self) -> tuple:
        obs = self._backend.reset()
        self._step_count = 0
        return obs, {"step_count": 0}

    def step(self, action: dict) -> tuple:
        self._backend.apply_action(action)
        obs = self._backend.read_state()
        self._step_count += 1
        info = {"step_count": self._step_count}
        return obs, 0.0, False, info

    def observe(self) -> tuple:
        return self._backend.read_state(), {"step_count": self._step_count}

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "RobotArmBridge":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False
