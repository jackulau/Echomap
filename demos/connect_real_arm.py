#!/usr/bin/env python3
"""Drive a real (or mock) robot arm with the same agent code as the sim.

The point of this demo: the agent class doesn't care whether it's talking
to a Rust simulator or a physical Dynamixel arm. Same observation / action
schema, same loop. Swap the backend, keep the agent.

Usage:
    python3 demos/connect_real_arm.py --backend mock
    python3 demos/connect_real_arm.py --backend serial --port /dev/tty.usbserial-XXXX

The serial backend is a stub — it frames packets per the documented wire
format but does NOT open the port (use a vendor driver plugin for that).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from echomap_client.hardware import MockArm, RobotArmBridge, SerialArm


class SinusoidalAgent:
    """Toy agent — sweeps every joint with a different phase.

    Any agent with `decide(observation, info) -> (action, optional_message)`
    works against the bridge.
    """

    def __init__(self, num_joints: int, speed: float = 0.5):
        self.num_joints = num_joints
        self.speed = speed
        self._t = 0.0

    def decide(self, observation, info):
        self._t += 0.05
        vels = [
            self.speed * math.sin(self._t + i * (math.pi / self.num_joints))
            for i in range(self.num_joints)
        ]
        return {"motor_velocities": vels, "gripper_commands": []}, None


def make_backend(args):
    if args.backend == "mock":
        return MockArm(num_joints=args.num_joints, has_gripper=args.gripper)
    if args.backend == "serial":
        if not args.port:
            raise SystemExit("--port required for serial backend")
        return SerialArm(
            port=args.port,
            num_joints=args.num_joints,
            has_gripper=args.gripper,
        )
    raise SystemExit(f"unknown backend: {args.backend}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "serial"], default="mock")
    parser.add_argument("--port", default=None, help="serial device path")
    parser.add_argument("--num-joints", type=int, default=6)
    parser.add_argument("--gripper", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    backend = make_backend(args)
    print(f"Backend: {type(backend).__name__} (joints={backend.num_joints})")

    with RobotArmBridge(backend=backend) as bridge:
        agent = SinusoidalAgent(num_joints=bridge.observation_space["num_joints"])

        obs, info = bridge.reset()
        print(f"reset -> joints={obs['joint_positions']}")
        print(f"capabilities: {bridge.capabilities}")

        for step in range(args.steps):
            action, _msg = agent.decide(obs, info)
            obs, _reward, _done, info = bridge.step(action)
            if step % 5 == 0:
                pos = [round(p, 3) for p in obs["joint_positions"]]
                print(f"step={step:3d}  joint_pos={pos}")

    print("done.")


if __name__ == "__main__":
    main()
