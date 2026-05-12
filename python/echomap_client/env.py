"""Gym-compatible environment client for EchoMap simulation."""

import json


class EchoMapEnv:
    """Gym-compatible client that connects to an EchoMap simulation server via WebSocket.

    Provides reset(), step(action), close() methods following the OpenAI Gym interface.
    Connects to the EchoMap WebSocket server and communicates using the JSON protocol
    defined in src/agent/protocol.rs.

    Usage:
        with EchoMapEnv(host="localhost", port=9002, robot_id=0) as env:
            env.connect()
            obs = env.reset()
            for _ in range(100):
                obs, reward, done, info = env.step({
                    "motor_velocities": [1.0, -0.5],
                    "gripper_commands": [True],
                })
                if done:
                    break
    """

    def __init__(self, host="localhost", port=9002, robot_id=0):
        """Initialize the environment client.

        Args:
            host: Server hostname (default: "localhost").
            port: WebSocket server port (default: 9002).
            robot_id: Robot ID to connect to (default: 0).
        """
        self._host = host
        self._port = port
        self._robot_id = robot_id
        self._ws = None
        self._observation_space = None
        self._action_space = None

    @property
    def observation_space(self):
        """Return the observation space dict received from the server on connect."""
        return self._observation_space

    @property
    def action_space(self):
        """Return the action space dict received from the server on connect."""
        return self._action_space

    def connect(self):
        """Connect to the EchoMap server via WebSocket.

        Sends a Connect message and waits for the Connected response,
        which provides observation_space and action_space.

        Raises:
            ConnectionError: If the server returns an error or unexpected response.
            ImportError: If websocket-client is not installed.
        """
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client is required. Install it with: pip install websocket-client"
            )

        url = f"ws://{self._host}:{self._port}"
        self._ws = websocket.create_connection(url)

        # Send Connect message
        connect_msg = json.dumps({"type": "connect", "robot_id": self._robot_id})
        self._ws.send(connect_msg)

        # Receive Connected response
        response = json.loads(self._ws.recv())
        if response.get("type") == "error":
            raise ConnectionError(
                f"Server error: {response.get('message', 'unknown error')}"
            )
        if response.get("type") != "connected":
            raise ConnectionError(
                f"Unexpected response type: {response.get('type')}"
            )

        self._observation_space = response.get("observation_space")
        self._action_space = response.get("action_space")

    def reset(self):
        """Reset the environment and return the initial observation.

        Returns:
            tuple: (state, info) where info contains messages.

        Raises:
            RuntimeError: If not connected or server returns an error.
        """
        self._ensure_connected()

        reset_msg = json.dumps({"type": "reset"})
        self._ws.send(reset_msg)

        response = json.loads(self._ws.recv())
        self._check_error(response)

        if response.get("type") != "observation":
            raise RuntimeError(
                f"Expected observation response, got: {response.get('type')}"
            )

        info = {
            "messages": response.get("messages", []),
            "match_state": response.get("match_state"),
        }
        return response.get("state"), info

    def step(self, action):
        """Take a step in the environment with the given action.

        Args:
            action: Dict with "motor_velocities" (list of floats) and optionally
                    "gripper_commands" (list of bools).

        Returns:
            tuple: (observation, reward, done, info) gym-style tuple where:
                - observation (dict): Current state.
                - reward (float): Reward for this step.
                - done (bool): Whether the episode is finished.
                - info (dict): Additional info including step_count.

        Raises:
            RuntimeError: If not connected or server returns an error.
        """
        self._ensure_connected()

        # Build the action payload
        action_payload = {
            "motor_velocities": action.get("motor_velocities", []),
        }
        if "gripper_commands" in action:
            action_payload["gripper_commands"] = action["gripper_commands"]
        else:
            action_payload["gripper_commands"] = []

        step_msg = json.dumps({"type": "step", "action": action_payload})
        self._ws.send(step_msg)

        response = json.loads(self._ws.recv())
        self._check_error(response)

        if response.get("type") != "observation":
            raise RuntimeError(
                f"Expected observation response, got: {response.get('type')}"
            )

        observation = response.get("state")
        reward = response.get("reward", 0.0)
        done = response.get("done", False)
        info = {
            "step_count": response.get("step_count", 0),
            "messages": response.get("messages", []),
            "match_state": response.get("match_state"),
        }

        return observation, reward, done, info

    def observe(self):
        """Get the current observation without stepping the simulation.

        Returns:
            tuple: (state, info) where info contains messages.

        Raises:
            RuntimeError: If not connected or server returns an error.
        """
        self._ensure_connected()

        observe_msg = json.dumps({"type": "observe"})
        self._ws.send(observe_msg)

        response = json.loads(self._ws.recv())
        self._check_error(response)

        if response.get("type") != "observation":
            raise RuntimeError(
                f"Expected observation response, got: {response.get('type')}"
            )

        info = {
            "messages": response.get("messages", []),
            "match_state": response.get("match_state"),
        }
        return response.get("state"), info

    def send_message(self, to_robot_id, content):
        """Send a text message to another agent.

        Args:
            to_robot_id: Target robot ID.
            content: Message text (max 1024 bytes).

        Raises:
            RuntimeError: If not connected or server returns an error.
        """
        self._ensure_connected()

        msg = json.dumps({
            "type": "send_message",
            "to_robot_id": to_robot_id,
            "content": content,
        })
        self._ws.send(msg)

        response = json.loads(self._ws.recv())
        self._check_error(response)

        if response.get("type") != "message_sent":
            raise RuntimeError(
                f"Expected message_sent response, got: {response.get('type')}"
            )

    def close(self):
        """Close the connection to the server.

        Sends a Close message and shuts down the WebSocket connection.
        Safe to call multiple times.
        """
        if self._ws is not None:
            try:
                close_msg = json.dumps({"type": "close"})
                self._ws.send(close_msg)
                # Wait for Closed acknowledgment
                response = json.loads(self._ws.recv())
                # Response should be {"type": "closed"} but we don't enforce it
            except Exception:
                pass  # Best-effort close
            finally:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None

    def _ensure_connected(self):
        """Raise RuntimeError if not connected."""
        if self._ws is None:
            raise RuntimeError(
                "Not connected. Call connect() first or use as context manager."
            )

    def _check_error(self, response):
        """Raise RuntimeError if the server returned an error response."""
        if response.get("type") == "error":
            raise RuntimeError(
                f"Server error: {response.get('message', 'unknown error')}"
            )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the connection."""
        self.close()
        return False
