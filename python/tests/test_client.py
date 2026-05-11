"""Tests for echomap_client Python SDK.

Works with both pytest and unittest. Run with:
    python3 -m pytest python/tests/test_client.py -v
    python3 python/tests/test_client.py
"""

import sys
import os
import unittest

# Add the python directory to sys.path so echomap_client is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestImport(unittest.TestCase):
    """Test that the package imports correctly."""

    def test_import(self):
        """Importing echomap_client should succeed."""
        from echomap_client import EchoMapEnv

        self.assertIsNotNone(EchoMapEnv)

    def test_version(self):
        """Package should have a version string."""
        import echomap_client

        self.assertIsInstance(echomap_client.__version__, str)
        self.assertTrue(len(echomap_client.__version__) > 0)


class TestEnvClassExists(unittest.TestCase):
    """Test that EchoMapEnv class exists and can be instantiated."""

    def test_env_class_exists(self):
        """EchoMapEnv should be callable and instantiable without connecting."""
        from echomap_client import EchoMapEnv

        env = EchoMapEnv()
        self.assertIsNotNone(env)

    def test_env_default_params(self):
        """EchoMapEnv should accept host, port, robot_id parameters."""
        from echomap_client import EchoMapEnv

        env = EchoMapEnv(host="127.0.0.1", port=9999, robot_id=5)
        self.assertIsNotNone(env)

    def test_env_no_connection_on_init(self):
        """Creating an EchoMapEnv should not attempt to connect."""
        from echomap_client import EchoMapEnv

        env = EchoMapEnv()
        # _ws should be None since we haven't connected
        self.assertIsNone(env._ws)


class TestEnvHasGymInterface(unittest.TestCase):
    """Test that EchoMapEnv has the expected gym-compatible interface."""

    def setUp(self):
        from echomap_client import EchoMapEnv

        self.env = EchoMapEnv()

    def test_has_reset(self):
        """EchoMapEnv should have a reset method."""
        self.assertTrue(hasattr(self.env, "reset"))
        self.assertTrue(callable(self.env.reset))

    def test_has_step(self):
        """EchoMapEnv should have a step method."""
        self.assertTrue(hasattr(self.env, "step"))
        self.assertTrue(callable(self.env.step))

    def test_has_close(self):
        """EchoMapEnv should have a close method."""
        self.assertTrue(hasattr(self.env, "close"))
        self.assertTrue(callable(self.env.close))

    def test_has_observe(self):
        """EchoMapEnv should have an observe method."""
        self.assertTrue(hasattr(self.env, "observe"))
        self.assertTrue(callable(self.env.observe))

    def test_has_connect(self):
        """EchoMapEnv should have a connect method."""
        self.assertTrue(hasattr(self.env, "connect"))
        self.assertTrue(callable(self.env.connect))

    def test_has_observation_space(self):
        """EchoMapEnv should have an observation_space property."""
        self.assertTrue(hasattr(self.env, "observation_space"))
        # Should be None before connect
        self.assertIsNone(self.env.observation_space)

    def test_has_action_space(self):
        """EchoMapEnv should have an action_space property."""
        self.assertTrue(hasattr(self.env, "action_space"))
        # Should be None before connect
        self.assertIsNone(self.env.action_space)

    def test_context_manager(self):
        """EchoMapEnv should support context manager protocol."""
        self.assertTrue(hasattr(self.env, "__enter__"))
        self.assertTrue(hasattr(self.env, "__exit__"))

    def test_context_manager_returns_self(self):
        """__enter__ should return self."""
        from echomap_client import EchoMapEnv

        env = EchoMapEnv()
        result = env.__enter__()
        self.assertIs(result, env)
        env.__exit__(None, None, None)

    def test_reset_raises_without_connection(self):
        """reset() should raise RuntimeError when not connected."""
        with self.assertRaises(RuntimeError):
            self.env.reset()

    def test_step_raises_without_connection(self):
        """step() should raise RuntimeError when not connected."""
        with self.assertRaises(RuntimeError):
            self.env.step({"motor_velocities": [1.0]})

    def test_observe_raises_without_connection(self):
        """observe() should raise RuntimeError when not connected."""
        with self.assertRaises(RuntimeError):
            self.env.observe()

    def test_close_safe_when_not_connected(self):
        """close() should not raise when not connected."""
        self.env.close()  # Should not raise


if __name__ == "__main__":
    unittest.main()
