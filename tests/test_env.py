"""
Tests for the Gymnasium environment.
"""

import numpy as np
import pytest

import apollo_lander.envs  # noqa: F401
import gymnasium as gym

from apollo_lander.envs.apollo_lander_env import ApolloLanderEnv


class TestApolloLanderEnv:
    """Tests for ApolloLanderEnv."""

    def test_env_creation(self):
        """Environment can be created via gymnasium.make."""
        env = gym.make("ApolloLander-v0")
        assert env is not None
        env.close()

    def test_reset_returns_valid_observation(self):
        """Reset should return an observation within the obs space."""
        env = ApolloLanderEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (12,)
        assert env.observation_space.contains(obs)
        assert "altitude" in info
        assert info["altitude"] > 0
        env.close()

    def test_step_returns_valid_tuple(self):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = ApolloLanderEnv()
        env.reset(seed=42)

        action = {"rhc": np.zeros(3, dtype=np.float32), "rod": 0}
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (12,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_action_space_sample(self):
        """Random actions from the action space should work."""
        env = ApolloLanderEnv()
        env.reset(seed=42)

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        env.close()

    def test_episode_terminates(self):
        """An episode should eventually terminate (crash or land)."""
        env = ApolloLanderEnv(max_steps=5000)
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 5000:
            action = {"rhc": np.zeros(3, dtype=np.float32), "rod": 0}
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done
        env.close()

    def test_fuel_depletion_terminates(self):
        """Running out of fuel should terminate the episode."""
        env = ApolloLanderEnv()
        env.reset(seed=42)

        # Set mass to just above dry mass to force quick fuel depletion
        env._state[6] = 6705.0  # dry + ascent + tiny fuel

        done = False
        for _ in range(5000):
            action = {"rhc": np.zeros(3, dtype=np.float32), "rod": 0}
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        assert done
        env.close()

    def test_deterministic_reset(self):
        """Same seed should produce same initial state."""
        env = ApolloLanderEnv()

        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        assert np.allclose(obs1, obs2)
        env.close()
