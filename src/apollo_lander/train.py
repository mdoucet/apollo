"""
RL training script for the Apollo Lander.

Uses Stable Baselines3 (PPO or SAC) to train an agent to land
the Lunar Module using the P66 control interface.

Usage:
    python -m apollo_lander.train
    python -m apollo_lander.train --algo sac --timesteps 500000
"""

from pathlib import Path

import click

import apollo_lander.envs  # noqa: F401
import gymnasium as gym

from apollo_lander.wrappers import FlatActionWrapper


def make_training_env() -> gym.Env:
    """Create and wrap the environment for RL training."""
    env = gym.make("ApolloLander-v0")
    env = FlatActionWrapper(env)
    return env


@click.command("train")
@click.option(
    "--algo",
    type=click.Choice(["ppo", "sac"]),
    default="ppo",
    help="RL algorithm (default: ppo)",
)
@click.option(
    "--timesteps",
    type=int,
    default=100_000,
    help="Total training timesteps (default: 100000)",
)
@click.option(
    "--output",
    type=click.Path(),
    default="models/apollo_lander",
    help="Path to save the trained model",
)
@click.option(
    "--tensorboard",
    type=click.Path(),
    default="logs/",
    help="Tensorboard log directory",
)
def train(algo: str, timesteps: int, output: str, tensorboard: str) -> None:
    """Train an RL agent to land the Apollo Lunar Module."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError:
        click.echo(
            "stable-baselines3 is required for training. "
            "Install with: pip install apollo-lander[rl]"
        )
        raise SystemExit(1)

    env = make_training_env()

    click.echo(f"Training with {algo.upper()} for {timesteps:,} timesteps...")
    click.echo(f"Observation space: {env.observation_space}")
    click.echo(f"Action space: {env.action_space}")

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard,
            learning_rate=3e-4,
            batch_size=256,
        )

    model.learn(total_timesteps=timesteps)

    # Save model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    click.echo(f"Model saved to {output_path}")

    env.close()


if __name__ == "__main__":
    train()
