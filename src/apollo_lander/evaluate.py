"""
Evaluate a trained RL agent on the Apollo Lander environment.

Loads a saved model and runs episodes, reporting success rate
and landing statistics. Optionally renders the agent's performance.

Usage:
    python -m apollo_lander.evaluate
    python -m apollo_lander.evaluate --model models/apollo_lander --episodes 50 --render
"""

from pathlib import Path

import click
import numpy as np

import apollo_lander.envs  # noqa: F401
import gymnasium as gym

from apollo_lander.wrappers import FlatActionWrapper


@click.command("evaluate")
@click.option(
    "--model",
    type=click.Path(exists=False),
    default="models/apollo_lander",
    help="Path to saved model (without extension)",
)
@click.option(
    "--algo", type=click.Choice(["ppo", "sac"]), default="ppo", help="Algorithm used"
)
@click.option("--episodes", type=int, default=20, help="Number of evaluation episodes")
@click.option("--render/--no-render", default=False, help="Render the agent visually")
def evaluate(model: str, algo: str, episodes: int, render: bool) -> None:
    """Evaluate a trained agent on the Apollo Lander."""
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError:
        click.echo(
            "stable-baselines3 is required. Install with: pip install apollo-lander[rl]"
        )
        raise SystemExit(1)

    model_path = Path(f"{model}.zip")
    if not model_path.exists():
        click.echo(f"Model not found: {model_path}")
        raise SystemExit(1)

    env = gym.make("ApolloLander-v0")
    env = FlatActionWrapper(env)

    if render:
        click.echo(
            "Visual rendering uses the web UI. "
            "Run 'apollo play' to launch the browser-based renderer."
        )

    algo_cls = PPO if algo == "ppo" else SAC
    agent = algo_cls.load(str(model), env=env)

    click.echo(f"Evaluating {algo.upper()} agent for {episodes} episodes...")

    successes = 0
    total_rewards = []
    final_velocities = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        success = info.get("landing_success", False)
        if success:
            successes += 1
        total_rewards.append(ep_reward)
        final_velocities.append(
            (info["vertical_velocity"], info["horizontal_velocity"])
        )

        status = "LANDED" if success else "CRASHED"
        click.echo(
            f"  Episode {ep + 1:3d}: {status} | "
            f"Reward: {ep_reward:8.1f} | "
            f"Vv: {info['vertical_velocity']:6.2f} | "
            f"Vh: {info['horizontal_velocity']:6.2f} | "
            f"Fuel: {info['fuel_remaining']:7.1f}"
        )

    click.echo(f"\n{'=' * 50}")
    click.echo(
        f"Success rate: {successes}/{episodes} ({100 * successes / episodes:.1f}%)"
    )
    click.echo(f"Avg reward:   {np.mean(total_rewards):.1f}")
    click.echo(
        f"Avg |Vv|:     {np.mean([abs(v) for v, _ in final_velocities]):.2f} m/s"
    )
    click.echo(
        f"Avg |Vh|:     {np.mean([abs(h) for _, h in final_velocities]):.2f} m/s"
    )

    env.close()


if __name__ == "__main__":
    evaluate()
