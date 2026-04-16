"""
Command-line interface for Apollo Lander.

Example usage:
    $ apollo play
    $ apollo train
    $ apollo evaluate
"""

import click


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Apollo Lander — Moon landing simulation based on Apollo 11 AGC code."""


@main.command()
def play() -> None:
    """Launch manual play mode in the web browser.

    Starts a local Flask server and opens the game page.

    Controls:
        Arrow keys: Pitch/Roll (Rotational Hand Controller)
        Q/E: Yaw
        Page Up: Decrease sink rate (ROD up)
        Page Down: Increase sink rate (ROD down)
        Enter: Reset after game over
    """
    from apollo_lander.manual import play as run_play

    run_play()


@main.command()
@click.option("--episodes", type=int, default=0, help="Headless episodes (0 = launch browser)")
def autopilot(episodes: int) -> None:
    """Run the classical AGC autopilot (no RL).

    With no options, launches the browser to watch the autopilot fly.
    With --episodes N, runs N headless episodes and prints statistics.
    """
    from apollo_lander.autopilot import AGCAutopilot

    if episodes > 0:
        # Headless evaluation
        import apollo_lander.envs  # noqa: F401
        import gymnasium as gym

        env = gym.make("ApolloLander-v0")
        agent = AGCAutopilot()

        successes = 0
        total_rewards: list[float] = []
        final_velocities: list[tuple[float, float]] = []

        click.echo(f"Running AGC autopilot for {episodes} episodes...")

        for ep in range(episodes):
            obs, info = env.reset()
            agent.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action, _ = agent.predict(obs)
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
            reason = info.get("termination_reason", "")
            click.echo(
                f"  Episode {ep + 1:3d}: {status:7s} ({reason:7s}) | "
                f"Reward: {ep_reward:8.1f} | "
                f"Vv: {info['vertical_velocity']:6.2f} | "
                f"Vh: {info['horizontal_velocity']:6.2f} | "
                f"Fuel: {info['fuel_remaining']:7.1f}"
            )

        import numpy as np

        click.echo(f"\n{'='*55}")
        click.echo(f"AGC Autopilot — {successes}/{episodes} landings ({100*successes/episodes:.1f}%)")
        click.echo(f"Avg reward:   {np.mean(total_rewards):.1f}")
        click.echo(f"Avg |Vv|:     {np.mean([abs(v) for v, _ in final_velocities]):.2f} m/s")
        click.echo(f"Avg Vh:       {np.mean([h for _, h in final_velocities]):.2f} m/s")
        env.close()
    else:
        # Launch browser mode
        from apollo_lander.manual import play as run_play

        run_play(mode="autopilot")


@main.command()
@click.option("--algo", type=click.Choice(["ppo", "sac"]), default="ppo", help="RL algorithm")
@click.option("--timesteps", type=int, default=100_000, help="Training timesteps")
@click.option("--output", default="models/apollo_lander", help="Model save path")
@click.option("--tensorboard", default="logs/", help="Tensorboard log dir")
def train(algo: str, timesteps: int, output: str, tensorboard: str) -> None:
    """Train an RL agent to land the Lunar Module."""
    from apollo_lander.train import train as run_train

    # Invoke the train click command programmatically
    ctx = click.Context(run_train)
    ctx.invoke(run_train, algo=algo, timesteps=timesteps, output=output, tensorboard=tensorboard)


@main.command()
@click.option("--model", default="models/apollo_lander", help="Path to saved model")
@click.option("--algo", type=click.Choice(["ppo", "sac"]), default="ppo", help="Algorithm used")
@click.option("--episodes", type=int, default=20, help="Number of episodes")
@click.option("--render/--no-render", default=False, help="Render visually")
def evaluate(model: str, algo: str, episodes: int, render: bool) -> None:
    """Evaluate a trained agent."""
    from apollo_lander.evaluate import evaluate as run_eval

    ctx = click.Context(run_eval)
    ctx.invoke(run_eval, model=model, algo=algo, episodes=episodes, render=render)


if __name__ == "__main__":
    main()
