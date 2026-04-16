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
