"""
Flask web application for the Apollo Lander simulation.

Serves an HTML5 Canvas-based UI for playing the lunar descent
simulation in a web browser. The simulation runs server-side
with the client sending keyboard state at 10 Hz.

Usage:
    python -m apollo_lander.webapp
    # or via CLI:
    apollo play
"""

from typing import Any

import numpy as np

from flask import Flask, jsonify, render_template, request

import apollo_lander.envs  # noqa: F401 — triggers env registration
import gymnasium as gym


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)

    # Game state stored on the app instance (single-player)
    app.config["env"] = None
    app.config["obs"] = None
    app.config["info"] = None
    app.config["total_reward"] = 0.0
    app.config["game_over"] = False
    app.config["landing_success"] = False

    def _reset_game() -> None:
        """Initialize or reset the Gymnasium environment."""
        if app.config["env"] is None:
            app.config["env"] = gym.make("ApolloLander-v0")

        obs, info = app.config["env"].reset()
        app.config["obs"] = obs
        app.config["info"] = info
        app.config["total_reward"] = 0.0
        app.config["game_over"] = False
        app.config["landing_success"] = False

    def _make_state() -> dict[str, Any]:
        """Build a JSON-serializable state dict from the current env."""
        info = app.config["info"]
        obs = app.config["obs"]
        return {
            "altitude": float(info["altitude"]),
            "vertical_velocity": float(info["vertical_velocity"]),
            "horizontal_velocity": float(info["horizontal_velocity"]),
            "fuel_remaining": float(info["fuel_remaining"]),
            "mass": float(info["mass"]),
            "target_descent_rate": float(info["target_descent_rate"]),
            "total_reward": float(app.config["total_reward"]),
            "game_over": app.config["game_over"],
            "landing_success": app.config["landing_success"],
            "termination_reason": info.get("termination_reason", ""),
            "obs": obs.tolist() if obs is not None else [],
        }

    @app.route("/")
    def index():
        """Serve the main game page."""
        return render_template("index.html")

    @app.route("/help")
    def help_page():
        """Serve the flight manual / help page."""
        return render_template("help.html")

    @app.route("/api/reset", methods=["POST"])
    def reset():
        """Reset the environment and return initial state."""
        _reset_game()
        return jsonify(_make_state())

    @app.route("/api/step", methods=["POST"])
    def step():
        """
        Step the simulation with the given action.

        Expects JSON body:
            {"rhc": [pitch, roll, yaw], "rod": 0|1|2}
        """
        if app.config["game_over"]:
            return jsonify(_make_state())

        data = request.get_json()
        if data is None:
            return jsonify({"error": "Missing JSON body"}), 400

        rhc = data.get("rhc", [0, 0, 0])
        rod = data.get("rod", 0)

        # Validate inputs
        if not isinstance(rhc, list) or len(rhc) != 3:
            return jsonify({"error": "rhc must be a list of 3 floats"}), 400
        if rod not in (0, 1, 2):
            return jsonify({"error": "rod must be 0, 1, or 2"}), 400

        # Clamp RHC values to [-1, 1]
        rhc_array = np.clip(
            np.array(rhc, dtype=np.float32), -1.0, 1.0
        )

        action = {"rhc": rhc_array, "rod": int(rod)}
        env = app.config["env"]
        obs, reward, terminated, truncated, info = env.step(action)

        app.config["obs"] = obs
        app.config["info"] = info
        app.config["total_reward"] += reward

        if terminated or truncated:
            app.config["game_over"] = True
            app.config["landing_success"] = info.get(
                "landing_success", False
            )

        return jsonify(_make_state())

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5050)
