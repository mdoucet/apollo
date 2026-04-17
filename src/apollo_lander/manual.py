"""
Manual play mode for the Apollo Lander.

Launches a Flask web server and opens the simulation in the
user's default web browser. The Lunar Module is controlled via
keyboard input using authentic P66 controls (RHC + ROD switch).
"""

import webbrowser

from apollo_lander.webapp import create_app


def play(host: str = "127.0.0.1", port: int = 5050, mode: str = "manual", crazy: bool = False) -> None:
    """
    Run the manual play mode via web browser.

    Starts a local Flask server and opens the game page.

    Controls:
        Arrow keys: Pitch/Roll (Rotational Hand Controller)
        Q/E: Yaw
        Page Up: Decrease sink rate (ROD up)
        Page Down: Increase sink rate (ROD down)
        Enter: Reset after game over

    Args:
        host: Hostname to bind to.
        port: Port number.
        mode: Play mode — "manual" for keyboard, "autopilot" for AGC.
        crazy: If True, use much wider random initial conditions.
    """
    app = create_app(mode=mode, crazy=crazy)
    url = f"http://{host}:{port}"
    print(f"Starting Apollo Lander at {url} (mode: {mode})")
    print("Press Ctrl+C to stop the server.")
    webbrowser.open(url)
    app.run(host=host, port=port)


if __name__ == "__main__":
    play()
