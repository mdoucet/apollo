"""
Renderer module for the Apollo Lander simulation.

The visualization has moved to a Flask web app.
See :mod:`apollo_lander.webapp` for the browser-based UI.

This module is kept as a thin entry point for backward compatibility.
"""

from apollo_lander.webapp import create_app


def launch(host: str = "127.0.0.1", port: int = 5000) -> None:
    """
    Launch the web-based renderer.

    Args:
        host: Hostname to bind to.
        port: Port number.
    """
    app = create_app()
    app.run(host=host, port=port)
