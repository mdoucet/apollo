"""
Tests for the Apollo Lander CLI.
"""

from click.testing import CliRunner

from apollo_lander.cli import main


def test_cli_version():
    """Test CLI --version flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    """Test CLI --help shows available commands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "play" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
