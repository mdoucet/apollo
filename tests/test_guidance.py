"""
Tests for the P66 guidance controller and coordinate transforms.
"""

import numpy as np
import pytest

from apollo_lander.constants import (
    LM_TOTAL_MASS,
    MAX_THRUST,
    MIN_THRUST,
    P66_INITIAL_DESCENT_RATE,
    R_MOON,
    ROD_INCREMENT,
)
from apollo_lander.guidance import ApolloP66Guidance
from apollo_lander.transforms import body_to_world, world_to_body


class TestBodyToWorld:
    """Tests for coordinate transformations."""

    def test_identity_rotation(self):
        """Zero attitude should preserve the body-frame vector."""
        attitude = np.array([0.0, 0.0, 0.0])
        result = body_to_world(1000.0, attitude)

        # Thrust along +Z body should stay along +Z world
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)
        assert result[2] == pytest.approx(1000.0, abs=1e-10)

    def test_pitch_90_degrees(self):
        """90-degree pitch should rotate Z thrust toward X or Y."""
        attitude = np.array([0.0, np.pi / 2, 0.0])  # 90° pitch
        result = body_to_world(1000.0, attitude)

        # Magnitude should be preserved
        assert np.linalg.norm(result) == pytest.approx(1000.0, abs=0.1)
        # Z component should be near zero
        assert abs(result[2]) < 1.0

    def test_roundtrip(self):
        """body_to_world and world_to_body should be inverses."""
        attitude = np.array([0.1, 0.2, 0.3])
        t_world = body_to_world(500.0, attitude)
        t_body = world_to_body(t_world, attitude)

        expected_body = np.array([0.0, 0.0, 500.0])
        assert np.allclose(t_body, expected_body, atol=1e-10)


class TestApolloP66Guidance:
    """Tests for the P66 guidance controller."""

    def _make_hover_state(self) -> np.ndarray:
        """Create a state hovering 100m above the surface."""
        r = R_MOON + 100.0
        return np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])

    def test_rod_click_up_decreases_sink_rate(self):
        """ROD click up should make descent rate less negative."""
        guidance = ApolloP66Guidance()
        initial_rate = guidance.target_descent_rate

        state = self._make_hover_state()
        attitude = np.zeros(3)
        rhc = np.zeros(3)

        guidance.process_controls(state, attitude, rhc, rod_action=1, dt=0.1)

        assert guidance.target_descent_rate == pytest.approx(
            initial_rate + ROD_INCREMENT
        )

    def test_rod_click_down_increases_sink_rate(self):
        """ROD click down should make descent rate more negative."""
        guidance = ApolloP66Guidance()
        initial_rate = guidance.target_descent_rate

        state = self._make_hover_state()
        attitude = np.zeros(3)
        rhc = np.zeros(3)

        guidance.process_controls(state, attitude, rhc, rod_action=2, dt=0.1)

        assert guidance.target_descent_rate == pytest.approx(
            initial_rate - ROD_INCREMENT
        )

    def test_no_rod_click_preserves_rate(self):
        """No ROD input should not change descent rate."""
        guidance = ApolloP66Guidance()
        initial_rate = guidance.target_descent_rate

        state = self._make_hover_state()
        attitude = np.zeros(3)
        rhc = np.zeros(3)

        guidance.process_controls(state, attitude, rhc, rod_action=0, dt=0.1)

        assert guidance.target_descent_rate == pytest.approx(initial_rate)

    def test_rhc_input_changes_target_attitude(self):
        """RHC stick deflection should update target attitude."""
        guidance = ApolloP66Guidance()
        state = self._make_hover_state()
        attitude = np.zeros(3)
        rhc = np.array([1.0, 0.0, 0.0])  # full pitch input

        guidance.process_controls(state, attitude, rhc, rod_action=0, dt=0.1)

        # Target attitude pitch should have changed
        assert guidance.target_attitude[0] != 0.0

    def test_thrust_within_engine_limits(self):
        """Commanded thrust should always be within DPS operating range."""
        guidance = ApolloP66Guidance()
        state = self._make_hover_state()
        attitude = np.zeros(3)
        rhc = np.zeros(3)

        thrust = guidance.process_controls(state, attitude, rhc, rod_action=0, dt=0.1)
        thrust_mag = np.linalg.norm(thrust)

        assert thrust_mag >= MIN_THRUST - 1.0  # small tolerance for rotation
        assert thrust_mag <= MAX_THRUST + 1.0

    def test_reset_restores_defaults(self):
        """Reset should restore initial controller state."""
        guidance = ApolloP66Guidance()
        state = self._make_hover_state()
        attitude = np.zeros(3)

        # Make some changes
        guidance.process_controls(
            state, attitude, np.array([1.0, 0.5, 0.0]), rod_action=1, dt=0.1
        )

        guidance.reset()

        assert guidance.target_descent_rate == pytest.approx(
            P66_INITIAL_DESCENT_RATE
        )
        assert np.allclose(guidance.target_attitude, 0.0)
