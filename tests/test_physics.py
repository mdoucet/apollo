"""
Tests for the physics engine (RK4 integrator and equations of motion).
"""

import numpy as np
import pytest

from apollo_lander.constants import LM_TOTAL_MASS, MU_MOON, R_MOON
from apollo_lander.physics import (
    compute_altitude,
    compute_surface_velocity,
    lunar_equations_of_motion,
    rk4_step,
)


class TestLunarEquationsOfMotion:
    """Tests for lunar_equations_of_motion."""

    def test_free_fall_produces_downward_acceleration(self):
        """Object above Moon with no thrust should accelerate downward."""
        altitude = 1000.0
        r = R_MOON + altitude
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])
        thrust = np.zeros(3)

        d_state = lunar_equations_of_motion(0, state, thrust)

        # dr/dt should be zero (no initial velocity)
        assert np.allclose(d_state[0:3], 0.0)
        # dv/dt should point toward Moon center (negative z)
        assert d_state[5] < 0
        # dm/dt should be zero (no thrust)
        assert d_state[6] == pytest.approx(0.0)

    def test_thrust_increases_acceleration(self):
        """Thrust along +z should reduce downward acceleration."""
        altitude = 1000.0
        r = R_MOON + altitude
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])

        d_no_thrust = lunar_equations_of_motion(0, state, np.zeros(3))
        d_with_thrust = lunar_equations_of_motion(
            0, state, np.array([0.0, 0.0, 20000.0])
        )

        # With upward thrust, z-acceleration should be less negative
        assert d_with_thrust[5] > d_no_thrust[5]

    def test_mass_depletes_with_thrust(self):
        """Firing the engine should deplete mass."""
        r = R_MOON + 1000.0
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])
        thrust = np.array([0.0, 0.0, 10000.0])

        d_state = lunar_equations_of_motion(0, state, thrust)

        assert d_state[6] < 0  # mass decreasing

    def test_below_surface_returns_zeros(self):
        """State below lunar surface should return zero derivatives."""
        state = np.array([0.0, 0.0, R_MOON - 1.0, 0.0, 0.0, -10.0, LM_TOTAL_MASS])
        thrust = np.zeros(3)

        d_state = lunar_equations_of_motion(0, state, thrust)

        assert np.allclose(d_state, 0.0)


class TestRK4Step:
    """Tests for the RK4 integrator."""

    def test_free_fall_trajectory(self):
        """Free-falling object moves toward Moon over time."""
        r = R_MOON + 500.0
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])
        thrust = np.zeros(3)
        dt = 0.1

        new_state = rk4_step(state, thrust, dt)

        # Should have moved slightly toward Moon
        assert new_state[2] < state[2]
        # Should have gained downward velocity
        assert new_state[5] < 0
        # Mass unchanged (no thrust)
        assert new_state[6] == pytest.approx(LM_TOTAL_MASS)

    def test_energy_conservation_in_free_fall(self):
        """Total mechanical energy should be approximately conserved."""
        r = R_MOON + 200.0
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, LM_TOTAL_MASS])
        thrust = np.zeros(3)
        dt = 0.01
        mass = LM_TOTAL_MASS

        def energy(s: np.ndarray) -> float:
            r_norm = np.linalg.norm(s[0:3])
            v_norm = np.linalg.norm(s[3:6])
            return 0.5 * mass * v_norm**2 - MU_MOON * mass / r_norm

        e_initial = energy(state)

        # Run 100 steps of free fall
        for _ in range(100):
            state = rk4_step(state, thrust, dt)

        e_final = energy(state)

        # Energy should be conserved to ~0.01% for small dt
        assert abs(e_final - e_initial) / abs(e_initial) < 1e-4

    def test_mass_floor(self):
        """Mass should not go below zero even with heavy thrust."""
        r = R_MOON + 100.0
        state = np.array([0.0, 0.0, r, 0.0, 0.0, 0.0, 1.0])  # very low mass
        thrust = np.array([0.0, 0.0, 45040.0])  # max thrust
        dt = 10.0  # large step

        new_state = rk4_step(state, thrust, dt)

        assert new_state[6] > 0


class TestAltitudeAndVelocity:
    """Tests for helper functions."""

    def test_compute_altitude(self):
        """Altitude is distance above surface."""
        alt = 500.0
        state = np.array([0.0, 0.0, R_MOON + alt, 0, 0, 0, LM_TOTAL_MASS])

        assert compute_altitude(state) == pytest.approx(alt)

    def test_compute_altitude_at_surface(self):
        """Altitude at surface should be zero."""
        state = np.array([0.0, 0.0, R_MOON, 0, 0, 0, LM_TOTAL_MASS])

        assert compute_altitude(state) == pytest.approx(0.0)

    def test_surface_velocity_decomposition(self):
        """Radial and tangential velocity components."""
        r = R_MOON + 100.0
        # Pure radial velocity (downward along z)
        state = np.array([0.0, 0.0, r, 0.0, 0.0, -5.0, LM_TOTAL_MASS])

        v_vert, v_horiz = compute_surface_velocity(state)

        assert v_vert == pytest.approx(-5.0)
        assert v_horiz == pytest.approx(0.0, abs=1e-10)

    def test_surface_velocity_horizontal(self):
        """Pure horizontal velocity should have zero vertical component."""
        r = R_MOON + 100.0
        state = np.array([0.0, 0.0, r, 3.0, 0.0, 0.0, LM_TOTAL_MASS])

        v_vert, v_horiz = compute_surface_velocity(state)

        assert v_vert == pytest.approx(0.0, abs=1e-10)
        assert v_horiz == pytest.approx(3.0)
