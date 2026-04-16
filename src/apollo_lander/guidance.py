"""
Apollo P66 guidance and Digital Autopilot (DAP) controller.

Implements the semi-manual control mode used during the final phase
of lunar descent. In P66, the astronaut controlled attitude via the
Rotational Hand Controller (RHC) and adjusted the rate of descent
via the ROD switch, while the AGC handled throttle and attitude hold.

This module translates abstract actions (RHC deflection, ROD clicks)
into physical thrust vectors for the RK4 physics engine.
"""

import numpy as np

from apollo_lander.constants import (
    MAX_ROTATION_RATE,
    MAX_THRUST,
    MIN_THRUST,
    MU_MOON,
    P66_INITIAL_DESCENT_RATE,
    R_MOON,
    ROD_INCREMENT,
)
from apollo_lander.transforms import body_to_world


class ApolloP66Guidance:
    """
    P66 flight control system.

    Mimics the Apollo Guidance Computer's Program 66 mode where:
    - The astronaut commands attitude rates via a 3-axis hand controller
    - The astronaut adjusts sink rate via ROD switch clicks
    - The AGC handles throttle to maintain the commanded descent rate
    - The DAP holds attitude when no input is given

    Attributes:
        target_descent_rate: Current commanded descent rate (m/s, negative=down).
        target_attitude: Current targeted attitude [roll, pitch, yaw] (rad).
        max_rate: Maximum rotation rate from full stick deflection (rad/s).
    """

    def __init__(self) -> None:
        self.target_descent_rate = P66_INITIAL_DESCENT_RATE
        self.target_attitude = np.array([0.0, 0.0, 0.0])
        self.max_rate = MAX_ROTATION_RATE

        # PD attitude controller gains
        self._kp_attitude = 2.0
        self._kd_attitude = 1.5

        # PI throttle controller gains
        self._kp_throttle = 0.5
        self._ki_throttle = 0.1
        self._throttle_error_integral = 0.0

        # Previous attitude for derivative computation
        self._prev_attitude_error = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        self.target_descent_rate = P66_INITIAL_DESCENT_RATE
        self.target_attitude = np.array([0.0, 0.0, 0.0])
        self._throttle_error_integral = 0.0
        self._prev_attitude_error = np.zeros(3)

    def process_controls(
        self,
        state: np.ndarray,
        attitude: np.ndarray,
        rhc_input: np.ndarray,
        rod_action: int,
        dt: float,
    ) -> np.ndarray:
        """
        Translate pilot inputs into a physical thrust vector.

        Args:
            state: 7D state vector [rx, ry, rz, vx, vy, vz, mass].
            attitude: Current spacecraft attitude [roll, pitch, yaw] (rad).
            rhc_input: 3D array [-1..1] for RHC stick deflection
                [pitch, roll, yaw].
            rod_action: ROD switch action: 0=none, 1=decrease sink (up),
                2=increase sink (down).
            dt: Time step in seconds.

        Returns:
            3D thrust vector in the inertial frame (N).
        """
        # 1. Update target descent rate from ROD switch
        if rod_action == 1:
            self.target_descent_rate += ROD_INCREMENT  # decrease sink rate
        elif rod_action == 2:
            self.target_descent_rate -= ROD_INCREMENT  # increase sink rate

        # 2. Update target attitude from RHC input
        # RHC commands attitude rate; when centered, attitude is held
        commanded_rates = rhc_input * self.max_rate
        self.target_attitude = self.target_attitude + commanded_rates * dt

        # 3. DAP: PD controller to track target attitude
        attitude_error = self.target_attitude - attitude
        attitude_error_deriv = (attitude_error - self._prev_attitude_error) / dt
        self._prev_attitude_error = attitude_error.copy()

        # RCS torques would apply here in a full 6-DOF sim.
        # For now, attitude tracks the target directly
        # (the env will apply the effective attitude from the target).

        # 4. Throttle control: maintain target descent rate
        r_vec = state[0:3]
        v_vec = state[3:6]
        mass = state[6]

        r_norm = np.linalg.norm(r_vec)
        r_hat = r_vec / max(r_norm, 1.0)

        # Current vertical (radial) velocity
        current_vz = float(np.dot(v_vec, r_hat))

        # Gravity force magnitude along radial direction
        gravity_accel = MU_MOON / max(r_norm**2, R_MOON**2)
        gravity_force = mass * gravity_accel

        # Velocity error for throttle PI controller
        vz_error = self.target_descent_rate - current_vz
        self._throttle_error_integral += vz_error * dt

        # Clamp integral to prevent windup
        self._throttle_error_integral = np.clip(
            self._throttle_error_integral, -10.0, 10.0
        )

        # Commanded thrust: gravity compensation + PI correction
        correction = (
            self._kp_throttle * vz_error * mass
            + self._ki_throttle * self._throttle_error_integral * mass
        )
        commanded_thrust = gravity_force + correction

        # Clip to engine operating range
        commanded_thrust = float(np.clip(commanded_thrust, MIN_THRUST, MAX_THRUST))

        # 5. Resolve thrust into inertial frame using current attitude
        thrust_vector_world = body_to_world(commanded_thrust, attitude)

        return thrust_vector_world
