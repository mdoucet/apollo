"""
Apollo P66 guidance and Digital Autopilot (DAP) controller.

Implements the semi-manual control mode used during the final phase
of lunar descent. In P66, the astronaut controlled attitude via the
Rotational Hand Controller (RHC) and adjusted the rate of descent
via the ROD switch, while the AGC handled throttle and attitude hold.

The throttle controller faithfully implements the real AGC RODCOMP
algorithm from LUNAR_LANDING_GUIDANCE_EQUATIONS.agc (pp. 816-819):
    a_cmd = (VDGVERT - HDOTDISP) / TAUROD
where:
    VDGVERT  = desired vertical velocity (set by ROD switch clicks)
    HDOTDISP = measured vertical velocity
    TAUROD   = time constant (~2 seconds, pad-loaded erasable)

The real AGC used a simple proportional control law with gravity
compensation and a lead-lag term for throttle actuator lag. There
was NO integral term — P66 was intentionally simple and stable.

This module translates abstract actions (RHC deflection, ROD clicks)
into physical thrust vectors for the RK4 physics engine.
"""

import numpy as np

from apollo_lander.constants import (
    LAG_OVER_TAU,
    MAX_ROTATION_RATE,
    MAX_THRUST,
    MIN_THRUST,
    MU_MOON,
    P66_INITIAL_DESCENT_RATE,
    R_MOON,
    ROD_INCREMENT,
    TAUROD,
)
from apollo_lander.transforms import body_to_world


class ApolloP66Guidance:
    """
    P66 flight control system.

    Mimics the Apollo Guidance Computer's Program 66 mode where:
    - The RHC operates in Rate Command / Attitude Hold (RCAH) mode:
      stick deflected → commands rotation rate; stick centered → AGC
      captures current attitude and holds it indefinitely
    - The ROD switch is edge-triggered: each click adjusts VDGVERT
      by ±1 ft/s; holding the switch has no additional effect
    - The AGC handles throttle to maintain the commanded descent rate
    - The DAP holds attitude when no input is given

    The throttle controller uses the real RODCOMP proportional law
    from the AGC source (not a PI controller):
        a_cmd = (VDGVERT - HDOTDISP) / TAUROD
    with gravity compensation and throttle lag lead-lag filter.

    Attributes:
        target_descent_rate: Current commanded descent rate (m/s, negative=down).
            Corresponds to AGC erasable VDGVERT.
        target_attitude: Current targeted attitude [roll, pitch, yaw] (rad).
        max_rate: Maximum rotation rate from full stick deflection (rad/s).
    """

    def __init__(self) -> None:
        self.target_descent_rate = P66_INITIAL_DESCENT_RATE
        self.target_attitude = np.array([0.0, 0.0, 0.0])
        self.max_rate = MAX_ROTATION_RATE

        # PD attitude controller gains (DAP, separate from RODCOMP)
        self._kp_attitude = 2.0
        self._kd_attitude = 1.5

        # RODCOMP state: previous commanded force for lag compensation
        # AGC: FCOLD — previous throttle command (LUNAR_LANDING_GUIDANCE_EQUATIONS.agc)
        self._fcold = 0.0

        # Previous attitude for derivative computation
        self._prev_attitude_error = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        self.target_descent_rate = P66_INITIAL_DESCENT_RATE
        self.target_attitude = np.array([0.0, 0.0, 0.0])
        self._fcold = 0.0
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

        # 2. RHC: Rate Command / Attitude Hold (RCAH)
        #
        # Real AGC P66 behavior:
        #   OUT OF DETENT (stick deflected): The stick deflection is read
        #     as a commanded rotation RATE. The DAP fires RCS jets to
        #     maintain that rotational velocity. Full deflection = max rate
        #     (20°/s). target_attitude integrates accordingly.
        #
        #   IN DETENT (stick centered): The moment the transducers read
        #     zero, the AGC captures the CURRENT spacecraft attitude as
        #     the new hold target and fires opposing jets to kill any
        #     residual rotation, freezing the craft at that angle.
        #
        # Source: Apollo Operations Handbook, P66 RCAH mode description
        rhc_magnitude = float(np.linalg.norm(rhc_input))

        if rhc_magnitude > 0.01:
            # Rate Command mode: integrate commanded rate into target
            commanded_rates = rhc_input * self.max_rate
            self.target_attitude = self.target_attitude + commanded_rates * dt
        else:
            # Attitude Hold mode: capture current attitude as hold target
            # This is the key RCAH behavior — when the stick centers,
            # the AGC freezes the craft at its CURRENT orientation,
            # not wherever target_attitude had drifted to.
            self.target_attitude = attitude.copy()

        # 3. DAP: PD controller to track target attitude
        attitude_error = self.target_attitude - attitude
        # attitude_error_deriv = (attitude_error - self._prev_attitude_error) / dt
        self._prev_attitude_error = attitude_error.copy()

        # RCS torques would apply here in a full 6-DOF sim.
        # For now, attitude tracks the target directly
        # (the env will apply the effective attitude from the target).

        # 4. Throttle control: AGC RODCOMP proportional law
        # Real AGC source: LUNAR_LANDING_GUIDANCE_EQUATIONS.agc pp.816-819
        # Algorithm: a_cmd = (VDGVERT - HDOTDISP) / TAUROD
        r_vec = state[0:3]
        v_vec = state[3:6]
        mass = state[6]

        r_norm = np.linalg.norm(r_vec)
        r_hat = r_vec / max(r_norm, 1.0)

        # Current vertical (radial) velocity = HDOTDISP
        current_vz = float(np.dot(v_vec, r_hat))

        # Gravity acceleration along radial direction
        gravity_accel = MU_MOON / max(r_norm**2, R_MOON**2)

        # RODCOMP: commanded acceleration
        # a_cmd = (VDGVERT - HDOTDISP) / TAUROD
        a_cmd = (self.target_descent_rate - current_vz) / TAUROD

        # Total commanded acceleration: RODCOMP + gravity compensation
        total_accel = a_cmd + gravity_accel

        # Convert acceleration to force
        commanded_force = total_accel * mass

        # Throttle lag compensation: FC += LAG/TAU * (FC - FCOLD)
        # This lead-lag filter compensates for the DPS throttle actuator
        # time constant (THROTLAG = 0.2s). Source: AGC RODCOMP code.
        commanded_force += LAG_OVER_TAU * (commanded_force - self._fcold)
        self._fcold = commanded_force

        # Clip to engine operating range (MINFORCE..MAXFORCE)
        commanded_thrust = float(np.clip(commanded_force, MIN_THRUST, MAX_THRUST))

        # 5. Resolve thrust into inertial frame using current attitude
        thrust_vector_world = body_to_world(commanded_thrust, attitude)

        return thrust_vector_world
