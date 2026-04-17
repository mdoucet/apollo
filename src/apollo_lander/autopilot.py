"""
Classical AGC autopilot agent for the Apollo Lander environment.

Implements a deterministic controller that mimics the real Apollo P66
landing procedure, producing actions (RHC + ROD) at each DAP cycle
based on the current observation.

This is NOT a reinforcement-learning agent — it uses hand-crafted control
laws derived from two distinct sources:

**AGC Software (what the computer did):**
    The throttle is controlled by the RODCOMP algorithm in guidance.py,
    which faithfully implements the real AGC proportional law:
        a_cmd = (VDGVERT - HDOTDISP) / TAUROD
    The autopilot interacts with this via ROD switch clicks, exactly
    as the real hardware interface worked. The AGC processed the clicks
    to adjust VDGVERT and managed the throttle autonomously.
    Source: LUNAR_LANDING_GUIDANCE_EQUATIONS.agc pp. 816-819

**Crew Procedures (what the astronaut did):**
    The AGC did NOT automate horizontal velocity control — that was
    100% manual by the Commander using the RHC and the Landing Point
    Designator (LPD) display. Our autopilot simulates a skilled CDR by:
    1. Nulling horizontal velocity via PD control on the RHC
    2. Scheduling descent rate by altitude (matching Armstrong's profile)
    3. Damping attitude back to vertical for final touchdown

    In the real mission, the CDR would look out the window, check the
    LPDT cross-pointers showing forward/lateral velocity (driven by
    the landing radar via the FLASHVHZ/FLASHVHY routines), and
    manually steer toward the landing pad.

Usage:
    apollo autopilot              # watch in browser
    apollo autopilot --episodes 20  # headless evaluation
"""

import numpy as np

from apollo_lander.constants import (
    LM_CG_TO_FOOTPAD,
    MAX_LANDING_HORIZONTAL_VEL,
    ROD_INCREMENT,
)


class AGCAutopilot:
    """
    Classical controller implementing the Apollo P66 landing procedure.

    The controller reads the 12-D observation vector and produces
    Dict actions (rhc, rod) compatible with ``ApolloLanderEnv``.

    **AGC RODCOMP (automated by the computer):**
    The throttle is handled entirely by guidance.py, which implements
    the real RODCOMP proportional law from the AGC source code. This
    autopilot only sends ROD switch clicks to adjust VDGVERT — exactly
    as a real astronaut would interact with the ROD toggle switch.

    **Crew Procedures (simulating the Commander):**
    Horizontal velocity nulling was NOT part of the AGC software.
    The real P66 only computed VHORIZ for the cross-pointer display —
    the CDR steered manually using the RHC. This autopilot simulates
    a skilled CDR using a PD controller to null lateral velocity.

    Attributes:
        rod_cooldown: Ticks remaining before next ROD click (prevents
            spamming the switch faster than the RODCOMP 1-second cycle).
    """

    ROD_COOLDOWN_TICKS = 10  # 1 second between ROD clicks (10 × 0.1s)

    # Maximum tilt from vertical (rad). Real Apollo crews rarely
    # exceeded 15° during P66; we allow a bit more for margin.
    MAX_TILT = 0.25  # ~14 degrees

    # Descent-rate schedule: (altitude_threshold, target_rate_fps)
    # Apollo crews typically used: −5 at high alt, −3 mid, −1 low
    # Minimum −1 ft/s prevents the LM from hovering indefinitely.
    DESCENT_SCHEDULE = [
        (100.0, -5.0),  # above 100 m: −5 ft/s
        (50.0, -3.0),  # 50–100 m: −3 ft/s
        (15.0, -2.0),  # 15–50 m: −2 ft/s
        (0.0, -1.0),  # below 15 m: −1 ft/s (final approach)
    ]

    def __init__(self) -> None:
        self.rod_cooldown = 0
        self._prev_v_horiz_vec = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        self.rod_cooldown = 0
        self._prev_v_horiz_vec = np.zeros(3)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[dict, None]:
        """
        Compute the next action from the current observation.

        Follows the same ``predict()`` interface as stable-baselines3
        agents so it can be used interchangeably in evaluation loops.

        Args:
            obs: 12-D observation vector from the environment.
            deterministic: Ignored (always deterministic).

        Returns:
            Tuple of (action_dict, None) matching SB3 convention.
        """
        # Unpack observation
        # obs[0:3] = position [rx, ry, rz]
        # obs[3:6] = velocity [vx, vy, vz]
        # obs[6:9] = attitude [roll, pitch, yaw]
        # obs[9]   = mass
        # obs[10]  = fuel remaining
        # obs[11]  = target descent rate (m/s)

        r_vec = obs[0:3]
        v_vec = obs[3:6]
        attitude = obs[6:9]
        fuel = float(obs[10])
        current_target_dr = float(obs[11])

        # Compute derived quantities
        r_norm = np.linalg.norm(r_vec)
        r_hat = r_vec / max(r_norm, 1.0)

        # Altitude above surface (footpad altitude, consistent with env)
        from apollo_lander.constants import R_MOON

        altitude = r_norm - R_MOON - LM_CG_TO_FOOTPAD

        # Decompose velocity into vertical (radial) and horizontal
        v_vertical = float(np.dot(v_vec, r_hat))

        # Horizontal velocity vector (tangent to surface)
        v_horiz_vec = v_vec - v_vertical * r_hat
        v_horizontal = float(np.linalg.norm(v_horiz_vec))

        # ============================================================
        # 1. RHC: CREW PROCEDURE — Pitch/Roll to null horizontal velocity
        #    The real AGC did NOT automate this. The CDR manually used
        #    the RHC while watching the LPDT cross-pointers (Vh display).
        #    We simulate a skilled CDR with a PD controller.
        # ============================================================
        # Due to the guidance→env mapping:
        #   rhc[0] → target_attitude[0] → env roll → Y-thrust component
        #   rhc[1] → target_attitude[1] → env pitch → X-thrust component
        # So to cancel X-velocity, use rhc[1]; to cancel Y-velocity, use rhc[0].
        rhc = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # PD controller for horizontal velocity: proportional + derivative
        # Derivative term damps oscillations by reacting to velocity changes
        dv_horiz = v_horiz_vec - self._prev_v_horiz_vec
        self._prev_v_horiz_vec = v_horiz_vec.copy()

        kp = 0.04  # proportional gain (gentle — avoid overshoot)
        kd = 0.3  # derivative gain (strong damping)

        # Below 20m, tighten control to prevent marginal failures
        if altitude < 20.0:
            kp = 0.08
            horiz_threshold = MAX_LANDING_HORIZONTAL_VEL * 0.5
        else:
            horiz_threshold = MAX_LANDING_HORIZONTAL_VEL

        if v_horizontal > horiz_threshold:
            # X-component: control via rhc[1] (pitch)
            v_hx = float(v_horiz_vec[0])
            dv_hx = float(dv_horiz[0])
            cmd_x = -(kp * v_hx + kd * dv_hx)
            rhc[1] = float(np.clip(cmd_x, -self.MAX_TILT, self.MAX_TILT))

            # Y-component: control via rhc[0] (roll), opposite sign
            v_hy = float(v_horiz_vec[1])
            dv_hy = float(dv_horiz[1])
            cmd_y = kp * v_hy + kd * dv_hy
            rhc[0] = float(np.clip(cmd_y, -self.MAX_TILT, self.MAX_TILT))
        else:
            # Horizontal nulled — damp attitude back to vertical
            roll_angle = attitude[0]
            pitch_angle = attitude[1]
            if abs(roll_angle) > 0.01:
                rhc[0] = float(np.clip(-roll_angle * 1.0, -0.2, 0.2))
            if abs(pitch_angle) > 0.01:
                rhc[1] = float(np.clip(-pitch_angle * 1.0, -0.2, 0.2))

        # ============================================================
        # 2. ROD: CREW PROCEDURE — Schedule descent rate by altitude
        #    The real CDR clicked the ROD toggle switch to adjust
        #    VDGVERT. The AGC's RODCOMP processed these clicks and
        #    managed the throttle autonomously. The schedule below
        #    mirrors Armstrong's descent profile on Apollo 11.
        # ============================================================
        rod_action = 0

        if self.rod_cooldown > 0:
            self.rod_cooldown -= 1
        else:
            # Determine desired descent rate for current altitude
            desired_rate_fps = -5.0  # default
            for alt_thresh, rate_fps in self.DESCENT_SCHEDULE:
                if altitude >= alt_thresh:
                    desired_rate_fps = rate_fps
                    break

            # Convert to m/s
            desired_rate_ms = desired_rate_fps * 0.3048

            # Compare with current target and click ROD if needed
            rate_error = desired_rate_ms - current_target_dr

            # One ROD click = 0.3048 m/s; only click if error > half a click
            if rate_error > ROD_INCREMENT * 0.5:
                # Need to make descent rate less negative (slower sink)
                rod_action = 1
                self.rod_cooldown = self.ROD_COOLDOWN_TICKS
            elif rate_error < -ROD_INCREMENT * 0.5:
                # Need to make descent rate more negative (faster sink)
                rod_action = 2
                self.rod_cooldown = self.ROD_COOLDOWN_TICKS

        action = {
            "rhc": rhc,
            "rod": rod_action,
        }

        return action, None
