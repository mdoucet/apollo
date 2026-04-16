"""
ApolloLander-v0 Gymnasium environment.

Simulates the P66 terminal descent phase of the Apollo Lunar Module,
where the pilot controls attitude via RHC and descent rate via ROD switch,
while the AGC manages throttle and attitude hold.

Supports both manual play (human keyboard input) and reinforcement
learning agents (via standard Gymnasium API).
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from apollo_lander.constants import (
    DAP_DT,
    GUIDANCE_DT,
    LM_DRY_MASS,
    LM_ASCENT_MASS,
    LM_FUEL_MASS,
    LM_TOTAL_MASS,
    MAX_LANDING_HORIZONTAL_VEL,
    MAX_LANDING_VERTICAL_VEL,
    P66_INITIAL_ALTITUDE,
    P66_INITIAL_DESCENT_RATE,
    P66_INITIAL_HORIZONTAL_VEL,
    R_MOON,
)
from apollo_lander.guidance import ApolloP66Guidance
from apollo_lander.physics import (
    compute_altitude,
    compute_surface_velocity,
    rk4_step,
)


class ApolloLanderEnv(gym.Env):
    """
    Gymnasium environment for Apollo Lunar Module P66 descent.

    Observation Space (12D Box):
        0-2: Position [rx, ry, rz] relative to landing site (m)
        3-5: Velocity [vx, vy, vz] (m/s)
        6-8: Attitude [roll, pitch, yaw] (rad)
        9:   Mass (kg)
        10:  Fuel remaining (kg)
        11:  Target descent rate (m/s)

    Action Space (Dict):
        rhc: Box(-1, 1, shape=(3,)) — Rotational Hand Controller
        rod: Discrete(3) — Rate of Descent switch (0=none, 1=up, 2=down)

    Reward:
        Dense shaping penalizing horizontal velocity, descent rate deviation,
        and excessive control input, with terminal bonus/penalty for landing/crash.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        dt: float = DAP_DT,
        max_steps: int = 3000,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation bounds (generous to avoid clipping during normal flight)
        obs_high = np.array(
            [
                5e6, 5e6, 5e6,     # position (m)
                2000, 2000, 2000,  # velocity (m/s)
                np.pi, np.pi, np.pi,  # attitude (rad)
                20000,             # mass (kg)
                10000,             # fuel (kg)
                10.0,              # target descent rate (m/s)
            ],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float64,
        )

        self.action_space = spaces.Dict(
            {
                "rhc": spaces.Box(
                    low=-1.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                "rod": spaces.Discrete(3),
            }
        )

        # Reward weights
        self._w_horizontal = 0.1
        self._w_descent = 0.05
        self._w_control = 0.01
        self._w_landing = 100.0
        self._w_crash = -100.0

        # Internal state
        self._guidance = ApolloP66Guidance()
        self._state: np.ndarray = np.zeros(7)
        self._attitude: np.ndarray = np.zeros(3)
        self._step_count = 0
        self._guidance_step_counter = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to P66 start conditions.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        # Position: directly above landing site at P66 altitude
        # In MCI frame, place LM above the surface along +Z axis
        altitude = P66_INITIAL_ALTITUDE
        r_surface = R_MOON + altitude

        # Small random perturbation for training variety
        rng = self.np_random
        pos_noise = rng.uniform(-10.0, 10.0, size=3) if options is None else np.zeros(3)
        position = np.array([0.0, 0.0, r_surface]) + pos_noise

        # Velocity: small horizontal + descent rate
        vel_noise = rng.uniform(-0.5, 0.5, size=3) if options is None else np.zeros(3)
        velocity = np.array(
            [P66_INITIAL_HORIZONTAL_VEL, 0.0, P66_INITIAL_DESCENT_RATE]
        ) + vel_noise

        self._state = np.array([
            position[0], position[1], position[2],
            velocity[0], velocity[1], velocity[2],
            LM_TOTAL_MASS,
        ])

        # Attitude: nearly upright (engine pointing down = +Z body axis up)
        att_noise = rng.uniform(-0.02, 0.02, size=3) if options is None else np.zeros(3)
        self._attitude = np.zeros(3) + att_noise

        self._guidance.reset()
        self._step_count = 0
        self._guidance_step_counter = 0

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one DAP cycle (0.1s).

        Args:
            action: Dict with 'rhc' (3D float array) and 'rod' (int 0-2).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        rhc_input = np.asarray(action["rhc"], dtype=np.float64)
        rod_action = int(action["rod"])

        # Compute thrust via P66 guidance
        thrust_vector = self._guidance.process_controls(
            self._state, self._attitude, rhc_input, rod_action, self.dt
        )

        # Advance physics
        self._state = rk4_step(self._state, thrust_vector, self.dt)

        # Update attitude toward target (simplified: direct tracking)
        # In a full 6-DOF sim, RCS torques would rotate the craft
        attitude_rate = 0.3  # convergence rate per step
        self._attitude = (
            self._attitude
            + attitude_rate * (self._guidance.target_attitude - self._attitude)
        )

        self._step_count += 1
        self._guidance_step_counter += 1

        # Check termination
        altitude = compute_altitude(self._state)
        v_vertical, v_horizontal = compute_surface_velocity(self._state)
        fuel_remaining = self._state[6] - (LM_DRY_MASS + LM_ASCENT_MASS)

        terminated = False
        truncated = False
        landing_success = False
        termination_reason = ""

        if altitude <= 0:
            terminated = True
            if (
                abs(v_vertical) <= MAX_LANDING_VERTICAL_VEL
                and v_horizontal <= MAX_LANDING_HORIZONTAL_VEL
            ):
                landing_success = True
                termination_reason = "landing"
            else:
                termination_reason = "crash"

        if fuel_remaining <= 0:
            terminated = True
            if not termination_reason:
                termination_reason = "fuel"

        if self._step_count >= self.max_steps:
            truncated = True
            if not termination_reason:
                termination_reason = "timeout"

        # Compute reward
        reward = self._compute_reward(
            v_vertical, v_horizontal, rhc_input, rod_action,
            terminated, landing_success,
        )

        info = self._get_info()
        info["landing_success"] = landing_success
        info["termination_reason"] = termination_reason

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(
        self,
        v_vertical: float,
        v_horizontal: float,
        rhc_input: np.ndarray,
        rod_action: int,
        terminated: bool,
        landing_success: bool,
    ) -> float:
        """Compute dense reward with terminal bonus/penalty."""
        # Penalize horizontal velocity (should be near zero for landing)
        r_horizontal = -self._w_horizontal * abs(v_horizontal)

        # Penalize deviation from target descent rate
        descent_error = abs(
            self._guidance.target_descent_rate - v_vertical
        )
        r_descent = -self._w_descent * descent_error

        # Penalize excessive control input (encourage smooth flying)
        control_magnitude = float(np.linalg.norm(rhc_input)) + (
            1.0 if rod_action != 0 else 0.0
        )
        r_control = -self._w_control * control_magnitude

        reward = r_horizontal + r_descent + r_control

        # Terminal reward
        if terminated:
            if landing_success:
                reward += self._w_landing
            else:
                reward += self._w_crash

        return reward

    def _get_obs(self) -> np.ndarray:
        """Build the 12D observation vector."""
        fuel_remaining = self._state[6] - (LM_DRY_MASS + LM_ASCENT_MASS)
        return np.array(
            [
                self._state[0],  # rx
                self._state[1],  # ry
                self._state[2],  # rz
                self._state[3],  # vx
                self._state[4],  # vy
                self._state[5],  # vz
                self._attitude[0],  # roll
                self._attitude[1],  # pitch
                self._attitude[2],  # yaw
                self._state[6],  # mass
                max(fuel_remaining, 0.0),  # fuel remaining
                self._guidance.target_descent_rate,  # target descent rate
            ],
            dtype=np.float64,
        )

    def _get_info(self) -> dict[str, Any]:
        """Build the info dictionary."""
        altitude = compute_altitude(self._state)
        v_vertical, v_horizontal = compute_surface_velocity(self._state)
        fuel_remaining = self._state[6] - (LM_DRY_MASS + LM_ASCENT_MASS)

        return {
            "altitude": altitude,
            "vertical_velocity": v_vertical,
            "horizontal_velocity": v_horizontal,
            "fuel_remaining": max(fuel_remaining, 0.0),
            "mass": self._state[6],
            "target_descent_rate": self._guidance.target_descent_rate,
            "step": self._step_count,
        }
