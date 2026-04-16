"""
Gymnasium wrapper that flattens the Dict action space for RL agents.

Stable Baselines3 works most naturally with flat Box action spaces.
This wrapper converts the Dict(rhc=Box(3), rod=Discrete(3)) action
space into a single Box(4) where the 4th dimension encodes the
ROD switch as a continuous value mapped to discrete choices.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlatActionWrapper(gym.ActionWrapper):
    """
    Flatten Dict action space to Box(4) for RL compatibility.

    Action mapping:
        [0:3] → RHC (pitch, roll, yaw) in [-1, 1]
        [3]   → ROD: <-0.33 → click down, >0.33 → click up, else → none
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> dict[str, Any]:
        """Convert flat array to Dict action."""
        rhc = np.clip(action[0:3], -1.0, 1.0).astype(np.float32)

        rod_val = float(action[3])
        if rod_val > 0.33:
            rod = 1  # ROD up
        elif rod_val < -0.33:
            rod = 2  # ROD down
        else:
            rod = 0  # no click

        return {"rhc": rhc, "rod": rod}
