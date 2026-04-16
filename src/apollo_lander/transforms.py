"""
Coordinate frame transformations for the lunar descent simulation.

Handles rotation between the spacecraft body frame and the
Moon-Centered Inertial (MCI) frame using scipy's Rotation class
(quaternion-based internally to avoid gimbal lock).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def body_to_world(
    thrust_magnitude: float,
    attitude: np.ndarray,
) -> np.ndarray:
    """
    Rotate the spacecraft's local thrust vector into the inertial frame.

    The descent engine fires along the spacecraft's local +Z axis.
    This function applies the current attitude rotation to produce
    the thrust vector in MCI coordinates.

    Args:
        thrust_magnitude: Engine thrust in Newtons (scalar).
        attitude: 3D array [roll, pitch, yaw] in radians.

    Returns:
        3D thrust vector [Tx, Ty, Tz] in the inertial frame (N).
    """
    # Thrust acts along local +Z body axis (engine points "down",
    # thrust pushes "up" through the spacecraft)
    t_body = np.array([0.0, 0.0, thrust_magnitude])

    # Build rotation: intrinsic ZYX (Yaw, Pitch, Roll) sequence
    r = Rotation.from_euler(
        "ZYX",
        [attitude[2], attitude[1], attitude[0]],
        degrees=False,
    )

    return r.apply(t_body)


def world_to_body(
    vector_world: np.ndarray,
    attitude: np.ndarray,
) -> np.ndarray:
    """
    Rotate a vector from the inertial frame into the body frame.

    Args:
        vector_world: 3D vector in MCI coordinates.
        attitude: 3D array [roll, pitch, yaw] in radians.

    Returns:
        3D vector in body-frame coordinates.
    """
    r = Rotation.from_euler(
        "ZYX",
        [attitude[2], attitude[1], attitude[0]],
        degrees=False,
    )

    return r.inv().apply(vector_world)
