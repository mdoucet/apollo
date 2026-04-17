"""
Lunar physics engine with RK4 integration.

Implements the equations of motion for the Lunar Module descent
in a Moon-Centered Inertial (MCI) coordinate frame, using a
4th-order Runge-Kutta integrator.

State vector (7D):
    [rx, ry, rz, vx, vy, vz, mass]

Where position/velocity are in MCI frame and mass is the
instantaneous spacecraft mass including remaining fuel.
"""

import numpy as np

from apollo_lander.constants import DPS_ISP, G0, LM_CG_TO_FOOTPAD, MU_MOON, R_MOON


def lunar_equations_of_motion(
    t: float,
    state: np.ndarray,
    thrust_vector: np.ndarray,
) -> np.ndarray:
    """
    Compute the derivative of the state vector.

    Args:
        t: Current time (s). Unused in this time-invariant formulation
            but kept for standard ODE solver compatibility.
        state: 7D array [rx, ry, rz, vx, vy, vz, mass].
        thrust_vector: 3D array [Tx, Ty, Tz] in Newtons (inertial frame).

    Returns:
        7D derivative array [vx, vy, vz, ax, ay, az, dm/dt].
    """
    r_vec = state[0:3]
    v_vec = state[3:6]
    mass = state[6]

    # Kinematics: dr/dt = v
    dr_dt = v_vec

    # Gravity: a = -mu / |r|^3 * r
    r_norm = np.linalg.norm(r_vec)
    if r_norm < R_MOON:
        return np.zeros(7)

    a_gravity = -(MU_MOON / (r_norm**3)) * r_vec

    # Thrust acceleration: a = F / m
    a_thrust = thrust_vector / mass

    dv_dt = a_gravity + a_thrust

    # Mass depletion: dm/dt = -|F| / (Isp * g0)
    thrust_mag = np.linalg.norm(thrust_vector)
    dm_dt = -(thrust_mag / (DPS_ISP * G0))

    return np.concatenate((dr_dt, dv_dt, [dm_dt]))


def rk4_step(
    state: np.ndarray,
    thrust_vector: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Advance the state by one time step using 4th-order Runge-Kutta.

    Args:
        state: Current 7D state vector.
        thrust_vector: Current 3D commanded thrust in Newtons (inertial frame).
        dt: Time step in seconds.

    Returns:
        Updated 7D state vector.
    """
    k1 = lunar_equations_of_motion(0, state, thrust_vector)
    k2 = lunar_equations_of_motion(0, state + k1 * (dt / 2), thrust_vector)
    k3 = lunar_equations_of_motion(0, state + k2 * (dt / 2), thrust_vector)
    k4 = lunar_equations_of_motion(0, state + k3 * dt, thrust_vector)

    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Prevent mass from going negative (fuel exhaustion)
    if new_state[6] < 0:
        new_state[6] = 0.001

    return new_state


def compute_altitude(state: np.ndarray) -> float:
    """
    Compute altitude of the landing gear footpads above the lunar surface.

    The physics state vector tracks the LM center of gravity (CG).
    This function subtracts the CG-to-footpad offset so that
    altitude = 0 corresponds to the footpads touching the surface.

    Args:
        state: 7D state vector.

    Returns:
        Footpad altitude in meters (0 = footpads on the surface).
    """
    r_norm = np.linalg.norm(state[0:3])
    return float(r_norm - R_MOON - LM_CG_TO_FOOTPAD)


def compute_surface_velocity(state: np.ndarray) -> tuple[float, float]:
    """
    Decompose velocity into vertical and horizontal components
    relative to the local surface.

    Args:
        state: 7D state vector.

    Returns:
        Tuple of (vertical_velocity, horizontal_velocity) in m/s.
        Vertical is positive upward.
        Horizontal is signed: positive = moving in the +X direction
        (rightward in the 2D side-view), negative = leftward.
    """
    r_vec = state[0:3]
    v_vec = state[3:6]

    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-10:
        return 0.0, 0.0

    # Unit radial vector (outward from Moon center)
    r_hat = r_vec / r_norm

    # Vertical velocity: projection of v onto radial direction
    v_vertical = float(np.dot(v_vec, r_hat))

    # Horizontal velocity: signed component in the local horizontal plane.
    # "East" direction = perpendicular to r_hat in the X-Z plane,
    # pointing in the +X direction (rightward in the side-view).
    # east_hat = cross([0,1,0], r_hat) → [rz, 0, -rx] / norm
    v_tangential = v_vec - v_vertical * r_hat
    east_hat = np.array([r_hat[2], 0.0, -r_hat[0]])
    east_norm = np.linalg.norm(east_hat)
    if east_norm < 1e-10:
        return v_vertical, 0.0
    east_hat = east_hat / east_norm

    v_horizontal = float(np.dot(v_tangential, east_hat))

    return v_vertical, v_horizontal
