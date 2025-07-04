from scipy.integrate import solve_ivp
import numpy as np
from typing import Tuple, Dict, Any
import datetime
import math

from . import GNSS_CONSTANTS

const = GNSS_CONSTANTS["GLONASS"]


def glonass_derivatives(t, state, const, ae):
    """Compute derivatives for GLONASS satellite motion"""
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)

    # Earth gravity + zonal harmonic
    earth_grav = -const.gm * r / r_norm**3
    zonal_term = 1.5 * const.c20 * const.gm * const.a**2 / r_norm**5
    zonal_correction = zonal_term * np.array(
        [
            r[0] * (1 - 5 * (r[2] / r_norm) ** 2),
            r[1] * (1 - 5 * (r[2] / r_norm) ** 2),
            r[2] * (3 - 5 * (r[2] / r_norm) ** 2),
        ]
    )

    # Total acceleration (Earth gravity + zonal harmonic + lunisolar)
    acceleration = earth_grav + zonal_correction + ae

    return np.concatenate([v, acceleration])


def get_gmst(ymd: list) -> float:
    """Compute Greenwich Mean Sidereal Time (simplified version)

    Args:
        ymd: [year, month, day] list

    Returns:
        GMST in radians
    """
    # Note: This is a simplified version. For precise calculations
    # TODO
    dt = datetime.datetime(ymd[0], ymd[1], ymd[2])
    jd = dt.toordinal() + 1721425.5  # Convert to Julian Date
    t = (jd - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t**2
        - t**3 / 38710000
    )
    return math.radians(gmst % 360)


def glonass_satellite_coordinates(
    ephem_dict: Dict[str, Dict[str, Any]],
    sv_id: str,
    delta_seconds: float = 300.0,  # default: 5 minuti dopo l'epoca delle effemeridi
    t_res: float = 60.0,
    rtol: float = 1e-8,
    atol: float = 1e-11,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute GLONASS satellite position from ephemeris data only,
    propagating motion for a given number of seconds from ephemeris time.

    Args:
        ephem_dict: Dictionary containing ephemeris data
        sv_id: Satellite identifier (e.g., 'R01')
        delta_seconds: Time in seconds to integrate from eph_time (can be negative)
        t_res: Time resolution for ODE solver output
        rtol: Relative tolerance for solver
        atol: Absolute tolerance for solver

    Returns:
        pos: [3] array of ECEF coordinates [X, Y, Z] (meters)
        aux: Dictionary with solver information
    """
    data = ephem_dict[sv_id]

    # Converting km/s → m/s
    re = np.array([data["satPosX"], data["satPosY"], data["satPosZ"]]) * 1000
    ve = np.array([data["velX"], data["velY"], data["velZ"]]) * 1000

    # Converting km/s² → m/s²
    ae = (
        np.array(
            [
                0.0 if data["accelX"] is None else data["accelX"],
                0.0 if data["accelY"] is None else data["accelY"],
                0.0 if data["accelZ"] is None else data["accelZ"],
            ]
        )
        * 1000
    )

    eph_time = datetime.datetime.fromtimestamp(
        data["gps_seconds"], datetime.timezone.utc
    )
    te = eph_time.hour * 3600 + eph_time.minute * 60 + eph_time.second
    ymd = [eph_time.year, eph_time.month, eph_time.day]

    # GMST at eph_time
    theta_ge = get_gmst(ymd) + const.we * (te % 86400)
    rot_matrix = np.array(
        [
            [math.cos(theta_ge), -math.sin(theta_ge), 0],
            [math.sin(theta_ge), math.cos(theta_ge), 0],
            [0, 0, 1],
        ]
    )

    # Inertial coordinates at epoch
    ra = rot_matrix @ re
    va = rot_matrix @ ve + const.we * np.array([-ra[1], ra[0], 0])
    initial_state = np.concatenate([ra, va])

    # Integration time span
    t_span = (0, delta_seconds) if delta_seconds >= 0 else (delta_seconds, 0)

    sol = solve_ivp(
        fun=lambda t, y: glonass_derivatives(t, y, const, ae),
        t_span=t_span,
        y0=initial_state,
        t_eval=np.linspace(
            t_span[0], t_span[1], max(2, int(abs(t_span[1] - t_span[0]) / t_res) + 1)
        ),
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    # Rotazione indietro in ECEF dopo delta_seconds
    theta_gi = theta_ge + const.we * delta_seconds
    rot_matrix_obs = np.array(
        [
            [math.cos(theta_gi), math.sin(theta_gi), 0],
            [-math.sin(theta_gi), math.cos(theta_gi), 0],
            [0, 0, 1],
        ]
    )

    inertial_pos = sol.y[:3, -1]
    ecef_pos = rot_matrix_obs @ inertial_pos

    aux = {
        "solution": sol,
        "integration_time": delta_seconds,
        "initial_state": initial_state,
        "rotation_matrix": rot_matrix_obs,
        "eph_time": eph_time,
        "final_time": eph_time + datetime.timedelta(seconds=delta_seconds),
    }

    return ecef_pos, aux
