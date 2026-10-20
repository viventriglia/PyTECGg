import datetime
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import DEFAULT_LEAP_SECONDS_UTC_GPST, GNSS_CONSTANTS
from pytecgg.satellites.kepler.orbits import _is_ephemeris_valid
from pytecgg.satellites.state_vector.orbits import _glonass_derivatives, _get_gmst


def _state_vector_satellite_coordinates(
    data: dict[str, Any],
    obs_time: datetime.datetime,
    t_res: float | None = None,
    error_estimate: Literal["coarse", "normal", "fine"] = "normal",
) -> np.ndarray:
    """
    Compute the Earth-Centered Earth-Fixed (ECEF) coordinates of a GLONASS
    satellite using state-vector numerical propagation model.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing ephemeris data for the specific satellite and epoch.
    obs_time : datetime.datetime
        OObservation time (datetime)
    t_res : float or None, optional
        Time resolution (seconds) for ODE solver output:
        - if float: the trajectory is sampled at fixed intervals
        - if None (default): ODE solver chooses internal time steps
    error_estimate : {"coarse", "normal", "fine"}, optional
        Error tolerance level for numerical integration:
        - "coarse": ~ 2000 meters precision
        - "normal": ~ 200 meters precision
        - "fine": ~ 20 meters precision
        by default "normal"

    Returns
    -------
    np.ndarray
        A 3-element NumPy array with the satellite's ECEF position [X, Y, Z] in
        meters; it returns an empty array if ephemeris data is invalid or incomplete
    """
    REQUIRED_KEYS = {
        "satPosX": "Satellite Position X (km)",
        "satPosY": "Satellite Position Y (km)",
        "satPosZ": "Satellite Position Z (km)",
        "velX": "Velocity X (km/s)",
        "velY": "Velocity Y (km/s)",
        "velZ": "Velocity Z (km/s)",
        "datetime": "Ephemeris datetime",
    }
    const = GNSS_CONSTANTS["GLONASS"]

    # Validation
    if not _is_ephemeris_valid(data, data.get("sv", "Unknown"), REQUIRED_KEYS):
        return np.array([], dtype=float)

    # Converting km/s → m/s
    re = np.array([data["satPosX"], data["satPosY"], data["satPosZ"]]) * 1000
    ve = np.array([data["velX"], data["velY"], data["velZ"]]) * 1000

    # Converting km/s² → m/s²
    ae = (
        np.array(
            [
                data.get("accelX") or 0.0,
                data.get("accelY") or 0.0,
                data.get("accelZ") or 0.0,
            ]
        )
        * 1000
    )

    eph_time = data["datetime"]
    if obs_time is not None and obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=datetime.timezone.utc)

    # obs_time is GPST, GLONASS ephemeris epochs are UTC: UTC = GPST - leap seconds
    delta_seconds = (obs_time - eph_time).total_seconds() - data.get(
        "leap_seconds", DEFAULT_LEAP_SECONDS_UTC_GPST
    )
    te = eph_time.hour * 3600 + eph_time.minute * 60 + eph_time.second
    ymd = [eph_time.year, eph_time.month, eph_time.day]

    # GMST at eph_time
    theta_ge = _get_gmst(ymd) + const.we * (te % 86400)
    cos_tg, sin_tg = np.cos(theta_ge), np.sin(theta_ge)
    rot_matrix = np.array(
        [
            [cos_tg, -sin_tg, 0],
            [sin_tg, cos_tg, 0],
            [0, 0, 1],
        ]
    )

    # Inertial coordinates at ephemeris time
    ra = rot_matrix @ re
    va = rot_matrix @ ve + const.we * np.array([-ra[1], ra[0], 0])
    initial_state = np.concatenate([ra, va])

    t_span = (0, delta_seconds)

    if t_res is None:
        t_eval = None
    else:
        n_steps = max(2, int(abs(t_span[1] - t_span[0]) / t_res) + 1)
        t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    tolerances = {
        "coarse": (1e-4, 1e-6),
        "normal": (1e-5, 1e-7),
        "fine": (1e-6, 1e-8),
    }
    rtol, atol = tolerances.get(error_estimate, tolerances["normal"])

    sol = solve_ivp(
        fun=lambda t, y: _glonass_derivatives(y, const.gm, const.c20, const.a, ae),
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(
            f"ODE integration failed for {data.get('sv')}: {sol.message}"
        )

    # Rotate back to ECEF after delta_seconds
    theta_gi = theta_ge + const.we * delta_seconds
    cos_ti, sin_ti = np.cos(theta_gi), np.sin(theta_gi)
    rot_matrix_obs = np.array(
        [
            [cos_ti, sin_ti, 0],
            [-sin_ti, cos_ti, 0],
            [0, 0, 1],
        ]
    )

    inertial_pos = sol.y[:3, -1]
    ecef_pos = rot_matrix_obs @ inertial_pos

    return ecef_pos
