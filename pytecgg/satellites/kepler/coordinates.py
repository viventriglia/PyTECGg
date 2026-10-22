from typing import Literal, Mapping

import numpy as np

from ..constants import GNSS_CONSTANTS
from pytecgg.satellites.kepler.orbits import (
    _compute_time_elapsed,
    _compute_anomalies,
    _apply_harmonic_corrections,
    _apply_geo_correction,
)


def _kepler_satellite_coordinates(
    eph: Mapping[str, np.ndarray],
    obs_gpst_s: np.ndarray,
    gnss_system: Literal["GPS", "GALILEO", "BEIDOU"],
) -> np.ndarray:
    """
    Compute the Earth-Centered Earth-Fixed (ECEF) coordinates of GNSS
    satellites using Keplerian orbital model with broadcast ephemeris parameters.

    Parameters
    ----------
    eph : Mapping[str, np.ndarray]
        Broadcast ephemeris parameters, one array per field (gps_week, toe,
        sqrta, deltaN, m0, e, omega, cuc, cus, crc, crs, cic, cis, i0, idot,
        omega0, omegaDot), with one element per observation.
    obs_gpst_s : np.ndarray
        Observation times, in seconds since the GPS epoch (GPST)
    gnss_system : Literal["GPS", "GALILEO", "BEIDOU"]
        GNSS constellation

    Returns
    -------
    np.ndarray
        Array of shape (n, 3) with the satellites' ECEF positions [X, Y, Z] in meters
    """
    if gnss_system not in GNSS_CONSTANTS:
        raise ValueError(
            "Unsupported GNSS system: choose one of ['GPS', 'GALILEO', 'BEIDOU']"
        )

    const = GNSS_CONSTANTS[gnss_system]
    gm, we = const.gm, const.we

    # Core computations
    A = eph["sqrta"] ** 2
    n0 = np.sqrt(gm / (A**3))
    tk = _compute_time_elapsed(
        obs_gpst_s,
        gps_week=eph["gps_week"],
        toe=eph["toe"],
        gnss_system=gnss_system,
    )

    # Orbital parameters
    n = n0 + eph["deltaN"]
    Mk, Ek, vk = _compute_anomalies(eph["e"], eph["m0"], n, tk)

    # Harmonic corrections
    Phik = np.fmod(vk + eph["omega"], 2 * np.pi)
    delta_uk, delta_rk, delta_ik = _apply_harmonic_corrections(
        Phik,
        eph["cuc"],
        eph["cus"],
        eph["crc"],
        eph["crs"],
        eph["cic"],
        eph["cis"],
    )

    # Corrected orbital parameters
    uk = np.fmod(Phik + delta_uk, 2 * np.pi)
    rk = A * (1 - eph["e"] * np.cos(Ek)) + delta_rk
    ik = eph["i0"] + eph["idot"] * tk + delta_ik

    # BeiDou GEO satellites are propagated in an inertial frame first,
    # then rotated to ECEF by _apply_geo_correction below (BDS ICD B1I
    # §5.2.4.12). Other Keplerian sats fold the ECEF rotation into lamk
    # via the -we*tk term.
    is_beidou_geo = (gnss_system == "BEIDOU") & (eph["i0"] <= 20 * (np.pi / 180))

    # Longitude of ascending node
    lamk = np.fmod(
        eph["omega0"]
        + (eph["omegaDot"] - np.where(is_beidou_geo, 0.0, we)) * tk
        - we * eph["toe"],
        2 * np.pi,
    )

    # Position in orbital plane
    xDash, yDash = rk * np.cos(uk), rk * np.sin(uk)

    # ECEF coordinates
    Xk = xDash * np.cos(lamk) - yDash * np.cos(ik) * np.sin(lamk)
    Yk = xDash * np.sin(lamk) + yDash * np.cos(ik) * np.cos(lamk)
    Zk = yDash * np.sin(ik)

    Xg, Yg, Zg = _apply_geo_correction(Xk, Yk, Zk, tk, we)
    return np.column_stack(
        [
            np.where(is_beidou_geo, Xg, Xk),
            np.where(is_beidou_geo, Yg, Yk),
            np.where(is_beidou_geo, Zg, Zk),
        ]
    )
