from typing import Any
import numpy as np
import math
import datetime

from pytec.satellites.kepler import kepler
from pytec.satellites import GNSS_CONSTANTS, TOL_KEPLER


def _validate_ephemeris(data: dict, required_keys: dict) -> None:
    """Validate the ephemeris data against required keys"""
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")


def _compute_time_elapsed(obs_time: datetime.datetime, toe: float) -> float:
    """Compute the time elapsed since the ephemeris reference epoch (ToE)"""
    if not isinstance(obs_time, datetime.datetime):
        raise TypeError("Invalid datetime format in ephemeris data")
    tk = (
        obs_time - datetime.datetime.fromtimestamp(toe, datetime.timezone.utc)
    ).total_seconds()
    # If time difference is negative (before toe), add 1 week (604800 seconds)
    return tk + 604800 if tk < 0 else tk


def _compute_anomalies(
    ecc: float, M0: float, n: float, tk: float
) -> tuple[float, float, float]:
    """Compute mean, eccentric, and true anomalies"""
    Mk = math.fmod(M0 + n * tk, 2 * math.pi)
    Ek = math.fmod(kepler(ecc, Mk, TOL_KEPLER), 2 * math.pi)
    vk = math.fmod(
        math.atan2(math.sqrt(1 - ecc**2) * math.sin(Ek), math.cos(Ek) - ecc),
        2 * math.pi,
    )
    return Mk, Ek, vk


def _apply_harmonic_corrections(
    Phik: float, cuc: float, cus: float, crc: float, crs: float, cic: float, cis: float
) -> tuple[float, float, float]:
    """Apply harmonic corrections to orbital parameters"""
    sin2P, cos2P = math.sin(2 * Phik), math.cos(2 * Phik)
    delta_uk = cuc * cos2P + cus * sin2P
    delta_rk = crc * cos2P + crs * sin2P
    delta_ik = cic * cos2P + cis * sin2P
    return delta_uk, delta_rk, delta_ik


def _apply_geo_correction(
    Xk: float, Yk: float, Zk: float, tk: float, we: float
) -> tuple[float, float, float]:
    """Transform coordinates for GEO satellites (BeiDou specific)"""
    sin5, cos5 = math.sin(math.radians(-5)), math.cos(math.radians(-5))
    X_GK, Y_GK, Z_GK = Xk, Yk, Zk
    Xk = (
        X_GK * math.cos(we * tk)
        + Y_GK * math.sin(we * tk) * cos5
        + Z_GK * math.sin(we * tk) * sin5
    )
    Yk = (
        -X_GK * math.sin(we * tk)
        + Y_GK * math.cos(we * tk) * cos5
        + Z_GK * math.cos(we * tk) * sin5
    )
    Zk = -Y_GK * sin5 + Z_GK * cos5
    return Xk, Yk, Zk


def get_sat_pos_bds(
    ephem_dict: dict[str, dict[str, Any]], fieldname: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute position of BeiDou satellites using broadcast ephemeris parameters.

    Parameters:
    - ephem_dict: Dictionary containing ephemeris data for multiple satellites
    - fieldname: Specific satellite identifier to process (e.g., 'C01')

    Returns:
    - pos: [3] array containing satellite coordinates [X, Y, Z]
    - aux: [8] array with auxiliary variables from computations
        [tk, Mk, Ek, vk, uk, rk, ik, Omegak]

    Raises:
    - ValueError: If required ephemeris data is missing or invalid
    - KeyError: If the specified fieldname is not found in ephem_dict
    """
    const = GNSS_CONSTANTS["BeiDou"]
    mu = const["gm"]
    we = const["we"]

    # Verify the requested satellite exists
    if fieldname not in ephem_dict:
        raise KeyError(f"Satellite {fieldname} not found in ephemeris data")

    data = ephem_dict[fieldname]

    REQUIRED_KEYS = {
        "toe": "Time of Ephemeris",
        "sqrta": "Square Root of Semi-Major Axis",
        "deltaN": "Mean Motion Difference",
        "m0": "Mean Anomaly at Reference Time",
        "e": "Eccentricity",
        "omega": "Argument of Perigee",
        "cuc": "Latitude Cosine Harmonic Correction",
        "cus": "Latitude Sine Harmonic Correction",
        "crc": "Orbit Radius Cosine Harmonic Correction",
        "crs": "Orbit Radius Sine Harmonic Correction",
        "cic": "Inclination Cosine Harmonic Correction",
        "cis": "Inclination Sine Harmonic Correction",
        "i0": "Inclination at Reference Time",
        "idot": "Rate of Inclination Angle",
        "omega0": "Longitude of Ascending Node",
        "omegaDot": "Rate of Right Ascension",
        "datetime": "Observation datetime",
    }

    _validate_ephemeris(data, REQUIRED_KEYS)

    try:
        # Core computations
        A = data["sqrta"] * data["sqrta"]
        n0 = math.sqrt(mu / (A * A * A))
        tk = _compute_time_elapsed(data["datetime"], data["toe"])

        # Orbital parameters
        n = n0 + data["deltaN"]
        Mk, Ek, vk = _compute_anomalies(data["e"], data["m0"], n, tk)

        # Harmonic corrections
        Phik = math.fmod(vk + data["omega"], 2 * math.pi)
        delta_uk, delta_rk, delta_ik = _apply_harmonic_corrections(
            Phik,
            data["cuc"],
            data["cus"],
            data["crc"],
            data["crs"],
            data["cic"],
            data["cis"],
        )

        # Corrected orbital parameters
        uk = math.fmod(Phik + delta_uk, 2 * math.pi)
        rk = A * (1 - data["e"] * math.cos(Ek)) + delta_rk
        ik = data["i0"] + data["idot"] * tk + delta_ik

        # Position in orbital plane
        xDash, yDash = rk * math.cos(uk), rk * math.sin(uk)

        # Longitude of ascending node
        Omegak = math.fmod(
            data["omega0"] + (data["omegaDot"] - we) * tk - we * data["toe"],
            2 * math.pi,
        )

        # ECEF coordinates
        Xk = xDash * math.cos(Omegak) - yDash * math.cos(ik) * math.sin(Omegak)
        Yk = xDash * math.sin(Omegak) + yDash * math.cos(ik) * math.cos(Omegak)
        Zk = yDash * math.sin(ik)

        # GEO special case
        if data["i0"] <= 20 * (math.pi / 180):
            Xk, Yk, Zk = _apply_geo_correction(Xk, Yk, Zk, tk, we)

        return (
            np.array([Xk, Yk, Zk], dtype=float),
            np.array([tk, Mk, Ek, vk, uk, rk, ik, Omegak], dtype=float),
        )

    except Exception as e:
        raise RuntimeError(f"Error computing position for {fieldname}: {str(e)}")
