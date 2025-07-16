from typing import Any, Literal
import math
import warnings
import datetime

import numpy as np


from pytecal.satellites.kepler import kepler
from pytecal.satellites import GNSS_CONSTANTS, TOL_KEPLER


def _is_ephemeris_valid(data: dict, required_keys: dict) -> bool:
    """Check ephemeris validaty against required keys"""
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        warnings.warn(f"Missing ephemeris parameters: {missing}", RuntimeWarning)
        return False
    return True


def _compute_time_elapsed(obs_time: datetime.datetime, toe: float) -> float:
    """Compute the time elapsed since the ephemeris reference epoch (ToE)"""
    if not isinstance(obs_time, datetime.datetime):
        raise TypeError("Invalid datetime format in ephemeris data")

    toe_dt = datetime.datetime.fromtimestamp(toe, datetime.timezone.utc)

    # Assicura che entrambi siano aware o naive
    if obs_time.tzinfo is not None and toe_dt.tzinfo is None:
        toe_dt = toe_dt.replace(tzinfo=datetime.timezone.utc)
    elif obs_time.tzinfo is None and toe_dt.tzinfo is not None:
        obs_time = obs_time.replace(tzinfo=datetime.timezone.utc)

    tk = (obs_time - toe_dt).total_seconds()
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


def satellite_coordinates(
    ephem_dict: dict[str, dict[str, Any]],
    sv_id: str,
    gnss_system: Literal["GPS", "Galileo", "QZSS", "BeiDou"],
    obs_time: datetime.datetime | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GNSS satellite position in ECEF coordinates using broadcast ephemeris.

    Parameters:
    - ephem_dict: Dictionary containing ephemeris data
    - sv_id: Satellite identifier (e.g., 'E23')
    - gnss_system: GNSS constellation ('GPS', 'Galileo', 'QZSS' or 'BeiDou')
    - obs_time: Optional observation time (datetime). If None, uses ephemeris timestamp

    Returns:
    - pos: [3] array of ECEF coordinates [X, Y, Z] (meters)
    - aux: [8] array of auxiliary variables [tk, Mk, Ek, vk, uk, rk, ik, lamk]
    """
    if gnss_system not in GNSS_CONSTANTS:
        raise ValueError(
            "Unsupported GNSS system: choose one of ['GPS', 'Galileo', 'QZSS', 'BeiDou']"
        )

    const = GNSS_CONSTANTS[gnss_system]
    gm, we = const.gm, const.we

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

    if sv_id not in ephem_dict:
        raise KeyError(f"Satellite {sv_id} not found in ephemeris data")

    data = ephem_dict[sv_id]
    if not _is_ephemeris_valid(data, REQUIRED_KEYS):
        return np.array([]), np.array([])

    computation_time = obs_time if obs_time is not None else data["datetime"]

    try:
        # Core computations
        A = data["sqrta"] ** 2
        n0 = math.sqrt(gm / (A**3))
        tk = _compute_time_elapsed(computation_time, data["toe"])

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

        # Longitude of ascending node
        lamk = math.fmod(
            data["omega0"] + (data["omegaDot"] - we) * tk - we * data["toe"],
            2 * math.pi,
        )

        # Position in orbital plane
        xDash, yDash = rk * math.cos(uk), rk * math.sin(uk)

        # ECEF coordinates
        Xk = xDash * math.cos(lamk) - yDash * math.cos(ik) * math.sin(lamk)
        Yk = xDash * math.sin(lamk) + yDash * math.cos(ik) * math.cos(lamk)
        Zk = yDash * math.sin(ik)

        if gnss_system == "BeiDou":
            # GEO orbits correction
            if data["i0"] <= 20 * (math.pi / 180):
                Xk, Yk, Zk = _apply_geo_correction(Xk, Yk, Zk, tk, we)

        pos = np.array([Xk, Yk, Zk], dtype=float)
        aux = np.array([tk, Mk, Ek, vk, uk, rk, ik, lamk], dtype=float)

        return pos, aux

    except Exception as e:
        raise RuntimeError(
            f"{gnss_system} position computation failed for {sv_id}: {str(e)}"
        )
