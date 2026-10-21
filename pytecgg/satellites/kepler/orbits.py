import datetime
import warnings
from typing import Optional

import numpy as np

from ..constants import BDT_OFFSET_FROM_GPST, DEFAULT_LEAP_SECONDS_UTC_GPST, TOL_KEPLER


def _is_ephemeris_valid(data: dict, sv_id: str, required_keys: dict) -> bool:
    """Check ephemeris validaty against required keys"""
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        warnings.warn(
            f"Satellite {sv_id} | Missing ephemeris parameters: {missing}",
            RuntimeWarning,
        )
        return False
    return True


def _gps_to_datetime(time_week, time_s, leap_seconds=0):
    gps_epoch = datetime.datetime(1980, 1, 6, tzinfo=datetime.timezone.utc)
    return gps_epoch + datetime.timedelta(
        weeks=time_week, seconds=time_s - leap_seconds
    )


def _utc_to_system_time(
    obs_time_utc: datetime.datetime,
    gnss_system: Optional[str],
    leap_seconds: int,
) -> datetime.datetime:
    """Shift a UTC observation datetime into the constellation's time system.

    GPS / Galileo / QZSS reference broadcast ephemerides to GPST, BeiDou to BDT,
    GLONASS to UTC. The ICDs fix all offsets:

    - GPST = UTC + leap_seconds (currently 18 s, IERS Bulletin C).
    - GST  = GPST (Galileo ICD).
    - QZSST = GPST (QZSS ICD).
    - BDT  = GPST - 14 s (BDS ICD, BDT epoch 2006-01-01).
    - GLONASS uses UTC directly (no shift).
    """
    if gnss_system in ("GPS", "GALILEO", "QZSS"):
        return obs_time_utc + datetime.timedelta(seconds=leap_seconds)
    if gnss_system == "BEIDOU":
        return obs_time_utc + datetime.timedelta(
            seconds=leap_seconds - BDT_OFFSET_FROM_GPST
        )
    return obs_time_utc


def _compute_time_elapsed(
    obs_time: datetime.datetime,
    gps_week: int,
    toe: int,
    gnss_system: Optional[str] = None,
    leap_seconds: int = DEFAULT_LEAP_SECONDS_UTC_GPST,
) -> float:
    """Compute the time elapsed since the ephemeris reference epoch (ToE).

    ``obs_time`` is interpreted as UTC and shifted into the constellation's
    own time system before subtracting the toe. ``gnss_system`` selects which
    offset applies (GPST / GST / BDT / UTC for GLONASS).
    """
    if not isinstance(obs_time, datetime.datetime):
        raise TypeError("Invalid datetime format in ephemeris data")

    # Ensure obs_time is tz-aware UTC before applying the leap-second shift.
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=datetime.timezone.utc)

    obs_time_in_sys = _utc_to_system_time(obs_time, gnss_system, leap_seconds)

    toe_dt = _gps_to_datetime(
        time_week=gps_week,
        time_s=toe,
    )
    if toe_dt.tzinfo is None:
        toe_dt = toe_dt.replace(tzinfo=datetime.timezone.utc)

    return (obs_time_in_sys - toe_dt).total_seconds()


def _kepler(e: float, mk: float, tol: float) -> float:
    """
    Computes the eccentric anomaly (ek) from the mean anomaly (mk)
    using Kepler's equation and a fixed-point iteration method

    Parameters:
    - e: numerical eccentricity of the orbit (dimensionless)
    - mk: mean anomaly in radians
    - tol: convergence tolerance in arcseconds (")

    Returns:
    - ek: eccentric anomaly in radians
    """
    tol_rad = tol * (np.pi / 648_000)

    e_prev = mk
    e_curr = mk + e * np.sin(e_prev)

    while abs(e_curr - e_prev) > tol_rad:
        e_prev, e_curr = e_curr, mk + e * np.sin(e_curr)

    return e_curr


def _compute_anomalies(
    ecc: float, M0: float, n: float, tk: float
) -> tuple[float, float, float]:
    """Compute mean, eccentric, and true anomalies"""
    Mk = np.fmod(M0 + n * tk, 2 * np.pi)
    Ek = np.fmod(_kepler(ecc, Mk, TOL_KEPLER), 2 * np.pi)
    vk = np.fmod(
        np.atan2(np.sqrt(1 - ecc**2) * np.sin(Ek), np.cos(Ek) - ecc),
        2 * np.pi,
    )
    return Mk, Ek, vk


def _apply_harmonic_corrections(
    Phik: float, cuc: float, cus: float, crc: float, crs: float, cic: float, cis: float
) -> tuple[float, float, float]:
    """Apply harmonic corrections to orbital parameters"""
    sin2P, cos2P = np.sin(2 * Phik), np.cos(2 * Phik)
    delta_uk = cuc * cos2P + cus * sin2P
    delta_rk = crc * cos2P + crs * sin2P
    delta_ik = cic * cos2P + cis * sin2P
    return delta_uk, delta_rk, delta_ik


def _apply_geo_correction(
    Xk: float, Yk: float, Zk: float, tk: float, we: float
) -> tuple[float, float, float]:
    """Transform coordinates for GEO satellites (BeiDou specific)"""
    sin5, cos5 = np.sin(np.radians(-5)), np.cos(np.radians(-5))
    X_GK, Y_GK, Z_GK = Xk, Yk, Zk
    Xk = (
        X_GK * np.cos(we * tk)
        + Y_GK * np.sin(we * tk) * cos5
        + Z_GK * np.sin(we * tk) * sin5
    )
    Yk = (
        -X_GK * np.sin(we * tk)
        + Y_GK * np.cos(we * tk) * cos5
        + Z_GK * np.cos(we * tk) * sin5
    )
    Zk = -Y_GK * sin5 + Z_GK * cos5
    return Xk, Yk, Zk
