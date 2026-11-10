import warnings
from typing import Optional

import numpy as np

from ..constants import BDT_OFFSET_FROM_GPST, TOL_KEPLER


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


def _compute_time_elapsed(
    obs_gpst_s: np.ndarray,
    gps_week: np.ndarray,
    toe: np.ndarray,
    gnss_system: Optional[str] = None,
) -> np.ndarray:
    """Compute the time elapsed since the ephemeris reference epoch (ToE), in seconds.

    ``obs_gpst_s`` is the observation time in seconds since the GPS epoch, on
    the GPST grid of ``read_rinex_obs``. ``gps_week`` is the week of the
    record's datetime, ``toe`` its seconds of week in the constellation's own
    time system:

    - GPST for GPS, GST = GPST for Galileo, QZSST = GPST for QZSS (no shift).
    - BDT = GPST - 14 s for BeiDou (BDS ICD, BDT epoch 2006-01-01).
    """
    offset = BDT_OFFSET_FROM_GPST if gnss_system == "BEIDOU" else 0
    return obs_gpst_s - offset - (gps_week * 604_800 + toe)


def _kepler(e: np.ndarray, mk: np.ndarray, tol: float) -> np.ndarray:
    """
    Computes the eccentric anomaly (ek) from the mean anomaly (mk)
    using Kepler's equation and a fixed-point iteration method

    Parameters:
    - e: numerical eccentricity of the orbit (dimensionless), scalar or array
    - mk: mean anomaly in radians, scalar or array
    - tol: convergence tolerance in arcseconds ("), met by every element

    Returns:
    - ek: eccentric anomaly in radians
    """
    tol_rad = tol * (np.pi / 648_000)

    e_prev = mk
    e_curr = mk + e * np.sin(e_prev)

    while np.max(np.abs(e_curr - e_prev), initial=0.0) > tol_rad:
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
