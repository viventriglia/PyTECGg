import datetime
from typing import Any, Literal

import numpy as np

from ..constants import DEFAULT_LEAP_SECONDS_UTC_GPST, GNSS_CONSTANTS
from pytecgg.satellites.kepler.orbits import (
    _is_ephemeris_valid,
    _compute_time_elapsed,
    _compute_anomalies,
    _apply_harmonic_corrections,
    _apply_geo_correction,
)


def _kepler_satellite_coordinates(
    ephem_dict: dict[str, dict[str, Any]],
    sv_id: str,
    gnss_system: Literal["GPS", "GALILEO", "BEIDOU"],
    obs_time: datetime.datetime | None = None,
) -> np.ndarray:
    """
    Compute the Earth-Centered Earth-Fixed (ECEF) coordinates of a GNSS
    satellite using Keplerian orbital model with broadcast ephemeris parameters.

    Parameters
    ----------
    ephem_dict : dict[str, dict[str, Any]]
        Dictionary containing ephemeris data
    sv_id : str
        Satellite identifier (e.g., 'E23')
    gnss_system : Literal["GPS", "GALILEO", "BEIDOU"]
        GNSS constellation
    obs_time : datetime.datetime | None, optional
        Optional observation time (datetime); if None, uses ephemeris timestamp

    Returns
    -------
    np.ndarray
        A 3-element NumPy array with the satellite's ECEF position [X, Y, Z] in
        meters; it returns an empty array if ephemeris data is invalid or incomplete
    """
    if gnss_system not in GNSS_CONSTANTS:
        raise ValueError(
            "Unsupported GNSS system: choose one of ['GPS', 'GALILEO', 'BEIDOU']"
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
    # prepare_ephemeris now stores a list of broadcast records per Keplerian SV.
    # Pick the one nearest to obs_time when called with the full list.
    if isinstance(data, list):
        if not data:
            return np.array([], dtype=float)
        if obs_time is not None:
            data = min(data, key=lambda r: abs(r["datetime"] - obs_time))
        else:
            data = data[len(data) // 2]
    if not _is_ephemeris_valid(data, sv_id, REQUIRED_KEYS):
        return np.array([], dtype=float)

    computation_time = obs_time if obs_time is not None else data["datetime"]

    try:
        # Core computations
        A = data["sqrta"] ** 2
        n0 = np.sqrt(gm / (A**3))
        tk = _compute_time_elapsed(
            computation_time,
            gps_week=data["gps_week"],
            toe=data["toe"],
            gnss_system=gnss_system,
            leap_seconds=data.get("leap_seconds", DEFAULT_LEAP_SECONDS_UTC_GPST),
        )

        # Orbital parameters
        n = n0 + data["deltaN"]
        Mk, Ek, vk = _compute_anomalies(data["e"], data["m0"], n, tk)

        # Harmonic corrections
        Phik = np.fmod(vk + data["omega"], 2 * np.pi)
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
        uk = np.fmod(Phik + delta_uk, 2 * np.pi)
        rk = A * (1 - data["e"] * np.cos(Ek)) + delta_rk
        ik = data["i0"] + data["idot"] * tk + delta_ik

        # BeiDou GEO satellites are propagated in an inertial frame first,
        # then rotated to ECEF by _apply_geo_correction below (BDS ICD B1I
        # §5.2.4.12). Other Keplerian sats fold the ECEF rotation into lamk
        # via the -we*tk term.
        is_beidou_geo = gnss_system == "BEIDOU" and data["i0"] <= 20 * (np.pi / 180)

        # Longitude of ascending node
        if is_beidou_geo:
            lamk = np.fmod(
                data["omega0"] + data["omegaDot"] * tk - we * data["toe"],
                2 * np.pi,
            )
        else:
            lamk = np.fmod(
                data["omega0"] + (data["omegaDot"] - we) * tk - we * data["toe"],
                2 * np.pi,
            )

        # Position in orbital plane
        xDash, yDash = rk * np.cos(uk), rk * np.sin(uk)

        # ECEF coordinates
        Xk = xDash * np.cos(lamk) - yDash * np.cos(ik) * np.sin(lamk)
        Yk = xDash * np.sin(lamk) + yDash * np.cos(ik) * np.cos(lamk)
        Zk = yDash * np.sin(ik)

        if is_beidou_geo:
            Xk, Yk, Zk = _apply_geo_correction(Xk, Yk, Zk, tk, we)

        return np.array([Xk, Yk, Zk], dtype=float)

    except Exception as e:
        raise RuntimeError(
            f"{gnss_system} position computation failed for {sv_id}: {str(e)}"
        )
