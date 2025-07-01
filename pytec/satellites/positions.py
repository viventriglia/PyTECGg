from typing import Any
import numpy as np
import math
import datetime
from warnings import warn

from pytec.satellites.kepler import kepler
from pytec.satellites import GNSS_CONSTANTS, TOL_KEPLER


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
    # Constants for BeiDou
    const = GNSS_CONSTANTS["BeiDou"]
    mu = const["gm"]

    # Verify the requested satellite exists
    if fieldname not in ephem_dict:
        raise KeyError(f"Satellite {fieldname} not found in ephemeris data")

    data = ephem_dict[fieldname]

    # Required parameters with descriptive names
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

    # Check for missing parameters
    missing_keys = [k for k in REQUIRED_KEYS if k not in data or data[k] is None]
    if missing_keys:
        raise ValueError(
            f"Missing required ephemeris parameters for {fieldname}: {missing_keys}"
        )

    try:
        # Extract and validate parameters
        A = data["sqrta"] * data["sqrta"]
        n0 = math.sqrt(mu / (A * A * A))

        # Time calculations
        if not isinstance(data["datetime"], datetime.datetime):
            raise TypeError("Invalid datetime format in ephemeris data")

        # Time since ephemeris reference epoch (in seconds)
        tk = (
            data["datetime"]
            - datetime.datetime.fromtimestamp(data["toe"], datetime.timezone.utc)
        ).total_seconds()

        # If time difference is negative (before toe), add 1 week (604800 seconds)
        if tk < 0:
            tk += 604800
            warn(
                f"Time {data['datetime']} is before TOE for {fieldname}, added 1 week to tk"
            )

        # Mean motion and anomaly
        n = n0 + data["deltaN"]
        Mk = math.fmod(data["m0"] + n * tk, 2 * math.pi)

        # Solve Kepler's equation for eccentric anomaly
        Ek = math.fmod(kepler(data["e"], Mk, TOL_KEPLER), 2 * math.pi)

        # Verify Kepler solution
        if abs(Ek - data["e"] * math.sin(Ek) - Mk) > TOL_KEPLER * (math.pi / 648000):
            raise ValueError("Kepler equation solution did not converge")

        # True anomaly
        vk = math.fmod(
            math.atan2(
                math.sqrt(1.0 - data["e"] * data["e"]) * math.sin(Ek),
                math.cos(Ek) - data["e"],
            ),
            2 * math.pi,
        )

        # Radius and argument of latitude
        r0k = A * (1 - data["e"] * math.cos(Ek))
        Phik = math.fmod(vk + data["omega"], 2 * math.pi)

        # Harmonic corrections
        delta_uk = data["cuc"] * math.cos(2 * Phik) + data["cus"] * math.sin(2 * Phik)
        delta_rk = data["crc"] * math.cos(2 * Phik) + data["crs"] * math.sin(2 * Phik)
        delta_ik = data["cic"] * math.cos(2 * Phik) + data["cis"] * math.sin(2 * Phik)

        # Corrected orbital parameters
        uk = math.fmod(Phik + delta_uk, 2 * math.pi)
        rk = r0k + delta_rk
        ik = data["i0"] + data["idot"] * tk + delta_ik

        # Position in orbital plane
        xDash = rk * math.cos(uk)
        yDash = rk * math.sin(uk)

        # Longitude of ascending node
        Omegak = math.fmod(
            data["omega0"]
            + (data["omegaDot"] - const["we"]) * tk
            - const["we"] * data["toe"],
            2 * math.pi,
        )

        # ECEF coordinates (MEO/IGSO case)
        Xk = xDash * math.cos(Omegak) - yDash * math.cos(ik) * math.sin(Omegak)
        Yk = xDash * math.sin(Omegak) + yDash * math.cos(ik) * math.cos(Omegak)
        Zk = yDash * math.sin(ik)

        # Handle GEO case if needed (i0 <= 20°)
        if data["i0"] <= 20 * (math.pi / 180):
            sin5 = math.sin(math.radians(-5))
            cos5 = math.cos(math.radians(-5))
            # GEO transformation
            X_GK, Y_GK, Z_GK = Xk, Yk, Zk
            Xk = (
                X_GK * math.cos(const["we"] * tk)
                + Y_GK * math.sin(const["we"] * tk) * cos5
                + Z_GK * math.sin(const["we"] * tk) * sin5
            )

            Yk = (
                -X_GK * math.sin(const["we"] * tk)
                + Y_GK * math.cos(const["we"] * tk) * cos5
                + Z_GK * math.cos(const["we"] * tk) * sin5
            )

            Zk = -Y_GK * sin5 + Z_GK * cos5

        pos = np.array([Xk, Yk, Zk], dtype=float)
        aux = np.array([tk, Mk, Ek, vk, uk, rk, ik, Omegak], dtype=float)

        return pos, aux

    except Exception as e:
        raise RuntimeError(f"Error computing position for {fieldname}: {str(e)}")
