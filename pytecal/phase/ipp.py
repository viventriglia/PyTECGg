import numpy as np
from typing import Tuple, Optional
from pymap3d import ecef2geodetic, ecef2aer

from . import RE


def calculate_ipp(
    rec_ecef_coords: Tuple[float, float, float],
    sat_ecef_coords: Tuple[float, float, float],
    h_ipp: float,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Calculate the Ionospheric Pierce Point (IPP) location given the receiver and satellite ECEF positions
    and the ionospheric shell height.

    Parameters:
    - rec_ecef_coords: Receiver ECEF coordinates (x, y, z) in meters
    - sat_ecef_coords: Satellite ECEF coordinates (x, y, z) in meters
    - h_ipp: Mean height of the ionosphere shell in meters

    Returns:
    - lat: Latitude of IPP in degrees (None, if calculation fails)
    - lon: Longitude of IPP in degrees (None, if calculation fails)
    - azi: Azimuth angle from receiver to satellite in degrees (None, if calculation fails)
    - ele: Elevation angle from receiver to satellite in degrees (None, if calculation fails)
    """

    # Check for NaN values in satellite position
    if any(np.isnan(coord) for coord in sat_ecef_coords):
        return None, None, None, None

    # Ionospheric shell radius in meters
    r = h_ipp + RE

    xA, yA, zA = rec_ecef_coords
    xB, yB, zB = sat_ecef_coords

    # Coefficients for quadratic equation
    a = (xB - xA) ** 2 + (yB - yA) ** 2 + (zB - zA) ** 2
    b = 2 * ((xB - xA) * xA + (yB - yA) * yA + (zB - zA) * zA)
    c = xA**2 + yA**2 + zA**2 - r**2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No real solution
        return None, None, None, None

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # We take the solution between 0 and 1 (point between receiver and satellite)
    valid_ts = [t for t in (t1, t2) if 0 <= t <= 1]
    if not valid_ts:
        return None, None, None, None
    t = min(valid_ts)

    x_ipp = xA + (xB - xA) * t
    y_ipp = yA + (yB - yA) * t
    z_ipp = zA + (zB - zA) * t

    # Convert IPP ECEF coordinates to geodetic (latitude, longitude, altitude)
    lat, lon, _ = ecef2geodetic(x_ipp, y_ipp, z_ipp)

    # Calculate azimuth and elevation angles
    rec_lat, rec_lon, rec_alt = ecef2geodetic(xA, yA, zA)
    azi, ele, _ = ecef2aer(
        xB,
        yB,
        zB,
        rec_lat,
        rec_lon,
        rec_alt,
        deg=True,
    )

    return lat, lon, azi, ele
