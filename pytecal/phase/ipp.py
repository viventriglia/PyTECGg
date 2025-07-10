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

    # Center of Earth
    xc, yc, zc = 0.0, 0.0, 0.0

    # Two points defining the line from receiver to satellite
    x1, y1, z1 = xA, yA, zA
    x2, y2, z2 = xB, yB, zB

    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    b = 2 * ((x2 - x1) * (x1 - xc) + (y2 - y1) * (y1 - yc) + (z2 - z1) * (z1 - zc))
    c = (
        xc**2
        + yc**2
        + zc**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (xc * x1 + yc * y1 + zc * z1)
        - r**2
    )
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No real solution
        x_ipp, y_ipp, z_ipp = 0.0, 0.0, 0.0
    else:
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # We take the solution between 0 and 1 (point between receiver and satellite)
        t = min(max(t1, t2), 1.0)

        x_ipp = x1 + (x2 - x1) * t
        y_ipp = y1 + (y2 - y1) * t
        z_ipp = z1 + (z2 - z1) * t

    # Convert IPP ECEF coordinates to geodetic (latitude, longitude, altitude)
    lat, lon, alt = ecef2geodetic(x_ipp, y_ipp, z_ipp)

    # Calculate azimuth and elevation angles
    # First convert receiver position to geodetic
    rec_lat, rec_lon, rec_alt = ecef2geodetic(xA, yA, zA)

    # Calculate azimuth and elevation
    azi, ele, _ = ecef2aer(
        sat_ecef_coords[0],
        sat_ecef_coords[1],
        sat_ecef_coords[2],
        rec_lat,
        rec_lon,
        rec_alt,
        deg=True,
    )

    return lat, lon, azi, ele
