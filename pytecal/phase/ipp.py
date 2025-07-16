import numpy as np
from typing import Tuple, Optional
from pymap3d import ecef2geodetic, ecef2aer

from . import RE


def calculate_ipp(
    rec_ecef: Tuple[float, float, float],
    sat_ecef_array: np.ndarray,
    h_ipp: float,
    rec_geodetic: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Ionospheric Pierce Point (IPP) location; it allows pre-computation
    of receiver geodetic coordinates to avoid redundant calculations

    Parameters:
    - rec_ecef: Receiver ECEF coordinates (x, y, z) in meters
    - sat_ecef: Satellite ECEF coordinates (x, y, z) in meters
    - h_ipp: Mean height of the ionosphere shell in meters
    - rec_geodetic: Optional pre-computed receiver geodetic coordinates (lat, lon, alt) in degrees/meters

    Returns:
    - lat_ipp, lon_ipp, azi, ele: (N,) NumPy array with:
        - lat: Latitude of IPP in degrees (None, if calculation fails)
        - lon: Longitude of IPP in degrees (None, if calculation fails)
        - azi: Azimuth angle from receiver to satellite in degrees (None, if calculation fails)
        - ele: Elevation angle from receiver to satellite in degrees (None, if calculation fails)
    """
    xA, yA, zA = rec_ecef
    xB, yB, zB = sat_ecef_array[:, 0], sat_ecef_array[:, 1], sat_ecef_array[:, 2]

    dx = xB - xA
    dy = yB - yA
    dz = zB - zA

    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * xA + dy * yA + dz * zA)
    c = xA**2 + yA**2 + zA**2 - (RE + h_ipp) ** 2

    disc = b**2 - 4 * a * c
    mask = disc >= 0

    # Init arrays with NaN
    lat_ipp = np.full_like(xB, np.nan, dtype=float)
    lon_ipp = np.full_like(xB, np.nan, dtype=float)
    azi = np.full_like(xB, np.nan, dtype=float)
    ele = np.full_like(xB, np.nan, dtype=float)

    if not np.any(mask):
        return lat_ipp, lon_ipp, azi, ele

    # Compute valid solutions
    sqrt_disc = np.sqrt(disc[mask])
    denom = 2 * a[mask]

    t1 = (-b[mask] + sqrt_disc) / denom
    t2 = (-b[mask] - sqrt_disc) / denom

    # Choose valid t (0 <= t <= 1), preferring the smaller one
    t1_valid = (0 <= t1) & (t1 <= 1)
    t2_valid = (0 <= t2) & (t2 <= 1)

    t = np.where(
        t1_valid & t2_valid,
        np.minimum(t1, t2),
        np.where(t1_valid, t1, np.where(t2_valid, t2, np.nan)),
    )

    valid = ~np.isnan(t)

    dxv, dyv, dzv = dx[mask][valid], dy[mask][valid], dz[mask][valid]
    tv = t[valid]

    x_ipp = xA + dxv * tv
    y_ipp = yA + dyv * tv
    z_ipp = zA + dzv * tv

    # Geodetic coordinates of IPP
    latv, lonv, _ = ecef2geodetic(x_ipp, y_ipp, z_ipp)

    # Azimuth and elevation angles for valid IPP points
    idx_out = np.flatnonzero(mask)[valid]
    lat_ipp[idx_out] = latv
    lon_ipp[idx_out] = lonv

    if rec_geodetic is None and rec_ecef is not None:
        # Compute receiver geodetic coordinates, if not provided, but ECEF is available
        rec_geodetic = ecef2geodetic(*rec_ecef)

    if rec_geodetic is not None and len(idx_out) > 0:
        aziv, elev, _ = ecef2aer(
            xB[mask][valid], yB[mask][valid], zB[mask][valid], *rec_geodetic, deg=True
        )
        azi[idx_out] = aziv
        ele[idx_out] = elev

    return lat_ipp, lon_ipp, azi, ele
