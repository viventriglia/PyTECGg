from typing import Optional

import numpy as np
import polars as pl
from pymap3d import ecef2geodetic, ecef2aer

from .constants import RE
from pytecgg.context import GNSSContext


def calculate_ipp(
    df: pl.DataFrame,
    ctx: GNSSContext,
    min_elevation: Optional[float] = None,
) -> pl.DataFrame:
    """
    Calculate the Ionospheric Pierce Point (IPP) coordinates and satellite geometry.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing satellite ECEF coordinates ('sat_x', 'sat_y', 'sat_z').
    ctx : GNSSContext
        Context containing receiver position and IPP height.
    min_elevation : float, optional
        Minimum elevation angle in degrees. If provided, observations below
        this threshold are filtered out.

    Returns
    -------
    pl.DataFrame
        DataFrame with added columns: 'lat_ipp', 'lon_ipp', 'azi', 'ele'.
    """
    if df.is_empty():
        return df

    sat_ecef = df.select(["sat_x", "sat_y", "sat_z"]).to_numpy()
    xA, yA, zA = ctx.receiver_pos
    xB, yB, zB = sat_ecef[:, 0], sat_ecef[:, 1], sat_ecef[:, 2]
    h_ipp = ctx.h_ipp

    dx, dy, dz = xB - xA, yB - yA, zB - zA

    # Intersection segment-sphere (thin-shell approximation)
    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * xA + dy * yA + dz * zA)
    c = xA**2 + yA**2 + zA**2 - (RE + h_ipp) ** 2

    disc = b**2 - 4 * a * c
    mask = disc >= 0

    size_ = sat_ecef.shape[0]
    lat_ipp = np.full(size_, np.nan, dtype=float)
    lon_ipp = np.full(size_, np.nan, dtype=float)
    azi = np.full(size_, np.nan, dtype=float)
    ele = np.full(size_, np.nan, dtype=float)

    if np.any(mask):
        sqrt_disc = np.sqrt(disc[mask])
        denom = 2 * a[mask]
        t1 = (-b[mask] + sqrt_disc) / denom
        t2 = (-b[mask] - sqrt_disc) / denom

        # Choose valid t (0 <= t <= 1), preferring the smaller (closer) one
        t1_valid = (t1 >= 0) & (t1 <= 1)
        t2_valid = (t2 >= 0) & (t2 <= 1)

        t = np.select(
            [t1_valid & t2_valid, t1_valid, t2_valid],
            [np.minimum(t1, t2), t1, t2],
            default=np.nan,
        )

        valid_mask = ~np.isnan(t)
        if np.any(valid_mask):
            idx_out = np.flatnonzero(mask)[valid_mask]

            x_ipp = xA + dx[mask][valid_mask] * t[valid_mask]
            y_ipp = yA + dy[mask][valid_mask] * t[valid_mask]
            z_ipp = zA + dz[mask][valid_mask] * t[valid_mask]

            latv, lonv, _ = ecef2geodetic(x_ipp, y_ipp, z_ipp)
            lat_ipp[idx_out] = latv
            lon_ipp[idx_out] = lonv

            rec_geodetic = ecef2geodetic(xA, yA, zA)
            aziv, elev, _ = ecef2aer(
                xB[mask][valid_mask],
                yB[mask][valid_mask],
                zB[mask][valid_mask],
                *rec_geodetic,
                deg=True,
            )
            azi[idx_out] = aziv
            ele[idx_out] = elev

    df_result = df.with_columns(
        [
            pl.Series("lat_ipp", lat_ipp),
            pl.Series("lon_ipp", lon_ipp),
            pl.Series("azi", azi),
            pl.Series("ele", ele),
        ]
    )

    if min_elevation is not None:
        # fill_nan: NaN >= x is True in polars; null rows are dropped by filter
        df_result = df_result.filter(pl.col("ele").fill_nan(None) >= min_elevation)

    return df_result
