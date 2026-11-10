import datetime
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Sequence, Union, Literal

import numpy as np

from .constants import ALTITUDE_KM, LONGITUDES, LATITUDES


def _calculate_modip_grid(year: int, altitude_km: float = ALTITUDE_KM) -> np.ndarray:
    """
    Calculate MoDip grid for a specific year.

    MoDip (Modified Dip Latitude) is calculated from the IGRF magnetic field model.

    Parameters:
    -----------
    year : int
        Year for magnetic field calculation
    altitude_km : float
        Altitude in kilometers above Earth's surface

    Returns:
    --------
    np.ndarray
        MoDip grid values
    """
    from ppigrf import igrf

    dates = datetime.datetime(year, 1, 1)
    lon_grid, lat_grid = np.meshgrid(LONGITUDES, LATITUDES, indexing="ij")

    # Magnetic field components (east, north, upward) using IGRF model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Be, Bn, Bu = igrf(lon_grid, lat_grid, altitude_km, dates)

    # Horizontal component and the magnetic inclination
    Bh = np.sqrt(Be**2 + Bn**2)
    inclination_rad = -np.arctan2(Bu, Bh)

    # MoDip (Modified Dip Latitude)
    lat_rad = np.deg2rad(lat_grid)
    # modip = np.rad2deg(np.arctan(inclination_rad / np.sqrt(np.cos(lat_rad))))
    modip = np.arctan(inclination_rad / np.sqrt(np.cos(lat_rad)))

    # At the poles, cos(latitude) approaches zero
    modip = _polar_nan_values(modip, lat_grid)

    return modip[0]


def _polar_nan_values(modip: np.ndarray, lats_grid: np.ndarray) -> np.ndarray:
    """
    Handle NaN values at the poles by replacing them with ±π/2 (radians).

    At the poles, the calculation involves division by very small numbers
    which results in NaN values. These are replaced with the appropriate
    pole values (+π/2 for North pole, -π/2 for South pole).

    Parameters:
    -----------
    modip : np.ndarray
        MoDip grid with potential NaN values at poles
    lats_grid : np.ndarray
        Latitude grid for identifying pole locations

    Returns:
    --------
    modip : np.ndarray
        MoDip grid with NaN values replaced by pole values
    """
    for i in range(modip.shape[0]):
        nan_mask = np.isnan(modip[i])
        modip[i][nan_mask] = np.sign(lats_grid[nan_mask]) * np.pi / 2
    return modip


def _save_modip_grid(modip_grid: np.ndarray, year: int) -> None:
    """
    Save a MoDip grid for a given year to a compressed .npz file.
    For internal use to precompute MoDip grids distributed with the package.
    """
    outdir = Path(__file__).parent / "modip_grids"
    outfile = outdir / f"modip_{year}.npz"
    np.savez_compressed(outfile, modip_grid, allow_pickle=False)


@lru_cache(maxsize=5)
def _load_modip_grid(year: int) -> np.ndarray:
    """
    Load a precomputed MoDip grid for a given year, or compute it if missing.
    """
    from importlib.resources import files

    fname = f"modip_{year}.npz"
    modip_grids = files("pytecgg.tec_calibration.modip_grids").joinpath(fname)

    try:
        with modip_grids.open("rb") as f:
            npz = np.load(f, allow_pickle=False)
            return npz["arr_0"]
    except FileNotFoundError:
        return _calculate_modip_grid(year)


def _parse_coords(
    coords: Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        Sequence[tuple[float, float]],
        Sequence[tuple[float, float, float]],
        np.ndarray,
    ],
    coord_type: Literal["ecef", "geo"],
) -> np.ndarray:
    """
    Parse coordinates into lon/lat arrays.

    Parameters
    ----------
    coords : tuple, sequence, or np.ndarray
        Input coordinates, in one of the following forms:
        - (lon, lat) tuple of arrays
        - (x, y, z) tuple of arrays
        - Sequence of (lon, lat) tuples
        - Sequence of (x, y, z) tuples
        - np.ndarray of shape (N, 2) or (N, 3)
    coord_type : {"ecef", "geo"}
        Type of input coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with columns [lon, lat].
    """
    if coord_type == "geo":
        if isinstance(coords, tuple) and len(coords) == 2:
            lon, lat = coords
            points_geo = np.column_stack([lon, lat])
        else:
            points_geo = np.array(coords)
            if points_geo.ndim != 2 or points_geo.shape[1] != 2:
                raise ValueError(
                    f"Invalid coords for coord_type='{coord_type}': expected shape (N, 2), got {points_geo.shape}"
                )
    elif coord_type == "ecef":
        from pymap3d import ecef2geodetic

        if isinstance(coords, tuple) and len(coords) == 3:
            x, y, z = coords
            points_ecef = np.column_stack([x, y, z])
        else:
            points_ecef = np.array(coords)
            if points_ecef.ndim != 2 or points_ecef.shape[1] != 3:
                raise ValueError(
                    f"Invalid coords for coord_type='{coord_type}': expected shape (N, 3), got {points_ecef.shape}"
                )
        lat, lon, _ = ecef2geodetic(
            points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2]
        )
        points_geo = np.column_stack([lon, lat])
    return points_geo


def extract_modip(
    coords: Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
        Sequence[tuple[float, float]],
        Sequence[tuple[float, float, float]],
        np.ndarray,
    ],
    year: int,
    coord_type: Literal["ecef", "geo"],
) -> np.ndarray:
    """
    Interpolate MoDip values for given ECEF or geodetic (longitude/latitude)
    coordinates. The function interpolates MoDip values from precomputed grids,
    if available.

    Parameters
    ----------
    coords : tuple, sequence, or np.ndarray
        Input coordinates, in one of the following forms:
        - (lon, lat) tuple of arrays
        - (x, y, z) tuple of arrays
        - Sequence of (lon, lat) tuples
        - Sequence of (x, y, z) tuples
        - np.ndarray of shape (N, 2) or (N, 3)
    year : int
        Year for which the MoDip grid is used.
    coord_type : Literal["ecef", "geo"]
        Type of input coordinates:
        - "ecef" for ECEF coordinates.
        - "geo" for geographic coordinates; lon, lat expected.

    Returns
    --------
    np.ndarray
        Interpolated MoDip values for the specified coordinates [radians]
    """
    from scipy.interpolate import RegularGridInterpolator

    points_geo = _parse_coords(coords, coord_type)
    modip_grid = _load_modip_grid(year)
    interpolator = RegularGridInterpolator(
        (LONGITUDES, LATITUDES),
        modip_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    return interpolator(points_geo)
