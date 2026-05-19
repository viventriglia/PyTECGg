import numpy as np
import polars as pl
from pymap3d import ecef2geodetic

from pytecgg.tec_calibration.modip import extract_modip


def _polynomial_expansion(
    modip_ipp: np.ndarray,
    modip_rec: np.ndarray,
    lon_ipp: np.ndarray,
    lon_rec: np.ndarray,
    delta_lt: np.ndarray,
    max_degree: int,
) -> np.ndarray:
    """
    Compute polynomial expansion terms for TEC modeling.

    The expansion combines differences in longitude and modified dip latitude (MoDip)
    between ionospheric pierce point (IPP) and receiver location, plus a linear
    local-time term that lets vTEC vary across the batch's time window.

    Parameters
    ----------
    modip_ipp : np.ndarray
        MoDip values at IPP
    modip_rec : np.ndarray
        MoDip values at the receiver position
    lon_ipp : np.ndarray
        Longitudes of IPP [degrees]
    lon_rec : np.ndarray
        Longitudes of the receiver position [degrees]
    delta_lt : np.ndarray
        Local-time offset at IPP relative to the batch center [hours].
        Defined as (epoch - batch_center)_hours + (lon_ipp - lon_rec) / 15.
        Carries the diurnal gradient information: rows at the same epoch but
        different IPP longitudes get different values, and rows at the same
        IPP but different epochs also get different values.
    max_degree : int
        Maximum polynomial degree for MoDip terms

    Returns
    -------
    np.ndarray
        Matrix of polynomial expansion terms with shape (n_points, 3 + max_degree).
        Columns: [const, Δlon, Δmodip, Δmodip², ..., Δmodip^max_degree, Δt_LT].
        The local-time column is un-damped so it stays active for low-elevation
        IPPs (high |Δmodip|) where the diurnal gradient matters most.
    """
    delta_lon = lon_ipp - lon_rec
    delta_modip = modip_ipp - modip_rec
    const_norm = 1.0 / (1.0 + np.abs(delta_modip) ** (max_degree + 1))

    n_points = modip_ipp.shape[0]
    n_terms = 3 + max_degree
    pterms_matrix = np.zeros((n_points, n_terms))

    # Constant and longitude terms (2 terms)
    pterms_matrix[:, 0] = const_norm
    pterms_matrix[:, 1] = delta_lon * const_norm
    # MoDip terms (max_degree terms)
    for j in range(1, max_degree + 1):
        pterms_matrix[:, 1 + j] = (delta_modip**j) * const_norm
    # Local-time term (un-damped)
    pterms_matrix[:, 2 + max_degree] = delta_lt

    return pterms_matrix


def _mapping_function(elevation: np.ndarray, h_ipp: float) -> np.ndarray:
    """
    Compute the mapping function to convert slant to vertical TEC.

    Parameters
    ----------
    elevation : np.ndarray
        Satellite elevation angles [degrees]
    h_ipp : float
        Height of IPP [km]

    Returns
    -------
    np.ndarray
    """
    return np.cos(
        np.arcsin((6_371 / (6_371 + h_ipp / 1_000)) * np.cos(np.radians(elevation)))
    )


def _preprocessing(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    h_ipp: float,
) -> pl.DataFrame:
    """
    Preprocess GNSS data for TEC modeling. Adds mapping function, vTEC, and
    MoDip parameters for both IPP and receiver positions.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with columns:
        - epoch: observation timestamps
        - ele: satellite elevation [degrees]
        - lon_ipp, lat_ipp: IPP coordinates [degrees]
        - gflc_levelled: levelled GFLC values
    receiver_position : tuple[float, float, float]
        Receiver position in ECEF coordinates (x, y, z) [meters].
    h_ipp : float
        Height of the ionospheric pierce point [m].

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns:
        - mapping: mapping function values
        - gflc_vert: vTEC estimates
        - modip_ipp, modip_rec: MoDip values (IPP and receiver)
        - lon_rec: receiver longitude [degrees]
    """
    year = df["epoch"][0].year

    modip_ipp = extract_modip(
        coords=(df["lon_ipp"].to_numpy(), df["lat_ipp"].to_numpy()),
        year=year,
        coord_type="geo",
    )
    modip_rec = extract_modip(coords=receiver_position, year=year, coord_type="ecef")[0]
    _, lon_rec, _ = ecef2geodetic(*receiver_position)

    mapping = _mapping_function(df["ele"].to_numpy(), h_ipp)
    gflc_vert = df["gflc_levelled"].to_numpy() * mapping

    return df.with_columns(
        [
            pl.Series("mapping", mapping),
            pl.Series("gflc_vert", gflc_vert),
            pl.Series("modip_ipp", modip_ipp),
            pl.lit(modip_rec).alias("modip_rec"),
            pl.lit(lon_rec).alias("lon_rec"),
        ]
    ).drop_nans()
