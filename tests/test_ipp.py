import numpy as np
import polars as pl
import pytest

from pytecgg.satellites.ipp import calculate_ipp
from pytecgg.satellites.constants import RE
from pytecgg.context import GNSSContext


def test_visible_satellite():
    """
    Evaluate IPP with a visible satellite and realistic receiver and satellite positions
    """
    # Receiver in Asia
    rec_pos = (-1224960.9797, 5804226.5715, 2338188.7548)
    # GPS satellite
    df_sat = pl.DataFrame(
        {
            "sat_x": [-11177306.3509],
            "sat_y": [23710565.8502],
            "sat_z": [3758426.0384],
        }
    )

    ctx = GNSSContext(
        receiver_pos=rec_pos,
        receiver_name="TEST",
        rinex_version="3.04",
        h_ipp=350_000,
        systems=["G"],
    )

    df_res = calculate_ipp(df_sat, ctx)

    assert df_res["lat_ipp"][0] == pytest.approx(20.7, abs=0.5)
    assert df_res["lon_ipp"][0] == pytest.approx(102.6, abs=0.5)
    assert df_res["azi"][0] == pytest.approx(134.1, abs=0.5)
    assert df_res["ele"][0] == pytest.approx(65.9, abs=0.5)


def test_nan_input():
    """
    NaN values in satellite coordinates
    """
    df_sat = pl.DataFrame(
        {
            "sat_x": [np.nan],
            "sat_y": [np.nan],
            "sat_z": [np.nan],
        }
    )

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, RE),
        receiver_name="TEST",
        rinex_version="3.04",
        h_ipp=350_000,
        systems=["G"],
    )

    df_res = calculate_ipp(df_sat, ctx)

    assert df_res["lat_ipp"].is_nan().all()
    assert df_res["lon_ipp"].is_nan().all()
    assert df_res["azi"].is_nan().all()
    assert df_res["ele"].is_nan().all()


def test_no_intersection():
    """
    Satellite too close to the receiver to intersect the ionosphere
    """
    df_sat = pl.DataFrame(
        {
            "sat_x": [0.0],
            "sat_y": [0.0],
            "sat_z": [RE + 20_000.0],
        }
    )

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, RE),
        receiver_name="LOW_SAT_TEST",
        rinex_version="3.04",
        h_ipp=350_000,
        systems=["G"],
    )

    df_res = calculate_ipp(df_sat, ctx)

    assert df_res["lat_ipp"].is_nan().all()
    assert df_res["lon_ipp"].is_nan().all()
    assert df_res["azi"].is_nan().all()
    assert df_res["ele"].is_nan().all()


def test_min_elevation_drops_rows_without_geometry():
    """NaN compares greater than any number in polars, so rows with no
    satellite position must be dropped explicitly by the elevation filter."""
    df_sat = pl.DataFrame(
        {
            "sat_x": [-11177306.3509, np.nan],
            "sat_y": [23710565.8502, np.nan],
            "sat_z": [3758426.0384, np.nan],
        }
    )
    ctx = GNSSContext(
        receiver_pos=(-1224960.9797, 5804226.5715, 2338188.7548),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )

    df_res = calculate_ipp(df_sat, ctx, min_elevation=20)

    assert df_res.height == 1
    assert df_res["ele"][0] == pytest.approx(65.9, abs=0.5)
