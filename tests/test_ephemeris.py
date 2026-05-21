from datetime import datetime, timezone

import pytest
import polars as pl

from pytecgg.context import GNSSContext
from pytecgg.parsing import read_rinex_nav
from pytecgg.satellites.ephemeris import _get_gps_time, prepare_ephemeris


def test_get_gps_time():
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps_week, gps_seconds = _get_gps_time(dt)
    assert gps_week == 2295
    assert gps_seconds == 129600.0


def test_get_gps_time_type_safety():
    """
    Verify that _get_gps_time raises TypeError when provided with an invalid type.
    """
    with pytest.raises(TypeError):
        _get_gps_time(123456789)


def test_prepare_ephemeris_nav_v3(nav_v3_file):
    nav_data = read_rinex_nav(nav_v3_file)

    df_gps = nav_data.get("GPS")
    assert df_gps is not None
    assert isinstance(df_gps["epoch"].dtype, pl.Datetime)

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )

    ephemeris = prepare_ephemeris(nav_data, ctx)
    assert isinstance(ephemeris, dict)

    for sat, eph in ephemeris.items():
        assert sat.startswith("G")
        assert len(sat) == 3
        # Keplerian SVs now carry the full list of broadcast records so
        # downstream code can pick the one nearest to obs_time.
        assert isinstance(eph, list) and eph, f"expected non-empty list for {sat}"
        for record in eph:
            assert record["datetime"].tzinfo is not None
            assert record["constellation"] == "GPS"
