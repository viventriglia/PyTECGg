from datetime import datetime, timedelta, timezone

import polars as pl

from pytecggrs import read_rinex_nav
from pytecgg.satellites.ephemeris import _parse_time, prepare_ephemeris


def test_parse_time():
    time_str = "2024-01-01T12:00:00 GPST"
    time_offset = timedelta(seconds=18)
    dt = _parse_time(time_str, "GPST", time_offset)
    assert dt == datetime(2024, 1, 1, 11, 59, 42, tzinfo=timezone.utc)


def test_prepare_ephemeris_nav_v3(nav_v3_file):
    nav_data = read_rinex_nav(nav_v3_file)

    assert isinstance(nav_data, dict)
    assert "GPS" in nav_data
    assert isinstance(nav_data["GPS"], pl.DataFrame)

    ephemeris = prepare_ephemeris(nav_data, "GPS")

    assert isinstance(ephemeris, dict)
    assert len(ephemeris) > 0
    for sat, eph in ephemeris.items():
        assert "gps_week" in eph
        assert "gps_seconds" in eph
        assert "sv" in eph
        assert eph["constellation"] == "GPS"
