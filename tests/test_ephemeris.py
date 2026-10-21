from datetime import datetime, timezone

import pytest
import polars as pl

from pytecgg.context import GNSSContext
from pytecgg.parsing import read_rinex_nav
from pytecgg.satellites.ephemeris import (
    _KEPLERIAN_CADENCE_S,
    _drop_week_rollover_bug,
    _get_gps_time,
    prepare_ephemeris,
)


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


def _bad_row(epoch: datetime, cadence: float) -> dict:
    return {"epoch": epoch, "toe": cadence}


def _good_row(epoch: datetime, toe: float) -> dict:
    return {"epoch": epoch, "toe": toe}


def test_drop_week_rollover_bug_drops_corrupted_beidou_row():
    """At BDT Sunday 00:00:00, toe = cadence is the rinex-crate bug signature."""
    cadence = _KEPLERIAN_CADENCE_S["BEIDOU"]
    rollover_sunday = datetime(2025, 1, 5, 0, 0, 0)  # Sunday 00:00:00
    df = pl.DataFrame(
        [
            _bad_row(rollover_sunday, cadence),
            _good_row(datetime(2025, 1, 5, 1, 0, 0), cadence),
            _good_row(datetime(2025, 1, 5, 2, 0, 0), 7200.0),
        ]
    )

    with pytest.warns(UserWarning, match="week rollover"):
        out = _drop_week_rollover_bug(df, "BEIDOU", "C01")

    assert out.height == 2
    assert rollover_sunday not in out["epoch"].to_list()


def test_drop_week_rollover_bug_leaves_other_records_alone():
    """A toe equal to cadence on any non-rollover epoch must NOT be dropped."""
    cadence = _KEPLERIAN_CADENCE_S["GALILEO"]
    df = pl.DataFrame(
        [
            # Sunday but not at 00:00:00 -> keep
            _bad_row(datetime(2025, 1, 5, 0, 10, 0), cadence),
            # Monday at 00:00:00 -> keep (weekday() != 7)
            _bad_row(datetime(2025, 1, 6, 0, 0, 0), cadence),
        ]
    )

    out = _drop_week_rollover_bug(df, "GALILEO", "E01")

    assert out.height == 2


def test_drop_week_rollover_bug_noop_for_glonass():
    """GLONASS has no cadence entry, so the filter is a no-op."""
    df = pl.DataFrame([{"epoch": datetime(2025, 1, 5, 0, 0, 0), "toe": 600.0}])

    out = _drop_week_rollover_bug(df, "GLONASS", "R01")

    assert out.height == 1


def test_prepare_ephemeris_populates_leap_seconds(nav_v3_file):
    """Each record must carry leap_seconds, defaulting when ctx leaves it None."""
    nav_data = read_rinex_nav(nav_v3_file)

    ctx_default = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )
    ephem_default = prepare_ephemeris(nav_data, ctx_default)
    any_sv = next(iter(ephem_default))
    assert ephem_default[any_sv][0]["leap_seconds"] == 18  # DEFAULT_LEAP_SECONDS_UTC_GPST

    ctx_override = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
        leap_seconds=37,
    )
    ephem_override = prepare_ephemeris(nav_data, ctx_override)
    assert ephem_override[any_sv][0]["leap_seconds"] == 37
