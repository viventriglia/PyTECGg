from datetime import datetime, timezone

import pytest
import polars as pl

from pytecgg.context import GNSSContext
from pytecgg.parsing import read_rinex_nav
from pytecgg.satellites.ephemeris import (
    _get_gps_time,
    _leap_seconds_at,
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


def test_prepare_ephemeris_populates_leap_seconds(nav_v3_file):
    """Each record must carry leap_seconds, from the nav header when ctx leaves it None."""
    nav_data = read_rinex_nav(nav_v3_file)

    ctx_default = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
    )
    ephem_default = prepare_ephemeris(nav_data, ctx_default)
    any_sv = next(iter(ephem_default))
    assert ephem_default[any_sv][0]["leap_seconds"] == 18  # header LEAP SECONDS

    ctx_override = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["G"],
        leap_seconds=37,
    )
    ephem_override = prepare_ephemeris(nav_data, ctx_override)
    assert ephem_override[any_sv][0]["leap_seconds"] == 37


def test_prepare_ephemeris_keeps_one_galileo_record_per_epoch(nav_v3_file):
    """Galileo nav files repeat each record once per data source (F/NAV 258,
    I/NAV 513/516/517) with the same orbit; keep one, preferring I/NAV E1+E5b."""
    nav_data = read_rinex_nav(nav_v3_file)
    assert nav_data["GALILEO"].select("epoch", "sv").is_duplicated().any()

    ctx = GNSSContext(
        receiver_pos=(0.0, 0.0, 0.0),
        receiver_name="TEST",
        rinex_version="3.04",
        systems=["E"],
    )
    ephemeris = prepare_ephemeris(nav_data, ctx)

    records = [record for sv_records in ephemeris.values() for record in sv_records]
    keys = [(record["sv"], record["datetime"]) for record in records]
    assert len(keys) == len(set(keys))
    assert {record["source"] for record in records} >= {517.0}


def test_leap_seconds_table():
    utc = timezone.utc
    assert _leap_seconds_at(datetime(1980, 1, 6, tzinfo=utc)) == 0
    assert _leap_seconds_at(datetime(2014, 4, 2, tzinfo=utc)) == 16
    assert _leap_seconds_at(datetime(2016, 12, 31, 23, 59, tzinfo=utc)) == 17
    assert _leap_seconds_at(datetime(2017, 1, 1, tzinfo=utc)) == 18
    with pytest.warns(UserWarning, match="past the built-in table"):
        assert _leap_seconds_at(datetime(2030, 1, 1, tzinfo=utc)) == 18


@pytest.mark.parametrize("strip_header_line", [False, True])
def test_glonass_leap_seconds_follow_the_nav_file(test_data_dir, tmp_path, strip_header_line):
    """A 2014 nav file needs 16 s, not today's 18: from its header, or from the
    date table when the header has no LEAP SECONDS line."""
    path = test_data_dir / "v3" / "nav" / "BRDC00GOP_R_20140920000_01D_MN.rnx"
    if strip_header_line:
        lines = path.read_text(encoding="ascii").splitlines(keepends=True)
        path = tmp_path / path.name
        path.write_text("".join(l for l in lines if "LEAP SECONDS" not in l[60:]), encoding="ascii")
    nav = read_rinex_nav(path)
    assert nav["GLONASS"]["leap_seconds"].to_list()[0] == (None if strip_header_line else 16)
    ctx = GNSSContext(receiver_pos=(0.0, 0.0, 0.0), receiver_name="TEST", rinex_version="3.04", systems=["R"])
    ephem = prepare_ephemeris(nav, ctx)
    assert {rec["leap_seconds"] for recs in ephem.values() for rec in recs} == {16}


def _glonass_nav(test_data_dir):
    return read_rinex_nav(test_data_dir / "v3" / "nav" / "BRDC00GOP_R_20140920000_01D_MN.rnx")


def _glonass_ephem(nav):
    ctx = GNSSContext(receiver_pos=(0.0, 0.0, 0.0), receiver_name="TEST", rinex_version="3.04", systems=["R"])
    return prepare_ephemeris(nav, ctx)


def test_glonass_foreign_orbit_with_own_clock_is_dropped(test_data_dir):
    """This file carries R06's state vector under R17 for 09:15-10:45, with R17's own
    clock offset and channel: only those 4 records go, every other one stays."""
    nav = _glonass_nav(test_data_dir)
    with pytest.warns(UserWarning, match="R17: dropped 4 ") as caught:
        ephem = _glonass_ephem(nav)
    assert sum("dropped" in str(w.message) for w in caught) == 1
    n_epochs = nav["GLONASS"].group_by("sv").agg(pl.col("epoch").n_unique()).rows()
    assert all(len(ephem[f"R{int(sv):02d}"]) == n - 4 * (sv == "17") for sv, n in n_epochs)
    r06 = {rec["datetime"]: rec["satPosX"] for rec in ephem["R06"]}
    assert not {rec["satPosX"] for rec in ephem["R17"]} & set(r06.values())


@pytest.mark.parametrize("n_foreign", [3, 9])
@pytest.mark.parametrize("replace", [False, True])
def test_glonass_mislabelled_records_are_dropped(test_data_dir, replace, n_foreign):
    """Another satellite's messages filed under R01, as in BRDC00IGS_R on 2024-01-01/02:
    alongside R01's own records for the same epochs, or in place of them, for up to
    several hours (9 records: 4.5 h, like R13 on 2024-01-01)."""
    nav = _glonass_nav(test_data_dir)
    glo = nav["GLONASS"]
    r01 = glo.filter(pl.col("sv") == "1").sort("epoch")
    epochs = r01["epoch"][10 : 10 + n_foreign]
    foreign = glo.filter((pl.col("sv") == "2") & pl.col("epoch").is_in(epochs.implode()))
    assert foreign.height == n_foreign
    if replace:
        glo = glo.filter(~((pl.col("sv") == "1") & pl.col("epoch").is_in(foreign["epoch"].implode())))
    nav = {**nav, "GLONASS": pl.concat([glo, foreign.with_columns(sv=pl.lit("1"))])}

    with pytest.warns(UserWarning, match=f"R01: dropped {n_foreign} "):
        ephem = _glonass_ephem(nav)

    kept = {rec["datetime"]: rec["satPosX"] for rec in ephem["R01"]}
    if replace:
        assert len(kept) == r01.height - n_foreign and not set(foreign["epoch"]) & set(kept)
    else:
        assert kept == dict(zip(r01["epoch"], r01["satPosX"]))

