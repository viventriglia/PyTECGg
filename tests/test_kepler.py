import datetime

import pytest
from math import isclose, pi
from numpy import sin

from pytecgg.satellites.constants import BDT_OFFSET_FROM_GPST
from pytecgg.satellites.kepler.orbits import (
    _compute_time_elapsed,
    _kepler,
    _utc_to_system_time,
)


def test_kepler_circular_orbit():
    """
    When the eccentricity is zero, the eccentric anomaly (ek) equals the mean anomaly (mk)
    """
    for mk in [0, pi / 4, pi / 2, pi, 2 * pi]:
        assert isclose(_kepler(0.0, mk, tol=0.01), mk, rel_tol=1e-12)


def test_kepler_low_eccentricity():
    """
    With low eccentricity, the eccentric anomaly (ek) should be close to the mean anomaly (mk)
    and satisfy the equation mk ≈ ek - e * sin(ek)
    """
    e = 0.1
    mk = pi / 3
    ek = _kepler(e, mk, tol=0.01)
    lhs = ek - e * sin(ek)
    assert isclose(lhs, mk, abs_tol=0.01 * (pi / 648_000))


def test_kepler_close_to_parabolic():
    """
    Close to the parabolic limit (e → 1), the eccentric anomaly (ek) should still satisfy
    the equation mk ≈ ek - e * sin(ek) with a reasonable tolerance
    """
    e = 0.999
    mk = pi / 2
    ek = _kepler(e, mk, tol=0.01)
    lhs = ek - e * sin(ek)
    assert isclose(lhs, mk, abs_tol=0.01 * (pi / 648_000))


# Time-system conversion (UTC -> GPST/GST/BDT/UTC)

_UTC_REF = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.mark.parametrize("system", ["GPS", "GALILEO", "QZSS"])
def test_utc_to_system_time_gpst_family_adds_leap_seconds(system):
    """GPST = UTC + leap_seconds; GST and QZSST align with GPST."""
    out = _utc_to_system_time(_UTC_REF, system, leap_seconds=18)
    assert (out - _UTC_REF).total_seconds() == 18


def test_utc_to_system_time_beidou_subtracts_bdt_offset():
    """BDT = GPST - BDT_OFFSET_FROM_GPST (14 s, fixed by BDS ICD)."""
    out = _utc_to_system_time(_UTC_REF, "BEIDOU", leap_seconds=18)
    assert (out - _UTC_REF).total_seconds() == 18 - BDT_OFFSET_FROM_GPST


def test_utc_to_system_time_glonass_is_identity():
    """GLONASS uses UTC directly: no shift applied."""
    out = _utc_to_system_time(_UTC_REF, "GLONASS", leap_seconds=18)
    assert out == _UTC_REF


def test_utc_to_system_time_unknown_system_falls_back_to_utc():
    out = _utc_to_system_time(_UTC_REF, None, leap_seconds=18)
    assert out == _UTC_REF


def test_compute_time_elapsed_applies_leap_seconds_for_gps():
    """GPS path shifts UTC into GPST before subtracting toe.

    Two identical UTC obs_times with different leap_seconds must produce
    elapsed values differing by exactly the leap-second delta.
    """
    # 2024-01-01 12:00:00 UTC <-> GPS week 2295, seconds 129600
    obs = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    gps_week = 2295
    toe = 129600

    dt_18 = _compute_time_elapsed(obs, gps_week, toe, gnss_system="GPS", leap_seconds=18)
    dt_20 = _compute_time_elapsed(obs, gps_week, toe, gnss_system="GPS", leap_seconds=20)

    assert dt_20 - dt_18 == 2.0


def test_compute_time_elapsed_beidou_offset_relative_to_gps():
    """For the same UTC obs_time, BeiDou elapsed = GPS elapsed - 14 s."""
    obs = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    gps_week = 2295
    toe = 129600

    dt_gps = _compute_time_elapsed(obs, gps_week, toe, gnss_system="GPS", leap_seconds=18)
    dt_bdt = _compute_time_elapsed(
        obs, gps_week, toe, gnss_system="BEIDOU", leap_seconds=18
    )

    assert dt_gps - dt_bdt == BDT_OFFSET_FROM_GPST


def test_compute_time_elapsed_naive_obs_is_treated_as_utc():
    """A tz-naive obs_time must be promoted to UTC, not raise."""
    obs_naive = datetime.datetime(2024, 1, 1, 12, 0, 0)
    gps_week = 2295
    toe = 129600

    dt = _compute_time_elapsed(
        obs_naive, gps_week, toe, gnss_system="GPS", leap_seconds=18
    )
    assert dt == 18.0
