import datetime

import pytest
from math import isclose, pi
from numpy import sin

from pytecgg.satellites.constants import BDT_OFFSET_FROM_GPST
from pytecgg.satellites.kepler.orbits import (
    _compute_time_elapsed,
    _kepler,
    _gpst_to_system_time,
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


# Time-system conversion (GPST obs epochs -> GPST/GST/BDT)

_GPST_REF = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.mark.parametrize("system", ["GPS", "GALILEO", "QZSS", None])
def test_gpst_to_system_time_gpst_family_is_identity(system):
    """Obs epochs are already GPST; GST and QZSST align with GPST."""
    assert _gpst_to_system_time(_GPST_REF, system) == _GPST_REF


def test_gpst_to_system_time_beidou_subtracts_bdt_offset():
    """BDT = GPST - BDT_OFFSET_FROM_GPST (14 s, fixed by BDS ICD)."""
    out = _gpst_to_system_time(_GPST_REF, "BEIDOU")
    assert (_GPST_REF - out).total_seconds() == BDT_OFFSET_FROM_GPST


def test_compute_time_elapsed_beidou_offset_relative_to_gps():
    """For the same GPST obs_time, BeiDou elapsed = GPS elapsed - 14 s."""
    obs = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    gps_week = 2295
    toe = 129600

    dt_gps = _compute_time_elapsed(obs, gps_week, toe, gnss_system="GPS")
    dt_bdt = _compute_time_elapsed(obs, gps_week, toe, gnss_system="BEIDOU")

    assert dt_gps - dt_bdt == BDT_OFFSET_FROM_GPST


def test_compute_time_elapsed_naive_obs_is_treated_as_gpst():
    """A tz-naive obs_time must be promoted, not raise; 2024-01-01 12:00 is toe."""
    obs_naive = datetime.datetime(2024, 1, 1, 12, 0, 0)
    gps_week = 2295
    toe = 129600

    assert _compute_time_elapsed(obs_naive, gps_week, toe, gnss_system="GPS") == 0.0
