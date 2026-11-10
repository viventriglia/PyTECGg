import pytest
from math import isclose, pi
import numpy as np
from numpy import sin

from pytecgg.satellites.constants import BDT_OFFSET_FROM_GPST
from pytecgg.satellites.kepler.orbits import (
    _compute_time_elapsed,
    _kepler,
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


def test_kepler_is_vectorized():
    """Each element of an array input converges like its scalar counterpart."""
    e = np.array([0.0, 0.1, 0.7])
    mk = np.array([pi / 4, pi / 3, 2.0])
    ek = _kepler(e, mk, tol=0.01)
    assert np.allclose(ek - e * np.sin(ek), mk, atol=0.01 * (pi / 648_000))
    assert _kepler(np.array([]), np.array([]), tol=0.01).size == 0


# Time elapsed since toe (GPST obs epochs -> GPST/GST/BDT)

# 2024-01-01 12:00:00 GPST is GPS week 2295, 129600 s of week
_OBS_GPST_S = 2295 * 604_800 + 129_600.0


@pytest.mark.parametrize("system", ["GPS", "GALILEO", "QZSS", None])
def test_compute_time_elapsed_gpst_family_needs_no_shift(system):
    """Obs epochs are already GPST; GST and QZSST align with GPST."""
    assert _compute_time_elapsed(_OBS_GPST_S, 2295, 129_600, gnss_system=system) == 0.0


def test_compute_time_elapsed_beidou_subtracts_bdt_offset():
    """BDT = GPST - BDT_OFFSET_FROM_GPST (14 s, fixed by BDS ICD)."""
    dt_bdt = _compute_time_elapsed(_OBS_GPST_S, 2295, 129_600, gnss_system="BEIDOU")
    assert dt_bdt == -BDT_OFFSET_FROM_GPST
