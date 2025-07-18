from math import isclose, pi
from pytecgg.satellites.kepler import kepler
from numpy import sin


def test_kepler_circular_orbit():
    """
    When the eccentricity is zero, the eccentric anomaly (ek) equals the mean anomaly (mk)
    """
    for mk in [0, pi / 4, pi / 2, pi, 2 * pi]:
        assert isclose(kepler(0.0, mk, tol=0.01), mk, rel_tol=1e-12)


def test_kepler_low_eccentricity():
    """
    With low eccentricity, the eccentric anomaly (ek) should be close to the mean anomaly (mk)
    and satisfy the equation mk ≈ ek - e * sin(ek)
    """
    e = 0.1
    mk = pi / 3
    ek = kepler(e, mk, tol=0.01)
    lhs = ek - e * sin(ek)
    assert isclose(lhs, mk, abs_tol=0.01 * (pi / 648_000))


def test_kepler_close_to_parabolic():
    """
    Close to the parabolic limit (e → 1), the eccentric anomaly (ek) should still satisfy
    the equation mk ≈ ek - e * sin(ek) with a reasonable tolerance
    """
    e = 0.999
    mk = pi / 2
    ek = kepler(e, mk, tol=0.01)
    lhs = ek - e * sin(ek)
    assert isclose(lhs, mk, abs_tol=0.01 * (pi / 648_000))
