import numpy as np

from pytecgg.satellites.positions import (
    satellite_coordinates,
    _compute_anomalies,
    _apply_geo_correction,
)
from pytecgg.satellites import GNSS_CONSTANTS


# def test_numerical_output(ephemeris_data):
#     pos, aux = satellite_coordinates(ephemeris_data, "G01", "GPS")

#     assert pos.shape == (3,)
#     assert aux.shape == (8,)

#     expected_pos = np.array([1.6e7, 1.3e7, 2.1e7])
#     assert np.allclose(pos, expected_pos, rtol=1e-2)


def test_compute_anomalies():
    """Test the computation of mean, eccentric, and true anomalies"""
    ecc = 0.01
    M0 = 1.0
    n = 0.0001
    tk = 1_000
    Mk, Ek, vk = _compute_anomalies(ecc, M0, n, tk)

    assert 0 <= Mk < 2 * np.pi
    assert 0 <= Ek < 2 * np.pi
    assert 0 <= vk < 2 * np.pi
    assert isinstance(Mk, float)
    assert isinstance(Ek, float)
    assert isinstance(vk, float)


def test_apply_geo_correction():
    """
    Test the transformation of coordinates for GEO satellites
    This test checks if the transformation correctly rotates a point on the X-axis
    """
    we = GNSS_CONSTANTS["GPS"].we
    Xk, Yk, Zk = 1.0, 0.0, 0.0
    tk = np.pi / (2 * we)

    X, Y, Z = _apply_geo_correction(Xk, Yk, Zk, tk, we)
    # After 90° rotation, the point should be on the Y-axis
    assert np.allclose([X, Y, Z], [0.0, -1.0, 0.0], atol=1e-6)
