import numpy as np
from pytecal.phase.ipp import calculate_ipp


def test_visible_satellite():
    """
    Evaluate IPP with a visible satellite and realistic receiver and satellite positions
    """
    rec_pos = (-1224960.9797, 5804226.5715, 2338188.7548)  # Receiver in Asia
    sat_pos = (-11177306.3509, 23710565.8502, 3758426.0384)  # GPS satellite
    h_ipp = 350_000

    lat, lon, azi, ele = calculate_ipp(rec_pos, sat_pos, h_ipp)

    assert -90 <= lat <= 90
    assert -180 <= lon <= 180
    assert 0 <= azi <= 360
    assert 0 <= ele <= 90


def test_nan_input():
    """
    NaN values in satellite coordinates
    """
    rec_pos = (0, 0, 0)
    sat_pos = (np.nan, np.nan, np.nan)
    h_ipp = 350_000

    assert calculate_ipp(rec_pos, sat_pos, h_ipp) == (None, None, None, None)


def test_no_intersection():
    """
    Satellite too close to the receiver to intersect the ionosphere
    """
    rec_pos = (0, 0, 0)
    sat_pos = (0, 0, 10_000)
    h_ipp = 350_000

    result = calculate_ipp(rec_pos, sat_pos, h_ipp)
    assert result == (None, None, None, None)
