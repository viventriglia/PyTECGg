import numpy as np
import pytest

from pytecgg.tec_calibration.calibration_preprocessing import _polynomial_expansion


def test_polynomial_expansion_wraps_longitude_across_dateline():
    """A pierce point at 179°W seen from a station at 179°E is 2° east,
    not 358° west."""
    terms = _polynomial_expansion(
        modip_ipp=np.array([10.0]),
        modip_rec=np.array([10.0]),
        lon_ipp=np.array([-179.0]),
        lon_rec=np.array([179.0]),
        max_degree=3,
    )
    assert terms[0, 1] == pytest.approx(2.0)
