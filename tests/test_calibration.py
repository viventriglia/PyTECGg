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


def test_modip_is_radians_and_unrounded():
    """MoDip stays within ±π/2 near the poles and resolves small latitude steps."""
    from pytecgg.tec_calibration.modip import extract_modip

    polar = extract_modip(
        coords=(np.array([0.0, 0.0]), np.array([89.5, -89.5])),
        year=2024,
        coord_type="geo",
    )
    assert np.all(np.abs(polar) <= np.pi / 2)

    rome = extract_modip(
        coords=(np.array([12.5, 12.5]), np.array([41.9, 42.0])),
        year=2024,
        coord_type="geo",
    )
    assert rome[1] > rome[0]
