import datetime

import numpy as np
import polars as pl

from pytecgg.satellites.kepler.orbits import _apply_geo_correction, _compute_anomalies
from pytecgg.satellites import GNSS_CONSTANTS
from pytecgg.satellites import positions as positions_module
from pytecgg.satellites.positions import _pick_nearest_keplerian_record


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


def test_glonass_ephemeris_schema_preserves_string_fields(monkeypatch):
    captured = {}

    def fake_compute_coordinates(sv_ids, epochs, ephem_data, coord_func, **kwargs):
        captured["schema"] = ephem_data.schema
        return pl.DataFrame(
            {
                "sv": sv_ids,
                "epoch": epochs,
                "sat_x": [0.0] * len(sv_ids),
                "sat_y": [0.0] * len(sv_ids),
                "sat_z": [0.0] * len(sv_ids),
            }
        )

    monkeypatch.setattr(
        positions_module, "_compute_coordinates", fake_compute_coordinates
    )

    epoch = pl.datetime(2023, 1, 1, 0, 0, 0, time_unit="us")
    obs_epoch = pl.select(epoch.alias("epoch")).item()
    sv_ids = pl.Series("sv", ["R01"])
    epochs = pl.Series("epoch", [obs_epoch])
    ephem_dict = {
        "R01": [
            {
                "constellation": "GLONASS",
                "sv": "R01",
                "datetime": obs_epoch,
                "gps_week": 0,
                "gps_seconds": 0.0,
                "satPosX": 1.0,
                "satPosY": 2.0,
                "satPosZ": 3.0,
                "velX": 0.1,
                "velY": 0.2,
                "velZ": 0.3,
                "accelX": 0.0,
                "accelY": 0.0,
                "accelZ": 0.0,
            }
        ]
    }

    result = positions_module.satellite_coordinates(sv_ids, epochs, ephem_dict)

    assert not result.is_empty()
    assert captured["schema"]["constellation"] == pl.String


# _pick_nearest_keplerian_record


def _record(dt: datetime.datetime) -> dict:
    return {"datetime": dt, "toe": 0.0}


def test_pick_nearest_keplerian_record_chooses_closest():
    epoch = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    records = [
        _record(datetime.datetime(2025, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)),
        _record(datetime.datetime(2025, 1, 1, 11, 30, 0, tzinfo=datetime.timezone.utc)),
        _record(datetime.datetime(2025, 1, 1, 14, 0, 0, tzinfo=datetime.timezone.utc)),
    ]

    chosen = _pick_nearest_keplerian_record(records, epoch)

    assert chosen is records[1]


def test_pick_nearest_keplerian_record_accepts_legacy_dict():
    """Manually built single-dict ephem entries must still work."""
    epoch = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    record = _record(datetime.datetime(2025, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc))

    chosen = _pick_nearest_keplerian_record(record, epoch)

    assert chosen is record


def test_pick_nearest_keplerian_record_empty_list_returns_none():
    epoch = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    assert _pick_nearest_keplerian_record([], epoch) is None
