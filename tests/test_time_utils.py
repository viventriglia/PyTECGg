import pytest
from datetime import datetime, timedelta, timezone

from pytecgg.satellites.ephemeris import _parse_time, _greg2gps


def test_parse_time_valid_gpst():
    """
    Test parsing a valid GPST time string with an offset
    """
    time_str = "2023-09-01T12:30:00 GPST"
    time_offset = timedelta(seconds=18)
    expected = datetime(2023, 9, 1, 12, 29, 42, tzinfo=timezone.utc)
    parsed = _parse_time(time_str, "GPST", time_offset)
    assert parsed == expected


def test_parse_time_no_offset():
    """
    Test parsing a valid GPST time string without an offset
    """
    time_str = "2023-09-01T12:00:00 GPST"
    parsed = _parse_time(time_str, "GPST", timedelta(0))
    expected = datetime(2023, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert parsed == expected


def test_parse_time_invalid_format():
    """
    Test parsing a time string with an invalid format: it should raise a ValueError
    """
    with pytest.raises(ValueError):
        _parse_time("bad format GPST", "GPST", timedelta(0))


def test_parse_time_wrong_type():
    """
    Test passing a non-string type for the time string: it should raise a TypeError
    """
    with pytest.raises(TypeError):
        _parse_time(12345, "GPST", timedelta(0))


def test_greg2gps_known_date():
    """
    Test the _greg2gps function with a known date
    """
    dt = datetime(2020, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
    # GPS week 2087, start of the week
    week, seconds = _greg2gps(dt)
    assert week == 2087
    assert seconds == 0.0
