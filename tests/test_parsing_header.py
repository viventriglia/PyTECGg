"""Tests for ``read_rinex_nav_header`` — the pure-Python header scanner."""

import gzip

from pytecgg.parsing import read_rinex_nav_header

# Minimal RINEX 3 nav header. Column 60-80 carries the label.
_HEADER_WITH_LEAP = (
    "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
    "ConvBin 0.1.0       AC                  20250328 000000 UTC PGM / RUN BY / DATE\n"
    "    18    18  2185     7                                    LEAP SECONDS\n"
    "                                                            END OF HEADER\n"
    "G01 2025 03 28 00 00 00 1.0e-04 0.0 0.0\n"  # one fake nav record after header
)

_HEADER_WITHOUT_LEAP = (
    "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
    "ConvBin 0.1.0       AC                  20250328 000000 UTC PGM / RUN BY / DATE\n"
    "                                                            END OF HEADER\n"
)


def test_read_rinex_nav_header_extracts_leap_seconds(tmp_path):
    nav_file = tmp_path / "fake.rnx"
    nav_file.write_text(_HEADER_WITH_LEAP, encoding="ascii")

    header = read_rinex_nav_header(nav_file)

    assert header == {"leap_seconds": 18}


def test_read_rinex_nav_header_returns_none_when_field_absent(tmp_path):
    nav_file = tmp_path / "no_leap.rnx"
    nav_file.write_text(_HEADER_WITHOUT_LEAP, encoding="ascii")

    header = read_rinex_nav_header(nav_file)

    assert header == {"leap_seconds": None}


def test_read_rinex_nav_header_handles_gzipped_input(tmp_path):
    nav_file = tmp_path / "fake.rnx.gz"
    with gzip.open(nav_file, "wt", encoding="ascii") as fh:
        fh.write(_HEADER_WITH_LEAP)

    header = read_rinex_nav_header(nav_file)

    assert header == {"leap_seconds": 18}


def test_read_rinex_nav_header_stops_at_end_of_header(tmp_path):
    """Body lines containing 'LEAP SECONDS' must not be parsed after END OF HEADER."""
    nav_file = tmp_path / "stop.rnx"
    body = (
        "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n"
        "                                                            END OF HEADER\n"
        # If the scanner kept reading, the line below would set leap_seconds=99.
        "    99    18  2185     7                                    LEAP SECONDS\n"
    )
    nav_file.write_text(body, encoding="ascii")

    header = read_rinex_nav_header(nav_file)

    assert header == {"leap_seconds": None}
