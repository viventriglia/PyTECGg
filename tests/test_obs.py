from pytecgg.parsing import read_rinex_obs
from polars import DataFrame
import pytest


def test_output_types(obs_v2_file):
    """Check that the output types of read_rinex_obs are correct"""
    df, coords, version = read_rinex_obs(obs_v2_file)
    assert isinstance(df, DataFrame)
    assert isinstance(coords, tuple)
    assert isinstance(version, str)
    assert len(coords) == 3
    assert all(isinstance(c, float) for c in coords)


def test_read_rinex_obs_valid_v2(obs_v2_file):
    """Test reading a RINEX v2 observation file"""
    df, (x, y, z), version = read_rinex_obs(obs_v2_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])
    assert version.startswith("2")


def test_read_rinex_obs_valid_v3(obs_v3_file):
    """Test reading a RINEX v3 observation file"""
    df, (x, y, z), version = read_rinex_obs(obs_v3_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])
    assert version.startswith("3")


def test_read_rinex_obs_valid_v3_hatanaka(obs_v3_hatanaka_compressed_file):
    """Test reading a (Hatanaka-compressed) RINEX v3 observation file"""
    df, (x, y, z), version = read_rinex_obs(obs_v3_hatanaka_compressed_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])
    assert version.startswith("3")


def test_read_rinex_obs_valid_v3_gzipped(obs_v3_gzip_file):
    df, (x, y, z), version = read_rinex_obs(obs_v3_gzip_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])
    assert version.startswith("3")


def test_read_rinex_obs_nonexistent_file(invalid_file):
    """Check that a FileNotFoundError is raised with a non-existent file"""
    with pytest.raises(FileNotFoundError):
        read_rinex_obs(invalid_file)


def test_sv_ids_use_standard_format(parsed_rinex_obs_data):
    """Every sv must be "<constellation letter><2-digit PRN>": the first
    character selects the constellation everywhere downstream. rinex >= 0.24
    names SBAS vehicles (S23 -> "ASTRA-5B") unless asked for this format."""
    sv = parsed_rinex_obs_data["obs_data"]["sv"].unique()
    assert "S23" in sv.to_list()
    assert sv.str.contains(r"^[A-Z]\d{2}$").all()


def test_malformed_leap_seconds_line_is_ignored(obs_v3_file, tmp_path):
    """rinex 0.24 panics on a short LEAP SECONDS line (REDZ00POL 2024-224 has
    "    18  2326     7", the future field missing); the reader drops that line."""
    text = open(obs_v3_file, encoding="ascii").read()
    good = "    18    18  1929     7                                    LEAP SECONDS"
    bad = "    18  1929     7                                          LEAP SECONDS"
    assert good in text
    path = tmp_path / "ASIR00ITA_R_20242810000_01D_30S_MO.rnx"
    path.write_text(text.replace(good, bad), encoding="ascii")
    df, pos, version = read_rinex_obs(str(path))
    expected = read_rinex_obs(obs_v3_file)
    assert df.equals(expected[0]) and (pos, version) == expected[1:]
