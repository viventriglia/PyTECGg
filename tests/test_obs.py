from pytecalrs import read_rinex_obs
from polars import DataFrame
import pytest


def test_output_types(obs_v2_file):
    """Check that the output types of read_rinex_obs are correct"""
    df, coords = read_rinex_obs(obs_v2_file)
    assert isinstance(df, DataFrame)
    assert isinstance(coords, tuple)
    assert len(coords) == 3
    assert all(isinstance(c, float) for c in coords)


def test_read_rinex_obs_valid_v2(obs_v2_file):
    """Test reading a RINEX v2 observation file"""
    df, (x, y, z) = read_rinex_obs(obs_v2_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])


def test_read_rinex_obs_valid_v3(obs_v3_file):
    """Test reading a RINEX v3 observation file"""
    df, (x, y, z) = read_rinex_obs(obs_v3_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])


def test_read_rinex_obs_valid_v3_hatanaka(obs_v3_hatanaka_compressed_file):
    """Test reading a (Hatanaka-compressed) RINEX v3 observation file"""
    df, (x, y, z) = read_rinex_obs(obs_v3_hatanaka_compressed_file)
    assert df.shape[0] > 0
    assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
    assert not all(v is None for v in [x, y, z])


# def test_read_rinex_obs_valid_v3_gzipped(obs_v3_gzip_file):
#     df, (x, y, z) = read_rinex_obs(obs_v3_gzip_file)
#     assert df.shape[0] > 0
#     assert all(col in df.columns for col in ["epoch", "sv", "observable", "value"])
#     assert not all(v is None for v in [x, y, z])


def test_read_rinex_obs_nonexistent_file(invalid_file):
    """Check that a FileNotFoundError is raised with a non-existent file"""
    with pytest.raises(FileNotFoundError):
        read_rinex_obs(invalid_file)
