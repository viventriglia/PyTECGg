import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "rinex"


@pytest.fixture
def obs_v2_file(test_data_dir):
    return str(test_data_dir / "v2" / "obs" / "cgtc0920.14o")


@pytest.fixture
def obs_v3_file(test_data_dir):
    return str(test_data_dir / "v3" / "obs" / "ASIR00ITA_R_20242810000_01D_30S_MO.crx")


@pytest.fixture
def nav_v2_file(test_data_dir):
    return str(test_data_dir / "v2" / "nav" / "BRDC00GOP_R_20140920000_01D_MN.rnx")


@pytest.fixture
def invalid_file(tmp_path):
    return str(tmp_path / "nonexistent.obs")
