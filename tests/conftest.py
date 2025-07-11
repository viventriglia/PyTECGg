from pathlib import Path
import datetime

import pytest


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "rinex"


# Observation File Fixtures


@pytest.fixture
def obs_v2_file(test_data_dir):
    return str(test_data_dir / "v2" / "obs" / "cgtc0920.14o")


@pytest.fixture
def obs_v3_file(test_data_dir):
    return str(test_data_dir / "v3" / "obs" / "ASIR00ITA_R_20242810000_01D_30S_MO.rnx")


@pytest.fixture
def obs_v3_hatanaka_compressed_file(test_data_dir):
    return str(test_data_dir / "v3" / "obs" / "ASIR00ITA_R_20242810000_01D_30S_MO.crx")


@pytest.fixture
def obs_v3_gzip_file(test_data_dir):
    return str(
        test_data_dir / "v3" / "obs" / "ASIR00ITA_R_20242810000_01D_30S_MO.crx.gz"
    )


# Navigation File Fixtures


@pytest.fixture
def nav_v3_file(test_data_dir):
    return str(test_data_dir / "v3" / "nav" / "BRDC00WRD_R_20250870000_01D_MN.rnx")


# Other File Fixtures
@pytest.fixture
def invalid_file(tmp_path):
    return str(tmp_path / "nonexistent.obs")


@pytest.fixture
def ephemeris_data():
    dt = datetime.datetime(2024, 1, 1, 12, tzinfo=datetime.timezone.utc)
    return {
        "G01": {
            "toe": dt.timestamp(),
            "sqrta": 5153.7954775,
            "deltaN": 4.847e-9,
            "m0": 0.977384,
            "e": 0.0082,
            "omega": 1.640466,
            "cuc": 9.313e-6,
            "cus": 9.313e-6,
            "crc": 279.75,
            "crs": -88.0,
            "cic": 1.49e-7,
            "cis": 1.49e-7,
            "i0": 0.961685,
            "idot": 2.235e-10,
            "omega0": 1.640466,
            "omegaDot": -8.295e-9,
            "datetime": dt,
        }
    }
