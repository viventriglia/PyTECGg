from pathlib import Path
from unittest.mock import MagicMock, patch
from requests.exceptions import RequestException

import pytest

from pytecgg.utils.download_rinex import (
    download_obs_ring,
    download_obs_euref,
    download_nav_bkg,
    _download_file,
)


def test_obs_ring_url_construction():
    """
    Verify that the RINEX observation file URL and path are constructed correctly.
    """
    # Mocking _batch_download to avoid downloads
    with patch("pytecgg.utils.download_rinex._batch_download") as mock_batch:
        station = "GROT"
        year = 2023
        doys = [1]
        out = Path("/tmp/gnss")

        download_obs_ring(station, year, doys, out)

        assert mock_batch.called

        tasks = mock_batch.call_args[0][0]
        url, path = tasks[0]

        assert "GROT00ITA" in url
        assert "2023001" in url
        assert path.name == "GROT00ITA_R_20230010000_01D_30S_MO.crx.gz"
        assert path.parent.name == "GROT"


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_obs_euref_url_construction_from_short_code(
    mock_get, mock_batch_download, tmp_path
):
    """
    Verify that a 4-character code is resolved against EUREF filenames and that
    the correct download task is constructed.
    """
    mock_response = MagicMock()
    mock_response.text = '<a href="BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz">link</a>'
    mock_get.return_value = mock_response

    download_obs_euref("BRUX", 2025, [1], tmp_path)

    assert mock_batch_download.called
    tasks = mock_batch_download.call_args[0][0]
    url, path = tasks[0]

    assert url == (
        "https://epncb.oma.be/pub/RINEX/2025/001/"
        "BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz"
    )
    assert path.name == "BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz"
    assert path.parent.name == "BRUX"


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_obs_euref_prefers_standard_r_filename(
    mock_get, mock_batch_download, tmp_path
):
    """
    Verify that the standard `_R_` filename is preferred when multiple EUREF
    filenames match the same station/day.
    """
    mock_response = MagicMock()
    mock_response.text = """
        <a href="BRUX00BEL_S_20250010000_01D_30S_MO.crx.gz">link</a>
        <a href="BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz">link</a>
    """
    mock_get.return_value = mock_response

    download_obs_euref("BRUX00BEL", 2025, [1], tmp_path)

    tasks = mock_batch_download.call_args[0][0]
    url, path = tasks[0]

    assert "BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz" in url
    assert path.name == "BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz"
    assert path.parent.name == "BRUX00BEL"


def test_download_file_cleanup_on_failure(tmp_path):
    """
    Verify that temporary files (.tmp) are removed in case of an error.
    """
    mock_session = MagicMock()
    # Emulating a network error (RequestException)
    mock_session.get.side_effect = RequestException("Connection refused")

    dest = tmp_path / "test_file.crx.gz"
    tmp_file = dest.with_suffix(".tmp")
    tmp_file.write_text("dati parziali")

    with pytest.raises(RequestException):
        _download_file(mock_session, "http://fake-url.com", dest)

    assert not tmp_file.exists(), "Temporary file not removed after download failure"
    assert not dest.exists(), "The destination file should not exist"


def test_download_file_success(tmp_path):
    """
    Verify that a file is downloaded successfully and temporary files are cleaned up.
    """
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_response.status_code = 200
    mock_session.get.return_value = mock_response

    dest = tmp_path / "success_file.crx.gz"

    _download_file(mock_session, "http://fake-url.com", dest)

    assert dest.exists()
    assert dest.read_bytes() == b"chunk1chunk2"
    assert not dest.with_suffix(".tmp").exists()


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_nav_bkg_scraping_priority_1_igs(mock_get, mock_batch_download, tmp_path):
    """
    Verify that the function prioritises IGS files when available and constructs
    the correct download task.
    """
    mock_response = MagicMock()
    mock_response.text = """
        <a href="BRDC00IGS_R_20250870000_01D_MN.rnx.gz">link</a>
        <a href="brdc0870.25p.gz">link</a>
    """
    mock_get.return_value = mock_response
    download_nav_bkg(2025, [87], tmp_path)

    assert mock_batch_download.called
    tasks = mock_batch_download.call_args[0][0]
    assert len(tasks) == 1

    url, path = tasks[0]
    assert "BRDC00IGS" in url
    assert path.name == "BRDC00IGS_R_20250870000_01D_MN.rnx.gz"


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_nav_bkg_scraping_priority_2_fallback(mock_get, mock_batch_download, tmp_path):
    """
    Verify that the function prioritises alternative centers (e.g. WRD) when IGS files
    are not available.
    """
    mock_response = MagicMock()
    mock_response.text = '<a href="BRDC00WRD_R_20201420000_01D_MN.rnx.gz">link</a>'
    mock_get.return_value = mock_response
    download_nav_bkg(2020, [142], tmp_path)
    tasks = mock_batch_download.call_args[0][0]
    url, path = tasks[0]
    assert "BRDC00WRD" in url
    assert path.name == "BRDC00WRD_R_20201420000_01D_MN.rnx.gz"


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_nav_bkg_scraping_priority_3_legacy(mock_get, mock_batch_download, tmp_path):
    """
    Verify the fallback to legacy .p.gz files for older years.
    """
    mock_response = MagicMock()
    mock_response.text = '<a href="brdc1190.14p.gz">link</a>'
    mock_get.return_value = mock_response
    download_nav_bkg(2014, [119], tmp_path)
    tasks = mock_batch_download.call_args[0][0]
    url, path = tasks[0]
    assert "brdc1190.14p.gz" in url
    assert path.name == "brdc1190.14p.gz"


@patch("pytecgg.utils.download_rinex._batch_download")
@patch("pytecgg.utils.download_rinex.requests.Session.get")
def test_nav_bkg_scraping_no_match(mock_get, mock_batch_download, tmp_path):
    """
    Verify that _batch_download is NOT called if there are no useful files.
    """
    mock_response = MagicMock()
    mock_response.text = (
        '<a href="brdc1190.14n.Z">link</a> <a href="brdc1190.14p.Z">link</a>'
    )
    mock_get.return_value = mock_response
    download_nav_bkg(2014, [119], tmp_path)
    mock_batch_download.assert_not_called()
