import re
import requests
import logging
from pathlib import Path
from typing import Iterable
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

USER_AGENT = "GNSS-TEC-Researcher-Python"
EUREF_BASE_URL = "https://epncb.oma.be/pub/RINEX"


def _download_file(
    session: requests.Session, url: str, dest: Path, timeout: int = 15
) -> None:
    """
    Perform a streaming download of a single file.

    This function implements an atomic download strategy: data is first written
    to a temporary file (.tmp) and renamed to the final destination only upon
    successful completion to prevent file corruption.

    Parameters
    ----------
    session : requests.Session
        An active HTTP session for connection pooling
    url : str
        The full URL of the file to be downloaded
    dest : Path
        The final local destination path (including filename)
    timeout : int, optional
        Maximum time in seconds to wait for a server response (default: 15)

    Raises
    ------
    RequestException
        If an HTTP error occurs (e.g., 404, 500) or network timeout
    OSError
        If there are issues writing to the disk or managing permissions
    """
    tmp_path = dest.with_suffix(".tmp")
    try:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        tmp_path.rename(dest)
        logger.info(f"Successfully downloaded: {dest.name}")

    except (RequestException, OSError) as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"Failed to download {url}: {e}")
        raise


def _batch_download(tasks: Iterable[tuple[str, Path]]) -> None:
    """
    Orchestrate the bulk download of multiple files.

    Handles HTTP session management, directory verification, and skips
    files that are already present on the local file system.

    Parameters
    ----------
    tasks : Iterable[Tuple[str, Path]]
        A sequence of tuples where each tuple contains (source_url, destination_path).
    """
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        for url, filepath in tasks:
            if filepath.exists():
                continue

            filepath.parent.mkdir(parents=True, exist_ok=True)

            try:
                _download_file(session, url, filepath)
            except Exception:
                continue


def download_obs_ring(
    station_code: str, year: int, doys: list[int], output_path: Path
) -> None:
    """
    Download RINEX observation files (Hatanaka crx.gz) from the INGV RING server.

    Automatically handles station code conversion (e.g., converts 4-character
    codes like 'GRO2' to 'GRO200ITA') and organizes files into station-specific
    subdirectories.

    Parameters
    ----------
    station_code : str
        The station identifier (4 or 9 characters). Example: 'GRO2' or 'GRO200ITA'.
    year : int
        The observation year (e.g., 2023).
    doys : list[int]
        A list of Days Of Year (DOY), e.g., [1, 2, 3].
    output_path : Path
        The root directory where the files will be saved.

    Notes
    -----
    Data Source: [https://webring.gm.ingv.it:44324/rinex/RING](https://webring.gm.ingv.it:44324/rinex/RING)
    """
    station_full = station_code.upper()
    if len(station_code) == 4:
        station_full = f"{station_code.upper()}00ITA"

    base_url = "https://webring.gm.ingv.it:44324/rinex/RING"
    station_dir = output_path / station_code.upper()

    tasks = []
    for doy in doys:
        doy_str = f"{doy:03d}"
        filename = f"{station_full}_R_{year}{doy_str}0000_01D_30S_MO.crx.gz"
        url = f"{base_url}/{year}/{doy_str}/{filename}"
        tasks.append((url, station_dir / filename))

    _batch_download(tasks)


def _select_euref_obs_filename(
    available_files: list[str], station_code: str, year: int, doy: int
) -> str | None:
    """
    Select the best matching EUREF observation filename for a station and DOY.

    Parameters
    ----------
    available_files : list[str]
        The list of filenames discovered in the remote EUREF directory.
    station_code : str
        The station identifier, typically 4 or 9 characters.
    year : int
        The observation year.
    doy : int
        The Day Of Year.

    Returns
    -------
    str | None
        The preferred matching filename, or None if no compatible file is found.

    Notes
    -----
    If a 4-character station code is provided, this function matches any EPN
    long-name filename starting with that prefix. If multiple candidates are
    present, files using the standard `_R_` naming are preferred.
    """
    station_prefix = station_code.upper()
    doy_str = f"{doy:03d}"
    expected_suffix = f"_{year}{doy_str}0000_01D_30S_MO.crx.gz"

    candidates = [
        filename
        for filename in available_files
        if filename.startswith(station_prefix) and filename.endswith(expected_suffix)
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda name: ("_R_" not in name, name))
    return candidates[0]


def download_obs_euref(
    station_code: str, year: int, doys: list[int], output_path: Path
) -> None:
    """
    Download RINEX observation files (Hatanaka crx.gz) from the EUREF EPN server.

    EUREF station long names cannot always be inferred from a 4-character code
    alone. If a short code is provided, the function inspects the remote daily
    directory and selects the matching EPN long-name filename. Downloaded files
    are organized into station-specific subdirectories.

    Parameters
    ----------
    station_code : str
        The station identifier (4 or 9 characters). Example: 'BRUX' or 'BRUX00BEL'.
    year : int
        The observation year (e.g., 2023).
    doys : list[int]
        A list of Days Of Year (DOY), e.g., [1, 2, 3].
    output_path : Path
        The root directory where the files will be saved.

    Notes
    -----
    Data Source: [https://epncb.oma.be/pub/RINEX](https://epncb.oma.be/pub/RINEX)
    """
    station_dir = output_path / station_code.upper()

    tasks = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        for doy in doys:
            doy_str = f"{doy:03d}"
            dir_url = f"{EUREF_BASE_URL}/{year}/{doy_str}/"

            try:
                resp = session.get(dir_url, timeout=15)
                resp.raise_for_status()
                available_files = re.findall(r'href="([^"/]+\.crx\.gz)"', resp.text)
            except Exception as e:
                logger.error(f"Could not access EUREF directory {dir_url}: {e}")
                continue

            target_filename = _select_euref_obs_filename(
                available_files, station_code, year, doy
            )

            if target_filename is None:
                logger.warning(
                    f"No compatible OBS file found on EUREF server for "
                    f"station {station_code.upper()} and {year}/DOY {doy_str}."
                )
                continue

            url = f"{dir_url}{target_filename}"
            tasks.append((url, station_dir / target_filename))

    if tasks:
        _batch_download(tasks)


def download_nav_bkg(year: int, doys: list[int], output_path: Path) -> None:
    """
    Download global navigation RINEX files (BRDC) from the BKG server.

    BRDC files contain multi-constellation navigation messages aggregated
    from the global IGS station network. The function dynamically scrapes
    the BKG HTTP directories to find the best available navigation file format
    (modern IGS, other RINEX 3 centers like WRD/DLR, or legacy .gz formats)
    for each given DOY.

    Parameters
    ----------
    year : int
        The observation year (e.g., 2023).
    doys : list[int]
        A list of Days Of Year (DOY).
    output_path : Path
        The directory where the navigation files will be saved.

    Notes
    -----
    Data Source: [https://igs.bkg.bund.de/root_ftp/IGS/BRDC](https://igs.bkg.bund.de/root_ftp/IGS/BRDC)
    """
    base_url = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC"

    tasks = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})

        for doy in doys:
            doy_str = f"{doy:03d}"
            yy = str(year)[-2:]
            dir_url = f"{base_url}/{year}/{doy_str}/"

            try:
                resp = session.get(dir_url, timeout=15)
                resp.raise_for_status()
                # Extract only .gz files from the HTML directory listing
                available_files = re.findall(r'href="([^"/]+\.gz)"', resp.text)
            except Exception as e:
                logger.error(f"Could not access BKG directory {dir_url}: {e}")
                continue

            # Priority list for file formats
            patterns = [
                r"^BRDC00IGS_R_.*_MN\.rnx\.gz$",  # 1: Official IGS
                r"^BRDC00.*_R_.*_MN\.rnx\.gz$",  # 2: Other RINEX 3 centers (WRD, DLR, etc.)
                rf"^brdc{doy_str}0\.{yy}p\.gz$",  # 3: Legacy Mixed .p.gz
            ]

            target_filename = None
            for pattern in patterns:
                for file in available_files:
                    if re.match(pattern, file):
                        target_filename = file
                        break
                if target_filename:
                    break

            if target_filename:
                url = f"{dir_url}{target_filename}"
                tasks.append((url, output_path / target_filename))
            else:
                logger.warning(
                    f"No compatible NAV file found on BKG server for {year}/DOY {doy_str}."
                )

    if tasks:
        _batch_download(tasks)
