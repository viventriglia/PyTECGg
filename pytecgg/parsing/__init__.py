import gzip
from pathlib import Path
from typing import Union

import polars as pl

from ..pytecgg import (
    read_rinex_obs as _read_rinex_obs,
    read_rinex_nav as _read_rinex_nav,
)

__all__ = ["read_rinex_obs", "read_rinex_nav", "read_rinex_nav_header"]


def read_rinex_obs(
    path: Union[str, Path],
) -> tuple[pl.DataFrame, tuple[float, float, float], str]:
    """
    Parses a RINEX observation file and returns the extracted observation data as a Polars DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the RINEX observation file (.rnx, .crx, or .gz).

    Returns
    -------
    tuple
        - pl.DataFrame: DataFrame with columns 'epoch', 'sv', 'observable', 'value'
        - tuple[float, float, float]: Receiver's position in ECEF coordinates (meters)
        - str: RINEX version
    """
    path_str = str(path)
    df, rec_pos, rinex_version = _read_rinex_obs(path_str)
    return (
        df.with_columns(pl.col("epoch").dt.replace_time_zone("UTC")),
        rec_pos,
        rinex_version,
    )


def _open_rinex_text(path: Union[str, Path]):
    """Open a RINEX file as text, transparently handling .gz."""
    p = Path(path)
    if p.suffix.lower() == ".gz":
        return gzip.open(p, "rt", encoding="ascii", errors="replace")
    return open(p, "r", encoding="ascii", errors="replace")


def read_rinex_nav_header(path: Union[str, Path]) -> dict[str, Union[int, None]]:
    """Read a few selected fields from a RINEX nav file header.

    The compiled parser does not surface the header. This pure-Python scan
    reads only the lines up to ``END OF HEADER`` and extracts:

    - ``leap_seconds`` : int or None
        Value of the ``LEAP SECONDS`` header field (UTC -> GPST offset). ``None``
        if the field is absent.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the RINEX navigation file (.rnx, .nav, or .gz).

    Returns
    -------
    dict
        Header fields. Currently ``{"leap_seconds": int | None}``.
    """
    leap_seconds: Union[int, None] = None
    with _open_rinex_text(path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").rstrip("\r")
            # RINEX header label occupies columns 60-80 (0-indexed 60:80).
            label = line[60:80].strip() if len(line) >= 60 else ""
            if label == "LEAP SECONDS":
                # The first numeric token in the data area is the leap-second count.
                # RINEX 3 may carry up to four integer fields (LS, ΔLS_future,
                # week, day) — we only need the first.
                token = line[:60].split()
                if token:
                    try:
                        leap_seconds = int(token[0])
                    except ValueError:
                        leap_seconds = None
            if label == "END OF HEADER":
                break
    return {"leap_seconds": leap_seconds}


def read_rinex_nav(path: Union[str, Path]) -> dict[str, pl.DataFrame]:
    """
    Parses a RINEX navigation file into a dictionary of DataFrames.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the RINEX navigation file.

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary keyed by constellation (e.g., 'GPS'), containing
        DataFrames with 'epoch' as datetime[μs, UTC] and orbital parameters.
    """
    path_str = str(path)
    nav_dict = _read_rinex_nav(path_str)
    return {
        const: df.with_columns(pl.col("epoch").dt.replace_time_zone("UTC"))
        for const, df in nav_dict.items()
    }
