from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from pytecgg.satellites import CONSTELLATION_PARAMS


def _parse_time(time_str: str, time_system: str, time_offset: timedelta) -> datetime:
    """
    Parse RINEX time string with time system awareness

    Parameters:
    time_str (str): Time string from RINEX file
    time_system (str): Time system identifier ('GPST', 'BDT', etc.)
    time_offset (timedelta): Offset to apply for conversion to UTC

    Returns:
    datetime: Timezone-aware datetime in UTC
    """
    if isinstance(time_str, str):
        # Remove time system suffix if present
        clean_str = time_str.split(f" {time_system}")[0].strip()

        try:
            # Parse naive datetime
            dt = datetime.fromisoformat(clean_str)

            # Apply time system offset and convert to UTC
            if time_offset:
                dt = dt - time_offset

            # Make timezone-aware (UTC)
            return dt.replace(tzinfo=timezone.utc)

        except ValueError as e:
            raise ValueError(f"Failed to parse time string '{time_str}': {e}")

    raise TypeError(f"Unsupported time format: {type(time_str)}")


def _greg2gps(dt: datetime) -> tuple[int, float]:
    """
    Convert Gregorian date to GPS week and seconds

    Parameters:
    dt (datetime): Datetime object to convert

    Returns:
    tuple: (GPS week, GPS seconds)
    """
    epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    delta = dt - epoch
    return (
        delta.days // 7,
        (delta.days % 7) * 86400 + delta.seconds + delta.microseconds / 1e6,
    )


def prepare_ephemeris(
    nav: dict[str, pl.DataFrame], constellation: str
) -> dict[str, dict[str, Any]]:
    """
    Prepare ephemeris data for specified constellation (GPS, BeiDou, etc.)

    Parameters:
    nav (dict): Dictionary of DataFrames containing navigation data from RINEX file (keyed by constellation)
    constellation (str): Constellation name ('GPS', 'BeiDou', etc.)

    Returns:
    dict: Dictionary with prepared ephemeris data for each satellite
    """
    if constellation not in CONSTELLATION_PARAMS or constellation not in nav:
        return {}

    params = CONSTELLATION_PARAMS[constellation]
    ephem_dict = {}

    for sat in nav[constellation]["sv"].unique().to_list():
        sat_data = nav[constellation].filter(pl.col("sv") == sat)
        if sat_data.is_empty():
            continue

        # Middle ephemeris
        ephe_row = sat_data[len(sat_data) // 2]

        ephe_time = _parse_time(
            ephe_row["epoch"].item(), params.time_system, params.time_offset
        )
        gps_week, gps_sec = _greg2gps(ephe_time)

        # Base structure
        ephem = {
            "constellation": constellation,
            "sv": sat,
            "datetime": ephe_time,
            "gps_week": gps_week,
            "gps_seconds": gps_sec,
            **{
                field: ephe_row[field].item()
                for field in params.fields
                if field in ephe_row
            },
        }

        ephem_dict[f"{params.prefix}{int(sat):02d}"] = ephem

    return ephem_dict
