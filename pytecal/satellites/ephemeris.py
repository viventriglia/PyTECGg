from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from pytecal.satellites import CONSTELLATION_PARAMS, EPHEMERIS_FIELDS


def prepare_ephemeris(
    nav: dict[str, pl.DataFrame], constellation: str
) -> dict[str, dict[str, Any]]:
    """
    Prepare ephemeris data for specified constellation (GPS, BeiDou, etc.)

    Parameters:
    nav (dict): Dictionary containing navigation data from RINEX file
    constellation (str): Constellation name ('GPS', 'BeiDou', etc.)

    Returns:
    dict: Dictionary with prepared ephemeris data for each satellite
    """
    ephem_dict = {}

    if constellation not in nav:
        return ephem_dict

    params = CONSTELLATION_PARAMS[constellation]

    unique_sats = nav[constellation].get_column("sv").unique().to_list()

    for sat_ in unique_sats:
        ephe = nav[constellation].filter(pl.col("sv") == sat_)

        if ephe.is_empty():
            continue

        # Middle ephemeris
        mid_idx = len(ephe) // 2
        ephe_row = ephe[mid_idx]

        # Convert and normalize time
        ephe_time = parse_time(
            time_str=ephe_row["epoch"].item(),
            time_system=params["time_system"],
            time_offset=params["time_offset"],
        )

        # Calculate GPS week and seconds
        # gps_week, gps_sec = greg2gps(ephe_time)

        # Common fields for all constellations
        ephe_tab = {
            "constellation": constellation,
            "sv": sat_,
            "datetime": ephe_time,
            # "gps_week": gps_week,
            # "gps_seconds": gps_sec,
            "year": ephe_time.year,
            "month": ephe_time.month,
            "day": ephe_time.day,
            "hour": ephe_time.hour,
            "minute": ephe_time.minute,
            "second": ephe_time.second,
            "doy": int(ephe_time.strftime("%j")),
            "weekday": ephe_time.weekday(),
        }

        # Constellation-specific fields
        if constellation in EPHEMERIS_FIELDS:
            ephe_tab.update(
                {
                    field: ephe_row[field].item()
                    for field in EPHEMERIS_FIELDS[constellation]
                    if field in ephe_row
                }
            )

        fieldname = f"{params['prefix']}{sat_:}"
        ephem_dict[fieldname] = ephe_tab

    return ephem_dict


def parse_time(
    time_str: str, time_system: str = "UTC", time_offset: timedelta = timedelta(0)
) -> datetime:
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


def greg2gps(dt: datetime) -> tuple[int, float]:
    """
    Convert Gregorian date to GPS week and seconds

    Parameters:
    dt (datetime): Datetime object to convert

    Returns:
    tuple: (GPS week, GPS seconds)
    """
    # FIXME
    # GPS epoch is January 6, 1980
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch

    gps_week = delta.days // 7
    gps_seconds = delta.seconds + (delta.days % 7) * 86400 + delta.microseconds / 1e6

    return gps_week, gps_seconds
