import polars as pl
from datetime import datetime, timedelta
from typing import Any


def prepare_bds(nav: dict[str, pl.DataFrame]) -> dict[str, dict[str, Any]]:
    """
    Prepare BeiDou ephemeris

    Parameters:
    nav (dict): Dictionary containing BeiDou navigation data from RINEX file

    Returns:
    dict: Dictionary with prepared ephemeris data for each BeiDou satellite
    """
    BDSNav = {}

    if "BeiDou" not in nav:
        return BDSNav

    unique_sats = nav["BeiDou"].get_column("sv").unique().to_list()

    for sat_ in unique_sats:
        ephe = nav["BeiDou"].filter(pl.col("sv") == sat_)

        if ephe.is_empty():
            continue

        # Take the middle ephemeris
        mid_idx = len(ephe) // 2
        ephe_row = ephe[mid_idx]

        # Convert time to datetime object
        ephe_time = ephe_row["epoch"][0]
        if isinstance(ephe_time, str):
            # Handle BDT time (UTC+8)
            if "BDT" in ephe_time:
                dt_str = ephe_time.split(" BDT")[0].strip()
                try:
                    naive_dt = datetime.fromisoformat(dt_str)
                    # Apply UTC+8 offset (BeiDou Time)
                    ephe_time = naive_dt - timedelta(hours=8)
                except ValueError as e:
                    raise ValueError(
                        f"Failed to parse BDT time string '{ephe_time}': {e}"
                    )
            else:
                try:
                    ephe_time = datetime.fromisoformat(ephe_time)
                except ValueError as e:
                    raise ValueError(f"Failed to parse time string '{ephe_time}': {e}")
        elif isinstance(ephe_time, (int, float)):
            # Handle potential timestamp cases
            ephe_time = datetime.fromtimestamp(ephe_time)

        # GPS week and seconds conversion
        # gps_week, gps_sec = greg2gps(ephe_time)

        # Create ephemeris dictionary
        ephe_tab = {
            "year": ephe_time.year,
            "month": ephe_time.month,
            "day": ephe_time.day,
            "hour": ephe_time.hour,
            "minute": ephe_time.minute,
            "second": ephe_time.second,
            # "GPSweek": gps_week,
            # "GPSsec": gps_sec,
            "weekday": ephe_time.weekday() + 1,
            "doy": int(ephe_time.strftime("%j")),  # Day of year
            "datenum": to_datenum(ephe_time),  # MATLAB datenum equivalent
            "SVClockBias": ephe_row["clock_bias"],
            "SVClockDrift": ephe_row["clock_drift"],
            "SVClockDriftRate": ephe_row["clock_drift_rate"],
            "AODE": ephe_row["aode"],
            "Crs": ephe_row["crs"],
            "Delta_n": ephe_row["deltaN"],
            "M0": ephe_row["m0"],
            "Cuc": ephe_row["cuc"],
            "Eccentricity": ephe_row["e"],
            "Cus": ephe_row["cus"],
            "sqrtA_squared": ephe_row["sqrta"] ** 2,
            "Toe": ephe_row["toe"],
            "Cic": ephe_row["cic"],
            "OMEGA0": ephe_row["omega0"],
            "Cis": ephe_row["cis"],
            "i0": ephe_row["i0"],
            "Crc": ephe_row["crc"],
            "omega": ephe_row["omega"],
            "OMEGA_DOT": ephe_row["omegaDot"],
            "IDOT": ephe_row["idot"],
            # "BRDCOrbit5Spare2": ephe_row["BRDCOrbit5Spare2"],
            # "BDTWeek": ephe_row["BDTWeek"],
            # "BRDCOrbit5Spare4": ephe_row["BRDCOrbit5Spare4"],
            "SVAccuracy": ephe_row["accuracy"],
            # "SatH1": ephe_row["SatH1"],
            "TGD1": ephe_row["tgd1b1b3"],
            "TGD2": ephe_row["tgd2b2b3"],
            # "TransmissionTime": ephe_row["TransmissionTime"],
            "AODC": ephe_row["aodc"],
            # "BRDCOrbit7Spare3": ephe_row["BRDCOrbit7Spare3"],
            # "BRDCOrbit7Spare4": ephe_row["BRDCOrbit7Spare4"],
        }

        # Create field name (e.g., 'C01' for satellite 1)
        fieldname = f"C{sat_}"
        BDSNav[fieldname] = ephe_tab

    return BDSNav


def greg2gps(dt: datetime) -> tuple[int, float]:
    """
    Convert Gregorian date to GPS week and seconds

    Parameters:
    dt (datetime): Datetime object to convert

    Returns:
    tuple: (GPS week, GPS seconds)
    """
    # GPS epoch is January 6, 1980
    gps_epoch = datetime(1980, 1, 6)
    delta = dt - gps_epoch

    gps_week = delta.days // 7
    gps_seconds = delta.seconds + (delta.days % 7) * 86400 + delta.microseconds / 1e6

    return gps_week, gps_seconds


def to_datenum(dt: datetime) -> float:
    """
    Convert datetime to MATLAB datenum equivalent

    Parameters:
    dt (datetime): Datetime object to convert

    Returns:
    float: MATLAB datenum value
    """
    return (
        366
        + dt.toordinal()
        + (dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0)
    )
