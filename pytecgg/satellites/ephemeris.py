from datetime import datetime
from typing import Any

import polars as pl

from .constants import (
    CONSTELLATION_PARAMS,
    DEFAULT_LEAP_SECONDS_UTC_GPST,
    GPS_EPOCH,
    KEPLERIAN_POSITION_FIELDS,
)
from pytecgg.context import GNSSContext

Ephem = dict[str, dict[str, Any] | list[dict[str, Any]]]
"""Type alias for a dictionary containing processed ephemeris data.

It maps satellite IDs (e.g., 'G01') to their specific orbital parameters, 
handling both Keplerian (single dict) and state-vector (list of dicts) models.
"""


def _get_gps_time(dt: datetime) -> tuple[int, float]:
    """Convert UTC-aware datetime to GPS week and seconds of the week."""
    delta = dt - GPS_EPOCH
    gps_week = delta.days // 7
    gps_seconds = (delta.days % 7) * 86400 + delta.seconds + delta.microseconds / 1e6
    return gps_week, gps_seconds


def prepare_ephemeris(nav: dict[str, pl.DataFrame], ctx: GNSSContext) -> Ephem:
    """
    Prepare ephemeris data from RINEX navigation data using the settings in GNSSContext.

    This function processes multiple GNSS constellations and formats data based on
    their specific orbit propagation models:

    1.  Keplerian Orbits (GPS, Galileo, BeiDou):
        Selects a single representative ephemeris message (the central one) per satellite.

    2.  State-Vector Orbits (GLONASS):
        All available ephemeris messages are collected for the satellite. This is
        required because GLONASS messages contain instantaneous state vectors (position/
        velocity/acceleration) valid only for short periods (typically ± 15 minutes),
        requiring numerical integration from the closest epoch.

    Parameters
    ----------
    nav : dict[str, pl.DataFrame]
        Navigation data from RINEX, keyed by constellation name (e.g., 'GPS', 'GLONASS').
    ctx : GNSSContext
        Execution context containing target systems and settings.

    Returns
    -------
    Ephem
        Dictionary keyed by satellite ID (e.g., 'G01', 'R09').
        Values are a single dict for Keplerian systems or a list of dicts for GLONASS.
    """
    ephem_dict: Ephem = {}
    inverse_map = ctx.symbol_to_name
    leap_seconds = (
        ctx.leap_seconds if ctx.leap_seconds is not None else DEFAULT_LEAP_SECONDS_UTC_GPST
    )

    for symbol_ in ctx.systems:
        const_name = inverse_map.get(symbol_)
        if const_name not in nav:
            continue

        params = CONSTELLATION_PARAMS[const_name]
        is_state_vector = symbol_ == "R"

        unique_svs = nav[const_name]["sv"].unique().to_list()
        for sat_id_ in unique_svs:
            normalised_sat_id = f"{symbol_}{int(sat_id_):02d}"
            sat_data = nav[const_name].filter(pl.col("sv") == sat_id_)

            if sat_data.is_empty():
                continue

            if is_state_vector:
                # GLONASS: Logic for state-vector models requires the full history
                sat_data = sat_data.sort("epoch")

                # Extract FDMA channel
                channel_val = sat_data.get_column("channel")[0]
                ctx.glonass_channels[normalised_sat_id] = channel_val

                sat_ephems_list = []
                for row in sat_data.to_dicts():
                    ephe_time = row["epoch"]
                    gps_week, gps_sec = _get_gps_time(ephe_time)

                    ephem = {
                        "constellation": const_name,
                        "sv": normalised_sat_id,
                        "datetime": ephe_time,
                        "gps_week": gps_week,
                        "gps_seconds": gps_sec,
                        "leap_seconds": leap_seconds,
                        **{field: row.get(field) for field in params.fields},
                    }
                    sat_ephems_list.append(ephem)

                ephem_dict[normalised_sat_id] = sat_ephems_list

            else:
                # Keplerian models (GPS, Galileo, BeiDou): keep every valid record
                # so the caller can pick the one nearest to the observation epoch.
                # Broadcast Keplerian parameters are only valid for ~±2 h around toe,
                # so reusing one record across a multi-hour file causes 50-200 m bias.
                # Validate only on fields required for position computation, not all EPHEMERIS_FIELDS
                # (e.g. bgdE5bE1 for Galileo may be absent for I/NAV-only satellites).
                position_fields = [f for f in KEPLERIAN_POSITION_FIELDS if f in sat_data.columns]
                valid_data = sat_data.drop_nulls(subset=position_fields).sort("epoch")
                if valid_data.is_empty():
                    continue

                sat_ephems_list = []
                for row in valid_data.to_dicts():
                    ephe_time = row["epoch"]
                    gps_week, gps_sec = _get_gps_time(ephe_time)

                    ephem = {
                        "constellation": const_name,
                        "sv": normalised_sat_id,
                        "datetime": ephe_time,
                        "gps_week": gps_week,
                        "gps_seconds": gps_sec,
                        "leap_seconds": leap_seconds,
                        **{field: row.get(field) for field in params.fields},
                    }
                    sat_ephems_list.append(ephem)

                ephem_dict[normalised_sat_id] = sat_ephems_list

    return ephem_dict
