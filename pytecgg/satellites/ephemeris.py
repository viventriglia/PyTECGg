import bisect
import warnings
from datetime import datetime
from typing import Any

import polars as pl

from .constants import (
    CONSTELLATION_PARAMS,
    GPS_EPOCH,
    KEPLERIAN_POSITION_FIELDS,
    LEAP_SECONDS_UTC_GPST,
    LEAP_SECONDS_VALID_UNTIL,
)
from pytecgg.context import GNSSContext

Ephem = dict[str, dict[str, Any] | list[dict[str, Any]]]
"""Type alias for a dictionary containing processed ephemeris data.

It maps satellite IDs (e.g., 'G01') to their orbital parameters. Values are
a list of broadcast records for every constellation (Keplerian and
state-vector); downstream code picks the record nearest the observation
epoch. A plain dict is still accepted for legacy / hand-built ephemerides.
"""


def _get_gps_time(dt: datetime) -> tuple[int, float]:
    """Convert UTC-aware datetime to GPS week and seconds of the week."""
    delta = dt - GPS_EPOCH
    gps_week = delta.days // 7
    gps_seconds = (delta.days % 7) * 86400 + delta.seconds + delta.microseconds / 1e6
    return gps_week, gps_seconds


def _leap_seconds_at(epoch: datetime) -> int:
    """UTC -> GPST leap seconds at `epoch`, from the table in constants.py."""
    day = epoch.date()
    if day >= LEAP_SECONDS_VALID_UNTIL:
        warnings.warn(
            f"Leap seconds for nav records from {LEAP_SECONDS_VALID_UNTIL} on are "
            "past the built-in table and the nav header has no LEAP SECONDS value: "
            "assuming no new leap second. Set GNSSContext.leap_seconds if one was "
            "announced.",
            stacklevel=3,
        )
    i = bisect.bisect_right([start for start, _ in LEAP_SECONDS_UTC_GPST], day)
    return LEAP_SECONDS_UTC_GPST[i - 1][1] if i else 0


def prepare_ephemeris(nav: dict[str, pl.DataFrame], ctx: GNSSContext) -> Ephem:
    """
    Prepare ephemeris data from RINEX navigation data using the settings in GNSSContext.

    This function processes multiple GNSS constellations and formats data based on
    their specific orbit propagation models:

    1.  Keplerian Orbits (GPS, Galileo, BeiDou):
        Keeps every valid broadcast record per satellite. Position computation
        later picks the record nearest to the observation epoch, since broadcast
        Keplerian parameters are only valid for ~±2 h around toe.

    2.  State-Vector Orbits (GLONASS):
        All available ephemeris messages are collected for the satellite. This is
        required because GLONASS messages contain instantaneous state vectors (position/
        velocity/acceleration) valid only for short periods (typically ± 15 minutes),
        requiring numerical integration from the closest epoch.

    Each record carries a ``leap_seconds`` value (UTC -> GPST), taken from
    ``ctx.leap_seconds`` if set, else from the nav header LEAP SECONDS field
    (the ``leap_seconds`` column added by :func:`pytecgg.parsing.read_rinex_nav`),
    else from the leap-second table for the record date.

    Parameters
    ----------
    nav : dict[str, pl.DataFrame]
        Navigation data from RINEX, keyed by constellation name (e.g., 'GPS', 'GLONASS').
    ctx : GNSSContext
        Execution context containing target systems and settings.

    Returns
    -------
    Ephem
        Dictionary keyed by satellite ID (e.g., 'G01', 'R09'). Values are a
        list of broadcast records (one dict per valid message).
    """
    ephem_dict: Ephem = {}
    inverse_map = ctx.symbol_to_name

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
                # GLONASS: state-vector propagation needs the full history
                sat_data = sat_data.sort("epoch")
                ctx.glonass_channels[normalised_sat_id] = sat_data.get_column("channel")[0]
            else:
                # Keplerian models (GPS, Galileo, BeiDou): keep every valid record
                # so the caller can pick the one nearest to the observation epoch.
                # Broadcast Keplerian parameters are only valid for ~±2 h around toe,
                # so reusing one record across a multi-hour file causes 50-200 m bias.
                # Validate only on fields required for position computation, not all EPHEMERIS_FIELDS
                # (e.g. bgdE5bE1 for Galileo may be absent for I/NAV-only satellites).
                position_fields = [f for f in KEPLERIAN_POSITION_FIELDS if f in sat_data.columns]
                sat_data = sat_data.drop_nulls(subset=position_fields)
                if "source" in sat_data.columns:
                    # Galileo files repeat each record per data source (F/NAV 258, I/NAV
                    # 513/516/517) with the same orbit: keep I/NAV E1+E5b (517) if present
                    sat_data = sat_data.sort("source", descending=True).unique(
                        "epoch", keep="first"
                    )
                sat_data = sat_data.sort("epoch")
                if sat_data.is_empty():
                    continue

            sat_ephems_list = []
            for row in sat_data.to_dicts():
                gps_week, gps_sec = _get_gps_time(row["epoch"])
                leap_seconds = (
                    ctx.leap_seconds
                    if ctx.leap_seconds is not None
                    else row.get("leap_seconds")
                )
                if leap_seconds is None:
                    leap_seconds = _leap_seconds_at(row["epoch"])
                sat_ephems_list.append(
                    {
                        "constellation": const_name,
                        "sv": normalised_sat_id,
                        "datetime": row["epoch"],
                        "gps_week": gps_week,
                        "gps_seconds": gps_sec,
                        "leap_seconds": leap_seconds,
                        **{field: row.get(field) for field in params.fields},
                    }
                )
            ephem_dict[normalised_sat_id] = sat_ephems_list

    return ephem_dict
