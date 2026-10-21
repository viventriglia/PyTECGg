import warnings
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


# Broadcast ephemeris cadence (seconds) per Keplerian constellation, used to
# detect a known rinex-crate parsing bug at the BDT/GST week rollover.
_KEPLERIAN_CADENCE_S: dict[str, float] = {
    "BEIDOU": 3600.0,
    "GALILEO": 600.0,
    "GPS": 7200.0,
}


def _drop_week_rollover_bug(
    valid_data: pl.DataFrame, const_name: str, sv_id: str
) -> pl.DataFrame:
    """Drop ephemeris records corrupted by the rinex-crate (<=0.22) toe bug.

    The crate misreads the toe field at the very first record of a new BDT/GST
    week (toc seconds-of-week == 0), returning roughly one broadcast-cadence
    step instead of 0.0. The misparse is not always exactly one cadence step:
    GPS records at the 2021-01-03 rollover were observed with toe in
    {7168, 7184, 7200} where the file says 0.0 (cadence 7200). The corrupted
    record's Keplerian propagation is off by ~one cadence step, producing
    ~26000 km position errors for GPS MEO, ~13000 km for BeiDou MEO and
    ~2300 km for Galileo MEO.

    Detection: at a genuine week-start record (toc at Sunday 00:00:00) the true
    toe must be ~0, so any toe in the upper half of the cadence interval
    (>= cadence/2) is a misparse. We drop such records and warn the caller;
    propagation falls back to the next valid record.
    """
    cadence = _KEPLERIAN_CADENCE_S.get(const_name)
    if cadence is None or "toe" not in valid_data.columns:
        return valid_data

    # At a true week-start record toe ~ 0; a misparse lands near a full cadence
    # step (observed slightly below, e.g. 7168/7184 for GPS's 7200). Half the
    # cadence is a safe separator: a legitimate first-of-week toe is nowhere
    # near cadence/2.
    mask_bad = (
        (pl.col("epoch").dt.weekday() == 7)
        & (pl.col("epoch").dt.hour() == 0)
        & (pl.col("epoch").dt.minute() == 0)
        & (pl.col("epoch").dt.second() == 0)
        & (pl.col("toe") >= cadence / 2)
    )
    bad_count = valid_data.filter(mask_bad).height
    if bad_count == 0:
        return valid_data

    bad_toes = sorted(
        v for v in valid_data.filter(mask_bad)["toe"].unique().to_list()
        if v is not None
    )
    warnings.warn(
        f"Dropped {bad_count} {const_name} ephemeris record(s) for {sv_id} "
        f"at BDT/GST week rollover: rinex-crate toe parsing bug returned "
        f"toe={bad_toes}s where the file says 0.0s. Affected epoch(s) "
        f"will fall back to the next valid record (~{cadence/60:.0f} min "
        f"later) for position propagation.",
        stacklevel=3,
    )
    return valid_data.filter(~mask_bad)


def prepare_ephemeris(nav: dict[str, pl.DataFrame], ctx: GNSSContext) -> Ephem:
    """
    Prepare ephemeris data from RINEX navigation data using the settings in GNSSContext.

    This function processes multiple GNSS constellations and formats data based on
    their specific orbit propagation models:

    1.  Keplerian Orbits (GPS, Galileo, BeiDou):
        Keeps every valid broadcast record per satellite. Position computation
        later picks the record nearest to the observation epoch, since broadcast
        Keplerian parameters are only valid for ~±2 h around toe. Records
        corrupted by the rinex-crate week-rollover bug are filtered out (see
        ``_drop_week_rollover_bug``).

    2.  State-Vector Orbits (GLONASS):
        All available ephemeris messages are collected for the satellite. This is
        required because GLONASS messages contain instantaneous state vectors (position/
        velocity/acceleration) valid only for short periods (typically ± 15 minutes),
        requiring numerical integration from the closest epoch.

    Each record carries the ``leap_seconds`` value taken from
    ``ctx.leap_seconds`` (populated from the RINEX nav header via
    :func:`pytecgg.parsing.read_rinex_nav_header`) or
    ``DEFAULT_LEAP_SECONDS_UTC_GPST`` when the context does not provide one.

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
                valid_data = _drop_week_rollover_bug(valid_data, const_name, normalised_sat_id)
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
