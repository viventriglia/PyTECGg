import bisect
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from .constants import (
    CONSTELLATION_PARAMS,
    GNSS_CONSTANTS,
    GPS_EPOCH,
    KEPLERIAN_POSITION_FIELDS,
    LEAP_SECONDS_UTC_GPST,
    LEAP_SECONDS_VALID_UNTIL,
)
from pytecgg.context import GNSSContext

GLONASS_SLOT_TOLERANCE_DEG = 5.0
"""Largest accepted offset of a GLONASS record from its satellite's orbital slot.
Neighbouring slots are 45 deg apart along the orbit and the planes 120 deg apart."""

_GLONASS_DRACONIC_PERIOD_S = 40544.0
"""Nominal GLONASS draconic period (11 h 15 min 44 s)."""

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


def _glonass_slot_angles(sat_data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Right ascension of the ascending node and along-track phase (u - n t) of each
    record, in degrees. Both are constant for one satellite over days; the inertial
    frame is ECEF rotated by we * t, and its unknown constant offset cancels."""
    we = GNSS_CONSTANTS["GLONASS"].we
    t = sat_data["epoch"].dt.epoch("us").to_numpy() / 1e6
    r = sat_data.select("satPosX", "satPosY", "satPosZ").to_numpy()
    v = sat_data.select("velX", "velY", "velZ").to_numpy()
    v = v + np.cross([0.0, 0.0, we], r)  # rotating-frame velocity -> inertial
    cos_t, sin_t = np.cos(we * t), np.sin(we * t)

    def to_inertial(x):
        return np.column_stack(
            [x[:, 0] * cos_t - x[:, 1] * sin_t, x[:, 0] * sin_t + x[:, 1] * cos_t, x[:, 2]]
        )

    r, v = to_inertial(r), to_inertial(v)
    h = np.cross(r, v)
    raan = np.arctan2(h[:, 0], -h[:, 1])
    node = np.column_stack([np.cos(raan), np.sin(raan), np.zeros_like(raan)])
    in_plane = np.cross(h / np.linalg.norm(h, axis=1, keepdims=True), node)
    u = np.arctan2((r * in_plane).sum(axis=1), (r * node).sum(axis=1))
    return np.degrees(raan), np.degrees(u - 2 * np.pi * t / _GLONASS_DRACONIC_PERIOD_S)


def _offset_from_median_deg(angle: np.ndarray) -> np.ndarray:
    """Absolute offset of each angle from their circular median, in degrees."""
    wrap = lambda a: (a + 180.0) % 360.0 - 180.0  # noqa: E731
    centre = np.degrees(np.angle(np.exp(1j * np.radians(angle)).mean()))
    centre += np.median(wrap(angle - centre))
    return np.abs(wrap(angle - centre))


def _drop_mislabelled_glonass_records(sat_data: pl.DataFrame, sat_id: str) -> pl.DataFrame:
    """
    Drop GLONASS records whose state vector is not on this satellite's orbit.

    Merged broadcast files sometimes carry another satellite's message under this
    slot (BRDC00IGS_R on 2024-01-01/02 does, for most of the constellation, for up
    to several hours). Its state vector is a valid orbit, just not this satellite's,
    so positions jump to another slot for as long as it is used.

    Every record gives the satellite's orbital plane (right ascension of the
    ascending node) and its phase along the orbit; both stay put for days. A record
    more than ``GLONASS_SLOT_TOLERANCE_DEG`` from the satellite's median in either
    is rejected, so this holds while most of the satellite's records are its own.
    Of several records for one epoch, the one nearest the median slot is kept.

    The clock offset is not used: in the same files many records carry a wrong
    clock offset with a correct orbit, and TEC does not need the satellite clock.
    """
    if sat_data.height < 3:
        return sat_data
    raan, phase = _glonass_slot_angles(sat_data)
    checked = sat_data.with_columns(
        _off=np.maximum(_offset_from_median_deg(raan), _offset_from_median_deg(phase))
    )
    on_slot = checked.filter(pl.col("_off") <= GLONASS_SLOT_TOLERANCE_DEG)
    if n_dropped := checked.height - on_slot.height:
        warnings.warn(
            f"[GLONASS] {sat_id}: dropped {n_dropped} broadcast record(s) more than "
            f"{GLONASS_SLOT_TOLERANCE_DEG:g} deg off the satellite's orbital slot "
            "(likely another satellite's message under this slot).",
            stacklevel=3,
        )
    return (
        on_slot.sort("epoch", "_off")
        .unique("epoch", keep="first", maintain_order=True)
        .drop("_off")
    )


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
        requiring numerical integration from the closest epoch. Records off the
        satellite's orbital slot are dropped as mislabelled
        (see :func:`_drop_mislabelled_glonass_records`).

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
                sat_data = _drop_mislabelled_glonass_records(sat_data.sort("epoch"), normalised_sat_id)
                if sat_data.is_empty():
                    continue
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
