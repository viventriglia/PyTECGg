import datetime
import warnings
from typing import Any, Union

import numpy as np
import polars as pl

from pytecgg.context import SUPPORTED_SYSTEMS
from pytecgg.satellites.constants import GPS_EPOCH, KEPLERIAN_POSITION_FIELDS
from pytecgg.satellites.kepler.coordinates import _kepler_satellite_coordinates
from pytecgg.satellites.state_vector.coordinates import (
    _state_vector_satellite_coordinates,
)

PREFIX_MAP = {v: k for k, v in SUPPORTED_SYSTEMS.items()}

# Rejects stale records (days or years old). Broadcast records are nominally
# valid ±2 h, but propagated 19 h they still give positions within ~50 m (GPS)
# to ~350 m (BeiDou), far below what TEC geometry needs; sparse nav files rely on it
MAX_KEPLERIAN_RECORD_AGE = datetime.timedelta(hours=24)

_SORTEDNESS_WARNING = "Sortedness of columns cannot be checked when 'by' groups provided"


def _infer_schema_overrides(records: list[dict[str, Any]]) -> dict[str, pl.DataType]:
    """Infer stable schema overrides for mixed GLONASS ephemeris records."""
    overrides: dict[str, pl.DataType] = {}

    if not records:
        return overrides

    for key in records[0]:
        sample = next((record.get(key) for record in records if record.get(key) is not None), None)
        if isinstance(sample, str):
            overrides[key] = pl.String
        elif isinstance(sample, bool):
            overrides[key] = pl.Boolean
        elif isinstance(sample, (int, np.integer)):
            overrides[key] = pl.Int64
        elif isinstance(sample, (float, np.floating)):
            overrides[key] = pl.Float64

    return overrides


def _emit_warnings(system: str, missing: set[str], failed: set[str]) -> None:
    """Helper function to emit warnings for missing or failed satellite coordinate calculations."""

    if missing:
        sorted_missing = sorted(missing)
        n_sv = len(missing)

        # If more than 8 missing, show only some
        if n_sv > 8:
            sv_ids = ", ".join(sorted_missing[:5]) + f"... (+ {n_sv - 5} more)"
        else:
            sv_ids = ", ".join(sorted_missing)

        verb = "has" if n_sv == 1 else "have"
        pron = "It" if n_sv == 1 else "These"

        if system == "GLONASS":
            info = f"One or more ephemerides are missing for {sv_ids}"
            reason = (
                f"{pron} may be missing from NAV file or the time gap may be too large"
            )
        else:
            info = f"{sv_ids} {verb} no valid ephemerides"
            reason = f"{pron} may be missing from NAV file, more than 24 h away from the observation, or {verb} been excluded due to invalid/incomplete parameters"

        warnings.warn(
            f"[{system}] {info}. Coordinates set to NaN.\n" f"ℹ️  {reason}.\n",
            RuntimeWarning,
        )

    if failed:
        sv_ids = ", ".join(sorted(failed))
        warnings.warn(
            f"[{system}] Calculation failed for {sv_ids}. Coordinates set to NaN.\n",
            RuntimeWarning,
        )


def _keplerian_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict: dict[str, Union[dict[str, Any], list[dict[str, Any]]]],
    gnss_system: str,
) -> pl.DataFrame:
    """
    Compute coordinates of Keplerian satellites (GPS, Galileo, BeiDou), using
    for each observation the broadcast record nearest in time, within
    ``MAX_KEPLERIAN_RECORD_AGE``. Rows without such a record get NaN.
    """
    fields = ["gps_week", *KEPLERIAN_POSITION_FIELDS]
    records = [
        {"sv": sv, "datetime": record["datetime"], **{f: record.get(f) for f in fields}}
        for sv, sv_records in ephem_dict.items()
        # A plain dict is a single legacy / hand-built record
        for record in ([sv_records] if isinstance(sv_records, dict) else sv_records)
    ]
    df_eph = (
        pl.DataFrame(
            records,
            schema={"sv": pl.String, "datetime": epochs.dtype, **{f: pl.Float64 for f in fields}},
        )
        .drop_nulls()
        .sort("datetime")
    )

    df_obs = pl.DataFrame({"sv": sv_ids, "epoch": epochs}).with_row_index("_row")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SORTEDNESS_WARNING)
        df_joined = (
            df_obs.sort("epoch")
            .join_asof(
                df_eph,
                left_on="epoch",
                right_on="datetime",
                by="sv",
                strategy="nearest",
                tolerance=MAX_KEPLERIAN_RECORD_AGE,
            )
            .sort("_row")
        )

    has_eph = df_joined["sqrta"].is_not_null().to_numpy()
    df_valid = df_joined.filter(pl.Series(has_eph))
    obs_gpst_s = df_valid["epoch"].dt.epoch("us").to_numpy() / 1e6 - GPS_EPOCH.timestamp()

    xyz = np.full((df_joined.height, 3), np.nan)
    xyz[has_eph] = _kepler_satellite_coordinates(
        {f: df_valid[f].to_numpy() for f in fields}, obs_gpst_s, gnss_system
    )

    sv_arr = df_joined["sv"].to_numpy()
    failed = has_eph & ~np.isfinite(xyz).all(axis=1)
    _emit_warnings(gnss_system, set(sv_arr[~has_eph]), set(sv_arr[failed]))

    return pl.DataFrame(
        {"sv": sv_ids, "epoch": epochs, "sat_x": xyz[:, 0], "sat_y": xyz[:, 1], "sat_z": xyz[:, 2]}
    )


def _state_vector_coordinates(df_joined: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:
    """
    Compute GLONASS coordinates by state-vector integration, one row at a time.

    ``df_joined`` is the join_asof result of observations (sv, epoch) with the
    nearest ephemeris message; rows without one have null ``satPosX``.
    """
    missing_eph = set()
    calculation_failed = set()

    data_rows = df_joined.to_dicts()
    x = np.full(len(data_rows), np.nan)
    y = np.full(len(data_rows), np.nan)
    z = np.full(len(data_rows), np.nan)

    for i, row in enumerate(data_rows):
        sv_id = row["sv"]

        # If satPosX is None, no ephemeris was found within tolerance in join_asof
        if row.get("satPosX") is None:
            missing_eph.add(sv_id)
            continue

        try:
            pos = _state_vector_satellite_coordinates(row, row["epoch"], **kwargs)
            if pos is not None and pos.size == 3:
                x[i], y[i], z[i] = pos
            else:
                calculation_failed.add(sv_id)
        except Exception:
            calculation_failed.add(sv_id)

    _emit_warnings("GLONASS", missing_eph, calculation_failed)

    return pl.DataFrame(
        {"sv": df_joined["sv"], "epoch": df_joined["epoch"], "sat_x": x, "sat_y": y, "sat_z": z}
    )


def satellite_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_dict: dict[str, Union[dict[str, Any], list[dict[str, Any]]]],
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Compute Earth-Centered Earth-Fixed (ECEF) coordinates for GNSS satellites.

    The function supports GPS, Galileo, BeiDou (using Keplerian orbits)
    and GLONASS (using state-vector propagation).

    Parameters
    ----------
    sv_ids : pl.Series
        Series containing satellite identifiers (e.g., 'G01', 'E23', 'R01')
    epochs : pl.Series
        Series containing observation times as datetime objects
    ephem_dict : dict
        Ephemeris data as returned by `prepare_ephemeris`: {sv_id: list[dict]}.
        For Keplerian systems a single dict per satellite is also accepted.
    **kwargs : Any
        Additional parameters for GLONASS state-vector propagation:
        - t_res : float or None, optional
            Time resolution (in seconds) for sampling the ODE solver solution
            - if float: the trajectory is sampled at fixed intervals
            - if None (default): the solver selects internal time steps automatically.
        - error_estimate : Literal["coarse", "normal", "fine"], optional
            Error tolerance level:
            - "coarse": ~2000 meters precision, faster
            - "normal": ~200 meters precision, balanced (default)
            - "fine": ~20 meters precision, slower

        These additional parameters are ignored for non-GLONASS systems

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: 'sv', 'epoch', 'sat_x', 'sat_y', 'sat_z'
        containing satellite ECEF coordinates in meters
    """
    unique_svs = sv_ids.unique().to_list()
    systems_in_data = {sv[0] for sv in unique_svs if sv[0] in PREFIX_MAP}

    results = []

    for prefix in systems_in_data:
        system_name = PREFIX_MAP[prefix]

        # Filter data for the current system
        mask = sv_ids.str.starts_with(prefix)
        sys_sv_ids = sv_ids.filter(mask)
        sys_epochs = epochs.filter(mask)

        if system_name == "GLONASS":
            # State-vector logic (DataFrame join_asof)
            eph_list = []
            for sv_ in ephem_dict:
                if sv_.startswith("R"):
                    # Ensure SV ID is present in the dictionaries for joining
                    for record in ephem_dict[sv_]:  # type: ignore
                        record["sv"] = sv_
                        eph_list.append(record)

            df_obs = pl.DataFrame({"sv": sys_sv_ids, "epoch": sys_epochs}).sort(["sv", "epoch"])
            if eph_list:
                df_eph = pl.DataFrame(
                    eph_list,
                    schema_overrides=_infer_schema_overrides(eph_list),
                    infer_schema_length=len(eph_list),
                ).sort("datetime")
            else:
                df_eph = pl.DataFrame(schema={"sv": pl.String, "datetime": pl.Datetime})

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=_SORTEDNESS_WARNING)
                df_joined = df_obs.join_asof(
                    df_eph,
                    left_on="epoch",
                    right_on="datetime",
                    by="sv",
                    strategy="nearest",
                    tolerance="45m",
                )

            results.append(_state_vector_coordinates(df_joined, **kwargs))

        else:
            sys_ephem_dict = {
                k: v for k, v in ephem_dict.items() if k.startswith(prefix)
            }
            results.append(
                _keplerian_coordinates(sys_sv_ids, sys_epochs, sys_ephem_dict, system_name)
            )

    if not results:
        return pl.DataFrame(
            schema={
                "sv": pl.String,
                "epoch": epochs.dtype,
                "sat_x": pl.Float64,
                "sat_y": pl.Float64,
                "sat_z": pl.Float64,
            }
        )

    return pl.concat(results).sort("epoch")
