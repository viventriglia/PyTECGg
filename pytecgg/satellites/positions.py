import warnings
from typing import Any, Callable, Union

import numpy as np
import polars as pl

from pytecgg.context import SUPPORTED_SYSTEMS
from pytecgg.satellites.kepler.coordinates import _kepler_satellite_coordinates
from pytecgg.satellites.state_vector.coordinates import (
    _state_vector_satellite_coordinates,
)

PREFIX_MAP = {v: k for k, v in SUPPORTED_SYSTEMS.items()}


def _pick_nearest_keplerian_record(
    records: Union[dict[str, Any], list[dict[str, Any]]],
    epoch: Any,
) -> Union[dict[str, Any], None]:
    """Pick the broadcast record whose datetime is nearest to `epoch`.

    Accepts either a list of records (current format) or a single record dict
    (legacy format) so manually built ephem_dicts keep working.
    """
    if isinstance(records, dict):
        return records
    if not records:
        return None
    return min(records, key=lambda r: abs(r["datetime"] - epoch))


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
            reason = f"{pron} may be missing from NAV file or {verb} been excluded due to invalid/incomplete parameters"

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


def _compute_coordinates(
    sv_ids: pl.Series,
    epochs: pl.Series,
    ephem_data: Union[dict[str, Any], pl.DataFrame],
    coord_func: Callable[..., np.ndarray],
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Compute satellite coordinates for multiple satellites and epochs.

    Parameters
    ----------
    sv_ids : pl.Series
        Series containing satellite identifiers (e.g., 'G01', 'E23', 'R01')
    epochs : pl.Series
        Series containing observation times as datetime objects
    ephem_data : Union[dict[str, Any], pl.DataFrame]
        Dictionary (Keplerian) or DataFrame (GLONASS) containing ephemeris data.
    coord_func : Callable[..., np.ndarray]
        Function to compute coordinates for a single satellite and epoch
    **kwargs : Any
        Additional keyword arguments to pass to the coordinate function

    Returns
    -------
    pl.DataFrame
        DataFrame with columns: 'sv', 'epoch', 'sat_x', 'sat_y', 'sat_z'
        containing satellite ECEF coordinates in meters.
    """
    # Track satellites with no or problematic ephemeris for warnings
    missing_eph = set()
    calculation_failed = set()
    system_label = kwargs.pop(
        "gnss_system_internal",
        "GLONASS" if isinstance(ephem_data, pl.DataFrame) else None,
    )

    # GLONASS (State-vector integration logic)
    if isinstance(ephem_data, pl.DataFrame):
        # ephem_data is the join_asof result, which is sorted by (sv, epoch).
        # Use its sv/epoch columns for the output so each computed (x, y, z)
        # stays aligned with the corresponding input pair — using the unsorted
        # input series here would mis-align rows.
        data_rows = ephem_data.to_dicts()
        size = len(data_rows)
        out_sv = [row["sv"] for row in data_rows]
        out_epoch = [row["epoch"] for row in data_rows]
        x = np.full(size, np.nan)
        y = np.full(size, np.nan)
        z = np.full(size, np.nan)

        for i, row in enumerate(data_rows):
            sv_id = row["sv"]

            # If satPosX is None, no ephemeris was found within tolerance in join_asof
            if row.get("satPosX") is None:
                missing_eph.add(sv_id)
                continue

            try:
                pos = coord_func(row, row["epoch"], **kwargs)
                if pos is not None and pos.size == 3:
                    x[i], y[i], z[i] = pos
                else:
                    calculation_failed.add(sv_id)
            except Exception:
                calculation_failed.add(sv_id)
                continue

    # Keplerian systems (GPS, Galileo, BeiDou, QZSS)
    else:
        sv_arr = sv_ids.to_numpy()
        size = len(sv_arr)
        out_sv = sv_arr
        out_epoch = epochs
        x = np.full(size, np.nan)
        y = np.full(size, np.nan)
        z = np.full(size, np.nan)

        for i, (sv, epoch) in enumerate(zip(sv_arr, epochs)):
            records = ephem_data.get(sv)
            if not records:
                missing_eph.add(sv)
                continue

            chosen = _pick_nearest_keplerian_record(records, epoch)
            if chosen is None:
                missing_eph.add(sv)
                continue

            try:
                pos = coord_func({sv: chosen}, sv, system_label, epoch, **kwargs)
                if pos is not None and pos.size == 3:
                    x[i], y[i], z[i] = pos
                else:
                    calculation_failed.add(sv)
            except Exception:
                calculation_failed.add(sv)
                continue

    _emit_warnings(system_label, missing_eph, calculation_failed)

    return pl.DataFrame(
        {
            "sv": out_sv,
            "epoch": out_epoch,
            "sat_x": x,
            "sat_y": y,
            "sat_z": z,
        }
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
        Dictionary containing ephemeris data
        Expected format: {sv_id: dict} for Keplerian systems or {sv_id: list[dict]}
        for GLONASS.
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
                warnings.filterwarnings(
                    "ignore",
                    message="Sortedness of columns cannot be checked when 'by' groups provided",
                )
                df_joined = df_obs.join_asof(
                    df_eph,
                    left_on="epoch",
                    right_on="datetime",
                    by="sv",
                    strategy="nearest",
                    tolerance="45m",
                )

            res = _compute_coordinates(
                sys_sv_ids,
                sys_epochs,
                df_joined,
                _state_vector_satellite_coordinates,
                **kwargs,
            )
            results.append(res)

        else:
            # Keplerian logic (Dictionary lookup)
            # We filter the global ephem_dict to pass only relevant satellites
            sys_ephem_dict = {
                k: v for k, v in ephem_dict.items() if k.startswith(prefix)
            }

            res = _compute_coordinates(
                sys_sv_ids,
                sys_epochs,
                sys_ephem_dict,
                _kepler_satellite_coordinates,
                gnss_system_internal=system_name,
                **kwargs,
            )
            results.append(res)

    if not results:
        return pl.DataFrame(
            schema={
                "sv": pl.String,
                "epoch": pl.Datetime,
                "sat_x": pl.Float64,
                "sat_y": pl.Float64,
                "sat_z": pl.Float64,
            }
        )

    return pl.concat(results).sort("epoch")
