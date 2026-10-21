from typing import Literal

import polars as pl

from pytecgg.context import GNSSContext, SUPPORTED_SYSTEMS

from .constants import FREQ_BANDS, BEIDOU_FREQ_BANDS_LEGACY, BEIDOU_FREQ_BANDS_MODERN
from .gflc import _calculate_gflc_code, _calculate_gflc_phase
from .iflc import _calculate_iflc_code, _calculate_iflc_phase
from .mw import _calculate_melbourne_wubbena
from .observables import (
    SelectedSignalPair,
    SelectionMode,
    select_observable_pair,
    select_observable_pair_from_pivot,
)

PER_SATELLITE_SELECTION_SYSTEMS = {"C"}


def _beidou_freq_bands(rinex_version: str) -> dict[str, float]:
    """Return the correct BeiDou frequency mapping for the given RINEX version."""
    try:
        minor = int(str(rinex_version).split(".")[1])
    except (IndexError, ValueError):
        minor = 99
    if minor <= 2:
        return BEIDOU_FREQ_BANDS_LEGACY
    return BEIDOU_FREQ_BANDS_MODERN


def _scalar_freqs_for_sv(
    system: str,
    sv: str,
    band1: str,
    band2: str,
    ctx: GNSSContext,
) -> tuple[float, float] | None:
    """Resolve scalar (f1, f2) in Hz for a single satellite, or None if unavailable."""
    if system == "C":
        beidou_bands = _beidou_freq_bands(ctx.rinex_version)
        try:
            return beidou_bands[band1], beidou_bands[band2]
        except KeyError:
            return None
    if system == "R":
        if not ctx.glonass_channels:
            raise RuntimeError(
                "GLONASS requested but `glonass_channels` is empty in the provided context. "
                "Did you forget to call `prepare_ephemeris()`?"
            )
        k = ctx.glonass_channels.get(sv)
        if k is None:
            return None
        f1_map = FREQ_BANDS["R"][band1]
        f2_map = FREQ_BANDS["R"][band2]
        return f1_map(k), f2_map(k)
    try:
        return FREQ_BANDS[system][band1], FREQ_BANDS[system][band2]
    except KeyError:
        return None


def _process_system_per_sv(
    df_sys: pl.DataFrame,
    system: str,
    ctx: GNSSContext,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ],
    selection_mode: SelectionMode,
    band_override: tuple[str, str] | None,
    obs_overrides_for_system: dict[str, dict[str, tuple[str, str]]],
) -> list[pl.DataFrame]:
    """Compute linear combinations one satellite at a time for a constellation."""
    df_pivot = df_sys.pivot(
        values="value",
        index=["epoch", "sv"],
        on="observable",
        aggregate_function="first",
    )
    if df_pivot.is_empty():
        return []

    results: list[pl.DataFrame] = []
    freq_meta_by_sv: dict[str, tuple[float, float]] = {}

    for df_sv in df_pivot.sort(["sv", "epoch"]).partition_by("sv", maintain_order=True):
        sv = df_sv.get_column("sv")[0]
        selection = select_observable_pair_from_pivot(
            df_sv,
            system=system,
            selection_mode=selection_mode,
            band_override=band_override,
            obs_override=obs_overrides_for_system.get(sv),
        )
        if selection is None:
            continue

        freqs = _scalar_freqs_for_sv(system, sv, selection.band1, selection.band2, ctx)
        if freqs is None:
            continue
        f1, f2 = freqs

        df_sv = df_sv.select(
            [
                "epoch",
                "sv",
                selection.phase1,
                selection.phase2,
                selection.code1,
                selection.code2,
            ]
        )
        freq_meta_by_sv[sv] = (f1 / 1e6, f2 / 1e6)
        results.append(
            _apply_linear_combinations(
                df_sv,
                selection=selection,
                freq1=pl.lit(f1),
                freq2=pl.lit(f2),
                combinations=combinations,
            )
        )

    if freq_meta_by_sv:
        ctx.freq_meta[system] = freq_meta_by_sv

    return results


def _apply_linear_combinations(
    df_step: pl.DataFrame,
    selection: SelectedSignalPair,
    freq1: pl.Expr,
    freq2: pl.Expr,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ],
) -> pl.DataFrame:
    if "gflc_phase" in combinations:
        df_step = df_step.with_columns(
            _calculate_gflc_phase(
                pl.col(selection.phase1), pl.col(selection.phase2), freq1, freq2
            ).alias("gflc_phase"),
            pl.lit(f"{selection.phase1},{selection.phase2}").alias(
                "gflc_phase_obs_used"
            ),
        )
    if "gflc_code" in combinations:
        df_step = df_step.with_columns(
            _calculate_gflc_code(
                pl.col(selection.code1), pl.col(selection.code2), freq1, freq2
            ).alias("gflc_code"),
            pl.lit(f"{selection.code1},{selection.code2}").alias("gflc_code_obs_used"),
        )
    if "mw" in combinations:
        df_step = df_step.with_columns(
            _calculate_melbourne_wubbena(
                pl.col(selection.phase1),
                pl.col(selection.phase2),
                pl.col(selection.code1),
                pl.col(selection.code2),
                freq1,
                freq2,
            ).alias("mw")
        )
    if "iflc_phase" in combinations:
        df_step = df_step.with_columns(
            _calculate_iflc_phase(
                pl.col(selection.phase1), pl.col(selection.phase2), freq1, freq2
            ).alias("iflc_phase")
        )
    if "iflc_code" in combinations:
        df_step = df_step.with_columns(
            _calculate_iflc_code(
                pl.col(selection.code1), pl.col(selection.code2), freq1, freq2
            ).alias("iflc_code")
        )

    drop_cols = [selection.phase1, selection.phase2, selection.code1, selection.code2]
    if "_k" in df_step.columns:
        drop_cols.append("_k")

    return df_step.drop(drop_cols)


def calculate_linear_combinations(
    obs_data: pl.DataFrame,
    ctx: GNSSContext,
    combinations: list[
        Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]
    ] = ["gflc_phase", "gflc_code", "mw"],
    selection_mode: SelectionMode = "quality",
    band_overrides: dict[str, tuple[str, str]] | None = None,
    obs_overrides: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> pl.DataFrame:
    """
    Process observations for multiple GNSS systems to calculate specific linear combinations.

    Parameters
    ----------
    obs_data : pl.DataFrame
        DataFrame containing observation data.
    ctx : GNSSContext
        Execution context containing GNSS systems, RINEX version, and support lookups.
    combinations : list[Literal["gflc_phase", "gflc_code", "mw", "iflc_phase", "iflc_code"]]
        List of combinations to calculate. Options:
            - "gflc_phase": Geometry-Free Linear Combination (Phase)
            - "gflc_code": Geometry-Free Linear (Code)
            - "mw": Melbourne-Wübbena combination
            - "iflc_phase": Ionosphere-Free Linear Combination (Phase)
            - "iflc_code": Ionosphere-Free Linear Combination (Code)
        Defaults to ["gflc_phase", "gflc_code", "mw"].
    selection_mode : {"quality", "availability", "legacy"}
        Policy used to rank viable dual-frequency band pairs.
        Defaults to "quality".
    band_overrides : dict[str, tuple[str, str]] | None
        Optional per-system band overrides. Keys can be constellation symbols or
        full names, for example {"G": ("L1", "L5")} or {"GPS": ("L1", "L5")}.
    obs_overrides : dict[str, dict[str, tuple[str, str]]] | None
        Optional per-satellite observable pins. Keys are satellite ids and values
        fully specify the phase and code observables to use, for example
        ``{"C06": {"phase": ("L1P", "L2X"), "code": ("C1X", "C2P")}}``. A
        constellation with any pinned satellite is processed per satellite;
        pinned satellites use the forced observables (skipped if any are missing)
        while the others fall back to the normal selection. Mutually exclusive
        with ``band_overrides``.

    Returns
    -------
    pl.DataFrame
        DataFrame with the requested linear combinations. When ``"gflc_code"``
        is requested, an additional ``gflc_code_obs_used`` string column is
        included, containing the two code observable names used to compute
        ``gflc_code`` joined with a comma (for example ``"C1C,C2W"``).
        Likewise, when ``"gflc_phase"`` is requested, a ``gflc_phase_obs_used``
        string column is included with the two phase observable names used to
        compute ``gflc_phase`` (for example ``"L1C,L2W"``). For BeiDou these
        values can vary per satellite because band selection is applied per
        ``sv``; for the other systems they are constant within the
        constellation block.

    Notes
    -----
    BeiDou currently uses per-satellite band-pair selection to cope with mixed
    signal availability across satellite generations, while other systems retain
    the per-constellation selection policy.
    """
    if band_overrides and obs_overrides:
        raise ValueError(
            "Pass either `band_overrides` or `obs_overrides`, not both."
        )

    results = []
    normalised_overrides: dict[str, tuple[str, str]] = {}

    if band_overrides:
        for system_key, band_pair in band_overrides.items():
            system_symbol = SUPPORTED_SYSTEMS.get(system_key.upper(), system_key.upper())
            normalised_overrides[system_symbol] = band_pair

    obs_overrides = obs_overrides or {}
    systems_with_obs_override = {sv[:1] for sv in obs_overrides}

    for system_ in ctx.systems:
        df_sys = obs_data.filter(pl.col("sv").str.starts_with(system_))
        if df_sys.is_empty():
            continue

        use_per_sv = (
            system_ in PER_SATELLITE_SELECTION_SYSTEMS
            or system_ in systems_with_obs_override
        )
        if use_per_sv:
            obs_overrides_for_system = {
                sv: override
                for sv, override in obs_overrides.items()
                if sv[:1] == system_
            }
            results.extend(
                _process_system_per_sv(
                    df_sys,
                    system=system_,
                    ctx=ctx,
                    combinations=combinations,
                    selection_mode=selection_mode,
                    band_override=normalised_overrides.get(system_),
                    obs_overrides_for_system=obs_overrides_for_system,
                )
            )
            continue

        selection = select_observable_pair(
            obs_data,
            system=system_,
            rinex_version=ctx.rinex_version,
            selection_mode=selection_mode,
            band_override=normalised_overrides.get(system_),
        )
        if selection is None:
            continue

        required_cols = {
            selection.phase1,
            selection.phase2,
            selection.code1,
            selection.code2,
        }

        df_sys = df_sys.filter(pl.col("observable").is_in(required_cols))
        if df_sys.is_empty():
            continue

        df_pivot = df_sys.pivot(
            values="value",
            index=["epoch", "sv"],
            on="observable",
            aggregate_function="first",
        )
        if not required_cols.issubset(df_pivot.columns):
            continue

        if system_ == "R":
            if not ctx.glonass_channels:
                raise RuntimeError(
                    "GLONASS requested but `glonass_channels` is empty in the provided context. "
                    "Did you forget to call `prepare_ephemeris()`?"
                )

            f1_map = FREQ_BANDS["R"][selection.band1]
            f2_map = FREQ_BANDS["R"][selection.band2]

            ctx.freq_meta[system_] = {
                sv: (f1_map(k) / 1e6, f2_map(k) / 1e6)
                for sv, k in ctx.glonass_channels.items()
                if k is not None
            }

            df_step = df_pivot.with_columns(
                pl.col("sv").replace(ctx.glonass_channels).cast(pl.Float32).alias("_k")
            )
            freq1, freq2 = f1_map(pl.col("_k")), f2_map(pl.col("_k"))
        else:
            try:
                f1 = FREQ_BANDS[system_][selection.band1]
                f2 = FREQ_BANDS[system_][selection.band2]
            except KeyError:
                continue

            ctx.freq_meta[system_] = (f1 / 1e6, f2 / 1e6)
            freq1, freq2 = pl.lit(f1), pl.lit(f2)
            df_step = df_pivot

        results.append(
            _apply_linear_combinations(
                df_step,
                selection=selection,
                freq1=freq1,
                freq2=freq2,
                combinations=combinations,
            )
        )

    if not results:
        return pl.DataFrame()

    return pl.concat(results)
