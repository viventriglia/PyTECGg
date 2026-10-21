from warnings import warn
from datetime import timedelta
from typing import Optional, Any

import polars as pl

from pytecgg.linear_combinations import detect_cs_lol
from pytecgg.context import GNSSContext


def _add_arc_id(
    min_arc_length: int = 30, receiver_acronym: str = None
) -> list[pl.Expr]:
    """
    Identify continuous TEC arcs in GNSS observations

    Arcs are defined as sequences of observations without loss-of-lock events.
    Cycle slips do not break arcs, as they can be repaired in subsequent processing.
    Arcs shorter than the minimum length (in epochs) are discarded.

    Parameters:
    ----------
    min_arc_length : int
        Minimum number of consecutive valid observations required for an arc to be considered valid.
        Default: 30 epochs.
    receiver_acronym : str, optional
        Acronym of the receiver to prepend to the arc identifier.
        If provided, the arc ID format will be "<receiver>_<sv>_<YYYYMMDD>_<arcnumber>".
        Otherwise, the format will be "<sv>_<YYYYMMDD>_<arcnumber>".
        Default: None.

    Returns:
    -------
    list[pl.Expr]
        List of Polars expressions representing:
        - id_arc: Arc identifier
        - id_arc_valid: Valid arc identifier (None for arcs shorter than `min_arc_length`)
    """
    _id_arc = pl.col("is_loss_of_lock").cum_sum().over("sv") + 1
    _arc_length = pl.col("gflc_code").is_not_null().sum().over(["sv", _id_arc])
    _arc_start_date = pl.col("epoch").min().over(["sv", _id_arc]).dt.strftime("%Y%m%d")
    _id_arc = _id_arc.cast(pl.Int64).cast(pl.Utf8).str.zfill(3)

    if receiver_acronym:
        id_arc = pl.format(
            "{}_{}_{}_{}",
            pl.lit(receiver_acronym),
            pl.col("sv").str.to_lowercase(),
            _arc_start_date,
            _id_arc,
        )
    else:
        id_arc = pl.format(
            "{}_{}_{}",
            pl.col("sv").str.to_lowercase(),
            _arc_start_date,
            _id_arc,
        )
    id_arc_valid = pl.when(_arc_length >= min_arc_length).then(id_arc).otherwise(None)

    return [id_arc.alias("id_arc"), id_arc_valid.alias("id_arc_valid")]


def _remove_cs_jumps(df: pl.DataFrame, threshold_jump: float = 10.0) -> pl.DataFrame:
    """
    Fix GNSS combinations by removing cycle-slip jumps within valid arcs

    For each column in `linear_combinations`, the function:
    1. Calculates differences between consecutive observations within valid arcs
    2. Identifies cycle slip contributions
    3. Computes cumulative sum of cycle slip jumps within each arc
    4. Fixes linear combinations by removing cumulative cycle slip effects
    5. Additional check: detects and corrects jumps above threshold between consecutive epochs

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - id_arc_valid: Valid arc identifiers
        - is_cycle_slip: Cycle slip indicators
        - linear_combinations, as calculated in `calculate_linear_combinations`
    threshold_jump : float, optional
        Threshold for detecting significant jumps between consecutive epochs.
        If the absolute difference between consecutive values exceeds this threshold,
        it will be treated as a jump and corrected. Default is 10

    Returns
    -------
    pl.DataFrame
        DataFrame with fixed columns (suffix '_fix') added
    """
    predefined_lin_combs = ["gflc_phase", "mw", "iflc_phase"]
    lin_combs = [lc_ for lc_ in predefined_lin_combs if lc_ in df.columns]

    if len(lin_combs) == 0:
        warn(
            "No linear combinations found in DataFrame columns; expected at least one of: "
            + ", ".join(predefined_lin_combs)
        )
        return df

    df_ = df.clone()

    for lc_ in lin_combs:
        # Calculate differences within valid arcs
        df_ = df_.with_columns(
            pl.when(pl.col("id_arc_valid").is_not_null())
            .then(pl.col(lc_) - pl.col(lc_).shift(1).over("id_arc_valid"))
            .otherwise(None)
            .alias(f"_delta_{lc_}")
        )

        # Identify cycle slip contributions
        df_ = df_.with_columns(
            pl.when(pl.col("is_cycle_slip"))
            .then(pl.col(f"_delta_{lc_}"))
            .otherwise(0.0)
            .alias(f"_cs_delta_{lc_}")
        )

        # Additional check: detect significant jumps between consecutive epochs
        # This catches jumps that might not be flagged as cycle slips
        df_ = df_.with_columns(
            pl.when(
                (pl.col("id_arc_valid").is_not_null())
                & (pl.col(f"_delta_{lc_}").abs() > threshold_jump)
                & (
                    ~pl.col("is_cycle_slip")
                )  # Only if not already flagged as cycle slip
            )
            .then(pl.col(f"_delta_{lc_}"))
            .otherwise(0.0)
            .alias(f"_jump_delta_{lc_}")
        )

        # Combine cycle slip and jump contributions
        total_jump = pl.col(f"_cs_delta_{lc_}") + pl.col(f"_jump_delta_{lc_}")

        # Cumulative sum of cycle slips and jumps within each arc
        df_ = df_.with_columns(
            total_jump.cum_sum().over("id_arc_valid").alias(f"_total_cumsum_{lc_}")
        )

        df_ = df_.with_columns(
            (pl.col(lc_) - pl.col(f"_total_cumsum_{lc_}")).alias(f"{lc_}_fix")
        )

    return df_.drop(pl.col("^_.*$"))


def _level_phase_to_code(df: pl.DataFrame) -> pl.DataFrame:
    """
    Level phase measurements to code measurements within valid arcs

    For each valid arc, the function computes the mean difference between
    phase and code linear combinations and adjusts the phase measurements accordingly.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations with:
        - id_arc_valid: Valid arc identifiers
        - gflc_phase_fix: Fixed phase linear combination
        - gflc_code: Code linear combination

    Returns
    -------
    pl.DataFrame
        DataFrame with leveled phase column added
    """
    if "gflc_phase_fix" not in df.columns or "gflc_code" not in df.columns:
        warn(
            "Both 'gflc_phase_fix' and 'gflc_code' must be present in DataFrame columns to level phase to code."
        )
        return df

    df_ = df.with_columns(
        (pl.col("gflc_phase_fix") - pl.col("gflc_code")).alias("_phase_code_diff")
    )

    # Calculate the mean of (phase - code) over each valid arc
    df_ = df_.with_columns(
        pl.when(pl.col("id_arc_valid").is_not_null())
        .then(pl.col("_phase_code_diff").mean().over("id_arc_valid"))
        .otherwise(None)
        .alias("_mean_phase_code_diff")
    )

    # Calculate the final, levelled value
    df_ = df_.with_columns(
        pl.when(pl.col("id_arc_valid").is_not_null())
        .then(pl.col("gflc_phase_fix") - pl.col("_mean_phase_code_diff"))
        .otherwise(None)
        .alias("gflc_levelled")
    )

    return df_.drop(pl.col("^_.*$"))


def extract_arcs(
    df: pl.DataFrame,
    ctx: GNSSContext,
    threshold_abs: float = 5.0,
    threshold_std: float = 5.0,
    min_arc_length: int = 30,
    max_gap: Optional[timedelta] = None,
    threshold_jump: float = 10.0,
) -> pl.DataFrame:
    """
    Extract continuous TEC arcs and fix GNSS linear combinations for multiple constellations.

    The function performs the following steps:
    1. Detects loss-of-lock events and cycle slips per constellation.
    2. Identifies valid arcs, discarding short ones.
    3. Removes cycle-slip jumps within valid arcs.
    4. Corrects significant jumps between consecutive epochs.
    5. Calculates arc-levelled GFLC values.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations.
    ctx : GNSSContext
        Execution context containing system configurations and frequency metadata.
    threshold_abs : float, optional
        Absolute threshold for detecting cycle slips; default is 5.
    threshold_std : float, optional
        Standard deviation multiplier threshold for cycle slips; default is 5.
    min_arc_length : int, optional
        Minimum number of consecutive valid observations for an arc; default is 30.
    max_gap : timedelta, optional
        Maximum allowed time gap before declaring Loss-of-Lock.
    threshold_jump : float, optional
        Threshold for detecting significant jumps between epochs; default is 10.

    Returns
    -------
    pl.DataFrame
        DataFrame with arc identifiers and levelled GFLC values.
    """
    cs_results = []

    # Iterate through systems defined in the context to handle constellation-specific noise/frequencies
    for sys_ in ctx.systems:
        f1, f2 = None, None
        satellite_freq = None
        if sys_ in ctx.freq_meta:
            meta = ctx.freq_meta[sys_]
            if sys_ != "R":
                if isinstance(meta, dict):
                    satellite_freq = {
                        sv: (freq1 * 1e6, freq2 * 1e6)
                        for sv, (freq1, freq2) in meta.items()
                    }
                else:
                    f1, f2 = meta[0] * 1e6, meta[1] * 1e6

        # Detect CS and LoL for this specific system block
        df_sys = df.filter(pl.col("sv").str.starts_with(sys_))

        if df_sys.is_empty():
            continue

        df_cs = detect_cs_lol(
            df_sys,
            system=sys_,
            threshold_abs=threshold_abs,
            threshold_std=threshold_std,
            max_gap=max_gap,
            glonass_freq=ctx.glonass_channels,
            satellite_freq=satellite_freq,
            f1=f1,
            f2=f2,
        )
        cs_results.append(df_cs)

    # Recombine results, join back the slip detection results and
    # assign unique arc identifiers
    df_all_cs = pl.concat(cs_results)
    df_lc_arcs = df.join(df_all_cs, on=["epoch", "sv"], how="left").with_columns(
        _add_arc_id(min_arc_length=min_arc_length, receiver_acronym=ctx.receiver_name)
    )

    # Drop loss-of-lock and null-LC epochs before fixing.  Arc IDs are already
    # assigned above (boundaries are defined by the LoL cum-sum), so removing
    # these rows here preserves the arc segmentation while keeping garbage
    # re-acquisition values out of the jump/cumsum baseline in _remove_cs_jumps.
    # Note: is_loss_of_lock may be null for rows with no CS-detection match
    # (left join). `col != True` returns null for those and filter() drops nulls,
    # so use fill_null(False) to drop ONLY genuine loss-of-lock rows.
    df_lc_arcs = df_lc_arcs.filter(
        (~pl.col("is_loss_of_lock").fill_null(False))
        & pl.col("gflc_phase").is_not_null()
        & pl.col("gflc_code").is_not_null()
        & pl.col("mw").is_not_null()
    )

    # Apply corrections to the linear combinations and perform phase-to-code levelling
    df_lc_arcs_fix = _remove_cs_jumps(df=df_lc_arcs, threshold_jump=threshold_jump)

    return _level_phase_to_code(df=df_lc_arcs_fix)
