import polars as pl
from typing import Optional, Literal

from . import OBS_MAPPING, FREQ_BANDS, C


def _calculate_gflc(
    phase1: pl.Expr, phase2: pl.Expr, freq1: pl.Expr, freq2: pl.Expr
) -> pl.Expr:
    """
    Calculate the geometry-free linear combination (GFLC) from two phase observations

    Args:
        phase1 (pl.Expr): Phase observation for frequency 1
        phase2 (pl.Expr): Phase observation for frequency 2
        freq1 (pl.Expr): Frequency 1 in Hz
        freq2 (pl.Expr): Frequency 2 in Hz
    Returns:
        pl.Expr: Expression for the calculated GFLC
    """
    lambda1 = C / freq1
    lambda2 = C / freq2
    pr_to_tec = (1 / 40.308) * (freq1**2 * freq2**2) / (freq1**2 - freq2**2) / 1e16
    return (phase1 * lambda1 - phase2 * lambda2) * pr_to_tec


def process_observations(
    obs_data: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    glonass_freq: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    """
    Process observations for a specific GNSS system to calculate GFLC
    Args:
        obs_data (pl.DataFrame): DataFrame containing observation data
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        glonass_freq (Optional[dict[str, int]]): Frequency mapping for GLONASS, required if system is "R"

    Returns:
        pl.DataFrame: DataFrame with calculated GFLC values
    """
    mapping = OBS_MAPPING[system]["phase"]
    obs_keys = list(mapping.keys())  # e.g. ["L1", "L2"] or ["L1", "L5"]
    obs1, obs2 = mapping[obs_keys[0]], mapping[obs_keys[1]]

    df_sys = obs_data.filter(
        (pl.col("sv").str.starts_with(system))
        & (pl.col("observable").is_in([obs1, obs2]))
    )

    if df_sys.is_empty():
        return pl.DataFrame()

    df_pivot = df_sys.pivot(
        values="value",
        index=["epoch", "sv"],
        columns="observable",
        aggregate_function="first",
    )

    if obs1 not in df_pivot.columns or obs2 not in df_pivot.columns:
        return pl.DataFrame()

    # Frequency handling
    if system == "R":
        if glonass_freq is None:
            raise ValueError("glonass_freq is required for GLONASS processing")
        df_pivot = df_pivot.with_columns(
            pl.col("sv").map_dict(glonass_freq).alias("freq_number")
        )
        freq1 = (1602 + pl.col("freq_number") * 0.5625) * 1e6
        freq2 = (1246 + pl.col("freq_number") * 0.4375) * 1e6
    elif system in ["G", "E", "C"]:
        obs_to_band = {v: k for k, v in mapping.items()}
        band1 = obs_to_band.get(obs1)
        band2 = obs_to_band.get(obs2)

        try:
            freq1 = FREQ_BANDS[system][band1]
            freq2 = FREQ_BANDS[system][band2]
        except KeyError as e:
            raise KeyError(
                f"Missing frequency for band '{e.args[0]}' in system '{system}'"
            )

    return df_pivot.with_columns(
        _calculate_gflc(pl.col(obs1), pl.col(obs2), freq1, freq2).alias("GFLC")
    )
