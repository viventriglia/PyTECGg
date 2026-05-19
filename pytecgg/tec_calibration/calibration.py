import datetime
import warnings

import numpy as np
import polars as pl
from scipy.linalg import qr, solve, solve_triangular
from scipy.sparse import csr_matrix

from pytecgg.tec_calibration.calibration_preprocessing import (
    _polynomial_expansion,
    _preprocessing,
    _mapping_function,
)
from pytecgg.context import GNSSContext


def _gg_calibration(
    df_clean: pl.DataFrame,
    batch_length_mins: int = 15,
    max_degree: int = 3,
) -> dict[str, float]:
    """
    Calibrate GNSS arcs using polynomial expansion and batch QR decomposition.

    The function processes batches of epochs, computes polynomial terms, and solves
    for arc-level biases using a QR approach.

    Parameters
    ----------
    df_clean : pl.DataFrame
        Preprocessed DataFrame containing:
        - gflc_vert: vertical TEC observations
        - mapping: mapping function values
        - modip_ipp, modip_rec: MoDip parameters
        - lon_ipp, lon_rec: longitudes
        - id_arc_valid: validated arc identifiers
    batch_length_mins : int, optional
        Length in minutes of the calibration batch. Default is 15.
    max_degree : int, optional
        Maximum degree of the polynomial expansion. Default is 3.

    Returns
    -------
    dict[str, float]
        Dictionary mapping arc identifiers to estimated biases.
    """

    df_clean = df_clean.sort(by="epoch")
    n_coeffs = max_degree + 3

    min_time, max_time = df_clean["epoch"].min(), df_clean["epoch"].max()
    interval_td = datetime.timedelta(minutes=batch_length_mins)

    int_starts = pl.datetime_range(
        start=min_time, end=max_time, interval=f"{batch_length_mins}m", eager=True
    ).alias("epoch")
    int_number = len(int_starts)

    int_mat_list = []
    int_arcs_list = []
    known_terms_list = []

    list_all_used_arcs = df_clean["id_arc_valid"].unique(maintain_order=True).to_numpy()

    for int_idx_ in range(int_number):
        start_time = int_starts[int_idx_]
        end_time = start_time + interval_td
        batch_center = start_time + interval_td / 2

        int_df = df_clean.filter(
            (pl.col("epoch") >= start_time) & (pl.col("epoch") < end_time)
        )

        if int_df.height == 0:
            continue

        arc_list_interval, equation_arc_idx = np.unique(
            int_df["id_arc_valid"], return_inverse=True
        )

        epochs_np = int_df["epoch"].to_numpy()
        batch_center_np = np.datetime64(batch_center)
        dt_utc_hours = (epochs_np - batch_center_np) / np.timedelta64(1, "h")
        lon_ipp_np = int_df["lon_ipp"].to_numpy()
        lon_rec_np = int_df["lon_rec"].to_numpy()
        delta_lt = dt_utc_hours + (lon_ipp_np - lon_rec_np) / 15.0

        poly_coeffs_matrix = _polynomial_expansion(
            int_df["modip_ipp"].to_numpy(),
            int_df["modip_rec"].to_numpy(),
            lon_ipp_np,
            lon_rec_np,
            delta_lt,
            max_degree,
        )

        # Mapping function matrix
        num_rows_poly = poly_coeffs_matrix.shape[0]

        row_indices = np.arange(num_rows_poly)
        col_indices = equation_arc_idx
        data_values = int_df["mapping"].to_numpy()

        mapping_f_sparse = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(num_rows_poly, len(arc_list_interval)),
        )
        mapping_f_matrix = mapping_f_sparse.toarray()

        gflc_vert_col = int_df["gflc_vert"].to_numpy().reshape(-1, 1)

        AA = np.hstack([poly_coeffs_matrix, mapping_f_matrix, gflc_vert_col])

        triang_mat = qr(AA, mode="r")[0]

        # Extract sub-matrices containing arc biases and known terms
        int_mat = triang_mat[n_coeffs : triang_mat.shape[1] - 1, n_coeffs:-1]
        known_term = triang_mat[n_coeffs : triang_mat.shape[1] - 1, -1]

        int_mat_list.append(int_mat)
        int_arcs_list.append(arc_list_interval)
        known_terms_list.append(known_term.reshape(-1, 1))

    # Block lengths = number of equations per interval
    blocks_length = np.array([m.shape[0] for m in int_mat_list])
    blocks_start = np.cumsum(blocks_length) - blocks_length

    total_rows = np.sum(blocks_length)
    num_all_arcs = len(list_all_used_arcs)

    full_arcbias_mat = np.zeros((total_rows, num_all_arcs))

    arc_to_full_idx = {arc: i for i, arc in enumerate(list_all_used_arcs)}

    for int_idx in range(len(int_mat_list)):

        int_arcs = int_arcs_list[int_idx]
        int_mat = int_mat_list[int_idx]

        start_row = blocks_start[int_idx]
        block_len = blocks_length[int_idx]
        end_row = start_row + block_len

        int_arcs_idx_in_full_list = np.array(
            [arc_to_full_idx[arc_id] for arc_id in int_arcs], dtype=int
        )

        full_arcbias_mat[start_row:end_row, int_arcs_idx_in_full_list] = int_mat

    known_terms_full_vec = np.vstack(known_terms_list)

    global_system_matrix = np.hstack([full_arcbias_mat, known_terms_full_vec])
    full_triang_mat = qr(global_system_matrix, mode="r")[0]

    non_zero_rows_global = np.any(np.abs(full_triang_mat) > 1e-12, axis=1)
    full_triang_mat_cleaned = full_triang_mat[non_zero_rows_global, :]

    last_row = full_triang_mat_cleaned[-1, :]

    if np.all(np.abs(last_row[:-1]) < 1e-12):
        lin_sys_mat = full_triang_mat_cleaned[:-1, :]
    else:
        lin_sys_mat = full_triang_mat_cleaned

    # Extract the coefficient matrix (unknowns) and the known terms vector
    num_independent_eq = lin_sys_mat.shape[0]
    unknowns_mat = lin_sys_mat[:, :num_independent_eq]
    known_term_vec = lin_sys_mat[:, -1]

    # Solve the linear system (upper triangular)
    arc_biases = solve(unknowns_mat, known_term_vec)

    return {arc_id: bias for arc_id, bias in zip(list_all_used_arcs, arc_biases)}


def _estimate_bias(
    df: pl.DataFrame,
    receiver_position: tuple[float, float, float],
    max_degree: int,
    n_epochs: int,
    h_ipp: float,
) -> dict[str, float]:
    """
    Estimate arc-level TEC biases. The function preprocesses the data, computes vTEC,
    mapping function and MoDip parameters, then applies a calibration procedure to
    estimate biases per arc.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with GNSS observations.
    receiver_position : tuple[float, float, float]
        Receiver position in ECEF coordinates (x, y, z) [meters].
    max_degree : int, optional
        Maximum degree of polynomial expansion.
    n_epochs : int, optional
        Number of epochs per batch for calibration.
    h_ipp : float, optional
        Height of the IPP [m].

    Returns
    -------
    dict[str, float]
        Dictionary mapping arc identifiers to estimated biases.
    """
    df_clean = _preprocessing(df, receiver_position=receiver_position, h_ipp=h_ipp)
    return _gg_calibration(df_clean, batch_length_mins=n_epochs, max_degree=max_degree)


def _estimate_veq_batches(
    df_clean: pl.DataFrame,
    max_degree: int = 3,
    batch_length_mins: int = 15,
) -> dict[datetime.datetime, float]:
    """
    Internal helper to extract the zenithal coefficient (vEq) per batch.

    Evaluates the ionospheric model at the receiver's location (delta=0).

    Parameters
    ----------
    df_clean : pl.DataFrame
        Preprocessed DataFrame including estimated biases.
    max_degree : int
        Polynomial degree.
    batch_length_mins : int
        Batch temporal window (in minutes).

    Returns
    -------
    dict[datetime.datetime, float]
        Zenithal TEC values at batch centers.
    """
    df_clean = df_clean.sort("epoch")
    n_coeffs = max_degree + 3

    min_time, max_time = df_clean["epoch"].min(), df_clean["epoch"].max()
    batch_starts = pl.datetime_range(
        start=min_time, end=max_time, interval=f"{batch_length_mins}m", eager=True
    )

    veq_results = {}
    interval_td = datetime.timedelta(minutes=batch_length_mins)

    for start_time in batch_starts:
        end_time = start_time + interval_td
        batch_center = start_time + interval_td / 2
        batch_df = df_clean.filter(
            (pl.col("epoch") >= start_time) & (pl.col("epoch") < end_time)
        )

        if batch_df.height < n_coeffs:
            continue

        epochs_np = batch_df["epoch"].to_numpy()
        batch_center_np = np.datetime64(batch_center)
        dt_utc_hours = (epochs_np - batch_center_np) / np.timedelta64(1, "h")
        lon_ipp_np = batch_df["lon_ipp"].to_numpy()
        lon_rec_np = batch_df["lon_rec"].to_numpy()
        delta_lt = dt_utc_hours + (lon_ipp_np - lon_rec_np) / 15.0

        # Basis functions for the current batch
        P = _polynomial_expansion(
            batch_df["modip_ipp"].to_numpy(),
            batch_df["modip_rec"].to_numpy(),
            lon_ipp_np,
            lon_rec_np,
            delta_lt,
            max_degree,
        )

        # De-biased observations for model fitting: (levelled - bias) * mapping
        target = (batch_df["gflc_levelled"] - batch_df["bias"]).to_numpy() * batch_df[
            "mapping"
        ].to_numpy()

        # Local model solving via QR
        triang_mat = qr(np.hstack([P, target.reshape(-1, 1)]), mode="r")[0]
        R = triang_mat[:n_coeffs, :n_coeffs]
        rhs = triang_mat[:n_coeffs, -1]

        try:
            coeffs = solve_triangular(R, rhs)
            # vEq is the C0 coefficient (zenithal equivalent)
            veq_results[start_time + interval_td / 2] = coeffs[0]
        except:
            continue

    return veq_results


def calculate_tec(
    df: pl.DataFrame,
    ctx: GNSSContext,
    max_polynomial_degree: int = 3,
    batch_size_epochs: int = 30,
) -> pl.DataFrame:
    """
    Compute slant and vertical TEC (sTEC, vTEC) after per-arc bias estimation.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing GNSS observations, including:
        - gflc_levelled: leveled sTEC measurements
        - id_arc_valid: valid arc identifiers
        - ele: satellite elevation angles
    ctx : GNSSContext
        Execution context containing receiver position and IPP height.
    max_polynomial_degree : int
        Maximum degree of polynomial expansion used in calibration.
    batch_size_epochs : int
        Number of epochs per batch for calibration.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns:
        - bias: estimated arc-level bias
        - stec: bias-corrected slant TEC
        - vtec: vertical TEC after mapping function correction
    """
    if df.get_column("id_arc_valid").null_count() == df.shape[0]:
        warnings.warn(
            "No valid arcs found in the DataFrame, calibration cannot be performed. "
            "Try adjusting the arcs extraction parameters or try with another constellation.",
        )
        return df.with_columns(
            [
                pl.lit(None).alias("bias"),
                pl.lit(None).alias("stec"),
                pl.lit(None).alias("vtec"),
            ]
        )

    offset_by_arc = _estimate_bias(
        df=df,
        receiver_position=ctx.receiver_pos,
        max_degree=max_polynomial_degree,
        n_epochs=batch_size_epochs,
        h_ipp=ctx.h_ipp,
    )

    map_ = pl.DataFrame(
        {
            "id_arc_valid": list(offset_by_arc.keys()),
            "bias": list(offset_by_arc.values()),
        }
    )

    return (
        df.join(
            map_,
            on="id_arc_valid",
            how="left",
        )
        .with_columns((pl.col("gflc_levelled") - pl.col("bias")).alias("stec"))
        .with_columns(
            (pl.col("stec") * _mapping_function(pl.col("ele"), h_ipp=ctx.h_ipp)).alias(
                "vtec"
            )
        )
    )


def calculate_vertical_equivalent(
    df_calibrated: pl.DataFrame,
    ctx: GNSSContext,
    max_polynomial_degree: int = 3,
    batch_size_epochs: int = 30,
) -> pl.DataFrame:
    """
    Compute the Vertical Equivalent (VEq) series at the station zenith,
    broadcasting a unique station-wide vertical TEC value for each epoch.

    Parameters
    ----------
    df_calibrated : pl.DataFrame
        DataFrame already processed by calculate_tec().
    ctx : GNSSContext
        Context with receiver position and IPP height.
    max_polynomial_degree : int, optional
        Same degree used for calibration. Default is 3.
    batch_size_epochs : int, optional
        Same batch size used for calibration. Default is 30.

    Returns
    -------
    pl.DataFrame
        DataFrame with an additional 'veq' column.
    """
    # Align MoDip and geometry for batch solving
    df_prep = _preprocessing(
        df_calibrated, receiver_position=ctx.receiver_pos, h_ipp=ctx.h_ipp
    )

    veq_dict = _estimate_veq_batches(
        df_prep, max_degree=max_polynomial_degree, batch_length_mins=batch_size_epochs
    )

    if not veq_dict:
        return df_calibrated.with_columns(pl.lit(None).alias("veq"))

    # Ensure schema consistency for join (e.g., UTC timezone)
    epoch_dtype = df_calibrated.schema["epoch"]

    veq_batches_df = (
        pl.DataFrame(
            {"epoch": list(veq_dict.keys()), "veq_raw": list(veq_dict.values())}
        )
        .with_columns(pl.col("epoch").cast(epoch_dtype))
        .sort("epoch")
    )

    # Single-row timeline to prevent duplication during interpolation
    unique_timeline = df_calibrated.select("epoch").unique().sort("epoch")

    # Distribute zenithal value to all satellites per epoch
    interp_values = (
        unique_timeline.join_asof(
            veq_batches_df,
            on="epoch",
            strategy="nearest",
            tolerance="1m",
        )
        .sort("epoch")
        .with_columns(
            pl.col("veq_raw")
            .interpolate(method="linear")
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias("veq")
        )
        .select(["epoch", "veq"])
    )

    return df_calibrated.join(interp_values, on="epoch", how="left")
