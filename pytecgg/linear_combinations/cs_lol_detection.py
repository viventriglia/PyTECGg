from typing import Literal, Optional
from datetime import timedelta

import polars as pl
import numpy as np

from .constants import FREQ_BANDS, C, PHASE_FREQ_PRIORITY


def _infer_temporal_resolution(df: pl.DataFrame) -> timedelta:
    epochs = df.get_column("epoch")
    return epochs.unique().sort().diff().drop_nulls().mode()[0]


def detect_cs_lol(
    df: pl.DataFrame,
    system: Literal["G", "E", "C", "R"],
    threshold_std: float = 5.0,
    threshold_abs: float = 5.0,
    max_gap: timedelta = None,
    glonass_freq: Optional[dict[str, int]] = None,
    satellite_freq: Optional[dict[str, tuple[float, float]]] = None,
    f1: Optional[float] = None,
    f2: Optional[float] = None,
) -> pl.DataFrame:
    """
    Detect cycle slip (CS) and loss-of-lock (LoL) in GNSS observations using
    Melbourne-Wübbena combination

    Parameters:
        df (pl.DataFrame): DataFrame containing GNSS observation data with Melbourne-Wübbena values
        system (Literal["G", "E", "C", "R"]): GNSS system identifier
        threshold_std (float): Threshold in standard deviations for CS detection (default: 5.0)
        threshold_abs (float): Absolute threshold in meters for CS detection (default: 5.0)
        max_gap (timedelta): Maximum allowed time gap between observations before declaring
            LoL (default: inferred from df's temporal resolution)
        glonass_freq (dict[str, int], optional): Frequency mapping for GLONASS satellites
        satellite_freq (dict[str, tuple[float, float]], optional): Per-satellite
            frequency mapping for non-GLONASS systems, in Hz
        f1 (float or pl.Series, optional): Frequency of the first band (Hz)
        f2 (float or pl.Series, optional): Frequency of the second band (Hz)

    Returns:
        pl.DataFrame: DataFrame with CS and LoL detections, containing:
            - epoch: Observation timestamp
            - sv: Satellite PRN number
            - is_loss_of_lock: Boolean indicating LoL (or satellite setting)
            - is_cycle_slip: Boolean indicating CS (None, when LoL occurs)

    Notes:
        This function implements a CS detection algorithm based on the Melbourne-Wübbena
        combination, following the methodology described in:
        ESA Navipedia (https://gssc.esa.int/navipedia/index.php?title=Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination)
    """
    if system == "R":
        if glonass_freq is None:
            raise ValueError("glonass_freq is required for GLONASS processing")
        valid_svs = [
            sv
            for sv in df.get_column("sv").unique()
            if glonass_freq.get(sv) is not None
        ]
        if not valid_svs:
            return pl.DataFrame()
    elif satellite_freq is not None:
        valid_svs = [
            sv for sv in df.get_column("sv").unique() if satellite_freq.get(sv) is not None
        ]
        if not valid_svs:
            return pl.DataFrame()
    else:
        valid_svs = df.get_column("sv").unique()

    if max_gap is None:
        max_gap = _infer_temporal_resolution(df)

    result = []

    result_schema = {
        "epoch": df.schema["epoch"],
        "sv": df.schema["sv"],
        "is_loss_of_lock": pl.Boolean,
        "is_cycle_slip": pl.Boolean,
    }

    if system in ["G", "E", "C"] and satellite_freq is None:
        if f1 is None or f2 is None:
            # Fallback to defaults from constants if frequencies are not provided
            band2, band1 = PHASE_FREQ_PRIORITY[system][0]
            f1 = FREQ_BANDS[system][band1]
            f2 = FREQ_BANDS[system][band2]
        lambda_w_const = C / (f1 - f2)
        sigma_0_const = lambda_w_const / 2

    for sv in valid_svs:
        df_sv = df.filter(pl.col("sv") == sv).sort("epoch")

        if system == "R":
            if sv not in glonass_freq:
                raise ValueError(f"Missing GLONASS frequency number for {sv}")
            n = glonass_freq[sv]
            f1 = FREQ_BANDS["R"]["L1"](n)
            f2 = FREQ_BANDS["R"]["L2"](n)
            lambda_w = C / (f1 - f2)
            sigma_0 = lambda_w / 2
        elif satellite_freq is not None:
            if sv not in satellite_freq:
                continue
            f1_sv, f2_sv = satellite_freq[sv]
            lambda_w = C / (f1_sv - f2_sv)
            sigma_0 = lambda_w / 2
        else:
            sigma_0 = sigma_0_const

        k = 0
        m_mw = None
        sigma2_mw = None
        last_valid_epoch = None
        m_prev = None

        for row in df_sv.iter_rows(named=True):
            epoch = row["epoch"]
            mw = row["mw"]

            # Melbourne-Wübbena value missing: loss of lock
            if mw is None:
                result.append(
                    {
                        "epoch": epoch,
                        "sv": sv,
                        "is_loss_of_lock": True,
                        "is_cycle_slip": None,
                    }
                )
                k = 0
                m_mw = None
                sigma2_mw = None
                m_prev = None
                continue

            # Gap in the current epoch: loss of lock/satellite setting
            is_loss_of_lock = False
            if last_valid_epoch is not None:
                gap = epoch - last_valid_epoch
                if gap > max_gap:
                    is_loss_of_lock = True
                    result.append(
                        {
                            "epoch": epoch,
                            "sv": sv,
                            "is_loss_of_lock": True,
                            "is_cycle_slip": None,
                        }
                    )
                    k = 0
                    m_mw = None
                    sigma2_mw = None
                    m_prev = None

            if is_loss_of_lock:
                last_valid_epoch = epoch
                continue

            # First valid point after a gap or NaN
            if k == 0 or m_mw is None or sigma2_mw is None or m_prev is None:
                m_mw = mw
                sigma2_mw = sigma_0**2
                is_cycle_slip = False
            else:
                sigma = np.sqrt(sigma2_mw)
                deviation = np.abs(mw - m_mw)
                delta = np.abs(mw - m_prev)

                if (deviation > threshold_std * sigma) and (delta > threshold_abs):
                    is_cycle_slip = True
                    k = 0
                    m_mw = None
                    sigma2_mw = None
                    m_prev = None
                else:
                    is_cycle_slip = False
                    m_prev = m_mw
                    m_mw = (k * m_mw + mw) / (k + 1)
                    sigma2_mw = (k * sigma2_mw + (mw - m_prev) ** 2) / (k + 1)

            result.append(
                {
                    "epoch": epoch,
                    "sv": sv,
                    "is_loss_of_lock": is_loss_of_lock,
                    "is_cycle_slip": is_cycle_slip if not is_loss_of_lock else None,
                }
            )

            last_valid_epoch = epoch
            if not is_cycle_slip and not is_loss_of_lock:
                k += 1
                m_prev = mw

    return pl.DataFrame(result, schema=result_schema)
