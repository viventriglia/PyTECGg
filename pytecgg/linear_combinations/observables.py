import re
from dataclasses import dataclass
from itertools import product
from typing import Literal, Optional

import polars as pl

from pytecgg.context import SUPPORTED_SYSTEMS

from .constants import (
    BAND_PAIR_PRIORITY_BY_MODE,
    CODE_CHAN_PRIORITY,
    FREQ_BANDS,
    PHASE_CHAN_PRIORITY,
)

SelectionMode = Literal["quality", "availability", "legacy"]

_band_re = re.compile(r"^([A-Za-z]\d+)")
_band_key_re = re.compile(r"^[A-Za-z](\d+)")


@dataclass(frozen=True)
class SelectedSignalPair:
    system: str
    band1: str
    band2: str
    phase1: str
    phase2: str
    code1: str
    code2: str
    joint_coverage: int


def _extract_band(obs: str) -> Optional[str]:
    m = _band_re.match(obs)
    return m.group(1) if m else None


def _extract_band_key(obs: str) -> Optional[str]:
    m = _band_key_re.match(obs)
    return f"L{m.group(1)}" if m else None


def _extract_suffix(obs: str) -> str:
    band = _extract_band(obs)
    return obs[len(band) :] if band else ""


def _suffix_rank(obs: str, suffix_priority: list[str]) -> int:
    suffix = _extract_suffix(obs)
    if not suffix:
        return 0
    head = suffix[0]
    return suffix_priority.index(head) if head in suffix_priority else 99


def _normalise_system(system: str) -> str:
    system_up = system.upper()
    return SUPPORTED_SYSTEMS.get(system_up, system_up)


def _band_frequency(system: str, band: str) -> float:
    value = FREQ_BANDS[system][band]
    return value(0) if callable(value) else value


def _normalise_band_pair(system: str, band_pair: tuple[str, str]) -> tuple[str, str]:
    band1, band2 = (band_pair[0].upper(), band_pair[1].upper())
    if band1 == band2:
        raise ValueError("Band overrides must specify two distinct bands")
    if band1 not in FREQ_BANDS[system] or band2 not in FREQ_BANDS[system]:
        raise ValueError(
            f"Invalid band override {band_pair} for system '{system}'. "
            f"Expected bands among {list(FREQ_BANDS[system])}."
        )
    return tuple(sorted((band1, band2), key=lambda band: _band_frequency(system, band)))


def _validate_selection_mode(selection_mode: SelectionMode) -> None:
    if selection_mode not in BAND_PAIR_PRIORITY_BY_MODE:
        raise ValueError(
            f"Unsupported selection_mode '{selection_mode}'. "
            "Expected one of: 'quality', 'availability', 'legacy'."
        )


def _complete_rows(df_pivot: pl.DataFrame, columns: list[str]) -> int:
    condition = pl.lit(True)
    for column in columns:
        condition = condition & pl.col(column).is_not_null()
    return df_pivot.filter(condition).height


def _candidate_columns(
    available: list[str], band: str, observable_type: Literal["phase", "code"]
) -> list[str]:
    if observable_type == "phase":
        return [
            obs
            for obs in available
            if obs.startswith("L") and _extract_band_key(obs) == band
        ]
    return [
        obs
        for obs in available
        if obs[:1] in {"C", "P"} and _extract_band_key(obs) == band
    ]


def _select_best_combo_for_band_pair(
    df_pivot: pl.DataFrame,
    band1: str,
    band2: str,
    selection_mode: SelectionMode,
) -> Optional[SelectedSignalPair]:
    available = [col for col in df_pivot.columns if col not in {"epoch", "sv"}]
    phase1_candidates = _candidate_columns(available, band1, "phase")
    phase2_candidates = _candidate_columns(available, band2, "phase")
    code1_candidates = _candidate_columns(available, band1, "code")
    code2_candidates = _candidate_columns(available, band2, "code")

    if not all(
        [phase1_candidates, phase2_candidates, code1_candidates, code2_candidates]
    ):
        return None

    best_choice = None
    best_key = None

    for phase1, phase2, code1, code2 in product(
        phase1_candidates, phase2_candidates, code1_candidates, code2_candidates
    ):
        joint_coverage = _complete_rows(df_pivot, [phase1, phase2, code1, code2])
        if joint_coverage == 0:
            continue

        total_coverage = sum(
            int(df_pivot.get_column(column).is_not_null().sum())
            for column in [phase1, phase2, code1, code2]
        )
        suffix_penalty = (
            _suffix_rank(phase1, PHASE_CHAN_PRIORITY)
            + _suffix_rank(phase2, PHASE_CHAN_PRIORITY)
            + _suffix_rank(code1, CODE_CHAN_PRIORITY)
            + _suffix_rank(code2, CODE_CHAN_PRIORITY)
        )

        if selection_mode == "availability":
            score = (joint_coverage, total_coverage, -suffix_penalty)
        else:
            score = (joint_coverage, -suffix_penalty, total_coverage)

        if best_key is None or score > best_key:
            best_key = score
            best_choice = SelectedSignalPair(
                system="",
                band1=band1,
                band2=band2,
                phase1=phase1,
                phase2=phase2,
                code1=code1,
                code2=code2,
                joint_coverage=joint_coverage,
            )

    return best_choice


def _resolve_band_pairs(
    system_symbol: str,
    selection_mode: SelectionMode,
    band_override: tuple[str, str] | None,
) -> list[tuple[str, str]]:
    if band_override is not None:
        return [_normalise_band_pair(system_symbol, band_override)]
    return BAND_PAIR_PRIORITY_BY_MODE.get(selection_mode, {}).get(system_symbol, [])


def select_observable_pair_from_pivot(
    df_pivot: pl.DataFrame,
    system: str,
    selection_mode: SelectionMode = "quality",
    band_override: tuple[str, str] | None = None,
) -> Optional[SelectedSignalPair]:
    """
    Select a coherent dual-frequency band pair from a pivoted observation table.

    Parameters
    ----------
    df_pivot : pl.DataFrame
        Wide-form DataFrame indexed by ['epoch', 'sv'] with observables as columns.
    system : str
        GNSS system identifier ("G", "R", "E", "C") or full name.
    selection_mode : {"quality", "availability", "legacy"}
        Policy used to rank viable band pairs.
    band_override : tuple[str, str] | None
        Optional user-forced band pair. Bands can be provided in any order, for
        example ("L1", "L5") or ("L5", "L1").

    Returns
    -------
    SelectedSignalPair | None
        The selected coherent band pair with phase/code observables, or None if no
        viable combination is available.
    """
    _validate_selection_mode(selection_mode)
    system_symbol = _normalise_system(system)

    if df_pivot.is_empty():
        return None

    band_pairs = _resolve_band_pairs(system_symbol, selection_mode, band_override)
    if not band_pairs:
        return None

    if selection_mode == "legacy" or band_override is not None:
        for band1, band2 in band_pairs:
            best_choice = _select_best_combo_for_band_pair(
                df_pivot, band1, band2, selection_mode
            )
            if best_choice is not None:
                return SelectedSignalPair(
                    system=system_symbol,
                    band1=best_choice.band1,
                    band2=best_choice.band2,
                    phase1=best_choice.phase1,
                    phase2=best_choice.phase2,
                    code1=best_choice.code1,
                    code2=best_choice.code2,
                    joint_coverage=best_choice.joint_coverage,
                )
        return None

    ranked_choices = []
    total_rows = max(df_pivot.height, 1)

    for rank, (band1, band2) in enumerate(band_pairs):
        best_choice = _select_best_combo_for_band_pair(
            df_pivot, band1, band2, selection_mode
        )
        if best_choice is None:
            continue

        coverage_ratio = best_choice.joint_coverage / total_rows
        priority_bonus = (len(band_pairs) - rank) / (len(band_pairs) * 10)

        if selection_mode == "quality":
            score = (
                coverage_ratio + priority_bonus,
                best_choice.joint_coverage,
                -rank,
            )
        else:
            score = (coverage_ratio, best_choice.joint_coverage, -rank)

        ranked_choices.append(
            (
                score,
                SelectedSignalPair(
                    system=system_symbol,
                    band1=best_choice.band1,
                    band2=best_choice.band2,
                    phase1=best_choice.phase1,
                    phase2=best_choice.phase2,
                    code1=best_choice.code1,
                    code2=best_choice.code2,
                    joint_coverage=best_choice.joint_coverage,
                ),
            )
        )

    if not ranked_choices:
        return None

    return max(ranked_choices, key=lambda item: item[0])[1]


def select_observable_pair(
    obs_data: pl.DataFrame,
    system: str,
    rinex_version: str,
    selection_mode: SelectionMode = "quality",
    band_override: tuple[str, str] | None = None,
) -> Optional[SelectedSignalPair]:
    """
    Select a coherent dual-frequency band pair and the best phase/code observables
    within that pair for a given GNSS system.

    Parameters
    ----------
    obs_data : pl.DataFrame
        DataFrame with columns ['epoch', 'sv', 'observable', 'value'].
    system : str
        GNSS system identifier ("G", "R", "E", "C") or full name.
    rinex_version : str
        RINEX version ("2" or "3"). Present for API compatibility and future policies.
    selection_mode : {"quality", "availability", "legacy"}
        Policy used to rank viable band pairs.
    band_override : tuple[str, str] | None
        Optional user-forced band pair. Bands can be provided in any order, for
        example ("L1", "L5") or ("L5", "L1").

    Returns
    -------
    SelectedSignalPair | None
        The selected coherent band pair with phase/code observables, or None if no
        viable combination is available.
    """
    del rinex_version

    if "system" not in obs_data.columns:
        obs_data = obs_data.with_columns([pl.col("sv").str.slice(0, 1).alias("system")])

    system_symbol = _normalise_system(system)
    df_sys = obs_data.filter(pl.col("system") == system_symbol)
    if df_sys.is_empty():
        return None

    df_pivot = df_sys.pivot(
        values="value",
        index=["epoch", "sv"],
        on="observable",
        aggregate_function="first",
    )

    return select_observable_pair_from_pivot(
        df_pivot,
        system=system_symbol,
        selection_mode=selection_mode,
        band_override=band_override,
    )


def retrieve_observable_pairs(
    obs_data: pl.DataFrame,
    system: str,
    rinex_version: str,
    prefer_by_suffix: bool = True,
    df_for_counts: pl.DataFrame | None = None,
    selection_mode: SelectionMode = "quality",
    band_override: tuple[str, str] | None = None,
) -> Optional[tuple[tuple[str, str], tuple[str, str]]]:
    """
    Backward-compatible wrapper returning only the selected phase/code observable names.
    """
    del prefer_by_suffix, df_for_counts
    selection = select_observable_pair(
        obs_data,
        system=system,
        rinex_version=rinex_version,
        selection_mode=selection_mode,
        band_override=band_override,
    )
    if selection is None:
        return None
    return (selection.phase1, selection.phase2), (selection.code1, selection.code2)
