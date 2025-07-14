import polars as pl

from typing import Optional, Literal

# Costanti fisiche
C = 299792458.0  # Velocità della luce in m/s

# Supported GNSS systems
System = Literal["G", "R", "E", "C"]

# Central frequencies for observables bands
FREQ_BANDS: dict[str, dict[str, float]] = {
    "G": {"L1": 1575.42e6, "L2": 1227.60e6, "L5": 1176.45e6},
    "R": {
        "L1": lambda n: (1602 + n * 0.5625) * 1e6,
        "L2": lambda n: (1246 + n * 0.4375) * 1e6,
    },
    "E": {"L1": 1575.42e6, "L5": 1176.45e6},
    "C": {"B1": 1561.098e6, "B2": 1207.14e6, "B3": 1268.52e6},
}

# Bands → Observables mapping
OBS_MAPPING: dict[System, dict[str, str]] = {
    "G": {"band1": "L1", "band2": "L2", "obs1": "L1C", "obs2": "L2W"},
    "R": {"band1": "L1", "band2": "L2", "obs1": "L1C", "obs2": "L2C"},
    "E": {"band1": "L1", "band2": "L5", "obs1": "L1X", "obs2": "L5X"},
    "C": {"band1": "B1", "band2": "B2", "obs1": "L1X", "obs2": "L7I"},
}


def calculate_gflc(phase1: float, phase2: float, freq1: float, freq2: float) -> float:
    """Calcola la combinazione geometry-free"""
    lambda1 = C / freq1
    lambda2 = C / freq2
    pr_to_tec = (1 / 40.308) * (freq1**2 * freq2**2) / (freq1**2 - freq2**2) / 1e16
    return (phase1 * lambda1 - phase2 * lambda2) * pr_to_tec


def process_observations(
    df: pl.DataFrame,
    rinex_version: str,
    systems: Optional[list[Literal["G", "R", "E", "C", "J"]]] = None,
    glonass_freq: Optional[dict[str, int]] = None,
) -> pl.DataFrame:
    version_key = rinex_version.split(".")[0]
    results = []

    for system in systems:
        cfg = OBS_MAPPING.get(version_key, OBS_MAPPING["3"]).get(system, {})
        phase_bands = cfg.get("phase", {})

        if not phase_bands:
            continue

        # Ottieni i nomi esatti delle osservabili
        obs_band1 = phase_bands.get("L1")  # Es. 'L1X' per Galileo
        obs_band2 = phase_bands.get("L5")  # Es. 'L5X' per Galileo

        if not obs_band1 or not obs_band2:
            continue

        # Filtra i dati
        system_df = df.filter(
            (pl.col("sv").str.starts_with(system))
            & (pl.col("observable").is_in([obs_band1, obs_band2]))
        )

        if system_df.is_empty():
            print(
                f"Nessun dato di fase trovato per {system} con osservabili {obs_band1}, {obs_band2}"
            )
            continue

        # Pivota il DataFrame
        pivot_df = system_df.pivot(
            values="value",
            index=["epoch", "sv"],
            columns="observable",
            aggregate_function="first",
        )

        # Verifica che le colonne richieste esistano
        if obs_band1 not in pivot_df.columns or obs_band2 not in pivot_df.columns:
            print(f"Osservabili mancanti nel pivot: {obs_band1} o {obs_band2}")
            continue

        # Calcola frequenze
        if system == "R" and glonass_freq:
            pivot_df = pivot_df.with_columns(
                pl.col("sv").map_dict(glonass_freq).alias("freq_number")
            )
            freq1 = FREQ_BANDS["R"]["L1"](pl.col("freq_number"))
            freq2 = FREQ_BANDS["R"]["L2"](pl.col("freq_number"))
        else:
            freq1 = FREQ_BANDS[system]["L1"]
            freq2 = FREQ_BANDS[system]["L5"]

        # Calcola GFLC
        pivot_df = pivot_df.with_columns(
            (calculate_gflc(pl.col(obs_band1), pl.col(obs_band2), freq1, freq2)).alias(
                "GFLC"
            )
        )

        results.append(pivot_df)

    return pl.concat(results) if results else pl.DataFrame()
