# Preprocessing & Geometry 🛰️

This stage transforms raw observations into high-quality, geometrically-referenced time series, determining exactly where the ionospheric observations took place.

## Ephemeris Preparation 📐

The first step is filtering global navigation messages and preparing ephemerides. Beyond preparing data for orbital propagation, this function enriches the `GNSSContext` with GLONASS `glonass_channels` (if `R` is among the requested constellations).

```python
from pytecgg.satellites import prepare_ephemeris

ephem_dict = prepare_ephemeris(nav_dict, ctx=ctx)
```

## Signal Processing 🔎

Starting from the basic observables and given a `GNSSContext`, `PyTECGg` can compute the following [linear combinations](https://gssc.esa.int/navipedia/index.php/Combination_of_GNSS_Measurements), useful for removing biases or isolating physical effects:

- [Geometry-Free](https://gssc.esa.int/navipedia/index.php/Detector_based_in_carrier_phase_data:_The_geometry-free_combination) Linear Combination (GFLC), sensitive to ionospheric effects.
- [Ionosphere-Free](https://gssc.esa.int/navipedia/index.php/Ionosphere-free_Combination_for_Dual_Frequency_Receivers) Linear Combination (IFLC), used to eliminate the ionospheric delay.
- [Melbourne-Wübbena](https://gssc.esa.int/navipedia/index.php/Detector_based_in_code_and_carrier_phase_data:_The_Melbourne-W%C3%BCbbena_combination) (MW) combination, useful for cycle-slip detection and ambiguity resolution.

!!! info "BeiDou Selection Is Now Per Satellite"
    GPS, Galileo, and GLONASS still use one selected dual-frequency pair per constellation, but BeiDou is handled satellite by satellite. This lets mixed BDS-2/BDS-3 observation files keep the best valid pair for each `Cxx` satellite instead of forcing one pair across the whole constellation.

```python
from pytecgg.linear_combinations import calculate_linear_combinations
from pytecgg.tec_calibration import extract_arcs

# Compute selected linear combinations
df_lc = calculate_linear_combinations(
    df_obs,
    ctx=ctx,
    combinations=["gflc_phase", "gflc_code", "mw"],
    selection_mode="quality",
    band_overrides={"G": ("L1", "L5")},
)

# Identify continuous arcs, detect cycle slips
# and loss-of-lock events
df_arcs = extract_arcs(
    df=df_lc,
    ctx=ctx,
    threshold_abs=10,
    threshold_std=10,
    threshold_jump=5
)
```

Processing parameters and options:

- `selection_mode`: ranking policy for automatic dual-frequency selection. Use `quality` to prefer higher-quality band pairs when coverage is comparable, `availability` to maximise complete observations, or `legacy` to keep legacy-style priorities. For BeiDou, this policy is applied per satellite rather than once per constellation.
- `band_overrides`: optional per-system override for the selected band pair, for example `{"G": ("L1", "L5")}` or `{"BEIDOU": ("L1", "L7")}`. For BeiDou, the override is still evaluated per satellite, so satellites without that pair are skipped.
- `combinations`: defines which signals to compute. Options include GFLC (`gflc_phase`, `gflc_code`), IFLC (`iflc_phase`, `iflc_code`), and MW (`mw`). When `gflc_code` is requested, the output also contains a `gflc_code_obs_used` column with the two code observable names (e.g. `"C1C,C2W"`) used to compute it; for BeiDou the value can vary per satellite.
- `extract_arcs`: outputs a Polars `DataFrame` containing unique arc identifiers (`id_arc_valid`) and arc-levelled GFLC values, essential for accurate bias estimation.
- `threshold_abs` & `threshold_std`: absolute and standard deviation thresholds used on the MW combination to detect cycle slips.
- `threshold_jump`: tolerance for detecting and correcting residual jumps within an arc.
- `max_gap`: can be explicitly set or automatically inferred from the data; if the time gap between consecutive epochs becomes too large, a loss-of-lock is declared.


## Orbital Propagation 🌍

To locate the ionospheric samples, we must first compute the satellite positions in the ECEF frame.

```python
from pytecgg.satellites import satellite_coordinates

# Compute satellites' ECEF coordinates
df_coords = satellite_coordinates(
    sv_ids=df_arcs["sv"],
    epochs=df_arcs["epoch"],
    ephem_dict=ephem_dict
)

# Join coordinates with the observation data
df_geom = df_arcs.join(df_coords, on=["sv", "epoch"], how="left")
```

`satellite_coordinates` provides the `X`, `Y`, `Z` positions of each satellite for every observed epoch.

## Ionospheric Pierce Point (IPP) 📌

The IPP is the theoretical intersection between the satellite–receiver line of sight and a thin-shell ionospheric model at the altitude defined in the `GNSSContext` (default: 350 km).

```python
from pytecgg.satellites import calculate_ipp

df_final = calculate_ipp(df_geom, ctx=ctx, min_elevation=20)
```

This function appends geographic coordinates (`lat_ipp`, `lon_ipp`) and the mapping function values to the Polars `DataFrame`. The `min_elevation` argument sets a cut-off angle (in degrees) to filter satellite elevation.

!!! info "Elevation Masking"
    Filtering out elevations below 20° is highly recommended to feed cleaner data to the calibration model, as low-elevation signals are more susceptible to multipath and noise.

## Wrapping up 🏁

After completing all the preprocessing steps, your `DataFrame` is ready for the calibration engine: it now contains a rich set of features, that you can inspect using Polars selection methods.

As an example, we may want to inspect arc metadata, like the unique identifier for each continuous observation sequence (`id_arc_valid`); the corrected and levelled GFLC (`gflc_levelled`, in TECu); and some geometric features, like the geographic IPP location (`lat_ipp`, `lon_ipp`) and the viewing geometry from the station (`azi`, `ele`).

=== "Code"
    ```python
    import polars as pl

    df_final.filter(
        pl.col("id_arc_valid").is_not_null()
    ).select(
        [
            "epoch",
            "sv",
            "id_arc_valid",
            "gflc_levelled",
            "lat_ipp",
            "lon_ipp",
            "azi",
            "ele",
        ]
    ).head()
    ```
=== "Example Output"
    | epoch | sv | id_arc_valid | gflc_levelled | lat_ipp | lon_ipp | azi | ele |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | 00:00:00 | E02 | "bneu_e02_001" | 42.39 | 20.40 | 96.31 | 257.73 | 26.55 |
    | 00:00:00 | E03 | "bneu_e03_001" | 42.11 | 23.51 | 99.84 | 314.36 | 47.16 |
    | 00:00:00 | E05 | "bneu_e05_001" | 44.97 | 20.75 | 101.93 | 179.50 | 73.10 |
    | 00:00:00 | E16 | "bneu_e16_001" | 11.03 | 23.88 | 102.24 | 27.57 | 52.26 |
    | 00:00:00 | E25 | "bneu_e25_001" | 34.64 | 22.97 | 100.76 | 321.15 | 59.68 |


--8<-- "includes/abbreviations.md"
