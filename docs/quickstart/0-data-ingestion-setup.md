# Data Ingestion & Setup 🏗️

The first step in any TEC analysis is loading the raw data and preparing the execution environment. `PyTECGg` uses a high-performance Rust backend to parse RINEX files, delivering data directly into a [Polars `DataFrame`](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html) for maximum efficiency.

## Parse RINEX files — fast ⚡

The `parsing` module provides two main functions: `read_rinex_obs` for observations and `read_rinex_nav` for navigation messages.

```python
from pathlib import Path
from pytecgg.parsing import read_rinex_nav, read_rinex_obs

# Define paths to your local files
NAV_PATH = Path("./path/to/your/nav_file.rnx")
OBS_PATH = Path("./path/to/your/obs_file.rnx")

# Load navigation data
nav_dict = read_rinex_nav(NAV_PATH)

# Load observations and metadata
df_obs, rec_pos, rinex_version = read_rinex_obs(OBS_PATH)

# Derive a standard 4-character receiver name from the filename
rec_name = OBS_PATH.name[:4].lower()
```

!!! note "Metadata Extraction"
    `read_rinex_obs` returns a `tuple`: along with the observation `DataFrame`, it automatically extracts the approximate ECEF position of the receiver and the RINEX version directly from the file header.

## Setting the Context ⚙️

The `GNSSContext` acts as a centralised configuration and state manager for the entire processing pipeline. It ensures that coordinate validation, constellation normalisation, and metadata management are handled consistently across all modules.

```python
from pytecgg import GNSSContext

ctx = GNSSContext(
    receiver_pos=rec_pos,
    receiver_name=rec_name,
    rinex_version=rinex_version,
    h_ipp=350_000,
    systems=['G', 'E'],
)
```

Context parameters are:

- `receiver_pos`, `receiver_name`, `rinex_version`: receiver position in ECEF coordinates (in meters), station identifier, and RINEX version string, respectively.
- `h_ipp`: the height (in meters) of the Ionospheric Pierce Point (IPP). This defines the altitude of the theoretical "thin shell" used for vertical mapping (typically 350 km).
- `systems`: a list of constellation identifiers to be processed. `PyTECGg` supports `GPS` (`G`), `Galileo` (`E`), `GLONASS` (`R`), and `BeiDou` (`C`). You can provide full names (*e.g.* `Galileo`) or symbols (*e.g.* `E`).

Initialising this object is mandatory: it serves as the required argument for all subsequent processing steps, including cycle-slip detection, IPP calculation, and bias estimation.

!!! info "Internal State Management"
    The `GNSSContext` also maintains internal attributes like `glonass_channels` and `freq_meta`: these are automatically populated during the ephemeris and signal processing stages. For GPS and Galileo, `freq_meta` is typically a single frequency pair per constellation; for GLONASS and BeiDou it may store per-satellite frequency metadata used downstream during TEC arc extraction and levelling.

## RINEX utilities 🔧

`PyTECGg` also provides utility functions to perform quick health checks on the parsed datasets and automate data retrieval.

### Data Diagnostics

Once data is loaded, you can use `summarise_rinex_data` to verify signal availability and navigation coverage.

=== "Code"
    ```python
    from pytecgg.utils import summarise_rinex_data

    summarise_rinex_data(obs_data, nav_dict)
    ```
=== "Example Output"
    ```text
    =======================================================
    🛰️  GNSS RINEX DATA SUMMARY
    =======================================================

    📅 Temporal Coverage (OBS)

    - Start:    2025-03-28 00:00:00+00:00
    - End:      2025-03-28 23:59:30+00:00
    - Duration: 23:59:30
    - Sampling: 30.0s

    📡 Constellations Breakdown (OBS)

    Sys  | SVs  | Total Records   | Available Signals
    -------------------------------------------------
    C    | 39   | 561,006         | C1X, C2I, C5X, C6I, C7I, D2I, L1X, L2I, L5X, L6I, L7I, S1X, S2I, S5X, S6I, S7I
    E    | 27   | 352,425         | C1X, C5X, C6X, C7X, C8X, D1X, L1X, L5X, L6X, L7X, L8X, S1X, S5X, S6X, S7X, S8X
    G    | 31   | 220,278         | C1C, C2W, C5X, L1C, L2W, L5X, S1C, S2W, S5X
    R    | 23   | 153,268         | C1C, C1P, C2C, D1C, L1C, L2C, S1C, S2C

    📖 Ephemeris Coverage (NAV)

    - BEIDOU  :  54 satellites have at least an ephemeris
    - GALILEO :  31 satellites have at least an ephemeris
    - GLONASS :  25 satellites have at least an ephemeris
    - GPS     :  31 satellites have at least an ephemeris
    ```

### Download RINEX files

=== "INGV RING (OBS)"
    The `download_obs_ring` function targets the [INGV RING](https://webring.gm.ingv.it/) network. This network manages continuous GNSS stations across Italy and archives GNSS data from thousands of stations along the Eurasia-Africa plate boundary. The function handles station naming conventions: if a 4-character code like `GRO2` is provided, it is converted to the long-name format (e.g., `GRO200ITA`).
    
    ```python
    from pytecgg.utils import download_obs_ring

    download_obs_ring(
        station_code="GRO2",
        year=2025,
        doys=[1, 55, 252],
        output_path=Path("./data")
    )
    ```

    The function creates a subdirectory named after the `station_code` under the provided `output_path`. The above example will produce the following structure:

    ```bash
    data/
    └── GRO2/
        ├── GRO200ITA_R_20250010000_01D_30S_MO.crx.gz
        ├── GRO200ITA_R_20250550000_01D_30S_MO.crx.gz
        └── GRO200ITA_R_20252520000_01D_30S_MO.crx.gz
    ```
=== "EUREF EPN (OBS)"
    The `download_obs_euref` function targets the [EUREF Permanent GNSS Network](https://epncb.oma.be/). EPN stores daily observation data in RINEX 3 long-name format. If you provide a 4-character code like `BRUX`, the function scans the remote daily directory and resolves the matching long-name station file automatically.

    ```python
    from pytecgg.utils import download_obs_euref

    download_obs_euref(
        station_code="BRUX",
        year=2025,
        doys=[1, 55, 252],
        output_path=Path("./data")
    )
    ```

    The function creates a subdirectory named after the `station_code` under the provided `output_path`. The above example will produce a structure like:

    ```bash
    data/
    └── BRUX/
        ├── BRUX00BEL_R_20250010000_01D_30S_MO.crx.gz
        ├── BRUX00BEL_R_20250550000_01D_30S_MO.crx.gz
        └── BRUX00BEL_R_20252520000_01D_30S_MO.crx.gz
    ```
=== "BKG IGS (NAV)"
    Multi-constellation navigation messages, providing aggregated data from the global [IGS network](https://network.igs.org/), can be downloaded from the [BKG](https://igs.bkg.bund.de/) GNSS Data Center.
    
    ```python
    from pytecgg.utils import download_nav_bkg

    download_nav_bkg(
        year=2025,
        doys=[1, 55, 252],
        output_path=Path("./data")
    )
    ```

    The RINEX files are saved under the provided `output_path`.

--8<-- "includes/abbreviations.md"
