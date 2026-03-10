# PyTECGg Batch Calibrator 🐳

This tool provides a containerised, out-of-the-box solution to batch process and calibrate GNSS RINEX observation files using `PyTECGg`, without the need to manually configure a Python environment.

The Docker container automatically handles the entire pipeline:

- Reads RINEX observation files (`.rnx`, `.crx`, `.gz`).
- Downloads missing navigation messages (NAV) automatically, if not provided.
- Calibrates the TEC and extracts the VEq.
- Exports the calibrated data in `.parquet` or `.csv` files.
- Generates high-quality static (`.png`) and interactive (`.html` via Plotly) plots.

## ⚙️ Setup and Configuration

### Prerequisites
Before starting, ensure you have **Docker** installed and running on your system (*e.g.*, [Docker Desktop](https://www.docker.com/products/docker-desktop/)).

### Environment Configuration
The behavior of the calibrator and the data paths are fully customisable using environment variables. We provide an `.env.example` template to get you started.

You need to create a local `.env` file by copying the provided example. Open a terminal in this directory and run:

**Windows:**
> copy .env.example .env

**Linux / macOS:**
> cp .env.example .env

Open the newly created `.env` file, define your external paths and toggle the desired outputs.

### Configuration Reference

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `INPUT_DIR` | Path | - | Directory containing input RINEX observation files |
| `OUTPUT_DIR` | Path | - | Directory where results and logs will be saved |
| `NAV_DIR` | Path | - | Directory for navigation (BRDC) files |
| `SAVE_PARQUET` | Boolean | `True` | Exports calibrated data as a `.parquet` file |
| `SAVE_CSV` | Boolean | `False` | Exports calibrated data as a `.csv` file |
| `SAVE_STATIC_PLOTS` | Boolean | `True` | Generates a static `.png` plot |
| `SAVE_INTERACTIVE_PLOTS`| Boolean | `False` | Generates an interactive `.html` plot |
| `PLOT_DPI` | Integer | `300` | Resolution for the static `.png` plots |
| `VERBOSE` | Integer | `1` | Logging level. `0`: Silent, `1`: Standard, `2`: Verbose, `3`: Verbose + `.log` file |
| `PROCESS_MODE` | String | `DAILY` | Processing mode: `DAILY` (file-by-file) or `STATION_BATCH` (multi-day grouping) |
| `H_IPP` | Float | `350000` | Height of the Ionospheric Pierce Point (IPP) in meters |
| `MIN_ELEVATION` | Float | `20` | Minimum satellite elevation angle in degrees |
| `ARC_THRESHOLD_ABS` | Float | `10` | Absolute threshold for cycle slip/arc detection |
| `ARC_THRESHOLD_STD` | Float | `10` | Standard deviation threshold for cycle slip/arc detection |
| `ARC_THRESHOLD_JUMP` | Float | `5` | Jump threshold for continuous phase arc extraction |
| `ARC_MIN_LENGTH` | Integer | `120` | Minimum number of valid epochs required to keep an arc |


## 🚀 How to Run

Before starting the container, ensure your directories are set up correctly according to the paths defined in your `.env` file:

* **`INPUT_DIR`**: Place your GNSS RINEX OBS files here (`.rnx`, `.crx`, or `.gz`). The container will scan this folder and process all valid files.
* **`NAV_DIR`**: Place your broadcast navigation files (BRDC) here. **Note:** This is optional! If you don't provide the required NAV files, the container will automatically attempt to download the missing ones from the BKG servers based on the observation's date.
* **`OUTPUT_DIR`**: This is where your results will appear. The tool will automatically create subdirectories for each station and save the requested outputs.

Once your data is in place, open a terminal in this directory and simply run:

```sh
docker compose up
```