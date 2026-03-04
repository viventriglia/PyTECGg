# PyTECGg Batch Calibrator 🐳

This tool provides a containerised, out-of-the-box solution to batch process and calibrate GNSS RINEX observation files using `PyTECGg`, without the need to manually configure a Python environment.

The Docker container automatically handles the entire pipeline:

- Reads RINEX observation files (`.rnx`, `.crx`, `.gz`).

- Downloads missing navigation messages (NAV) automatically, if not provided.

- Calibrates the TEC and extracts the VEq.

- Exports the calibrated data in `.parquet` or `.csv` files, and optionally generates static (`.png`) or interactive (`.html`) plots.

## ⚙️ Setup and Configuration

The behavior of the calibrator and the data paths are fully customisable using environment variables. We provide an `.env.example` template to get you started.

You need to create a local `.env` file by copying the provided example. Open a terminal in this directory and run:

```sh
copy .env.example .env
```

or on Linux / macOS:

```sh
cp .env.example .env
```

Open the newly created `.env` file, define your external paths and toggle the desired outputs.

## 🚀 How to Run

Once you set `INPUT_DIR`, `OUTPUT_DIR`, and `NAV_DIR` to the actual absolute paths on your system, and once you dropped your RINEX files under the local `INPUT_DIR` (optionally, under the local `NAV_DIR` as well), you can run the container:

```sh
docker compose up
```

The container will automatically read and process the files, and save the requested results in your specified output directory.