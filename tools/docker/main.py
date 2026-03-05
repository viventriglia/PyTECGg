from pathlib import Path

import polars as pl

from pytecgg import GNSSContext
from pytecgg.parsing import read_rinex_obs, read_rinex_nav
from pytecgg.satellites import prepare_ephemeris, satellite_coordinates, calculate_ipp
from pytecgg.linear_combinations import calculate_linear_combinations
from pytecgg.tec_calibration import (
    extract_arcs,
    calculate_tec,
    calculate_vertical_equivalent,
)
from pytecgg.utils import download_nav_bkg
from utils import (
    print_logo,
    plot_static,
    plot_interactive,
    setup_logging,
    INPUT_DIR,
    OUTPUT_DIR,
    NAV_DIR,
    SAVE_CSV,
    SAVE_PARQUET,
    SAVE_INTERACTIVE_PLOTS,
    SAVE_STATIC_PLOTS,
    COLS_TO_KEEP,
    PLOT_DPI,
)

print_logo()
logger = setup_logging()


def process_file(obs_path: Path):
    logger.info("")
    logger.info(f">>> Working on: {obs_path.name}")

    try:
        df_obs, rec_pos, rinex_version = read_rinex_obs(str(obs_path))
        rec_name = obs_path.name[:4].lower()

        # Create output directory for the station
        station_dir = OUTPUT_DIR / rec_name.upper()
        station_dir.mkdir(parents=True, exist_ok=True)

        # Find Year e DOY to download NAV if missing
        first_epoch = df_obs["epoch"][0]
        year = first_epoch.year
        doy = first_epoch.timetuple().tm_yday

        # Look for local NAV files containing "BRDC", year, and DOY
        nav_files = list(NAV_DIR.glob(f"*BRDC*{year}*{doy:03d}*"))
        if not nav_files:
            logger.info(
                f"📥 Missing NAV for {year}/DOY {doy}. Downloading from BKG servers..."
            )
            download_nav_bkg(year=year, doys=[doy], output_path=NAV_DIR)
            nav_files = list(NAV_DIR.glob(f"*BRDC*{year}*{doy:03d}*"))

        if not nav_files:
            logger.error(
                f"🚫 Skipping: Unable to find or download NAV for {obs_path.name}"
            )
            return

        nav_dict = read_rinex_nav(str(nav_files[0]))

        ctx = GNSSContext(
            receiver_pos=rec_pos,
            receiver_name=rec_name,
            rinex_version=rinex_version,
            systems=["G", "E", "R", "C"],  # FIXME possiamo anche filtrare
        )

        ephem_dict = prepare_ephemeris(nav_dict, ctx=ctx)
        df_lc = calculate_linear_combinations(df_obs, ctx=ctx)
        df_arcs = extract_arcs(df=df_lc, ctx=ctx)

        df_coords = satellite_coordinates(df_arcs["sv"], df_arcs["epoch"], ephem_dict)
        df_geom = df_arcs.join(df_coords, on=["sv", "epoch"], how="left")
        df_final = calculate_ipp(df_geom, ctx=ctx, min_elevation=20)

        df_tec = calculate_tec(df_final, ctx=ctx)
        df_veq = calculate_vertical_equivalent(df_tec, ctx=ctx)

        df_spool = (
            df_veq.filter(pl.col("id_arc_valid").is_not_null())
            .select(COLS_TO_KEEP)
            .with_columns(pl.col(pl.Float32, pl.Float64).round(2))
            .sort(["epoch", "sv"])
        )

        base_filename = station_dir / obs_path.stem

        if SAVE_PARQUET:
            df_spool.write_parquet(f"{base_filename}_calibrated.parquet")
            logger.info(f"💾 .parquet file saved")

        if SAVE_CSV:
            df_spool.write_csv(f"{base_filename}_calibrated.csv")
            logger.info(f"💾 .csv file saved")

        if SAVE_STATIC_PLOTS:
            plot_path = f"{base_filename}_calibrated.png"
            date_str = first_epoch.strftime("%d/%m/%Y")
            plot_static(df_spool, plot_path, PLOT_DPI, rec_name, date_str)
            logger.info(f"📈 .png static plot saved")

        if SAVE_INTERACTIVE_PLOTS:
            plot_path = f"{base_filename}_calibrated.html"
            date_str = first_epoch.strftime("%d/%m/%Y")
            plot_interactive(df_spool, plot_path, rec_name, date_str)
            logger.info(f"📈 .html interactive plot saved")

    except Exception as e:
        logger.error(f"❌ Error during processing of {obs_path.name}: {str(e)}")


def main():
    extensions = [".rnx", ".crx", ".gz"]
    files = [
        f
        for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in extensions or f.name.endswith(("o", "O"))
    ]

    if not files:
        print(f"No files found in {INPUT_DIR}")
        return

    for f in sorted(files):
        process_file(f)


if __name__ == "__main__":
    main()
