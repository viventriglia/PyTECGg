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
    plot_interactive,
    plot_static,
    print_logo,
    setup_logging,
    ARC_THRESHOLD_ABS,
    ARC_THRESHOLD_STD,
    ARC_THRESHOLD_JUMP,
    ARC_MIN_LENGTH,
    COLS_TO_KEEP,
    H_IPP,
    INPUT_DIR,
    MIN_ELEVATION,
    NAV_DIR,
    OUTPUT_DIR,
    PLOT_DPI,
    PROCESS_MODE,
    SAVE_CSV,
    SAVE_PARQUET,
    SAVE_INTERACTIVE_PLOTS,
    SAVE_STATIC_PLOTS,
)

print_logo()
logger = setup_logging()

# ==========================================
# CORE PROCESSING PIPELINE
# ==========================================


def get_nav_files(year: int, doy: int) -> list[Path]:
    yy = str(year)[-2:]
    files = list(NAV_DIR.glob(f"*BRDC*{year}*{doy:03d}*"))
    files.extend(NAV_DIR.glob(f"brdc{doy:03d}0.{yy}*"))
    return files


def merge_nav_dicts(nav_dicts: list[dict]) -> dict:
    master_nav = {}
    for day_nav in nav_dicts:
        for system, df_nav in day_nav.items():
            if system not in master_nav:
                master_nav[system] = df_nav
            else:
                master_nav[system] = pl.concat(
                    [master_nav[system], df_nav], how="diagonal"
                )
    return master_nav


def run_calibration_pipeline(
    df_obs: pl.DataFrame,
    nav_dict: dict,
    rec_pos: list,
    rec_name: str,
    rinex_version: float,
) -> pl.DataFrame:
    ctx = GNSSContext(
        receiver_pos=rec_pos,
        receiver_name=rec_name,
        rinex_version=rinex_version,
        h_ipp=H_IPP,
        systems=["G", "E", "R", "C"],
    )

    ephem_dict = prepare_ephemeris(nav_dict, ctx=ctx)
    df_lc = calculate_linear_combinations(df_obs, ctx=ctx)

    df_arcs = extract_arcs(
        df=df_lc,
        ctx=ctx,
        threshold_abs=ARC_THRESHOLD_ABS,
        threshold_std=ARC_THRESHOLD_STD,
        threshold_jump=ARC_THRESHOLD_JUMP,
        min_arc_length=ARC_MIN_LENGTH,
    )

    df_coords = satellite_coordinates(df_arcs["sv"], df_arcs["epoch"], ephem_dict)
    df_geom = df_arcs.join(df_coords, on=["sv", "epoch"], how="left")
    df_final = calculate_ipp(df_geom, ctx=ctx, min_elevation=MIN_ELEVATION)

    df_tec = calculate_tec(df_final, ctx=ctx)
    df_veq = calculate_vertical_equivalent(df_tec, ctx=ctx)

    df_spool = (
        df_veq.filter(pl.col("id_arc_valid").is_not_null())
        .select(COLS_TO_KEEP)
        .with_columns(pl.col(pl.Float32, pl.Float64).round(2))
        .sort(["epoch", "sv"])
    )
    return df_spool


def save_outputs(
    df_spool: pl.DataFrame, base_filename: Path, station_name: str, date_str: str
):
    if SAVE_PARQUET:
        df_spool.write_parquet(f"{base_filename}_calibrated.parquet")
        logger.info(f"💾 .parquet file saved")

    if SAVE_CSV:
        df_spool.write_csv(f"{base_filename}_calibrated.csv")
        logger.info(f"💾 .csv file saved")

    if SAVE_STATIC_PLOTS:
        plot_static(
            df_spool,
            f"{base_filename}_calibrated.png",
            PLOT_DPI,
            station_name,
            date_str,
        )
        logger.info(f"📈 .png static plot saved")

    if SAVE_INTERACTIVE_PLOTS:
        plot_interactive(
            df_spool, f"{base_filename}_calibrated.html", station_name, date_str
        )
        logger.info(f"📈 .html interactive plot saved")


# ==========================================
# RUN MODES
# ==========================================


def process_daily(obs_path: Path):
    logger.info(f">>> Working on DAILY: {obs_path.name}")
    try:
        df_obs, rec_pos, rinex_version = read_rinex_obs(str(obs_path))
        rec_name = obs_path.name[:4].lower()

        station_dir = OUTPUT_DIR / rec_name.upper()
        station_dir.mkdir(parents=True, exist_ok=True)

        first_epoch = df_obs["epoch"][0]
        year, doy = first_epoch.year, first_epoch.timetuple().tm_yday

        nav_files = get_nav_files(year, doy)
        if not nav_files:
            logger.info(f"📥 Missing NAV for {year}/DOY {doy}. Downloading...")
            download_nav_bkg(year=year, doys=[doy], output_path=NAV_DIR)
            nav_files = get_nav_files(year, doy)

        if not nav_files:
            logger.error(f"🚫 Unable to find or auto-download NAV for {obs_path.name}")
            logger.error(
                f"💡 ACTION REQUIRED: Please manually set a valid NAV file for {year} DOY {doy},"
            )
            logger.error(
                f"   under your local 'nav' directory, and run the container again."
            )
            logger.info("")
            return

        nav_dict = read_rinex_nav(str(nav_files[0]))

        df_spool = run_calibration_pipeline(
            df_obs, nav_dict, rec_pos, rec_name, rinex_version
        )

        base_filename = station_dir / obs_path.stem
        date_str = first_epoch.strftime("%d/%m/%Y")
        save_outputs(df_spool, base_filename, rec_name, date_str)
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Error during processing of {obs_path.name}: {str(e)}")
        logger.info("")


def process_station_batch(station_id: str, obs_files: list[Path]):
    logger.info(
        f">>> Working on BATCH: {station_id.upper()} ({len(obs_files)} files merged)"
    )

    try:
        all_obs = []
        all_doys = set()
        rec_pos, rinex_version = None, None

        for f in sorted(obs_files):
            df, pos, ver = read_rinex_obs(str(f))
            all_obs.append(df)
            if rec_pos is None:
                rec_pos, rinex_version = pos, ver
            first_epoch = df["epoch"][0]
            all_doys.add((first_epoch.year, first_epoch.timetuple().tm_yday))

        df_obs = pl.concat(all_obs).sort("epoch")

        nav_dicts = []
        for year, doy in sorted(all_doys):
            nav_files = get_nav_files(year, doy)
            if not nav_files:
                logger.info(f"📥 Missing NAV for {year}/DOY {doy}. Downloading...")
                download_nav_bkg(year=year, doys=[doy], output_path=NAV_DIR)
                nav_files = get_nav_files(year, doy)

            if nav_files:
                nav_dicts.append(read_rinex_nav(str(nav_files[0])))
            else:
                logger.warning(f"⚠️ Could not obtain NAV for {year} DOY {doy}")

        if not nav_dicts:
            logger.error(
                f"🚫 Skipping batch {station_id}: Unable to find or auto-download required NAV files."
            )
            logger.error(
                f"💡 ACTION REQUIRED: Please manually set the NAV files for the period"
            )
            logger.error(
                f"   under your local 'nav' directory, and run the container again."
            )
            logger.info("")
            return

        master_nav_dict = merge_nav_dicts(nav_dicts)

        df_spool = run_calibration_pipeline(
            df_obs, master_nav_dict, rec_pos, station_id, rinex_version
        )

        start_dt = df_obs["epoch"].min()
        end_dt = df_obs["epoch"].max()

        station_dir = OUTPUT_DIR / station_id.upper()
        station_dir.mkdir(parents=True, exist_ok=True)
        base_filename = (
            station_dir
            / f"{station_id.upper()}_{start_dt.strftime('%Y%j')}_{end_dt.strftime('%Y%j')}"
        )

        date_str = f"{start_dt.strftime('%d/%m/%Y')} - {end_dt.strftime('%d/%m/%Y')}"
        save_outputs(df_spool, base_filename, station_id, date_str)
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Error during batch processing of {station_id}: {str(e)}")
        logger.info("")


# ==========================================
# MAIN
# ==========================================


def main():
    extensions = [".rnx", ".crx", ".gz"]
    files = [
        f
        for f in INPUT_DIR.iterdir()
        if f.is_file()
        and (f.suffix.lower() in extensions or f.name.endswith(("o", "O")))
    ]

    if not files:
        logger.error(f"No files found in {INPUT_DIR}")
        return

    if PROCESS_MODE == "STATION_BATCH":
        groups = {}
        for f in files:
            sid = f.name[:4].lower()
            groups.setdefault(sid, []).append(f)

        for sid, f_list in sorted(groups.items()):
            process_station_batch(sid, f_list)
    elif PROCESS_MODE == "DAILY":
        for f in sorted(files):
            process_daily(f)


if __name__ == "__main__":
    main()
