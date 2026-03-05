import os
import logging
import warnings
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import polars as pl

## Variables

VERBOSE = int(os.getenv("VERBOSE"))
SAVE_PARQUET = os.getenv("SAVE_PARQUET").lower() == "true"
SAVE_CSV = os.getenv("SAVE_CSV").lower() == "true"
SAVE_STATIC_PLOTS = os.getenv("SAVE_STATIC_PLOTS").lower() == "true"
SAVE_INTERACTIVE_PLOTS = os.getenv("SAVE_INTERACTIVE_PLOTS").lower() == "true"
PLOT_DPI = int(os.getenv("PLOT_DPI"))

INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
NAV_DIR = Path("/data/nav")
COLS_TO_KEEP = [
    "epoch",
    "sv",
    "id_arc",
    "lat_ipp",
    "lon_ipp",
    "azi",
    "ele",
    "bias",
    "stec",
    "vtec",
    "veq",
]

CONSTELLATION_COLORS = {
    "G": "#4682B4",  # GPS: Steel Blue
    "E": "#009E73",  # Galileo: Okabe-Ito Green
    "R": "#D55E00",  # GLONASS: Vermilion
    "C": "#CC79A7",  # BDS: Reddish Purple
}

CONSTELLATION_NAMES = {"G": "GPS", "E": "Galileo", "R": "GLONASS", "C": "BeiDou"}

## Helper functions


def setup_logging():
    if VERBOSE == 0:
        log_level = logging.ERROR
        warnings.filterwarnings("ignore")
    elif VERBOSE == 1:
        log_level = logging.INFO
        warnings.filterwarnings("ignore")
    else:
        log_level = logging.DEBUG
        logging.captureWarnings(True)

    _original_formatwarning = warnings.formatwarning

    def _clean_formatwarning(*args, **kwargs):
        return _original_formatwarning(*args, **kwargs).rstrip()

    warnings.formatwarning = _clean_formatwarning

    handlers = [logging.StreamHandler()]

    if VERBOSE == 3:
        log_file_path = OUTPUT_DIR / "pytecgg_batch.log"
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=log_level, format="%(message)s", handlers=handlers)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger = logging.getLogger("pytecgg-batch")

    if VERBOSE == 3:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n{'='*55}\n NEW BATCH RUN STARTED: {start_time}\n{'='*55}")

    return logger


def get_pytecgg_version():
    return version("pytecgg")


def print_logo():
    logo = rf"""
    ____        ______________________      
   / __ \__  _/_  __/ ____/ ____/ ____/____ 
  / /_/ / / / // / / __/ / /    / / __/ __ `/
 / ____/ /_/ // / / /___/ /___ / /_/ / /_/ / 
/_/    \__, //_/ /_____/\____/ \____/\__, /  
      /____/                        /____/   

    Batch Calibrator — Based on PyTECGg v{get_pytecgg_version()}
    ------------------------------------------
    """
    print(logo)


def plot_static(
    df_plt: pl.DataFrame, output_path: str, dpi: int, station_name: str, date_str: str
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "liberation sans"]

    bg_color = "#FFFFFF"
    veq_color = "#212121"
    grid_color = "#dee2e6"

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(grid_color)
    ax.spines["bottom"].set_color(grid_color)
    ax.yaxis.grid(True, linestyle="--", alpha=0.8, color=grid_color)
    ax.set_axisbelow(True)
    ax.tick_params(colors=veq_color, which="both", labelsize=11)

    ax.set_title(
        f"TEC over {station_name.upper()} station",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=25,
        color=veq_color,
    )
    ax.text(
        0,
        1.03,
        f"Calibrated sTEC and VEq – {date_str}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="normal",
        color=veq_color,
        alpha=0.8,
    )

    df_plt = df_plt.with_columns(pl.col("sv").str.slice(0, 1).alias("constellation"))
    for const in ["G", "E", "R", "C"]:
        df_const = df_plt.filter(pl.col("constellation") == const)
        if len(df_const) > 0:
            ax.scatter(
                df_const["epoch"],
                df_const["stec"],
                color=CONSTELLATION_COLORS.get(const),
                alpha=0.5,
                s=3.25,
                edgecolor="none",
                label=f"sTEC ({CONSTELLATION_NAMES.get(const, const)})",
            )

    df_veq_line = df_plt.select(["epoch", "veq"]).unique().sort("epoch")
    ax.plot(
        df_veq_line["epoch"],
        df_veq_line["veq"],
        color=veq_color,
        linewidth=3.5,
        zorder=10,
        label="VEq",
    )

    ax.set_ylabel("TECu", fontsize=13, color=veq_color)
    ax.set_xlabel("Epoch (UTC)", fontsize=13, color=veq_color, labelpad=10)
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=10,
        markerscale=2.5,
        facecolor="w",
        framealpha=1,
        edgecolor="none",
        borderpad=0.7,
    )

    ax.set_xlim(df_plt["epoch"].min(), df_plt["epoch"].max())
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_interactive(
    df_plt: pl.DataFrame, output_path: str, station_name: str, date_str: str
):
    bg_color = "#FFFFFF"
    veq_color = "#212121"
    grid_color = "#dee2e6"

    fig = go.Figure()

    df_plt = df_plt.with_columns(pl.col("sv").str.slice(0, 1).alias("constellation"))

    for const in ["G", "E", "R", "C"]:
        df_const = df_plt.filter(pl.col("constellation") == const)
        if len(df_const) > 0:
            fig.add_trace(
                go.Scatter(
                    x=df_const["epoch"],
                    y=df_const["stec"],
                    mode="markers",
                    marker=dict(
                        color=CONSTELLATION_COLORS.get(const, "#808080"),
                        size=4,
                        opacity=0.45,
                    ),
                    name=f"sTEC ({CONSTELLATION_NAMES.get(const, const)})",
                    text=df_const["sv"],
                    hovertemplate="<b>SV:</b> %{text}<br><b>Time:</b> %{x|%H:%M:%S}<br><b>sTEC:</b> %{y:.2f} TECu<extra></extra>",
                )
            )

    df_veq_line = df_plt.select(["epoch", "veq"]).unique().sort("epoch")
    fig.add_trace(
        go.Scatter(
            x=df_veq_line["epoch"],
            y=df_veq_line["veq"],
            mode="lines",
            line=dict(color=veq_color, width=3.5),
            name="VEq",
            hovertemplate="<b>Time:</b> %{x|%H:%M:%S}<br><b>VEq:</b> %{y:.2f} TECu<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>TEC over {station_name.upper()} station</b><br><sup>Calibrated sTEC and VEq – {date_str}</sup>",
            font=dict(size=20, color=veq_color, family="Arial, sans-serif"),
            x=0.02,
            y=0.95,
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Arial, sans-serif", color=veq_color),
        xaxis=dict(
            title="Epoch (UTC)",
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            griddash="dash",
            linecolor=grid_color,
            tickformat="%H:%M",
        ),
        yaxis=dict(
            title="TECu",
            showgrid=True,
            gridcolor=grid_color,
            gridwidth=1,
            griddash="dash",
            linecolor=grid_color,
        ),
        legend=dict(
            bgcolor="white",
            bordercolor="rgba(0,0,0,0)",
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
        ),
        margin=dict(l=60, r=30, t=100, b=60),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
