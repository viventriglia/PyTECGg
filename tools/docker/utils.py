from importlib.metadata import version

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import polars as pl

DOCKER_CALIBRATOR_VERSION = "0.1.0"


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

    Batch Calibrator v{DOCKER_CALIBRATOR_VERSION}
    Based on PyTECGg v{get_pytecgg_version()}
    --------------------------------------
    """
    print(logo)


def plot_tec(
    df_plt: pl.DataFrame, output_path: str, dpi: int, station_name: str, date_str: str
) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    bg_color = "#FFFFFF"
    veq_color = "#212121"
    grid_color = "#dee2e6"
    vtec_color = "#4682B4"

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
        f"Calibrated vTEC and VEq – {date_str}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="normal",
        color=veq_color,
        alpha=0.8,
    )

    ax.scatter(
        df_plt["epoch"],
        df_plt["vtec"],
        color=vtec_color,
        alpha=0.35,
        s=3.25,
        edgecolor="none",
        label="vTEC",
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
