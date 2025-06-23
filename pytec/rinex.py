from libtecrs import read_rinex_obs_to_polars
import polars as pl
import time


def main():
    files = [
        "./rinex/v2/obs/cgtc0920.14o",
        "./rinex/v3/obs/ASIR00ITA_R_20242810000_01D_30S_MO.crx",
    ]

    for file_path in files:
        print(f"\nProcessing file: {file_path}")

        start_time = time.time()
        df, (x, y, z) = read_rinex_obs_to_polars(file_path)
        load_time = time.time() - start_time

        print(f"Load time: {load_time:.2f} s")
        print(f"Number of rows: {len(df):,}")
        print(f"ECEF coordinates: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
        print(df.head(5))

        df_pivot = df.pivot(
            values="value",
            index=["epoch", "sv"],
            on="observable",
            aggregate_function="first",
        )

        print(df_pivot.head(5))


if __name__ == "__main__":
    main()
