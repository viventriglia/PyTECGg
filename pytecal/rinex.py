from pytecal import read_rinex_obs, read_rinex_nav
import time


def test_obs():
    files = [
        "./rinex/v2/obs/cgtc0920.14o",
        "./rinex/v3/obs/ASIR00ITA_R_20242810000_01D_30S_MO.crx",
    ]

    for file_path in files:
        print(f"\nProcessing file: {file_path}")

        start_time = time.time()
        df, (x, y, z) = read_rinex_obs(file_path)
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


def test_nav():
    files = [
        "./rinex/v3/nav/BRDC00GOP_R_20140920000_01D_MN.rnx",
        "./rinex/v3/nav/BRDC00WRD_R_20250870000_01D_MN.rnx",
    ]

    for file_path in files:
        print(f"\nProcessing file: {file_path}")

        start_time = time.time()
        nav_by_constellation_dict = read_rinex_nav(file_path)
        load_time = time.time() - start_time

        print(f"Load time: {load_time:.2f} s")
        print(nav_by_constellation_dict)


def main():
    # test_obs()
    test_nav()


if __name__ == "__main__":
    main()
