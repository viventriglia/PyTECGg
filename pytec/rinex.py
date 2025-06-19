import libtecrs


def main():
    input_path = "./rinex/v2/obs/cgtc0920.14o"
    output_path = "./output/cgtc0920.parquet"
    # libtecrs.convert_rinex_to_parquet(input_path, output_path)


if __name__ == "__main__":
    main()
