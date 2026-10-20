# Parsing

The `parsing` module is the entry point for data ingestion in `PyTECGg`. It leverages a high-performance **Rust backend** to handle the heavy lifting of reading RINEX files, ensuring that even large multi-constellation observation files are processed with minimal latency. By delegating the parsing logic to Rust and returning native [Polars](https://pola.rs/) `DataFrame`s, `PyTECGg` avoids the common bottlenecks of Python-based RINEX readers. The module automatically handles:

* Format detection: support for standard `.rnx`, Hatanaka-compressed `.crx`, and gz-ipped `.gz` files.
* Metadata extraction: retrieval of the receiver ECEF position and RINEX version.
* Epochs: returned as written in the file, with the UTC time zone attached but no time-scale shift. Observation epochs stay on the GPST grid; navigation epochs stay in each constellation's own time (e.g. UTC for GLONASS). Satellite position computation handles the conversion (see [Satellites & Geometry](satellites.md#time-systems-and-leap-seconds)).

---

## API Reference

::: pytecgg.parsing
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
      docstring_section_style: table
      members:
        - read_rinex_obs
        - read_rinex_nav
        - read_rinex_nav_header