# Utils 🛠️

The `utils` module provides helper functions to inspect the parsed datasets, and to download RINEX files. About the latter, to perform TEC analysis you need both Observation (from the station) and Navigation (global ephemerides) files: `PyTECGg` simplifies this with dedicated downloaders:

* [**INGV RING Network**](https://webring.gm.ingv.it/): targeted at the Italian GNSS network, downloading high-quality 30s observation files.
* [**EUREF EPN**](https://epncb.oma.be/): targeted at the European Permanent Network, downloading daily observation files from the EPN archive.
* [**BKG IGS Global**](https://igs.bkg.bund.de/): Downloads aggregated multi-constellation navigation files (BRDC), essential for orbit propagation.

---

## API Reference

::: pytecgg.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
      docstring_section_style: table
      members:
        - summarise_rinex_data
        - download_obs_euref
        - download_obs_ring
        - download_nav_bkg
