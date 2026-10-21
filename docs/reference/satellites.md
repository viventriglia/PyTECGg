# Satellites & Geometry

The `satellites` module provides essential functions for GNSS orbit propagation and observation geometry. It handles the transition from raw navigation messages to precise satellite positions and Ionospheric Pierce Points (IPP).

## Orbital Models

`PyTECGg` supports different orbital propagation models depending on the GNSS constellation:

1.  **Keplerian** model: used for GPS, Galileo, and BeiDou; it computes positions based on orbital elements valid for a few hours.
2.  **State-Vector** model: used for GLONASS; it performs numerical integration (via a [Numba](https://numba.pydata.org/)-accelerated ODE solver) of instantaneous position, velocity, and acceleration vectors.

## Ephemeris Preparation Notes

`prepare_ephemeris` now validates navigation records according to the orbit model actually used downstream:

- **GPS, Galileo, BeiDou**: the function keeps every valid broadcast record per satellite, checking only the orbital fields strictly required for Keplerian position propagation. Position computation later selects the record whose `toe` is nearest to the observation epoch, since broadcast Keplerian parameters are only valid for ~±2 h around `toe` — reusing a single record across a multi-hour file would otherwise introduce 50–200 m biases. Records more than 24 h from the observation epoch are not used. Galileo nav files repeat each record once per data source (F/NAV and I/NAV) with the same orbit; one record per epoch is kept, preferring I/NAV E1+E5b (`source` 517).
- **GLONASS**: the full message history is retained because state-vector propagation needs the closest valid epoch and the FDMA channel.

This means a navigation message is not discarded, as long as the orbital solution is still sufficient to compute satellite geometry.

### Time Systems and Leap Seconds

Each broadcast record carries a `leap_seconds` field (UTC → GPST offset), chosen in this order:

1. `GNSSContext.leap_seconds`, if set (an explicit override).
2. The `LEAP SECONDS` field of the nav file header, which `read_rinex_nav` adds as a `leap_seconds` column. Nav files written after a new leap second carry the new value, so this follows future changes without a `PyTECGg` update. Files merged from several days keep each file's value.
3. The leap-second table `LEAP_SECONDS_UTC_GPST` in `pytecgg.satellites.constants`, for the record date, when the header has no `LEAP SECONDS` line (the field is optional). The table is checked against IERS Bulletin C up to `LEAP_SECONDS_VALID_UNTIL`; for later records a warning says the value may be stale.

Observation epochs from `read_rinex_obs` are on the GPST grid. During position computation they are shifted into the constellation's own time system before the elapsed time from the ephemeris epoch is computed: no shift for GPS / Galileo / QZSS, BDT for BeiDou (BDT = GPST − 14 s), and UTC for GLONASS (UTC = GPST − `leap_seconds`).

---

## API Reference

::: pytecgg.satellites
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
      docstring_section_style: table
      group_by_category: false
      members:
        - prepare_ephemeris
        - satellite_coordinates
        - calculate_ipp
        - Ephem
