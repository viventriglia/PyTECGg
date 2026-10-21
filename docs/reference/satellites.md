# Satellites & Geometry

The `satellites` module provides essential functions for GNSS orbit propagation and observation geometry. It handles the transition from raw navigation messages to precise satellite positions and Ionospheric Pierce Points (IPP).

## Orbital Models

`PyTECGg` supports different orbital propagation models depending on the GNSS constellation:

1.  **Keplerian** model: used for GPS, Galileo, and BeiDou; it computes positions based on orbital elements valid for a few hours.
2.  **State-Vector** model: used for GLONASS; it performs numerical integration (via a [Numba](https://numba.pydata.org/)-accelerated ODE solver) of instantaneous position, velocity, and acceleration vectors.

## Ephemeris Preparation Notes

`prepare_ephemeris` now validates navigation records according to the orbit model actually used downstream:

- **GPS, Galileo, BeiDou**: the function keeps every valid broadcast record per satellite, checking only the orbital fields strictly required for Keplerian position propagation. Position computation later selects the record whose `toe` is nearest to the observation epoch, since broadcast Keplerian parameters are only valid for ~±2 h around `toe` — reusing a single record across a multi-hour file would otherwise introduce 50–200 m biases. Records corrupted by the rinex-crate week-rollover `toe` parsing bug are filtered out and a warning is emitted.
- **GLONASS**: the full message history is retained because state-vector propagation needs the closest valid epoch and the FDMA channel.

This means a navigation message is not discarded, as long as the orbital solution is still sufficient to compute satellite geometry.

### Time Systems and Leap Seconds

Each broadcast record carries a `leap_seconds` field (UTC → GPST offset). The value is taken from the RINEX nav header via [`read_rinex_nav_header`][pytecgg.parsing.read_rinex_nav_header] (assigned to `GNSSContext.leap_seconds`) or falls back to `DEFAULT_LEAP_SECONDS_UTC_GPST` from `pytecgg.satellites.constants`. During position computation, observation epochs are shifted into the constellation's own time system before the elapsed-time-from-`toe` is computed: GPST for GPS / Galileo / QZSS, BDT for BeiDou (BDT = GPST − 14 s), and UTC for GLONASS.

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
