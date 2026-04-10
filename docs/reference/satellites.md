# Satellites & Geometry

The `satellites` module provides essential functions for GNSS orbit propagation and observation geometry. It handles the transition from raw navigation messages to precise satellite positions and Ionospheric Pierce Points (IPP).

## Orbital Models

`PyTECGg` supports different orbital propagation models depending on the GNSS constellation:

1.  **Keplerian** model: used for GPS, Galileo, and BeiDou; it computes positions based on orbital elements valid for a few hours.
2.  **State-Vector** model: used for GLONASS; it performs numerical integration (via a [Numba](https://numba.pydata.org/)-accelerated ODE solver) of instantaneous position, velocity, and acceleration vectors.

## Ephemeris Preparation Notes

`prepare_ephemeris` now validates navigation records according to the orbit model actually used downstream:

- **GPS, Galileo, BeiDou**: the function keeps one representative broadcast message per satellite and checks only the orbital fields strictly required for Keplerian position propagation.
- **GLONASS**: the full message history is retained because state-vector propagation needs the closest valid epoch and the FDMA channel.

This means a navigation message is not discarded, as long as the orbital solution is still sufficient to compute satellite geometry.

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
