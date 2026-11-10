# TEC calibration ⚖️

The calibration process is a fundamental step in GNSS-based ionospheric studies. Its goal is to estimate and remove the hardware-related inter-frequency biases (often referred to as Differential Code Biases, or DCBs) from both the transmitting satellites and the receiving hardware.

`PyTECGg` implements a modern version of the calibration algorithm described by [**Ciraolo et al. (2007)**](https://link.springer.com/article/10.1007/s00190-006-0093-1). This approach addresses the systematic errors introduced during the "levelling" process (where carrier-phase observations are aligned with code-delay data) and accounts for the short-term instability of receiver biases.

## How it works 💬

The calibration computes both slant TEC (**sTEC**) and vertical TEC (**vTEC**) by modeling the ionosphere as a thin shell at a fixed altitude. The spatial variation of the ionospheric delay is represented using a polynomial expansion of the shell up to a configurable degree (`max_degree`, default is 3).

To ensure temporal stability and account for intra-day variations of the hardware biases, the calibration is performed in time batches whose length, in minutes, is set by `batch_size_epochs`. This allows the model to remain responsive to ionospheric variations while mitigating the impact of code-delay multipath.

To calibrate your TEC data, you need the prepared `DataFrame` (containing arc-levelled combinations and IPP coordinates) and the `GNSSContext`:

```python
from pytecgg.tec_calibration import calculate_tec

df_calibrated = calculate_tec(
    df_final, 
    ctx=ctx,
    max_polynomial_degree=3,
    batch_size_epochs=30
)
```

Calibration Parameters:

- `max_polynomial_degree`: maximum degree of the polynomial expansion used to model the ionosphere; default is 3.
- `batch_size_epochs`: length of each batch **in minutes** (the name is historical; it does not depend on the sampling rate); smaller batches track bias variations more closely, while larger ones offer more stability; default is 30.

## Vertical Equivalent ↕️

While `vtec` provides a vertical estimation for each satellite's specific IPP, the Vertical Equivalent (**VEq**) represents the synthetic state of the ionosphere directly above the GNSS station (zenith).

`veq` is derived from the constant coefficient of the ionospheric model computed during the calibration process. Since the model is centered on the receiver's coordinates, evaluating the polynomial at the station's zenith isolates the local ionospheric contribution, effectively filtering out spatial gradients captured by different satellites and providing a unique, continuous value for the station.

```python
from pytecgg.tec_calibration import calculate_vertical_equivalent

df_veq = calculate_vertical_equivalent(
    df_calibrated, 
    ctx=ctx,
    max_polynomial_degree=3,
    batch_size_epochs=30
)
```

`veq` becomes valuable for station-level time-series analysis, as it mitigates individual satellite vulnerabilities.

## Wrapping up 🏁

These functions estimate instrumental offsets and append calibrated physical values to the `DataFrame`. You can verify the results by inspecting the estimated biases, absolute TEC values, and the vertical equivalent over the station.

=== "Code"
    ```python
    df_veq.filter(
        pl.col("id_arc_valid").is_not_null()
    ).select(
        [
            "epoch",
            "sv",
            "id_arc_valid",
            "gflc_levelled",
            "azi",
            "ele",
            "bias",
            "stec",
            "vtec",
            "veq",
        ]
    ).head()
    ```
=== "Example Output"

    | epoch | sv | id_arc_valid | gflc_levelled | azi | ele | bias | stec | vtec | veq |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | 2025-04-05 00:00:00 | G05 | "grot_g05_20250405_001" | -21.81 | 313.73 | 30.60 | -34.04 | 12.23 | 7.07 | 9.77 |
    | 2025-04-05 00:00:00 | G07 | "grot_g07_20250405_001" | -29.87 | 49.06 | 61.13 | -40.82 | 10.94 | 9.73 | 9.77 |
    | 2025-04-05 00:00:00 | G14 | "grot_g14_20250405_001" | -10.65 | 165.57 | 24.03 | -32.77 | 22.12 | 11.07 | 9.77 |
    | 2025-04-05 00:00:00 | G30 | "grot_g30_20250405_001" | 0.05 | 237.84 | 79.88 | -10.34 | 10.39 | 10.24 | 9.77 |
    | 2025-04-05 00:00:30 | G05 | "grot_g05_20250405_001" | -21.84 | 13.67 | 30.80 | -34.04 | 12.20 | 7.08 | 9.77 |

The new columns represent:

- `bias`: estimated instrumental bias (in TECu) for the specific arc and batch.
- `stec`: sTEC, corrected for hardware biases.
- `vtec`: final vTEC, obtained by applying the mapping function to the sTEC.
- `veq`: time series of the "equivalent" vertical TEC above the station (zenith).

As shown below, while the scattered points represent **vTEC** at each satellite's IPP, the **VEq** solid line represents an estimation of the ionosphere directly above the station zenith. A good alignment of the different satellite tracks, which would otherwise be scattered at different vertical offsets, indicates that instrumental biases have been successfully estimated and removed.

![TEC calibration](../images/tec_calibration.png)

!!! tip "Absolute TEC"
    Unlike the relative geometry-free combination, vTEC provides the absolute electron content, making it directly comparable across different stations and satellites.

!!! tip "vTEC *vs* VEq"
    Unlike `vtec`, which shows different values for each satellite due to their different geographic IPP positions, `veq` provides a unique value per epoch, making it the ideal product for local ionospheric trend analysis.

--8<-- "includes/abbreviations.md"