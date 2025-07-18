# PyTECGg

<!-- Add PyPi version when published -->
![Python version](https://img.shields.io/badge/python-3.11--3.13-blue.svg)
![License](https://img.shields.io/github/license/viventriglia/pyTEC)
![Tests](https://github.com/viventriglia/pyTEC/actions/workflows/pytest.yml/badge.svg)

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core

## Table of Contents

- [What is it?](#what-is-it)

- [Example usage](#example-usage)

- [How can I help?](#how-can-i-help)

## What is it?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the [Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content) (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

This package:
- is open source: read and access all the code!
- supports all modern GNSS constellations, codes and signals:
    - GPS, Galileo, BeiDou, GLONASS and QZSS
- supports RINEX V2-3-4
- provides seamless de-compression for RINEX files

| ![Earth's ionosphere and GNSS satellites](images/project_cover.png) | 
|:--:| 
| *Generated image of Earth's ionosphere with GNSS satellites studying TEC* |


## Example usage

### Read RINEX files — fast ⚡

```python
from pytecggrs import read_rinex_nav, read_rinex_obs

# Read a RINEX navigation file
nav_dict = read_rinex_nav("./path/to/your/file.rnx")

# Read a RINEX observation file, and extract receiver position and RINEX version
obs_data, rec_pos, rinex_version = read_rinex_obs("./path/to/your/file.rnx")
```

### Prepare Satellite Ephemerides 🛰️

```python
from pytecgg.satellites.ephemeris import prepare_ephemeris

ephem_dict = prepare_ephemeris(nav_dict, constellation='Galileo')
```

Supported constellations are: ```'Galileo', 'GPS', 'GLONASS', 'BeiDou'```

### Compute Satellite Coordinates 🧭

```python
from pytecgg.satellites.positions import satellite_coordinates

# Compute the position of a Galileo satellite (space vehicle #25)
satellite_coordinates(ephem_dict=ephem_dict, sv_id='E25', gnss_system='Galileo')
```

## How can I help?

Contributions are what make the open source community an amazing place to learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature_amazing_feature`)
3. Commit your Changes (`git commit -m 'Add some amazing stuff'`)
4. Push to the Branch (`git push origin feature_amazing_feature`)
5. Open a Pull Request

