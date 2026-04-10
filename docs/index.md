![logo](images/pytecgg_logo.svg)

<p align="center">
  <a href="https://pypi.org/project/pytecgg/">
    <img src="https://img.shields.io/pypi/v/pytecgg.svg" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/badge/python-3.11--3.13-blue.svg" alt="Python version">
  <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="License">
  <img src="https://img.shields.io/pypi/dm/PyTECGg" alt="Downloads">
</p>

<p align="center">
  <img src="https://github.com/viventriglia/PyTECGg/actions/workflows/pytest.yml/badge.svg" alt="Tests">
  <img src="https://github.com/viventriglia/pytecgg/actions/workflows/build_publish.yml/badge.svg" alt="CI">
  <a href="https://gemini.google.com/gem/1qc1bu6XL6UVUtrd1MLuCQVrnLI8_nU9_?usp=sharing">
    <img src="https://img.shields.io/badge/Gemini-PyTECGg-1a73e8?logo=googlegemini&logoColor=white" alt="Gemini Gem">
  </a>
</p>

---

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core.

## What is PyTECGg?

`PyTECGg` is a fast, lightweight Python package that helps **reconstruct and calibrate** the **[Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content)** (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

## Main features

This package is designed for researchers and engineers who need speed and reliability.

- **Blazing fast**: powered by a [Rust](https://rust-lang.org/) core and [Polars](https://pola.rs/), it is much faster than existing pure-Python implementations when parsing large datasets.
- **Open source**: read, access, and contribute to [all the code](https://github.com/viventriglia/PyTECGg).
- **Modern GNSS support**: compatible with GPS, Galileo, BeiDou, GLONASS.
- **Adaptive signal selection**: automatic dual-frequency ranking now supports per-satellite BeiDou selection and RINEX-aware BeiDou band/frequency mapping.
- **RINEX ready**: supports versions 2, 3, and 4.
- **Seamless decompression**: built-in support for Hatanaka-compressed files — no need to decompress your data manually.

!!! info "Citing `PyTECGg`"
    If you use `PyTECGg` for your research, please cite
    [*PyTECGg: Total Electron Content reconstruction with GNSS data*](https://dx.doi.org/10.2139/ssrn.6549526) (preprint) by V. Ventriglia, M. Guerra, D. Okoh, P. Vermicelli, L. Ciraolo, and C. Cesaroni.

## Get started

📥 [**Install** `PyTECGg`](installation.md)

🚀 [**Quickstart** with an hands-on tutorial](quickstart/0-data-ingestion-setup.md)

🤖 <a href="https://gemini.google.com/gem/1qc1bu6XL6UVUtrd1MLuCQVrnLI8_nU9_?usp=sharing" target="_blank">**Ask** the official <code>PyTECGg</code> AI assistant for help</a>

🌱 [**Contribute** to the project](contributing.md)
