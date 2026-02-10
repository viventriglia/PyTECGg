# PyTECGg

[![PyPI version](https://img.shields.io/pypi/v/pytecgg?color=3776ab&logo=pypi&logoColor=white)](https://pypi.org/project/pytecgg/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pytecgg?color=3776ab&logo=pypi&logoColor=white)](https://img.shields.io/pypi/dm/PyTECGg)
[![Python version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776ab?logo=python&logoColor=white)](https://pypi.org/project/pytecgg/)
[![License](https://img.shields.io/badge/license-GPLv3-3776ab)](https://github.com/viventriglia/PyTECGg/blob/main/LICENSE)

[![Docs](https://img.shields.io/badge/docs-reference-blue?color=3776ab&logo=materialformkdocs&logoColor=white)](https://viventriglia.github.io/PyTECGg/)
[![Tests](https://github.com/viventriglia/PyTECGg/actions/workflows/pytest.yml/badge.svg)](https://github.com/viventriglia/PyTECGg/actions/workflows/pytest.yml)
[![Build & Publish](https://github.com/viventriglia/PyTECGg/actions/workflows/build_publish.yml/badge.svg)](https://github.com/viventriglia/PyTECGg/actions/workflows/build_publish.yml)
[![AI Assistant](https://img.shields.io/badge/AI_Assistant-Gemini-1a73e8?logo=googlegemini&logoColor=white)](https://gemini.google.com/gem/1qc1bu6XL6UVUtrd1MLuCQVrnLI8_nU9_?usp=sharing)


<p align="left">
  <img src="docs/images/pytecgg_logo.svg" width="400" title="Logo PyTECGg">
</p>

Total Electron Content (**TEC**) reconstruction with **GNSS** data – a Python 🐍 package with a Rust 🦀 core

## What is it?

PyTECGg is a fast, lightweight Python package that helps **reconstruct and calibrate** the [Total Electron Content](https://en.wikipedia.org/wiki/Total_electron_content) (TEC) from **GNSS data**.

Why calibration matters? Because without it, you don’t actually know the true value of TEC — only how it changes. Uncalibrated TEC is affected by unknown biases from satellites and receivers, as well as other sources of error.

This package:
- is open source: read and access all the code!
- supports all modern GNSS constellations, codes and signals:
    - GPS, Galileo, BeiDou, GLONASS
- supports RINEX V2-3-4
- provides seamless decompression for RINEX files


## Installation

You can install the package directly from [PyPI](https://pypi.org/project/pytecgg/):

```shell
pip install pytecgg
```

This will also install all required Python dependencies automatically.


## Documentation

Read the **documentation** [**here**](https://viventriglia.github.io/PyTECGg/).

> [!TIP]
> **In a hurry?** You can [**ask the PyTECGg AI assistant**](https://gemini.google.com/gem/1qc1bu6XL6UVUtrd1MLuCQVrnLI8_nU9_?usp=sharing) for instant help!

## Contributing

We welcome contributions from everyone!

👉 [**Contributing to PyTECGg**](./CONTRIBUTING.md)

Please read the contributing guide before submitting issues or pull requests.

## License

This project is released under the **GPLv3 License**.