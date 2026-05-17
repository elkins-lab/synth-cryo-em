# synth-cryo-em

[![Tests](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://elkins.github.io/synth-cryo-em/)

A lightweight Pythonic utility to convert atomic models (PDB/CIF) into synthetic 3D Cryo-EM maps with realistic noise, CTF effects, and varying resolutions.

## 🌟 Features
- **Voxelize** atomic models with accurate resolution simulation.
- **Simulate Physics:** Apply Contrast Transfer Functions (CTF) and envelope functions.
- **Noise Modeling:** Add adjustable Gaussian noise to simulate low-SNR experimental data.
- **Standard Format:** Export results to MRC files compatible with RELION, ChimeraX, and other tools.

## 🚀 Quick Start

### Installation
```bash
pip install synth-cryo-em
```

### Basic Generation
```bash
synth-cryo-em structure.pdb output.mrc --resolution 4.0
```

### Realistic Simulation
```bash
synth-cryo-em structure.pdb output.mrc --resolution 3.5 --apply-physics --snr 5
```

## 📚 Tutorials
Explore the project's functionality interactively via our Jupyter notebooks:

- **[Interactive Physics & Visualization](notebooks/visual_tutorial.ipynb)**: Explore how resolution, noise (SNR), and CTF effects change the visual features of a map in 3D.
- **[Core API Walkthrough](notebooks/interactive_tutorial.ipynb)**: A step-by-step guide to using the Python API for custom workflows.

## 📖 Documentation
For detailed guides and API reference, visit the [Documentation Site](https://elkins.github.io/synth-cryo-em/).

## 🛠️ Development
To install for development and documentation:
```bash
pip install -e ".[test,docs]"
```

Run tests:
```bash
pytest tests/
```

Build docs locally:
```bash
mkdocs serve
```
