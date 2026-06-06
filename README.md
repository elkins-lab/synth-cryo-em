# synth-cryo-em

[![PyPI version](https://img.shields.io/pypi/v/synth-cryo-em.svg)](https://pypi.org/project/synth-cryo-em/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://elkins.github.io/synth-cryo-em/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![codecov](https://codecov.io/gh/elkins/synth-cryo-em/branch/main/graph/badge.svg)](https://codecov.io/gh/elkins/synth-cryo-em)

A lightweight Pythonic utility to convert atomic models (PDB/CIF) into synthetic 3D Cryo-EM maps with realistic noise, CTF effects and varying resolutions.

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

- **Interactive Physics & Visualization**: Explore how resolution, noise (SNR), and CTF effects change the visual features of a map in 3D.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-cryo-em/blob/main/notebooks/visual_tutorial.ipynb)
- **Core API Walkthrough**: A step-by-step guide to using the Python API for custom workflows.
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-cryo-em/blob/main/notebooks/interactive_tutorial.ipynb)

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

## Related Projects

This library is part of the **synth-pdb ecosystem**:

- [synth-pdb](https://github.com/elkins/synth-pdb) — Core protein structure generator
- [synth-nmr](https://github.com/elkins/synth-nmr) — NMR observables simulator
- [synth-saxs](https://github.com/elkins/synth-saxs) — SAXS profile simulator
- [synth-dynamics](https://github.com/georgeelkins/synth-dynamics) — ANM/Langevin dynamics engine
- [diff-biophys](https://github.com/elkins/diff-biophys) — Differentiable JAX biophysics kernels

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/elkins/synth-cryo-em). Run `pre-commit run --all-files` before submitting.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{synth_cryo_em,
  author  = {Elkins, George},
  title   = {synth-cryo-em: Synthetic cryo-EM map generation from atomic models},
  year    = {2026},
  url     = {https://github.com/elkins/synth-cryo-em},
  version = {0.1.0}
}
```
