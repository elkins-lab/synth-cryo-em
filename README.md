# 🧬 synth-cryo-em: Synthetic Cryo-EM Map Generation

[![PyPI version](https://img.shields.io/pypi/v/synth-cryo-em.svg)](https://pypi.org/project/synth-cryo-em/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/synth-cryo-em.svg)](https://pypi.org/project/synth-cryo-em/)
[![Tests](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-cryo-em/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`synth-cryo-em` generates synthetic density maps from PDB structures, simulating the image-formation process of electron cryo-microscopy.

---

### 🧪 For Structural Biologists
*   **Map Simulation:** Create density maps at various resolutions (e.g., 2Å to 20Å) to assist in model building and validation.
*   **MRC Export:** Generates standard `.mrc` files compatible with ChimeraX and PyMOL.

### 🤖 For Machine Learning Geeks
*   **ML Training Data:** Generate large-scale voxel datasets for training 3D CNNs or diffusion models for map denoising and deconvolution.
*   **Physics-Grounded:** Models the atomic scattering factors for electrons to ensure physically accurate maps.

---

## 📦 Installation

```bash
pip install synth-cryo-em
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
