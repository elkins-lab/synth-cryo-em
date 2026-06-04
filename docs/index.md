# synth-cryo-em

A lightweight Pythonic utility to convert atomic models into synthetic 3D Cryo-EM maps.

## Overview
`synth-cryo-em` is designed to bridge the gap between atomic coordinates (PDB/CIF) and realistic density maps used in Cryo-Electron Microscopy. It simulates the physical processes of image formation, including resolution effects, Contrast Transfer Function (CTF), and noise.

## Why use synth-cryo-em?
- **ML Training:** Generate thousands of labeled synthetic maps for training denoisers, pickers, and segmentors.
- **Education:** Visualize how experimental parameters like resolution and defocus affect the resulting 3D density.
- **Validation:** Test the robustness of structural biology algorithms against varied noise and resolution levels.

## Scientific Basis
For details on the physical models and mathematical formulas used in this project, see our [Scientific Foundations](science.md) page.
