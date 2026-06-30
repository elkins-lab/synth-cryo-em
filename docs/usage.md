# Usage Guide

## Installation

```bash
pip install .
```

## Basic Usage

To generate a simple density map from a PDB file:

```bash
synth-cryo-em structure.pdb output.mrc --resolution 4.0
```

## Simulating Realistic Physics

To add CTF effects and noise:

```bash
synth-cryo-em structure.pdb output.mrc \
  --resolution 3.5 \
  --apply-physics \
  --defocus 2.0 \
  --bfactor 150 \
  --snr 5
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--resolution` / `-r` | Target resolution in Angstroms | 4.0 |
| `--spacing` / `-s` | Grid spacing (pixel size) in Angstroms | resolution / 3 |
| `--snr` | Signal-to-Noise Ratio for Gaussian noise | None |
| `--defocus` | Defocus in micrometers | 2.0 |
| `--voltage` | Acceleration voltage in kV | 300.0 |
| `--cs` | Spherical aberration in mm | 2.7 |
| `--bfactor` | Envelope function B-factor | 0.0 |
| `--apply-physics` | Enable CTF effects | False |

## Map Validation

The package also includes `synth-cryo-em-validate`, a tool to compare two Cryo-EM maps and compute their Cross-Correlation Coefficient (CCC) and Fourier Shell Correlation (FSC) curves.

```bash
synth-cryo-em-validate map1.mrc map2.mrc --output fsc_results.csv
```

### Validation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output` / `-o` | Path to save FSC data (CSV) | None |
