"""
synth-cryo-em: Synthetic Cryo-EM map generator from atomic models.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("synth-cryo-em")
except PackageNotFoundError:
    __version__ = "unknown"

from .core import apply_ctf, compute_ccc, generate_density_map, save_mrc
