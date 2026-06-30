"""
synth-cryo-em: Synthetic Cryo-EM map generator from atomic models.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("synth-cryo-em")
except PackageNotFoundError:
    __version__ = "unknown"

from .core import (
    generate_density_map,
    add_gaussian_noise,
    apply_ctf,
    compute_fsc,
    compute_ccc,
    save_mrc
)
