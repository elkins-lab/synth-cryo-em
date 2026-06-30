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
    apply_ctf,
    compute_ccc,
    save_mrc
)
