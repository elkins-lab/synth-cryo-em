import click
import numpy as np
from synth_core import add_gaussian_noise

from .core import apply_ctf, generate_density_map, save_mrc


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--resolution", "-r", default=4.0, help="Resolution in Angstroms")
@click.option("--spacing", "-s", default=None, type=float, help="Grid spacing in Angstroms")
@click.option("--snr", default=None, type=float, help="Signal-to-noise ratio")
@click.option("--defocus", default=2.0, help="Defocus in micrometers")
@click.option("--voltage", default=300.0, help="Acceleration voltage in kV")
@click.option("--cs", default=2.7, help="Spherical aberration in mm")
@click.option("--bfactor", default=0.0, help="Envelope B-factor")
@click.option(
    "--bfactors/--no-bfactors", default=False, help="Use atomic B-factors for local resolution"
)
@click.option("--apply-physics/--no-physics", default=False, help="Apply CTF effects")
def main(
    input_path: str,
    output_path: str,
    resolution: float,
    spacing: float | None,
    snr: float | None,
    defocus: float,
    voltage: float,
    cs: float,
    bfactor: float,
    bfactors: bool,
    apply_physics: bool,
) -> None:
    """
    Generate a synthetic Cryo-EM map from an atomic model (PDB, mmCIF, or BCIF).
    """
    try:
        click.echo(f"Generating map for {input_path} at {resolution}A resolution...")

        grid, origin = generate_density_map(
            input_path, resolution, grid_spacing=spacing, use_bfactors=bfactors
        )

        data = np.array(grid, copy=True)

        # Voxel size is from the unit cell
        uc = grid.unit_cell
        vox_size = (uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)

        if apply_physics:
            click.echo(
                f"Applying CTF (defocus={defocus}um, voltage={voltage}kV, B-factor={bfactor})..."
            )
            data = apply_ctf(
                data, vox_size, defoc=defocus, cs=cs, voltage=voltage, b_factor=bfactor
            )

        if snr is not None:
            click.echo(f"Adding Gaussian noise (SNR={snr})...")
            data = add_gaussian_noise(data, snr)

        save_mrc(data, output_path, origin=origin, spacing=vox_size)
        click.echo(f"Saved synthetic map to {output_path}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    main()
