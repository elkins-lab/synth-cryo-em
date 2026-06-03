import click
import mrcfile
import numpy as np

from .core import compute_ccc, compute_fsc


@click.command()
@click.argument("map1_path", type=click.Path(exists=True))
@click.argument("map2_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Path to save FSC data (CSV)")
def main(map1_path: str, map2_path: str, output: str | None) -> None:
    """
    Compare two Cryo-EM maps using Fourier Shell Correlation (FSC) and CCC.
    """
    click.echo(f"Comparing {map1_path} and {map2_path}...")

    with mrcfile.open(map1_path) as m1, mrcfile.open(map2_path) as m2:
        d1 = m1.data
        d2 = m2.data
        v1 = m1.voxel_size

        if d1.shape != d2.shape:
            click.echo("Error: Maps have different shapes. Resampling not yet supported.", err=True)
            return

        voxel_size = (v1.x, v1.y, v1.z)
        freqs, fsc = compute_fsc(d1, d2, voxel_size)
        ccc = compute_ccc(d1, d2)

    click.echo(f"\nOverall Cross-Correlation Coefficient (CCC): {ccc:.4f}\n")

    # Print some key values
    click.echo(f"{'Resolution (A)':<15} | {'FSC':<10}")
    click.echo("-" * 30)

    # Ensure step is at least 1
    step = max(1, len(freqs) // 10)
    for i in range(0, len(freqs), step):
        res = 1.0 / freqs[i] if freqs[i] > 0 else float("inf")
        click.echo(f"{res:<15.2f} | {fsc[i]:<10.4f}")

    # Find 0.5 and 0.143 crossings
    for val in [0.5, 0.143]:
        cross_idx = np.where(fsc < val)[0]
        if len(cross_idx) > 0:
            res = 1.0 / freqs[cross_idx[0]]
            click.echo(f"\nFSC={val} crossing at {res:.2f} Angstroms")

    if output:
        np.savetxt(output, np.column_stack((freqs, fsc)), delimiter=",", header="frequency,fsc")
        click.echo(f"\nFSC data saved to {output}")


if __name__ == "__main__":
    main()
