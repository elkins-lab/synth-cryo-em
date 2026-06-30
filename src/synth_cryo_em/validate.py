import click
import mrcfile
import numpy as np
from synth_core import compute_fsc

from .core import compute_ccc


def compute_and_report_fsc(
    freqs: np.ndarray, fsc: np.ndarray, ccc: float, output: str | None = None
) -> None:
    """
    Report FSC and CCC results to the console and optionally to a file.
    """
    click.echo(f"Overall Cross-Correlation Coefficient (CCC): {ccc:.4f}")
    click.echo("\nFSC Summary:")
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


@click.command()
@click.argument("map1_path", type=click.Path(exists=True))
@click.argument("map2_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Path to save FSC data (CSV)")
def main(map1_path: str, map2_path: str, output: str | None) -> None:
    """
    Compare two Cryo-EM maps using Fourier Shell Correlation (FSC) and CCC.

    This function acts as the CLI entrypoint for map validation. It loads two
    MRC maps, verifies their dimensions, and computes correlation metrics.
    """
    try:
        with mrcfile.open(map1_path) as m1, mrcfile.open(map2_path) as m2:
            data1 = m1.data
            data2 = m2.data
            voxel_size = (m1.voxel_size.z, m1.voxel_size.y, m1.voxel_size.x)

        ccc = compute_ccc(data1, data2)
        freqs, fsc = compute_fsc(data1, data2, voxel_size)

        compute_and_report_fsc(freqs, fsc, ccc, output)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    main()
