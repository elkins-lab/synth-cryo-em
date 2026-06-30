import os

import mrcfile
import numpy as np
from click.testing import CliRunner

from synth_cryo_em.cli import main


def test_cli_basic() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy PDB file
        with open("test.pdb", "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
            )

        result = runner.invoke(main, ["test.pdb", "output.mrc", "--resolution", "4.0"])
        assert result.exit_code == 0
        assert os.path.exists("output.mrc")

        with mrcfile.open("output.mrc") as mrc:
            assert mrc.data.shape != (0, 0, 0)
            assert np.sum(mrc.data) > 0


def test_cli_with_physics_and_noise() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdb", "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
            )

        result = runner.invoke(
            main,
            [
                "test.pdb",
                "output.mrc",
                "--resolution",
                "4.0",
                "--snr",
                "10",
                "--apply-physics",
                "--defocus",
                "1.5",
                "--bfactors",
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists("output.mrc")


def test_cli_invalid_resolution() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdb", "w") as f:
            f.write(
                "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
            )

        # Negative resolution should trigger ValueError and be caught by the CLI
        result = runner.invoke(main, ["test.pdb", "output.mrc", "--resolution", "-1.0"])
        assert result.exit_code != 0
        assert "Error: Resolution must be positive" in result.output
