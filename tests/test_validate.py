import os


import mrcfile
import numpy as np
from click.testing import CliRunner

from synth_cryo_em.validate import main


def create_dummy_mrc(path: str, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = (1.0, 1.0, 1.0)


def test_validate_basic() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        data1 = np.random.rand(20, 20, 20).astype(np.float32)
        data2 = data1 + np.random.normal(0, 0.1, (20, 20, 20)).astype(np.float32)

        create_dummy_mrc("map1.mrc", data1)
        create_dummy_mrc("map2.mrc", data2)

        result = runner.invoke(main, ["map1.mrc", "map2.mrc"])
        assert result.exit_code == 0
        assert "Overall Cross-Correlation Coefficient" in result.output
        assert "Resolution (A)" in result.output


def test_validate_output_csv() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        data1 = np.random.rand(20, 20, 20).astype(np.float32)
        data2 = data1.copy()

        create_dummy_mrc("map1.mrc", data1)
        create_dummy_mrc("map2.mrc", data2)

        result = runner.invoke(main, ["map1.mrc", "map2.mrc", "--output", "fsc.csv"])
        assert result.exit_code == 0
        assert os.path.exists("fsc.csv")

        # Check if CSV has content
        fsc_data = np.loadtxt("fsc.csv", delimiter=",")
        assert fsc_data.shape[1] == 2


def test_validate_identical() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        data = np.random.rand(20, 20, 20).astype(np.float32)
        create_dummy_mrc("map1.mrc", data)
        create_dummy_mrc("map2.mrc", data)

        result = runner.invoke(main, ["map1.mrc", "map2.mrc"])
        assert result.exit_code == 0
        assert "Overall Cross-Correlation Coefficient (CCC): 1.0000" in result.output


def test_validate_mismatched_shapes_value_error() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Maps with different shapes
        data1 = np.zeros((10, 10, 10), dtype=np.float32)
        data2 = np.zeros((10, 10, 11), dtype=np.float32)

        create_dummy_mrc("map1.mrc", data1)
        create_dummy_mrc("map2.mrc", data2)

        # compute_ccc will raise ValueError due to shape mismatch
        result = runner.invoke(main, ["map1.mrc", "map2.mrc"])
        assert result.exit_code != 0
        assert "Error: Maps have different shapes" in result.output
