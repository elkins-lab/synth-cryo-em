import os
import tempfile
import unittest

import numpy as np

from synth_cryo_em.core import apply_ctf, compute_ccc, generate_density_map


class TestRobustness(unittest.TestCase):
    def setUp(self) -> None:
        self.pdb_content = (
            "ATOM      1  CA  ALA A   1      10.000  10.000  10.000  1.00 20.00           C\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            tmp.write(self.pdb_content)
            self.test_pdb = tmp.name

    def tearDown(self) -> None:
        if os.path.exists(self.test_pdb):
            os.remove(self.test_pdb)

    def test_generate_density_invalid_resolution(self) -> None:
        with self.assertRaises(ValueError):
            generate_density_map(self.test_pdb, resolution=0.0)
        with self.assertRaises(ValueError):
            generate_density_map(self.test_pdb, resolution=-1.0)

    def test_generate_density_invalid_spacing(self) -> None:
        with self.assertRaises(ValueError):
            generate_density_map(self.test_pdb, resolution=3.0, grid_spacing=0.0)
        with self.assertRaises(ValueError):
            generate_density_map(self.test_pdb, resolution=3.0, grid_spacing=-0.5)

    def test_generate_density_invalid_margin(self) -> None:
        with self.assertRaises(ValueError):
            generate_density_map(self.test_pdb, resolution=3.0, margin=-1.0)

    def test_apply_ctf_invalid_voxel_size(self) -> None:
        data = np.zeros((10, 10, 10))
        with self.assertRaises(ValueError):
            apply_ctf(data, (0.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            apply_ctf(data, (1.0, -1.0, 1.0))

    def test_apply_ctf_invalid_voltage(self) -> None:
        data = np.zeros((10, 10, 10))
        with self.assertRaises(ValueError):
            apply_ctf(data, (1.0, 1.0, 1.0), voltage=0)
        with self.assertRaises(ValueError):
            apply_ctf(data, (1.0, 1.0, 1.0), voltage=-300)

    def test_apply_ctf_invalid_amplitude_contrast(self) -> None:
        data = np.zeros((10, 10, 10))
        with self.assertRaises(ValueError):
            apply_ctf(data, (1.0, 1.0, 1.0), amplitude_contrast=1.5)
        with self.assertRaises(ValueError):
            apply_ctf(data, (1.0, 1.0, 1.0), amplitude_contrast=-1.5)

    def test_compute_ccc_mismatched_shapes(self) -> None:
        data1 = np.zeros((10, 10, 10))
        data2 = np.zeros((10, 10, 11))
        with self.assertRaises(ValueError):
            compute_ccc(data1, data2)


if __name__ == "__main__":
    unittest.main()
