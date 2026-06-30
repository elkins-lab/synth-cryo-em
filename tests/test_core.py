import os
import unittest

import numpy as np
from synth_core import add_gaussian_noise

from synth_cryo_em.core import apply_ctf, compute_ccc, generate_density_map


class TestSynthCryoEM(unittest.TestCase):
    def setUp(self) -> None:
        self.pdb_content = """ATOM      1  N   ALA A   1      11.104   6.132  11.469  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      3  C   ALA A   1      13.104  18.132  13.469  1.00 20.00           C
TER
END
"""
        self.test_pdb = "test_temp.pdb"
        with open(self.test_pdb, "w") as f:
            f.write(self.pdb_content)

    def tearDown(self) -> None:
        if os.path.exists(self.test_pdb):
            os.remove(self.test_pdb)

    def test_generate_density(self) -> None:
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=False)
        self.assertGreater(np.sum(data), 0)
        self.assertEqual(len(data.shape), 3)

    def test_apply_ctf(self) -> None:
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=True)
        uc = grid.unit_cell
        vox_size = (uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)
        data_ctf = apply_ctf(data, vox_size, defoc=1.0)
        self.assertEqual(data_ctf.shape, data.shape)
        # CTF should change the values
        self.assertFalse(np.allclose(data, data_ctf))

    def test_add_noise(self) -> None:
        data = np.ones((10, 10, 10))
        noisy = add_gaussian_noise(data, snr=10)
        self.assertEqual(noisy.shape, data.shape)
        self.assertNotEqual(np.mean(noisy), 1.0)

    def test_apply_ctf_with_bfactor(self) -> None:
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=True)
        uc = grid.unit_cell
        vox_size = (uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)
        data_ctf = apply_ctf(data, vox_size, defoc=1.0, b_factor=100.0)
        self.assertEqual(data_ctf.shape, data.shape)
        self.assertFalse(np.allclose(data, data_ctf))

    def test_generate_density_no_atoms(self) -> None:
        empty_pdb = "empty.pdb"
        with open(empty_pdb, "w") as f:
            f.write("END\n")
        try:
            with self.assertRaises(ValueError):
                generate_density_map(empty_pdb, resolution=4.0)
        finally:
            if os.path.exists(empty_pdb):
                os.remove(empty_pdb)

    def test_generate_with_bfactors(self) -> None:
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0, use_bfactors=True)
        data = np.array(grid, copy=False)
        self.assertGreater(np.sum(data), 0)

    def test_mmcif_support(self) -> None:
        import gemmi

        cif_path = "test_mmcif.cif"
        st = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(10, 10, 10)
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        st.add_model(model)
        st.make_mmcif_document().write_file(cif_path)

        try:
            grid, origin = generate_density_map(cif_path, resolution=4.0)
            data = np.array(grid, copy=False)
            self.assertGreater(np.sum(data), 0)
        finally:
            if os.path.exists(cif_path):
                os.remove(cif_path)

    def test_save_mrc(self) -> None:
        import mrcfile

        from synth_cryo_em.core import save_mrc

        data = np.zeros((10, 10, 10), dtype=np.float32)
        test_mrc = "test_output.mrc"
        try:
            save_mrc(data, test_mrc, origin=(1, 2, 3), spacing=(1.1, 1.1, 1.1))
            self.assertTrue(os.path.exists(test_mrc))
            with mrcfile.open(test_mrc) as mrc:
                self.assertEqual(mrc.data.shape, (10, 10, 10))
                self.assertAlmostEqual(mrc.voxel_size.x, 1.1, places=5)
                self.assertAlmostEqual(mrc.header.origin.x, 1.0)
        finally:
            if os.path.exists(test_mrc):
                os.remove(test_mrc)

    def test_compute_ccc_zero_den(self) -> None:
        # Create an array of constant values
        data1 = np.ones((5, 5, 5))
        data2 = np.ones((5, 5, 5))
        # This will result in 0 variance -> den == 0
        ccc = compute_ccc(data1, data2)
        self.assertEqual(ccc, 0.0)

    def test_compute_fsc_anisotropic(self) -> None:
        """
        Regression test for axis mismatch in compute_fsc.
        Ensures that voxel_size dimensions are correctly mapped to array axes
        by using a highly anisotropic box and a signal that would be scrambled
        if axes were swapped.
        """
        from synth_core import compute_fsc

        # Create a highly anisotropic box: (40, 10, 20)
        shape = (40, 10, 20)
        # Voxel sizes are very different to make any swap obvious
        voxel_size = (1.0, 2.0, 4.0)

        z, y, x = np.indices(shape)
        # Center in grid units
        cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2

        # A simple Gaussian signal
        # Use a sigma that is large in grid units but small in Angstroms
        # to ensure it's well-sampled but still drops off.
        dist2 = (
            ((z - cz) * voxel_size[0]) ** 2
            + ((y - cy) * voxel_size[1]) ** 2
            + ((x - cx) * voxel_size[2]) ** 2
        )
        data = np.exp(-dist2 / 50.0)

        # 1. Self-correlation should be 1.0
        freqs, fsc = compute_fsc(data, data, voxel_size)
        self.assertTrue(
            np.allclose(fsc, 1.0, atol=1e-5), f"FSC with self should be 1.0, got {fsc[0]}"
        )

        # 2. Add noise to one map. FSC should drop at high frequencies.
        # This is the most reliable way to test frequency-dependent drop.
        noise = np.random.normal(0, 0.05, data.shape)
        data_noisy = data + noise

        freqs, fsc = compute_fsc(data, data_noisy, voxel_size)

        # Low frequency should still have high correlation
        self.assertGreater(fsc[0], 0.8)
        # High frequency should be much lower due to noise dominance
        self.assertGreater(fsc[0], fsc[-1], "FSC should decrease with frequency")

    def test_apply_ctf_anisotropic(self) -> None:
        """
        Regression test for axis mismatch in apply_ctf.
        Ensures that voxel_size dimensions are correctly mapped to array axes.
        """
        # Create a signal that is very different in each dimension
        # A long "cigar" shape along one axis.
        shape = (40, 10, 20)
        voxel_size = (1.0, 2.0, 4.0)
        z, y, x = np.indices(shape)
        cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2

        # Sigma is 5.0 Angstroms. In grid units, this is:
        # sz = 5.0, sy = 2.5, sx = 1.25
        dist2 = (
            ((z - cz) * voxel_size[0]) ** 2
            + ((y - cy) * voxel_size[1]) ** 2
            + ((x - cx) * voxel_size[2]) ** 2
        )
        data = np.exp(-dist2 / 50.0)

        # Apply CTF. If axes are swapped, the phase shifts will be applied
        # at the wrong frequencies for the signal's elongation.
        data_ctf = apply_ctf(data, voxel_size, defoc=1.0)

        self.assertEqual(data_ctf.shape, data.shape)
        self.assertFalse(np.allclose(data, data_ctf))

        # We can also check that a B-factor works correctly
        data_b = apply_ctf(data, voxel_size, defoc=0.0, b_factor=100.0)
        # B-factor should blur the map
        self.assertLess(np.max(data_b), np.max(data))


if __name__ == "__main__":
    unittest.main()
