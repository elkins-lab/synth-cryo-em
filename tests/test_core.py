import unittest
import os
import numpy as np
from synth_cryo_em.core import generate_density_map, apply_ctf, add_gaussian_noise

class TestSynthCryoEM(unittest.TestCase):
    def setUp(self):
        self.pdb_content = """ATOM      1  N   ALA A   1      11.104   6.132  11.469  1.00 20.00           N  
ATOM      2  CA  ALA A   1      12.000  12.000  12.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      13.104  18.132  13.469  1.00 20.00           C  
TER
END
"""
        self.test_pdb = "test_temp.pdb"
        with open(self.test_pdb, "w") as f:
            f.write(self.pdb_content)

    def tearDown(self):
        if os.path.exists(self.test_pdb):
            os.remove(self.test_pdb)

    def test_generate_density(self):
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=False)
        self.assertGreater(np.sum(data), 0)
        self.assertEqual(len(data.shape), 3)

    def test_apply_ctf(self):
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=True)
        uc = grid.unit_cell
        vox_size = (uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)
        data_ctf = apply_ctf(data, vox_size, defoc=1.0)
        self.assertEqual(data_ctf.shape, data.shape)
        # CTF should change the values
        self.assertFalse(np.allclose(data, data_ctf))

    def test_add_noise(self):
        data = np.ones((10, 10, 10))
        noisy = add_gaussian_noise(data, snr=10)
        self.assertEqual(noisy.shape, data.shape)
        self.assertNotEqual(np.mean(noisy), 1.0)

    def test_apply_ctf_with_bfactor(self):
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0)
        data = np.array(grid, copy=True)
        uc = grid.unit_cell
        vox_size = (uc.a / grid.nu, uc.b / grid.nv, uc.c / grid.nw)
        data_ctf = apply_ctf(data, vox_size, defoc=1.0, b_factor=100.0)
        self.assertEqual(data_ctf.shape, data.shape)
        self.assertFalse(np.allclose(data, data_ctf))

    def test_generate_density_no_atoms(self):
        empty_pdb = "empty.pdb"
        with open(empty_pdb, "w") as f:
            f.write("END\n")
        try:
            with self.assertRaises(ValueError):
                generate_density_map(empty_pdb, resolution=4.0)
        finally:
            if os.path.exists(empty_pdb):
                os.remove(empty_pdb)

    def test_generate_with_bfactors(self):
        grid, origin = generate_density_map(self.test_pdb, resolution=4.0, use_bfactors=True)
        data = np.array(grid, copy=False)
        self.assertGreater(np.sum(data), 0)

    def test_mmcif_support(self):
        import gemmi
        cif_path = "test_mmcif.cif"
        st = gemmi.Structure()
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(1, ' ')
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

    def test_save_mrc(self):
        from synth_cryo_em.core import save_mrc
        import mrcfile
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

if __name__ == '__main__':
    unittest.main()
