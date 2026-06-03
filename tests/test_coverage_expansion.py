import os
import tempfile
import unittest

import gemmi
import numpy as np

from synth_cryo_em.core import compute_ccc, generate_density_map
from synth_cryo_em.validate import compute_and_report_fsc


class TestCoverageExpansion(unittest.TestCase):
    def test_generate_density_map_no_atoms(self) -> None:
        """Test ValueError when no atoms are found in structure."""
        # Create an empty PDB file
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            tmp.write("HEADER    EMPTY STRUCTURE\n")
            tmp.write("END\n")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as cm:
                generate_density_map(tmp_path, resolution=3.0)
            self.assertEqual(str(cm.exception), "No atoms found in structure")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_compute_ccc_zero_den(self) -> None:
        """Test CCC when denominator is zero (e.g. flat maps)."""
        data1 = np.zeros((10, 10, 10))
        data2 = np.zeros((10, 10, 10))
        ccc = compute_ccc(data1, data2)
        self.assertEqual(ccc, 0.0)

    def test_compute_and_report_fsc_output(self) -> None:
        """Test compute_and_report_fsc with output file to cover the output branch."""
        freqs = np.array([0.1, 0.2, 0.3])
        fsc = np.array([1.0, 0.5, 0.1])
        ccc = 0.8

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "fsc_out.csv")
            compute_and_report_fsc(freqs, fsc, ccc, output=output_file)

            self.assertTrue(os.path.exists(output_file))
            content = np.loadtxt(output_file, delimiter=",", skiprows=1)
            self.assertEqual(content.shape, (3, 2))

    def test_generate_density_map_use_bfactors(self) -> None:
        """Test generate_density_map with use_bfactors=True."""
        st = gemmi.Structure()
        model = gemmi.Model(1)
        chain = gemmi.Chain("A")
        res = gemmi.Residue()
        res.name = "ALA"
        # Let's try gemmi.SeqId("1") which is supported by overload 2
        res.seqid = gemmi.SeqId("1")
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(10, 10, 10)
        atom.b_iso = 20.0
        res.add_atom(atom)
        chain.add_residue(res)
        model.add_chain(chain)
        st.add_model(model)
        st.cell = gemmi.UnitCell(20, 20, 20, 90, 90, 90)

        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            st.write_pdb(tmp.name)
            tmp_path = tmp.name

        try:
            grid, _ = generate_density_map(tmp_path, resolution=3.0, use_bfactors=True)
            self.assertIsInstance(grid, gemmi.FloatGrid)
            data = np.array(grid)
            self.assertGreater(np.max(data), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
