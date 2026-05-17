import unittest
import os
import numpy as np
import mrcfile
import gemmi
import urllib.request
from synth_cryo_em.core import generate_density_map, compute_ccc, compute_fsc

class TestEmpiricalValidation(unittest.TestCase):
    """
    Functional tests comparing synthetic results with empirical expectations.
    Uses small real-world structures to validate correlation.
    """
    
    @classmethod
    def setUpClass(cls):
        # We use a very small, well-known protein: Crambin (PDB: 1CRN)
        # It's small (46 residues) and stable.
        cls.pdb_id = "1crn"
        cls.pdb_path = f"{cls.pdb_id}.pdb"
        
        if not os.path.exists(cls.pdb_path):
            url = f"https://files.rcsb.org/download/{cls.pdb_id}.pdb"
            try:
                urllib.request.urlretrieve(url, cls.pdb_path)
            except Exception as e:
                print(f"Skipping empirical test: Could not download {url}. Error: {e}")
                cls.pdb_path = None

    @classmethod
    def tearDownClass(cls):
        if cls.pdb_path and os.path.exists(cls.pdb_path):
            os.remove(cls.pdb_path)

    def test_crambin_reconstruction_consistency(self):
        """
        Validate that generating a map at a specific resolution and 
        re-evaluating it against the same model yields high correlation.
        This serves as a baseline 'internal' empirical check.
        """
        if not self.pdb_path:
            self.skipTest("PDB file not available")

        res = 3.0
        # Use fixed parameters for consistency
        spacing = 1.0
        margin = 10.0
        grid, _ = generate_density_map(self.pdb_path, resolution=res, grid_spacing=spacing, margin=margin)
        data = np.array(grid, copy=True)

        # Now generate a 'reference' map using the same parameters
        grid_ref, _ = generate_density_map(self.pdb_path, resolution=res, grid_spacing=spacing, margin=margin)
        data_ref = np.array(grid_ref, copy=True)

        ccc = compute_ccc(data, data_ref)

        # Identical parameters should yield identical results
        self.assertAlmostEqual(ccc, 1.0, places=5)

    def test_resolution_cutoffs(self):
        """
        Validate that the FSC correctly reflects the simulated resolution.
        If we simulate at 6A, the FSC against a higher resolution (3A) version 
        should show a significant drop.
        """
        if not self.pdb_path:
            self.skipTest("PDB file not available")

        res_low = 8.0
        res_high = 2.0
        # Use a fixed grid spacing and margin for both to ensure same shape
        spacing = 1.0
        margin = 15.0

        grid_low, _ = generate_density_map(self.pdb_path, resolution=res_low, grid_spacing=spacing, margin=margin)
        grid_high, _ = generate_density_map(self.pdb_path, resolution=res_high, grid_spacing=spacing, margin=margin)

        # Ensure grids are the same size for comparison
        data_low = np.array(grid_low, copy=True)
        data_high = np.array(grid_high, copy=True)

        uc = grid_low.unit_cell
        voxel_size = (uc.a / grid_low.nu, uc.b / grid_low.nv, uc.c / grid_low.nw)

        freqs, fsc = compute_fsc(data_low, data_high, voxel_size)

        # The FSC should be lower at higher frequencies
        # Check that FSC at high frequency (near Nyquist) is much lower than at low frequency
        self.assertGreater(fsc[1], fsc[-1], "FSC should decrease with frequency")

        # Check for a drop below 0.5 at some point
        low_fsc_indices = np.where(fsc < 0.5)[0]
        self.assertGreater(len(low_fsc_indices), 0, "FSC should drop below 0.5 for different resolutions")

if __name__ == '__main__':
    unittest.main()
