import os
import unittest
import numpy as np
import mrcfile
from click.testing import CliRunner
from synth_cryo_em.validate import main

class TestValidate(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.map1 = "val_map1.mrc"
        self.map2 = "val_map2.mrc"
        self.map3 = "val_map3.mrc"
        self.csv_out = "val_out.csv"
        
        np.random.seed(42)
        data1 = np.random.rand(10, 10, 10).astype(np.float32)
        with mrcfile.new(self.map1, overwrite=True) as m:
            m.set_data(data1)
            m.voxel_size = 1.0
            
        with mrcfile.new(self.map2, overwrite=True) as m:
            m.set_data(data1)
            m.voxel_size = 1.0
            
        data3 = np.random.rand(5, 5, 5).astype(np.float32)
        with mrcfile.new(self.map3, overwrite=True) as m:
            m.set_data(data3)
            m.voxel_size = 1.0

    def tearDown(self):
        for f in [self.map1, self.map2, self.map3, self.csv_out]:
            if os.path.exists(f):
                os.remove(f)

    def test_basic_validation(self):
        result = self.runner.invoke(main, [self.map1, self.map2])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Comparing", result.output)
        self.assertIn("CCC", result.output)

    def test_validation_different_shapes(self):
        result = self.runner.invoke(main, [self.map1, self.map3])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error: Maps have different shapes", result.output)

    def test_validation_with_output(self):
        result = self.runner.invoke(main, [self.map1, self.map2, "--output", self.csv_out])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists(self.csv_out))

if __name__ == '__main__':
    unittest.main()
