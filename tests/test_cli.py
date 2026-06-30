import os
import unittest
from click.testing import CliRunner
from synth_cryo_em.cli import main

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.pdb_content = """ATOM      1  N   ALA A   1      11.104   6.132  11.469  1.00 20.00           N  
ATOM      2  CA  ALA A   1      12.000  12.000  12.000  1.00 20.00           C  
ATOM      3  C   ALA A   1      13.104  18.132  13.469  1.00 20.00           C  
TER
END
"""
        self.test_pdb = "test_cli.pdb"
        with open(self.test_pdb, "w") as f:
            f.write(self.pdb_content)

    def tearDown(self):
        if os.path.exists(self.test_pdb):
            os.remove(self.test_pdb)
        if os.path.exists("out.mrc"):
            os.remove("out.mrc")

    def test_basic_generation(self):
        result = self.runner.invoke(main, [self.test_pdb, "out.mrc"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists("out.mrc"))

    def test_with_options(self):
        result = self.runner.invoke(main, [
            self.test_pdb, "out.mrc",
            "--resolution", "5.0",
            "--spacing", "1.5",
            "--snr", "10",
            "--apply-physics",
            "--defocus", "1.5",
            "--voltage", "300",
            "--cs", "2.7",
            "--bfactor", "50.0",
            "--bfactors"
        ])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.exists("out.mrc"))

if __name__ == '__main__':
    unittest.main()
