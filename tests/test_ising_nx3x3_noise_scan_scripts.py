import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyqed._letta_one_site_opt import LatticeLETTA


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "letta_3D"
    / "Jul24th_TFI_n33"
)
RUNNER_PATH = SCRIPT_ROOT / "run_ising_Nx3x3_letta_D2_noise_scan.py"
LAUNCHER_PATH = SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D2_noise_scan.sh"
COLLECTOR_PATH = SCRIPT_ROOT / "collect_ising_Nx3x3_letta_D2_noise_scan.py"


def _load_runner():
    if not RUNNER_PATH.exists():
        raise AssertionError(f"missing noise-scan runner: {RUNNER_PATH}")
    specification = importlib.util.spec_from_file_location(
        RUNNER_PATH.stem,
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class IsingNx3x3NoiseScanScriptTests(unittest.TestCase):
    def test_noise_scan_files_exist(self):
        self.assertTrue(RUNNER_PATH.is_file())
        self.assertTrue(LAUNCHER_PATH.is_file())
        self.assertTrue(COLLECTOR_PATH.is_file())

    def test_noise_scan_has_the_four_requested_fixed_noises(self):
        runner = _load_runner()
        self.assertEqual(
            runner.NOISE_LEVELS,
            (0.002, 0.001, 0.0005, 0.0001),
        )
        self.assertEqual(
            [runner._noise_label(noise) for noise in runner.NOISE_LEVELS],
            ["0p002", "0p001", "0p0005", "0p0001"],
        )

    def test_scan_uses_existing_d1_parent_and_isolates_each_noise(self):
        runner = _load_runner()
        results = Path("/tmp/results/letta_warmstart")
        states = Path("/tmp/states/letta_warmstart")
        paths = runner._scan_paths(results, states, n_layers=3, noise=0.0005)

        self.assertEqual(paths["d1_result"], results / "D1" / "N03.json")
        self.assertEqual(paths["d1_state"], states / "D1" / "N03.npz")
        self.assertEqual(
            paths["d2_result"],
            results / "D2_from_D1_noise0p0005_qr" / "N03.json",
        )
        self.assertEqual(
            paths["d2_state"],
            states / "D2_from_D1_noise0p0005_qr" / "N03.npz",
        )

    def test_scan_expansion_uses_exact_requested_noise_and_full_d2_rank(self):
        runner = _load_runner()
        parent = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )
        expanded, ranks, kick_seed = runner._expand_fixed_noise(
            parent,
            noise=0.0005,
            seed=4,
        )
        expected = parent.expand_bond_dimension(
            2,
            noise=0.0005,
            seed=1004,
        )

        self.assertEqual(kick_seed, 1004)
        self.assertEqual(ranks, (2,) * 7)
        for actual, reference in zip(expanded.tensors, expected.tensors):
            np.testing.assert_allclose(actual, reference, atol=1.0e-13)
        with self.assertRaisesRegex(ValueError, "not in the configured scan"):
            runner._expand_fixed_noise(parent, noise=0.005, seed=4)

    def test_scan_launcher_submits_four_parallel_d2_cases_from_existing_d1(self):
        self.assertTrue(LAUNCHER_PATH.is_file())
        launcher = LAUNCHER_PATH.read_text()

        self.assertIn("N_LAYERS=${N_LAYERS:-3}", launcher)
        self.assertIn("D2_SWEEPS=${D2_SWEEPS:-30}", launcher)
        self.assertIn("--array=0-3", launcher)
        self.assertNotIn("--dependency=afterok", launcher)
        self.assertIn("--dependency=afterany:${D2_JOB}", launcher)
        self.assertIn("--mem=256G", launcher)
        self.assertIn(
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg",
            launcher,
        )
        for noise in ("0.002", "0.001", "0.0005", "0.0001"):
            self.assertIn(noise, launcher)

    def test_collector_reports_successful_noise_cases_without_requiring_all(self):
        specification = importlib.util.spec_from_file_location(
            COLLECTOR_PATH.stem,
            COLLECTOR_PATH,
        )
        collector = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "solver": "letta",
                "bond_dimension": 2,
                "N": 3,
                "energy": -61.97,
                "runtime_seconds": 1.0,
                "requested_kick": 0.001,
                "parent_state_sha256": "same-parent",
            }
            path = root / "D2_from_D1_noise0p001_qr" / "N03.json"
            path.parent.mkdir(parents=True)
            path.write_text(__import__("json").dumps(payload))
            rows = collector.collect(root, n_layers=3)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["noise"], 0.001)


if __name__ == "__main__":
    unittest.main()
