import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "letta_3D"
    / "Jul24th_TFI_n33"
)
COLLECTOR_PATH = SCRIPT_ROOT / "collect_ising_Nx3x3_all.py"
LAUNCHER_PATH = SCRIPT_ROOT / "collect_ising_Nx3x3_all.sh"


def _load_collector():
    if not COLLECTOR_PATH.exists():
        raise AssertionError(f"missing collector: {COLLECTOR_PATH}")
    specification = importlib.util.spec_from_file_location(
        COLLECTOR_PATH.stem,
        COLLECTOR_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_result(root, relative_path, **overrides):
    payload = {
        "solver": "letta",
        "bond_dimension": 3,
        "N": 3,
        "lattice_shape": [3, 3, 3],
        "nsites": 27,
        "energy": -61.9,
        "runtime_seconds": 12.5,
        "converged": False,
        "coupling": 1.0,
        "field": 1.5,
        "tolerance": 1.0e-8,
        "seed": 4,
        "ordering": "compact",
        "requested_sweeps": 20,
        "actual_sweeps": 20,
        "initialization": "D2_embedding_plus_random_kick",
        "requested_kick": 1.0e-3,
        "used_kick": 1.0e-4,
        "energy_history": [
            {
                "iteration": 1,
                "direction": "lr",
                "energy": -61.8,
                "energy_change": 0.1,
                "energy_density_change": 0.1 / 27,
            }
        ],
    }
    payload.update(overrides)
    path = Path(root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class IsingNx3x3CollectionScriptTests(unittest.TestCase):
    def test_collector_and_launcher_exist(self):
        self.assertTrue(COLLECTOR_PATH.is_file())
        self.assertTrue(LAUNCHER_PATH.is_file())

    def test_method_matrix_contains_all_requested_dimensions_and_scans(self):
        collector = _load_collector()
        approximate = {
            (
                method.solver,
                method.scan_sequence,
                method.bond_dimension,
            )
            for method in collector.METHOD_SPECS
            if method.solver != "exact"
        }

        self.assertEqual(
            {
                dimension
                for solver, scan, dimension in approximate
                if solver == "letta" and scan == "compact"
            },
            {1, 2, 3, 4, 8},
        )
        for scan_sequence in ("compact", "continuous-snake"):
            self.assertEqual(
                {
                    dimension
                    for solver, scan, dimension in approximate
                    if solver == "mps" and scan == scan_sequence
                },
                {1, 2, 4, 8, 16},
            )
        self.assertIn(
            "letta_noise0p05_D2",
            {method.method_id for method in collector.METHOD_SPECS},
        )
        self.assertIn(
            "letta_noise0p005_D2",
            {method.method_id for method in collector.METHOD_SPECS},
        )
        self.assertIn(
            "letta_restart_noise0p001_D2",
            {method.method_id for method in collector.METHOD_SPECS},
        )
        self.assertIn(
            "letta_restart_noise0p001_kick14_D2",
            {method.method_id for method in collector.METHOD_SPECS},
        )

    def test_collection_normalizes_warm_letta_and_snake_mps(self):
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / "results"
            _write_result(
                results_root,
                "letta_warmstart/D3_from_D2/N03.json",
            )
            _write_result(
                results_root,
                "mps_snake_D8/N03.json",
                solver="mps",
                bond_dimension=8,
                ordering="continuous-snake",
                converged=True,
                sweeps=20,
                initialization=None,
                requested_sweeps=None,
                actual_sweeps=None,
                requested_kick=None,
                used_kick=None,
            )

            rows, availability = collector.collect_results(
                results_root,
                min_n=3,
                max_n=3,
                coupling=1.0,
                field=1.5,
            )

        self.assertEqual(len(rows), 2)
        by_method = {row["method_id"]: row for row in rows}
        self.assertEqual(
            by_method["letta_warm_D3"]["initialization_sequence"],
            "D2_to_D3",
        )
        self.assertEqual(
            by_method["letta_warm_D3"]["requested_sweeps"],
            20,
        )
        self.assertEqual(
            by_method["letta_warm_D3"]["actual_sweeps"],
            20,
        )
        self.assertEqual(
            by_method["mps_snake_D8"]["scan_sequence"],
            "continuous-snake",
        )
        self.assertEqual(
            by_method["mps_snake_D8"]["actual_sweeps"],
            1,
        )
        self.assertEqual(len(availability), len(collector.METHOD_SPECS))
        self.assertEqual(
            sum(row["status"] == "collected" for row in availability),
            2,
        )

    def test_invalid_metadata_is_reported_without_stopping_collection(self):
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            results_root = Path(directory) / "results"
            _write_result(
                results_root,
                "mps_compact_D1/N03.json",
                solver="mps",
                bond_dimension=1,
                ordering="continuous-snake",
                sweeps=20,
            )

            rows, availability = collector.collect_results(
                results_root,
                min_n=3,
                max_n=3,
                coupling=1.0,
                field=1.5,
            )

        self.assertEqual(rows, [])
        status = {
            row["method_id"]: row
            for row in availability
        }["mps_compact_D1"]
        self.assertEqual(status["status"], "invalid")
        self.assertIn("ordering", status["detail"])

    def test_collection_writes_summary_history_availability_and_manifest(self):
        collector = _load_collector()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_root = root / "results"
            output_root = root / "collected"
            _write_result(
                results_root,
                "letta_warmstart/D3_from_D2/N03.json",
            )
            rows, availability = collector.collect_results(
                results_root,
                min_n=3,
                max_n=3,
                coupling=1.0,
                field=1.5,
            )

            paths = collector.write_collection(
                rows,
                availability,
                output_root,
                results_root=results_root,
                min_n=3,
                max_n=3,
                coupling=1.0,
                field=1.5,
            )

            for path in paths.values():
                self.assertTrue(path.is_file())
            with paths["summary_csv"].open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            with paths["history_csv"].open(newline="") as handle:
                history = list(csv.DictReader(handle))
            manifest = json.loads(paths["manifest_json"].read_text())

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["method_id"], "letta_warm_D3")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["direction"], "lr")
        self.assertEqual(manifest["counts"]["collected"], 1)
        self.assertEqual(
            manifest["counts"]["expected"],
            len(collector.METHOD_SPECS),
        )

    def test_launcher_uses_cluster_branch_and_collection_paths(self):
        self.assertTrue(LAUNCHER_PATH.is_file())
        launcher = LAUNCHER_PATH.read_text()

        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn(
            'collect_ising_Nx3x3_all.py',
            launcher,
        )
        self.assertIn(
            '--results-root "${RUN_ROOT}/results"',
            launcher,
        )
        self.assertIn(
            '--output-root "${RUN_ROOT}/collected_data"',
            launcher,
        )
        self.assertIn('--min-N "${MIN_N}"', launcher)
        self.assertIn('--max-N "${MAX_N}"', launcher)
        self.assertIn("--mem=4G", launcher)


if __name__ == "__main__":
    unittest.main()
