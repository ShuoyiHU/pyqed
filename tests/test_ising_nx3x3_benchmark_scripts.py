import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "letta_3D"
    / "Jul24th_TFI_n33"
)


def _load_script(name):
    path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsingNx3x3BenchmarkScriptTests(unittest.TestCase):
    def test_runner_uses_n_as_the_extension_direction(self):
        runner = _load_script("run_ising_Nx3x3.py")
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return {
                "energy": -50.0,
                "runtime_seconds": 3.0,
                "converged": True,
                "energy_history": [
                    {"iteration": 1, "energy": -49.0},
                    {"iteration": 2, "energy": -50.0},
                ],
                "metadata": {
                    "lattice_shape": (5, 3, 3),
                    "ordering": "compact",
                    "environment_granularity": "site",
                    "nsites": 45,
                    "nbonds": 96,
                    "mpo_bond_dimension": 11,
                },
            }

        with self._temporary_directory() as tmp_path:
            if not hasattr(runner, "_run_variational_case"):
                self.fail("runner must expose _run_variational_case")
            with patch.object(runner, "_run_variational_case", fake_run):
                output = runner.run_case(
                    solver="letta",
                    n_layers=5,
                    bond_dim=2,
                    output_root=tmp_path,
                    force=True,
                )

            payload = json.loads(output.read_text())

        self.assertEqual(calls[0]["lattice_shape"], (5, 3, 3))
        self.assertEqual(calls[0]["solver"], "letta")
        self.assertEqual(calls[0]["bond_dim"], 2)
        self.assertEqual(calls[0]["ordering"], "compact")
        self.assertEqual(calls[0]["sweeps"], 20)
        self.assertEqual(payload["lattice_shape"], [5, 3, 3])
        self.assertEqual(payload["bond_dimension"], 2)
        self.assertEqual(payload["sweeps"], 20)
        self.assertEqual(
            payload["energy_history"],
            [
                {"iteration": 1, "energy": -49.0},
                {"iteration": 2, "energy": -50.0},
            ],
        )

    def test_mps_orderings_use_separate_result_paths(self):
        runner = _load_script("run_ising_Nx3x3.py")

        def fake_run(**kwargs):
            ordering = kwargs["ordering"]
            return {
                "energy": -30.0,
                "runtime_seconds": 1.0,
                "converged": True,
                "energy_history": [
                    {"iteration": 1, "energy": -30.0}
                ],
                "metadata": {
                    "lattice_shape": (3, 3, 3),
                    "ordering": ordering,
                    "environment_granularity": "site",
                    "nsites": 27,
                    "nbonds": 54,
                    "mpo_bond_dimension": 11,
                },
            }

        with self._temporary_directory() as tmp_path:
            with patch.object(runner, "_run_variational_case", fake_run):
                compact = runner.run_case(
                    solver="mps",
                    ordering="compact",
                    n_layers=3,
                    bond_dim=2,
                    output_root=tmp_path,
                    force=True,
                )
                snake = runner.run_case(
                    solver="mps",
                    ordering="continuous-snake",
                    n_layers=3,
                    bond_dim=2,
                    output_root=tmp_path,
                    force=True,
                )

            compact_payload = json.loads(compact.read_text())
            snake_payload = json.loads(snake.read_text())

        self.assertEqual(compact.parent.name, "mps_compact_D2")
        self.assertEqual(snake.parent.name, "mps_snake_D2")
        self.assertEqual(compact_payload["ordering"], "compact")
        self.assertEqual(snake_payload["ordering"], "continuous-snake")

    def test_matrix_free_tfim_action_matches_dense_reference(self):
        runner = _load_script("run_ising_Nx3x3.py")
        shape = (2, 2, 2)
        coupling = 0.7
        field = 1.2
        operator = runner._tfim_linear_operator(
            shape, coupling=coupling, field=field
        )
        vector = np.linspace(-1.0, 1.0, operator.shape[0])
        actual = operator @ vector

        coordinates = tuple(np.ndindex(*shape))
        site_for = {
            coordinate: site for site, coordinate in enumerate(coordinates)
        }
        bonds = []
        for coordinate in coordinates:
            for axis in range(3):
                neighbor = list(coordinate)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor in site_for:
                    bonds.append((site_for[coordinate], site_for[neighbor]))
        expected = np.zeros_like(vector)
        for basis_index in range(vector.size):
            spins = 1 - 2 * np.array(
                [
                    (basis_index >> site) & 1
                    for site in range(len(coordinates))
                ]
            )
            expected[basis_index] -= coupling * sum(
                spins[left] * spins[right] for left, right in bonds
            ) * vector[basis_index]
            for site in range(len(coordinates)):
                expected[basis_index] -= (
                    field * vector[basis_index ^ (1 << site)]
                )
        np.testing.assert_allclose(actual, expected, atol=1.0e-13)

    def test_result_without_iteration_history_is_recomputed(self):
        runner = _load_script("run_ising_Nx3x3.py")
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs)
            return {
                "energy": -30.0,
                "runtime_seconds": 2.0,
                "converged": True,
                "energy_history": [
                    {"iteration": 1, "energy": -30.0}
                ],
                "metadata": {
                    "lattice_shape": (3, 3, 3),
                    "ordering": "compact",
                    "environment_granularity": "site",
                    "nsites": 27,
                    "nbonds": 54,
                    "mpo_bond_dimension": 11,
                },
            }

        with self._temporary_directory() as tmp_path:
            stale = tmp_path / "letta_D1" / "N03.json"
            stale.parent.mkdir(parents=True)
            stale.write_text(
                json.dumps(
                    {
                        "solver": "letta",
                        "bond_dimension": 1,
                        "N": 3,
                        "lattice_shape": [3, 3, 3],
                        "coupling": 1.0,
                        "field": 1.5,
                        "sweeps": 20,
                        "tolerance": 1.0e-8,
                        "seed": 4,
                    }
                )
            )
            with patch.object(runner, "_run_variational_case", fake_run):
                output = runner.run_case(
                    solver="letta",
                    n_layers=3,
                    bond_dim=1,
                    output_root=tmp_path,
                )

            payload = json.loads(output.read_text())

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            payload["energy_history"],
            [{"iteration": 1, "energy": -30.0}],
        )

    def test_exact_is_matrix_free_for_each_requested_n(self):
        runner = _load_script("run_ising_Nx3x3.py")
        with self._temporary_directory() as tmp_path:
            with patch.object(
                runner,
                "_matrix_free_exact_ground_state",
                return_value=(-60.0, 15.0),
            ) as exact_solver:
                output = runner.run_case(
                    solver="exact",
                    n_layers=3,
                    bond_dim=None,
                    output_root=tmp_path,
                    force=True,
                )

            payload = json.loads(output.read_text())
            exact_solver.assert_called_once()
            self.assertEqual(payload["lattice_shape"], [3, 3, 3])
            self.assertEqual(payload["nsites"], 27)
            self.assertEqual(payload["exact_backend"], "matrix-free Lanczos")

            with patch.object(
                runner,
                "_matrix_free_exact_ground_state",
                return_value=(-80.0, 20.0),
            ):
                n4_output = runner.run_case(
                    solver="exact",
                    n_layers=4,
                    bond_dim=None,
                    output_root=tmp_path,
                    force=True,
                )
            self.assertEqual(
                json.loads(n4_output.read_text())["lattice_shape"],
                [4, 3, 3],
            )

    def test_plotter_collects_requested_results_and_plots_iterations(self):
        plotter = _load_script("plot_ising_Nx3x3.py")
        with self._temporary_directory() as tmp_path:
            results = tmp_path / "results"
            methods = [
                ("letta", "compact", 1),
                ("letta", "compact", 2),
                ("mps", "compact", 1),
                ("mps", "compact", 2),
                ("mps", "compact", 4),
                ("mps", "compact", 8),
                ("mps", "compact", 16),
                ("mps", "continuous-snake", 1),
                ("mps", "continuous-snake", 2),
                ("mps", "continuous-snake", 4),
                ("mps", "continuous-snake", 8),
                ("mps", "continuous-snake", 16),
            ]
            for solver, ordering, bond_dim in methods:
                if solver == "letta":
                    method = f"letta_D{bond_dim}"
                else:
                    short = (
                        "compact"
                        if ordering == "compact"
                        else "snake"
                    )
                    method = f"mps_{short}_D{bond_dim}"
                directory = results / method
                directory.mkdir(parents=True)
                for n_layers in range(3, 11):
                    (directory / f"N{n_layers:02d}.json").write_text(
                        json.dumps(
                            {
                                "solver": solver,
                                "bond_dimension": bond_dim,
                                "N": n_layers,
                                "lattice_shape": [n_layers, 3, 3],
                                "nsites": 9 * n_layers,
                                "energy": -10.0 * n_layers
                                + 0.01 * bond_dim,
                                "runtime_seconds": n_layers * bond_dim,
                                "converged": True,
                                "sweeps": 20,
                                "ordering": ordering,
                                "energy_history": [
                                    {
                                        "iteration": iteration,
                                        "energy": (
                                            -10.0 * n_layers
                                            + 0.01 * bond_dim
                                            + 1.0 / iteration
                                        ),
                                    }
                                    for iteration in range(1, 21)
                                ],
                            }
                        )
                    )
            exact = results / "exact"
            exact.mkdir()
            for n_layers in range(3, 11):
                (exact / f"N{n_layers:02d}.json").write_text(
                    json.dumps(
                        {
                            "solver": "exact",
                            "bond_dimension": None,
                            "N": n_layers,
                            "lattice_shape": [n_layers, 3, 3],
                            "nsites": 9 * n_layers,
                            "energy": -10.0 * n_layers - 1.0,
                            "runtime_seconds": 100.0 * n_layers,
                            "converged": None,
                            "ordering": "compact",
                        }
                    )
                )

            rows = plotter.collect_results(
                results, min_n=3, max_n=10
            )
            plotter.plot_results(rows, tmp_path / "plots", dpi=50)
            self.assertTrue(
                (tmp_path / "plots" / "ising_Nx3x3_energy_vs_N.png").is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "plots"
                    / "ising_Nx3x3_energy_differences.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path / "plots" / "ising_Nx3x3_runtime_vs_N.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    tmp_path
                    / "plots"
                    / "ising_Nx3x3_energy_vs_iteration.png"
                ).is_file()
            )
            iteration_csv = (
                tmp_path / "plots" / "ising_Nx3x3_energy_vs_iteration.csv"
            )
            plotter.write_iteration_csv(rows, iteration_csv)
            self.assertEqual(
                len(iteration_csv.read_text().splitlines()),
                1 + 12 * 8 * 20,
            )

        self.assertEqual(len(rows), 104)
        differences = plotter.energy_difference_series(
            rows, reference=("letta", "compact", 1)
        )
        self.assertEqual(len(differences), 12)
        self.assertEqual(
            [
                point["N"]
                for point in differences[
                    ("mps", "continuous-snake", 16)
                ]
            ],
            list(range(3, 11)),
        )
        exact_differences = plotter.energy_difference_series(
            rows, reference=("exact", "compact", None)
        )
        self.assertEqual(len(exact_differences), 12)
        self.assertTrue(
            all(
                [point["N"] for point in series] == list(range(3, 11))
                for series in exact_differences.values()
            )
        )

    def test_plotter_skips_missing_or_failed_jobs(self):
        plotter = _load_script("plot_ising_Nx3x3.py")
        with self._temporary_directory() as tmp_path:
            result = tmp_path / "letta_D1" / "N03.json"
            result.parent.mkdir(parents=True)
            result.write_text(
                json.dumps(
                    {
                        "solver": "letta",
                        "bond_dimension": 1,
                        "N": 3,
                        "lattice_shape": [3, 3, 3],
                        "nsites": 27,
                        "energy": -30.0,
                        "runtime_seconds": 1.0,
                        "sweeps": 20,
                        "ordering": "compact",
                        "energy_history": [
                            {"iteration": 1, "energy": -30.0}
                        ],
                    }
                )
            )
            stale = tmp_path / "mps_compact_D1" / "N03.json"
            stale.parent.mkdir(parents=True)
            stale.write_text(
                json.dumps(
                    {
                        "solver": "mps",
                        "bond_dimension": 1,
                        "N": 3,
                        "lattice_shape": [3, 3, 3],
                        "nsites": 27,
                        "energy": -31.0,
                        "runtime_seconds": 2.0,
                        "sweeps": 50,
                        "ordering": "compact",
                        "energy_history": [
                            {"iteration": 1, "energy": -31.0}
                        ],
                    }
                )
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rows = plotter.collect_results(
                    tmp_path, min_n=3, max_n=10
                )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["N"], 3)

    def test_slurm_launcher_uses_requested_arrays_memory_and_dependency(self):
        launcher = (SCRIPT_ROOT / "submit_ising_Nx3x3.sh").read_text()

        self.assertIn(
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg", launcher
        )
        self.assertIn(
            "/storage/gubingLab/hushuoyi/letta/letta_3D/Jul24th_TFI_n33",
            launcher,
        )
        self.assertIn("SWEEPS=${SWEEPS:-20}", launcher)
        self.assertIn("--array=0-15", launcher)
        self.assertIn("LETTA_DIMS=(1 2)", launcher)
        self.assertNotIn("LETTA_DIMS=(1 2 3 4)", launcher)
        self.assertIn("--array=0-79", launcher)
        self.assertIn(
            "MPS_ORDERINGS=(compact continuous-snake)",
            launcher,
        )
        self.assertIn('--ordering "${ORDERING}"', launcher)
        self.assertIn("--array=0-7", launcher)
        self.assertEqual(launcher.count("--mem=128G"), 3)
        self.assertIn('DEPENDENCY="afterany:', launcher)
        self.assertIn('--dependency="${DEPENDENCY}"', launcher)
        self.assertIn("plot_ising_Nx3x3.py", launcher)

    @staticmethod
    def _temporary_directory():
        import tempfile

        class TemporaryDirectoryPath(tempfile.TemporaryDirectory):
            def __enter__(self):
                return Path(super().__enter__())

        return TemporaryDirectoryPath()


if __name__ == "__main__":
    unittest.main()
