import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "letta_3D"
    / "Jul24th_TFI"
)


def _load_script(name):
    path = SCRIPT_ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IsingBenchmarkScriptTests(unittest.TestCase):
    def test_runner_dispatches_one_solver_and_saves_json(self):
        cases = [
            ("letta", 1, "letta_energy", "letta_runtime_seconds"),
            ("mps", 4, "mps_energy", "mps_runtime_seconds"),
            ("exact", None, "exact_energy", "exact_runtime_seconds"),
        ]
        for solver, bond_dim, energy_field, runtime_field in cases:
            with self.subTest(solver=solver), self._temporary_directory() as tmp_path:
                runner = _load_script("run_ising_2x2xN.py")
                calls = []

                def fake_compare(**kwargs):
                    calls.append(kwargs)
                    values = {
                        "letta_energy": None,
                        "mps_energy": None,
                        "exact_energy": None,
                        "letta_runtime_seconds": None,
                        "mps_runtime_seconds": None,
                        "exact_runtime_seconds": None,
                        "letta_converged": None,
                        "mps_converged": None,
                    }
                    values[energy_field] = -12.5
                    values[runtime_field] = 3.25
                    if solver != "exact":
                        values[f"{solver}_converged"] = True
                    return SimpleNamespace(
                        lattice_shape=(3, 2, 2),
                        ordering="compact",
                        environment_granularity="site",
                        nsites=12,
                        nbonds=20,
                        mpo_bond_dimension=6,
                        **values,
                    )

                with patch.object(runner, "compare_3d_ising", fake_compare):
                    output = runner.run_case(
                        solver=solver,
                        n_layers=3,
                        bond_dim=bond_dim,
                        output_root=tmp_path,
                        sweeps=7,
                        force=True,
                    )

                self.assertEqual(calls[0]["lattice_shape"], (3, 2, 2))
                self.assertIs(calls[0]["run_letta"], solver == "letta")
                self.assertIs(calls[0]["run_mps"], solver == "mps")
                self.assertIs(calls[0]["run_exact"], solver == "exact")
                self.assertEqual(calls[0]["letta_sweeps"], 7)
                self.assertEqual(calls[0]["mps_sweeps"], 7)

                payload = json.loads(output.read_text())
                self.assertEqual(payload["solver"], solver)
                self.assertEqual(payload["N"], 3)
                self.assertEqual(payload["energy"], -12.5)
                self.assertEqual(payload["runtime_seconds"], 3.25)
                self.assertEqual(payload["bond_dimension"], bond_dim)
                self.assertEqual(payload["lattice_shape"], [3, 2, 2])

    def test_runner_rejects_exact_diagonalization_above_twenty_four_sites(self):
        runner = _load_script("run_ising_2x2xN.py")
        with self._temporary_directory() as tmp_path:
            with self.assertRaisesRegex(ValueError, "N <= 6"):
                runner.run_case(
                    solver="exact",
                    n_layers=7,
                    bond_dim=None,
                    output_root=tmp_path,
                )

    def test_matrix_free_tfim_action_matches_dense_reference(self):
        runner = _load_script("run_ising_2x2xN.py")
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

    def test_runner_uses_matrix_free_exact_diagonalization_for_n6(self):
        runner = _load_script("run_ising_2x2xN.py")
        with self._temporary_directory() as tmp_path:
            with patch.object(
                runner,
                "_matrix_free_exact_ground_state",
                return_value=(-31.25, 12.5),
            ) as exact_solver:
                output = runner.run_case(
                    solver="exact",
                    n_layers=6,
                    bond_dim=None,
                    output_root=tmp_path,
                    force=True,
                )

            exact_solver.assert_called_once()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["lattice_shape"], [6, 2, 2])
            self.assertEqual(payload["energy"], -31.25)
            self.assertEqual(payload["runtime_seconds"], 12.5)
            self.assertEqual(payload["exact_backend"], "matrix-free Lanczos")

    def test_runner_does_not_reuse_results_from_old_axis_order(self):
        runner = _load_script("run_ising_2x2xN.py")
        with self._temporary_directory() as tmp_path:
            result_path = tmp_path / "mps_D1" / "N03.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "solver": "mps",
                        "bond_dimension": 1,
                        "N": 3,
                        "lattice_shape": [2, 2, 3],
                        "coupling": 1.0,
                        "field": 1.5,
                        "sweeps": 50,
                        "tolerance": 1.0e-8,
                        "seed": 4,
                    }
                )
            )

            self.assertFalse(
                runner._existing_result_matches(
                    result_path,
                    solver="mps",
                    bond_dim=1,
                    n_layers=3,
                    coupling=1.0,
                    field=1.5,
                    sweeps=50,
                    tolerance=1.0e-8,
                    seed=4,
                )
            )

    def test_plotter_collects_five_approximate_series_and_exact_points(self):
        plotter = _load_script("plot_ising_2x2xN.py")
        with self._temporary_directory() as tmp_path:
            results = tmp_path / "results"

            for solver, bond_dim in [
                ("letta", 1),
                ("mps", 1),
                ("mps", 2),
                ("mps", 4),
                ("mps", 8),
            ]:
                method = f"{solver}_D{bond_dim}"
                method_dir = results / method
                method_dir.mkdir(parents=True)
                for n_layers in range(2, 11):
                    (method_dir / f"N{n_layers:02d}.json").write_text(
                        json.dumps(
                            {
                                "solver": solver,
                                "bond_dimension": bond_dim,
                                "N": n_layers,
                                "lattice_shape": [n_layers, 2, 2],
                                "nsites": 4 * n_layers,
                                "energy": -float(n_layers),
                                "runtime_seconds": float(n_layers),
                                "converged": True,
                            }
                        )
                    )

            exact_dir = results / "exact"
            exact_dir.mkdir()
            for n_layers in range(2, 7):
                (exact_dir / f"N{n_layers:02d}.json").write_text(
                    json.dumps(
                        {
                            "solver": "exact",
                            "bond_dimension": None,
                            "N": n_layers,
                            "lattice_shape": [n_layers, 2, 2],
                            "nsites": 4 * n_layers,
                            "energy": -float(n_layers) - 0.1,
                            "runtime_seconds": float(n_layers) * 2,
                            "converged": None,
                        }
                    )
                )

            rows = plotter.collect_results(
                results, min_n=2, max_n=10, exact_max_n=6
            )

        self.assertEqual(len(rows), 50)
        self.assertEqual(
            {(row["solver"], row["bond_dimension"]) for row in rows},
            {
                ("letta", 1),
                ("mps", 1),
                ("mps", 2),
                ("mps", 4),
                ("mps", 8),
                ("exact", None),
            },
        )

        exact_differences = plotter.energy_difference_series(
            rows, reference=("exact", None)
        )
        self.assertEqual(
            set(exact_differences),
            {
                ("letta", 1),
                ("mps", 1),
                ("mps", 2),
                ("mps", 4),
                ("mps", 8),
            },
        )
        self.assertTrue(
            all(
                [point["N"] for point in series] == list(range(2, 7))
                for series in exact_differences.values()
            )
        )
        self.assertTrue(
            all(
                abs(point["energy_difference"] - 0.1) < 1.0e-12
                for series in exact_differences.values()
                for point in series
            )
        )

    def test_plotter_rejects_results_from_old_axis_order(self):
        plotter = _load_script("plot_ising_2x2xN.py")
        with self._temporary_directory() as tmp_path:
            result = tmp_path / "N03.json"
            result.write_text(
                json.dumps(
                    {
                        "solver": "letta",
                        "bond_dimension": 1,
                        "N": 3,
                        "lattice_shape": [2, 2, 3],
                        "nsites": 12,
                        "energy": -10.0,
                        "runtime_seconds": 1.0,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "lattice_shape"):
                plotter._load_result(result, "letta", 1, 3)

    def test_slurm_launcher_uses_bg_checkout_and_plot_dependency(self):
        launcher = (SCRIPT_ROOT / "submit_ising_2x2xN.sh").read_text()

        self.assertIn(
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg", launcher
        )
        self.assertIn("--array=0-8", launcher)
        self.assertIn("--array=0-35", launcher)
        self.assertIn("--array=0-4", launcher)
        self.assertIn('DEPENDENCY="afterany:', launcher)
        self.assertIn('--dependency="${DEPENDENCY}"', launcher)
        self.assertIn("plot_ising_2x2xN.py", launcher)
        self.assertIn("--exact-max-N 6", launcher)

    @staticmethod
    def _temporary_directory():
        import tempfile

        class TemporaryDirectoryPath(tempfile.TemporaryDirectory):
            def __enter__(self):
                return Path(super().__enter__())

        return TemporaryDirectoryPath()


if __name__ == "__main__":
    unittest.main()
