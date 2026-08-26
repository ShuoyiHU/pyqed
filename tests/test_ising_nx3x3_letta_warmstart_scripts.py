import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyqed._letta_one_site_opt import LatticeLETTA, identity_mpo
from pyqed._letta_one_site_opt._letta_for_3d import ordered_coordinates


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "letta_3D"
    / "Jul24th_TFI_n33"
)
RUNNER_PATH = SCRIPT_ROOT / "run_ising_Nx3x3_letta_warmstart.py"
LAUNCHER_PATH = SCRIPT_ROOT / "submit_ising_Nx3x3_letta_warmstart.sh"
CONTINUATION_RUNNER_PATH = (
    SCRIPT_ROOT / "run_ising_Nx3x3_letta_D3_D4_warmstart.py"
)
CONTINUATION_LAUNCHER_PATH = (
    SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D3_D4_warmstart.sh"
)
D8_RUNNER_PATH = SCRIPT_ROOT / "run_ising_Nx3x3_letta_D8_from_D4.py"
D8_LAUNCHER_PATH = (
    SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D8_from_D4.sh"
)
FIXED_NOISE_RUNNER_PATH = (
    SCRIPT_ROOT / "run_ising_Nx3x3_letta_D2_noise0p05_qr.py"
)
FIXED_NOISE_LAUNCHER_PATH = (
    SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D2_noise0p05_qr.sh"
)
SMALL_NOISE_RUNNER_PATH = (
    SCRIPT_ROOT / "run_ising_Nx3x3_letta_D2_noise0p005_qr.py"
)
SMALL_NOISE_LAUNCHER_PATH = (
    SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D2_noise0p005_qr.sh"
)
D2_RESTART_RUNNER_PATH = (
    SCRIPT_ROOT / "run_ising_Nx3x3_letta_D2_restart_noise0p001_qr.py"
)
D2_RESTART_LAUNCHER_PATH = (
    SCRIPT_ROOT / "submit_ising_Nx3x3_letta_D2_restart_noise0p001_qr.sh"
)
KICK14_RUNNER_PATH = (
    SCRIPT_ROOT
    / "run_ising_Nx3x3_letta_D2_restart_noise0p001_kick14.py"
)
KICK14_LAUNCHER_PATH = (
    SCRIPT_ROOT
    / "submit_ising_Nx3x3_letta_D2_restart_noise0p001_kick14.sh"
)


def _load_runner():
    if not RUNNER_PATH.exists():
        raise AssertionError(f"missing warm-start runner: {RUNNER_PATH}")
    specification = importlib.util.spec_from_file_location(
        RUNNER_PATH.stem,
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_continuation_runner():
    if not CONTINUATION_RUNNER_PATH.exists():
        raise AssertionError(
            f"missing D=3/D=4 continuation runner: "
            f"{CONTINUATION_RUNNER_PATH}"
        )
    specification = importlib.util.spec_from_file_location(
        CONTINUATION_RUNNER_PATH.stem,
        CONTINUATION_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_d8_runner():
    if not D8_RUNNER_PATH.exists():
        raise AssertionError(f"missing D=8 runner: {D8_RUNNER_PATH}")
    specification = importlib.util.spec_from_file_location(
        D8_RUNNER_PATH.stem,
        D8_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_fixed_noise_runner():
    if not FIXED_NOISE_RUNNER_PATH.exists():
        raise AssertionError(
            f"missing fixed-noise D=2 runner: {FIXED_NOISE_RUNNER_PATH}"
        )
    specification = importlib.util.spec_from_file_location(
        FIXED_NOISE_RUNNER_PATH.stem,
        FIXED_NOISE_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_small_noise_runner():
    if not SMALL_NOISE_RUNNER_PATH.exists():
        raise AssertionError(
            f"missing 0.005-noise D=2 runner: {SMALL_NOISE_RUNNER_PATH}"
        )
    specification = importlib.util.spec_from_file_location(
        SMALL_NOISE_RUNNER_PATH.stem,
        SMALL_NOISE_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_d2_restart_runner():
    if not D2_RESTART_RUNNER_PATH.exists():
        raise AssertionError(
            f"missing D=2 restart runner: {D2_RESTART_RUNNER_PATH}"
        )
    specification = importlib.util.spec_from_file_location(
        D2_RESTART_RUNNER_PATH.stem,
        D2_RESTART_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_kick14_runner():
    if not KICK14_RUNNER_PATH.exists():
        raise AssertionError(
            f"missing 14-kick D=2 runner: {KICK14_RUNNER_PATH}"
        )
    specification = importlib.util.spec_from_file_location(
        KICK14_RUNNER_PATH.stem,
        KICK14_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class IsingNx3x3LETTAWarmStartScriptTests(unittest.TestCase):
    def test_warmstart_runner_and_launcher_exist(self):
        self.assertTrue(RUNNER_PATH.is_file())
        self.assertTrue(LAUNCHER_PATH.is_file())

    def test_saved_letta_state_round_trips_without_pickle(self):
        runner = _load_runner()
        shape = (2, 2, 2)
        state = LatticeLETTA.random(
            shape,
            physical_dim=2,
            bond_dim=1,
            seed=7,
            coordinates=ordered_coordinates(shape, ordering="compact"),
        )
        metadata = {
            "stage": "D1",
            "N": 2,
            "coupling": 1.0,
            "field": 1.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            runner._save_letta_state(path, state, metadata)
            restored, restored_metadata = runner._load_letta_state(path)

        self.assertEqual(restored.lattice_shape, state.lattice_shape)
        self.assertEqual(restored.coordinates, state.coordinates)
        self.assertEqual(restored.physical_dim, state.physical_dim)
        self.assertEqual(restored_metadata, metadata)
        self.assertEqual(len(restored.tensors), len(state.tensors))
        for actual, expected in zip(restored.tensors, state.tensors):
            np.testing.assert_allclose(actual, expected, atol=1.0e-13)

    def test_d1_state_is_expanded_with_a_full_rank_d2_kick(self):
        runner = _load_runner()
        shape = (2, 2, 2)
        state = LatticeLETTA.random(
            shape,
            physical_dim=2,
            bond_dim=1,
            seed=4,
            coordinates=ordered_coordinates(shape, ordering="compact"),
        )

        expanded, ranks = runner._expand_with_kick(
            state,
            target_bond_dim=2,
            kick=1.0e-3,
            seed=9,
        )

        self.assertEqual(expanded.bond_dimensions, (2,) * (state.nsites - 1))
        self.assertEqual(ranks, (2,) * (state.nsites - 1))
        self.assertTrue(
            all(tensor.shape[0] == 2 for tensor in expanded.tensors[1:])
        )
        self.assertTrue(
            all(tensor.shape[-1] == 2 for tensor in expanded.tensors[:-1])
        )

    def test_zero_kick_is_rejected_for_d1_to_d2_expansion(self):
        runner = _load_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )
        with self.assertRaisesRegex(ValueError, "kick must be positive"):
            runner._expand_with_kick(
                state,
                target_bond_dim=2,
                kick=0.0,
                seed=9,
            )

    def test_prepared_warm_state_reports_the_kick_seed_it_used(self):
        runner = _load_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )
        hamiltonian = identity_mpo(
            state.nsites,
            physical_dim=2,
            lattice_shape=state.lattice_shape,
        )

        prepared = runner._prepare_warm_state(
            state,
            hamiltonian,
            parent_energy=1.0,
            requested_kick=1.0e-3,
            seed=9,
            max_energy_increase_per_site=1.0e-12,
        )
        self.assertEqual(len(prepared), 5)
        _state, _energy, used_kick, ranks, kick_seed = prepared

        self.assertEqual(used_kick, 1.0e-3)
        self.assertEqual(ranks, (2,) * (state.nsites - 1))
        self.assertEqual(kick_seed, 1009)

    def test_warmstart_outputs_do_not_overwrite_random_start_results(self):
        runner = _load_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._case_paths(result_root, state_root, n_layers=5)

        self.assertEqual(
            paths["d1_result"],
            result_root / "D1" / "N05.json",
        )
        self.assertEqual(
            paths["d2_result"],
            result_root / "D2_from_D1" / "N05.json",
        )
        self.assertEqual(
            paths["d1_state"],
            state_root / "D1" / "N05.npz",
        )
        self.assertEqual(
            paths["d2_state"],
            state_root / "D2_from_D1" / "N05.npz",
        )

    def test_launcher_runs_parallel_n_tasks_with_cluster_pyqed_branch(self):
        self.assertTrue(LAUNCHER_PATH.is_file())
        launcher = LAUNCHER_PATH.read_text()

        self.assertIn("--array=0-7", launcher)
        self.assertIn("--cpus-per-task=16", launcher)
        self.assertIn("--mem=128G", launcher)
        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn('N_LAYERS=$((TASK_ID + 3))', launcher)
        self.assertIn("run_ising_Nx3x3_letta_warmstart.py", launcher)
        self.assertIn('--d1-sweeps "${D1_SWEEPS}"', launcher)
        self.assertIn('--d2-sweeps "${D2_SWEEPS}"', launcher)
        self.assertIn('--kick "${KICK}"', launcher)
        self.assertIn(
            '--output-root "${RUN_ROOT}/results/letta_warmstart"',
            launcher,
        )
        self.assertIn(
            '--state-root "${RUN_ROOT}/states/letta_warmstart"',
            launcher,
        )

    def test_d3_d4_continuation_runner_and_launcher_exist(self):
        self.assertTrue(CONTINUATION_RUNNER_PATH.is_file())
        self.assertTrue(CONTINUATION_LAUNCHER_PATH.is_file())

    def test_continuation_outputs_are_separate_from_d1_and_d2(self):
        runner = _load_continuation_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._continuation_paths(
            result_root,
            state_root,
            n_layers=7,
        )

        self.assertEqual(
            paths["d2_result"],
            result_root / "D2_from_D1" / "N07.json",
        )
        self.assertEqual(
            paths["d2_state"],
            state_root / "D2_from_D1" / "N07.npz",
        )
        self.assertEqual(
            paths["d3_result"],
            result_root / "D3_from_D2" / "N07.json",
        )
        self.assertEqual(
            paths["d3_state"],
            state_root / "D3_from_D2" / "N07.npz",
        )
        self.assertEqual(
            paths["d4_result"],
            result_root / "D4_from_D3" / "N07.json",
        )
        self.assertEqual(
            paths["d4_state"],
            state_root / "D4_from_D3" / "N07.npz",
        )

    def test_continuation_kicks_activate_d3_then_d4_sectors(self):
        runner = _load_continuation_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=2,
            seed=4,
        )

        d3_state, d3_ranks = runner._expand_with_kick(
            state,
            target_bond_dim=3,
            kick=1.0e-3,
            seed=3004,
        )
        d4_state, d4_ranks = runner._expand_with_kick(
            d3_state,
            target_bond_dim=4,
            kick=1.0e-3,
            seed=4004,
        )

        self.assertEqual(d3_state.bond_dimensions, (3,) * 7)
        self.assertEqual(
            d3_ranks,
            runner._maximum_virtual_bond_ranks(d3_state, 3),
        )
        self.assertEqual(d4_state.bond_dimensions, (4,) * 7)
        self.assertEqual(
            d4_ranks,
            runner._maximum_virtual_bond_ranks(d4_state, 4),
        )

    def test_continuation_rejects_a_zero_kick(self):
        runner = _load_continuation_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=2,
            seed=4,
        )
        with self.assertRaisesRegex(ValueError, "kick must be positive"):
            runner._expand_with_kick(
                state,
                target_bond_dim=3,
                kick=0.0,
                seed=3004,
            )

    def test_continuation_launcher_uses_matching_parent_array_tasks(self):
        self.assertTrue(CONTINUATION_LAUNCHER_PATH.is_file())
        launcher = CONTINUATION_LAUNCHER_PATH.read_text()

        self.assertIn(
            "PARENT_ARRAY_JOB_ID=${PARENT_ARRAY_JOB_ID:-11657856}",
            launcher,
        )
        self.assertIn(
            "--dependency=aftercorr:${PARENT_ARRAY_JOB_ID}",
            launcher,
        )
        self.assertIn("--array=0-7", launcher)
        self.assertIn("--cpus-per-task=16", launcher)
        self.assertIn("--mem=128G", launcher)
        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn('N_LAYERS=$((TASK_ID + 3))', launcher)
        self.assertIn(
            "run_ising_Nx3x3_letta_D3_D4_warmstart.py",
            launcher,
        )
        self.assertIn('--d3-sweeps "${D3_SWEEPS}"', launcher)
        self.assertIn('--d4-sweeps "${D4_SWEEPS}"', launcher)
        self.assertIn('--d3-kick "${D3_KICK}"', launcher)
        self.assertIn('--d4-kick "${D4_KICK}"', launcher)

    def test_d8_runner_and_launcher_exist(self):
        self.assertTrue(D8_RUNNER_PATH.is_file())
        self.assertTrue(D8_LAUNCHER_PATH.is_file())

    def test_d8_outputs_are_separate_and_use_d4_as_parent(self):
        runner = _load_d8_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._d8_paths(
            result_root,
            state_root,
            n_layers=6,
        )

        self.assertEqual(
            paths["d4_result"],
            result_root / "D4_from_D3" / "N06.json",
        )
        self.assertEqual(
            paths["d4_state"],
            state_root / "D4_from_D3" / "N06.npz",
        )
        self.assertEqual(
            paths["d8_result"],
            result_root / "D8_from_D4" / "N06.json",
        )
        self.assertEqual(
            paths["d8_state"],
            state_root / "D8_from_D4" / "N06.npz",
        )

    def test_d4_state_is_kicked_directly_into_active_d8_sectors(self):
        runner = _load_d8_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=4,
            seed=4,
        )

        expanded, ranks = runner._expand_with_kick(
            state,
            target_bond_dim=8,
            kick=1.0e-3,
            seed=8004,
        )

        self.assertEqual(expanded.bond_dimensions, (8,) * 7)
        self.assertEqual(
            ranks,
            runner._maximum_virtual_bond_ranks(expanded, 8),
        )

    def test_parent_target_dimension_allows_exact_boundary_reduction(self):
        runner = _load_continuation_runner()
        original = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=4,
            seed=4,
        )
        tensors = [tensor.copy() for tensor in original.tensors]
        tensors[-2] = tensors[-2][..., :2]
        tensors[-1] = tensors[-1][:2, ...]
        reduced = LatticeLETTA(
            original.lattice_shape,
            original.physical_dim,
            tensors,
            coordinates=original.coordinates,
        )

        self.assertEqual(
            runner._parent_bond_dimension(
                reduced,
                {"bond_dimension": 4},
            ),
            4,
        )

    def test_d8_parent_loader_accepts_physical_boundary_reduction(self):
        runner = _load_d8_runner()
        original = LatticeLETTA.random(
            (3, 3, 3),
            physical_dim=2,
            bond_dim=4,
            seed=4,
            coordinates=ordered_coordinates(
                (3, 3, 3),
                ordering="compact",
            ),
        )
        tensors = [tensor.copy() for tensor in original.tensors]
        tensors[-2] = tensors[-2][..., :2]
        tensors[-1] = tensors[-1][:2, ...]
        reduced = LatticeLETTA(
            original.lattice_shape,
            original.physical_dim,
            tensors,
            coordinates=original.coordinates,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = runner._d8_paths(
                root / "results",
                root / "states",
                n_layers=3,
            )
            metadata = {
                "solver": "letta",
                "bond_dimension": 4,
                "N": 3,
                "lattice_shape": [3, 3, 3],
                "coupling": 1.0,
                "field": 1.5,
                "ordering": "compact",
            }
            runner._COMMON._save_letta_state(
                paths["d4_state"],
                reduced,
                metadata,
            )
            payload = {
                **metadata,
                "energy": -61.0,
                "state_sha256": runner._COMMON._sha256(
                    paths["d4_state"]
                ),
            }
            runner._COMMON._atomic_write_json(
                paths["d4_result"],
                payload,
            )

            loaded, loaded_payload, _sha256 = runner._load_parent_d4(
                paths,
                n_layers=3,
                coupling=1.0,
                field=1.5,
            )

        self.assertEqual(max(loaded.bond_dimensions), 4)
        self.assertEqual(loaded.bond_dimensions[-1], 2)
        self.assertEqual(loaded_payload["bond_dimension"], 4)

    def test_d8_launcher_has_no_job_dependency(self):
        self.assertTrue(D8_LAUNCHER_PATH.is_file())
        launcher = D8_LAUNCHER_PATH.read_text()

        self.assertNotIn("--dependency", launcher)
        self.assertNotIn("PARENT_ARRAY_JOB_ID", launcher)
        self.assertIn("--array=0-7", launcher)
        self.assertIn("--cpus-per-task=16", launcher)
        self.assertIn("--mem=128G", launcher)
        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn('N_LAYERS=$((TASK_ID + 3))', launcher)
        self.assertIn(
            "run_ising_Nx3x3_letta_D8_from_D4.py",
            launcher,
        )
        self.assertIn('--d8-sweeps "${D8_SWEEPS}"', launcher)
        self.assertIn('--kick "${KICK}"', launcher)

    def test_fixed_noise_d2_runner_and_launcher_exist(self):
        self.assertTrue(FIXED_NOISE_RUNNER_PATH.is_file())
        self.assertTrue(FIXED_NOISE_LAUNCHER_PATH.is_file())

    def test_fixed_noise_outputs_do_not_overwrite_other_d2_runs(self):
        runner = _load_fixed_noise_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._fixed_noise_paths(
            result_root,
            state_root,
            n_layers=5,
        )

        self.assertEqual(
            paths["d1_result"],
            result_root / "D1" / "N05.json",
        )
        self.assertEqual(
            paths["d1_state"],
            state_root / "D1" / "N05.npz",
        )
        self.assertEqual(
            paths["d2_result"],
            result_root / "D2_from_D1_noise0p05_qr" / "N05.json",
        )
        self.assertEqual(
            paths["d2_state"],
            state_root / "D2_from_D1_noise0p05_qr" / "N05.npz",
        )

    def test_fixed_noise_expansion_uses_exactly_point_zero_five(self):
        runner = _load_fixed_noise_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )

        expanded, ranks, kick_seed = runner._expand_fixed_noise(
            state,
            noise=0.05,
            seed=4,
        )
        expected = state.expand_bond_dimension(
            2,
            noise=0.05,
            seed=1004,
        )

        self.assertEqual(kick_seed, 1004)
        self.assertEqual(ranks, (2,) * 7)
        for actual_tensor, expected_tensor in zip(
            expanded.tensors,
            expected.tensors,
        ):
            np.testing.assert_allclose(
                actual_tensor,
                expected_tensor,
                atol=1.0e-13,
            )

    def test_fixed_noise_variant_rejects_a_different_noise_value(self):
        runner = _load_fixed_noise_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )
        with self.assertRaisesRegex(ValueError, "fixed at 0.05"):
            runner._expand_fixed_noise(
                state,
                noise=0.5,
                seed=4,
            )

    def test_right_qr_gauge_preserves_state_and_fixes_initial_center(self):
        runner = _load_fixed_noise_runner()
        parent = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )
        expanded, _ranks, _kick_seed = runner._expand_fixed_noise(
            parent,
            noise=0.05,
            seed=4,
        )
        state_before = expanded.state_vector()

        gauged, residuals = runner._right_qr_gauge(expanded)

        np.testing.assert_allclose(
            gauged.state_vector(),
            state_before,
            atol=1.0e-12,
        )
        self.assertEqual(len(residuals), gauged.nsites - 1)
        self.assertLess(max(residuals), 1.0e-12)
        self.assertEqual(gauged.bond_dimensions, (2,) * 7)

    def test_fixed_noise_launcher_is_dependency_free_and_explicit(self):
        self.assertTrue(FIXED_NOISE_LAUNCHER_PATH.is_file())
        launcher = FIXED_NOISE_LAUNCHER_PATH.read_text()

        self.assertNotIn("--dependency", launcher)
        self.assertIn("KICK=${KICK:-0.05}", launcher)
        self.assertIn("D2_SWEEPS=${D2_SWEEPS:-30}", launcher)
        self.assertIn("--array=0-7", launcher)
        self.assertIn("--cpus-per-task=16", launcher)
        self.assertIn("--mem=128G", launcher)
        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn('N_LAYERS=$((TASK_ID + 3))', launcher)
        self.assertIn(
            "run_ising_Nx3x3_letta_D2_noise0p05_qr.py",
            launcher,
        )
        self.assertIn('--d2-sweeps "${D2_SWEEPS}"', launcher)
        self.assertIn('--noise "${KICK}"', launcher)

    def test_point_zero_zero_five_runner_uses_separate_outputs(self):
        runner = _load_small_noise_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._fixed_noise_paths(
            result_root,
            state_root,
            n_layers=5,
        )

        self.assertEqual(
            paths["d2_result"],
            result_root / "D2_from_D1_noise0p005_qr" / "N05.json",
        )
        self.assertEqual(
            paths["d2_state"],
            state_root / "D2_from_D1_noise0p005_qr" / "N05.npz",
        )

    def test_point_zero_zero_five_expansion_is_exact_and_keeps_d2(self):
        runner = _load_small_noise_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=1,
            seed=4,
        )

        expanded, ranks, kick_seed = runner._expand_fixed_noise(
            state,
            noise=0.005,
            seed=4,
        )
        expected = state.expand_bond_dimension(
            2,
            noise=0.005,
            seed=1004,
        )

        self.assertEqual(kick_seed, 1004)
        self.assertEqual(ranks, (2,) * 7)
        for actual_tensor, expected_tensor in zip(
            expanded.tensors,
            expected.tensors,
        ):
            np.testing.assert_allclose(
                actual_tensor,
                expected_tensor,
                atol=1.0e-13,
            )
        with self.assertRaisesRegex(ValueError, "fixed at 0.005"):
            runner._expand_fixed_noise(state, noise=0.05, seed=4)

    def test_point_zero_zero_five_launcher_defaults_to_thirty_sweeps(self):
        self.assertTrue(SMALL_NOISE_LAUNCHER_PATH.is_file())
        launcher = SMALL_NOISE_LAUNCHER_PATH.read_text()

        self.assertNotIn("--dependency", launcher)
        self.assertIn("KICK=${KICK:-0.005}", launcher)
        self.assertIn("D2_SWEEPS=${D2_SWEEPS:-30}", launcher)
        self.assertIn("--array=0-7", launcher)
        self.assertIn(
            "run_ising_Nx3x3_letta_D2_noise0p005_qr.py",
            launcher,
        )

    def test_d2_restart_uses_original_d2_parent_and_separate_outputs(self):
        runner = _load_d2_restart_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._restart_paths(
            result_root,
            state_root,
            n_layers=3,
        )

        self.assertEqual(
            paths["parent_result"],
            result_root / "D2_from_D1" / "N03.json",
        )
        self.assertEqual(
            paths["parent_state"],
            state_root / "D2_from_D1" / "N03.npz",
        )
        self.assertEqual(
            paths["restart_result"],
            result_root
            / "D2_restart_from_D2_noise0p001_qr"
            / "N03.json",
        )
        self.assertEqual(
            paths["restart_state"],
            state_root
            / "D2_restart_from_D2_noise0p001_qr"
            / "N03.npz",
        )

    def test_d2_restart_kick_changes_only_canonical_center_tensor(self):
        runner = _load_d2_restart_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=2,
            seed=8,
        )
        canonical, _ = runner._right_qr_gauge(state)
        kicked, kick_seed = runner._kick_same_dimension(
            canonical,
            noise=0.001,
            seed=4,
        )

        rng = np.random.default_rng(3004)
        self.assertEqual(kick_seed, 3004)
        self.assertEqual(
            kicked.bond_dimensions,
            canonical.bond_dimensions,
        )
        center = canonical.tensors[0]
        scale = 0.001 * np.linalg.norm(center) / np.sqrt(center.size)
        perturbation = rng.normal(size=center.shape)
        if np.issubdtype(center.dtype, np.complexfloating):
            perturbation = (
                perturbation
                + 1j * rng.normal(size=center.shape)
            ) / np.sqrt(2.0)
        np.testing.assert_allclose(
            kicked.tensors[0],
            center + scale * perturbation,
            atol=1.0e-13,
        )
        for canonical_tensor, kicked_tensor in zip(
            canonical.tensors[1:],
            kicked.tensors[1:],
        ):
            np.testing.assert_array_equal(
                kicked_tensor,
                canonical_tensor,
            )

        with self.assertRaisesRegex(ValueError, "fixed at 0.001"):
            runner._kick_same_dimension(
                canonical,
                noise=0.005,
                seed=4,
            )

    def test_d2_restart_qr_preserves_state_and_center_kick_keeps_environment(self):
        runner = _load_d2_restart_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=2,
            seed=8,
        )
        vector_before = state.state_vector()

        canonical, residuals = runner._right_qr_gauge(state)
        kicked, _ = runner._kick_same_dimension(
            canonical,
            noise=0.001,
            seed=4,
        )

        np.testing.assert_allclose(
            canonical.state_vector(),
            vector_before,
            atol=1.0e-12,
        )
        self.assertEqual(
            kicked.bond_dimensions,
            state.bond_dimensions,
        )
        self.assertLess(max(residuals), 1.0e-12)
        self.assertLess(
            max(runner._right_gauge_residuals(kicked)),
            1.0e-12,
        )

    def test_d2_restart_launcher_runs_n3_for_thirty_additional_sweeps(self):
        self.assertTrue(D2_RESTART_LAUNCHER_PATH.is_file())
        launcher = D2_RESTART_LAUNCHER_PATH.read_text()

        self.assertNotIn("--array", launcher)
        self.assertNotIn("--dependency", launcher)
        self.assertIn("N_LAYERS=${N_LAYERS:-3}", launcher)
        self.assertIn("ADDITIONAL_SWEEPS=${ADDITIONAL_SWEEPS:-30}", launcher)
        self.assertIn("KICK=${KICK:-0.001}", launcher)
        self.assertIn(
            "PYQED_REPO=${PYQED_REPO:-"
            "/share/home/gubingLab/hushuoyi/software/pyqed_bg}",
            launcher,
        )
        self.assertIn(
            "run_ising_Nx3x3_letta_D2_restart_noise0p001_qr.py",
            launcher,
        )

    def test_kick14_continues_from_one_kick_restart_without_overwrite(self):
        runner = _load_kick14_runner()
        result_root = Path("/tmp/results/letta_warmstart")
        state_root = Path("/tmp/states/letta_warmstart")

        paths = runner._case_paths(
            result_root,
            state_root,
            n_layers=3,
        )

        self.assertEqual(
            paths["parent_result"],
            result_root
            / "D2_restart_from_D2_noise0p001_qr"
            / "N03.json",
        )
        self.assertEqual(
            paths["parent_state"],
            state_root
            / "D2_restart_from_D2_noise0p001_qr"
            / "N03.npz",
        )
        self.assertEqual(
            paths["result"],
            result_root
            / "D2_restart_from_restart_noise0p001_kick14"
            / "N03.json",
        )
        self.assertEqual(
            paths["state"],
            state_root
            / "D2_restart_from_restart_noise0p001_kick14"
            / "N03.npz",
        )

    def test_kick14_plan_kicks_first_fourteen_of_exactly_thirty_sweeps(self):
        runner = _load_kick14_runner()

        plan = runner._sweep_plan(
            total_sweeps=30,
            kicked_sweeps=14,
            seed=4,
        )

        self.assertEqual(len(plan), 30)
        self.assertEqual(
            [step["direction"] for step in plan[:4]],
            ["lr", "rl", "lr", "rl"],
        )
        self.assertEqual(
            [step["center_site"] for step in plan[:4]],
            ["first", "last", "first", "last"],
        )
        self.assertEqual(
            [step["kick_seed"] for step in plan[:14]],
            list(range(4005, 4019)),
        )
        self.assertTrue(
            all(step["kick_seed"] is None for step in plan[14:])
        )

    def test_kick14_left_qr_and_last_center_kick_preserve_d2(self):
        runner = _load_kick14_runner()
        state = LatticeLETTA.random(
            (2, 2, 2),
            physical_dim=2,
            bond_dim=2,
            seed=8,
        )
        vector_before = state.state_vector()

        canonical, residuals = runner._left_qr_gauge(state)
        kicked = runner._kick_center_tensor(
            canonical,
            center_site=canonical.nsites - 1,
            noise=0.001,
            kick_seed=4006,
        )

        np.testing.assert_allclose(
            canonical.state_vector(),
            vector_before,
            atol=1.0e-12,
        )
        self.assertLess(max(residuals), 1.0e-12)
        self.assertEqual(kicked.bond_dimensions, state.bond_dimensions)
        for canonical_tensor, kicked_tensor in zip(
            canonical.tensors[:-1],
            kicked.tensors[:-1],
        ):
            np.testing.assert_array_equal(
                kicked_tensor,
                canonical_tensor,
            )
        self.assertFalse(
            np.array_equal(
                kicked.tensors[-1],
                canonical.tensors[-1],
            )
        )

    def test_kick14_launcher_defaults_are_explicit(self):
        self.assertTrue(KICK14_LAUNCHER_PATH.is_file())
        launcher = KICK14_LAUNCHER_PATH.read_text()

        self.assertNotIn("--array", launcher)
        self.assertNotIn("--dependency", launcher)
        self.assertIn("N_LAYERS=${N_LAYERS:-3}", launcher)
        self.assertIn("TOTAL_SWEEPS=${TOTAL_SWEEPS:-30}", launcher)
        self.assertIn("KICKED_SWEEPS=${KICKED_SWEEPS:-14}", launcher)
        self.assertIn("KICK=${KICK:-0.001}", launcher)
        self.assertIn(
            "run_ising_Nx3x3_letta_D2_restart_noise0p001_kick14.py",
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
