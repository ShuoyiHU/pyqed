from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyqed._letta_one_site_opt import LETTADMROptions, LatticeLETTA
from pyqed._letta_one_site_opt._letta_for_3d import (
    MPSDMRGOptions,
    SnakeMPS,
    letta_ground_state,
    mps_dmrg,
    nearest_neighbor_bonds,
    ordered_coordinates,
    snake_letta_state,
    transverse_field_ising_mpo,
)


# =============================================================================
# USER CONFIGURATION -- edit these values, then click Run
# =============================================================================

# The lattice is (N_x, 3, 3). Multiple N_x values may be supplied.
N_X_VALUES = (3,)

# Coupling scan. The transverse field is fixed to g=1, so J equals J/g.
J_OVER_G_VALUES = (2,)
FIELD_G = 1

# Warm-start dimensions must be strictly increasing and start at D=1.
LETTA_BOND_DIMS = (1,2)
MPS_BOND_DIMS = (1, 2)

# Select which solver families to run.
RUN_LETTA = True
RUN_MPS_TWO_SITE = True
RUN_MPS_ONE_SITE = False

# Use "compact"/"raster" or "snake"/"continuous-snake". Snake LETTA carries
# only physical indices belonging to later sites in the one-dimensional chain.
ORDERING = "raster"
MAX_SWEEPS = 20
ENERGY_DENSITY_TOLERANCE = 1.0e-8
EIGENSOLVER_TOLERANCE = 1.0e-10
EIGENSOLVER_MAX_ITERATIONS = 300
RANDOM_SEED = 4

# Noisy D -> larger-D embedding used on the cluster.
INITIAL_NOISE = 5.0e-4
NOISE_ROUNDS = 6
NOISE_DECAY = 0.5

# Existing valid results are resumed by default.
FORCE_RERUN = True

# Results are kept beside this example unless an absolute path is supplied.
OUTPUT_ROOT = Path(__file__).resolve().with_name("tfi_jg_scan_output")

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


RESULTS_ROOT = OUTPUT_ROOT / "results"
STATES_ROOT = OUTPUT_ROOT / "states"
SUMMARY_PATH = OUTPUT_ROOT / "energy_summary.csv"

POSITIVE_COORDINATE_CONVENTION = "positive-coordinate-neighbors"
LATER_CHAIN_CONVENTION = "later-in-chain-neighbors"
_LATER_CHAIN_STATE_CLASS = None


def _normalized_ordering() -> str:
    key = str(ORDERING).strip().lower().replace("_", "-")
    aliases = {
        "compact": "compact",
        "raster": "compact",
        "lexicographic": "compact",
        "c-order": "compact",
        "snake": "continuous-snake",
        "continuous": "continuous-snake",
        "continuous-snake": "continuous-snake",
    }
    try:
        return aliases[key]
    except KeyError as error:
        choices = "compact, raster, snake, or continuous-snake"
        raise ValueError(f"ORDERING must be {choices}; received {ORDERING!r}") from error


def _ordering_slug() -> str:
    return "snake" if _normalized_ordering() == "continuous-snake" else "compact"


def _physical_index_convention() -> str:
    if _normalized_ordering() == "continuous-snake":
        return LATER_CHAIN_CONVENTION
    return POSITIVE_COORDINATE_CONVENTION


def _letta_dataset(bond_dim: int) -> str:
    if _normalized_ordering() == "continuous-snake":
        # Do not reuse states produced by the old positive-coordinate convention.
        return f"letta_continuous_snake_later_chain_D{int(bond_dim)}"
    return f"letta_compact_D{int(bond_dim)}"


def _later_chain_neighborhoods(lattice_shape, coordinates):
    """Assign every geometric edge to its earlier endpoint in chain order."""

    shape = tuple(int(length) for length in lattice_shape)
    coordinates = tuple(tuple(int(value) for value in item) for item in coordinates)
    if len(shape) not in {2, 3}:
        raise ValueError("later-chain LETTA supports 2D or 3D lattices")
    if len(coordinates) != math.prod(shape):
        raise ValueError("coordinates do not cover the lattice")
    site_for = {coordinate: site for site, coordinate in enumerate(coordinates)}
    if len(site_for) != len(coordinates):
        raise ValueError("coordinates contain duplicates")

    neighborhoods = []
    for site, coordinate in enumerate(coordinates):
        later_neighbors = []
        for axis, axis_length in enumerate(shape):
            for displacement in (-1, 1):
                neighbor = list(coordinate)
                neighbor[axis] += displacement
                if not 0 <= neighbor[axis] < axis_length:
                    continue
                neighbor_site = site_for.get(tuple(neighbor))
                if neighbor_site is not None and neighbor_site > site:
                    later_neighbors.append(neighbor_site)
        neighborhoods.append((site,) + tuple(sorted(later_neighbors)))
    return tuple(neighborhoods)


def _later_chain_state_class():
    """Return a LatticeLETTA subtype that survives copying and D expansion."""

    global _LATER_CHAIN_STATE_CLASS
    if _LATER_CHAIN_STATE_CLASS is not None:
        return _LATER_CHAIN_STATE_CLASS

    class LaterChainLETTA(LatticeLETTA):
        physical_index_convention = LATER_CHAIN_CONVENTION

        def _build_neighborhood(self, coordinate):
            site = self._coordinate_to_site[coordinate]
            later_neighbors = []
            for axis, axis_length in enumerate(self.lattice_shape):
                for displacement in (-1, 1):
                    neighbor = list(coordinate)
                    neighbor[axis] += displacement
                    if not 0 <= neighbor[axis] < axis_length:
                        continue
                    neighbor_site = self._coordinate_to_site.get(tuple(neighbor))
                    if neighbor_site is not None and neighbor_site > site:
                        later_neighbors.append(neighbor_site)
            return (site,) + tuple(sorted(later_neighbors))

        @classmethod
        def random(
            cls,
            lattice_shape,
            *,
            physical_dim=2,
            bond_dim=2,
            seed=None,
            real=True,
            coordinates=None,
        ):
            shape = tuple(int(length) for length in lattice_shape)
            if coordinates is None:
                coordinates = tuple(np.ndindex(*shape))
            else:
                coordinates = tuple(tuple(item) for item in coordinates)
            neighborhoods = _later_chain_neighborhoods(shape, coordinates)
            rng = np.random.default_rng(seed)
            tensors = []
            nsites = len(coordinates)
            for site, neighborhood in enumerate(neighborhoods):
                left_dim = 1 if site == 0 else int(bond_dim)
                right_dim = 1 if site == nsites - 1 else int(bond_dim)
                tensor_shape = (
                    (left_dim,)
                    + (int(physical_dim),) * len(neighborhood)
                    + (right_dim,)
                )
                tensor = rng.normal(size=tensor_shape)
                if not real:
                    tensor = tensor + 1j * rng.normal(size=tensor_shape)
                tensor /= np.sqrt(tensor.size)
                tensors.append(tensor)
            return cls(
                shape,
                int(physical_dim),
                tensors,
                coordinates=coordinates,
            )

        def copy(self):
            return type(self)(
                self.lattice_shape,
                self.physical_dim,
                [tensor.copy() for tensor in self.tensors],
                coordinates=self.coordinates,
            )

        def expand_bond_dimension(self, bond_dim, *, noise=0.0, seed=None):
            bond_dim = int(bond_dim)
            noise = float(noise)
            if bond_dim <= 0:
                raise ValueError("bond_dim must be positive")
            if noise < 0.0:
                raise ValueError("noise must be nonnegative")
            if any(value > bond_dim for value in self.bond_dimensions):
                raise ValueError("bond_dim cannot shrink the state")

            rng = np.random.default_rng(seed)
            dtype = np.result_type(*self.tensors)
            expanded_tensors = []
            for site, tensor in enumerate(self.tensors):
                left_dim = 1 if site == 0 else bond_dim
                right_dim = 1 if site == self.nsites - 1 else bond_dim
                tensor_shape = (
                    (left_dim,) + tensor.shape[1:-1] + (right_dim,)
                )
                expanded = np.zeros(tensor_shape, dtype=dtype)
                old_block = (
                    (slice(0, tensor.shape[0]),)
                    + (slice(None),) * (tensor.ndim - 2)
                    + (slice(0, tensor.shape[-1]),)
                )
                expanded[old_block] = tensor
                if noise:
                    mask = np.ones(tensor_shape, dtype=bool)
                    mask[old_block] = False
                    scale = noise * np.linalg.norm(tensor) / np.sqrt(tensor.size)
                    perturbation = rng.normal(size=tensor_shape)
                    if np.issubdtype(dtype, np.complexfloating):
                        perturbation = (
                            perturbation + 1j * rng.normal(size=tensor_shape)
                        ) / np.sqrt(2.0)
                    expanded[mask] = scale * perturbation[mask]
                expanded_tensors.append(expanded)
            return type(self)(
                self.lattice_shape,
                self.physical_dim,
                expanded_tensors,
                coordinates=self.coordinates,
            )

    _LATER_CHAIN_STATE_CLASS = LaterChainLETTA
    return LaterChainLETTA


def _new_letta_state(shape, coordinates):
    if _normalized_ordering() == "continuous-snake":
        return _later_chain_state_class().random(
            shape,
            physical_dim=2,
            bond_dim=1,
            seed=RANDOM_SEED,
            coordinates=coordinates,
        )
    return snake_letta_state(
        shape,
        physical_dim=2,
        bond_dim=1,
        seed=RANDOM_SEED,
        ordering=_normalized_ordering(),
    )


def _validate_letta_state(state, *, shape, coordinates, bond_dim: int) -> None:
    if tuple(state.lattice_shape) != tuple(shape):
        raise ValueError("LETTA state has the wrong lattice shape")
    if tuple(state.coordinates) != tuple(coordinates):
        raise ValueError("LETTA state has the wrong chain ordering")
    if max(state.bond_dimensions, default=1) > int(bond_dim):
        raise ValueError(f"LETTA state exceeds D={int(bond_dim)}")

    if _normalized_ordering() != "continuous-snake":
        if (
            getattr(state, "physical_index_convention", None)
            == LATER_CHAIN_CONVENTION
        ):
            raise ValueError("compact LETTA state uses the snake physical-index convention")
        return

    if getattr(state, "physical_index_convention", None) != LATER_CHAIN_CONVENTION:
        raise ValueError("snake LETTA state does not use later-in-chain physical legs")
    expected = _later_chain_neighborhoods(shape, coordinates)
    actual = tuple(state.site_neighborhood(site) for site in range(state.nsites))
    if actual != expected:
        raise ValueError("snake LETTA state has incorrect later-chain neighborhoods")

    # This is the defining invariant: tensor i may carry s_j only when j > i.
    if any(
        neighbor <= site
        for site, group in enumerate(actual)
        for neighbor in group[1:]
    ):
        raise ValueError("snake LETTA state contains an earlier-in-chain carried index")


def _validate_configuration() -> None:
    _normalized_ordering()
    if FIELD_G != 1.0:
        raise ValueError("FIELD_G must remain 1.0 when scanning J/g directly")
    if not N_X_VALUES or any(int(value) <= 0 for value in N_X_VALUES):
        raise ValueError("N_X_VALUES must contain positive integers")
    if not J_OVER_G_VALUES or any(float(value) < 0.0 for value in J_OVER_G_VALUES):
        raise ValueError("J_OVER_G_VALUES must contain nonnegative values")
    for label, dimensions, warm_start in (
        ("LETTA_BOND_DIMS", LETTA_BOND_DIMS, RUN_LETTA),
        ("MPS_BOND_DIMS", MPS_BOND_DIMS, RUN_MPS_ONE_SITE),
    ):
        if not dimensions or any(int(value) <= 0 for value in dimensions):
            raise ValueError(f"{label} must contain positive integers")
        if tuple(sorted(set(dimensions))) != tuple(dimensions):
            raise ValueError(f"{label} must be unique and strictly increasing")
        if warm_start and dimensions[0] != 1:
            raise ValueError(f"{label} must start at D=1 for warm starts")
    if MAX_SWEEPS <= 0 or NOISE_ROUNDS <= 0:
        raise ValueError("MAX_SWEEPS and NOISE_ROUNDS must be positive")
    if INITIAL_NOISE < 0.0 or not 0.0 < NOISE_DECAY <= 1.0:
        raise ValueError("the noise schedule is invalid")


def _jg_slug(value: float) -> str:
    return f"jg_{float(value):.5f}".replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_tensor_state(
    path: Path,
    state,
    metadata: dict,
    *,
    letta: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "physical_dim": np.asarray(state.physical_dim, dtype=np.int64),
        "tensor_count": np.asarray(len(state.tensors), dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if letta:
        arrays["lattice_shape"] = np.asarray(state.lattice_shape, dtype=np.int64)
        arrays["coordinates"] = np.asarray(state.coordinates, dtype=np.int64)
    arrays.update(
        {f"tensor_{index:04d}": np.asarray(tensor) for index, tensor in enumerate(state.tensors)}
    )
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_tensor_state(path: Path, *, letta: bool):
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 1:
            raise ValueError(f"unsupported state format in {path}")
        count = int(archive["tensor_count"])
        tensors = [
            np.asarray(archive[f"tensor_{index:04d}"]).copy()
            for index in range(count)
        ]
        metadata = json.loads(str(archive["metadata_json"].item()))
        if letta:
            shape = tuple(int(value) for value in archive["lattice_shape"])
            coordinates = tuple(
                tuple(int(value) for value in coordinate)
                for coordinate in archive["coordinates"]
            )
            state_class = (
                _later_chain_state_class()
                if metadata.get("physical_index_convention")
                == LATER_CHAIN_CONVENTION
                else LatticeLETTA
            )
            state = state_class(
                shape,
                int(archive["physical_dim"]),
                tensors,
                coordinates=coordinates,
            )
        else:
            state = SnakeMPS(tensors)
    return state, metadata


def _problem(n_x: int, coupling: float):
    shape = (int(n_x), 3, 3)
    ordering = _normalized_ordering()
    coordinates = ordered_coordinates(shape, ordering=ordering)
    bonds = nearest_neighbor_bonds(shape, ordering=ordering)
    mpo = transverse_field_ising_mpo(
        shape,
        coupling=float(coupling),
        field=FIELD_G,
        ordering=ordering,
    )
    return shape, coordinates, bonds, mpo


def _history(result) -> list[dict]:
    records = []
    for position, sweep in enumerate(result.history, start=1):
        iteration = getattr(sweep, "sweep", getattr(sweep, "iteration", position))
        record = {
            "iteration": int(iteration),
            "energy": float(sweep.energy),
            "energy_change": float(sweep.energy_change),
            "energy_density_change": float(sweep.energy_density_change),
        }
        for name in ("direction", "bond_dimension", "max_discarded_weight"):
            if hasattr(sweep, name):
                value = getattr(sweep, name)
                record[name] = float(value) if name == "max_discarded_weight" else value
        records.append(record)
    return records


def _result_payload(
    *,
    solver: str,
    update_scheme: str,
    n_x: int,
    bond_dim: int,
    coupling: float,
    result,
    runtime: float,
    state_path: Path,
    initial_energy: float,
    coordinates,
    bonds,
    mpo,
) -> dict:
    return {
        "solver": solver,
        "update_scheme": update_scheme,
        "ordering": _normalized_ordering(),
        "bond_dimension": int(bond_dim),
        "N": int(n_x),
        "lattice_shape": [int(n_x), 3, 3],
        "nsites": len(coordinates),
        "nbonds": len(bonds),
        "mpo_bond_dimension": max(mpo.bond_dimensions, default=1),
        "coupling": float(coupling),
        "field": FIELD_G,
        "J_over_g": float(coupling),
        "energy": float(result.energy),
        "initial_energy": float(initial_energy),
        "runtime_seconds": float(runtime),
        "converged": bool(result.converged),
        "sweeps": int(result.sweeps),
        "requested_sweeps": int(MAX_SWEEPS),
        "tolerance": float(ENERGY_DENSITY_TOLERANCE),
        "seed": int(RANDOM_SEED),
        "energy_history": _history(result),
        "message": str(result.message),
        "state_path": str(state_path.resolve()),
        "hostname": socket.gethostname(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _case_paths(dataset: str, coupling: float, n_x: int) -> tuple[Path, Path]:
    slug = _jg_slug(coupling)
    stem = f"N{n_x:02d}"
    return (
        RESULTS_ROOT / dataset / slug / f"{stem}.json",
        STATES_ROOT / dataset / slug / f"{stem}.npz",
    )


def _completed_state(
    result_path: Path,
    state_path: Path,
    *,
    solver: str,
    n_x: int,
    coupling: float,
    bond_dim: int,
    letta: bool,
):
    if FORCE_RERUN or not result_path.is_file() or not state_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        valid = (
            payload.get("solver") == solver
            and payload.get("ordering") == _normalized_ordering()
            and payload.get("N") == n_x
            and payload.get("bond_dimension") == bond_dim
            and math.isclose(
                float(payload.get("J_over_g", float("nan"))),
                coupling,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and payload.get("energy_history")
            and payload.get("state_sha256") == _sha256(state_path)
        )
        if letta:
            valid = valid and payload.get(
                "physical_index_convention"
            ) == _physical_index_convention()
        if not valid:
            return None
        state, metadata = _load_tensor_state(state_path, letta=letta)
        if letta:
            shape, coordinates, _bonds, _mpo = _problem(n_x, coupling)
            _validate_letta_state(
                state,
                shape=shape,
                coordinates=coordinates,
                bond_dim=bond_dim,
            )
        return state, metadata, payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _letta_qr_gauge_residual(state, direction: str) -> float:
    residuals = []
    if direction == "lr":
        tensors = state.tensors[:-1]
        for tensor in tensors:
            matrix = tensor.reshape(-1, tensor.shape[-1])
            gram = matrix.conj().T @ matrix
            residuals.append(np.linalg.norm(gram - np.eye(gram.shape[0])))
    elif direction == "rl":
        tensors = state.tensors[1:]
        for tensor in tensors:
            matrix = tensor.reshape(tensor.shape[0], -1)
            gram = matrix @ matrix.conj().T
            residuals.append(np.linalg.norm(gram - np.eye(gram.shape[0])))
    else:
        raise ValueError("direction must be 'lr' or 'rl'")
    return max((float(value) for value in residuals), default=0.0)


def _letta_qr_gauge_pass(state, direction: str = "rl") -> float:
    if direction == "lr":
        for site in range(state.nsites - 1):
            tensor = state.tensors[site]
            matrix = tensor.reshape(-1, tensor.shape[-1])
            q_matrix, r_matrix = np.linalg.qr(matrix, mode="reduced")
            state.tensors[site] = q_matrix.reshape(
                tensor.shape[:-1] + (q_matrix.shape[1],)
            )
            state.tensors[site + 1] = np.tensordot(
                r_matrix, state.tensors[site + 1], axes=([1], [0])
            )
    elif direction == "rl":
        for site in range(state.nsites - 1, 0, -1):
            tensor = state.tensors[site]
            matrix = tensor.reshape(tensor.shape[0], -1)
            q_matrix, r_matrix = np.linalg.qr(matrix.T, mode="reduced")
            state.tensors[site] = q_matrix.T.reshape(
                (q_matrix.shape[1],) + tensor.shape[1:]
            )
            state.tensors[site - 1] = np.tensordot(
                state.tensors[site - 1], r_matrix.T, axes=([-1], [0])
            )
    else:
        raise ValueError("direction must be 'lr' or 'rl'")
    return _letta_qr_gauge_residual(state, direction)


def _letta_expand(parent, mpo, target: int, parent_energy: float, seed: int):
    candidates = []
    for round_index in range(NOISE_ROUNDS):
        noise = INITIAL_NOISE * NOISE_DECAY**round_index
        candidate = parent.expand_bond_dimension(
            target,
            noise=noise,
            seed=seed + 1000 + round_index,
        )
        gauge_residual = _letta_qr_gauge_pass(candidate, direction="rl")
        energy = float(candidate.expectation(mpo))
        candidates.append((energy, noise, candidate, gauge_residual))
        if energy <= parent_energy + 1.0e-4 * candidate.nsites:
            return candidate, energy, noise, gauge_residual
    energy, noise, candidate, gauge_residual = min(
        candidates, key=lambda item: item[0]
    )
    return candidate, energy, noise, gauge_residual


def _letta_optimize(mpo, shape, state, bond_dim: int):
    return letta_ground_state(
        mpo,
        lattice_shape=shape,
        bond_dim=bond_dim,
        seed=RANDOM_SEED,
        state=state,
        options=LETTADMROptions(
            max_sweeps=MAX_SWEEPS,
            tolerance=ENERGY_DENSITY_TOLERANCE,
            gauge_mode="qr",
            start_direction="lr",
            environment_granularity="site",
            use_sparse_mpo=True,
        ),
        use_bond_schedule=False,
        ordering=_normalized_ordering(),
    )


def _run_letta_chain(n_x: int, coupling: float) -> list[Path]:
    shape, coordinates, bonds, mpo = _problem(n_x, coupling)
    parent = None
    parent_energy = None
    parent_sha = None
    outputs = []
    for index, bond_dim in enumerate(LETTA_BOND_DIMS):
        dataset = _letta_dataset(bond_dim)
        result_path, state_path = _case_paths(dataset, coupling, n_x)
        completed = _completed_state(
            result_path,
            state_path,
            solver="letta",
            n_x=n_x,
            coupling=coupling,
            bond_dim=bond_dim,
            letta=True,
        )
        if completed is not None:
            parent, _metadata, payload = completed
            parent_energy = float(payload["energy"])
            parent_sha = payload["state_sha256"]
            outputs.append(result_path)
            print(f"  reuse LETTA D={bond_dim}: E={parent_energy:.15g}")
            continue
        if index == 0:
            parent = _new_letta_state(shape, coordinates)
            gauge_residual = _letta_qr_gauge_pass(parent, direction="rl")
            initial_energy = float(parent.expectation(mpo))
            initialization = "random_D1"
            used_noise = None
        else:
            parent, initial_energy, used_noise, gauge_residual = _letta_expand(
                parent,
                mpo,
                bond_dim,
                float(parent_energy),
                RANDOM_SEED,
            )
            initialization = (
                f"D{LETTA_BOND_DIMS[index - 1]}_embedding_noise_schedule"
            )
        _validate_letta_state(
            parent,
            shape=shape,
            coordinates=coordinates,
            bond_dim=bond_dim,
        )
        start = perf_counter()
        result = _letta_optimize(mpo, shape, parent, bond_dim)
        runtime = perf_counter() - start
        parent = result.state
        _validate_letta_state(
            parent,
            shape=shape,
            coordinates=coordinates,
            bond_dim=bond_dim,
        )
        parent_energy = float(result.energy)
        state_metadata = {
            "solver": "letta",
            "update_scheme": (
                "letta_continuous_snake_later_chain"
                if _normalized_ordering() == "continuous-snake"
                else "letta_compact"
            ),
            "ordering": _normalized_ordering(),
            "physical_index_convention": _physical_index_convention(),
            "bond_dimension": bond_dim,
            "N": n_x,
            "coupling": coupling,
            "field": FIELD_G,
            "sweeps": MAX_SWEEPS,
            "tolerance": ENERGY_DENSITY_TOLERANCE,
            "seed": RANDOM_SEED,
            "initialization": initialization,
            "used_kick": used_noise,
            "initial_qr_gauge_direction": "rl",
            "initial_qr_gauge_max_residual": gauge_residual,
        }
        _save_tensor_state(state_path, parent, state_metadata, letta=True)
        state_sha = _sha256(state_path)
        payload = _result_payload(
            solver="letta",
            update_scheme=(
                "site_recursive_continuous_snake_later_chain"
                if _normalized_ordering() == "continuous-snake"
                else "site_recursive_compact"
            ),
            n_x=n_x,
            bond_dim=bond_dim,
            coupling=coupling,
            result=result,
            runtime=runtime,
            state_path=state_path,
            initial_energy=initial_energy,
            coordinates=coordinates,
            bonds=bonds,
            mpo=mpo,
        )
        payload.update(
            {
                "physical_index_convention": _physical_index_convention(),
                "initialization": initialization,
                "used_kick": used_noise,
                "initial_qr_gauge_direction": "rl",
                "initial_qr_gauge_max_residual": gauge_residual,
                "state_sha256": state_sha,
                "parent_state_sha256": parent_sha,
            }
        )
        _atomic_json(result_path, payload)
        parent_sha = state_sha
        outputs.append(result_path)
        print(
            f"  LETTA D={bond_dim}: E={result.energy:.15g}, "
            f"sweeps={result.sweeps}, converged={result.converged}"
        )
    return outputs


def _run_mps_two_site(n_x: int, coupling: float, bond_dim: int) -> Path:
    shape, coordinates, bonds, mpo = _problem(n_x, coupling)
    del shape
    dataset = f"mps_two_site_{_ordering_slug()}_D{bond_dim}"
    result_path, state_path = _case_paths(dataset, coupling, n_x)
    completed = _completed_state(
        result_path,
        state_path,
        solver="mps",
        n_x=n_x,
        coupling=coupling,
        bond_dim=bond_dim,
        letta=False,
    )
    if completed is not None:
        energy = float(completed[2]["energy"])
        print(f"  reuse MPS two-site D={bond_dim}: E={energy:.15g}")
        return result_path
    state = SnakeMPS.random(
        mpo.nsites,
        physical_dim=2,
        bond_dim=bond_dim,
        seed=RANDOM_SEED,
    )
    initial_energy = float(state.expectation(mpo))
    start = perf_counter()
    result = mps_dmrg(
        mpo,
        state=state,
        bond_dim=bond_dim,
        seed=RANDOM_SEED,
        options=MPSDMRGOptions(
            max_sweeps=MAX_SWEEPS,
            tolerance=ENERGY_DENSITY_TOLERANCE,
            eigensolver_tolerance=EIGENSOLVER_TOLERANCE,
            eigensolver_max_iterations=EIGENSOLVER_MAX_ITERATIONS,
        ),
    )
    runtime = perf_counter() - start
    state_metadata = {
        "solver": "mps",
        "update_scheme": "two_site",
        "ordering": _normalized_ordering(),
        "bond_dimension": bond_dim,
        "N": n_x,
        "coupling": coupling,
        "field": FIELD_G,
        "sweeps": MAX_SWEEPS,
        "tolerance": ENERGY_DENSITY_TOLERANCE,
        "seed": RANDOM_SEED,
    }
    _save_tensor_state(state_path, result.state, state_metadata, letta=False)
    state_sha = _sha256(state_path)
    payload = _result_payload(
        solver="mps",
        update_scheme="two_site",
        n_x=n_x,
        bond_dim=bond_dim,
        coupling=coupling,
        result=result,
        runtime=runtime,
        state_path=state_path,
        initial_energy=initial_energy,
        coordinates=coordinates,
        bonds=bonds,
        mpo=mpo,
    )
    payload["state_sha256"] = state_sha
    _atomic_json(result_path, payload)
    print(
        f"  MPS two-site D={bond_dim}: E={result.energy:.15g}, "
        f"sweeps={result.sweeps}, converged={result.converged}"
    )
    return result_path


def _one_site_action(left, right, transitions, shape, vector):
    theta = np.asarray(vector).reshape(shape)
    result = np.zeros(shape, dtype=np.result_type(left, right, theta))
    for left_channel, right_channel, operator in transitions:
        value = np.tensordot(left[left_channel], theta, axes=([1], [0]))
        value = np.tensordot(operator, value, axes=([1], [1])).transpose(1, 0, 2)
        value = np.tensordot(value, right[right_channel], axes=([2], [1]))
        result += value
    return result.reshape(-1)


def _lowest_one_site_vector(left, right, transitions, theta, options):
    shape = theta.shape
    dimension = theta.size

    def action(vector):
        return _one_site_action(left, right, transitions, shape, vector)

    if dimension <= 64:
        identity = np.eye(dimension, dtype=np.result_type(theta))
        matrix = np.column_stack([action(identity[:, column]) for column in range(dimension)])
        matrix = 0.5 * (matrix + matrix.conj().T)
        values, vectors = linalg.eigh(
            matrix,
            subset_by_index=[0, 0],
            check_finite=False,
        )
        return float(np.real(values[0])), vectors[:, 0]
    operator = LinearOperator(
        (dimension, dimension),
        matvec=action,
        dtype=np.result_type(theta),
    )
    initial = theta.reshape(-1).copy()
    initial /= np.linalg.norm(initial)
    try:
        values, vectors = eigsh(
            operator,
            k=1,
            which="SA",
            v0=initial,
            tol=options.eigensolver_tolerance,
            maxiter=options.eigensolver_max_iterations,
        )
    except ArpackNoConvergence as error:
        if error.eigenvectors is None or error.eigenvectors.shape[1] == 0:
            raise
        values, vectors = error.eigenvalues, error.eigenvectors
    return float(np.real(values[0])), vectors[:, 0]


def _left_canonicalize_and_push(state, site: int) -> None:
    tensor = state.tensors[site]
    left_dim, physical_dim, right_dim = tensor.shape
    q_matrix, r_matrix = np.linalg.qr(
        tensor.reshape(left_dim * physical_dim, right_dim),
        mode="reduced",
    )
    keep = q_matrix.shape[1]
    state.tensors[site] = q_matrix.reshape(left_dim, physical_dim, keep)
    state.tensors[site + 1] = np.tensordot(
        r_matrix, state.tensors[site + 1], axes=([1], [0])
    )


def _right_canonicalize_and_push(state, site: int) -> None:
    tensor = state.tensors[site]
    left_dim, physical_dim, right_dim = tensor.shape
    q_matrix, r_matrix = np.linalg.qr(
        tensor.reshape(left_dim, physical_dim * right_dim).T,
        mode="reduced",
    )
    keep = q_matrix.shape[1]
    state.tensors[site] = q_matrix.T.reshape(keep, physical_dim, right_dim)
    state.tensors[site - 1] = np.tensordot(
        state.tensors[site - 1], r_matrix.T, axes=([-1], [0])
    )


def _one_site_expand(state, target: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    tensors = []
    for site, old in enumerate(state.tensors):
        left = 1 if site == 0 else target
        right = 1 if site == state.nsites - 1 else target
        new = np.zeros((left, old.shape[1], right), dtype=old.dtype)
        new[: old.shape[0], :, : old.shape[2]] = old
        if noise > 0.0:
            mask = np.ones(new.shape, dtype=bool)
            mask[: old.shape[0], :, : old.shape[2]] = False
            new[mask] = noise * rng.normal(size=int(mask.sum()))
        tensors.append(new)
    return SnakeMPS(tensors).right_canonicalize().normalize()


def _mps_one_site_optimize(mpo, state):
    from pyqed._letta_one_site_opt._letta_for_3d.mps import (
        _build_left_environments,
        _build_right_environments,
        _extend_left,
        _extend_right,
    )

    options = MPSDMRGOptions(
        max_sweeps=MAX_SWEEPS,
        tolerance=ENERGY_DENSITY_TOLERANCE,
        eigensolver_tolerance=EIGENSOLVER_TOLERANCE,
        eigensolver_max_iterations=EIGENSOLVER_MAX_ITERATIONS,
    )
    state = state.copy().right_canonicalize().normalize()
    previous_energy = float(state.expectation(mpo))
    history = []
    converged = False
    message = "STOP: MAXIMUM SWEEPS REACHED"
    for sweep in range(1, MAX_SWEEPS + 1):
        right_environments = _build_right_environments(state, mpo)
        left = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
        for site in range(state.nsites):
            theta = state.tensors[site]
            _energy, vector = _lowest_one_site_vector(
                left,
                right_environments[site + 1],
                mpo.transitions[site],
                theta,
                options,
            )
            state.tensors[site] = vector.reshape(theta.shape)
            if site < state.nsites - 1:
                _left_canonicalize_and_push(state, site)
                left = _extend_left(left, state.tensors[site], mpo.factors[site])
        left_environments = _build_left_environments(state, mpo)
        right = np.ones((1, 1, 1), dtype=np.result_type(*state.tensors))
        for site in range(state.nsites - 1, -1, -1):
            theta = state.tensors[site]
            _energy, vector = _lowest_one_site_vector(
                left_environments[site],
                right,
                mpo.transitions[site],
                theta,
                options,
            )
            state.tensors[site] = vector.reshape(theta.shape)
            if site > 0:
                _right_canonicalize_and_push(state, site)
                right = _extend_right(right, state.tensors[site], mpo.factors[site])
        energy = float(state.expectation(mpo))
        change = abs(energy - previous_energy)
        history.append(
            SimpleNamespace(
                iteration=sweep,
                energy=energy,
                energy_change=change,
                energy_density_change=change / state.nsites,
                direction="lr_rl",
            )
        )
        if change / state.nsites <= ENERGY_DENSITY_TOLERANCE:
            converged = True
            message = "CONVERGENCE: SWEEP ENERGY DENSITY CHANGE <= TOLERANCE"
            break
        previous_energy = energy
    return SimpleNamespace(
        state=state,
        energy=float(history[-1].energy),
        converged=converged,
        sweeps=len(history),
        history=tuple(history),
        message=message,
    )


def _run_mps_one_site_chain(n_x: int, coupling: float) -> list[Path]:
    shape, coordinates, bonds, mpo = _problem(n_x, coupling)
    del shape
    parent = None
    parent_energy = None
    parent_sha = None
    outputs = []
    for index, bond_dim in enumerate(MPS_BOND_DIMS):
        dataset = f"mps_one_site_{_ordering_slug()}_D{bond_dim}"
        result_path, state_path = _case_paths(dataset, coupling, n_x)
        completed = _completed_state(
            result_path,
            state_path,
            solver="mps",
            n_x=n_x,
            coupling=coupling,
            bond_dim=bond_dim,
            letta=False,
        )
        if completed is not None:
            parent, _metadata, payload = completed
            parent_energy = float(payload["energy"])
            parent_sha = payload["state_sha256"]
            outputs.append(result_path)
            print(f"  reuse MPS one-site D={bond_dim}: E={parent_energy:.15g}")
            continue
        if index == 0:
            parent = SnakeMPS.random(
                mpo.nsites,
                physical_dim=2,
                bond_dim=1,
                seed=RANDOM_SEED,
            )
            initial_energy = float(parent.expectation(mpo))
            initialization = "random_D1"
            used_noise = None
        else:
            candidates = []
            for round_index in range(NOISE_ROUNDS):
                noise = INITIAL_NOISE * NOISE_DECAY**round_index
                candidate = _one_site_expand(
                    parent,
                    bond_dim,
                    noise,
                    RANDOM_SEED + 1000 * index + round_index,
                )
                energy = float(candidate.expectation(mpo))
                candidates.append((energy, noise, candidate))
                if energy <= float(parent_energy) + 1.0e-4 * candidate.nsites:
                    break
            initial_energy, used_noise, parent = min(
                candidates, key=lambda item: item[0]
            )
            initialization = (
                f"D{MPS_BOND_DIMS[index - 1]}_embedding_noise_schedule"
            )
        start = perf_counter()
        result = _mps_one_site_optimize(mpo, parent)
        runtime = perf_counter() - start
        parent = result.state
        parent_energy = float(result.energy)
        state_metadata = {
            "solver": "mps",
            "update_scheme": "one_site",
            "ordering": _normalized_ordering(),
            "bond_dimension": bond_dim,
            "N": n_x,
            "coupling": coupling,
            "field": FIELD_G,
            "sweeps": MAX_SWEEPS,
            "tolerance": ENERGY_DENSITY_TOLERANCE,
            "seed": RANDOM_SEED,
            "initialization": initialization,
            "used_kick": used_noise,
        }
        _save_tensor_state(state_path, parent, state_metadata, letta=False)
        state_sha = _sha256(state_path)
        payload = _result_payload(
            solver="mps",
            update_scheme="one_site",
            n_x=n_x,
            bond_dim=bond_dim,
            coupling=coupling,
            result=result,
            runtime=runtime,
            state_path=state_path,
            initial_energy=initial_energy,
            coordinates=coordinates,
            bonds=bonds,
            mpo=mpo,
        )
        payload.update(
            {
                "initialization": initialization,
                "used_kick": used_noise,
                "state_sha256": state_sha,
                "parent_state_sha256": parent_sha,
            }
        )
        _atomic_json(result_path, payload)
        parent_sha = state_sha
        outputs.append(result_path)
        print(
            f"  MPS one-site D={bond_dim}: E={result.energy:.15g}, "
            f"sweeps={result.sweeps}, converged={result.converged}"
        )
    return outputs


SUMMARY_FIELDS = (
    "N",
    "nsites",
    "J_over_g",
    "solver",
    "update_scheme",
    "ordering",
    "physical_index_convention",
    "bond_dimension",
    "energy",
    "energy_per_site",
    "initial_energy",
    "converged",
    "sweeps",
    "requested_sweeps",
    "runtime_seconds",
    "initialization",
    "used_kick",
    "result_file",
    "state_file",
)


def _write_summary(result_paths: list[Path]) -> None:
    rows = []
    for result_path in sorted(set(result_paths)):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        nsites = int(payload["nsites"])
        rows.append(
            {
                "N": payload["N"],
                "nsites": nsites,
                "J_over_g": payload["J_over_g"],
                "solver": payload["solver"],
                "update_scheme": payload["update_scheme"],
                "ordering": payload["ordering"],
                "physical_index_convention": payload.get(
                    "physical_index_convention", ""
                ),
                "bond_dimension": payload["bond_dimension"],
                "energy": payload["energy"],
                "energy_per_site": float(payload["energy"]) / nsites,
                "initial_energy": payload["initial_energy"],
                "converged": payload["converged"],
                "sweeps": payload["sweeps"],
                "requested_sweeps": payload["requested_sweeps"],
                "runtime_seconds": payload["runtime_seconds"],
                "initialization": payload.get("initialization", "random"),
                "used_kick": payload.get("used_kick", ""),
                "result_file": str(result_path.resolve()),
                "state_file": payload["state_path"],
            }
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_name(f".{SUMMARY_PATH.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, SUMMARY_PATH)


def main() -> None:
    _validate_configuration()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_start = perf_counter()
    print("3D transverse-field Ising scan")
    print(f"  N_x values: {N_X_VALUES}")
    print(f"  J/g values: {J_OVER_G_VALUES}")
    print(f"  ordering: {_normalized_ordering()}")
    if RUN_LETTA:
        print(f"  LETTA physical legs: {_physical_index_convention()}")
    print(f"  output: {OUTPUT_ROOT}")
    for n_x in N_X_VALUES:
        for coupling in J_OVER_G_VALUES:
            print(f"\nN_x={n_x}, J/g={coupling:g}")
            if RUN_MPS_TWO_SITE:
                for bond_dim in MPS_BOND_DIMS:
                    all_results.append(
                        _run_mps_two_site(int(n_x), float(coupling), int(bond_dim))
                    )
                    _write_summary(all_results)
            if RUN_MPS_ONE_SITE:
                all_results.extend(
                    _run_mps_one_site_chain(int(n_x), float(coupling))
                )
                _write_summary(all_results)
            if RUN_LETTA:
                all_results.extend(_run_letta_chain(int(n_x), float(coupling)))
                _write_summary(all_results)
    elapsed = perf_counter() - total_start
    print(f"\nCompleted {len(all_results)} cases in {elapsed:.3f} s")
    print(f"Energy summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
