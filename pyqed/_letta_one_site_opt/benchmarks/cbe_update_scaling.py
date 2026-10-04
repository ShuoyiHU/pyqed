"""Click-run complete-pass costs on controlled 1D/2D/3D tensor networks.

The Hamiltonian is a sum of real Hermitian product operators, a cost probe
rather than a condensed-matter convergence test. Timings include all solver
work and environment setup. Instrumented contraction/decomposition estimates
are collected in a separate run and are not measured peak process memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pyqed._letta_one_site_opt import cbe_general
from pyqed._letta_one_site_opt.operators import LatticeMPO
from pyqed._letta_one_site_opt.solver import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.state import LatticeLETTA
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
from pyqed._letta_one_site_opt.benchmarks.cbe_scaling import _profile_call

GEOMETRIES = {"1d": (1, 6), "2d": (2, 3), "3d": (2, 2, 2)}
METHODS = ("one_site", "strict_cbe", "two_site")


def probe_mpo(lattice_shape, *, physical_dimension, width, seed):
    """Fixed-width sum of products; every local factor is real symmetric."""
    rng = np.random.default_rng(seed)
    factors = []
    nsites, d = int(np.prod(lattice_shape)), int(physical_dimension)
    for site in range(nsites):
        factor = np.zeros((1 if site == 0 else width,
                           1 if site == nsites - 1 else width, d, d))
        for channel in range(width):
            matrix = rng.normal(size=(d, d))
            matrix = matrix + matrix.T
            matrix /= np.linalg.norm(matrix, ord=2)
            factor[0 if site == 0 else channel,
                   0 if site == nsites - 1 else channel] = matrix
        factors.append(factor)
    return LatticeMPO(factors, lattice_shape=lattice_shape)


def _fingerprint(state):
    digest = hashlib.sha256()
    for tensor in state.tensors:
        digest.update(str(tensor.shape).encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def profile_update_point(*, lattice_shape, bond_dimension, physical_dimension,
                         mpo_width=2, preselection_dimension=2,
                         expansion_dimension=1, repeats=3, passes=1, seed=817):
    if min(bond_dimension, physical_dimension, mpo_width, repeats, passes,
           expansion_dimension) <= 0 or preselection_dimension < expansion_dimension:
        raise ValueError("positive dimensions/counts and preselection >= expansion required")
    state = LatticeLETTA.random(lattice_shape, physical_dim=physical_dimension,
                                bond_dim=bond_dimension, seed=seed)
    mpo = probe_mpo(lattice_shape, physical_dimension=physical_dimension,
                    width=mpo_width, seed=seed + 1)
    fingerprint = _fingerprint(state)
    inventories = [cbe_general.PhysicalIndexInventory.from_state(state, k)
                   for k in range(state.nsites - 1)]
    point = {
        "scope": "complete_solver_including_environment_setup",
        "lattice_shape": list(lattice_shape), "bond_dimension": bond_dimension,
        "physical_dimension": physical_dimension, "mpo_width": mpo_width,
        "preselection_dimension": preselection_dimension,
        "expansion_dimension": expansion_dimension, "requested_passes": passes,
        "seed": seed, "initial_state_fingerprint": fingerprint,
        "initial_state_kind": "random LETTA (not an embedded MPS)",
        "initial_energy": float(state.expectation(mpo)),
        "categories_by_cut": [i.categories for i in inventories],
        "memory_note": "Largest instrumented array estimate; not measured peak RSS. "
                       "Excludes simultaneous arrays and library workspaces.",
        "timing_note": "Uninstrumented repeated complete solves after a profiled warm-up; "
                       "each starts from the same state. Pass means one LR or RL traversal.",
        "work_note": "opt_cost counts labelled contractions; SVD/eigh work are proxies. "
                     "Uninstrumented matrix products and iterative vector algebra are excluded.",
        "records": {},
    }

    def solve(method):
        if method == "two_site":
            return letta_two_site_dmrg(
                mpo, state=state, bond_dim=bond_dimension,
                options=LETTATwoSiteOptions(max_sweeps=passes, tolerance=1.e-12))
        return letta_dmrg(
            mpo, state=state,
            options=LETTADMROptions(
                max_sweeps=passes, tolerance=1.e-12,
                cbe_enabled=method == "strict_cbe", cbe_selector="shrewd",
                cbe_expansion_dimension=expansion_dimension,
                cbe_preselection_dimension=preselection_dimension))

    # The profiler wraps contraction execution, and restores it
    # before wall-clock repetitions. It also provides a per-method warm-up.
    for method in METHODS:
        result, profile = _profile_call(lambda: solve(method),
                                        live_tensors=[t.size for t in state.tensors])
        profile["estimated_largest_array_bytes"] = profile["largest_live_tensor"] * 8
        updates = [u for sweep in result.history for u in sweep.updates]
        diagnostics = [u.cbe_selection_diagnostics for u in updates
                       if getattr(u, "cbe_selection_diagnostics", None) is not None]
        point["records"][method] = {
            "profile": profile, "profiled_energy": float(result.energy),
            "profiled_passes": int(result.sweeps), "profiled_updates": len(updates),
            "selection_diagnostics": diagnostics,
            "initial_state_fingerprint": fingerprint,
            "elapsed_seconds": [], "energies": [], "passes": [],
        }
    for repeat in range(repeats):
        # Reverse order on alternating repeats to reduce simple order bias.
        for method in METHODS if repeat % 2 == 0 else METHODS[::-1]:
            started = perf_counter()
            result = solve(method)
            elapsed = perf_counter() - started
            record = point["records"][method]
            record["elapsed_seconds"].append(elapsed)
            record["energies"].append(float(result.energy))
            record["passes"].append(int(result.sweeps))
    for record in point["records"].values():
        record["median_seconds"] = float(np.median(record["elapsed_seconds"]))
    if _fingerprint(state) != fingerprint:
        raise AssertionError("a solver mutated the shared initial state")
    return point


def run_update_scaling(*, geometries=("1d", "2d", "3d"),
                       bond_dimensions=(2, 4), physical_dimensions=(2, 3),
                       preselection_dimensions=(2, 4), repeats=3, passes=1,
                       output=None, progress=False):
    report = {"points": [], "failures": {}, "repeats": repeats,
              "probe": "sum of Hermitian product operators; cost, not convergence",
              "axis_design": "one axis at a time around D=2,d=2,preselection=2"}
    for geometry in geometries:
        shape = GEOMETRIES[geometry]
        settings = dict.fromkeys(
            [(d, 2, 2) for d in bond_dimensions]
            + [(2, d, 2) for d in physical_dimensions]
            + [(2, 2, p) for p in preselection_dimensions])
        for D, d, p in settings:
            key = f"{geometry}:D={D},d={d},pre={p}"
            if progress:
                print("Starting " + key, flush=True)
            try:
                point = profile_update_point(
                    lattice_shape=shape, bond_dimension=D, physical_dimension=d,
                    preselection_dimension=p, repeats=repeats, passes=passes)
                point["geometry"] = geometry
                report["points"].append(point)
                if progress:
                    print("  " + ", ".join(
                        f"{m}={r['median_seconds']:.4f}s"
                        for m, r in point["records"].items()), flush=True)
            except Exception as error:
                report["failures"][key] = f"{type(error).__name__}: {error}"
                if progress:
                    print("  FAILED: " + report["failures"][key], flush=True)
            if output is not None:
                Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    integers = lambda value: tuple(map(int, value.split(",")))
    parser.add_argument("--geometries", type=lambda value: tuple(value.split(",")),
                        default=tuple(GEOMETRIES))
    parser.add_argument("--bond-dimensions", type=integers, default=(2, 4))
    parser.add_argument("--physical-dimensions", type=integers, default=(2, 3))
    parser.add_argument("--preselection-dimensions", type=integers, default=(2, 4))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_update_scaling(**vars(args), progress=True)
    print(f"Completed {len(report['points'])} points; {len(report['failures'])} failures.")
    return report


if __name__ == "__main__":
    main()
