"""Compare one-site, ALS-only, and ALS-plus-energy sweeps on small NN chains.

Run from the repository root with PYTHONPATH=. and BLAS/OpenMP threads set
to one. Dense Hilbert-space Hamiltonians are not needed by this benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import make_shared_initial_state
from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg


DEFAULT_CASES = (
    ("heisenberg", 6, 2),
    ("ising", 6, 2),
    ("blume_capel", 4, 2),
    ("spin1_heisenberg", 4, 2),
)


def run_case(model_name, length, bond_dim, *, seed=731, max_sweeps=200, tolerance=1e-11):
    model = build_model(model_name, dimension="1d", size=length)
    initial = make_shared_initial_state(model, bond_dim=bond_dim, seed=seed)
    records = []
    for method in ("one-site", "metric-als", "metric-als-energy"):
        started = perf_counter()
        if method == "one-site":
            result = letta_dmrg(
                model.mpo, state=initial.letta,
                options=LETTADMROptions(max_sweeps=max_sweeps, tolerance=tolerance),
            )
        else:
            result = letta_two_site_dmrg(
                model.mpo, state=initial.letta, bond_dim=bond_dim,
                options=LETTATwoSiteOptions(
                    max_sweeps=max_sweeps, tolerance=tolerance,
                    split_method=method, one_site_polish_sweeps=0,
                ),
            )
        elapsed = perf_counter() - started
        records.append({
            "method": method,
            "energy": result.energy,
            "physical_energy": float(result.state.expectation(model.mpo)),
            "converged": result.converged,
            "sweeps": result.sweeps,
            "elapsed_seconds": elapsed,
            "sweep_energies": [s.energy for s in result.history],
            "rejected_updates": sum(not u.accepted for s in result.history for u in s.updates),
            "energy_refinement_iterations": sum(
                getattr(u, "energy_refinement_iterations", 0)
                for s in result.history for u in s.updates
            ),
        })
    return {
        "model": model_name, "length": length, "bond_dim": bond_dim,
        "seed": seed, "max_sweeps": max_sweeps, "tolerance": tolerance,
        "initial_state_fingerprint": initial.fingerprint,
        "records": records,
        "als_minus_one_site": records[1]["energy"] - records[0]["energy"],
        "refined_minus_one_site": records[2]["energy"] - records[0]["energy"],
    }


def _case(value):
    try:
        model, length, bond_dim = value.split(":")
        length, bond_dim = int(length), int(bond_dim)
        if length < 2 or bond_dim < 1:
            raise ValueError
        return model, length, bond_dim
    except ValueError as error:
        raise argparse.ArgumentTypeError("use MODEL:L:D with L >= 2 and D >= 1") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=_case, action="append", help="repeat MODEL:L:D to select cases")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-sweeps", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = []
    for model, length, bond in args.case or DEFAULT_CASES:
        report = run_case(model, length, bond, seed=args.seed,
                          max_sweeps=args.max_sweeps, tolerance=args.tolerance)
        reports.append(report)
        print(f"{model} L={length} D={bond}: "
              f"ALS - one-site = {report['als_minus_one_site']:.6e}; "
              f"refined - one-site = {report['refined_minus_one_site']:.6e}; "
              f"converged = {[r['converged'] for r in report['records']]}", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(reports, indent=2) + "\n")
    return reports


if __name__ == "__main__":
    main()
