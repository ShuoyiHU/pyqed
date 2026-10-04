"""Run all registered condensed-model comparisons, including 1D NN chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import parse_solvers
from pyqed._letta_one_site_opt.benchmarks.condensed_models import MODEL_CASES
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
    SOLVERS,
    run_benchmark,
)


SUITE_CASES = MODEL_CASES


def run_suite(
    *,
    length=4,
    shape=(2, 2),
    bond_dim=4,
    expansion_dimension=1,
    cbe_baseline_guard_fraction=0.2,
    max_sweeps=50,
    seed=731,
    tolerance=1.0e-9,
    exact_max_dimension=4096,
    eigensolver_tolerance=1.0e-10,
    eigensolver_max_iterations=300,
    cbe_conditional_trim=True,
    solvers=SOLVERS,
):
    """Run every registered case, isolating whole-case failures."""

    cases = []
    failures = {}
    for offset, (model_name, dimension) in enumerate(SUITE_CASES):
        key = f"{model_name}_{dimension}"
        size = length if dimension == "1d" else tuple(shape)
        try:
            cases.append(
                run_benchmark(
                    model_name,
                    dimension=dimension,
                    size=size,
                    bond_dim=bond_dim,
                    expansion_dimension=expansion_dimension,
                    cbe_baseline_guard_fraction=cbe_baseline_guard_fraction,
                    max_sweeps=max_sweeps,
                    seed=int(seed) + offset,
                    tolerance=tolerance,
                    exact_max_dimension=exact_max_dimension,
                    eigensolver_tolerance=eigensolver_tolerance,
                    eigensolver_max_iterations=eigensolver_max_iterations,
                    cbe_conditional_trim=cbe_conditional_trim,
                    solvers=solvers,
                )
            )
        except Exception as error:
            failures[key] = f"{type(error).__name__}: {error}"
    return {
        "length": int(length),
        "shape": list(shape),
        "bond_dim": int(bond_dim),
        "expansion_dimension": int(expansion_dimension),
        "cbe_baseline_guard_fraction": float(cbe_baseline_guard_fraction),
        "max_sweeps": int(max_sweeps),
        "seed": int(seed),
        "tolerance": float(tolerance),
        "eigensolver_tolerance": float(eigensolver_tolerance),
        "eigensolver_max_iterations": int(eigensolver_max_iterations),
        "cbe_conditional_trim": bool(cbe_conditional_trim),
        "solvers": list(solvers),
        "cases": cases,
        "failures": failures,
    }


def format_suite_table(report):
    lines = [
        "case                       solver                 energy             error    seconds  swp  conv     dE/site  cbe-ok/fallback"
    ]
    for case in report["cases"]:
        case_name = case.get("case_name", f"{case['model']}_{case['dimension']}")
        for record in case["records"]:
            error = record["energy_error"]
            error_text = "n/a" if error is None else f"{error:.3e}"
            change = record["final_energy_density_change"]
            change_text = "n/a" if change is None else f"{change:.2e}"
            lines.append(
                f"{case_name:<26} {record['solver']:<22} "
                f"{record['energy']: .12f}  {error_text:>11}  "
                f"{record['elapsed_seconds']:7.3f}  "
                f"{record['sweeps']:3d}  "
                f"{str(record['converged']):>5}  "
                f"{change_text:>10}  "
                f"{record['cbe_accepted']:3d}/{record['cbe_fallbacks']:<3d}"
            )
        for solver, error in case["solver_failures"].items():
            lines.append(f"{case_name:<26} {solver:<22} FAILED: {error}")
    for case_name, error in report["failures"].items():
        lines.append(f"{case_name:<26} CASE FAILED: {error}")
    lines.append("LETTA swp = one directional pass (LR or RL); two passes make one LR+RL cycle.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--expansion-dimension", type=int, default=1)
    parser.add_argument("--cbe-baseline-guard-fraction", type=float, default=0.2)
    parser.add_argument("--global-trim", action="store_true")
    parser.add_argument("--max-sweeps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--eigensolver-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--eigensolver-max-iterations", type=int, default=300)
    parser.add_argument("--exact-max-dimension", type=int, default=4096)
    parser.add_argument("--solvers", type=parse_solvers, default=SOLVERS)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args(argv)
    report = run_suite(
        length=arguments.length,
        shape=(arguments.rows, arguments.columns),
        bond_dim=arguments.bond_dim,
        expansion_dimension=arguments.expansion_dimension,
        cbe_baseline_guard_fraction=arguments.cbe_baseline_guard_fraction,
        cbe_conditional_trim=not arguments.global_trim,
        max_sweeps=arguments.max_sweeps,
        seed=arguments.seed,
        tolerance=arguments.tolerance,
        eigensolver_tolerance=arguments.eigensolver_tolerance,
        eigensolver_max_iterations=arguments.eigensolver_max_iterations,
        exact_max_dimension=arguments.exact_max_dimension,
        solvers=arguments.solvers,
    )
    print(
        json.dumps(report, indent=2, sort_keys=True)
        if arguments.json
        else format_suite_table(report)
    )
    return report


if __name__ == "__main__":
    main()
