"""2D LETTA comparison from a shared initial physical state; no exact CBE oracle."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import _parser, _model_parameters
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import run_benchmark, format_table
from pyqed._letta_one_site_opt.state import _validate_neighborhoods


MODELS = ("ising", "heisenberg", "bose_hubbard", "fermi_hubbard")


def neighborhoods_for_shape(rows, columns, pattern):
    """Return physical dependency axes, independently of Hamiltonian bonds."""
    if pattern == "lattice":
        return None
    directions = [(0, 1), (1, 0)]
    if pattern == "diagonal":
        directions.append((1, 1))
    elif pattern == "bidirectional":
        directions.extend([(0, -1), (-1, 0)])
    else:
        raise ValueError("unknown tie pattern")
    result = []
    for row in range(rows):
        for column in range(columns):
            neighbors = [(row + dr) * columns + column + dc for dr, dc in directions
                         if 0 <= row + dr < rows and 0 <= column + dc < columns]
            result.append((row * columns + column, *neighbors))
    return tuple(result)


def main(argv=None, *, model=None, include_two_site=True):
    # Model-specific files expose their usual physical parameters through the
    # existing parser. The common entry point selects a model first.
    import argparse
    if model is None:
        chooser = argparse.ArgumentParser(add_help=False)
        chooser.add_argument("--model", choices=MODELS, default="ising")
        known, remaining = chooser.parse_known_args(argv)
        model, argv = known.model, remaining
    parser = _parser(model, "2d")
    parser.description = f"Compare one-site, strict CBE, and optional two-site LETTA for 2D {model}."
    default_solvers = ("letta_one_site", "letta_cbe_strict", "letta_two_site")
    if not include_two_site:
        default_solvers = default_solvers[:-1]
    parser.set_defaults(solvers=default_solvers, bond_dim=2, max_sweeps=10,
                        exact_max_dimension=256, raise_on_failure=True)
    parser.add_argument("--skip-two-site", action="store_true",
                        help="run one-site and strict CBE only")
    parser.add_argument("--two-site-max-sweeps", type=int,
                        help="separate directional sweep cap for the slower two-site solver")
    tying = parser.add_mutually_exclusive_group()
    tying.add_argument("--tie-pattern", choices=("lattice", "diagonal", "bidirectional"), default="lattice")
    tying.add_argument("--ties-json", type=Path,
                       help="JSON list of physical site indices per tensor, home site first")
    parser.add_argument("--output", type=Path, help="save energies, timings and convergence histories as JSON")
    args = parser.parse_args(argv)
    solvers = tuple(s for s in args.solvers if not (args.skip_two_site and s == "letta_two_site"))
    if not solvers:
        parser.error("at least one solver must remain after --skip-two-site")
    neighborhoods = (json.loads(args.ties_json.read_text()) if args.ties_json else
                     neighborhoods_for_shape(args.rows, args.columns, args.tie_pattern))
    if neighborhoods is not None:
        neighborhoods = _validate_neighborhoods(neighborhoods, args.rows * args.columns)
    report = run_benchmark(
        model, dimension="2d", size=(args.rows, args.columns),
        model_parameters=_model_parameters(model, args),
        bond_dim=args.bond_dim, expansion_dimension=args.expansion_dimension,
        max_sweeps=args.max_sweeps, two_site_max_sweeps=args.two_site_max_sweeps,
        tolerance=args.tolerance, seed=args.seed, neighborhoods=neighborhoods,
        eigensolver_tolerance=args.eigensolver_tolerance,
        eigensolver_max_iterations=args.eigensolver_max_iterations,
        cbe_baseline_guard_fraction=args.cbe_baseline_guard_fraction,
        cbe_conditional_trim=not args.global_trim,
        exact_max_dimension=args.exact_max_dimension, solvers=solvers,
        raise_on_failure=args.raise_on_failure,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{model}: shape={tuple(report['shape'])}, D={args.bond_dim}, seed={args.seed}")
        print(format_table(report))
        if args.two_site_max_sweeps is not None and "letta_two_site" in solvers:
            print(f"Two-site sweep cap: {args.two_site_max_sweeps}; other methods: {args.max_sweeps}.")
        for record in report["records"]:
            if not record["converged"]:
                print(f"{record['solver']}: {record['message']}")
    return report


if __name__ == "__main__":
    main()
