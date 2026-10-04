"""Command-line plumbing shared by the click-run model benchmarks."""

from __future__ import annotations

import argparse
import json

from .condensed_runner import SOLVERS, format_table, run_benchmark
from .nn_chain_models import CHAIN_MODEL_DEFAULTS


def parse_solvers(value):
    solvers = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(solvers) - set(SOLVERS)
    if not solvers or unknown:
        raise argparse.ArgumentTypeError(
            f"solvers must be a comma-separated subset of {SOLVERS}"
        )
    return solvers


def _parser(model_name, dimension):
    parser = argparse.ArgumentParser(
        description=(
            f"Compare LETTA one-site, exact/strict CBE, LETTA two-site, and "
            f"MPS two-site for the {dimension.upper()} {model_name} model."
        )
    )
    if dimension == "1d":
        parser.add_argument("--length", type=int, default=3 if model_name == "hubbard_holstein" else 4)
    else:
        parser.add_argument("--rows", type=int, default=2)
        parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--bond-dim", type=int, default=4)
    parser.add_argument("--expansion-dimension", type=int, default=1)
    parser.add_argument("--cbe-baseline-guard-fraction", type=float, default=0.2)
    parser.add_argument("--global-trim", action="store_true", help="ablate the strict-CBE conditional compression fix")
    parser.add_argument("--max-sweeps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--eigensolver-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--eigensolver-max-iterations", type=int, default=300,
                        help="maximum ARPACK restart iterations per iterative local solve")
    parser.add_argument("--exact-max-dimension", type=int, default=4096)
    parser.add_argument("--solvers", type=parse_solvers, default=SOLVERS)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--raise-on-failure", action="store_true")
    if model_name in CHAIN_MODEL_DEFAULTS:
        for name, default in CHAIN_MODEL_DEFAULTS[model_name].items():
            parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    elif model_name == "ising":
        parser.add_argument("--J", type=float, default=1.0)
        parser.add_argument("--h", type=float, default=1.0)
    elif model_name == "heisenberg":
        parser.add_argument("--J", type=float, default=1.0)
        parser.add_argument("--delta", type=float, default=1.0)
        parser.add_argument("--h", type=float, default=0.0)
    elif model_name == "bose_hubbard":
        parser.add_argument("--t", type=float, default=1.0)
        parser.add_argument("--U", type=float, default=4.0)
        parser.add_argument("--mu", type=float, default=2.0)
        parser.add_argument("--max-occupancy", type=int, default=2)
    elif model_name == "hubbard_holstein":
        parser.add_argument("--t", type=float, default=1.0)
        parser.add_argument("--U", type=float, default=0.8)
        parser.add_argument("--mu", type=float, default=0.0,
                            help="chemical potential; paper Hamiltonian uses 0, but fixes N=L separately")
        parser.add_argument("--omega", type=float, default=0.5)
        parser.add_argument("--g", type=float, default=0.2**0.5)
        parser.add_argument("--max-phonons", type=int, default=1,
                            help="maximum phonon occupation per site; local d=4*(max_phonons+1)")
    else:
        parser.add_argument("--t", type=float, default=1.0)
        parser.add_argument("--U", type=float, default=4.0)
        parser.add_argument("--mu", type=float, default=2.0)
    return parser


def _model_parameters(model_name, arguments):
    if model_name in CHAIN_MODEL_DEFAULTS:
        return {name: getattr(arguments, name) for name in CHAIN_MODEL_DEFAULTS[model_name]}
    if model_name == "hubbard_holstein":
        return {"t": arguments.t, "U": arguments.U, "mu": arguments.mu,
                "omega": arguments.omega, "g": arguments.g,
                "max_phonons": arguments.max_phonons}
    if model_name == "ising":
        return {"J": arguments.J, "h": arguments.h}
    if model_name == "heisenberg":
        return {"J": arguments.J, "delta": arguments.delta, "h": arguments.h}
    if model_name == "bose_hubbard":
        return {
            "t": arguments.t,
            "U": arguments.U,
            "mu": arguments.mu,
            "max_occupancy": arguments.max_occupancy,
        }
    return {"t": arguments.t, "U": arguments.U, "mu": arguments.mu}


def run_model_cli(model_name, dimension, argv=None):
    parser = _parser(model_name, dimension)
    arguments = parser.parse_args(argv)
    size = (
        arguments.length
        if dimension == "1d"
        else (arguments.rows, arguments.columns)
    )
    report = run_benchmark(
        model_name,
        dimension=dimension,
        size=size,
        model_parameters=_model_parameters(model_name, arguments),
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
        raise_on_failure=arguments.raise_on_failure,
    )
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"model={report['model']}  dimension={report['dimension']}  "
            f"shape={tuple(report['shape'])}  d={report['physical_dim']}  "
            f"D={report['bond_dim']}  deltaD={report['expansion_dimension']}"
        )
        exact = report["exact_energy"]
        exact_text = "disabled" if exact is None else f"{exact:.12f}"
        print(
            f"parameters={report['parameters']}  initial={report['initial_energy']:.12f}  "
            f"exact={exact_text}"
        )
        if model_name == "hubbard_holstein":
            print("Sector note: N and total spin are not fixed; this is not the paper's N=L, S=0 calculation. "
                  "The exact reference uses the same phonon cutoff and unrestricted sectors.")
        print(format_table(report))
    return report


__all__ = ["parse_solvers", "run_model_cli"]
