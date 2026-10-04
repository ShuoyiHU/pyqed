"""Click Run in an IDE, or execute this file from any working directory.

Edit DEFAULTS below for a no-argument run. CLI arguments override those values.
Outputs: per-sweep CSV, full JSON (including local diagnostics), and PNG curves.
The solver is timed separately from exact references, validation, and plotting.
"""
from pathlib import Path
import os
import sys

# Set before importing numerical libraries. In an already-running IDE kernel,
# restart the kernel to ensure these settings take effect.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------- Edit these, then click Run ----------------
DEFAULTS = dict(
    rows=1, columns=20, bond_dims=(4, 8, 16), seed=733,
    field=0.9, max_sweeps=12, repeats=3,
    tolerance=1e-10, metric_tolerance=1e-12,
    eigensolver_tolerance=1e-10, eigensolver_max_iterations=300,
    target_error=1e-8, exact_max_dimension=1024,
    output=ROOT / "output" / "benchmarks" / "gauge_convergence",
)
# ------------------------------------------------------------

import argparse
from collections import Counter
import csv
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
import scipy

from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, frontier_gauge_cuts, letta_dmrg
from pyqed._letta_one_site_opt._letta_for_2d import transverse_field_ising_mpo


def run_comparison(config):
    shape = (config["rows"], config["columns"])
    mpo = transverse_field_ising_mpo(shape, field=config["field"])
    dense = None
    exact_energy = None
    reference_started = perf_counter()
    if mpo.shape[0] <= config["exact_max_dimension"]:
        dense = mpo.to_dense(max_sites=mpo.nsites)
        exact_energy = float(eigh(dense, eigvals_only=True, subset_by_index=(0, 0))[0])
    report = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        threads={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        exact_energy=exact_energy, reference_seconds=perf_counter() - reference_started,
        sweep_definition="One directional pass; LR+RL is two sweeps.",
        convergence_rule="abs(E_current - E_previous) / nsites <= tolerance",
        runs=[],
    )
    print(f"Ising {shape}, field={config['field']}; exact E={exact_energy}", flush=True)
    print("One sweep = LR or RL. Solver time includes gauge initialization; references and plots are excluded.", flush=True)
    for dimension in config["bond_dims"]:
        initial = LatticeLETTA.random(shape, bond_dim=dimension, seed=config["seed"], real=True)
        initial_energy = initial.expectation(mpo)
        initial_hash = hashlib.sha256(b"".join(a.tobytes() for a in initial.tensors)).hexdigest()
        admissible = sum(c.admissible for c in frontier_gauge_cuts(initial))
        for repeat in range(config["repeats"]):
            modes = ("qr", "frontier") if repeat % 2 == 0 else ("frontier", "qr")
            for mode in modes:
                options = LETTADMROptions(
                    gauge_mode=mode, max_sweeps=config["max_sweeps"],
                    tolerance=config["tolerance"], metric_tolerance=config["metric_tolerance"],
                    eigensolver_tolerance=config["eigensolver_tolerance"],
                    eigensolver_max_iterations=config["eigensolver_max_iterations"],
                    matrix_free=True, dense_solver_threshold=1,
                )
                started = perf_counter()
                result = letta_dmrg(mpo, state=initial, options=options)
                seconds = perf_counter() - started
                # Fresh physical contraction is deliberately outside the timed solve.
                physical_energy = result.state.expectation(mpo)
                if not np.isfinite(physical_energy) or abs(physical_energy - result.energy) > 1e-8:
                    raise AssertionError("cached and fresh physical energies disagree")
                if initial_hash != hashlib.sha256(b"".join(a.tobytes() for a in initial.tensors)).hexdigest():
                    raise AssertionError("solver modified the shared initial state")
                physical_residual = None
                if dense is not None:
                    vector = result.state.state_vector()
                    vector = vector / np.linalg.norm(vector)
                    physical_residual = float(np.linalg.norm(dense @ vector - physical_energy * vector))
                history = []
                previous_seconds = 0.
                for sweep in result.history:
                    kinds = Counter(u.metric_kind for u in sweep.updates)
                    history.append(dict(
                        sweep=sweep.sweep, direction=sweep.direction, energy=sweep.energy,
                        energy_error=None if exact_energy is None else sweep.energy - exact_energy,
                        energy_density_change=sweep.energy_density_change,
                        elapsed_seconds=sweep.elapsed_seconds,
                        pass_seconds=sweep.elapsed_seconds - previous_seconds,
                        metric_kinds=dict(kinds), compact_updates=sum(v for k, v in kinds.items() if k != "general"),
                        total_updates=len(sweep.updates), accepted_updates=sum(u.accepted for u in sweep.updates),
                        hamiltonian_applications=sum(u.hamiltonian_applications for u in sweep.updates),
                        local_updates=[dict(site=u.site, energy=u.energy, accepted=u.accepted,
                                            metric_kind=u.metric_kind, metric_rank=u.metric_rank,
                                            local_dimension=u.local_dimension, residual_norm=u.residual_norm)
                                       for u in sweep.updates],
                    ))
                    previous_seconds = sweep.elapsed_seconds
                hit = next((h for h in history if h["energy_error"] is not None
                            and abs(h["energy_error"]) <= config["target_error"]), None)
                record = dict(
                    bond_dim=dimension, gauge=mode, repeat=repeat + 1, initial_hash=initial_hash,
                    initial_energy=initial_energy, admissible_cuts=admissible,
                    seconds=seconds, energy=result.energy, physical_energy=physical_energy,
                    physical_residual=physical_residual, sweeps=result.sweeps, converged=result.converged,
                    message=result.message, energy_error=None if exact_energy is None else result.energy - exact_energy,
                    target_sweep=None if hit is None else hit["sweep"],
                    target_seconds=None if hit is None else hit["elapsed_seconds"],
                    canonical_environment_reuses=result.canonical_environment_reuses,
                    canonical_metric_hits=result.canonical_metric_hits, history=history,
                )
                report["runs"].append(record)
                error = "n/a" if record["energy_error"] is None else f"{record['energy_error']:.2e}"
                print(f"D={dimension:<3} {mode:8} repeat={repeat+1} E={result.energy:.12f} "
                      f"error={error:>10} sweeps={result.sweeps:2} conv={result.converged} "
                      f"time={seconds:.3f}s compact={result.canonical_metric_hits}", flush=True)
    return report


def save_report(report, output, *, plot=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ("bond_dim", "gauge", "repeat", "sweep", "direction", "energy", "energy_error",
              "energy_density_change", "elapsed_seconds", "pass_seconds", "compact_updates",
              "total_updates", "accepted_updates", "hamiltonian_applications")
    with (output / "sweeps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in report["runs"]:
            for step in run["history"]:
                row = dict(run, **step)
                writer.writerow({key: row[key] for key in fields})
    if plot:
        save_plot(report, output / "convergence.png")
    print(f"Saved results to {output.resolve()}")


def save_plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dimensions = report["config"]["bond_dims"]
    fig, axes = plt.subplots(len(dimensions), 3, figsize=(13, 3.4 * len(dimensions)), squeeze=False)
    exact = report["exact_energy"]
    for row, dimension in enumerate(dimensions):
        # Use a real complete trajectory from the median-runtime repetition.
        for mode, color in (("qr", "tab:blue"), ("frontier", "tab:orange")):
            runs = sorted((r for r in report["runs"] if r["bond_dim"] == dimension and r["gauge"] == mode),
                          key=lambda r: r["seconds"])
            run = runs[len(runs) // 2]
            steps = run["history"]
            x = [0] + [s["sweep"] for s in steps]
            t = [0.] + [s["elapsed_seconds"] for s in steps]
            energy = np.array([run["initial_energy"]] + [s["energy"] for s in steps])
            y = energy if exact is None else np.maximum(abs(energy - exact), 1e-14)
            for column, horizontal in ((0, x), (1, t)):
                axes[row, column].plot(horizontal, y, ".-", color=color, label=mode)
                if exact is not None:
                    axes[row, column].set_yscale("log")
                    axes[row, column].axhline(report["config"]["target_error"], color="gray", linestyle=":")
            axes[row, 2].semilogy(x[1:], np.maximum([s["energy_density_change"] for s in steps], 1e-16),
                                  ".-", color=color, label=mode)
        axes[row, 0].set_ylabel(f"D={dimension}\n" + ("Energy" if exact is None else "|E - exact E|"))
        axes[row, 0].set_xlabel("Directional sweep")
        axes[row, 1].set_xlabel("Elapsed solver time (s)")
        axes[row, 2].set_xlabel("Directional sweep")
        axes[row, 2].set_ylabel("|sweep energy change| / sites")
        axes[row, 2].axhline(report["config"]["tolerance"], color="gray", linestyle=":")
        for ax in axes[row]:
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle("QR vs frontier: median-runtime repetition for each mode and D\n"
                 "Time includes initialization; error display floor 1e-14, change floor 1e-16")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "columns", "seed", "max_sweeps", "repeats", "exact_max_dimension", "eigensolver_max_iterations"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    for name in ("field", "tolerance", "metric_tolerance", "eigensolver_tolerance", "target_error"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name])
    parser.add_argument("--bond-dims", nargs="+", type=int, default=DEFAULTS["bond_dims"])
    parser.add_argument("--output", type=Path, default=DEFAULTS["output"])
    parser.add_argument("--no-plot", action="store_true")
    config = vars(parser.parse_args(argv))
    plot = not config.pop("no_plot")
    positive = ("rows", "columns", "max_sweeps", "repeats", "eigensolver_max_iterations",
                "tolerance", "metric_tolerance", "eigensolver_tolerance", "target_error")
    if any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive) or any(d <= 0 for d in config["bond_dims"]):
        parser.error("dimensions, counts, and tolerances must be positive and finite")
    if config["exact_max_dimension"] < 0 or not np.isfinite(config["field"]):
        parser.error("exact-max-dimension must be nonnegative and field must be finite")
    report = run_comparison(config)
    save_report(report, config["output"], plot=plot)
    return report


if __name__ == "__main__":
    main()
