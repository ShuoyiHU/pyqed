#!/usr/bin/env python3
"""Plan independent large LETTA jobs, run one solver, and collect partial results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

MODELS = ("ising", "heisenberg", "bose_hubbard", "fermi_hubbard")
SOLVERS = ("one_site", "cbe", "two_site")
SHAPES = "3x3 3x6 3x9 4x4 4x8 4x12 5x5 5x10 6x6 6x12 7x7 8x8 9x9"
DEFAULT_REPO = "/share/home/gubingLab/hushuoyi/software/pyqed_bg_letta_cbe"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def memory_estimate(model, shape, bond_dim):
    """Exact dense frontier storage; working-space estimate is heuristic.

    Count label incidences without allocating model tensors. The production
    product-term MPO retains one channel per term and dense frontier arrays.
    These estimates are for current real, positive-neighbor 2D benchmarks.
    """
    rows, columns = shape
    n = rows * columns
    d = {"ising": 2, "heisenberg": 2, "bose_hubbard": 3, "fermi_hubbard": 4}[model]
    edges = rows * (columns - 1) + columns * (rows - 1)
    width = {"ising": 1, "heisenberg": 3, "bose_hubbard": 2, "fermi_hubbard": 4}[model] * edges + n
    incidence = [{i} for i in range(n)]
    for i in range(n):
        r, c = divmod(i, columns)
        for j in ([i + 1] if c + 1 < columns else []) + ([i + columns] if r + 1 < rows else []):
            incidence[j].add(i)
    cuts = [sum(min(sites) < cut <= max(sites) for sites in incidence) for cut in range(1, n)]
    h_sizes = [8 * bond_dim**2 * width * d**(2 * count) for count in cuts]
    n_sizes = [8 * bond_dim**2 * d**count for count in cuts]
    mpo_bytes = 8 * d**2 * (2 * width + (n - 2) * width**2)
    saved_bytes = sum(h_sizes) + sum(n_sizes) + mpo_bytes
    # Extra live boundaries, stacked channels, MPO construction, and local
    # eigensolver/ALS/CBE buffers are not all represented by saved_bytes.
    suggested_bytes = 3 * saved_bytes + 8 * 1024**3
    return dict(max_frontier_sites=max(cuts), mpo_channels=width,
                largest_h_environment_gib=max(h_sizes) / 1024**3,
                saved_environment_and_mpo_gib=saved_bytes / 1024**3,
                suggested_memory_gib=suggested_bytes / 1024**3,
                note="Heuristic, not a peak-RSS guarantee; local CBE/two-site temporaries can exceed it.")


def make_plan(args):
    shapes = []
    for text in args.shapes:
        shape = tuple(int(x) for x in text.lower().split("x"))
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError(f"invalid 2D shape: {text}")
        shapes.append(shape)
    if min(args.bond_dims) < 1 or min(args.max_sweeps, args.two_site_max_sweeps) < 1:
        raise ValueError("bond dimensions and sweep limits must be positive")
    if args.tolerance <= 0 or args.memory_gib <= 0:
        raise ValueError("tolerance and memory must be positive")
    tasks = []
    for model in dict.fromkeys(args.models):
        for requested in dict.fromkeys(shapes):
            # Rotate isotropic rectangles so C-order chains have smaller width.
            shape = tuple(sorted(requested, reverse=True))
            for bond in dict.fromkeys(args.bond_dims):
                for seed in dict.fromkeys(args.seeds):
                    case = f"{model}_{requested[0]}x{requested[1]}_D{bond}_seed{seed}"
                    for solver in dict.fromkeys(args.solvers):
                        tasks.append(dict(index=len(tasks), case=case, model=model,
                            requested_shape=requested, shape=shape, bond_dim=bond, seed=seed,
                            solver=solver, max_sweeps=(args.two_site_max_sweeps if solver == "two_site" else args.max_sweeps),
                            tolerance=args.tolerance, memory=memory_estimate(model, shape, bond)))
    plan = dict(schema=1, created=datetime.now(timezone.utc).isoformat(),
                repo=str(Path(args.repo).resolve()), memory_gib=args.memory_gib,
                output_dir=str(args.output.parent.resolve() / "results"), tasks=tasks)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace an existing plan: {args.output}")
    save_json(args.output, plan)
    over = sum(t["memory"]["suggested_memory_gib"] > args.memory_gib for t in tasks)
    print(f"{len(tasks)} tasks; {over} exceed the {args.memory_gib:g} GiB memory estimate.")
    print(f"Plan: {args.output}")
    print(f"Array indices: 0-{len(tasks)-1} (no concurrency cap)")


def load_runtime(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    import pyqed
    Path(pyqed.__file__).resolve().relative_to(repo)
    import numpy, scipy, opt_einsum
    from pyqed._letta_one_site_opt import LatticeLETTA, LETTADMROptions, letta_dmrg
    from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
    from pyqed._letta_one_site_opt.benchmarks import condensed_runner
    # Editable installations can silently supply missing submodules even when
    # pyqed.__file__ points into this bundle. Check every loaded submodule.
    outside = {}
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if (name == "pyqed" or name.startswith("pyqed.")) and origin:
            if not Path(origin).resolve().is_relative_to(repo):
                outside[name] = origin
    if outside:
        raise ImportError(f"pyqed imports escaped the selected source {repo}: {outside}")
    return dict(python=sys.version, executable=sys.executable, pyqed=pyqed.__file__,
                numpy=numpy.__version__, scipy=scipy.__version__, opt_einsum=opt_einsum.__version__)


def run_task(args):
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.task_index < len(plan["tasks"]):
        raise ValueError("task index is outside the plan")
    task = plan["tasks"][args.task_index]
    target = Path(plan["output_dir"]) / task["case"] / (task["solver"] + ".json")
    if target.exists() and not args.force:
        old = json.loads(target.read_text())
        if old.get("status") == "completed":
            print(f"Already completed: {target}")
            return 0
        raise FileExistsError(f"existing incomplete result: {target}; use --force to retry")
    report = dict(task=task, status="starting", started=datetime.now(timezone.utc).isoformat(),
                  host=platform.node(), job_id=os.getenv("SLURM_JOB_ID"),
                  array_task_id=os.getenv("SLURM_ARRAY_TASK_ID"),
                  plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    save_json(target, report)
    started = time.perf_counter()
    try:
        budget = args.memory_gib or plan["memory_gib"]
        if not args.allow_large_memory and task["memory"]["suggested_memory_gib"] > budget:
            report.update(status="resource_blocked", message=(
                f"Estimated working memory {task['memory']['suggested_memory_gib']:.1f} GiB exceeds "
                f"{budget:g} GiB. Increase MEMORY_GIB or set ALLOW_LARGE_MEMORY=1 to attempt anyway."))
            return 2
        report["runtime"] = load_runtime(plan["repo"])
        manifest = Path(plan["repo"]) / "SYNC_MANIFEST.json"
        if manifest.exists():
            report["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        from pyqed._letta_one_site_opt import LETTADMROptions, letta_dmrg
        from pyqed._letta_two_site_opt import LETTATwoSiteOptions, letta_two_site_dmrg
        from pyqed._letta_one_site_opt.benchmarks.condensed_models import build_model
        from pyqed._letta_one_site_opt.benchmarks.condensed_runner import (
            make_shared_initial_state, _letta_one_site_record, _letta_two_site_record)
        print(json.dumps(task, indent=2), flush=True)
        model = build_model(task["model"], dimension="2d", size=task["shape"])
        initial = make_shared_initial_state(model, bond_dim=task["bond_dim"], seed=task["seed"])
        report.update(status="running", parameters=dict(model.parameters),
                      initial_state_fingerprint=initial.fingerprint, initial_energy=initial.energy)
        save_json(target, report)
        common = dict(max_sweeps=task["max_sweeps"], tolerance=task["tolerance"],
                      eigensolver_tolerance=1e-10, eigensolver_max_iterations=300, verbosity=1)
        solve_started = time.perf_counter()
        if task["solver"] == "two_site":
            result = letta_two_site_dmrg(model.mpo, state=initial.letta, bond_dim=task["bond_dim"],
                options=LETTATwoSiteOptions(**common, split_method="metric-als-energy"))
            record = _letta_two_site_record(result, time.perf_counter()-solve_started, initial.fingerprint, None)
        else:
            cbe = task["solver"] == "cbe"
            result = letta_dmrg(model.mpo, state=initial.letta,
                options=LETTADMROptions(**common, cbe_enabled=cbe, cbe_selector="shrewd",
                                       cbe_expansion_dimension=1))
            record = _letta_one_site_record(result, "letta_cbe_strict" if cbe else "letta_one_site",
                time.perf_counter()-solve_started, initial.fingerprint, None)
        report.update(status="completed", record=record)
        print(f"Energy: {result.energy:.14g}; converged={result.converged}; sweeps={result.sweeps}", flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["max_rss_gib"] = rss / (1024**3 if sys.platform == "darwin" else 1024**2)
        save_json(target, report)
        print(f"Status: {report['status']}; result: {target}", flush=True)


def collect(args):
    plan = json.loads(args.plan.read_text())
    rows, fingerprints = [], {}
    for task in plan["tasks"]:
        path = Path(plan["output_dir"]) / task["case"] / (task["solver"] + ".json")
        r = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        record = r.get("record", {})
        row = dict(case=task["case"], solver=task["solver"], status=r["status"],
                   energy=record.get("energy"), converged=record.get("converged"),
                   sweeps=record.get("sweeps"), seconds=record.get("elapsed_seconds"),
                   max_rss_gib=r.get("max_rss_gib"), result=str(path))
        if r.get("initial_state_fingerprint"):
            fingerprints.setdefault(task["case"], set()).add(r["initial_state_fingerprint"])
        rows.append(row)
    for row in rows:
        baseline = next((r["energy"] for r in rows if r["case"] == row["case"] and r["solver"] == "one_site"), None)
        row["energy_minus_one_site"] = (row["energy"] - baseline if row["energy"] is not None and baseline is not None else None)
        row["initial_states_match"] = len(fingerprints.get(row["case"], ())) <= 1
    output = args.plan.parent / "summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {output}; {sum(r['status']=='completed' for r in rows)}/{len(rows)} completed.")
    print("starting/running after Slurm termination means incomplete; inspect sacct and logs for OOM/native failures.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo", default=os.getenv("PYQED_REPO", DEFAULT_REPO))
    p.add_argument("--shapes", nargs="+", default=SHAPES.split())
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--solvers", nargs="+", choices=SOLVERS, default=SOLVERS)
    p.add_argument("--bond-dims", nargs="+", type=int, default=[4, 8])
    p.add_argument("--seeds", nargs="+", type=int, default=[731, 732])
    p.add_argument("--max-sweeps", type=int, default=100)
    p.add_argument("--two-site-max-sweeps", type=int, default=100)
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--memory-gib", type=float, default=256)
    p.set_defaults(function=make_plan)
    p = sub.add_parser("preflight")
    p.add_argument("--repo", default=os.getenv("PYQED_REPO", DEFAULT_REPO))
    p.set_defaults(function=lambda a: print(json.dumps(load_runtime(a.repo), indent=2)))
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--memory-gib", type=float)
    p.add_argument("--allow-large-memory", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(function=run_task)
    p = sub.add_parser("collect")
    p.add_argument("--plan", type=Path, required=True)
    p.set_defaults(function=collect)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    sys.exit(main())
