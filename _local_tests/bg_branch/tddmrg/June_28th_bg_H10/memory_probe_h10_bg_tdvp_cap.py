#!/usr/bin/env python3
"""Memory diagnostics for the H10 BG GDVR-TDDMRG CAP run.

This intentionally does not change the solver.  It monkeypatches TDMPS.step at
runtime and writes one diagnostics row after each TDVP step.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import gc
import importlib.util
import json
import os
import pickle
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))


def load_cap_driver():
    candidates = [
        SCRIPT_PATH.with_name("h10_bg_tdvp_cap.py"),
    ]
    pyqed_repo = os.environ.get("PYQED_REPO")
    if pyqed_repo:
        candidates.append(
            Path(pyqed_repo)
            / "_local_tests"
            / "bg_branch"
            / "tddmrg"
            / "June_28th_bg_H10"
            / "h10_bg_tdvp_cap.py"
        )
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("h10_bg_tdvp_cap", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["h10_bg_tdvp_cap"] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError(
        "Could not find h10_bg_tdvp_cap.py. Put it next to this probe or set "
        "PYQED_REPO to a checkout containing _local_tests/bg_branch/tddmrg/June_28th_bg_H10/h10_bg_tdvp_cap.py."
    )


cap_driver = load_cap_driver()


CSV_FIELDS = [
    "wall_time_s",
    "phase",
    "global_step",
    "local_step",
    "t_au",
    "rss_gb",
    "rss_delta_gb",
    "rss_hwm_gb",
    "ru_maxrss_gb",
    "mps_sites",
    "mps_blocks",
    "mps_elements",
    "mps_bytes_gb",
    "max_block_elements",
    "max_block_bytes_gb",
    "max_block_shape",
    "max_bond_dim",
    "sum_bond_dim",
    "tdvp_engine_cache_size",
    "block_heff_plan_cache_size",
    "block_heff_backend_cache_size",
    "affine_block_sparse_mpo_cache_size",
    "engine_prepared",
    "engine_projection_backend",
    "engine_block_sparse_mpo_cache_size",
    "engine_block_sparse_sector_cache_size",
    "moving_env_environment_plan_records",
    "moving_env_environment_plan_builds",
    "moving_env_environment_plan_replacements",
    "moving_env_environment_plan_cache_hits",
    "moving_env_environment_plan_advance_calls",
    "moving_env_sweep_environment_step_auto_calls",
    "moving_env_environment_plan_last_routes",
    "moving_env_environment_plan_last_blocks",
    "last_pre_norm2",
    "last_tdvp_truncation_error",
    "gc_objects",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=sorted(cap_driver.GEOMETRIES), required=True)
    parser.add_argument("--intensity", choices=sorted(cap_driver.INTENSITIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.05841455452769231)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--drive-amplitude", type=float, default=None)
    parser.add_argument("--newton-cycles", type=int, default=200)
    parser.add_argument("--newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--dmrg-sweeps", type=int, default=20)
    parser.add_argument("--cap-width", type=float, default=cap_driver.CAP_WIDTH)
    parser.add_argument("--cap-strength", type=float, default=cap_driver.CAP_STRENGTH)
    parser.add_argument("--cap-order", type=int, default=cap_driver.CAP_ORDER)
    parser.add_argument("--diag-every", type=int, default=1)
    parser.add_argument("--track-energy", action="store_true")
    parser.add_argument("--fresh-dmrg", action="store_true")
    parser.add_argument("--stop-rss-gb", type=float, default=0.0)
    parser.add_argument("--no-save-final-state", action="store_true")
    parser.add_argument("--krylov-dim", type=int, default=12)
    parser.add_argument("--krylov-tol", type=float, default=1.0e-13)
    parser.add_argument("--krylov-method", default="lanczos")
    return parser.parse_args(argv)


def read_proc_status_memory():
    out = {
        "rss_gb": np.nan,
        "rss_hwm_gb": np.nan,
    }
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            out["rss_gb"] = float(line.split()[1]) / 1024.0 / 1024.0
        elif line.startswith("VmHWM:"):
            out["rss_hwm_gb"] = float(line.split()[1]) / 1024.0 / 1024.0
    return out


def ru_maxrss_gb():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / 1024.0 / 1024.0 / 1024.0
    return value / 1024.0 / 1024.0


def block_nbytes(block):
    try:
        return int(np.asarray(block).nbytes)
    except Exception:
        return 0


def block_nelems(block):
    try:
        return int(np.asarray(block).size)
    except Exception:
        return 0


def mps_stats(psi):
    stats = {
        "mps_sites": 0,
        "mps_blocks": 0,
        "mps_elements": 0,
        "mps_bytes_gb": 0.0,
        "max_block_elements": 0,
        "max_block_bytes_gb": 0.0,
        "max_block_shape": "",
        "max_bond_dim": 0,
        "sum_bond_dim": 0,
    }
    factors = getattr(psi, "factors", None)
    if factors is None:
        factors = getattr(psi, "Bs", None)
    if factors is None:
        return stats
    stats["mps_sites"] = len(factors)

    max_block_bytes = 0
    left_dims = []
    right_dims = []
    for factor in factors:
        if hasattr(factor, "data") and isinstance(getattr(factor, "data"), dict):
            stats["mps_blocks"] += len(factor.data)
            for block in factor.data.values():
                arr = np.asarray(block)
                nbytes = int(arr.nbytes)
                nelems = int(arr.size)
                stats["mps_elements"] += nelems
                if nelems > stats["max_block_elements"]:
                    stats["max_block_elements"] = nelems
                    stats["max_block_shape"] = "x".join(str(x) for x in arr.shape)
                max_block_bytes = max(max_block_bytes, nbytes)
            qns = getattr(factor, "qns", None)
            if qns is not None and len(qns) >= 2:
                left_dims.append(len(qns[0]))
                right_dims.append(len(qns[1]))
            continue

        arr = np.asarray(factor)
        nbytes = int(arr.nbytes)
        nelems = int(arr.size)
        stats["mps_blocks"] += 1
        stats["mps_elements"] += nelems
        if nelems > stats["max_block_elements"]:
            stats["max_block_elements"] = nelems
            stats["max_block_shape"] = "x".join(str(x) for x in arr.shape)
        max_block_bytes = max(max_block_bytes, nbytes)
        if arr.ndim >= 3:
            left_dims.append(int(arr.shape[0]))
            right_dims.append(int(arr.shape[-1]))

    stats["mps_bytes_gb"] = sum(
        sum(block_nbytes(block) for block in factor.data.values())
        if hasattr(factor, "data") and isinstance(getattr(factor, "data"), dict)
        else block_nbytes(factor)
        for factor in factors
    ) / 1024.0**3
    stats["max_block_bytes_gb"] = max_block_bytes / 1024.0**3
    bond_dims = left_dims + right_dims
    if bond_dims:
        stats["max_bond_dim"] = int(max(bond_dims))
        stats["sum_bond_dim"] = int(sum(bond_dims))
    return stats


def tdvp_global_cache_stats():
    try:
        import pyqed.mps.tdvp as tdvp_mod
    except Exception:
        return {}
    return {
        "block_heff_plan_cache_size": len(getattr(tdvp_mod, "_BLOCK_HEFF_PLAN_CACHE", {})),
        "block_heff_backend_cache_size": len(getattr(tdvp_mod, "_BLOCK_HEFF_BACKEND_DECISION_CACHE", {})),
        "affine_block_sparse_mpo_cache_size": len(getattr(tdvp_mod, "_AFFINE_BLOCK_SPARSE_MPO_CACHE", {})),
    }


def engine_stats(tdmps):
    stats = {
        "tdvp_engine_cache_size": len(getattr(tdmps, "_tdvp_engine_cache", {}) or {}),
        "engine_prepared": "",
        "engine_projection_backend": "",
        "engine_block_sparse_mpo_cache_size": 0,
        "engine_block_sparse_sector_cache_size": 0,
        "moving_env_environment_plan_records": np.nan,
        "moving_env_environment_plan_builds": np.nan,
        "moving_env_environment_plan_replacements": np.nan,
        "moving_env_environment_plan_cache_hits": np.nan,
        "moving_env_environment_plan_advance_calls": np.nan,
        "moving_env_sweep_environment_step_auto_calls": np.nan,
        "moving_env_environment_plan_last_routes": np.nan,
        "moving_env_environment_plan_last_blocks": np.nan,
    }
    cache = getattr(tdmps, "_tdvp_engine_cache", {}) or {}
    if not cache:
        return stats
    engine = next(reversed(cache.values()))
    stats["engine_prepared"] = bool(getattr(engine, "_prepared", False))
    stats["engine_projection_backend"] = str(getattr(engine, "projection_backend", ""))
    stats["engine_block_sparse_mpo_cache_size"] = len(getattr(engine, "_block_sparse_mpo_cache", {}) or {})
    stats["engine_block_sparse_sector_cache_size"] = len(getattr(engine, "_block_sparse_sector_cache", {}) or {})
    env = getattr(engine, "_block_sparse_cpp_moving_environment", None)
    if env is not None and hasattr(env, "stats"):
        try:
            env_stats = dict(env.stats())
        except Exception:
            env_stats = {}
        for key in list(stats):
            if key.startswith("moving_env_"):
                raw_key = key[len("moving_env_") :]
                if raw_key in env_stats:
                    stats[key] = env_stats[raw_key]
    return stats


class StepDiagnostics:
    def __init__(self, output_dir, *, diag_every=1, stop_rss_gb=0.0, dt=0.1, t0=0.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "memory_step_diagnostics.csv"
        self.jsonl_path = self.output_dir / "memory_step_diagnostics.jsonl"
        self.diag_every = max(1, int(diag_every))
        self.stop_rss_gb = float(stop_rss_gb or 0.0)
        self.dt = float(dt)
        self.t0 = float(t0)
        self.started = time.time()
        self.global_step = 0
        self.last_rss_gb = None
        self.csv_file = self.csv_path.open("w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.jsonl_file = self.jsonl_path.open("w")

    def close(self):
        self.csv_file.close()
        self.jsonl_file.close()

    def row(self, phase, psi=None, tdmps=None, local_step=None):
        mem = read_proc_status_memory()
        rss = mem["rss_gb"]
        delta = np.nan if self.last_rss_gb is None or not np.isfinite(rss) else rss - self.last_rss_gb
        if np.isfinite(rss):
            self.last_rss_gb = rss
        row = {
            "wall_time_s": time.time() - self.started,
            "phase": phase,
            "global_step": self.global_step,
            "local_step": "" if local_step is None else int(local_step),
            "t_au": self.t0 + self.global_step * self.dt,
            "rss_gb": rss,
            "rss_delta_gb": delta,
            "rss_hwm_gb": mem["rss_hwm_gb"],
            "ru_maxrss_gb": ru_maxrss_gb(),
            "gc_objects": len(gc.get_objects()),
        }
        row.update(mps_stats(psi) if psi is not None else mps_stats(None))
        row.update(tdvp_global_cache_stats())
        row.update(engine_stats(tdmps) if tdmps is not None else {})
        for field in CSV_FIELDS:
            row.setdefault(field, "")
        self.writer.writerow(row)
        self.csv_file.flush()
        self.jsonl_file.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self.jsonl_file.flush()
        print(
            "[MEMPROBE] "
            f"step={row['global_step']} phase={phase} "
            f"rss={row['rss_gb']:.3f}GB d_rss={row['rss_delta_gb']:.3f}GB "
            f"blocks={row['mps_blocks']} elems={row['mps_elements']} "
            f"max_block={row['max_block_elements']} "
            f"env_records={row.get('moving_env_environment_plan_records', '')}",
            flush=True,
        )
        if self.stop_rss_gb > 0.0 and np.isfinite(rss) and rss >= self.stop_rss_gb:
            raise MemoryError(f"RSS {rss:.3f} GB reached --stop-rss-gb={self.stop_rss_gb:.3f}")


def install_step_probe(diag):
    from pyqed.mps.tdmps import TDMPS

    original_step = TDMPS.step

    def wrapped_step(self, psi, *args, **kwargs):
        out = original_step(self, psi, *args, **kwargs)
        diag.global_step += 1
        if diag.global_step % diag.diag_every == 0:
            diag.row("after_step", psi=out, tdmps=self, local_step=diag.global_step)
        return out

    TDMPS.step = wrapped_step
    return original_step


def restore_step_probe(original_step):
    from pyqed.mps.tdmps import TDMPS

    TDMPS.step = original_step


def load_or_build_ground_state(args, output_dir, grid, geometry, atom_z, coords):
    build_params = {"Lz": grid["lz"], "Nz": grid["nz"], "M": 1, "verbose": False}
    rhf_params = {"conv": 1.0e-8, "verbose": False}
    newton_params = {
        "max_cycles": int(args.newton_cycles),
        "sweep_iterations": 1,
        "tol": float(args.newton_tol),
        "ridge": 0.5,
        "trust_step": 0.5,
        "trust_radius": 1.0,
        "verbose": True,
    }
    dmrg_params = {
        "D": int(args.bond_dim),
        "nsweeps": int(args.dmrg_sweeps),
        "symmetry_list": ["charge", "sz"],
        "not_conv_err": False,
    }

    mol = cap_driver.AtomicChain(["H"] * len(atom_z), coords=coords)
    mol.build(**build_params)
    mf = mol.RHF().run(**rhf_params)
    print(f"[MEMPROBE] RHF energy = {cap_driver.safe_float(mf.e_tot):.12f} Ha", flush=True)
    mf.newton(**newton_params)
    print(
        f"[MEMPROBE] HF-TOO final = {cap_driver.safe_float(mf.e_tot):.12f} Ha "
        f"converged={mf.info.get('newton_converged')}",
        flush=True,
    )

    td = mf.TDDMRG().build()
    td.verbose = max(int(getattr(td, "verbose", 0) or 0), 1)
    cache_metadata = cap_driver.make_dmrg_cache_metadata(
        args,
        geometry,
        atom_z,
        grid,
        build_params,
        newton_params,
        dmrg_params,
    )
    cache_root = output_dir.parent / "dmrg_ground_states"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_state = cache_root / f"{cache_metadata['cache_key']}.pkl"
    cache_json = cache_root / f"{cache_metadata['cache_key']}.json"

    payload = None if args.fresh_dmrg else cap_driver.load_dmrg_cache(cache_state, cache_json, cache_metadata)
    if payload is None:
        print(
            f"[MEMPROBE] running static DMRG D={dmrg_params['D']} nsweeps={dmrg_params['nsweeps']}",
            flush=True,
        )
        td.optimize_ground_state(
            D=dmrg_params["D"],
            nsweeps=dmrg_params["nsweeps"],
            symmetry_list=dmrg_params["symmetry_list"],
            initial_guess=td.init_guess,
            not_conv_err=dmrg_params["not_conv_err"],
        )
        psi = td.dmrg.ground_state.copy()
        dmrg_energy = cap_driver.safe_float(getattr(td, "e_tot", None))
        payload = cap_driver.save_dmrg_cache(cache_state, cache_json, cache_metadata, psi, dmrg_energy)
        print(f"[MEMPROBE] saved DMRG cache {cache_state}", flush=True)
    else:
        print(f"[MEMPROBE] loaded DMRG cache {cache_state}", flush=True)

    return mol, mf, td, payload["psi"].copy(), cap_driver.safe_float(payload.get("dmrg_energy"))


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = cap_driver.GEOMETRIES[args.geometry]
    intensity = cap_driver.INTENSITIES[args.intensity]
    amplitude = float(args.drive_amplitude) if args.drive_amplitude is not None else float(intensity["drive_amplitude"])
    grid = cap_driver.expanded_grid(cap_width=args.cap_width)
    field, total_time = cap_driver.make_field(amplitude, args.omega, args.cycles)
    steps = int(args.steps) if args.steps is not None else int(np.ceil(total_time / float(args.dt)))
    cap_settings = {
        "width": float(args.cap_width),
        "strength": float(args.cap_strength),
        "order": int(args.cap_order),
    }
    atom_z = np.asarray(geometry["atom_z"], dtype=float)
    coords = [(0.0, 0.0, float(z)) for z in atom_z]

    setup = {
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": sys.version,
        "pyqed_path": cap_driver.pyqed.__file__,
        "geometry": args.geometry,
        "intensity": args.intensity,
        "amplitude": amplitude,
        "grid": grid,
        "steps": steps,
        "dt": float(args.dt),
        "bond_dim": int(args.bond_dim),
        "diag_every": int(args.diag_every),
        "stop_rss_gb": float(args.stop_rss_gb),
    }
    cap_driver.write_json_atomic(output_dir / "memory_probe_setup.json", setup)

    print("=" * 72, flush=True)
    print("[MEMPROBE] H10 BG TDVP CAP memory probe", flush=True)
    print(f"geometry/intensity: {args.geometry} / {args.intensity}", flush=True)
    print(f"steps/dt/D        : {steps} / {args.dt} / {args.bond_dim}", flush=True)
    print(f"output            : {output_dir}", flush=True)
    print("=" * 72, flush=True)

    diag = StepDiagnostics(
        output_dir,
        diag_every=args.diag_every,
        stop_rss_gb=args.stop_rss_gb,
        dt=args.dt,
    )
    original_step = install_step_probe(diag)
    try:
        mol, mf, td, psi, dmrg_energy = load_or_build_ground_state(
            args,
            output_dir,
            grid,
            geometry,
            atom_z,
            coords,
        )
        diag.row("before_td", psi=psi, tdmps=None)
        print(f"[MEMPROBE] DMRG energy = {cap_driver.format_float(dmrg_energy)} Ha", flush=True)

        td.run(
            psi0=psi,
            D=args.bond_dim,
            dt=args.dt,
            steps=steps,
            e_ops=["mu_z", cap_driver.force_mpo(mol)],
            field=field,
            cap=cap_settings,
            t0=0.0,
            integrator="tdvp",
            tdvp_projection_backend="block-sparse",
            track_energy=args.track_energy,
            progress=True,
            progress_every=max(1, min(10, steps)),
            krylov_dim=args.krylov_dim,
            krylov_tol=args.krylov_tol,
            krylov_method=args.krylov_method,
        )
        final_state = td.final_state.copy()
        diag.row("after_td", psi=final_state, tdmps=getattr(td, "tdmps", None))

        if not args.no_save_final_state:
            with (output_dir / "memory_probe_final_state.pkl").open("wb") as handle:
                pickle.dump(final_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        summary = {
            "completed": True,
            "steps": int(steps),
            "final_rss_gb": read_proc_status_memory()["rss_gb"],
            "final_ru_maxrss_gb": ru_maxrss_gb(),
            "diagnostics_csv": str(diag.csv_path),
            "diagnostics_jsonl": str(diag.jsonl_path),
        }
        cap_driver.write_json_atomic(output_dir / "memory_probe_summary.json", summary)
        print("[MEMPROBE] completed", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except MemoryError as exc:
        summary = {
            "completed": False,
            "stopped_reason": str(exc),
            "steps_logged": int(diag.global_step),
            "rss_gb": read_proc_status_memory()["rss_gb"],
            "ru_maxrss_gb": ru_maxrss_gb(),
            "diagnostics_csv": str(diag.csv_path),
            "diagnostics_jsonl": str(diag.jsonl_path),
        }
        cap_driver.write_json_atomic(output_dir / "memory_probe_summary.json", summary)
        print("[MEMPROBE] stopped early", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 2
    finally:
        restore_step_probe(original_step)
        diag.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
