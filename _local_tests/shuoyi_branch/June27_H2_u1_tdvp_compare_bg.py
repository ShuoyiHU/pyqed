#!/usr/bin/env python3
"""Run current-branch GDVR U(1) TDVP and compare against the saved BG result."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqed
from pyqed.qchem.gdvr.gdvr_mean_field import Molecule
from pyqed.qchem.gdvr.gdvr_tdmps import GDVRTDMPS


RUN_LABEL = "June27_H2_shuoyi_u1_tdvp"
OUTPUT_ROOT = SCRIPT_PATH.with_name("benchmark_results")
DEFAULT_BG_ROOT = (
    REPO_ROOT
    / "_local_tests"
    / "bg_branch"
    / "tddmrg"
    / "June_27th_bg"
    / "benchmark_results"
)

H_BASIS_CFG = {
    "s": [
        35.52322122,
        6.513143725,
        1.822142904,
        0.625955266,
        0.243076747,
        0.100112428,
    ]
}
TDVP_CUTOFF = 1.0e-10
TDVP_KRYLOV_DIM = 15
TDVP_KRYLOV_TOL = 1.0e-10


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def git_text(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def latest_bg_run(root):
    root = Path(root)
    candidates = sorted(root.rglob("benchmark_data.npz"))
    if not candidates:
        return None
    return candidates[-1].parent


def load_bg_setup(bg_dir):
    bg_dir = Path(bg_dir)
    setup_path = bg_dir / "setup.json"
    if not setup_path.exists():
        raise FileNotFoundError(f"Missing BG setup file: {setup_path}")
    return json.loads(setup_path.read_text()), bg_dir / "benchmark_data.npz"


class BGField:
    """BG branch finite sin-squared z-field."""

    def __init__(self, E0, omega, cycles):
        self.E0 = float(E0)
        self.omega = float(omega)
        self.cycles = float(cycles)
        self.total_time = self.cycles * 2.0 * np.pi / self.omega

    def __call__(self, time):
        time = float(time)
        if 0.0 <= time <= self.total_time:
            return self.E0 * np.sin(np.pi * time / self.total_time) ** 2 * np.sin(
                self.omega * time
            )
        return 0.0


def compute_acceleration(times, dipole):
    times = np.asarray(times, dtype=float)
    dipole = np.asarray(dipole, dtype=float)
    dt = float(np.mean(np.diff(times)))
    return np.gradient(np.gradient(dipole - np.mean(dipole), dt), dt)


def compute_hhg_from_acceleration(times, acceleration):
    times = np.asarray(times, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    dt = float(np.mean(np.diff(times)))
    omega = 2.0 * np.pi * np.fft.rfftfreq(acceleration.size, d=dt)
    power = np.abs(np.fft.rfft((acceleration - acceleration.mean()) * np.hanning(acceleration.size))) ** 2
    return omega, power


def compare_series(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    delta = current - reference
    delta_flip = -current - reference
    rmse = float(np.sqrt(np.mean(delta * delta)))
    rmse_flip = float(np.sqrt(np.mean(delta_flip * delta_flip)))
    return {
        "rmse": rmse,
        "max_abs": float(np.max(np.abs(delta))),
        "rmse_sign_flipped": rmse_flip,
        "max_abs_sign_flipped": float(np.max(np.abs(delta_flip))),
        "preferred_sign": -1 if rmse_flip < rmse else 1,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Current-branch H2 GDVR U1 TDVP benchmark against saved BG run."
    )
    parser.add_argument("--bg-dir", type=Path, default=None)
    parser.add_argument("--bg-root", type=Path, default=DEFAULT_BG_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dmrg-sweeps", type=int, default=10)
    parser.add_argument("--dmrg-cycles", type=int, default=1)
    parser.add_argument("--post-dmrg-opt-cycles", type=int, default=0)
    parser.add_argument("--progress-flush-interval", type=int, default=25)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bg_dir = args.bg_dir or latest_bg_run(args.bg_root)
    if bg_dir is None:
        raise FileNotFoundError(f"No BG benchmark_data.npz found under {args.bg_root}")
    bg_setup, bg_data_path = load_bg_setup(bg_dir)

    molecule_cfg = bg_setup["molecule"]
    field_cfg = bg_setup["field"]["params"]
    td_cfg = bg_setup["td_params"]
    newton_cfg = bg_setup["newton_params"]
    tddmrg_cfg = bg_setup["tddmrg_params"]

    run_id = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"{RUN_LABEL}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    dynamics_dir = output_dir / "tdvp_u1"

    elements = list(molecule_cfg["elements"])
    coords = np.asarray(molecule_cfg["coords"], dtype=float)
    charges = [1.0 if elem == "H" else float(elem) for elem in elements]
    build_params = dict(molecule_cfg["build_params"])
    Lz = float(build_params["Lz"])
    Nz = int(build_params["Nz"])
    nelec = len(elements)
    spin = 0

    dt = float(td_cfg["dt"])
    steps = int(td_cfg["steps"])
    bond_dim = int(tddmrg_cfg["D"])
    td_bond_dim = int(tddmrg_cfg.get("td_bond_dim", bond_dim))
    field = BGField(
        E0=field_cfg["E0"],
        omega=field_cfg["omega"],
        cycles=field_cfg["cycles"],
    )

    metadata = {
        "run_label": RUN_LABEL,
        "run_id": run_id,
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pyqed_path": pyqed.__file__,
        "environment_note": "Expected environment: pyqed-plot Python 3.9.23",
        "git": {
            "branch": git_text(["branch", "--show-current"]),
            "commit": git_text(["rev-parse", "HEAD"]),
            "short_commit": git_text(["rev-parse", "--short", "HEAD"]),
            "status": git_text(["status", "--short"]),
        },
        "bg_reference": {
            "directory": str(bg_dir),
            "data_file": str(bg_data_path),
            "setup_file": str(Path(bg_dir) / "setup.json"),
        },
        "molecule": {
            "elements": elements,
            "charges": charges,
            "coords": coords,
            "nelec": nelec,
            "spin": spin,
            "build_params": build_params,
            "basis_cfg": H_BASIS_CFG,
        },
        "newton_params": {
            "pre_opt_cycles": int(newton_cfg["max_cycles"]),
            "pre_opt_sweep_iterations": int(newton_cfg["sweep_iterations"]),
            "newton_ridge": float(newton_cfg["ridge"]),
            "newton_trust_step": float(newton_cfg["trust_step"]),
            "newton_trust_radius": float(newton_cfg["trust_radius"]),
        },
        "dmrg_params": {
            "dmrg_cycles": int(args.dmrg_cycles),
            "dmrg_bond_dim": bond_dim,
            "dmrg_sweeps": int(args.dmrg_sweeps),
            "post_dmrg_opt_cycles": int(args.post_dmrg_opt_cycles),
            "abelian_symmetry": True,
        },
        "td_params": {
            "dt": dt,
            "steps": steps,
            "td_bond_dim": td_bond_dim,
            "propagation_method": "tdvp_u1",
            "tdvp_cutoff": TDVP_CUTOFF,
            "tdvp_krylov_dim": TDVP_KRYLOV_DIM,
            "tdvp_krylov_tol": TDVP_KRYLOV_TOL,
            "flush_interval": int(args.progress_flush_interval),
        },
        "field": {
            "name": bg_setup["field"]["name"],
            "params": field_cfg,
            "formula": bg_setup["field"]["formula"],
        },
    }
    write_json(output_dir / "setup.json", metadata)

    code_diff = git_text(["diff", "--", "pyqed", str(SCRIPT_PATH.relative_to(REPO_ROOT))])
    if code_diff:
        (output_dir / "code_diff.patch").write_text(code_diff + "\n")

    print(f"[shuoyi benchmark] BG reference: {bg_data_path}", flush=True)
    print(f"[shuoyi benchmark] Saving current-branch run under: {output_dir}", flush=True)

    started = time.time()
    mol = Molecule(charges, coords, nelec=nelec, spin=spin)
    solver = GDVRTDMPS(
        mol=mol,
        Lz=Lz,
        Nz=Nz,
        basis_cfg=H_BASIS_CFG,
        e_field_func=field,
        D=bond_dim,
        abelian_symmetry=True,
    )
    ground_energy = solver.run_dmrg(
        pre_opt_cycles=int(newton_cfg["max_cycles"]),
        pre_opt_sweep_iterations=int(newton_cfg["sweep_iterations"]),
        newton_ridge=float(newton_cfg["ridge"]),
        newton_trust_step=float(newton_cfg["trust_step"]),
        newton_trust_radius=float(newton_cfg["trust_radius"]),
        dmrg_cycles=int(args.dmrg_cycles),
        dmrg_bond_dim=bond_dim,
        dmrg_sweeps=int(args.dmrg_sweeps),
        post_dmrg_opt_cycles=int(args.post_dmrg_opt_cycles),
        checkpoint_dir=str(output_dir / "ground_state"),
    )

    solver.run(
        dt=dt,
        steps=steps,
        e_ops=[],
        interval=1,
        flush_interval=int(args.progress_flush_interval),
        save_dir=str(dynamics_dir),
        kick_strength=None,
        escape_boundary=None,
        propagation_method="tdvp_u1",
        tdvp_cutoff=TDVP_CUTOFF,
        tdvp_krylov_dim=TDVP_KRYLOV_DIM,
        tdvp_krylov_tol=TDVP_KRYLOV_TOL,
        td_bond_dim=td_bond_dim,
        kick_bond_dim=td_bond_dim,
        collect_step_diagnostics=True,
        plot_population=False,
        save_density_csv=True,
    )

    times_full = np.asarray(solver.times, dtype=float)
    densities_full = np.asarray(solver.densities, dtype=float)
    z_grid = np.asarray(solver.z_grid, dtype=float)
    fields_full = np.asarray([field(t) for t in times_full], dtype=float)
    z_expect_full = densities_full @ z_grid
    dipole_full = -z_expect_full

    sample_times = times_full[1:]
    sample_fields = fields_full[1:]
    sample_z_expect = z_expect_full[1:]
    sample_dipole = dipole_full[1:]
    acceleration = compute_acceleration(sample_times, sample_dipole)
    hhg_omega, hhg_power = compute_hhg_from_acceleration(sample_times, acceleration)

    bg = np.load(bg_data_path, allow_pickle=True)
    bg_times = np.asarray(bg["times"], dtype=float)
    bg_field_z = np.asarray(bg["field_values"], dtype=float)[:, 2]
    bg_mu_z = np.asarray(bg["observables"], dtype=complex)[:, 0].real
    bg_acceleration = np.asarray(bg["acceleration"], dtype=float)
    bg_hhg_power = np.asarray(bg["hhg_power"], dtype=float)

    if sample_times.shape != bg_times.shape or not np.allclose(sample_times, bg_times):
        raise ValueError(
            "Current and BG time grids do not match. "
            f"current={sample_times.shape}, bg={bg_times.shape}"
        )

    min_hhg = min(hhg_power.size, bg_hhg_power.size)
    comparison = {
        "bg_dir": str(bg_dir),
        "elapsed_seconds": time.time() - started,
        "ground_energy": float(ground_energy),
        "time_grid_max_abs_diff": float(np.max(np.abs(sample_times - bg_times))),
        "field_max_abs_diff": float(np.max(np.abs(sample_fields - bg_field_z))),
        "dipole_vs_bg_mu_z": compare_series(bg_mu_z, sample_dipole),
        "z_expect_vs_bg_mu_z": compare_series(bg_mu_z, sample_z_expect),
        "acceleration_vs_bg_force_form_acceleration": compare_series(
            bg_acceleration, acceleration
        ),
        "hhg_power": {
            "rmse": float(np.sqrt(np.mean((hhg_power[:min_hhg] - bg_hhg_power[:min_hhg]) ** 2))),
            "max_abs": float(np.max(np.abs(hhg_power[:min_hhg] - bg_hhg_power[:min_hhg]))),
            "current_max": float(np.max(hhg_power)),
            "bg_max": float(np.max(bg_hhg_power)),
        },
        "final": {
            "current_dipole_z": float(sample_dipole[-1]),
            "current_z_expect": float(sample_z_expect[-1]),
            "bg_mu_z": float(bg_mu_z[-1]),
            "current_acceleration": float(acceleration[-1]),
            "bg_acceleration": float(bg_acceleration[-1]),
        },
    }

    np.savez_compressed(
        output_dir / "shuoyi_benchmark_data.npz",
        times=sample_times,
        times_full=times_full,
        z_grid=z_grid,
        densities=densities_full,
        field_z=sample_fields,
        field_z_full=fields_full,
        dipole_z=sample_dipole,
        dipole_z_full=dipole_full,
        z_expect=sample_z_expect,
        z_expect_full=z_expect_full,
        acceleration=acceleration,
        hhg_omega=hhg_omega,
        hhg_power=hhg_power,
        tdvp_step_diagnostics=np.asarray(getattr(solver, "tdvp_step_diagnostics", []), dtype=object),
    )
    np.savez_compressed(
        output_dir / "comparison_to_bg.npz",
        bg_times=bg_times,
        current_times=sample_times,
        bg_field_z=bg_field_z,
        current_field_z=sample_fields,
        bg_mu_z=bg_mu_z,
        current_dipole_z=sample_dipole,
        current_z_expect=sample_z_expect,
        bg_acceleration=bg_acceleration,
        current_acceleration=acceleration,
        bg_hhg_power=bg_hhg_power,
        current_hhg_power=hhg_power,
        hhg_omega=hhg_omega,
    )
    np.savetxt(
        output_dir / "comparison_timeseries.csv",
        np.column_stack(
            [
                sample_times,
                bg_field_z,
                sample_fields,
                bg_mu_z,
                sample_dipole,
                sample_z_expect,
                bg_acceleration,
                acceleration,
            ]
        ),
        delimiter=",",
        header=(
            "time,bg_field_z,current_field_z,bg_mu_z,current_dipole_z,"
            "current_z_expect,bg_force_form_acceleration,current_dipole_acceleration"
        ),
        comments="",
        fmt="%.16e",
    )
    write_json(output_dir / "comparison_summary.json", comparison)
    print(f"[shuoyi benchmark] Saved current data: {output_dir / 'shuoyi_benchmark_data.npz'}", flush=True)
    print(f"[shuoyi benchmark] Saved comparison: {output_dir / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
