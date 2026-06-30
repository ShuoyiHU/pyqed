#!/usr/bin/env python3
"""BG-branch H2 TDDMRG test with static DMRG initial state.

This keeps the same molecule, HF Newton optimization, field, TD step, bond
dimension, and observables as the saved BG benchmark, but it runs static DMRG
before time propagation and explicitly passes the converged DMRG state to
``td.run``.  The output is compared against the latest saved Shuoyi-branch U1
TDVP benchmark.
"""

from __future__ import annotations

import datetime as _datetime
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqed
from pyqed.qchem.gdvr import AtomicChain, force_mpo


RUN_LABEL = "June27_H2_bg_dmrg_init"
OUTPUT_ROOT = SCRIPT_PATH.with_name("benchmark_results")
DEFAULT_SHUOYI_ROOT = REPO_ROOT / "_local_tests" / "shuoyi_branch" / "benchmark_results"

ELEMENTS = ["H"] * 2
COORDS = [(0.0, 0.0, (i - 0.5) * 0.8) for i in range(2)]
BUILD_PARAMS = {"Lz": 5.0, "Nz": 16, "M": 1, "verbose": False}
RHF_PARAMS = {"conv": 1e-8, "verbose": False}
NEWTON_PARAMS = {
    "max_cycles": 10,
    "sweep_iterations": 1,
    "tol": 1e-7,
    "ridge": 0.5,
    "trust_step": 0.5,
    "trust_radius": 1.0,
    "verbose": True,
}
TDDMRG_PARAMS = {}
DMRG_PARAMS = {
    "D": 16,
    "nsweeps": 10,
    "symmetry_list": ["charge", "sz"],
    "initial_guess": "current_init_guess",
    "not_conv_err": False,
}
FIELD_PARAMS = {"E0": 0.08, "omega": 0.057, "cycles": 2}
TD_PARAMS = {
    "D": 16,
    "dt": 0.5,
    "integrator": "tdvp",
    "tdvp_projection_backend": "block-sparse",
    "track_energy": False,
    "progress": True,
}
CAP_PARAMS = None
# Example CAP settings supported by the current BG branch:
# CAP_PARAMS = {"width": 1.0, "strength": 0.01, "order": 2}


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


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def load_shuoyi_energy_log(shuoyi_dir):
    ground_dir = Path(shuoyi_dir) / "ground_state"
    final_meta = ground_dir / "04_DMRG_Final_meta.pkl"
    if not final_meta.exists():
        candidates = sorted(ground_dir.glob("*_meta.pkl"))
        final_meta = candidates[-1] if candidates else None
    if final_meta is None or not final_meta.exists():
        return {
            "source_file": None,
            "hf_initial": None,
            "hf_newton_history_excluding_initial": [],
            "hf_newton_history_including_initial": [],
            "hf_newton_final": None,
            "hf_too_energy": None,
            "dmrg_cycles": [],
            "dmrg_final": None,
            "final_overlap": None,
        }

    with final_meta.open("rb") as handle:
        meta = pickle.load(handle)
    log = dict(meta.get("log", {}))
    hf_initial = safe_float(log.get("hf_initial"))
    hf_pre_opt = [safe_float(x) for x in log.get("hf_pre_opt", [])]
    hf_pre_opt = [x for x in hf_pre_opt if x is not None]
    dmrg_cycles = []
    for item in log.get("dmrg_cycles", []):
        dmrg_cycles.append(
            {
                "cycle": int(item.get("cycle", len(dmrg_cycles))),
                "e_dmrg": safe_float(item.get("e_dmrg")),
                "ao_opt": bool(item.get("ao_opt", False)),
            }
        )
    dmrg_values = [x["e_dmrg"] for x in dmrg_cycles if x["e_dmrg"] is not None]
    return {
        "source_file": str(final_meta),
        "hf_initial": hf_initial,
        "hf_newton_history_excluding_initial": hf_pre_opt,
        "hf_newton_history_including_initial": (
            [hf_initial] + hf_pre_opt if hf_initial is not None else list(hf_pre_opt)
        ),
        "hf_newton_final": hf_pre_opt[-1] if hf_pre_opt else hf_initial,
        "hf_too_energy": hf_pre_opt[-1] if hf_pre_opt else hf_initial,
        "dmrg_cycles": dmrg_cycles,
        "dmrg_final": dmrg_values[-1] if dmrg_values else None,
        "final_overlap": log.get("final_overlap"),
    }


def scalar_delta(current, reference):
    current = safe_float(current)
    reference = safe_float(reference)
    if current is None or reference is None:
        return None
    return current - reference


def energy_difference_summary(bg_log, shuoyi_log):
    bg_hist = bg_log.get("hf_newton_history_including_initial", [])
    shuoyi_hist = shuoyi_log.get("hf_newton_history_including_initial", [])
    n = min(len(bg_hist), len(shuoyi_hist))
    per_cycle = []
    for idx in range(n):
        per_cycle.append(
            {
                "index": idx,
                "bg": bg_hist[idx],
                "shuoyi": shuoyi_hist[idx],
                "bg_minus_shuoyi": scalar_delta(bg_hist[idx], shuoyi_hist[idx]),
            }
        )
    return {
        "hf_initial_bg_minus_shuoyi": scalar_delta(bg_log.get("hf_initial"), shuoyi_log.get("hf_initial")),
        "hf_newton_final_bg_minus_shuoyi": scalar_delta(
            bg_log.get("hf_newton_final"), shuoyi_log.get("hf_newton_final")
        ),
        "hf_too_energy_bg_minus_shuoyi": scalar_delta(
            bg_log.get("hf_too_energy"), shuoyi_log.get("hf_too_energy")
        ),
        "dmrg_final_bg_minus_shuoyi": scalar_delta(bg_log.get("dmrg_final"), shuoyi_log.get("dmrg_final")),
        "newton_history_bg_minus_shuoyi": per_cycle,
    }


def td_energy_audit(td):
    static_energies = np.asarray(getattr(td, "static_energies", []), dtype=complex)
    energy_drift = np.asarray(getattr(td, "energy_drift", []), dtype=complex)
    pre_norm2 = np.asarray(getattr(td, "pre_normalization_norm2", []), dtype=float)
    finite_static = static_energies[np.isfinite(static_energies.real) & np.isfinite(static_energies.imag)]
    finite_drift = energy_drift[np.isfinite(energy_drift.real) & np.isfinite(energy_drift.imag)]
    finite_norm2 = pre_norm2[np.isfinite(pre_norm2) & (pre_norm2 >= 0.0)]
    cumulative_norm2 = np.cumprod(finite_norm2) if finite_norm2.size else np.asarray([], dtype=float)
    return {
        "track_energy": bool(TD_PARAMS.get("track_energy", False)),
        "energy_times": np.asarray(getattr(td, "energy_times", []), dtype=float),
        "static_energies_real": static_energies.real,
        "static_energies_imag": static_energies.imag,
        "energy_drift_real": energy_drift.real,
        "energy_drift_imag": energy_drift.imag,
        "static_energy_real_min": safe_float(np.min(finite_static.real)) if finite_static.size else None,
        "static_energy_real_max": safe_float(np.max(finite_static.real)) if finite_static.size else None,
        "static_energy_imag_min": safe_float(np.min(finite_static.imag)) if finite_static.size else None,
        "static_energy_imag_max": safe_float(np.max(finite_static.imag)) if finite_static.size else None,
        "energy_drift_max_abs": safe_float(np.max(np.abs(finite_drift))) if finite_drift.size else None,
        "pre_normalization_norm2": pre_norm2,
        "cumulative_pre_normalization_norm2": cumulative_norm2,
        "estimated_final_norm2_without_step_renormalization": safe_float(cumulative_norm2[-1])
        if cumulative_norm2.size
        else None,
    }


def latest_shuoyi_run(root=DEFAULT_SHUOYI_ROOT):
    candidates = sorted(Path(root).rglob("shuoyi_benchmark_data.npz"))
    if not candidates:
        return None
    return candidates[-1].parent


def pulse(t, E0=FIELD_PARAMS["E0"], omega=FIELD_PARAMS["omega"], cycles=FIELD_PARAMS["cycles"]):
    total = cycles * 2.0 * np.pi / omega
    field = np.zeros(3)
    if 0.0 <= t <= total:
        field[2] = E0 * np.sin(np.pi * t / total) ** 2 * np.sin(omega * t)
    return field


def compute_acceleration_from_dipole(times, dipole):
    times = np.asarray(times, dtype=float)
    dipole = np.asarray(dipole, dtype=float)
    dt = float(np.mean(np.diff(times)))
    return np.gradient(np.gradient(dipole - dipole.mean(), dt), dt)


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
    ref0 = reference - reference.mean()
    cur0 = current - current.mean()
    den = np.linalg.norm(ref0) * np.linalg.norm(cur0)
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "corr": float(np.dot(ref0, cur0) / den) if den else np.nan,
        "reference_max_abs": float(np.max(np.abs(reference))),
        "current_max_abs": float(np.max(np.abs(current))),
    }


def make_figures(output_dir, shuoyi, bg_dmrg, summary):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}", flush=True)
        return

    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)

    t = bg_dmrg["times"]
    field_z = bg_dmrg["field_values"][:, 2]
    bg_mu = bg_dmrg["mu_z"]
    bg_force_acc = bg_dmrg["force_acceleration"]
    bg_dip_acc = bg_dmrg["dipole_acceleration"]
    shuoyi_dip = shuoyi["dipole_z"]
    shuoyi_acc = shuoyi["acceleration"]
    harm = bg_dmrg["hhg_omega"] / FIELD_PARAMS["omega"]
    mask = harm <= 80.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, field_z, color="0.2", lw=1.2, label="field")
    axes[0].set_ylabel("E_z(t)")
    axes[0].legend(frameon=False)
    axes[1].plot(t, shuoyi_dip, lw=1.1, label="Shuoyi dipole_z")
    axes[1].plot(t, bg_mu, lw=1.1, label="BG+DMRG mu_z")
    axes[1].set_ylabel("dipole / mu_z")
    axes[1].legend(frameon=False)
    axes[2].plot(t, bg_mu - shuoyi_dip, color="tab:red", lw=1.0, label="BG+DMRG - Shuoyi")
    axes[2].axhline(0.0, color="0.5", lw=0.7)
    axes[2].set_xlabel("time (a.u.)")
    axes[2].set_ylabel("difference")
    axes[2].legend(frameon=False)
    fig.suptitle("BG with DMRG initial state vs Shuoyi current branch")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_dipole_trace.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(t, shuoyi_acc, lw=1.1, label="Shuoyi dipole acceleration")
    axes[0].plot(t, bg_force_acc, lw=1.1, label="BG+DMRG force-form acceleration")
    axes[0].plot(t, bg_dip_acc, lw=0.9, ls="--", label="BG+DMRG dipole acceleration")
    axes[0].set_ylabel("acceleration")
    axes[0].legend(frameon=False)
    axes[1].plot(t, bg_force_acc - shuoyi_acc, lw=1.0, color="tab:red", label="force acc. - Shuoyi")
    axes[1].plot(t, bg_dip_acc - shuoyi_acc, lw=0.9, color="tab:purple", label="dipole acc. - Shuoyi")
    axes[1].axhline(0.0, color="0.5", lw=0.7)
    axes[1].set_xlabel("time (a.u.)")
    axes[1].set_ylabel("difference")
    axes[1].legend(frameon=False)
    fig.suptitle("Acceleration comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_acceleration_trace.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    floor = 1e-18
    ax.semilogy(harm[mask], np.maximum(shuoyi["hhg_power"][mask], floor), label="Shuoyi")
    ax.semilogy(harm[mask], np.maximum(bg_dmrg["hhg_power"][mask], floor), label="BG+DMRG force acc.")
    ax.semilogy(
        harm[mask],
        np.maximum(bg_dmrg["dipole_hhg_power"][mask], floor),
        ls="--",
        label="BG+DMRG dipole acc.",
    )
    ax.set_xlabel("harmonic order")
    ax.set_ylabel("power")
    ax.set_title("HHG-like spectrum")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_hhg_spectrum.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axis("off")
    lines = [
        f"dipole RMSE = {summary['dipole_vs_shuoyi']['rmse']:.3e}",
        f"dipole corr = {summary['dipole_vs_shuoyi']['corr']:.4f}",
        f"force acc RMSE = {summary['force_acceleration_vs_shuoyi']['rmse']:.3e}",
        f"force acc corr = {summary['force_acceleration_vs_shuoyi']['corr']:.4f}",
        f"dipole acc RMSE = {summary['dipole_acceleration_vs_shuoyi']['rmse']:.3e}",
        f"dipole acc corr = {summary['dipole_acceleration_vs_shuoyi']['corr']:.4f}",
        f"BG+DMRG HHG max / Shuoyi = {summary['hhg_force_power']['bg_dmrg_max_over_shuoyi_max']:.3e}",
        f"BG+DMRG dipole-HHG max / Shuoyi = {summary['hhg_dipole_power']['bg_dmrg_max_over_shuoyi_max']:.3e}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "00_summary.png", dpi=220)
    plt.close(fig)


def main():
    shuoyi_dir = latest_shuoyi_run()
    if shuoyi_dir is None:
        raise FileNotFoundError(f"No Shuoyi benchmark found under {DEFAULT_SHUOYI_ROOT}")
    shuoyi_data_path = shuoyi_dir / "shuoyi_benchmark_data.npz"

    run_id = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{RUN_LABEL}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    setup = {
        "run_label": RUN_LABEL,
        "run_id": run_id,
        "script": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pyqed_path": pyqed.__file__,
        "git": {
            "branch": git_text(["branch", "--show-current"]),
            "commit": git_text(["rev-parse", "HEAD"]),
            "short_commit": git_text(["rev-parse", "--short", "HEAD"]),
            "status": git_text(["status", "--short"]),
        },
        "shuoyi_reference": {
            "directory": str(shuoyi_dir),
            "data_file": str(shuoyi_data_path),
        },
        "molecule": {
            "elements": ELEMENTS,
            "coords": COORDS,
            "build_params": BUILD_PARAMS,
        },
        "rhf_params": RHF_PARAMS,
        "newton_params": NEWTON_PARAMS,
        "tddmrg_params": TDDMRG_PARAMS,
        "dmrg_params": DMRG_PARAMS,
        "field": {
            "name": "sin2_sine_z",
            "params": FIELD_PARAMS,
            "formula": "E_z(t)=E0*sin(pi*t/T)^2*sin(omega*t), T=cycles*2*pi/omega, for 0<=t<=T",
        },
        "td_params": TD_PARAMS,
        "cap_params": CAP_PARAMS,
        "observable_labels": ["mu_z", "force_mpo"],
    }
    write_json(output_dir / "setup.json", setup)

    code_diff = git_text(["diff", "--", "pyqed", str(SCRIPT_PATH.relative_to(REPO_ROOT))])
    if code_diff:
        (output_dir / "code_diff.patch").write_text(code_diff + "\n")

    print(f"[BG+DMRG] Shuoyi reference: {shuoyi_data_path}", flush=True)
    print(f"[BG+DMRG] Saving output under: {output_dir}", flush=True)

    started_at = time.time()
    mol = AtomicChain(ELEMENTS, coords=COORDS)
    mol.build(**BUILD_PARAMS)
    mf = mol.RHF().run(**RHF_PARAMS)
    hf_initial_energy = safe_float(mf.e_tot)
    mf.newton(**NEWTON_PARAMS)
    hf_newton_history = [safe_float(x) for x in mf.info.get("newton_energy_history", [])]
    hf_newton_history = [x for x in hf_newton_history if x is not None]

    td = mf.TDDMRG(**TDDMRG_PARAMS).build()
    print("[BG+DMRG] Optimizing static DMRG ground state before TD...", flush=True)
    td.optimize_ground_state(
        D=DMRG_PARAMS["D"],
        nsweeps=DMRG_PARAMS["nsweeps"],
        symmetry_list=DMRG_PARAMS["symmetry_list"],
        initial_guess=td.init_guess,
        not_conv_err=DMRG_PARAMS["not_conv_err"],
    )
    dmrg_energy = safe_float(getattr(td, "e_tot", None))
    dmrg_state = td.dmrg.ground_state.copy()
    shuoyi_energy_log = load_shuoyi_energy_log(shuoyi_dir)
    bg_energy_log = {
        "source_file": str(SCRIPT_PATH),
        "hf_initial": hf_initial_energy,
        "hf_newton_history_excluding_initial": hf_newton_history[1:],
        "hf_newton_history_including_initial": hf_newton_history,
        "hf_newton_final": hf_newton_history[-1] if hf_newton_history else safe_float(mf.e_tot),
        "hf_too_energy": hf_newton_history[-1] if hf_newton_history else safe_float(mf.e_tot),
        "newton_cycles": int(mf.info.get("newton_cycles", max(len(hf_newton_history) - 1, 0))),
        "newton_converged": bool(mf.info.get("newton_converged", False)),
        "dmrg_cycles": [
            {
                "cycle": 0,
                "e_dmrg": dmrg_energy,
                "ao_opt": False,
                "nsweeps": DMRG_PARAMS["nsweeps"],
            }
        ],
        "dmrg_final": dmrg_energy,
        "dmrg_active_energy": safe_float(getattr(getattr(td, "dmrg", None), "e_active", None)),
        "dmrg_solver_e_tot": safe_float(getattr(getattr(td, "dmrg", None), "e_tot", None)),
        "e_core": safe_float(getattr(td, "e_core", None)),
    }
    energy_audit = {
        "units": "Hartree",
        "notes": {
            "hf_too_energy": "Alias for the final HF Newton tensor/orbital-optimized energy.",
            "hf_newton_history_including_initial": "Index 0 is the RHF energy before Newton optimization; later indices are Newton cycles.",
        },
        "bg": bg_energy_log,
        "shuoyi_reference": shuoyi_energy_log,
        "differences": energy_difference_summary(bg_energy_log, shuoyi_energy_log),
    }
    write_json(output_dir / "energy_audit.json", energy_audit)

    dt = TD_PARAMS["dt"]
    steps = int(np.ceil((FIELD_PARAMS["cycles"] * 2.0 * np.pi / FIELD_PARAMS["omega"]) / dt))
    setup["td_params"] = {**TD_PARAMS, "steps": steps}
    setup["cap_params"] = CAP_PARAMS
    write_json(output_dir / "setup.json", setup)

    td.run(
        psi0=dmrg_state,
        D=TD_PARAMS["D"],
        dt=dt,
        steps=steps,
        e_ops=["mu_z", force_mpo(mol)],
        field=pulse,
        cap=CAP_PARAMS,
        integrator=TD_PARAMS["integrator"],
        tdvp_projection_backend=TD_PARAMS["tdvp_projection_backend"],
        track_energy=TD_PARAMS["track_energy"],
        progress=TD_PARAMS["progress"],
    )
    energy_audit["td"] = td_energy_audit(td)
    write_json(output_dir / "energy_audit.json", energy_audit)

    times = np.asarray(td.times, dtype=float)
    field_values = np.asarray([pulse(t) for t in times], dtype=float)
    mu_z = np.asarray(td.observables[:, 0].real, dtype=float)
    force_obs = np.asarray(td.observables[:, 1].real, dtype=float)
    force_acceleration = force_obs + mol.nelec * field_values[:, 2]
    dipole_acceleration = compute_acceleration_from_dipole(times, mu_z)
    hhg_omega, hhg_power = compute_hhg_from_acceleration(times, force_acceleration)
    _, dipole_hhg_power = compute_hhg_from_acceleration(times, dipole_acceleration)

    shuoyi = np.load(shuoyi_data_path, allow_pickle=True)
    shuoyi_times = np.asarray(shuoyi["times"], dtype=float)
    if times.shape != shuoyi_times.shape or not np.allclose(times, shuoyi_times):
        raise ValueError(
            f"Time grid mismatch: BG+DMRG {times.shape}, Shuoyi {shuoyi_times.shape}"
        )
    shuoyi_dipole = np.asarray(shuoyi["dipole_z"], dtype=float)
    shuoyi_acc = np.asarray(shuoyi["acceleration"], dtype=float)
    shuoyi_hhg = np.asarray(shuoyi["hhg_power"], dtype=float)

    min_hhg = min(hhg_power.size, shuoyi_hhg.size)
    summary = {
        "elapsed_seconds": time.time() - started_at,
        "dmrg_energy": float(getattr(td, "e_tot", np.nan)),
        "energy_audit_file": str(output_dir / "energy_audit.json"),
        "energy_differences": energy_audit["differences"],
        "cap_params": CAP_PARAMS,
        "cap_settings": getattr(td, "cap_settings", None),
        "steps": steps,
        "dt": dt,
        "time_grid_max_abs_diff": float(np.max(np.abs(times - shuoyi_times))),
        "field_max_abs_diff": float(
            np.max(np.abs(field_values[:, 2] - np.asarray(shuoyi["field_z"], dtype=float)))
        ),
        "dipole_vs_shuoyi": compare_series(shuoyi_dipole, mu_z),
        "force_acceleration_vs_shuoyi": compare_series(shuoyi_acc, force_acceleration),
        "dipole_acceleration_vs_shuoyi": compare_series(shuoyi_acc, dipole_acceleration),
        "hhg_force_power": {
            "rmse": float(np.sqrt(np.mean((hhg_power[:min_hhg] - shuoyi_hhg[:min_hhg]) ** 2))),
            "max_abs": float(np.max(np.abs(hhg_power[:min_hhg] - shuoyi_hhg[:min_hhg]))),
            "bg_dmrg_max": float(np.max(hhg_power)),
            "shuoyi_max": float(np.max(shuoyi_hhg)),
            "bg_dmrg_max_over_shuoyi_max": float(np.max(hhg_power) / max(np.max(shuoyi_hhg), 1e-300)),
        },
        "hhg_dipole_power": {
            "rmse": float(np.sqrt(np.mean((dipole_hhg_power[:min_hhg] - shuoyi_hhg[:min_hhg]) ** 2))),
            "max_abs": float(np.max(np.abs(dipole_hhg_power[:min_hhg] - shuoyi_hhg[:min_hhg]))),
            "bg_dmrg_max": float(np.max(dipole_hhg_power)),
            "shuoyi_max": float(np.max(shuoyi_hhg)),
            "bg_dmrg_max_over_shuoyi_max": float(np.max(dipole_hhg_power) / max(np.max(shuoyi_hhg), 1e-300)),
        },
    }

    bg_dmrg = {
        "times": times,
        "field_values": field_values,
        "mu_z": mu_z,
        "force_observable": force_obs,
        "force_acceleration": force_acceleration,
        "dipole_acceleration": dipole_acceleration,
        "hhg_omega": hhg_omega,
        "hhg_power": hhg_power,
        "dipole_hhg_power": dipole_hhg_power,
    }
    shuoyi_compare = {
        "times": shuoyi_times,
        "dipole_z": shuoyi_dipole,
        "acceleration": shuoyi_acc,
        "hhg_power": shuoyi_hhg,
    }

    np.savez_compressed(
        output_dir / "benchmark_data.npz",
        times=times,
        observables=np.asarray(td.observables),
        fields=np.asarray(getattr(td, "fields", field_values)),
        field_values=field_values,
        mu_z=mu_z,
        force_observable=force_obs,
        acceleration=force_acceleration,
        dipole_acceleration=dipole_acceleration,
        hhg_omega=hhg_omega,
        hhg_power=hhg_power,
        dipole_hhg_power=dipole_hhg_power,
        pre_normalization_norms=np.asarray(getattr(td, "pre_normalization_norms", [])),
        pre_normalization_norm2=np.asarray(getattr(td, "pre_normalization_norm2", [])),
        tdvp_truncation_errors=np.asarray(getattr(td, "tdvp_truncation_errors", [])),
        static_energies=np.asarray(getattr(td, "static_energies", []), dtype=complex),
        energy_drift=np.asarray(getattr(td, "energy_drift", []), dtype=complex),
        cap_params=np.asarray([CAP_PARAMS], dtype=object),
    )
    np.savez_compressed(
        output_dir / "comparison_to_shuoyi.npz",
        shuoyi_times=shuoyi_times,
        bg_dmrg_times=times,
        shuoyi_dipole_z=shuoyi_dipole,
        bg_dmrg_mu_z=mu_z,
        shuoyi_acceleration=shuoyi_acc,
        bg_dmrg_force_acceleration=force_acceleration,
        bg_dmrg_dipole_acceleration=dipole_acceleration,
        shuoyi_hhg_power=shuoyi_hhg,
        bg_dmrg_hhg_power=hhg_power,
        bg_dmrg_dipole_hhg_power=dipole_hhg_power,
        hhg_omega=hhg_omega,
    )
    np.savetxt(
        output_dir / "comparison_timeseries.csv",
        np.column_stack(
            [
                times,
                field_values[:, 2],
                shuoyi_dipole,
                mu_z,
                shuoyi_acc,
                force_acceleration,
                dipole_acceleration,
            ]
        ),
        delimiter=",",
        header=(
            "time,field_z,shuoyi_dipole_z,bg_dmrg_mu_z,shuoyi_acceleration,"
            "bg_dmrg_force_acceleration,bg_dmrg_dipole_acceleration"
        ),
        comments="",
        fmt="%.16e",
    )
    write_json(output_dir / "comparison_summary.json", summary)
    make_figures(output_dir, shuoyi_compare, bg_dmrg, summary)

    print(f"[BG+DMRG] Saved data: {output_dir / 'benchmark_data.npz'}", flush=True)
    print(f"[BG+DMRG] Saved comparison: {output_dir / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
