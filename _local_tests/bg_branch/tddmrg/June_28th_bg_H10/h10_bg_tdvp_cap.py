from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqed
from pyqed.qchem.gdvr import AtomicChain, force_mpo


GEOMETRIES = {
    "afm": {
        "description": "near-uniform AFM dimerization with long outer bonds: short=1.75 bohr, long=1.85 bohr",
        "atom_z": [-8.125, -6.275, -4.525, -2.675, -0.925, 0.925, 2.675, 4.525, 6.275, 8.125],
    },
    "bonding": {
        "description": "bonding dimerization with short outer bonds: short=1.4 bohr, long=2.2 bohr",
        "atom_z": [-7.9, -6.5, -4.3, -2.9, -0.7, 0.7, 2.9, 4.3, 6.5, 7.9],
    },
    "edge_localized": {
        "description": "edge-localized dimerization with long outer bonds: short=1.4 bohr, long=2.2 bohr",
        "atom_z": [-8.3, -6.1, -4.7, -2.5, -1.1, 1.1, 2.5, 4.7, 6.1, 8.3],
    },
}

INTENSITIES = {
    "off": {"label": "0", "drive_amplitude": 0.0},
    "I1e13": {"label": "1e13", "drive_amplitude": 0.016880323915389028},
    "I5e14": {"label": "5e14", "drive_amplitude": 0.11936191509197033},
}

ORIGINAL_LZ = 18.0
ORIGINAL_NZ = 63
CAP_WIDTH = 2.0
CAP_STRENGTH = 0.01
CAP_ORDER = 2


def expanded_grid(original_lz=ORIGINAL_LZ, original_nz=ORIGINAL_NZ, cap_width=CAP_WIDTH):
    dz = 2.0 * float(original_lz) / (int(original_nz) + 1)
    extra_each_side = int(np.ceil(float(cap_width) / dz))
    nz = int(original_nz) + 2 * extra_each_side
    lz = 0.5 * dz * (nz + 1)
    actual_added_width = extra_each_side * dz
    return {
        "original_lz": float(original_lz),
        "original_nz": int(original_nz),
        "original_dz": float(dz),
        "cap_width_requested": float(cap_width),
        "extra_grid_points_each_side": int(extra_each_side),
        "actual_added_width_each_side": float(actual_added_width),
        "lz": float(lz),
        "nz": int(nz),
        "dz": float(dz),
    }


GRID = expanded_grid()


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
    if isinstance(obj, complex):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def format_float(value, digits=12):
    value = safe_float(value)
    return "None" if value is None else f"{value:.{digits}f}"


def make_field(amplitude, omega, cycles):
    total_time = float(cycles) * 2.0 * np.pi / float(omega)

    def field(t):
        out = np.zeros(3, dtype=float)
        if 0.0 <= float(t) <= total_time:
            out[2] = float(amplitude) * np.sin(np.pi * float(t) / total_time) ** 2 * np.sin(float(omega) * float(t))
        return out

    return field, total_time


def compute_dipole_acceleration(times, dipole):
    times = np.asarray(times, dtype=float)
    dipole = np.asarray(dipole, dtype=float)
    if times.size < 3:
        return np.full(times.shape, np.nan)
    dt = float(np.mean(np.diff(times)))
    return np.gradient(np.gradient(dipole - dipole.mean(), dt), dt)


def append_chunk(total, chunk):
    if not total:
        return {key: np.asarray(value) for key, value in chunk.items()}
    out = dict(total)
    for key in chunk:
        out[key] = np.concatenate([np.asarray(total[key]), np.asarray(chunk[key])], axis=0)
    return out


def save_npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def save_pickle_atomic(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def write_lock_heartbeat(lock_path, payload):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return
    payload = {**payload, "updated_at": _datetime.datetime.now().isoformat(timespec="seconds")}
    write_json_atomic(lock_path / "lock_info.json", payload)
    try:
        os.utime(lock_path, None)
    except OSError:
        pass


def remove_stale_lock_if_needed(lock_path, stale_seconds):
    lock_path = Path(lock_path)
    if stale_seconds is None or stale_seconds <= 0 or not lock_path.exists():
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    if age < stale_seconds:
        return False
    print(
        f"[H10 BG TDVP CAP] removing stale DMRG lock {lock_path} "
        f"age={age / 3600.0:.2f} h",
        flush=True,
    )
    try:
        shutil.rmtree(lock_path)
        return True
    except OSError as exc:
        print(f"[H10 BG TDVP CAP] could not remove stale DMRG lock: {exc}", flush=True)
        return False


def save_latest_outputs(output_dir, arrays, metadata):
    save_npz_atomic(output_dir / "td_timeseries_latest.npz", **arrays)
    write_json_atomic(output_dir / "progress_latest.json", metadata)


def load_td_resume(output_dir, steps, dt, disabled=False):
    if disabled:
        return None
    output_dir = Path(output_dir)
    data_path = output_dir / "td_timeseries_latest.npz"
    state_path = output_dir / "latest_state.pkl"
    progress_path = output_dir / "progress_latest.json"
    if not data_path.exists() or not state_path.exists() or not progress_path.exists():
        return None
    try:
        progress = read_json(progress_path)
        completed = int(progress.get("completed_steps", 0))
    except Exception as exc:
        print(f"[H10 BG TDVP CAP] ignoring unreadable TD resume progress: {exc}", flush=True)
        return None
    if completed <= 0:
        return None
    if completed > int(steps):
        print(
            f"[H10 BG TDVP CAP] ignoring TD resume with completed_steps={completed} > total_steps={steps}",
            flush=True,
        )
        return None
    try:
        with state_path.open("rb") as handle:
            psi = pickle.load(handle)
        with np.load(data_path, allow_pickle=True) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        print(f"[H10 BG TDVP CAP] ignoring unreadable TD resume checkpoint: {exc}", flush=True)
        return None

    required = ["times", "field_z", "mu_z", "force_observable", "force_acceleration"]
    missing = [key for key in required if key not in arrays]
    if missing:
        print(f"[H10 BG TDVP CAP] ignoring TD resume missing arrays: {missing}", flush=True)
        return None
    if len(arrays["times"]) != completed:
        print(
            f"[H10 BG TDVP CAP] ignoring TD resume length mismatch: "
            f"len(times)={len(arrays['times'])}, completed_steps={completed}",
            flush=True,
        )
        return None
    current_t0 = completed * float(dt)
    print(
        f"[H10 BG TDVP CAP] resuming TD from step {completed}/{steps}, t={current_t0:.6f}",
        flush=True,
    )
    return {
        "psi": psi,
        "arrays": arrays,
        "progress": progress,
        "completed": completed,
        "current_t0": current_t0,
    }


def make_dmrg_cache_metadata(args, geometry, atom_z, grid, build_params, newton_params, dmrg_params):
    inputs = {
        "geometry": {
            "name": args.geometry,
            "atom_z": [float(x) for x in atom_z],
            "description": geometry["description"],
        },
        "grid": grid,
        "build_params": build_params,
        "newton_params": newton_params,
        "dmrg_params": dmrg_params,
        "pyqed_path": pyqed.__file__,
    }
    cache_key = (
        f"h10_{args.geometry}_Nz{grid['nz']}_Lz{grid['lz']:.8f}_"
        f"D{dmrg_params['D']}_newton{newton_params['max_cycles']}_"
        f"dmrg{dmrg_params['nsweeps']}"
    )
    return {"cache_key": cache_key, "inputs": inputs}


def load_dmrg_cache(state_path, metadata_path, expected_metadata):
    state_path = Path(state_path)
    metadata_path = Path(metadata_path)
    if not state_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = read_json(metadata_path)
    except Exception as exc:
        print(f"[H10 BG TDVP CAP] ignoring unreadable DMRG cache metadata: {exc}", flush=True)
        return None
    if metadata.get("cache_key") != expected_metadata.get("cache_key"):
        return None
    if metadata.get("inputs") != expected_metadata.get("inputs"):
        print("[H10 BG TDVP CAP] ignoring DMRG cache with mismatched setup", flush=True)
        return None
    try:
        with state_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(f"[H10 BG TDVP CAP] ignoring unreadable DMRG cache state: {exc}", flush=True)
        return None
    if not isinstance(payload, dict) or "psi" not in payload:
        print("[H10 BG TDVP CAP] ignoring old-format DMRG cache state", flush=True)
        return None
    return payload


def save_dmrg_cache(state_path, metadata_path, expected_metadata, psi, dmrg_energy):
    payload = {
        "psi": psi,
        "dmrg_energy": dmrg_energy,
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "metadata": expected_metadata,
    }
    with Path(state_path).open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        **expected_metadata,
        "created_at": payload["created_at"],
        "dmrg_energy": dmrg_energy,
        "state_file": str(state_path),
    }
    write_json(metadata_path, metadata)
    return payload


def mirror_dmrg_state(output_dir, payload, cache_state_path, cache_metadata_path, cache_used):
    output_state_path = Path(output_dir) / "dmrg_ground_state.pkl"
    output_metadata_path = Path(output_dir) / "dmrg_ground_state.json"
    with output_state_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        **payload.get("metadata", {}),
        "dmrg_energy": payload.get("dmrg_energy"),
        "cache_used": bool(cache_used),
        "shared_cache_state_file": str(cache_state_path),
        "shared_cache_metadata_file": str(cache_metadata_path),
        "state_file": str(output_state_path),
    }
    write_json(output_metadata_path, metadata)
    return output_state_path, output_metadata_path


def td_energy_payload(td):
    static = np.asarray(getattr(td, "static_energies", []), dtype=complex)
    drift = np.asarray(getattr(td, "energy_drift", []), dtype=complex)
    norm2 = np.asarray(getattr(td, "pre_normalization_norm2", []), dtype=float)
    return {
        "energy_times": np.asarray(getattr(td, "energy_times", []), dtype=float),
        "static_energies_real": static.real,
        "static_energies_imag": static.imag,
        "energy_drift_real": drift.real,
        "energy_drift_imag": drift.imag,
        "pre_normalization_norm2": norm2,
        "tdvp_truncation_errors": np.asarray(getattr(td, "tdvp_truncation_errors", []), dtype=float),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), required=True)
    parser.add_argument("--intensity", choices=sorted(INTENSITIES), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.05841455452769231)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--drive-amplitude", type=float, default=None)
    parser.add_argument("--newton-cycles", type=int, default=200)
    parser.add_argument("--newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--dmrg-sweeps", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--cap-width", type=float, default=CAP_WIDTH)
    parser.add_argument("--cap-strength", type=float, default=CAP_STRENGTH)
    parser.add_argument("--cap-order", type=int, default=CAP_ORDER)
    parser.add_argument("--track-energy", action="store_true")
    parser.add_argument("--dmrg-lock-stale-minutes", type=float, default=180.0)
    parser.add_argument("--no-resume-td", action="store_true")
    parser.add_argument("--no-save-state", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    geometry = GEOMETRIES[args.geometry]
    intensity = INTENSITIES[args.intensity]
    amplitude = float(args.drive_amplitude) if args.drive_amplitude is not None else float(intensity["drive_amplitude"])

    grid = expanded_grid(cap_width=args.cap_width)
    field, total_time = make_field(amplitude, args.omega, args.cycles)
    steps = int(np.ceil(total_time / float(args.dt)))
    cap_settings = {"width": float(args.cap_width), "strength": float(args.cap_strength), "order": int(args.cap_order)}

    if args.output_dir is None:
        output_dir = (
            SCRIPT_PATH.with_name("benchmark_results")
            / f"h10_{args.geometry}_bg_tdvp_cap_Nz{grid['nz']}_D{args.bond_dim}_{args.intensity}"
        )
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    atom_z = np.asarray(geometry["atom_z"], dtype=float)
    coords = [(0.0, 0.0, float(z)) for z in atom_z]
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
    td_params = {
        "D": int(args.bond_dim),
        "dt": float(args.dt),
        "steps": int(steps),
        "save_every": int(args.save_every),
        "integrator": "tdvp",
        "tdvp_projection_backend": "block-sparse",
        "track_energy": bool(args.track_energy),
        "resume_td": not bool(args.no_resume_td),
        "cap": cap_settings,
    }

    setup = {
        "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "script": str(SCRIPT_PATH),
        "output_dir": str(output_dir),
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
        "geometry": {
            "name": args.geometry,
            "description": geometry["description"],
            "elements": ["H"] * len(atom_z),
            "atom_z": atom_z,
            "coords": coords,
        },
        "grid": grid,
        "build_params": build_params,
        "rhf_params": rhf_params,
        "newton_params": newton_params,
        "dmrg_params": dmrg_params,
        "td_params": td_params,
        "dmrg_cache": {
            "lock_stale_minutes": float(args.dmrg_lock_stale_minutes),
        },
        "field": {
            "intensity": args.intensity,
            "intensity_label_w_cm2": intensity["label"],
            "amplitude": amplitude,
            "omega": float(args.omega),
            "cycles": float(args.cycles),
            "total_time": float(total_time),
            "formula": "E_z(t)=A*sin(pi*t/T)^2*sin(omega*t), T=cycles*2*pi/omega",
        },
        "observables": ["mu_z", "force_mpo"],
    }
    write_json(output_dir / "setup.json", setup)

    print("=" * 72, flush=True)
    print("[H10 BG TDVP CAP] setup", flush=True)
    print(f"geometry        : {args.geometry}", flush=True)
    print(f"intensity       : {args.intensity} amplitude={amplitude:.16g}", flush=True)
    print(f"grid            : original Lz={ORIGINAL_LZ}, Nz={ORIGINAL_NZ}, dz={grid['dz']:.8f}", flush=True)
    print(f"expanded grid   : Lz={grid['lz']:.8f}, Nz={grid['nz']}, cap width={args.cap_width}", flush=True)
    print(f"TD              : dt={args.dt}, steps={steps}, save_every={args.save_every}, D={args.bond_dim}", flush=True)
    print(f"output          : {output_dir}", flush=True)
    print("=" * 72, flush=True)

    started = time.time()
    mol = AtomicChain(["H"] * len(atom_z), coords=coords)
    mol.build(**build_params)

    print("[H10 BG TDVP CAP] running RHF...", flush=True)
    mf = mol.RHF().run(**rhf_params)
    hf_initial = safe_float(mf.e_tot)
    print(f"[H10 BG TDVP CAP] RHF energy = {hf_initial:.12f} Ha", flush=True)

    print("[H10 BG TDVP CAP] running HF Newton orbital optimization...", flush=True)
    mf.newton(**newton_params)
    hf_history = [safe_float(x) for x in mf.info.get("newton_energy_history", [])]
    print(f"[H10 BG TDVP CAP] HF-TOO final = {hf_history[-1]:.12f} Ha", flush=True)

    td = mf.TDDMRG().build()
    td.verbose = max(int(getattr(td, "verbose", 0) or 0), 1)
    dmrg_cache_metadata = make_dmrg_cache_metadata(
        args, geometry, atom_z, grid, build_params, newton_params, dmrg_params
    )
    dmrg_cache_root = output_dir.parent / "dmrg_ground_states"
    dmrg_cache_root.mkdir(parents=True, exist_ok=True)
    dmrg_cache_state = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.pkl"
    dmrg_cache_json = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.json"
    dmrg_cache_lock = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.lock"

    cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
    dmrg_cache_used = cache_payload is not None
    dmrg_started = time.time()
    lock_stale_seconds = float(args.dmrg_lock_stale_minutes) * 60.0

    while cache_payload is None:
        remove_stale_lock_if_needed(dmrg_cache_lock, lock_stale_seconds)
        try:
            dmrg_cache_lock.mkdir()
            have_lock = True
        except FileExistsError:
            have_lock = False

        lock_payload = {
            "host": platform.node(),
            "pid": os.getpid(),
            "cache_state": str(dmrg_cache_state),
            "cache_key": dmrg_cache_metadata["cache_key"],
            "created_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        }
        if not have_lock:
            print(
                f"[H10 BG TDVP CAP] waiting for shared DMRG ground-state cache: {dmrg_cache_state}",
                flush=True,
            )
            while dmrg_cache_lock.exists():
                cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
                if cache_payload is not None:
                    dmrg_cache_used = True
                    break
                time.sleep(60.0)
                if remove_stale_lock_if_needed(dmrg_cache_lock, lock_stale_seconds):
                    break
                print(
                    f"[H10 BG TDVP CAP] still waiting for DMRG cache "
                    f"elapsed={time.time() - dmrg_started:.2f}s",
                    flush=True,
                )
            if cache_payload is None:
                cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
                dmrg_cache_used = cache_payload is not None
            continue

        try:
            cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
            if cache_payload is not None:
                dmrg_cache_used = True
                break
            write_lock_heartbeat(dmrg_cache_lock, lock_payload)

            print(
                f"[H10 BG TDVP CAP] running static DMRG: D={dmrg_params['D']}, nsweeps={dmrg_params['nsweeps']}...",
                flush=True,
            )
            dmrg_done = threading.Event()

            def dmrg_heartbeat():
                while not dmrg_done.wait(60.0):
                    write_lock_heartbeat(dmrg_cache_lock, lock_payload)
                    print(
                        f"[H10 BG TDVP CAP] static DMRG still running "
                        f"elapsed={time.time() - dmrg_started:.2f}s",
                        flush=True,
                    )

            heartbeat_thread = threading.Thread(target=dmrg_heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                td.optimize_ground_state(
                    D=dmrg_params["D"],
                    nsweeps=dmrg_params["nsweeps"],
                    symmetry_list=dmrg_params["symmetry_list"],
                    initial_guess=td.init_guess,
                    not_conv_err=dmrg_params["not_conv_err"],
                )
            finally:
                dmrg_done.set()
                heartbeat_thread.join(timeout=1.0)

            dmrg_energy = safe_float(getattr(td, "e_tot", None))
            psi = td.dmrg.ground_state.copy()
            cache_payload = save_dmrg_cache(
                dmrg_cache_state,
                dmrg_cache_json,
                dmrg_cache_metadata,
                psi,
                dmrg_energy,
            )
            print(
                f"[H10 BG TDVP CAP] saved shared DMRG ground-state cache: {dmrg_cache_state}",
                flush=True,
            )
        finally:
            try:
                shutil.rmtree(dmrg_cache_lock)
            except OSError:
                pass

    psi = cache_payload["psi"].copy()
    dmrg_energy = safe_float(cache_payload.get("dmrg_energy"))
    output_dmrg_state, output_dmrg_json = mirror_dmrg_state(
        output_dir,
        cache_payload,
        dmrg_cache_state,
        dmrg_cache_json,
        dmrg_cache_used,
    )
    print(
        f"[H10 BG TDVP CAP] DMRG energy = {format_float(dmrg_energy)} Ha "
        f"cache_used={dmrg_cache_used} elapsed={time.time() - dmrg_started:.2f}s",
        flush=True,
    )

    energy_audit = {
        "units": "Hartree",
        "hf_initial": hf_initial,
        "hf_newton_history": hf_history,
        "hf_too_energy": hf_history[-1] if hf_history else safe_float(mf.e_tot),
        "newton_info": {
            "newton_cycles": int(mf.info.get("newton_cycles", 0)),
            "newton_converged": bool(mf.info.get("newton_converged", False)),
        },
        "dmrg_energy": dmrg_energy,
        "dmrg_cache_used": bool(dmrg_cache_used),
        "dmrg_ground_state_file": str(output_dmrg_state),
        "dmrg_ground_state_metadata_file": str(output_dmrg_json),
        "shared_dmrg_ground_state_file": str(dmrg_cache_state),
    }
    write_json(output_dir / "energy_audit.json", energy_audit)

    all_data = {}
    all_norm2 = []
    all_trunc = []
    chunk_index = 0
    completed = 0
    current_t0 = 0.0
    resume_payload = load_td_resume(
        output_dir,
        steps=steps,
        dt=args.dt,
        disabled=args.no_resume_td,
    )
    if resume_payload is not None:
        psi = resume_payload["psi"].copy()
        completed = int(resume_payload["completed"])
        current_t0 = float(resume_payload["current_t0"])
        chunk_index = completed // max(int(args.save_every), 1)
        resume_arrays = resume_payload["arrays"]
        data_keys = ["times", "field_z", "mu_z", "force_observable", "force_acceleration"]
        all_data = {key: np.asarray(resume_arrays[key]) for key in data_keys if key in resume_arrays}
        if "dipole_acceleration" in resume_arrays:
            all_data["dipole_acceleration"] = np.asarray(resume_arrays["dipole_acceleration"])
        if "pre_normalization_norm2" in resume_arrays:
            all_norm2.append(np.asarray(resume_arrays["pre_normalization_norm2"], dtype=float))
        if "tdvp_truncation_errors" in resume_arrays:
            all_trunc.append(np.asarray(resume_arrays["tdvp_truncation_errors"], dtype=float))
        if "times" in all_data and "mu_z" in all_data:
            all_data["dipole_acceleration"] = compute_dipole_acceleration(all_data["times"], all_data["mu_z"])

    if completed >= steps:
        print(
            f"[H10 BG TDVP CAP] TD already completed in existing checkpoint: "
            f"{completed}/{steps} steps",
            flush=True,
        )
        return 0

    dipole_z_mpo = td.get_interaction_mpo(axis=2)
    force_z_mpo = force_mpo(mol)

    while completed < steps:
        chunk_index += 1
        chunk_steps = min(int(args.save_every), steps - completed)
        print(
            f"[H10 BG TDVP CAP] TD chunk {chunk_index}: steps {completed + 1}-{completed + chunk_steps} / {steps}, "
            f"t={current_t0:.4f}->{current_t0 + chunk_steps * args.dt:.4f}",
            flush=True,
        )
        chunk_started = time.time()
        td.run(
            psi0=psi,
            D=args.bond_dim,
            dt=args.dt,
            steps=chunk_steps,
            e_ops=[dipole_z_mpo, force_z_mpo],
            field=field,
            cap=cap_settings,
            t0=current_t0,
            integrator="tdvp",
            tdvp_projection_backend="block-sparse",
            track_energy=args.track_energy,
            progress=True,
            progress_every=chunk_steps,
        )
        psi = td.final_state.copy()

        times = np.asarray(td.times, dtype=float)
        fields = np.asarray(getattr(td, "fields", [field(t) for t in times]), dtype=float)
        obs = np.asarray(td.observables, dtype=complex)
        mu_z = obs[:, 0].real
        force_z = obs[:, 1].real
        force_acc = force_z + mol.nelec * fields[:, 2]
        chunk = {
            "times": times,
            "field_z": fields[:, 2],
            "mu_z": mu_z,
            "force_observable": force_z,
            "force_acceleration": force_acc,
        }
        all_data = append_chunk(all_data, chunk)
        all_norm2.append(np.asarray(getattr(td, "pre_normalization_norm2", []), dtype=float))
        all_trunc.append(np.asarray(getattr(td, "tdvp_truncation_errors", []), dtype=float))

        completed += chunk_steps
        current_t0 += chunk_steps * args.dt
        all_data["dipole_acceleration"] = compute_dipole_acceleration(all_data["times"], all_data["mu_z"])
        pre_norm2 = np.concatenate(all_norm2) if all_norm2 else np.asarray([], dtype=float)
        trunc = np.concatenate(all_trunc) if all_trunc else np.asarray([], dtype=float)
        survival = np.cumprod(pre_norm2[np.isfinite(pre_norm2)]) if pre_norm2.size else np.asarray([], dtype=float)

        progress = {
            "completed_steps": int(completed),
            "total_steps": int(steps),
            "time": float(current_t0),
            "elapsed_seconds": time.time() - started,
            "chunk_elapsed_seconds": time.time() - chunk_started,
            "latest_mu_z": safe_float(all_data["mu_z"][-1]),
            "latest_force_acceleration": safe_float(all_data["force_acceleration"][-1]),
            "latest_pre_normalization_norm2": safe_float(pre_norm2[-1]) if pre_norm2.size else None,
            "estimated_norm2_without_step_renormalization": safe_float(survival[-1]) if survival.size else None,
            "max_tdvp_truncation_error": safe_float(np.nanmax(trunc)) if trunc.size else None,
        }
        latest_arrays = {
            **all_data,
            "pre_normalization_norm2": pre_norm2,
            "tdvp_truncation_errors": trunc,
            "estimated_norm2_without_step_renormalization": survival,
        }
        save_latest_outputs(output_dir, latest_arrays, progress)
        save_npz_atomic(
            checkpoints_dir / f"step_{completed:06d}.npz",
            **latest_arrays,
            completed_steps=np.asarray(completed),
        )
        if not args.no_save_state:
            save_pickle_atomic(output_dir / "latest_state.pkl", psi)
            save_pickle_atomic(checkpoints_dir / f"state_step_{completed:06d}.pkl", psi)
        print(
            f"[H10 BG TDVP CAP] saved step {completed}/{steps}: "
            f"mu_z={progress['latest_mu_z']:.8e}, "
            f"norm2_est={progress['estimated_norm2_without_step_renormalization']}",
            flush=True,
        )

    final_summary = {
        "completed_at": _datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.time() - started,
        "completed_steps": int(completed),
        "total_steps": int(steps),
        "time_final": float(current_t0),
        "mu_z_min": safe_float(np.nanmin(all_data["mu_z"])),
        "mu_z_max": safe_float(np.nanmax(all_data["mu_z"])),
        "force_acceleration_min": safe_float(np.nanmin(all_data["force_acceleration"])),
        "force_acceleration_max": safe_float(np.nanmax(all_data["force_acceleration"])),
        "data_file": str(output_dir / "td_timeseries_latest.npz"),
        "setup_file": str(output_dir / "setup.json"),
        "energy_audit_file": str(output_dir / "energy_audit.json"),
        "dmrg_cache_used": bool(dmrg_cache_used),
        "dmrg_ground_state_file": str(output_dmrg_state),
        "dmrg_ground_state_metadata_file": str(output_dmrg_json),
        "shared_dmrg_ground_state_file": str(dmrg_cache_state),
    }
    write_json(output_dir / "summary.json", final_summary)
    if not args.no_save_state:
        save_pickle_atomic(output_dir / "final_state.pkl", psi)
    print("[H10 BG TDVP CAP] completed", flush=True)
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
