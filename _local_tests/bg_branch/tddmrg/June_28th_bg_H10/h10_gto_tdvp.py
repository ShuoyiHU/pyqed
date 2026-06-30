#!/usr/bin/env python3
"""GTO H10 active-space TDDMRG benchmark with chunked saves."""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import pickle
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyqed
from pyqed.qchem.mol import Molecule
from pyqed.qchem.dmrg.tddmrg import TDDMRG


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


def read_json(path):
    return json.loads(Path(path).read_text())


def json_canonical(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=json_default))


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


def density_z_grid(args, atom_z):
    zmin = float(args.density_zmin) if args.density_zmin is not None else float(np.min(atom_z) - args.density_padding)
    zmax = float(args.density_zmax) if args.density_zmax is not None else float(np.max(atom_z) + args.density_padding)
    return np.linspace(zmin, zmax, int(args.density_nz))


def active_state_density_matrix_ao(td, psi):
    dmrg = getattr(td, "dmrg", None)
    if dmrg is None:
        dmrg = SimpleNamespace(ground_state=None, states=None)
        td.dmrg = dmrg

    old_ground_state = getattr(dmrg, "ground_state", None)
    old_states = getattr(dmrg, "states", None)
    try:
        dmrg.ground_state = psi
        dmrg.states = None
        dm_mo = np.asarray(td.make_rdm1(spatial=True, with_core=True), dtype=complex)
    finally:
        dmrg.ground_state = old_ground_state
        dmrg.states = old_states

    mo = np.asarray(td.mo_coeff[:, : dm_mo.shape[0]], dtype=complex)
    return mo @ dm_mo @ mo.conj().T


def axis_density_from_state(mol, td, psi, z_grid):
    pmol = mol.topyscf()
    coords = np.zeros((len(z_grid), 3), dtype=float)
    coords[:, 2] = np.asarray(z_grid, dtype=float)
    ao = np.asarray(pmol.eval_gto("GTOval_sph", coords), dtype=complex)
    dm_ao = active_state_density_matrix_ao(td, psi)
    rho = np.einsum("zi,ij,zj->z", ao.conj(), dm_ao, ao, optimize=True)
    return np.real_if_close(rho).real


def save_density_outputs(output_dir, z_grid, times, rho_rows, metadata):
    if not rho_rows:
        return
    np.savez_compressed(
        output_dir / "density_timeseries_latest.npz",
        density_times=np.asarray(times, dtype=float),
        times=np.asarray(times, dtype=float),
        z_grid=np.asarray(z_grid, dtype=float),
        rho_z=np.asarray(rho_rows, dtype=float),
        charge_density_z=np.asarray(rho_rows, dtype=float),
    )
    write_json(output_dir / "density_progress_latest.json", metadata)


def append_chunk(total, chunk):
    if not total:
        return {key: np.asarray(value) for key, value in chunk.items()}
    out = dict(total)
    for key in chunk:
        out[key] = np.concatenate([np.asarray(total[key]), np.asarray(chunk[key])], axis=0)
    return out


def save_latest_outputs(output_dir, arrays, metadata):
    np.savez_compressed(output_dir / "td_timeseries_latest.npz", **arrays)
    write_json(output_dir / "progress_latest.json", metadata)


def make_dmrg_cache_metadata(args, geometry, atom_z, mol_params, rhf_params, dmrg_params):
    inputs = {
        "geometry": {
            "name": args.geometry,
            "atom_z": [float(x) for x in atom_z],
            "description": geometry["description"],
        },
        "molecule": mol_params,
        "rhf_params": rhf_params,
        "dmrg_params": dmrg_params,
        "pyqed_path": pyqed.__file__,
    }
    cache_key = (
        f"h10_gto_{args.geometry}_{args.basis.replace('-', '').replace('*', 's')}_"
        f"ncas{dmrg_params['ncas']}_nelecas{dmrg_params['nelecas']}_"
        f"D{dmrg_params['D']}_dmrg{dmrg_params['nsweeps']}"
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
        print(f"[H10 GTO TDVP] ignoring unreadable DMRG cache metadata: {exc}", flush=True)
        return None
    if metadata.get("cache_key") != expected_metadata.get("cache_key"):
        return None
    if json_canonical(metadata.get("inputs")) != json_canonical(expected_metadata.get("inputs")):
        print("[H10 GTO TDVP] ignoring DMRG cache with mismatched setup", flush=True)
        return None
    try:
        with state_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        print(f"[H10 GTO TDVP] ignoring unreadable DMRG cache state: {exc}", flush=True)
        return None
    if not isinstance(payload, dict) or "psi" not in payload:
        print("[H10 GTO TDVP] ignoring old-format DMRG cache state", flush=True)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=sorted(GEOMETRIES), required=True)
    parser.add_argument("--intensity", choices=sorted(INTENSITIES), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--ncas", type=int, default=14)
    parser.add_argument("--nelecas", type=int, default=10)
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--dmrg-sweeps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.05841455452769231)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--drive-amplitude", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--track-energy", action="store_true")
    parser.add_argument("--no-save-state", action="store_true")
    parser.add_argument("--tdvp-projection-backend", default=None)
    parser.add_argument("--tdvp-dynamic-mode", default="midpoint")
    parser.add_argument("--save-density", action="store_true", help="Save rho(z,t) on the molecular axis at checkpoint boundaries.")
    parser.add_argument("--density-nz", type=int, default=401)
    parser.add_argument("--density-padding", type=float, default=4.0)
    parser.add_argument("--density-zmin", type=float, default=None)
    parser.add_argument("--density-zmax", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    geometry = GEOMETRIES[args.geometry]
    intensity = INTENSITIES[args.intensity]
    amplitude = float(args.drive_amplitude) if args.drive_amplitude is not None else float(intensity["drive_amplitude"])
    field, total_time = make_field(amplitude, args.omega, args.cycles)
    steps = int(np.ceil(total_time / float(args.dt)))

    if args.output_dir is None:
        output_dir = (
            SCRIPT_PATH.with_name("benchmark_results")
            / f"h10_{args.geometry}_gto_tdvp_{args.basis}_CAS{args.nelecas}e_{args.ncas}o_D{args.bond_dim}_{args.intensity}"
        )
    else:
        output_dir = args.output_dir
    output_dir = output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    atom_z = np.asarray(geometry["atom_z"], dtype=float)
    coords = [(0.0, 0.0, float(z)) for z in atom_z]
    mol_params = {
        "atom": [("H", tuple(coord)) for coord in coords],
        "basis": args.basis,
        "unit": "bohr",
        "charge": 0,
        "spin": 0,
        "build_driver": "pyscf",
    }
    rhf_params = {"verbose": 0}
    dmrg_params = {
        "D": int(args.bond_dim),
        "nsweeps": int(args.dmrg_sweeps),
        "ncas": int(args.ncas),
        "nelecas": int(args.nelecas),
        "symmetry_list": ["charge", "sz"],
        "orbital_layout": "spatial",
        "initial_guess": "hf",
        "not_conv_err": False,
    }
    td_params = {
        "D": int(args.bond_dim),
        "dt": float(args.dt),
        "steps": int(steps),
        "save_every": int(args.save_every),
        "integrator": "tdvp",
        "tdvp_projection_backend": args.tdvp_projection_backend,
        "tdvp_dynamic_mode": args.tdvp_dynamic_mode,
        "track_energy": bool(args.track_energy),
        "save_density": bool(args.save_density),
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
        "molecule": mol_params,
        "rhf_params": rhf_params,
        "dmrg_params": dmrg_params,
        "td_params": td_params,
        "field": {
            "intensity": args.intensity,
            "intensity_label_w_cm2": intensity["label"],
            "amplitude": amplitude,
            "omega": float(args.omega),
            "cycles": float(args.cycles),
            "total_time": float(total_time),
            "formula": "E_z(t)=A*sin(pi*t/T)^2*sin(omega*t), T=cycles*2*pi/omega",
        },
        "observables": ["mu_z"],
        "density": {
            "enabled": bool(args.save_density),
            "quantity": "electron density on molecular axis, rho(x=0,y=0,z)",
            "density_nz": int(args.density_nz),
            "density_padding": float(args.density_padding),
            "density_zmin": args.density_zmin,
            "density_zmax": args.density_zmax,
            "save_cadence": "initial state and every TD checkpoint",
        },
        "cap": None,
    }
    write_json(output_dir / "setup.json", setup)

    print("=" * 72, flush=True)
    print("[H10 GTO TDVP] setup", flush=True)
    print(f"geometry        : {args.geometry}", flush=True)
    print(f"basis/active    : {args.basis} CAS({args.nelecas}e,{args.ncas}o)", flush=True)
    print(f"intensity       : {args.intensity} amplitude={amplitude:.16g}", flush=True)
    print(f"TD              : dt={args.dt}, steps={steps}, save_every={args.save_every}, D={args.bond_dim}", flush=True)
    print(f"output          : {output_dir}", flush=True)
    print("=" * 72, flush=True)

    started = time.time()
    mol = Molecule(
        atom=mol_params["atom"],
        basis=mol_params["basis"],
        unit=mol_params["unit"],
        charge=mol_params["charge"],
        spin=mol_params["spin"],
    )
    mol.build(driver=mol_params["build_driver"])

    print("[H10 GTO TDVP] running RHF...", flush=True)
    mf = mol.RHF().run(**rhf_params)
    hf_energy = safe_float(mf.e_tot)
    print(f"[H10 GTO TDVP] RHF energy = {hf_energy:.12f} Ha", flush=True)

    td = TDDMRG(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        orbital_layout=dmrg_params["orbital_layout"],
    ).build()
    td.verbose = max(int(getattr(td, "verbose", 0) or 0), 1)
    dmrg_cache_metadata = make_dmrg_cache_metadata(args, geometry, atom_z, mol_params, rhf_params, dmrg_params)
    dmrg_cache_root = output_dir.parent / "dmrg_ground_states_gto"
    dmrg_cache_root.mkdir(parents=True, exist_ok=True)
    dmrg_cache_state = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.pkl"
    dmrg_cache_json = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.json"
    dmrg_cache_lock = dmrg_cache_root / f"{dmrg_cache_metadata['cache_key']}.lock"

    cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
    dmrg_cache_used = cache_payload is not None
    dmrg_started = time.time()

    while cache_payload is None:
        try:
            dmrg_cache_lock.mkdir()
            have_lock = True
        except FileExistsError:
            have_lock = False

        if not have_lock:
            print(f"[H10 GTO TDVP] waiting for shared DMRG cache: {dmrg_cache_state}", flush=True)
            while dmrg_cache_lock.exists():
                cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
                if cache_payload is not None:
                    dmrg_cache_used = True
                    break
                time.sleep(60.0)
                print(f"[H10 GTO TDVP] still waiting for DMRG cache elapsed={time.time() - dmrg_started:.2f}s", flush=True)
            if cache_payload is None:
                cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
                dmrg_cache_used = cache_payload is not None
            continue

        try:
            cache_payload = load_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata)
            if cache_payload is not None:
                dmrg_cache_used = True
                break

            print(
                f"[H10 GTO TDVP] running static DMRG: D={dmrg_params['D']}, nsweeps={dmrg_params['nsweeps']}...",
                flush=True,
            )
            dmrg_done = threading.Event()

            def dmrg_heartbeat():
                while not dmrg_done.wait(60.0):
                    print(f"[H10 GTO TDVP] static DMRG still running elapsed={time.time() - dmrg_started:.2f}s", flush=True)

            heartbeat_thread = threading.Thread(target=dmrg_heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                td.optimize_ground_state(
                    D=dmrg_params["D"],
                    nsweeps=dmrg_params["nsweeps"],
                    symmetry_list=dmrg_params["symmetry_list"],
                    initial_guess=dmrg_params["initial_guess"],
                    not_conv_err=dmrg_params["not_conv_err"],
                )
            finally:
                dmrg_done.set()
                heartbeat_thread.join(timeout=1.0)

            dmrg_energy = safe_float(getattr(td, "e_tot", None))
            psi = td.export_ground_state(dense=True)
            cache_payload = save_dmrg_cache(dmrg_cache_state, dmrg_cache_json, dmrg_cache_metadata, psi, dmrg_energy)
            print(f"[H10 GTO TDVP] saved shared DMRG cache: {dmrg_cache_state}", flush=True)
        finally:
            try:
                dmrg_cache_lock.rmdir()
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
        f"[H10 GTO TDVP] DMRG energy = {format_float(dmrg_energy)} Ha "
        f"cache_used={dmrg_cache_used} elapsed={time.time() - dmrg_started:.2f}s",
        flush=True,
    )

    energy_audit = {
        "units": "Hartree",
        "hf_energy": hf_energy,
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
    all_energy_times = []
    all_static_energies = []
    all_energy_drift = []
    density_times = []
    density_rows = []
    z_density = density_z_grid(args, atom_z) if args.save_density else None
    completed = 0
    chunk_index = 0
    current_t0 = 0.0

    if args.save_density:
        print(
            f"[H10 GTO TDVP] saving axis density: nz={len(z_density)}, "
            f"z=[{z_density[0]:.4f}, {z_density[-1]:.4f}]",
            flush=True,
        )
        density_times.append(0.0)
        density_rows.append(axis_density_from_state(mol, td, psi, z_density))
        save_density_outputs(
            output_dir,
            z_density,
            density_times,
            density_rows,
            {
                "completed_steps": 0,
                "total_steps": int(steps),
                "n_density_frames": len(density_rows),
                "time": 0.0,
            },
        )

    while completed < steps:
        chunk_index += 1
        chunk_steps = min(int(args.save_every), steps - completed)
        print(
            f"[H10 GTO TDVP] TD chunk {chunk_index}: steps {completed + 1}-{completed + chunk_steps} / {steps}, "
            f"t={current_t0:.4f}->{current_t0 + chunk_steps * args.dt:.4f}",
            flush=True,
        )
        chunk_started = time.time()
        td.run(
            psi0=psi,
            D=args.bond_dim,
            dt=args.dt,
            steps=chunk_steps,
            e_ops=["mu_z"],
            field=field,
            t0=current_t0,
            integrator="tdvp",
            tdvp_projection_backend=args.tdvp_projection_backend,
            tdvp_dynamic_mode=args.tdvp_dynamic_mode,
            track_energy=args.track_energy,
            progress=True,
            progress_every=chunk_steps,
        )
        psi = td.final_state.copy()

        times = np.asarray(td.times, dtype=float)
        fields = np.asarray(getattr(td, "fields", [field(t) for t in times]), dtype=float)
        obs = np.asarray(td.observables, dtype=complex)
        chunk = {
            "times": times,
            "field_z": fields[:, 2],
            "mu_z": obs[:, 0].real,
        }
        all_data = append_chunk(all_data, chunk)
        all_data["dipole_acceleration"] = compute_dipole_acceleration(all_data["times"], all_data["mu_z"])
        all_norm2.append(np.asarray(getattr(td, "pre_normalization_norm2", []), dtype=float))
        all_trunc.append(np.asarray(getattr(td, "tdvp_truncation_errors", []), dtype=float))
        all_energy_times.append(np.asarray(getattr(td, "energy_times", []), dtype=float))
        all_static_energies.append(np.asarray(getattr(td, "static_energies", []), dtype=complex))
        all_energy_drift.append(np.asarray(getattr(td, "energy_drift", []), dtype=complex))

        completed += chunk_steps
        current_t0 += chunk_steps * args.dt
        if args.save_density:
            density_times.append(float(current_t0))
            density_rows.append(axis_density_from_state(mol, td, psi, z_density))
            save_density_outputs(
                output_dir,
                z_density,
                density_times,
                density_rows,
                {
                    "completed_steps": int(completed),
                    "total_steps": int(steps),
                    "n_density_frames": len(density_rows),
                    "time": float(current_t0),
                },
            )
        pre_norm2 = np.concatenate(all_norm2) if all_norm2 else np.asarray([], dtype=float)
        trunc = np.concatenate(all_trunc) if all_trunc else np.asarray([], dtype=float)
        survival = np.cumprod(pre_norm2[np.isfinite(pre_norm2)]) if pre_norm2.size else np.asarray([], dtype=float)
        energy_times = np.concatenate([x for x in all_energy_times if x.size]) if any(x.size for x in all_energy_times) else np.asarray([], dtype=float)
        static_energies = np.concatenate([x for x in all_static_energies if x.size]) if any(x.size for x in all_static_energies) else np.asarray([], dtype=complex)
        energy_drift = np.concatenate([x for x in all_energy_drift if x.size]) if any(x.size for x in all_energy_drift) else np.asarray([], dtype=complex)

        progress = {
            "completed_steps": int(completed),
            "total_steps": int(steps),
            "time": float(current_t0),
            "elapsed_seconds": time.time() - started,
            "chunk_elapsed_seconds": time.time() - chunk_started,
            "latest_mu_z": safe_float(all_data["mu_z"][-1]),
            "latest_pre_normalization_norm2": safe_float(pre_norm2[-1]) if pre_norm2.size else None,
            "estimated_norm2_without_step_renormalization": safe_float(survival[-1]) if survival.size else None,
            "max_tdvp_truncation_error": safe_float(np.nanmax(trunc)) if trunc.size else None,
        }
        latest_arrays = {
            **all_data,
            "pre_normalization_norm2": pre_norm2,
            "tdvp_truncation_errors": trunc,
            "estimated_norm2_without_step_renormalization": survival,
            "energy_times": energy_times,
            "static_energies_real": static_energies.real,
            "static_energies_imag": static_energies.imag,
            "energy_drift_real": energy_drift.real,
            "energy_drift_imag": energy_drift.imag,
        }
        save_latest_outputs(output_dir, latest_arrays, progress)
        np.savez_compressed(
            checkpoints_dir / f"step_{completed:06d}.npz",
            **latest_arrays,
            completed_steps=np.asarray(completed),
        )
        if args.save_density:
            np.savez_compressed(
                checkpoints_dir / f"density_step_{completed:06d}.npz",
                density_times=np.asarray(density_times, dtype=float),
                times=np.asarray(density_times, dtype=float),
                z_grid=np.asarray(z_density, dtype=float),
                rho_z=np.asarray(density_rows, dtype=float),
                charge_density_z=np.asarray(density_rows, dtype=float),
                completed_steps=np.asarray(completed),
            )
        if not args.no_save_state:
            with (output_dir / "latest_state.pkl").open("wb") as handle:
                pickle.dump(psi, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[H10 GTO TDVP] saved step {completed}/{steps}: "
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
        with (output_dir / "final_state.pkl").open("wb") as handle:
            pickle.dump(psi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("[H10 GTO TDVP] completed", flush=True)
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
