import datetime as _datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyqed
from pyqed.qchem.gdvr import AtomicChain, force_mpo

SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = SCRIPT_PATH.with_name("benchmark_results")
RUN_LABEL = "June27_H2_bg"

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
TDDMRG_PARAMS = {"D": 16, "td_bond_dim": 16}
FIELD_PARAMS = {"E0": 0.08, "omega": 0.057, "cycles": 2}
TD_PARAMS = {
    "dt": 0.5,
    "integrator": "tdvp",
    "tdvp_projection_backend": "block-sparse",
    "track_energy": False,
    "progress": True,
    "progress_every": 1,
}


def _git_text(args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=SCRIPT_PATH.parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def pulse(t, E0=FIELD_PARAMS["E0"], omega=FIELD_PARAMS["omega"], cycles=FIELD_PARAMS["cycles"]):
    T = cycles * 2 * np.pi / omega
    f = np.zeros(3)
    if 0 <= t <= T:
        f[2] = E0 * np.sin(np.pi * t / T)**2 * np.sin(omega * t)
    return f

run_id = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = OUTPUT_ROOT / f"{RUN_LABEL}_{run_id}"
output_dir.mkdir(parents=True, exist_ok=False)

setup_metadata = {
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
        "branch": _git_text(["branch", "--show-current"]),
        "commit": _git_text(["rev-parse", "HEAD"]),
        "short_commit": _git_text(["rev-parse", "--short", "HEAD"]),
        "status": _git_text(["status", "--short"]),
    },
    "molecule": {
        "elements": ELEMENTS,
        "coords": COORDS,
        "build_params": BUILD_PARAMS,
    },
    "rhf_params": RHF_PARAMS,
    "newton_params": NEWTON_PARAMS,
    "tddmrg_params": TDDMRG_PARAMS,
    "field": {
        "name": "sin2_sine_z",
        "params": FIELD_PARAMS,
        "formula": "E_z(t)=E0*sin(pi*t/T)^2*sin(omega*t), T=cycles*2*pi/omega, for 0<=t<=T",
    },
    "td_params": TD_PARAMS,
    "observable_labels": ["mu_z", "force_mpo"],
}
_write_json(output_dir / "setup.json", setup_metadata)
git_diff = _git_text(["diff", "--", "pyqed", "setup.py", str(SCRIPT_PATH.relative_to(SCRIPT_PATH.parents[3]))])
if git_diff:
    (output_dir / "code_diff.patch").write_text(git_diff + "\n")

print(f"[benchmark] Saving setup and results under: {output_dir}", flush=True)

# mol = AtomicChain(["H"] * 10, coords=[(0, 0, (i - 4.5) * 0.8) for i in range(10)])
started_at = time.time()
mol = AtomicChain(ELEMENTS, coords=COORDS)
mol.build(**BUILD_PARAMS)

mf = mol.RHF().run(**RHF_PARAMS)

mf.newton(**NEWTON_PARAMS) 

td = mf.TDDMRG(**TDDMRG_PARAMS).build()

dt = TD_PARAMS["dt"]
omega = FIELD_PARAMS["omega"]
steps = int(np.ceil((FIELD_PARAMS["cycles"] * 2 * np.pi / omega) / dt))
TD_PARAMS["steps"] = steps
setup_metadata["td_params"] = TD_PARAMS
_write_json(output_dir / "setup.json", setup_metadata)
td.run(dt=dt, steps=steps, e_ops=["mu_z", force_mpo(mol)], field=pulse,
       integrator=TD_PARAMS["integrator"],
       tdvp_projection_backend=TD_PARAMS["tdvp_projection_backend"],
       track_energy=TD_PARAMS["track_energy"],
       progress=TD_PARAMS["progress"],
       progress_every=TD_PARAMS["progress_every"])

acc = td.observables[:, 1].real + mol.nelec * np.array([pulse(t)[2] for t in td.times])
w = 2 * np.pi * np.fft.rfftfreq(acc.size, d=dt)
hhg = np.abs(np.fft.rfft((acc - acc.mean()) * np.hanning(acc.size)))**2

field_values = np.asarray([pulse(t) for t in td.times], dtype=float)
elapsed_seconds = time.time() - started_at
np.savez_compressed(
    output_dir / "benchmark_data.npz",
    times=np.asarray(td.times),
    observables=np.asarray(td.observables),
    fields=np.asarray(getattr(td, "fields", field_values)),
    field_values=field_values,
    acceleration=acc,
    hhg_omega=w,
    hhg_power=hhg,
    pre_normalization_norms=np.asarray(getattr(td, "pre_normalization_norms", [])),
    pre_normalization_norm2=np.asarray(getattr(td, "pre_normalization_norm2", [])),
    tdvp_truncation_errors=np.asarray(getattr(td, "tdvp_truncation_errors", [])),
    energy_times=np.asarray(getattr(td, "energy_times", [])),
    static_energies=np.asarray(getattr(td, "static_energies", [])),
    energy_drift=np.asarray(getattr(td, "energy_drift", [])),
    elements=np.asarray(ELEMENTS, dtype=object),
    coords=np.asarray(COORDS, dtype=float),
)

summary = {
    "run_label": RUN_LABEL,
    "run_id": run_id,
    "completed_at": _datetime.datetime.now().isoformat(timespec="seconds"),
    "elapsed_seconds": elapsed_seconds,
    "steps": steps,
    "dt": dt,
    "nelec": mol.nelec,
    "n_observables": int(td.observables.shape[1]),
    "time_final": float(td.times[-1]) if len(td.times) else None,
    "acceleration_min": float(np.min(acc)),
    "acceleration_max": float(np.max(acc)),
    "hhg_power_max": float(np.max(hhg)),
    "data_file": str(output_dir / "benchmark_data.npz"),
    "setup_file": str(output_dir / "setup.json"),
}
_write_json(output_dir / "summary.json", summary)
print(f"[benchmark] Saved data: {output_dir / 'benchmark_data.npz'}", flush=True)
print(f"[benchmark] Saved summary: {output_dir / 'summary.json'}", flush=True)
