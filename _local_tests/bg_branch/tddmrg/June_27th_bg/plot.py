from pathlib import Path
import json
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
BG_ROOT = HERE / "benchmark_results"
SHUOYI_ROOT = (
    HERE.parents[2]
    / "shuoyi_branch"
    / "benchmark_results"
)


def newest(pattern_root, glob_pattern, required_file):
    runs = sorted(
        p for p in Path(pattern_root).glob(glob_pattern)
        if (p / required_file).exists()
    )
    if not runs:
        raise FileNotFoundError(
            f"No run matching {glob_pattern!r} with {required_file!r} under {pattern_root}"
        )
    return runs[-1]


def load_bg(run):
    data = np.load(run / "benchmark_data.npz", allow_pickle=True)
    setup = json.loads((run / "setup.json").read_text())
    times = np.asarray(data["times"], dtype=float)
    obs = np.asarray(data["observables"])
    field = np.asarray(data["field_values"], dtype=float)[:, 2]
    mu_z = obs[:, 0].real
    force = obs[:, 1].real
    nelec = int(setup["molecule"]["elements"].count("H"))
    force_acc = np.asarray(data["acceleration"], dtype=float)
    if force_acc.size != times.size:
        force_acc = force + nelec * field
    dt = float(setup["td_params"]["dt"])
    omega = float(setup["field"]["params"]["omega"])
    dip_acc = np.gradient(np.gradient(mu_z - mu_z.mean(), dt), dt)
    w = 2 * np.pi * np.fft.rfftfreq(times.size, d=dt)
    return {
        "run": run,
        "times": times,
        "field": field,
        "mu_z": mu_z,
        "force": force,
        "force_acc": force_acc,
        "dip_acc": dip_acc,
        "omega": omega,
        "hhg_w": w,
        "hhg_force": np.abs(np.fft.rfft((force_acc - force_acc.mean()) * np.hanning(force_acc.size))) ** 2,
        "hhg_dip": np.abs(np.fft.rfft((dip_acc - dip_acc.mean()) * np.hanning(dip_acc.size))) ** 2,
    }


def load_shuoyi(run):
    data = np.load(run / "shuoyi_benchmark_data.npz", allow_pickle=True)
    times = np.asarray(data["times"], dtype=float)
    dip = np.asarray(data["dipole_z"], dtype=float)
    acc = np.asarray(data["acceleration"], dtype=float)
    return {
        "run": run,
        "times": times,
        "field": np.asarray(data["field_z"], dtype=float),
        "mu_z": dip,
        "dip_acc": acc,
        "hhg_w": np.asarray(data["hhg_omega"], dtype=float),
        "hhg_dip": np.asarray(data["hhg_power"], dtype=float),
        "z_grid": np.asarray(data["z_grid"], dtype=float),
        "densities": np.asarray(data["densities"], dtype=float),
        "times_full": np.asarray(data["times_full"], dtype=float),
    }


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def load_shuoyi_energy(run):
    ground_dir = Path(run) / "ground_state"
    candidates = sorted(ground_dir.glob("*_meta.pkl"))
    final_meta = ground_dir / "04_DMRG_Final_meta.pkl"
    if not final_meta.exists():
        final_meta = candidates[-1] if candidates else None
    if final_meta is None or not final_meta.exists():
        return {}
    with final_meta.open("rb") as handle:
        meta = pickle.load(handle)
    log = dict(meta.get("log", {}))
    hf_initial = safe_float(log.get("hf_initial"))
    hf_pre_opt = [safe_float(x) for x in log.get("hf_pre_opt", [])]
    hf_pre_opt = [x for x in hf_pre_opt if x is not None]
    dmrg_cycles = list(log.get("dmrg_cycles", []))
    dmrg_values = [safe_float(x.get("e_dmrg")) for x in dmrg_cycles]
    dmrg_values = [x for x in dmrg_values if x is not None]
    return {
        "source_file": str(final_meta),
        "hf_initial": hf_initial,
        "hf_newton_final": hf_pre_opt[-1] if hf_pre_opt else hf_initial,
        "hf_too_energy": hf_pre_opt[-1] if hf_pre_opt else hf_initial,
        "dmrg_final": dmrg_values[-1] if dmrg_values else None,
    }


def load_bg_energy(run):
    audit_path = Path(run) / "energy_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        return {
            "source_file": str(audit_path),
            **dict(audit.get("bg", {})),
        }
    summary_path = Path(run) / "comparison_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return {
            "source_file": str(summary_path),
            "dmrg_final": safe_float(summary.get("dmrg_energy")),
        }
    return {}


def energy_delta(bg_energy, shuoyi_energy):
    keys = ["hf_initial", "hf_too_energy", "hf_newton_final", "dmrg_final"]
    out = {}
    for key in keys:
        bg = safe_float(bg_energy.get(key))
        shuoyi = safe_float(shuoyi_energy.get(key))
        out[f"{key}_bg_minus_shuoyi"] = None if bg is None or shuoyi is None else bg - shuoyi
    return out


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a, b):
    a = np.asarray(a, dtype=float) - np.mean(a)
    b = np.asarray(b, dtype=float) - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else float("nan")


def spectrum_from_acc(times, acc):
    times = np.asarray(times, dtype=float)
    acc = np.asarray(acc, dtype=float)
    dt = float(np.mean(np.diff(times)))
    w = 2 * np.pi * np.fft.rfftfreq(acc.size, d=dt)
    p = np.abs(np.fft.rfft((acc - acc.mean()) * np.hanning(acc.size))) ** 2
    return w, p


def linear_fit_residual(reference, current):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    design = np.column_stack([reference, np.ones_like(reference)])
    scale, offset = np.linalg.lstsq(design, current, rcond=None)[0]
    residual = current - (scale * reference + offset)
    return {
        "scale": float(scale),
        "offset": float(offset),
        "rmse_after_scale_offset": rmse(residual, np.zeros_like(residual)),
    }


old_bg_run = newest(BG_ROOT, "June27_H2_bg_20*", "benchmark_data.npz")
dmrg_bg_run = newest(BG_ROOT, "June27_H2_bg_dmrg_init_*", "benchmark_data.npz")
shuoyi_run = newest(SHUOYI_ROOT, "June27_H2_shuoyi_u1_tdvp_*", "shuoyi_benchmark_data.npz")

old_bg = load_bg(old_bg_run)
dmrg_bg = load_bg(dmrg_bg_run)
shuoyi = load_shuoyi(shuoyi_run)
dmrg_bg_energy = load_bg_energy(dmrg_bg_run)
shuoyi_energy = load_shuoyi_energy(shuoyi_run)
energy_differences = energy_delta(dmrg_bg_energy, shuoyi_energy)

figdir = dmrg_bg_run / "figures"
figdir.mkdir(exist_ok=True)

t = dmrg_bg["times"]
omega = dmrg_bg["omega"]
harm = dmrg_bg["hhg_w"] / omega
mask = harm <= 80

summary = {
    "old_bg_run": str(old_bg_run),
    "dmrg_bg_run": str(dmrg_bg_run),
    "shuoyi_run": str(shuoyi_run),
    "field_max_abs_diff_dmrg_bg_vs_shuoyi": float(np.max(np.abs(dmrg_bg["field"] - shuoyi["field"]))),
    "dipole_rmse_old_bg_vs_shuoyi": rmse(old_bg["mu_z"], shuoyi["mu_z"]),
    "dipole_corr_old_bg_vs_shuoyi": corr(old_bg["mu_z"], shuoyi["mu_z"]),
    "dipole_rmse_dmrg_bg_vs_shuoyi": rmse(dmrg_bg["mu_z"], shuoyi["mu_z"]),
    "dipole_corr_dmrg_bg_vs_shuoyi": corr(dmrg_bg["mu_z"], shuoyi["mu_z"]),
    "dipole_linear_fit_dmrg_bg_from_shuoyi": linear_fit_residual(shuoyi["mu_z"], dmrg_bg["mu_z"]),
    "dipole_rmse_dmrg_bg_vs_old_bg": rmse(dmrg_bg["mu_z"], old_bg["mu_z"]),
    "force_acc_max_abs_dmrg_bg": float(np.max(np.abs(dmrg_bg["force_acc"]))),
    "dip_acc_max_abs_dmrg_bg": float(np.max(np.abs(dmrg_bg["dip_acc"]))),
    "dip_acc_max_abs_shuoyi": float(np.max(np.abs(shuoyi["dip_acc"]))),
    "force_acc_rmse_dmrg_bg_vs_shuoyi_dip_acc": rmse(dmrg_bg["force_acc"], shuoyi["dip_acc"]),
    "dip_acc_rmse_dmrg_bg_vs_shuoyi": rmse(dmrg_bg["dip_acc"], shuoyi["dip_acc"]),
    "hhg_force_max_dmrg_bg": float(np.max(dmrg_bg["hhg_force"])),
    "hhg_dip_max_dmrg_bg": float(np.max(dmrg_bg["hhg_dip"])),
    "hhg_dip_max_shuoyi": float(np.max(shuoyi["hhg_dip"])),
    "energy": {
        "bg": dmrg_bg_energy,
        "shuoyi": shuoyi_energy,
        "differences": energy_differences,
    },
}
(dmrg_bg_run / "comparison_fig_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

# Dashboard.
fig = plt.figure(figsize=(12.5, 8.5))
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax.plot(t, old_bg["mu_z"], lw=0.9, alpha=0.75, label="BG old init")
ax.plot(t, dmrg_bg["mu_z"], lw=1.2, label="BG DMRG init")
ax.plot(t, shuoyi["mu_z"], lw=1.0, ls="--", label="Shuoyi U1")
ax.set_title("Dipole-like response")
ax.set_xlabel("time (a.u.)")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, old_bg["mu_z"] - shuoyi["mu_z"], lw=0.9, label="old BG - Shuoyi")
ax.plot(t, dmrg_bg["mu_z"] - shuoyi["mu_z"], lw=1.1, label="DMRG BG - Shuoyi")
ax.axhline(0, color="0.5", lw=0.7)
ax.set_title("Dipole residuals")
ax.set_xlabel("time (a.u.)")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 0])
ax.semilogy(harm[mask], np.maximum(dmrg_bg["hhg_force"][mask], 1e-18), label="BG DMRG force-form")
ax.semilogy(harm[mask], np.maximum(dmrg_bg["hhg_dip"][mask], 1e-18), label="BG DMRG dipole-2nd-deriv")
ax.semilogy(harm[mask], np.maximum(shuoyi["hhg_dip"][mask], 1e-18), ls="--", label="Shuoyi dipole-2nd-deriv")
ax.set_title("HHG-like spectra")
ax.set_xlabel("harmonic order")
ax.set_ylabel("power")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 1])
text = "\n".join(
    [
        f"old BG dipole RMSE vs Shuoyi = {summary['dipole_rmse_old_bg_vs_shuoyi']:.3e}",
        f"DMRG BG dipole RMSE vs Shuoyi = {summary['dipole_rmse_dmrg_bg_vs_shuoyi']:.3e}",
        f"DMRG BG dipole corr vs Shuoyi = {summary['dipole_corr_dmrg_bg_vs_shuoyi']:.6f}",
        f"BG = {summary['dipole_linear_fit_dmrg_bg_from_shuoyi']['scale']:.6f} * Shuoyi + "
        f"{summary['dipole_linear_fit_dmrg_bg_from_shuoyi']['offset']:.2e}",
        f"RMSE after scale/offset = "
        f"{summary['dipole_linear_fit_dmrg_bg_from_shuoyi']['rmse_after_scale_offset']:.3e}",
        f"DMRG BG force-acc max = {summary['force_acc_max_abs_dmrg_bg']:.3e}",
        f"DMRG BG dip-acc max = {summary['dip_acc_max_abs_dmrg_bg']:.3e}",
        f"Shuoyi dip-acc max = {summary['dip_acc_max_abs_shuoyi']:.3e}",
        f"force-acc RMSE vs Shuoyi dip-acc = {summary['force_acc_rmse_dmrg_bg_vs_shuoyi_dip_acc']:.3e}",
        f"dip-acc RMSE vs Shuoyi = {summary['dip_acc_rmse_dmrg_bg_vs_shuoyi']:.3e}",
        f"BG force HHG max = {summary['hhg_force_max_dmrg_bg']:.3e}",
        f"BG dipole HHG max = {summary['hhg_dip_max_dmrg_bg']:.3e}",
        f"Shuoyi HHG max = {summary['hhg_dip_max_shuoyi']:.3e}",
        f"HF-TOO E diff = {energy_differences['hf_too_energy_bg_minus_shuoyi']}",
        f"DMRG E diff = {energy_differences['dmrg_final_bg_minus_shuoyi']}",
    ]
)
ax.axis("off")
ax.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10.5)
fig.suptitle("BG old init vs BG DMRG init vs Shuoyi current branch")
fig.tight_layout()
fig.savefig(figdir / "00_bg_dmrg_init_comparison_dashboard.png", dpi=220)
plt.close(fig)

# Dipole overlay.
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(t, dmrg_bg["field"], color="0.2", label="field")
axes[0].set_ylabel("E_z")
axes[0].legend(frameon=False)
axes[1].plot(t, old_bg["mu_z"], lw=0.9, alpha=0.75, label="BG old init")
axes[1].plot(t, dmrg_bg["mu_z"], lw=1.2, label="BG DMRG init")
axes[1].plot(t, shuoyi["mu_z"], lw=1.0, ls="--", label="Shuoyi U1")
axes[1].set_ylabel("dipole / mu_z")
axes[1].legend(frameon=False, ncol=3)
axes[2].plot(t, old_bg["mu_z"] - shuoyi["mu_z"], label="old BG - Shuoyi", lw=0.9)
axes[2].plot(t, dmrg_bg["mu_z"] - shuoyi["mu_z"], label="DMRG BG - Shuoyi", lw=1.1)
axes[2].axhline(0, color="0.5", lw=0.7)
axes[2].set_ylabel("residual")
axes[2].set_xlabel("time (a.u.)")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figdir / "01_dipole_overlay_old_bg_dmrg_bg_shuoyi.png", dpi=220)
plt.close(fig)

# Acceleration diagnostics.
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(t, dmrg_bg["force"], label="BG DMRG raw force_mpo", lw=1)
axes[0].plot(t, dmrg_bg["force_acc"], label="BG DMRG force + NeE", lw=1)
axes[0].set_ylabel("force")
axes[0].legend(frameon=False)
axes[1].plot(t, dmrg_bg["dip_acc"], label="BG DMRG d2(mu_z)/dt2", lw=1.1)
axes[1].plot(t, shuoyi["dip_acc"], label="Shuoyi d2(dipole)/dt2", ls="--", lw=1.0)
axes[1].set_ylabel("dipole accel")
axes[1].legend(frameon=False)
axes[2].plot(t, dmrg_bg["force_acc"] - dmrg_bg["dip_acc"], label="BG force-form - BG dipole-2nd-deriv", lw=1.0)
axes[2].plot(t, dmrg_bg["dip_acc"] - shuoyi["dip_acc"], label="BG dipole-2nd-deriv - Shuoyi", lw=1.0)
axes[2].axhline(0, color="0.5", lw=0.7)
axes[2].set_ylabel("residual")
axes[2].set_xlabel("time (a.u.)")
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figdir / "02_acceleration_force_vs_dipole_diagnostic.png", dpi=220)
plt.close(fig)

# Spectrum comparison absolute and normalized.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
series = [
    ("BG old force", old_bg["hhg_force"]),
    ("BG DMRG force", dmrg_bg["hhg_force"]),
    ("BG DMRG dipole", dmrg_bg["hhg_dip"]),
    ("Shuoyi dipole", shuoyi["hhg_dip"]),
]
for label, power in series:
    axes[0].semilogy(harm[mask], np.maximum(power[mask], 1e-18), label=label)
    axes[1].semilogy(harm[mask], np.maximum(power[mask] / max(np.max(power), 1e-300), 1e-18), label=label)
axes[0].set_title("Absolute power")
axes[1].set_title("Normalized shape")
for ax in axes:
    ax.set_xlabel("harmonic order")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("power")
axes[1].set_ylabel("normalized power")
fig.tight_layout()
fig.savefig(figdir / "03_hhg_old_bg_dmrg_bg_shuoyi.png", dpi=220)
plt.close(fig)

print(json.dumps({"figdir": str(figdir), **summary}, indent=2, sort_keys=True))
