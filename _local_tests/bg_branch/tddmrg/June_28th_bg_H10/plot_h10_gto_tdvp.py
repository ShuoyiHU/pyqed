#!/usr/bin/env python3
"""Generate quick-look figures for one H10 GTO-TDVP output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def finite_or_none(values):
    values = np.asarray(values)
    mask = np.isfinite(values)
    return values[mask] if np.any(mask) else None


def plot_field_dipole(output_dir, data, setup, figdir):
    t = np.asarray(data["times"], dtype=float)
    field = np.asarray(data["field_z"], dtype=float)
    mu = np.asarray(data["mu_z"], dtype=float)
    acc = np.asarray(data.get("dipole_acceleration", np.full_like(mu, np.nan)), dtype=float)

    title_bits = []
    geom = setup.get("geometry", {}).get("name")
    intensity = setup.get("field", {}).get("intensity")
    basis = setup.get("molecule", {}).get("basis")
    if geom:
        title_bits.append(str(geom))
    if basis:
        title_bits.append(str(basis))
    if intensity:
        title_bits.append(str(intensity))
    title = "H10 GTO TDVP" + (": " + ", ".join(title_bits) if title_bits else "")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, field, color="0.2", lw=1.6)
    axes[0].set_ylabel("E_z(t)")
    axes[0].set_title(title)
    axes[0].axhline(0.0, color="0.75", lw=0.8)

    axes[1].plot(t, mu, color="#1f77b4", lw=1.4)
    axes[1].set_ylabel("mu_z")
    axes[1].axhline(0.0, color="0.75", lw=0.8)

    axes[2].plot(t, acc, color="#d62728", lw=1.2)
    axes[2].set_ylabel("d2(mu_z)/dt2")
    axes[2].set_xlabel("time (a.u.)")
    axes[2].axhline(0.0, color="0.75", lw=0.8)

    fig.savefig(figdir / "01_field_dipole_acceleration.png", dpi=180)
    plt.close(fig)


def plot_tdvp_diagnostics(data, figdir):
    t = np.asarray(data["times"], dtype=float)
    norm2 = np.asarray(data.get("pre_normalization_norm2", []), dtype=float)
    survival = np.asarray(data.get("estimated_norm2_without_step_renormalization", []), dtype=float)
    trunc = np.asarray(data.get("tdvp_truncation_errors", []), dtype=float)
    step_t = t[: max(len(norm2), len(trunc), len(survival))]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False, constrained_layout=True)
    if norm2.size:
        axes[0].plot(step_t[: len(norm2)], norm2, lw=1.2)
    axes[0].set_ylabel("step norm2")
    axes[0].axhline(1.0, color="0.75", lw=0.8)

    if survival.size:
        axes[1].plot(step_t[: len(survival)], survival, lw=1.2)
    axes[1].set_ylabel("cum norm2")
    axes[1].axhline(1.0, color="0.75", lw=0.8)

    if trunc.size:
        axes[2].semilogy(step_t[: len(trunc)], np.maximum(trunc, 1.0e-300), lw=1.2)
    axes[2].set_ylabel("TDVP trunc")
    axes[2].set_xlabel("time (a.u.)")

    fig.savefig(figdir / "02_tdvp_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_energy(data, figdir):
    if "energy_times" not in data or "static_energies_real" not in data:
        return
    et = np.asarray(data["energy_times"], dtype=float)
    ere = np.asarray(data["static_energies_real"], dtype=float)
    eim = np.asarray(data.get("static_energies_imag", np.zeros_like(ere)), dtype=float)
    dre = np.asarray(data.get("energy_drift_real", np.full_like(ere, np.nan)), dtype=float)
    if et.size == 0 or ere.size == 0:
        return
    n = min(et.size, ere.size, eim.size, dre.size)
    et = et[:n]
    ere = ere[:n]
    eim = eim[:n]
    dre = dre[:n]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(et, ere, label="Re <H0>", lw=1.2)
    if finite_or_none(eim) is not None and np.nanmax(np.abs(eim)) > 0.0:
        axes[0].plot(et, eim, label="Im <H0>", lw=1.0)
    axes[0].set_ylabel("energy (Ha)")
    axes[0].legend(frameon=False)

    axes[1].plot(et, dre, color="#d62728", lw=1.2)
    axes[1].axhline(0.0, color="0.75", lw=0.8)
    axes[1].set_ylabel("energy drift")
    axes[1].set_xlabel("time (a.u.)")

    fig.savefig(figdir / "03_energy_trace.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    data_file = output_dir / "td_timeseries_latest.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"missing data file: {data_file}")

    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    data = np.load(data_file)
    setup = safe_load_json(output_dir / "setup.json")

    plot_field_dipole(output_dir, data, setup, figdir)
    plot_tdvp_diagnostics(data, figdir)
    plot_energy(data, figdir)

    summary = {
        "output_dir": str(output_dir),
        "figures_dir": str(figdir),
        "n_time_points": int(np.asarray(data["times"]).size),
        "mu_z_max_abs": float(np.nanmax(np.abs(np.asarray(data["mu_z"], dtype=float)))),
        "field_z_max_abs": float(np.nanmax(np.abs(np.asarray(data["field_z"], dtype=float)))),
    }
    (figdir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
