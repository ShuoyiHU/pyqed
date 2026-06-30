#!/usr/bin/env python3
"""Plot time-resolved H10 GTO charge density if density data were saved.

Expected density archive format:

    density_timeseries_latest.npz with keys:
        times or density_times          shape (nt,)
        z_grid, z, or z_points          shape (nz,)
        rho_z or charge_density_z       shape (nt, nz)

The script also accepts checkpoint .npz files containing the same density keys.
Scalar-only TD output files cannot be converted into a density heat map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DENSITY_KEYS = (
    "rho_z",
    "charge_density_z",
    "electron_density_z",
    "density_z",
)
TIME_KEYS = ("times", "density_times", "time")
Z_KEYS = ("z_grid", "z", "z_points")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def first_existing_key(data, candidates):
    for key in candidates:
        if key in data:
            return key
    return None


def normalize_density_array(rho, times, z_grid, source):
    rho = np.asarray(rho)
    times = np.asarray(times, dtype=float).reshape(-1)
    z_grid = np.asarray(z_grid, dtype=float).reshape(-1)

    if rho.ndim == 1:
        rho = rho.reshape(1, -1)
    if rho.ndim != 2:
        raise ValueError(f"{source}: density array must be 1D or 2D, got shape {rho.shape}")

    if rho.shape == (z_grid.size, times.size):
        rho = rho.T
    if rho.shape[1] != z_grid.size:
        raise ValueError(
            f"{source}: density second dimension must match z grid size "
            f"({rho.shape[1]} != {z_grid.size})"
        )
    if rho.shape[0] != times.size:
        if times.size == 1:
            times = np.full(rho.shape[0], float(times[0]))
        else:
            raise ValueError(
                f"{source}: density first dimension must match time size "
                f"({rho.shape[0]} != {times.size})"
            )
    return times, z_grid, np.real_if_close(rho).real


def load_density_archive(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        dkey = first_existing_key(data, DENSITY_KEYS)
        tkey = first_existing_key(data, TIME_KEYS)
        zkey = first_existing_key(data, Z_KEYS)
        if dkey is None or tkey is None or zkey is None:
            return None
        return normalize_density_array(data[dkey], data[tkey], data[zkey], path)


def load_from_density_files(output_dir):
    candidates = [
        output_dir / "density_timeseries_latest.npz",
        output_dir / "charge_density_latest.npz",
        output_dir / "td_density_latest.npz",
        output_dir / "td_timeseries_latest.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        loaded = load_density_archive(path)
        if loaded is not None:
            return loaded, path
    return None, None


def load_from_checkpoints(output_dir):
    checkpoints = sorted((output_dir / "checkpoints").glob("*.npz"))
    if not checkpoints:
        return None, None

    # Prefer the latest cumulative checkpoint if it already contains a full series.
    for path in reversed(checkpoints):
        loaded = load_density_archive(path)
        if loaded is None:
            continue
        times, z_grid, rho = loaded
        if rho.shape[0] > 1:
            return loaded, path

    rows = []
    row_times = []
    z_ref = None
    used = []
    for path in checkpoints:
        loaded = load_density_archive(path)
        if loaded is None:
            continue
        times, z_grid, rho = loaded
        if z_ref is None:
            z_ref = z_grid
        elif not np.allclose(z_ref, z_grid, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: z grid differs from earlier checkpoint density grid")
        rows.append(rho)
        row_times.append(times)
        used.append(path)

    if not rows:
        return None, None
    return (np.concatenate(row_times), z_ref, np.vstack(rows)), used[-1]


def load_density(output_dir):
    loaded, source = load_from_density_files(output_dir)
    if loaded is not None:
        return loaded, source
    loaded, source = load_from_checkpoints(output_dir)
    if loaded is not None:
        return loaded, source
    raise FileNotFoundError(
        "No saved density data found. I looked for density_timeseries_latest.npz "
        "or checkpoint .npz files containing one of "
        f"{DENSITY_KEYS}. Existing scalar files such as td_timeseries_latest.npz "
        "are not enough to reconstruct rho(z,t)."
    )


def atom_z_from_setup(setup):
    geometry = setup.get("geometry", {})
    atom_z = geometry.get("atom_z")
    if atom_z is not None:
        return np.asarray(atom_z, dtype=float)
    coords = geometry.get("coords") or setup.get("molecule", {}).get("atom")
    zs = []
    if coords:
        for item in coords:
            try:
                if len(item) == 2 and isinstance(item[1], (list, tuple)):
                    zs.append(float(item[1][2]))
                else:
                    zs.append(float(item[2]))
            except Exception:
                pass
    return np.asarray(zs, dtype=float)


def title_from_setup(setup, fallback):
    bits = []
    geom = setup.get("geometry", {}).get("name")
    basis = setup.get("molecule", {}).get("basis")
    intensity = setup.get("field", {}).get("intensity")
    for value in (geom, basis, intensity):
        if value:
            bits.append(str(value))
    return "H10 GTO density" + (": " + ", ".join(bits) if bits else f": {fallback.name}")


def robust_limits(values, symmetric=False):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), 99.0))
        return -vmax, vmax
    return (
        float(np.nanpercentile(finite, 1.0)),
        float(np.nanpercentile(finite, 99.0)),
    )


def plot_heatmap(times, z_grid, rho, setup, output_dir, figdir, *, delta=False, dpi=180):
    values = rho - rho[0:1, :] if delta else rho
    vmin, vmax = robust_limits(values, symmetric=delta)
    cmap = "RdBu_r" if delta else "viridis"
    label = "Delta rho(z,t)" if delta else "rho(z,t)"
    fname = "05_charge_density_delta_heatmap.png" if delta else "04_charge_density_heatmap.png"

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    image = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=(float(times[0]), float(times[-1]), float(z_grid[0]), float(z_grid[-1])),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("z (bohr)")
    ax.set_title(title_from_setup(setup, output_dir) + (" delta" if delta else ""))

    atom_z = atom_z_from_setup(setup)
    for z in atom_z:
        ax.axhline(float(z), color="w", lw=0.7, alpha=0.45)
        ax.axhline(float(z), color="k", lw=0.25, alpha=0.35)

    out = figdir / fname
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="One H10 GTO TDVP result directory.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Only plot rho(z,t), not rho(z,t)-rho(z,0).",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    figdir = output_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    try:
        (times, z_grid, rho), source = load_density(output_dir)
    except FileNotFoundError as exc:
        print(f"[density plot] {exc}", flush=True)
        return 2
    order = np.argsort(times)
    times = times[order]
    rho = rho[order]

    setup = read_json(output_dir / "setup.json")
    figures = [plot_heatmap(times, z_grid, rho, setup, output_dir, figdir, delta=False, dpi=args.dpi)]
    if not args.no_delta and rho.shape[0] > 1:
        figures.append(plot_heatmap(times, z_grid, rho, setup, output_dir, figdir, delta=True, dpi=args.dpi))

    summary = {
        "output_dir": str(output_dir),
        "source": str(source),
        "figures": [str(path) for path in figures],
        "n_times": int(times.size),
        "n_z": int(z_grid.size),
        "time_min": float(times[0]),
        "time_max": float(times[-1]),
        "z_min": float(z_grid[0]),
        "z_max": float(z_grid[-1]),
        "rho_min": float(np.nanmin(rho)),
        "rho_max": float(np.nanmax(rho)),
    }
    (figdir / "density_plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
