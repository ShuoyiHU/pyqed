"""Click-run comparisons with a separate energy-convergence figure per case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed._letta_one_site_opt.benchmarks.condensed_cli import parse_solvers
from pyqed._letta_one_site_opt.benchmarks.condensed_runner import SOLVERS, format_table, run_benchmark
from pyqed._letta_one_site_opt.benchmarks.run_condensed_suite import format_suite_table
from pyqed._letta_one_site_opt.benchmarks.nn_chain_cases import NN_CHAIN_CASES, NN_CHAIN_PARAMETERS
from pyqed._letta_one_site_opt.benchmarks.nn_chain_models import CHAIN_HAMILTONIAN_LABELS

# Each entry is (model, dimension, size, D). The random seed is set separately.
HARD_CASES = {
    # "hubbard_holstein_3_4": ("hubbard_holstein", "1d", 5, 4),  # Default max_phonons=1, d=8.
    # "hubbard_holstein_4_6": ("hubbard_holstein", "1d", 4, 6),
    # "hubbard_holstein_8_10": ("hubbard_holstein", "1d", 8, 10),
    # "ising_2x3": ("ising", "2d", (2, 3), 4),
    # "ising_2x3_alt": ("ising", "2d", (2, 3), 4),  # Use --seed for another start.
    # "heisenberg_2x3": ("heisenberg", "2d", (2, 3), 4),
    # "heisenberg_8": ("heisenberg", "1d", 8, 4),

    # "bose_hubbard_6_4": ("bose_hubbard", "1d", 6, 4),
    # "bose_hubbard_6_6": ("bose_hubbard", "1d", 6, 6),
    # "bose_hubbard_6_8": ("bose_hubbard", "1d", 6, 8),
    # "bose_hubbard_6_10": ("bose_hubbard", "1d", 6, 10),
    # "bose_hubbard_8_4": ("bose_hubbard", "1d", 8, 4),
    # "bose_hubbard_8_6": ("bose_hubbard", "1d", 8, 6),
    # "bose_hubbard_8_8": ("bose_hubbard", "1d", 8, 8),
    # "bose_hubbard_8_10": ("bose_hubbard", "1d", 8, 10),
    # "bose_hubbard_10_4": ("bose_hubbard", "1d", 10, 4),
    # "bose_hubbard_10_6": ("bose_hubbard", "1d", 10, 6),

    # "fermi_hubbard_6_4": ("fermi_hubbard", "1d", 6, 4),
    # "fermi_hubbard_6_6": ("fermi_hubbard", "1d", 6, 6),
    # "fermi_hubbard_6_8": ("fermi_hubbard", "1d", 6, 8),
    # "fermi_hubbard_8_4": ("fermi_hubbard", "1d", 8, 4),
    # "fermi_hubbard_8_6": ("fermi_hubbard", "1d", 8, 6),
    # "fermi_hubbard_8_8": ("fermi_hubbard", "1d", 8, 8),
    # "fermi_hubbard_10_4": ("fermi_hubbard", "1d", 10, 4),
    # "fermi_hubbard_10_6": ("fermi_hubbard", "1d", 10, 6),
    # "fermi_hubbard_10_8": ("fermi_hubbard", "1d", 10, 8),

    # 1D NN click-run menu: uncomment the (L, D) rows you want to run.
    # One row per family is enabled below; model parameters use their defaults.

    # # Spinless SSH: alternating t1=0.6, t2=1.4
    # "ssh_6_2": ("ssh", "1d", 6, 2),
    # "ssh_6_4": ("ssh", "1d", 6, 4),
    # "ssh_6_6": ("ssh", "1d", 6, 6),
    # "ssh_8_2": ("ssh", "1d", 8, 2),
    # "ssh_8_4": ("ssh", "1d", 8, 4),
    # "ssh_8_6": ("ssh", "1d", 8, 6),
    # "ssh_10_2": ("ssh", "1d", 10, 2),
    # "ssh_10_4": ("ssh", "1d", 10, 4),
    # "ssh_10_6": ("ssh", "1d", 10, 6),

    # # 1D transverse-field Ising: J=1, h=1
    # "ising_6_2": ("ising", "1d", 6, 2),
    # "ising_6_4": ("ising", "1d", 6, 4),
    # "ising_6_6": ("ising", "1d", 6, 6),
    # "ising_8_2": ("ising", "1d", 8, 2),
    # "ising_8_4": ("ising", "1d", 8, 4),
    # "ising_8_6": ("ising", "1d", 8, 6),
    # "ising_10_2": ("ising", "1d", 10, 2),
    # "ising_10_4": ("ising", "1d", 10, 4),
    # "ising_10_6": ("ising", "1d", 10, 6),

    # # Spin-1/2 Heisenberg chain: J=1, delta=1
    # "heisenberg_6_2": ("heisenberg", "1d", 6, 2),
    # "heisenberg_6_4": ("heisenberg", "1d", 6, 4),
    # "heisenberg_6_6": ("heisenberg", "1d", 6, 6),
    # "heisenberg_8_2": ("heisenberg", "1d", 8, 2),
    # "heisenberg_8_4": ("heisenberg", "1d", 8, 4),
    # "heisenberg_8_6": ("heisenberg", "1d", 8, 6),
    # "heisenberg_10_2": ("heisenberg", "1d", 10, 2),
    # "heisenberg_10_4": ("heisenberg", "1d", 10, 4),
    # "heisenberg_10_6": ("heisenberg", "1d", 10, 6),

    # # Rice-Mele: alternating hopping and stagger=0.4
    # "rice_mele_6_2": ("rice_mele", "1d", 6, 2),
    # "rice_mele_6_4": ("rice_mele", "1d", 6, 4),
    # "rice_mele_6_6": ("rice_mele", "1d", 6, 6),
    # "rice_mele_8_2": ("rice_mele", "1d", 8, 2),
    # "rice_mele_8_4": ("rice_mele", "1d", 8, 4),
    # "rice_mele_8_6": ("rice_mele", "1d", 8, 6),
    # "rice_mele_10_2": ("rice_mele", "1d", 10, 2),
    # "rice_mele_10_4": ("rice_mele", "1d", 10, 4),
    # "rice_mele_10_6": ("rice_mele", "1d", 10, 6),

    # # Spinless t-V: t=1, V=2, mu=0 (centered density)
    # "spinless_tv_6_2": ("spinless_tv", "1d", 6, 2),
    # "spinless_tv_6_4": ("spinless_tv", "1d", 6, 4),
    # "spinless_tv_6_6": ("spinless_tv", "1d", 6, 6),
    # "spinless_tv_8_2": ("spinless_tv", "1d", 8, 2),
    # "spinless_tv_8_4": ("spinless_tv", "1d", 8, 4),
    # "spinless_tv_8_6": ("spinless_tv", "1d", 8, 6),
    # "spinless_tv_10_2": ("spinless_tv", "1d", 10, 2),
    # "spinless_tv_10_4": ("spinless_tv", "1d", 10, 4),
    # "spinless_tv_10_6": ("spinless_tv", "1d", 10, 6),

    # # Kitaev chain: t=1, pairing=0.6, mu=0.5
    # "kitaev_6_2": ("kitaev", "1d", 6, 2),
    # "kitaev_6_4": ("kitaev", "1d", 6, 4),
    # "kitaev_6_6": ("kitaev", "1d", 6, 6),
    # "kitaev_8_2": ("kitaev", "1d", 8, 2),
    # "kitaev_8_4": ("kitaev", "1d", 8, 4),
    # "kitaev_8_6": ("kitaev", "1d", 8, 6),
    # "kitaev_10_2": ("kitaev", "1d", 10, 2),
    # "kitaev_10_4": ("kitaev", "1d", 10, 4),
    # "kitaev_10_6": ("kitaev", "1d", 10, 6),

    # # Spin-1/2 XY: J=1, gamma=0.5, h=0.3
    # "xy_6_2": ("xy", "1d", 6, 2),
    # "xy_6_4": ("xy", "1d", 6, 4),
    # "xy_6_6": ("xy", "1d", 6, 6),
    # "xy_8_2": ("xy", "1d", 8, 2),
    # "xy_8_4": ("xy", "1d", 8, 4),
    # "xy_8_6": ("xy", "1d", 8, 6),
    # "xy_10_2": ("xy", "1d", 10, 2),
    # "xy_10_4": ("xy", "1d", 10, 4),
    # "xy_10_6": ("xy", "1d", 10, 6),

    # # Spin-1/2 XYZ: Jx=1, Jy=0.7, Jz=1.3, h=0.2
    # "xyz_6_2": ("xyz", "1d", 6, 2),
    # "xyz_6_4": ("xyz", "1d", 6, 4),
    # "xyz_6_6": ("xyz", "1d", 6, 6),
    # "xyz_8_2": ("xyz", "1d", 8, 2),
    # "xyz_8_4": ("xyz", "1d", 8, 4),
    # "xyz_8_6": ("xyz", "1d", 8, 6),
    # "xyz_10_2": ("xyz", "1d", 10, 2),
    # "xyz_10_4": ("xyz", "1d", 10, 4),
    # "xyz_10_6": ("xyz", "1d", 10, 6),

    # # Dimerized Heisenberg: J=1, dimer=0.4, delta=1
    # "dimerized_heisenberg_6_2": ("dimerized_heisenberg", "1d", 6, 2),
    # "dimerized_heisenberg_6_4": ("dimerized_heisenberg", "1d", 6, 4),
    # "dimerized_heisenberg_6_6": ("dimerized_heisenberg", "1d", 6, 6),
    # "dimerized_heisenberg_8_2": ("dimerized_heisenberg", "1d", 8, 2),
    # "dimerized_heisenberg_8_4": ("dimerized_heisenberg", "1d", 8, 4),
    # "dimerized_heisenberg_8_6": ("dimerized_heisenberg", "1d", 8, 6),
    # "dimerized_heisenberg_10_2": ("dimerized_heisenberg", "1d", 10, 2),
    # "dimerized_heisenberg_10_4": ("dimerized_heisenberg", "1d", 10, 4),
    # "dimerized_heisenberg_10_6": ("dimerized_heisenberg", "1d", 10, 6),

    # # Spin-1 XXZ: J=1, delta=1, single_ion=0
    # "spin1_heisenberg_4_2": ("spin1_heisenberg", "1d", 4, 2),
    # "spin1_heisenberg_4_3": ("spin1_heisenberg", "1d", 4, 3),
    # "spin1_heisenberg_4_4": ("spin1_heisenberg", "1d", 4, 4),
    # "spin1_heisenberg_6_2": ("spin1_heisenberg", "1d", 6, 2),
    # "spin1_heisenberg_6_3": ("spin1_heisenberg", "1d", 6, 3),
    # "spin1_heisenberg_6_4": ("spin1_heisenberg", "1d", 6, 4),
    # "spin1_heisenberg_8_2": ("spin1_heisenberg", "1d", 8, 2),
    # "spin1_heisenberg_8_3": ("spin1_heisenberg", "1d", 8, 3),
    # "spin1_heisenberg_8_4": ("spin1_heisenberg", "1d", 8, 4),

    # Spin-1 Blume-Capel: J=1, h=0.7, single_ion=0.5
    "blume_capel_4_2": ("blume_capel", "1d", 4, 2),
    "blume_capel_4_3": ("blume_capel", "1d", 4, 3),
    "blume_capel_4_4": ("blume_capel", "1d", 4, 4),
    "blume_capel_6_2": ("blume_capel", "1d", 6, 2),
    "blume_capel_6_3": ("blume_capel", "1d", 6, 3),
    "blume_capel_6_4": ("blume_capel", "1d", 6, 4),
    "blume_capel_8_2": ("blume_capel", "1d", 8, 2),
    "blume_capel_8_3": ("blume_capel", "1d", 8, 3),
    "blume_capel_8_4": ("blume_capel", "1d", 8, 4),

    # "ising_3x3": ("ising", "2d", (3, 3), 6),
}
# Click Run uses every uncommented entry above, with its own L and D.
DEFAULT_CASES = tuple(HARD_CASES)
# Register optional chains after selecting the user's existing click-run cases.
for _name, _case in NN_CHAIN_CASES.items():
    HARD_CASES.setdefault(_name, _case)
# To click-run every new chain preset, use: DEFAULT_CASES = tuple(NN_CHAIN_CASES)
CASE_PARAMETERS = dict(NN_CHAIN_PARAMETERS)
DEFAULT_FIGURE_DIR = ROOT / "output" / "benchmarks" / "cbe_harder_cases"
INCLUDE_CBE_EXACT = False  # Set True for click-run, or pass --include-cbe-exact.
DEFAULT_SOLVERS = ("letta_one_site", "letta_two_site", "letta_cbe_strict")
PLOT_LINTHRESH = 1.e-10  # Signed-log plots are linear within +/- this energy difference.

PLOT_METHODS = (
    ("letta_one_site", "One-site LETTA", "#0072B2", "o", "-"),
    ("letta_two_site", "Two-site LETTA", "#D55E00", "s", "--"),
    ("letta_cbe_strict", "Strict CBE LETTA", "#009E73", "^", "-."),
)
HAMILTONIAN_LABELS = {
    **CHAIN_HAMILTONIAN_LABELS,
    "bose_hubbard": ("Bose-Hubbard", (
        r"$H=-t\sum_{\langle i,j\rangle}(b_i^\dagger b_j+\mathrm{h.c.})"
        r"+\frac{U}{2}\sum_i n_i(n_i-1)-\mu\sum_i n_i$",
    )),
    "fermi_hubbard": ("Fermi-Hubbard", (
        r"$H=-t\sum_{\langle i,j\rangle,\sigma}(c_{i\sigma}^\dagger c_{j\sigma}+\mathrm{h.c.})"
        r"+U\sum_i n_{i\uparrow}n_{i\downarrow}-\mu\sum_i n_i$",
    )),
    "hubbard_holstein": ("Hubbard-Holstein", (
        r"$H=-t\sum_{\langle i,j\rangle,\sigma}(c_{i\sigma}^\dagger c_{j\sigma}+\mathrm{h.c.})"
        r"+U\sum_i n_{i\uparrow}n_{i\downarrow}-\mu\sum_i n_i$",
        r"$\quad+\omega\sum_i b_i^\dagger b_i+g\sum_i(n_i-1)(b_i+b_i^\dagger)$",
    )),
    "ising": ("Transverse-field Ising", (
        r"$H=-J\sum_{\langle i,j\rangle}\sigma_i^z\sigma_j^z-h\sum_i\sigma_i^x$",
    )),
    "heisenberg": ("XXZ Heisenberg", (
        r"$H=J\sum_{\langle i,j\rangle}(S_i^x S_j^x+S_i^y S_j^y"
        r"+\Delta S_i^z S_j^z)-h\sum_i S_i^z$",
    )),
}


def build_convergence_figure(case):
    """Plot signed differences from the final two-site energy, without padding."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    fig = Figure(figsize=(8, 5.5), facecolor="white")
    FigureCanvasAgg(fig)  # Headless export; never open or block on a GUI window.
    ax = fig.add_subplot(111)
    title, equations = HAMILTONIAN_LABELS[case["model"]]
    geometry = (f"1D chain, L={case['nsites']}" if case["dimension"] == "1d"
                else f"{case['dimension'].upper()} lattice, shape={tuple(case['shape'])}, N={case['nsites']}")
    fig.suptitle(f"{title} | {geometry} | D={case['bond_dim']}", y=.97, fontsize=12)
    for line, equation in enumerate(equations):
        fig.text(.5, .90 - .065 * line, equation, ha="center", va="center", fontsize=11)
    parameters_y = .90 - .065 * len(equations)
    parameters = ", ".join(f"{key}={value:g}" for key, value in case["parameters"].items())
    fig.text(.5, parameters_y, parameters, ha="center", fontsize=9)
    fig.subplots_adjust(left=.15, right=.97, bottom=.20, top=parameters_y - .045)
    records = {row["solver"]: row for row in case["records"]}
    reference_row = records.get("letta_two_site", {})
    reference = reference_row.get("energy")
    has_reference = reference is not None and np.isfinite(reference)
    missing, longest = [], 1
    lowest, highest = -PLOT_LINTHRESH, PLOT_LINTHRESH
    for solver, label, color, marker, linestyle in PLOT_METHODS:
        energies = records.get(solver, {}).get("sweep_energies", [])
        if not energies:
            missing.append(label)
            continue
        if not has_reference:
            continue
        longest = max(longest, len(energies))
        difference = np.asarray(energies) - reference
        lowest = min(lowest, float(np.min(difference)) * 1.5)
        highest = max(highest, float(np.max(difference)) * 1.5)
        ax.plot(range(1, len(energies) + 1), difference, label=label,
                color=color, marker=marker, linestyle=linestyle,
                linewidth=1.5, markersize=3.5, markerfacecolor="none")
        if solver == "letta_cbe_strict":
            accepted = records[solver].get("sweep_cbe_accepted", [])
            indices = [index for index, count in enumerate(accepted)
                       if count > 0 and index < len(energies)]
            if indices:
                ax.scatter(np.asarray(indices) + 1, difference[indices],
                           color="magenta", marker=marker, s=3.5 ** 2,
                           zorder=3, label="Strict CBE: ≥1 CBE-ok update")
    ax.set_xlabel("Iteration (one directional pass: LR or RL)")
    ax.set_ylabel(r"$E(k)-E_{\mathrm{two-site,final}}$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(.75, longest + .25)
    ax.set_yscale("symlog", linthresh=PLOT_LINTHRESH, base=10)
    ax.set_ylim(lowest, highest)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2, linewidth=.6)
    if ax.lines:
        ax.legend(frameon=False, fontsize=9)
        ax.axhline(0, color="0.4", linewidth=.7, zorder=0)
    if has_reference:
        status = "" if reference_row.get("converged") else " (not converged)"
        note = f"Reference: final two-site energy = {reference:.12f}{status}."
        note += (f"\nSigned log scale; linear for |ΔE| < {PLOT_LINTHRESH:g}. "
                 "Negative = below reference.")
    else:
        note = "Two-site reference not available; energy differences cannot be plotted."
    if missing and has_reference:
        note += "\nNot run / not available: " + ", ".join(missing)
    fig.text(.5, .025, note, ha="center", fontsize=8, color="0.35")
    return fig


def save_convergence_figure(case, directory):
    """Save a case's PNG/PDF and original result data; same-name reruns overwrite."""
    import matplotlib as mpl

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["case_name"])
    stem = f"{name}_L{case['nsites']}_D{case['bond_dim']}_energy_difference"
    paths = {extension: str(directory / f"{stem}.{extension}")
             for extension in ("png", "pdf", "json")}
    # Preserve numerical results even if figure rendering fails.
    reference = next((row["energy"] for row in case["records"]
                      if row["solver"] == "letta_two_site"), None)
    source = dict(case, plot={"quantity": "E(k) - final two-site energy",
                             "reference_energy": reference, "scale": "symlog",
                             "linthresh": PLOT_LINTHRESH})
    Path(paths["json"]).write_text(json.dumps(source, indent=2) + "\n")
    with mpl.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "text.usetex": False}):
        fig = build_convergence_figure(case)
        try:
            fig.savefig(paths["png"], dpi=200)
            fig.savefig(paths["pdf"])
        finally:
            fig.clear()
    return paths


def run_harder_suite(*, cases=DEFAULT_CASES, bond_dim=None, max_sweeps=50,
                     seed=731, solvers=None, cbe_conditional_trim=True,
                     expansion_dimension=1, exact_max_dimension=4096,
                     output=None, progress=False, figure_dir=None,
                     include_cbe_exact=INCLUDE_CBE_EXACT):
    """Use each case's D unless bond_dim explicitly overrides every case."""
    cases = tuple(cases)
    unknown = set(cases) - set(HARD_CASES)
    if unknown or not cases:
        raise ValueError(f"unknown or empty harder cases: {sorted(unknown)}")
    solvers = tuple(DEFAULT_SOLVERS if solvers is None else solvers)
    if include_cbe_exact and "letta_cbe_exact" not in solvers:
        solvers += ("letta_cbe_exact",)
    report = {"cases": [], "failures": {}, "plot_failures": {}, "requested_cases": list(cases),
              "solvers": list(solvers),
              "bond_dim": bond_dim, "max_sweeps": max_sweeps, "seed": seed,
              "cbe_conditional_trim": bool(cbe_conditional_trim)}
    for name in cases:
        model, dimension, size, case_bond_dim = HARD_CASES[name]
        effective_bond_dim = case_bond_dim if bond_dim is None else bond_dim
        if progress:
            print(f"Starting {name}: D={effective_bond_dim}, seed={seed}", flush=True)
        try:
            case = run_benchmark(
                model, dimension=dimension, size=size, bond_dim=effective_bond_dim,
                model_parameters=CASE_PARAMETERS.get(name),
                max_sweeps=max_sweeps, seed=seed, solvers=solvers,
                cbe_conditional_trim=cbe_conditional_trim,
                expansion_dimension=expansion_dimension,
                exact_max_dimension=exact_max_dimension,
            )
            case["case_name"] = name
            report["cases"].append(case)
            if progress:
                print(format_table(case), flush=True)
            if figure_dir is not None:
                try:
                    case["figures"] = save_convergence_figure(case, figure_dir)
                    if progress:
                        print(f"Saved figure: {case['figures']['png']}", flush=True)
                except Exception as error:
                    report["plot_failures"][name] = f"{type(error).__name__}: {error}"
                    if progress:
                        print(f"{name} plot FAILED (solver results retained): {error}", flush=True)
        except Exception as error:
            report["failures"][name] = f"{type(error).__name__}: {error}"
            if progress:
                print(f"{name} FAILED: {error}", flush=True)
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cases", type=lambda value: tuple(value.split(",")),
                        default=DEFAULT_CASES, help=", ".join(HARD_CASES))
    selection.add_argument("--nn-chains", action="store_true",
                           help="run every optional 1D nearest-neighbor chain preset")
    parser.add_argument("--list-cases", action="store_true",
                        help="list case sizes, bond dimensions and parameter overrides, then exit")
    parser.add_argument("--bond-dim", type=int, default=None,
                        help="override D for every selected case; default: fourth value in HARD_CASES")
    parser.add_argument("--max-sweeps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--expansion-dimension", type=int, default=1)
    parser.add_argument("--exact-max-dimension", type=int, default=4096)
    parser.add_argument("--solvers", type=parse_solvers, default=DEFAULT_SOLVERS)
    parser.add_argument("--include-cbe-exact", action="store_true", default=INCLUDE_CBE_EXACT,
                        help="add the expensive exact CBE selector (off by default)")
    parser.add_argument("--global-trim", action="store_true",
                        help="disable shared-physical conditional compression; retain metric-aware SVD/ALS")
    parser.add_argument("--output", type=Path, help="save results after every completed case")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR,
                        help="directory for separate per-case PNG, PDF, and source JSON files")
    parser.add_argument("--no-plots", action="store_true", help="disable figure exports")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.list_cases:
        for name, (model, dimension, size, bond_dim) in HARD_CASES.items():
            print(f"{name}: {model}, {dimension}, size={size}, D={bond_dim}, "
                  f"overrides={CASE_PARAMETERS.get(name, {})}")
        return
    report = run_harder_suite(
        cases=tuple(NN_CHAIN_CASES) if args.nn_chains else args.cases,
        bond_dim=args.bond_dim, max_sweeps=args.max_sweeps,
        seed=args.seed, solvers=args.solvers, cbe_conditional_trim=not args.global_trim,
        expansion_dimension=args.expansion_dimension, exact_max_dimension=args.exact_max_dimension,
        output=args.output, progress=not args.json,
        figure_dir=None if args.no_plots else args.figure_dir,
        include_cbe_exact=args.include_cbe_exact,
    )
    print(json.dumps(report, indent=2) if args.json else format_suite_table(report))
    return report


if __name__ == "__main__":
    main()
