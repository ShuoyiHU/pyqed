#!/usr/bin/env python3
"""Local STO-6G H10 GTO-TDDMRG test driver.

Default mode is a short smoke test for a laptop. Use ``--mode full`` only when
you are ready to run all 3 geometries x 3 intensities locally.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
RUNNER = REPO_ROOT / "_local_tests" / "bg_branch" / "tddmrg" / "June_28th_bg_H10" / "h10_gto_tdvp.py"
DEFAULT_OUTPUT_ROOT = SCRIPT_PATH.with_name("benchmark_results")

GEOMETRIES = ("afm", "bonding", "edge_localized")
INTENSITIES = ("off", "I1e13", "I5e14")
AMPLITUDES = {
    "off": "0.0",
    "I1e13": "0.016880323915389028",
    "I5e14": "0.11936191509197033",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "single", "full"),
        default="smoke",
        help="smoke: tiny local check; single: one production-like case; full: 9 cases.",
    )
    parser.add_argument("--geometry", choices=GEOMETRIES, default="bonding")
    parser.add_argument("--intensity", choices=INTENSITIES, default="off")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bond-dim", type=int, default=40)
    parser.add_argument("--dmrg-sweeps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega", type=float, default=0.05841455452769231)
    parser.add_argument("--cycles", type=float, default=2.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--track-energy", action="store_true")
    return parser.parse_args(argv)


def case_settings(args):
    if args.mode == "smoke":
        return [
            {
                "geometry": args.geometry,
                "intensity": args.intensity,
                "ncas": 2,
                "nelecas": 2,
                "bond_dim": 2,
                "dmrg_sweeps": 1,
                "cycles": 0.001,
                "save_every": 1,
                "suffix": "smoke",
            }
        ]
    if args.mode == "single":
        return [
            {
                "geometry": args.geometry,
                "intensity": args.intensity,
                "ncas": 10,
                "nelecas": 10,
                "bond_dim": args.bond_dim,
                "dmrg_sweeps": args.dmrg_sweeps,
                "cycles": args.cycles,
                "save_every": args.save_every,
                "suffix": "single",
            }
        ]

    cases = []
    for geometry in GEOMETRIES:
        for intensity in INTENSITIES:
            cases.append(
                {
                    "geometry": geometry,
                    "intensity": intensity,
                    "ncas": 10,
                    "nelecas": 10,
                    "bond_dim": args.bond_dim,
                    "dmrg_sweeps": args.dmrg_sweeps,
                    "cycles": args.cycles,
                    "save_every": args.save_every,
                    "suffix": "full",
                }
            )
    return cases


def run_case(args, case):
    output_dir = (
        args.output_root.resolve()
        / f"h10_{case['geometry']}_gto_sto6g_{case['suffix']}_CAS{case['ncas']}_{case['nelecas']}_"
        f"D{case['bond_dim']}_{case['intensity']}"
    )
    cmd = [
        args.python,
        str(RUNNER),
        "--geometry",
        case["geometry"],
        "--intensity",
        case["intensity"],
        "--basis",
        "sto-6g",
        "--ncas",
        str(case["ncas"]),
        "--nelecas",
        str(case["nelecas"]),
        "--bond-dim",
        str(case["bond_dim"]),
        "--dmrg-sweeps",
        str(case["dmrg_sweeps"]),
        "--dt",
        str(args.dt),
        "--omega",
        str(args.omega),
        "--cycles",
        str(case["cycles"]),
        "--drive-amplitude",
        AMPLITUDES[case["intensity"]],
        "--save-every",
        str(case["save_every"]),
        "--tdvp-dynamic-mode",
        "midpoint",
        "--output-dir",
        str(output_dir),
    ]
    if args.track_energy:
        cmd.append("--track-energy")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("=" * 72, flush=True)
    print(
        f"[local STO-6G] {case['geometry']} {case['intensity']} "
        f"CAS({case['ncas']},{case['nelecas']}) D={case['bond_dim']}",
        flush=True,
    )
    print(f"[local STO-6G] output: {output_dir}", flush=True)
    print("=" * 72, flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main(argv=None):
    args = parse_args(argv)
    if not RUNNER.exists():
        raise FileNotFoundError(f"GTO runner not found: {RUNNER}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for case in case_settings(args):
        run_case(args, case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
