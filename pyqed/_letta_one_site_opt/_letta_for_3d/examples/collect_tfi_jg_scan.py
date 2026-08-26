#!/usr/bin/env python3
"""Collect TFI scan JSON results into summary and iteration CSV files.

The scan driver stores the per-sweep energies in ``energy_history`` inside
each result JSON.  This collector reads all result files below ``results/``
and writes:

* ``energy_summary.csv``: one final-result row per JSON file;
* ``iteration_energy.csv``: one row per recorded sweep/iteration.

Both files are regenerated atomically, so a failed collection does not leave
an incomplete CSV behind.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().with_name("tfi_jg_scan_output")

SUMMARY_FIELDS = (
    "N",
    "nsites",
    "J_over_g",
    "solver",
    "update_scheme",
    "ordering",
    "physical_index_convention",
    "bond_dimension",
    "energy",
    "energy_per_site",
    "initial_energy",
    "converged",
    "sweeps",
    "requested_sweeps",
    "runtime_seconds",
    "initialization",
    "used_kick",
    "result_file",
    "state_file",
)

ITERATION_FIELDS = (
    "N",
    "nsites",
    "J_over_g",
    "solver",
    "update_scheme",
    "ordering",
    "physical_index_convention",
    "bond_dimension",
    "iteration",
    "direction",
    "energy",
    "energy_per_site",
    "energy_change",
    "energy_density_change",
    "bond_dimension_at_iteration",
    "max_discarded_weight",
    "result_file",
)


def _result_files(results_root: Path) -> list[Path]:
    """Return result JSON files in deterministic path order."""

    if not results_root.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {results_root}")
    return sorted(
        path
        for path in results_root.rglob("N*.json")
        if path.is_file()
    )


def _load_results(result_files: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded = []
    for result_file in result_files:
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read result JSON {result_file}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"result JSON must contain an object: {result_file}")
        if not isinstance(payload.get("energy_history"), list):
            raise ValueError(
                f"result JSON has no list-valued energy_history: {result_file}"
            )
        loaded.append((result_file, payload))
    return loaded


def _summary_row(result_file: Path, payload: dict[str, Any]) -> dict[str, Any]:
    nsites = int(payload["nsites"])
    energy = float(payload["energy"])
    return {
        "N": payload["N"],
        "nsites": nsites,
        "J_over_g": payload["J_over_g"],
        "solver": payload["solver"],
        "update_scheme": payload["update_scheme"],
        "ordering": payload["ordering"],
        "physical_index_convention": payload.get("physical_index_convention", ""),
        "bond_dimension": payload["bond_dimension"],
        "energy": energy,
        "energy_per_site": energy / nsites,
        "initial_energy": payload["initial_energy"],
        "converged": payload["converged"],
        "sweeps": payload["sweeps"],
        "requested_sweeps": payload["requested_sweeps"],
        "runtime_seconds": payload["runtime_seconds"],
        "initialization": payload.get("initialization", "random"),
        "used_kick": payload.get("used_kick", ""),
        "result_file": str(result_file.resolve()),
        "state_file": payload.get("state_path", ""),
    }


def _iteration_rows(result_file: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    nsites = int(payload["nsites"])
    rows = []
    for position, history in enumerate(payload["energy_history"], start=1):
        if not isinstance(history, dict):
            raise ValueError(
                f"energy_history entry {position} is not an object: {result_file}"
            )
        if "energy" not in history:
            raise ValueError(
                f"energy_history entry {position} has no energy: {result_file}"
            )
        energy = float(history["energy"])
        rows.append(
            {
                "N": payload["N"],
                "nsites": nsites,
                "J_over_g": payload["J_over_g"],
                "solver": payload["solver"],
                "update_scheme": payload["update_scheme"],
                "ordering": payload["ordering"],
                "physical_index_convention": payload.get(
                    "physical_index_convention", ""
                ),
                "bond_dimension": payload["bond_dimension"],
                "iteration": history.get("iteration", position),
                "direction": history.get("direction", ""),
                "energy": energy,
                "energy_per_site": energy / nsites,
                "energy_change": history.get("energy_change", ""),
                "energy_density_change": history.get(
                    "energy_density_change", ""
                ),
                "bond_dimension_at_iteration": history.get(
                    "bond_dimension", payload["bond_dimension"]
                ),
                "max_discarded_weight": history.get("max_discarded_weight", ""),
                "result_file": str(result_file.resolve()),
            }
        )
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect(
    *,
    results_root: Path,
    summary_path: Path,
    iteration_path: Path,
) -> tuple[int, int]:
    """Collect result JSON files and return ``(result_count, iteration_count)``."""

    loaded = _load_results(_result_files(results_root))
    summary_rows = [_summary_row(path, payload) for path, payload in loaded]
    iteration_rows = [
        row
        for path, payload in loaded
        for row in _iteration_rows(path, payload)
    ]
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(iteration_path, ITERATION_FIELDS, iteration_rows)
    return len(summary_rows), len(iteration_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TFI scan result JSON files into CSV files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"scan output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="result JSON directory (default: OUTPUT_ROOT/results)",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="summary CSV path (default: OUTPUT_ROOT/energy_summary.csv)",
    )
    parser.add_argument(
        "--iteration-path",
        type=Path,
        default=None,
        help="per-iteration CSV path (default: OUTPUT_ROOT/iteration_energy.csv)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    output_root = arguments.output_root
    results_root = arguments.results_root or output_root / "results"
    summary_path = arguments.summary_path or output_root / "energy_summary.csv"
    iteration_path = arguments.iteration_path or output_root / "iteration_energy.csv"
    result_count, iteration_count = collect(
        results_root=results_root,
        summary_path=summary_path,
        iteration_path=iteration_path,
    )
    print(f"Collected {result_count} result files and {iteration_count} iterations.")
    print(f"Summary: {summary_path}")
    print(f"Iterations: {iteration_path}")


if __name__ == "__main__":
    main()
