"""Count stored scalar entries for the 3D TFI MPS and LETTA ansatzes.

This utility deliberately counts every scalar stored in every tensor.  It does
not subtract gauge freedoms, normalization constraints, or other redundancies.
Bond dimensions are capped only when the tensor dimensions on one side of a
cut cannot support the requested maximum bond dimension.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import prod
from pathlib import Path
from typing import Iterable, Sequence

NX_VALUES = (3, 6, 9)
MPS_BOND_DIMENSIONS = (1, 4, 8, 16, 32)
LETTA_BOND_DIMENSIONS = (1, 4, 6)
PHYSICAL_LOCAL_DIMENSION = 2


def _rank_capped_bonds(
    local_dimensions: Sequence[int], max_bond_dimension: int
) -> tuple[int, ...]:
    """Return the stored rank of every internal bond."""
    if max_bond_dimension < 1:
        raise ValueError("max_bond_dimension must be positive")
    if any(dimension < 1 for dimension in local_dimensions):
        raise ValueError("local dimensions must be positive")

    left_capacities: list[int] = []
    capacity = 1
    for dimension in local_dimensions[:-1]:
        capacity = min(max_bond_dimension, capacity * dimension)
        left_capacities.append(capacity)

    right_capacities: list[int] = []
    capacity = 1
    for dimension in reversed(local_dimensions[1:]):
        capacity = min(max_bond_dimension, capacity * dimension)
        right_capacities.append(capacity)
    right_capacities.reverse()

    return tuple(
        min(max_bond_dimension, left_capacity, right_capacity)
        for left_capacity, right_capacity in zip(
            left_capacities, right_capacities, strict=True
        )
    )


def _tensor_shapes(
    local_dimensions: Sequence[int], max_bond_dimension: int
) -> tuple[tuple[int, int, int], ...]:
    bonds = _rank_capped_bonds(local_dimensions, max_bond_dimension)
    left_bonds = (1,) + bonds
    right_bonds = bonds + (1,)
    return tuple(
        (left_bond, local_dimension, right_bond)
        for left_bond, local_dimension, right_bond in zip(
            left_bonds, local_dimensions, right_bonds, strict=True
        )
    )


def _compact_letta_local_dimensions(shape: tuple[int, int, int]) -> tuple[int, ...]:
    """Physical-block dimensions in compact LETTA tensor order."""
    lx, ly, lz = shape
    coordinates = tuple(
        (x, y, z)
        for x in range(lx)
        for y in range(ly)
        for z in range(lz)
    )
    coordinate_set = set(coordinates)
    positive_steps = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    local_dimensions: list[int] = []
    for x, y, z in coordinates:
        positive_neighbor_count = sum(
            (x + dx, y + dy, z + dz) in coordinate_set
            for dx, dy, dz in positive_steps
        )
        physical_axes = 1 + positive_neighbor_count
        local_dimensions.append(PHYSICAL_LOCAL_DIMENSION**physical_axes)
    return tuple(local_dimensions)


def _shape_histogram(shapes: Iterable[tuple[int, int, int]]) -> str:
    counts = Counter(shapes)
    return ";".join(
        f"{left}x{local}x{right}:{count}"
        for (left, local, right), count in sorted(counts.items())
    )


def _record(
    nx: int,
    ansatz: str,
    ordering: str,
    max_bond_dimension: int,
    local_dimensions: Sequence[int],
) -> dict[str, object]:
    shape = (nx, 3, 3)
    tensor_shapes = _tensor_shapes(local_dimensions, max_bond_dimension)
    internal_bonds = _rank_capped_bonds(local_dimensions, max_bond_dimension)
    return {
        "Nx": nx,
        "lattice_shape": "x".join(map(str, shape)),
        "nsites": prod(shape),
        "ansatz": ansatz,
        "ordering": ordering,
        "max_bond_dimension_D": max_bond_dimension,
        "physical_local_dimension": PHYSICAL_LOCAL_DIMENSION,
        "stored_tensor_entries": sum(prod(tensor_shape) for tensor_shape in tensor_shapes),
        "internal_bond_dimensions": list(internal_bonds),
        "stored_tensor_shape_histogram": _shape_histogram(tensor_shapes),
    }


def parameter_count_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for nx in NX_VALUES:
        nsites = nx * 3 * 3
        letta_local_dimensions = _compact_letta_local_dimensions((nx, 3, 3))
        for bond_dimension in LETTA_BOND_DIMENSIONS:
            records.append(
                _record(
                    nx,
                    "LETTA",
                    "compact",
                    bond_dimension,
                    letta_local_dimensions,
                )
            )
        for bond_dimension in MPS_BOND_DIMENSIONS:
            records.append(
                _record(
                    nx,
                    "MPS",
                    "raster",
                    bond_dimension,
                    (PHYSICAL_LOCAL_DIMENSION,) * nsites,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional audit JSON path. If omitted, write JSON to standard output.",
    )
    args = parser.parse_args()

    output = json.dumps(parameter_count_records(), indent=2) + "\n"
    if args.output_json is None:
        print(output, end="")
    else:
        args.output_json.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
