"""Three-dimensional open-lattice geometry and site orderings."""

from __future__ import annotations

from operator import index


def validate_shape(lattice_shape):
    try:
        shape = tuple(index(length) for length in lattice_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("lattice_shape must contain integers.") from error
    if len(shape) != 3:
        raise ValueError("a 3D lattice shape must contain exactly three lengths.")
    if any(length <= 0 for length in shape):
        raise ValueError("all lattice dimensions must be positive.")
    return shape


def ordered_coordinates(lattice_shape, *, ordering="continuous-snake"):
    """Return lattice coordinates in a selected one-dimensional ordering.

    ``continuous-snake`` reverses alternating complete planes and makes every
    chain step spatially local. ``layer-snake`` repeats the same 2D snake in
    each plane, reducing long inter-plane bond spans. ``compact`` is ordinary
    lexicographic order and minimizes the maximum cubic bond span for the
    rectangular grids used here.
    """

    lx, ly, lz = validate_shape(lattice_shape)
    ordering = str(ordering).lower().replace("_", "-")
    aliases = {
        "snake": "continuous-snake",
        "continuous": "continuous-snake",
        "layer": "layer-snake",
        "layered": "layer-snake",
        "lexicographic": "compact",
        "c-order": "compact",
    }
    ordering = aliases.get(ordering, ordering)
    if ordering == "compact":
        return tuple(
            (x, y, z)
            for x in range(lx)
            for y in range(ly)
            for z in range(lz)
        )
    if ordering not in {"continuous-snake", "layer-snake"}:
        raise ValueError(
            "ordering must be 'continuous-snake', 'layer-snake', or 'compact'."
        )

    plane = []
    for y in range(ly):
        z_values = range(lz) if y % 2 == 0 else range(lz - 1, -1, -1)
        plane.extend((y, z) for z in z_values)

    coordinates = []
    for x in range(lx):
        reverse_plane = ordering == "continuous-snake" and x % 2
        path = reversed(plane) if reverse_plane else plane
        coordinates.extend((x, y, z) for y, z in path)
    return tuple(coordinates)


def snake_coordinates(lattice_shape):
    """Return the continuous nearest-neighbor snake used by the first version."""

    return ordered_coordinates(lattice_shape, ordering="continuous-snake")


def coordinate_to_site(lattice_shape, *, ordering="continuous-snake"):
    return {
        coordinate: site
        for site, coordinate in enumerate(
            ordered_coordinates(lattice_shape, ordering=ordering)
        )
    }


def nearest_neighbor_bonds(lattice_shape, *, ordering="continuous-snake"):
    """Return every open-boundary cubic nearest-neighbor bond once."""

    shape = validate_shape(lattice_shape)
    coordinates = ordered_coordinates(shape, ordering=ordering)
    site_for = {coordinate: site for site, coordinate in enumerate(coordinates)}
    bonds = []
    for coordinate in coordinates:
        left = site_for[coordinate]
        for axis in range(3):
            neighbor = list(coordinate)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            if neighbor not in site_for:
                continue
            right = site_for[neighbor]
            bonds.append(tuple(sorted((left, right))))
    return tuple(sorted(set(bonds)))
