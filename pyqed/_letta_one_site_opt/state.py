"""Finite site-factorized LETTA states on rectangular lattices."""

from __future__ import annotations

from operator import index

import numpy as np

from .symmetry import AbelianSymmetry


def _validate_lattice_shape(lattice_shape):
    try:
        shape = tuple(index(length) for length in lattice_shape)
    except (TypeError, ValueError) as error:
        raise ValueError("lattice_shape must contain integers.") from error
    if len(shape) not in {2, 3}:
        raise ValueError("only two- and three-dimensional lattices are supported.")
    if any(length <= 0 for length in shape):
        raise ValueError("all lattice dimensions must be positive.")
    return shape


def _validate_coordinates(lattice_shape, coordinates):
    canonical = tuple(np.ndindex(*lattice_shape))
    if coordinates is None:
        return canonical
    try:
        coordinates = tuple(
            tuple(index(value) for value in coordinate)
            for coordinate in coordinates
        )
    except (TypeError, ValueError) as error:
        raise ValueError("coordinates must contain integer lattice points.") from error
    if len(coordinates) != len(canonical) or set(coordinates) != set(canonical):
        raise ValueError(
            "coordinates must be a permutation of all lattice points."
        )
    return coordinates


class LatticeLETTA:
    r"""Open-boundary LETTA with current and positive-neighbor physical legs.

    Sites use NumPy's C-order enumeration unless an explicit coordinate
    permutation is supplied. For a two-dimensional shape ``(Lx, Ly)``, the
    default is the column-major numbering in the user's sketch:
    ``(0,0), (0,1), ..., (1,0), ...``. Physical legs are ordered as the
    current site followed by existing positive-axis neighbors, with the last
    coordinate axis first. In 2D this means ``(current, down, right)``.
    """

    def __init__(
        self,
        lattice_shape,
        physical_dim,
        tensors,
        *,
        coordinates=None,
        symmetry=None,
        bond_charges=None,
    ):
        self.lattice_shape = _validate_lattice_shape(lattice_shape)
        try:
            self.physical_dim = index(physical_dim)
        except TypeError as error:
            raise ValueError("physical_dim must be an integer.") from error
        if self.physical_dim <= 0:
            raise ValueError("physical_dim must be positive.")
        self.coordinates = _validate_coordinates(
            self.lattice_shape,
            coordinates,
        )
        self._coordinate_to_site = {
            coordinate: site for site, coordinate in enumerate(self.coordinates)
        }
        self._neighborhoods = tuple(
            self._build_neighborhood(coordinate) for coordinate in self.coordinates
        )
        self.tensors = self._validate_tensors(tensors)
        if symmetry is not None and not isinstance(symmetry, AbelianSymmetry):
            raise TypeError("symmetry must be an AbelianSymmetry instance.")
        if symmetry is not None and len(symmetry.physical_charges) != self.physical_dim:
            raise ValueError(
                "symmetry physical charges must match the physical dimension."
            )
        if symmetry is None and bond_charges is not None:
            raise ValueError("bond_charges requires a symmetry.")
        self.symmetry = symmetry
        if symmetry is None:
            self.bond_charges = None
        else:
            if bond_charges is None:
                if len(set(self.bond_dimensions)) > 1:
                    raise ValueError(
                        "explicit bond_charges are required for nonuniform bonds."
                    )
                bond_dim = self.bond_dimensions[0] if self.bond_dimensions else 1
                bond_charges = symmetry.allocate_bond_charges(
                    self.nsites, bond_dim
                )
            self.bond_charges = symmetry.validate_bond_charges(
                bond_charges, self.bond_dimensions
            )
            self.enforce_symmetry()
        self.normalize()

    @classmethod
    def random(
        cls,
        lattice_shape,
        *,
        physical_dim=2,
        bond_dim=2,
        seed=None,
        real=True,
        coordinates=None,
        symmetry=None,
        bond_charges=None,
    ):
        lattice_shape = _validate_lattice_shape(lattice_shape)
        try:
            physical_dim = index(physical_dim)
            bond_dim = index(bond_dim)
        except TypeError as error:
            raise ValueError("physical_dim and bond_dim must be integers.") from error
        if physical_dim <= 0 or bond_dim <= 0:
            raise ValueError("physical_dim and bond_dim must be positive.")
        coordinates = _validate_coordinates(lattice_shape, coordinates)
        coordinate_to_site = {
            coordinate: site for site, coordinate in enumerate(coordinates)
        }
        rng = np.random.default_rng(seed)
        tensors = []
        nsites = len(coordinates)
        for site, coordinate in enumerate(coordinates):
            nneighbors = 0
            for axis in reversed(range(len(lattice_shape))):
                neighbor = list(coordinate)
                neighbor[axis] += 1
                if tuple(neighbor) in coordinate_to_site:
                    nneighbors += 1
            left_dim = 1 if site == 0 else bond_dim
            right_dim = 1 if site == nsites - 1 else bond_dim
            shape = (left_dim,) + (physical_dim,) * (1 + nneighbors) + (
                right_dim,
            )
            tensor = rng.normal(size=shape)
            if not real:
                tensor = tensor + 1j * rng.normal(size=shape)
            tensor /= np.sqrt(tensor.size)
            tensors.append(tensor)
        return cls(
            lattice_shape,
            physical_dim,
            tensors,
            coordinates=coordinates,
            symmetry=symmetry,
            bond_charges=bond_charges,
        )

    @property
    def ndim(self):
        return len(self.lattice_shape)

    @property
    def nsites(self):
        return len(self.coordinates)

    @property
    def hilbert_dim(self):
        return self.physical_dim**self.nsites

    @property
    def bond_dimensions(self):
        return tuple(tensor.shape[-1] for tensor in self.tensors[:-1])

    @property
    def dense_parameter_count(self):
        return sum(tensor.size for tensor in self.tensors)

    @property
    def parameter_count(self):
        if self.symmetry is None:
            return self.dense_parameter_count
        return sum(
            int(np.count_nonzero(self.symmetry_mask(site)))
            for site in range(self.nsites)
        )

    def _build_neighborhood(self, coordinate):
        sites = [self._coordinate_to_site[coordinate]]
        for axis in reversed(range(self.ndim)):
            neighbor = list(coordinate)
            neighbor[axis] += 1
            neighbor = tuple(neighbor)
            if neighbor in self._coordinate_to_site:
                sites.append(self._coordinate_to_site[neighbor])
        return tuple(sites)

    def site_neighborhood(self, site):
        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range.")
        return self._neighborhoods[site]

    def _validate_tensors(self, tensors):
        tensors = [np.asarray(tensor).copy() for tensor in tensors]
        if len(tensors) != self.nsites:
            raise ValueError("there must be one LETTA tensor per lattice site.")
        for site, tensor in enumerate(tensors):
            physical_rank = len(self._neighborhoods[site])
            if tensor.ndim != physical_rank + 2:
                raise ValueError(
                    f"tensor {site} must have {physical_rank} physical axes."
                )
            if tensor.shape[1:-1] != (self.physical_dim,) * physical_rank:
                raise ValueError(f"tensor {site} has incompatible physical axes.")
            if site == 0 and tensor.shape[0] != 1:
                raise ValueError("the first tensor must have left bond dimension one.")
            if site == self.nsites - 1 and tensor.shape[-1] != 1:
                raise ValueError("the last tensor must have right bond dimension one.")
            if site and tensors[site - 1].shape[-1] != tensor.shape[0]:
                raise ValueError(f"virtual bond mismatch before tensor {site}.")
            if not np.all(np.isfinite(tensor)):
                raise ValueError("LETTA tensors must contain finite values.")
        return tensors

    def copy(self):
        return LatticeLETTA(
            self.lattice_shape,
            self.physical_dim,
            [tensor.copy() for tensor in self.tensors],
            coordinates=self.coordinates,
            symmetry=self.symmetry,
            bond_charges=self.bond_charges,
        )

    def without_symmetry(self):
        """Return the identical dense LETTA wavefunction without charge metadata."""

        return LatticeLETTA(
            self.lattice_shape,
            self.physical_dim,
            [tensor.copy() for tensor in self.tensors],
            coordinates=self.coordinates,
        )

    def left_virtual_charges(self, site):
        if self.symmetry is None:
            return None
        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range.")
        if site == 0:
            return (self.symmetry.identity,)
        return self.bond_charges[site - 1]

    def right_virtual_charges(self, site):
        if self.symmetry is None:
            return None
        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range.")
        if site == self.nsites - 1:
            return (self.symmetry.sector,)
        return self.bond_charges[site]

    def symmetry_mask(self, site):
        """Return the dense mask of charge-conserving entries at one site."""

        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range.")
        tensor = self.tensors[site]
        if self.symmetry is None:
            return np.ones(tensor.shape, dtype=bool)
        left_charges = self.left_virtual_charges(site)
        right_charges = self.right_virtual_charges(site)
        local = np.zeros(
            (len(left_charges), self.physical_dim, len(right_charges)),
            dtype=bool,
        )
        for left, left_charge in enumerate(left_charges):
            for physical, physical_charge in enumerate(
                self.symmetry.physical_charges
            ):
                outgoing = self.symmetry.fuse(left_charge, physical_charge)
                for right, right_charge in enumerate(right_charges):
                    local[left, physical, right] = outgoing == right_charge
        local = local.reshape(
            (tensor.shape[0], self.physical_dim)
            + (1,) * (tensor.ndim - 3)
            + (tensor.shape[-1],)
        )
        return np.broadcast_to(local, tensor.shape)

    def symmetry_indices(self, site):
        return np.flatnonzero(self.symmetry_mask(site).reshape(-1))

    def enforce_symmetry(self):
        if self.symmetry is not None:
            for site, tensor in enumerate(self.tensors):
                tensor[~self.symmetry_mask(site)] = 0.0
        return self

    def symmetry_violation(self):
        if self.symmetry is None:
            return 0.0
        forbidden = 0.0
        total = 0.0
        for site, tensor in enumerate(self.tensors):
            forbidden += float(np.linalg.norm(tensor[~self.symmetry_mask(site)]) ** 2)
            total += float(np.linalg.norm(tensor) ** 2)
        if total <= np.finfo(float).tiny:
            return 0.0
        return float(np.sqrt(forbidden / total))

    def expand_bond_dimension(self, bond_dim, *, noise=0.0, seed=None):
        """Embed the state in a larger virtual space.

        Zero padding is an exact embedding: the enlarged tensors represent the
        same wavefunction.  Optional noise populates only newly introduced
        tensor entries and lets subsequent one-site sweeps discover directions
        outside the smaller-bond variational manifold.
        """

        try:
            bond_dim = index(bond_dim)
        except TypeError as error:
            raise ValueError("bond_dim must be an integer.") from error
        noise = float(noise)
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive.")
        if noise < 0.0:
            raise ValueError("noise must be nonnegative.")
        if any(dimension > bond_dim for dimension in self.bond_dimensions):
            raise ValueError("bond_dim cannot shrink an existing virtual bond.")

        rng = np.random.default_rng(seed)
        dtype = np.result_type(*self.tensors)
        tensors = []
        expanded_bond_charges = None
        if self.symmetry is not None:
            allocated = self.symmetry.allocate_bond_charges(
                self.nsites, bond_dim
            )
            expanded_bond_charges = []
            for old, new in zip(self.bond_charges, allocated):
                charges = list(new)
                charges[: len(old)] = old
                expanded_bond_charges.append(tuple(charges))
            expanded_bond_charges = tuple(expanded_bond_charges)
        for site, tensor in enumerate(self.tensors):
            left_dim = 1 if site == 0 else bond_dim
            right_dim = 1 if site == self.nsites - 1 else bond_dim
            shape = (left_dim,) + tensor.shape[1:-1] + (right_dim,)
            expanded = np.zeros(shape, dtype=dtype)
            old_block = (
                (slice(0, tensor.shape[0]),)
                + (slice(None),) * (tensor.ndim - 2)
                + (slice(0, tensor.shape[-1]),)
            )
            expanded[old_block] = tensor
            if noise:
                mask = np.ones(shape, dtype=bool)
                mask[old_block] = False
                scale = noise * np.linalg.norm(tensor) / np.sqrt(tensor.size)
                perturbation = rng.normal(size=shape)
                if np.issubdtype(dtype, np.complexfloating):
                    perturbation = (
                        perturbation + 1j * rng.normal(size=shape)
                    ) / np.sqrt(2.0)
                expanded[mask] = scale * perturbation[mask]
            tensors.append(expanded)
        return LatticeLETTA(
            self.lattice_shape,
            self.physical_dim,
            tensors,
            coordinates=self.coordinates,
            symmetry=self.symmetry,
            bond_charges=expanded_bond_charges,
        )

    def _physical_indices(self, site, configuration):
        return tuple(configuration[index_] for index_ in self._neighborhoods[site])

    def _site_matrix(self, site, configuration):
        physical = self._physical_indices(site, configuration)
        return self.tensors[site][(slice(None),) + physical + (slice(None),)]

    def amplitude(self, configuration):
        configuration = tuple(index(value) for value in configuration)
        if len(configuration) != self.nsites:
            raise ValueError("configuration length must equal the number of sites.")
        if any(value < 0 or value >= self.physical_dim for value in configuration):
            raise ValueError("configuration contains an invalid physical index.")
        value = np.array([1.0], dtype=np.result_type(*self.tensors))
        for site in range(self.nsites):
            value = value @ self._site_matrix(site, configuration)
        return value[0]

    def state_vector(self):
        vector = np.empty(self.hilbert_dim, dtype=np.result_type(*self.tensors))
        dimensions = (self.physical_dim,) * self.nsites
        for flat, configuration in enumerate(np.ndindex(*dimensions)):
            vector[flat] = self.amplitude(configuration)
        return vector

    def norm(self):
        from .contractions import network_overlap

        return network_overlap(self)

    def normalize(self):
        norm_squared = self.norm()
        if norm_squared <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a numerically zero LETTA state.")
        self._balance_tensor_scales(norm_squared**-0.5)
        return self

    def balance_tensor_scales(self):
        """Equalize tensor norms without changing the represented state."""

        self._balance_tensor_scales(1.0)
        return self

    def _balance_tensor_scales(self, amplitude_scale):
        tensor_norms = np.asarray(
            [np.linalg.norm(tensor) for tensor in self.tensors]
        )
        if np.any(tensor_norms <= np.finfo(float).tiny):
            raise ValueError("cannot balance a LETTA state with a zero tensor.")
        target_log_norm = np.mean(np.log(tensor_norms)) + (
            np.log(amplitude_scale) / self.nsites
        )
        for site, tensor_norm in enumerate(tensor_norms):
            scale = np.exp(target_log_norm - np.log(tensor_norm))
            self.tensors[site] = self.tensors[site] * scale

    def expectation(self, operator):
        from .operators import LatticeMPO

        if isinstance(operator, LatticeMPO):
            from .contractions import network_expectation

            return network_expectation(self, operator)
        vector = self.state_vector()
        applied = operator @ vector
        return float(np.real(np.vdot(vector, applied) / np.vdot(vector, vector)))

    def _left_partial(self, site, configuration):
        value = np.array([1.0], dtype=np.result_type(*self.tensors))
        for current in range(site):
            value = value @ self._site_matrix(current, configuration)
        return value

    def _right_partial(self, site, configuration):
        value = np.array([1.0], dtype=np.result_type(*self.tensors))
        for current in range(self.nsites - 1, site, -1):
            value = self._site_matrix(current, configuration) @ value
        return value

    def local_frame(self, site):
        """Return the linear map from one local tensor to the full state."""

        site = index(site)
        if site < 0 or site >= self.nsites:
            raise IndexError("site index out of range.")
        tensor = self.tensors[site]
        frame = np.zeros(
            (self.hilbert_dim, tensor.size),
            dtype=np.result_type(*self.tensors),
        )
        dimensions = (self.physical_dim,) * self.nsites
        for flat, configuration in enumerate(np.ndindex(*dimensions)):
            left = self._left_partial(site, configuration)
            right = self._right_partial(site, configuration)
            physical = self._physical_indices(site, configuration)
            for left_index in range(tensor.shape[0]):
                for right_index in range(tensor.shape[-1]):
                    local_index = (left_index,) + physical + (right_index,)
                    column = np.ravel_multi_index(local_index, tensor.shape)
                    frame[flat, column] = (
                        left[left_index] * right[right_index]
                    )
        return frame

    def column_tensor(self, column):
        """Contract one 2D column into an enlarged-physical pair tensor."""

        if self.ndim != 2:
            raise ValueError("column blocking is defined only for 2D lattices.")
        column = index(column)
        lx, ly = self.lattice_shape
        if column < 0 or column >= lx:
            raise IndexError("column index out of range.")
        q = self.physical_dim**ly
        first_site = column * ly
        last_site = first_site + ly - 1
        left_dim = self.tensors[first_site].shape[0]
        right_dim = self.tensors[last_site].shape[-1]
        next_dim = q if column + 1 < lx else 1
        blocked = np.empty(
            (left_dim, q, next_dim, right_dim),
            dtype=np.result_type(*self.tensors),
        )
        column_configs = tuple(np.ndindex(*(self.physical_dim,) * ly))
        next_configs = column_configs if next_dim == q else (None,)
        for current_flat, current in enumerate(column_configs):
            for next_flat, next_column in enumerate(next_configs):
                product = np.eye(left_dim, dtype=blocked.dtype)
                for offset in range(ly):
                    site = first_site + offset
                    physical = [current[offset]]
                    if offset + 1 < ly:
                        physical.append(current[offset + 1])
                    if next_column is not None:
                        physical.append(next_column[offset])
                    matrix = self.tensors[site][
                        (slice(None),) + tuple(physical) + (slice(None),)
                    ]
                    product = product @ matrix
                blocked[:, current_flat, next_flat, :] = product
        return blocked

    def blocked_column_amplitude(self, configuration):
        if self.ndim != 2:
            raise ValueError("column blocking is defined only for 2D lattices.")
        configuration = tuple(configuration)
        lx, ly = self.lattice_shape
        columns = [
            configuration[column * ly : (column + 1) * ly]
            for column in range(lx)
        ]
        qdims = (self.physical_dim,) * ly
        column_indices = [np.ravel_multi_index(column, qdims) for column in columns]
        value = np.array([1.0], dtype=np.result_type(*self.tensors))
        for column in range(lx):
            next_index = column_indices[column + 1] if column + 1 < lx else 0
            value = value @ self.column_tensor(column)[
                :, column_indices[column], next_index, :
            ]
        return value[0]
