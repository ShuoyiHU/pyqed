"""Direct tensor-network contractions for lattice LETTA states."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import opt_einsum as oe

from .operators import LatticeMPO


class BoundaryMPS:
    """Tensor-train representation of an ordered contraction frontier."""

    def __init__(self, cores, labels, discarded_weight=0.0):
        self.cores = tuple(np.asarray(core) for core in cores)
        self.labels = tuple(labels)
        self.discarded_weight = float(discarded_weight)

    @classmethod
    def from_dense(cls, tensor, labels, *, max_bond_dim, cutoff=0.0):
        tensor = np.asarray(tensor)
        labels = tuple(labels)
        if tensor.ndim != len(labels) or tensor.ndim == 0:
            raise ValueError("boundary tensor rank must match its labels.")
        max_bond_dim = int(max_bond_dim)
        if max_bond_dim <= 0 or cutoff < 0.0:
            raise ValueError("invalid boundary-MPS truncation settings.")
        dimensions = tensor.shape
        total_norm_squared = float(np.linalg.norm(tensor) ** 2)
        remainder = tensor
        left_rank = 1
        cores = []
        discarded = 0.0
        for dimension in dimensions[:-1]:
            matrix = remainder.reshape(left_rank * dimension, -1)
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            threshold = cutoff * singular_values[0] if singular_values.size else 0.0
            keep = min(
                max_bond_dim,
                max(1, int(np.count_nonzero(singular_values > threshold))),
            )
            discarded += float(np.sum(singular_values[keep:] ** 2))
            cores.append(u[:, :keep].reshape(left_rank, dimension, keep))
            remainder = singular_values[:keep, None] * vh[:keep]
            left_rank = keep
        cores.append(remainder.reshape(left_rank, dimensions[-1], 1))
        relative_discarded = (
            discarded / total_norm_squared if total_norm_squared > 0.0 else 0.0
        )
        return cls(cores, labels, relative_discarded)

    @property
    def shape(self):
        return tuple(core.shape[1] for core in self.cores)

    @property
    def nbytes(self):
        return sum(core.nbytes for core in self.cores)

    @property
    def dtype(self):
        return np.result_type(*self.cores)

    def to_dense(self):
        tensor = self.cores[0][0]
        for core in self.cores[1:]:
            tensor = np.tensordot(tensor, core, axes=([-1], [0]))
        return tensor[..., 0]

    def compress(self, *, max_bond_dim, cutoff=0.0):
        """Compress this boundary without reconstructing its dense tensor."""

        max_bond_dim = int(max_bond_dim)
        if max_bond_dim <= 0 or cutoff < 0.0:
            raise ValueError("invalid boundary-MPS truncation settings.")
        cores = [core.copy() for core in self.cores]
        for site in range(len(cores) - 1, 0, -1):
            left_rank, physical_dim, right_rank = cores[site].shape
            q, r = np.linalg.qr(
                cores[site].reshape(left_rank, physical_dim * right_rank).T,
                mode="reduced",
            )
            cores[site] = q.T.reshape(q.shape[1], physical_dim, right_rank)
            cores[site - 1] = np.tensordot(
                cores[site - 1],
                r.T,
                axes=([-1], [0]),
            )
        total_norm_squared = float(np.linalg.norm(cores[0]) ** 2)
        discarded = 0.0
        for site in range(len(cores) - 1):
            left_rank, physical_dim, right_rank = cores[site].shape
            matrix = cores[site].reshape(left_rank * physical_dim, right_rank)
            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            threshold = cutoff * singular_values[0] if singular_values.size else 0.0
            keep = min(
                max_bond_dim,
                max(1, int(np.count_nonzero(singular_values > threshold))),
            )
            discarded += float(np.sum(singular_values[keep:] ** 2))
            cores[site] = u[:, :keep].reshape(left_rank, physical_dim, keep)
            transfer = singular_values[:keep, None] * vh[:keep]
            cores[site + 1] = np.tensordot(
                transfer,
                cores[site + 1],
                axes=([1], [0]),
            )
        relative_discarded = (
            discarded / total_norm_squared if total_norm_squared > 0.0 else 0.0
        )
        return type(self)(
            cores,
            self.labels,
            self.discarded_weight + relative_discarded,
        )

    def apply_local_transfer(
        self,
        *,
        input_labels,
        output_labels,
        operands,
        operand_labels,
        label_dimensions,
        max_bond_dim,
        cutoff=0.0,
    ):
        """Apply one local contraction group as an MPO on this frontier."""

        input_labels = tuple(input_labels)
        output_labels = tuple(output_labels)
        if input_labels != self.labels:
            raise ValueError("boundary-MPS labels do not match the transfer input.")
        union_labels = tuple(sorted(set(input_labels) | set(output_labels)))
        state_cores = _align_boundary_cores(
            self.cores,
            input_labels,
            union_labels,
        )
        operator_cores = _local_transfer_cores(
            operands,
            operand_labels,
            input_labels,
            output_labels,
            union_labels,
            label_dimensions,
        )
        result_cores = []
        for state_core, operator_core in zip(state_cores, operator_cores):
            left_state, input_dim, right_state = state_core.shape
            left_operator, output_dim, operator_input, right_operator = (
                operator_core.shape
            )
            if input_dim != operator_input:
                raise ValueError("boundary and transfer physical dimensions differ.")
            applied = np.einsum(
                "aib,cjid->acjbd",
                state_core,
                operator_core,
                optimize=True,
            )
            result_cores.append(
                applied.reshape(
                    left_state * left_operator,
                    output_dim,
                    right_state * right_operator,
                )
            )
        retained_cores, retained_labels, scalar = _remove_unit_output_cores(
            result_cores,
            union_labels,
            output_labels,
        )
        if not output_labels:
            return np.asarray(scalar)
        result = type(self)(retained_cores, retained_labels)
        return result.compress(max_bond_dim=max_bond_dim, cutoff=cutoff)


def _align_boundary_cores(cores, labels, union_labels):
    aligned = []
    position = 0
    for label in union_labels:
        if position < len(labels) and labels[position] == label:
            aligned.append(cores[position])
            position += 1
            continue
        rank = aligned[-1].shape[-1] if aligned else 1
        aligned.append(np.eye(rank).reshape(rank, 1, rank))
    if position != len(labels):
        raise ValueError("boundary labels are not an ordered subset of the union.")
    return aligned


def _tensor_train_cores(tensor, dimensions):
    dimensions = tuple(int(dimension) for dimension in dimensions)
    if not dimensions:
        return [], np.asarray(tensor).reshape(()).item()
    remainder = np.asarray(tensor).reshape(dimensions)
    cores = []
    left_rank = 1
    for dimension in dimensions[:-1]:
        matrix = remainder.reshape(left_rank * dimension, -1)
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = singular_values.size
        cores.append(u.reshape(left_rank, dimension, rank))
        remainder = singular_values[:, None] * vh
        left_rank = rank
    cores.append(remainder.reshape(left_rank, dimensions[-1], 1))
    return cores, 1.0


def _local_transfer_cores(
    operands,
    operand_labels,
    input_labels,
    output_labels,
    union_labels,
    label_dimensions,
):
    factor_labels = {label for labels in operand_labels for label in labels}
    touched = tuple(label for label in union_labels if label in factor_labels)
    next_label = max(
        [label for labels in operand_labels for label in labels] + [-1]
    ) + 1
    input_axes = {}
    output_axes = {}
    for label in touched:
        if label in output_labels:
            output_axes[label] = next_label
            next_label += 1
        if label in input_labels:
            input_axes[label] = next_label
            next_label += 1

    transformed_labels = []
    for labels in operand_labels:
        transformed = []
        for label in labels:
            if label in output_axes:
                transformed.append(output_axes[label])
            elif label in input_axes:
                transformed.append(input_axes[label])
            else:
                transformed.append(label)
        transformed_labels.append(tuple(transformed))

    transfer_operands = list(operands)
    transfer_labels = list(transformed_labels)
    for label in touched:
        if label in input_axes and label in output_axes:
            dimension = label_dimensions[label]
            transfer_operands.append(np.eye(dimension))
            transfer_labels.append((output_axes[label], input_axes[label]))

    transfer_output = []
    combined_dimensions = []
    for label in touched:
        output_dimension = 1
        input_dimension = 1
        if label in output_axes:
            transfer_output.append(output_axes[label])
            output_dimension = label_dimensions[label]
        if label in input_axes:
            transfer_output.append(input_axes[label])
            input_dimension = label_dimensions[label]
        combined_dimensions.append(output_dimension * input_dimension)
    local_tensor = _contract_operands(
        transfer_operands,
        transfer_labels,
        tuple(transfer_output),
    )
    local_cores, scalar = _tensor_train_cores(
        local_tensor,
        combined_dimensions,
    )

    result = []
    local_position = 0
    current_rank = 1
    for label in union_labels:
        output_dimension = (
            label_dimensions[label] if label in output_labels else 1
        )
        input_dimension = (
            label_dimensions[label] if label in input_labels else 1
        )
        if label in touched:
            core = local_cores[local_position]
            local_position += 1
            result.append(
                core.reshape(
                    core.shape[0],
                    output_dimension,
                    input_dimension,
                    core.shape[-1],
                )
            )
            current_rank = core.shape[-1]
        else:
            core = np.zeros(
                (
                    current_rank,
                    output_dimension,
                    input_dimension,
                    current_rank,
                ),
                dtype=np.result_type(*operands),
            )
            diagonal_dimension = min(output_dimension, input_dimension)
            for rank in range(current_rank):
                for physical in range(diagonal_dimension):
                    core[rank, physical, physical, rank] = scalar
            scalar = 1.0
            result.append(core)
    if result and scalar != 1.0:
        result[0] = result[0] * scalar
    return result


def _remove_unit_output_cores(cores, labels, output_labels):
    cores = list(cores)
    labels = list(labels)
    output_set = set(output_labels)
    position = len(cores) - 1
    while position >= 0:
        if labels[position] in output_set:
            position -= 1
            continue
        matrix = cores[position][:, 0, :]
        if len(cores) == 1:
            return [], (), matrix.reshape(()).item()
        if position == 0:
            cores[1] = np.tensordot(matrix, cores[1], axes=([1], [0]))
        else:
            cores[position - 1] = np.tensordot(
                cores[position - 1],
                matrix,
                axes=([-1], [0]),
            )
        cores.pop(position)
        labels.pop(position)
        position -= 1
    return cores, tuple(labels), None


class BlockDiagonalMetric:
    """Block-diagonal local overlap metric indexed by physical sectors."""

    def __init__(self, size, blocks, indices):
        self.size = int(size)
        self.blocks = tuple(np.asarray(block) for block in blocks)
        self.indices = tuple(np.asarray(index, dtype=int) for index in indices)
        if len(self.blocks) != len(self.indices):
            raise ValueError("metric blocks and index sets must have equal length.")
        for block, index in zip(self.blocks, self.indices):
            if block.shape != (index.size, index.size):
                raise ValueError("a metric block does not match its index set.")
        self.shape = (self.size, self.size)
        self.dtype = np.result_type(*self.blocks)

    def __matmul__(self, value):
        value = np.asarray(value)
        if value.shape[0] != self.size:
            raise ValueError("metric operand has an incompatible leading dimension.")
        result = np.zeros_like(value, dtype=np.result_type(self.dtype, value))
        for block, index in zip(self.blocks, self.indices):
            result[index] = block @ value[index]
        return result

    def restrict(self, retained_indices):
        """Return this metric in a sorted subset of its coordinates."""

        retained_indices = np.asarray(retained_indices, dtype=int)
        if retained_indices.ndim != 1:
            raise ValueError("retained_indices must be one-dimensional.")
        if retained_indices.size == 0:
            raise ValueError("a restricted metric must retain at least one index.")
        if (
            np.any(retained_indices < 0)
            or np.any(retained_indices >= self.size)
            or np.unique(retained_indices).size != retained_indices.size
        ):
            raise ValueError("retained metric indices are invalid.")
        positions = np.full(self.size, -1, dtype=int)
        positions[retained_indices] = np.arange(retained_indices.size)
        blocks = []
        indices = []
        for block, block_indices in zip(self.blocks, self.indices):
            keep = positions[block_indices] >= 0
            if not np.any(keep):
                continue
            blocks.append(block[np.ix_(keep, keep)])
            indices.append(positions[block_indices[keep]])
        return BlockDiagonalMetric(retained_indices.size, blocks, indices)

    def to_dense(self):
        result = np.zeros(self.shape, dtype=self.dtype)
        for block, index in zip(self.blocks, self.indices):
            result[np.ix_(index, index)] = block
        return result

    def whitening_basis(self, tolerance):
        """Return a matrix that maps Euclidean vectors to metric-normalized ones."""

        decompositions = []
        scale = 0.0
        for block in self.blocks:
            hermitian = 0.5 * (block + block.conj().T)
            values, vectors = np.linalg.eigh(hermitian)
            decompositions.append((values, vectors))
            if values.size:
                scale = max(scale, float(values[-1]))
        if scale <= 0.0:
            raise ValueError("the local LETTA overlap metric has zero rank.")
        relative_cutoff = max(
            float(tolerance),
            np.finfo(float).eps * self.size,
        )
        cutoff = relative_cutoff * scale
        columns = []
        for index, (values, vectors) in zip(self.indices, decompositions):
            retained = values > cutoff
            for local_column in np.nonzero(retained)[0]:
                column = np.zeros(self.size, dtype=self.dtype)
                column[index] = (
                    vectors[:, local_column] / np.sqrt(values[local_column])
                )
                columns.append(column)
        if not columns:
            raise ValueError("the local LETTA overlap metric has zero rank.")
        return np.column_stack(columns), len(columns)


def _environment_operands(environment, labels, slot):
    if not isinstance(environment, BoundaryMPS):
        return [environment], [tuple(labels)]
    if tuple(labels) != environment.labels:
        raise ValueError("boundary-MPS labels do not match the contraction cut.")
    base = -1 - slot * 1_000_000
    bonds = tuple(base - index for index in range(len(labels) + 1))
    core_labels = [
        (bonds[index], label, bonds[index + 1])
        for index, label in enumerate(labels)
    ]
    return list(environment.cores), core_labels


def _validate_network(state, mpo):
    if not isinstance(mpo, LatticeMPO):
        raise TypeError("operator must be a LatticeMPO.")
    if mpo.nsites != state.nsites:
        raise ValueError("MPO length does not match the LETTA state.")
    if mpo.physical_dim != state.physical_dim:
        raise ValueError("MPO physical dimension does not match the LETTA state.")
    if (
        mpo.lattice_shape is not None
        and mpo.lattice_shape != state.lattice_shape
    ):
        raise ValueError("MPO lattice shape does not match the LETTA state.")


def _network_specification(state, mpo, active_site):
    nsites = state.nsites
    next_label = 0

    def labels(count):
        nonlocal next_label
        result = list(range(next_label, next_label + count))
        next_label += count
        return result

    bra_virtual = labels(nsites + 1)
    ket_virtual = labels(nsites + 1)
    mpo_virtual = labels(nsites + 1)
    bra_physical = labels(nsites)
    ket_physical = labels(nsites)
    operands = []
    operand_labels = []

    for site, tensor in enumerate(state.tensors):
        if site == active_site:
            continue
        neighborhood = state.site_neighborhood(site)
        bra_indices = (
            [bra_virtual[site]]
            + [bra_physical[index] for index in neighborhood]
            + [bra_virtual[site + 1]]
        )
        ket_indices = (
            [ket_virtual[site]]
            + [ket_physical[index] for index in neighborhood]
            + [ket_virtual[site + 1]]
        )
        operands.extend([tensor.conj(), tensor])
        operand_labels.extend([bra_indices, ket_indices])

    for site, factor in enumerate(mpo.factors):
        operands.append(factor)
        operand_labels.append(
            [
                mpo_virtual[site],
                mpo_virtual[site + 1],
                bra_physical[site],
                ket_physical[site],
            ]
        )

    if active_site is None:
        output = []
    else:
        neighborhood = state.site_neighborhood(active_site)
        output = (
            [bra_virtual[active_site]]
            + [bra_physical[index] for index in neighborhood]
            + [bra_virtual[active_site + 1]]
            + [ket_virtual[active_site]]
            + [ket_physical[index] for index in neighborhood]
            + [ket_virtual[active_site + 1]]
        )
        used = {label for indices in operand_labels for label in indices}
        tensor_shape = state.tensors[active_site].shape
        output_dimensions = tensor_shape + tensor_shape
        for label, dimension in zip(output, output_dimensions):
            if label not in used:
                operands.append(np.ones(dimension))
                operand_labels.append([label])

    return operands, operand_labels, output


def _canonical_signature(operands, labels, output):
    mapping = {}

    def canonical(label):
        if label not in mapping:
            mapping[label] = len(mapping)
        return mapping[label]

    canonical_labels = tuple(
        tuple(canonical(label) for label in indices) for indices in labels
    )
    canonical_output = tuple(canonical(label) for label in output)
    return (
        tuple(operand.shape for operand in operands),
        canonical_labels,
        canonical_output,
    )


@lru_cache(maxsize=512)
def _compiled_contraction(signature):
    shapes, labels, output = signature
    inputs = [
        "".join(oe.get_symbol(label) for label in indices) for indices in labels
    ]
    result = "".join(oe.get_symbol(label) for label in output)
    equation = ",".join(inputs) + "->" + result
    return oe.contract_expression(
        equation,
        *shapes,
        optimize="greedy",
    )


def _contract_operands(operands, labels, output):
    signature = _canonical_signature(operands, labels, output)
    expression = _compiled_contraction(signature)
    return expression(*operands)


def _contract(state, mpo, active_site=None):
    _validate_network(state, mpo)
    if active_site is not None:
        active_site = int(active_site)
        if active_site < 0 or active_site >= state.nsites:
            raise IndexError("active LETTA site is out of range.")
    operands, labels, output = _network_specification(
        state,
        mpo,
        active_site,
    )
    return _contract_operands(operands, labels, output)


class LETTAEnvironmentCache:
    """Reusable exact frontier contractions for one LETTA-MPO sweep."""

    def __init__(
        self,
        state,
        mpo,
        *,
        use_sparse_mpo=True,
        boundary_bond_dim=None,
        boundary_cutoff=0.0,
    ):
        _validate_network(state, mpo)
        self.state = state
        self.mpo = mpo
        self.boundary_bond_dim = boundary_bond_dim
        self.boundary_cutoff = float(boundary_cutoff)
        self.compression_errors = []
        self.use_sparse_mpo = bool(use_sparse_mpo) and boundary_bond_dim is None
        nsites = state.nsites
        next_label = 0

        def labels(count):
            nonlocal next_label
            result = tuple(range(next_label, next_label + count))
            next_label += count
            return result

        self.bra_virtual = labels(nsites + 1)
        self.ket_virtual = labels(nsites + 1)
        self.mpo_virtual = labels(nsites + 1)
        self.bra_physical = labels(nsites)
        self.ket_physical = labels(nsites)
        group_labels = tuple(self._group_labels(site) for site in range(nsites))
        self.label_dimensions = {}
        for site, group in enumerate(group_labels):
            shapes = (
                state.tensors[site].shape,
                state.tensors[site].shape,
                mpo.factors[site].shape,
            )
            for indices, shape in zip(group, shapes):
                for label, dimension in zip(indices, shape):
                    previous = self.label_dimensions.setdefault(label, dimension)
                    if previous != dimension:
                        raise ValueError("inconsistent tensor-network index dimension.")
        label_sites = {}
        for site, group in enumerate(group_labels):
            for indices in group:
                for label in indices:
                    label_sites.setdefault(label, set()).add(site)
        self.frontiers = tuple(
            tuple(
                sorted(
                    label
                    for label, sites in label_sites.items()
                    if min(sites) < cut <= max(sites)
                )
            )
            for cut in range(nsites + 1)
        )
        self.transition_groups = tuple(
            self._group_transitions(site) for site in range(nsites)
        )

    def _group_transitions(self, site):
        groups = {}
        for left_channel, right_channel, local_operator in self.mpo.transitions[site]:
            operator = np.ascontiguousarray(local_operator)
            key = (operator.dtype.str, operator.shape, operator.tobytes())
            if key not in groups:
                groups[key] = [operator, []]
            groups[key][1].append((left_channel, right_channel))
        return tuple(
            (operator, tuple(channels)) for operator, channels in groups.values()
        )

    def _group_labels(self, site):
        neighborhood = self.state.site_neighborhood(site)
        bra = (
            (self.bra_virtual[site],)
            + tuple(self.bra_physical[index] for index in neighborhood)
            + (self.bra_virtual[site + 1],)
        )
        ket = (
            (self.ket_virtual[site],)
            + tuple(self.ket_physical[index] for index in neighborhood)
            + (self.ket_virtual[site + 1],)
        )
        operator = (
            self.mpo_virtual[site],
            self.mpo_virtual[site + 1],
            self.bra_physical[site],
            self.ket_physical[site],
        )
        return bra, ket, operator

    def _group(self, site):
        tensor = self.state.tensors[site]
        return (
            [tensor.conj(), tensor, self.mpo.factors[site]],
            list(self._group_labels(site)),
        )

    def scalar_boundary(self):
        dtype = np.result_type(*self.state.tensors, *self.mpo.factors)
        return np.asarray(1.0, dtype=dtype)

    def _compress(self, tensor, labels):
        if self.boundary_bond_dim is None or not labels:
            return tensor
        compressed = BoundaryMPS.from_dense(
            tensor,
            labels,
            max_bond_dim=self.boundary_bond_dim,
            cutoff=self.boundary_cutoff,
        )
        self.compression_errors.append(compressed.discarded_weight)
        return compressed

    def extend_left(self, left, site):
        if self.use_sparse_mpo:
            return self._sparse_extend(left, site, "lr")
        operands, labels = self._group(site)
        if isinstance(left, BoundaryMPS):
            result = left.apply_local_transfer(
                input_labels=self.frontiers[site],
                output_labels=self.frontiers[site + 1],
                operands=operands,
                operand_labels=labels,
                label_dimensions=self.label_dimensions,
                max_bond_dim=self.boundary_bond_dim,
                cutoff=self.boundary_cutoff,
            )
            if isinstance(result, BoundaryMPS):
                self.compression_errors.append(result.discarded_weight)
            return result
        environment_operands, environment_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        result = _contract_operands(
            environment_operands + operands,
            environment_labels + labels,
            self.frontiers[site + 1],
        )
        return self._compress(result, self.frontiers[site + 1])

    def extend_right(self, right, site):
        if self.use_sparse_mpo:
            return self._sparse_extend(right, site, "rl")
        operands, labels = self._group(site)
        if isinstance(right, BoundaryMPS):
            result = right.apply_local_transfer(
                input_labels=self.frontiers[site + 1],
                output_labels=self.frontiers[site],
                operands=operands,
                operand_labels=labels,
                label_dimensions=self.label_dimensions,
                max_bond_dim=self.boundary_bond_dim,
                cutoff=self.boundary_cutoff,
            )
            if isinstance(result, BoundaryMPS):
                self.compression_errors.append(result.discarded_weight)
            return result
        environment_operands, environment_labels = _environment_operands(
            right, self.frontiers[site + 1], 0
        )
        result = _contract_operands(
            operands + environment_operands,
            labels + environment_labels,
            self.frontiers[site],
        )
        return self._compress(result, self.frontiers[site])

    def build_left_environments(self):
        environments = [None] * (self.state.nsites + 1)
        environments[0] = self.scalar_boundary()
        for site in range(self.state.nsites):
            environments[site + 1] = self.extend_left(
                environments[site],
                site,
            )
        return environments

    def build_right_environments(self):
        environments = [None] * (self.state.nsites + 1)
        environments[-1] = self.scalar_boundary()
        for site in range(self.state.nsites - 1, -1, -1):
            environments[site] = self.extend_right(
                environments[site + 1],
                site,
            )
        return environments

    @staticmethod
    def _select_channel(tensor, labels, channel_label, channel):
        labels = list(labels)
        if channel_label not in labels:
            if channel != 0:
                return None, None
            return tensor, labels
        axis = labels.index(channel_label)
        return np.take(tensor, channel, axis=axis), labels[:axis] + labels[axis + 1 :]

    def _empty_frontier(self, labels, dtype):
        nsites = self.state.nsites
        for cut in range(nsites + 1):
            if cut == 0:
                dimension = self.state.tensors[0].shape[0]
            elif cut == nsites:
                dimension = self.state.tensors[-1].shape[-1]
            else:
                dimension = self.state.tensors[cut].shape[0]
            self.label_dimensions[self.bra_virtual[cut]] = dimension
            self.label_dimensions[self.ket_virtual[cut]] = dimension
        return np.zeros(
            tuple(self.label_dimensions[label] for label in labels),
            dtype=dtype,
        )

    @staticmethod
    def _add_channel(result, labels, channel_label, channel, value):
        if channel_label not in labels:
            result[...] += value
            return
        axis = labels.index(channel_label)
        destination = [slice(None)] * result.ndim
        destination[axis] = channel
        result[tuple(destination)] += value

    def _sparse_extend(self, environment, site, direction):
        bra_labels, ket_labels, operator_labels = self._group_labels(site)
        bra_physical, ket_physical = operator_labels[2:]
        tensor = self.state.tensors[site]
        if direction == "lr":
            input_labels = self.frontiers[site]
            output_labels = self.frontiers[site + 1]
            input_channel = self.mpo_virtual[site]
            output_channel = self.mpo_virtual[site + 1]
        else:
            input_labels = self.frontiers[site + 1]
            output_labels = self.frontiers[site]
            input_channel = self.mpo_virtual[site + 1]
            output_channel = self.mpo_virtual[site]

        result = self._empty_frontier(
            output_labels,
            np.result_type(environment, tensor, self.mpo.factors[site]),
        )
        reduced_output = tuple(
            label for label in output_labels if label != output_channel
        )
        batch_label = -1
        for local_operator, channels in self.transition_groups[site]:
            selected_environments = []
            targets = []
            selected_labels = None
            for left_channel, right_channel in channels:
                selected = left_channel if direction == "lr" else right_channel
                target = right_channel if direction == "lr" else left_channel
                selected_environment, current_labels = self._select_channel(
                    environment,
                    input_labels,
                    input_channel,
                    selected,
                )
                if selected_environment is None:
                    continue
                selected_environments.append(selected_environment)
                targets.append(target)
                selected_labels = current_labels
            if not selected_environments:
                continue
            values = _contract_operands(
                [
                    np.stack(selected_environments),
                    tensor.conj(),
                    tensor,
                    local_operator,
                ],
                [
                    (batch_label,) + tuple(selected_labels),
                    bra_labels,
                    ket_labels,
                    (bra_physical, ket_physical),
                ],
                (batch_label,) + reduced_output,
            )
            for target, value in zip(targets, values):
                self._add_channel(
                    result,
                    output_labels,
                    output_channel,
                    target,
                    value,
                )
        return result

    def effective_matrix(self, left, right, site):
        if self.use_sparse_mpo:
            return self._sparse_effective_matrix(left, right, site)
        site = int(site)
        tensor = self.state.tensors[site]
        neighborhood = self.state.site_neighborhood(site)
        output = (
            (self.bra_virtual[site],)
            + tuple(self.bra_physical[index] for index in neighborhood)
            + (self.bra_virtual[site + 1], self.ket_virtual[site])
            + tuple(self.ket_physical[index] for index in neighborhood)
            + (self.ket_virtual[site + 1],)
        )
        operator_labels = self._group_labels(site)[2]
        left_operands, left_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        right_operands, right_labels = _environment_operands(
            right, self.frontiers[site + 1], 1
        )
        operands = left_operands + [self.mpo.factors[site]] + right_operands
        labels = left_labels + [operator_labels] + right_labels
        used = {label for indices in labels for label in indices}
        for label, dimension in zip(output, tensor.shape + tensor.shape):
            if label not in used:
                operands.append(np.ones(dimension))
                labels.append((label,))
        effective = _contract_operands(operands, labels, output)
        return effective.reshape(tensor.size, tensor.size)

    def _sparse_effective_matrix(self, left, right, site):
        site = int(site)
        tensor = self.state.tensors[site]
        neighborhood = self.state.site_neighborhood(site)
        output = (
            (self.bra_virtual[site],)
            + tuple(self.bra_physical[index] for index in neighborhood)
            + (self.bra_virtual[site + 1], self.ket_virtual[site])
            + tuple(self.ket_physical[index] for index in neighborhood)
            + (self.ket_virtual[site + 1],)
        )
        operator_labels = self._group_labels(site)[2]
        bra_physical, ket_physical = operator_labels[2:]
        effective = np.zeros(
            tensor.shape + tensor.shape,
            dtype=np.result_type(left, right, tensor, self.mpo.factors[site]),
        )
        batch_label = -1
        for local_operator, channels in self.transition_groups[site]:
            selected_lefts = []
            selected_rights = []
            left_labels = None
            right_labels = None
            for left_channel, right_channel in channels:
                selected_left, current_left_labels = self._select_channel(
                    left,
                    self.frontiers[site],
                    self.mpo_virtual[site],
                    left_channel,
                )
                selected_right, current_right_labels = self._select_channel(
                    right,
                    self.frontiers[site + 1],
                    self.mpo_virtual[site + 1],
                    right_channel,
                )
                if selected_left is None or selected_right is None:
                    continue
                selected_lefts.append(selected_left)
                selected_rights.append(selected_right)
                left_labels = current_left_labels
                right_labels = current_right_labels
            if not selected_lefts:
                continue
            operands = [
                np.stack(selected_lefts),
                local_operator,
                np.stack(selected_rights),
            ]
            labels = [
                (batch_label,) + tuple(left_labels),
                (bra_physical, ket_physical),
                (batch_label,) + tuple(right_labels),
            ]
            used = {label for indices in labels for label in indices}
            for label, dimension in zip(output, tensor.shape + tensor.shape):
                if label not in used:
                    operands.append(np.ones(dimension))
                    labels.append((label,))
            effective += _contract_operands(operands, labels, output)
        return effective.reshape(tensor.size, tensor.size)

    def effective_action(self, left, right, site, vector):
        site = int(site)
        tensor = self.state.tensors[site]
        vector = np.asarray(vector).reshape(tensor.shape)
        neighborhood = self.state.site_neighborhood(site)
        bra_output = (
            (self.bra_virtual[site],)
            + tuple(self.bra_physical[index] for index in neighborhood)
            + (self.bra_virtual[site + 1],)
        )
        ket_active = (
            (self.ket_virtual[site],)
            + tuple(self.ket_physical[index] for index in neighborhood)
            + (self.ket_virtual[site + 1],)
        )
        operator_labels = self._group_labels(site)[2]
        bra_physical, ket_physical = operator_labels[2:]
        if self.boundary_bond_dim is not None:
            left_operands, left_labels = _environment_operands(
                left, self.frontiers[site], 0
            )
            right_operands, right_labels = _environment_operands(
                right, self.frontiers[site + 1], 1
            )
            operands = (
                left_operands
                + [self.mpo.factors[site]]
                + right_operands
                + [vector]
            )
            labels = left_labels + [operator_labels] + right_labels + [ket_active]
            used = {label for indices in labels for label in indices}
            for label, dimension in zip(bra_output, tensor.shape):
                if label not in used:
                    operands.append(np.ones(dimension))
                    labels.append((label,))
            return _contract_operands(operands, labels, bra_output).reshape(-1)
        result = np.zeros(tensor.shape, dtype=np.result_type(left, right, vector))
        batch_label = -1
        for local_operator, channels in self.transition_groups[site]:
            selected_lefts = []
            selected_rights = []
            left_labels = None
            right_labels = None
            for left_channel, right_channel in channels:
                selected_left, current_left_labels = self._select_channel(
                    left,
                    self.frontiers[site],
                    self.mpo_virtual[site],
                    left_channel,
                )
                selected_right, current_right_labels = self._select_channel(
                    right,
                    self.frontiers[site + 1],
                    self.mpo_virtual[site + 1],
                    right_channel,
                )
                if selected_left is None or selected_right is None:
                    continue
                selected_lefts.append(selected_left)
                selected_rights.append(selected_right)
                left_labels = current_left_labels
                right_labels = current_right_labels
            if not selected_lefts:
                continue
            operands = [
                np.stack(selected_lefts),
                local_operator,
                np.stack(selected_rights),
                vector,
            ]
            labels = [
                (batch_label,) + tuple(left_labels),
                (bra_physical, ket_physical),
                (batch_label,) + tuple(right_labels),
                ket_active,
            ]
            used = {label for indices in labels for label in indices}
            for label, dimension in zip(bra_output, tensor.shape):
                if label not in used:
                    operands.append(np.ones(dimension))
                    labels.append((label,))
            result += _contract_operands(operands, labels, bra_output)
        return result.reshape(-1)


class IdentityEnvironmentCache:
    """Exact overlap environments with bra and ket physical labels fused."""

    def __init__(self, state, *, boundary_bond_dim=None, boundary_cutoff=0.0):
        self.state = state
        self.boundary_bond_dim = boundary_bond_dim
        self.boundary_cutoff = float(boundary_cutoff)
        self.compression_errors = []
        nsites = state.nsites
        next_label = 0

        def labels(count):
            nonlocal next_label
            result = tuple(range(next_label, next_label + count))
            next_label += count
            return result

        self.bra_virtual = labels(nsites + 1)
        self.ket_virtual = labels(nsites + 1)
        self.physical = labels(nsites)
        group_labels = tuple(self._group_labels(site) for site in range(nsites))
        self.label_dimensions = {}
        for site, group in enumerate(group_labels):
            tensor_shape = state.tensors[site].shape
            for indices, shape in zip(group, (tensor_shape, tensor_shape)):
                for label, dimension in zip(indices, shape):
                    previous = self.label_dimensions.setdefault(label, dimension)
                    if previous != dimension:
                        raise ValueError(
                            "inconsistent tensor-network index dimension."
                        )
        label_sites = {}
        for site, group in enumerate(group_labels):
            for indices in group:
                for label in indices:
                    label_sites.setdefault(label, set()).add(site)
        self.frontiers = tuple(
            tuple(
                sorted(
                    label
                    for label, sites in label_sites.items()
                    if min(sites) < cut <= max(sites)
                )
            )
            for cut in range(nsites + 1)
        )

    def _group_labels(self, site):
        neighborhood = self.state.site_neighborhood(site)
        physical = tuple(self.physical[index] for index in neighborhood)
        return (
            (self.bra_virtual[site],)
            + physical
            + (self.bra_virtual[site + 1],),
            (self.ket_virtual[site],)
            + physical
            + (self.ket_virtual[site + 1],),
        )

    def _group(self, site):
        tensor = self.state.tensors[site]
        return [tensor.conj(), tensor], list(self._group_labels(site))

    def scalar_boundary(self):
        return np.asarray(1.0, dtype=np.result_type(*self.state.tensors))

    def _compress(self, tensor, labels):
        if self.boundary_bond_dim is None or not labels:
            return tensor
        compressed = BoundaryMPS.from_dense(
            tensor,
            labels,
            max_bond_dim=self.boundary_bond_dim,
            cutoff=self.boundary_cutoff,
        )
        self.compression_errors.append(compressed.discarded_weight)
        return compressed

    def extend_left(self, left, site):
        operands, labels = self._group(site)
        if isinstance(left, BoundaryMPS):
            result = left.apply_local_transfer(
                input_labels=self.frontiers[site],
                output_labels=self.frontiers[site + 1],
                operands=operands,
                operand_labels=labels,
                label_dimensions=self.label_dimensions,
                max_bond_dim=self.boundary_bond_dim,
                cutoff=self.boundary_cutoff,
            )
            if isinstance(result, BoundaryMPS):
                self.compression_errors.append(result.discarded_weight)
            return result
        environment_operands, environment_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        result = _contract_operands(
            environment_operands + operands,
            environment_labels + labels,
            self.frontiers[site + 1],
        )
        return self._compress(result, self.frontiers[site + 1])

    def extend_right(self, right, site):
        operands, labels = self._group(site)
        if isinstance(right, BoundaryMPS):
            result = right.apply_local_transfer(
                input_labels=self.frontiers[site + 1],
                output_labels=self.frontiers[site],
                operands=operands,
                operand_labels=labels,
                label_dimensions=self.label_dimensions,
                max_bond_dim=self.boundary_bond_dim,
                cutoff=self.boundary_cutoff,
            )
            if isinstance(result, BoundaryMPS):
                self.compression_errors.append(result.discarded_weight)
            return result
        environment_operands, environment_labels = _environment_operands(
            right, self.frontiers[site + 1], 0
        )
        result = _contract_operands(
            operands + environment_operands,
            labels + environment_labels,
            self.frontiers[site],
        )
        return self._compress(result, self.frontiers[site])

    def build_left_environments(self):
        environments = [None] * (self.state.nsites + 1)
        environments[0] = self.scalar_boundary()
        for site in range(self.state.nsites):
            environments[site + 1] = self.extend_left(environments[site], site)
        return environments

    def build_right_environments(self):
        environments = [None] * (self.state.nsites + 1)
        environments[-1] = self.scalar_boundary()
        for site in range(self.state.nsites - 1, -1, -1):
            environments[site] = self.extend_right(environments[site + 1], site)
        return environments

    def _reduced_metric(self, left, right, site):
        site = int(site)
        tensor = self.state.tensors[site]
        neighborhood = self.state.site_neighborhood(site)
        physical = tuple(self.physical[index] for index in neighborhood)
        output = (
            (self.bra_virtual[site],)
            + physical
            + (
                self.bra_virtual[site + 1],
                self.ket_virtual[site],
                self.ket_virtual[site + 1],
            )
        )
        left_operands, left_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        right_operands, right_labels = _environment_operands(
            right, self.frontiers[site + 1], 1
        )
        operands = left_operands + right_operands
        labels = left_labels + right_labels
        used = {label for indices in labels for label in indices}
        reduced_dimensions = (
            (tensor.shape[0],)
            + tensor.shape[1:-1]
            + (tensor.shape[-1], tensor.shape[0], tensor.shape[-1])
        )
        for label, dimension in zip(output, reduced_dimensions):
            if label not in used:
                operands.append(np.ones(dimension))
                labels.append((label,))
        return _contract_operands(operands, labels, output)

    def effective_metric(self, left, right, site):
        site = int(site)
        tensor = self.state.tensors[site]
        reduced = self._reduced_metric(left, right, site)
        physical_shape = tensor.shape[1:-1]
        flat_indices = np.arange(tensor.size).reshape(tensor.shape)
        blocks = []
        indices = []
        for configuration in np.ndindex(*physical_shape):
            source = (slice(None),) + configuration + (slice(None),) * 3
            block = reduced[source].reshape(
                tensor.shape[0] * tensor.shape[-1],
                tensor.shape[0] * tensor.shape[-1],
            )
            index = flat_indices[
                (slice(None),) + configuration + (slice(None),)
            ].reshape(-1)
            blocks.append(block)
            indices.append(index)
        return BlockDiagonalMetric(tensor.size, blocks, indices)

    def effective_matrix(self, left, right, site):
        return self.effective_metric(left, right, site).to_dense()

    def effective_action(self, left, right, site, vector):
        site = int(site)
        tensor = self.state.tensors[site]
        vector = np.asarray(vector).reshape(tensor.shape)
        neighborhood = self.state.site_neighborhood(site)
        physical = tuple(self.physical[index] for index in neighborhood)
        bra_output = (
            (self.bra_virtual[site],)
            + physical
            + (self.bra_virtual[site + 1],)
        )
        ket_active = (
            (self.ket_virtual[site],)
            + physical
            + (self.ket_virtual[site + 1],)
        )
        left_operands, left_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        right_operands, right_labels = _environment_operands(
            right, self.frontiers[site + 1], 1
        )
        operands = left_operands + right_operands + [vector]
        labels = left_labels + right_labels + [ket_active]
        used = {label for indices in labels for label in indices}
        for label, dimension in zip(bra_output, tensor.shape):
            if label not in used:
                operands.append(np.ones(dimension))
                labels.append((label,))
        return _contract_operands(operands, labels, bra_output).reshape(-1)


def network_overlap(state):
    """Contract the LETTA norm without materializing its state vector."""

    cache = IdentityEnvironmentCache(state)
    overlap = cache.scalar_boundary()
    for site in range(state.nsites):
        overlap = cache.extend_left(overlap, site)
    return float(np.real_if_close(overlap))


def network_expectation(state, mpo):
    """Contract a normalized MPO expectation value directly."""

    cache = LETTAEnvironmentCache(state, mpo, use_sparse_mpo=True)
    numerator = cache.scalar_boundary()
    for site in range(state.nsites):
        numerator = cache.extend_left(numerator, site)
    denominator = network_overlap(state)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("cannot evaluate an operator on a zero LETTA state.")
    return np.real_if_close(numerator / denominator).item()


def network_operator_matrix(state, mpo, active_site):
    """Contract an effective one-tensor operator with active legs open."""

    active_site = int(active_site)
    tensor = state.tensors[active_site]
    effective_tensor = _contract(state, mpo, active_site)
    return effective_tensor.reshape(tensor.size, tensor.size)
