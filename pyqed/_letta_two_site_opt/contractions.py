"""Exact frontier contractions with two adjacent active LETTA tensors."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .._letta_one_site_opt.contractions import (
    BlockDiagonalMetric,
    IdentityEnvironmentCache,
    LETTAEnvironmentCache,
    _contract_operands,
    _environment_operands,
    _prepare_contraction,
)
from .pair import LETTAPairLayout


def _validate_layout(cache, layout):
    if not isinstance(layout, LETTAPairLayout):
        raise TypeError("layout must be a LETTAPairLayout.")
    if layout.left_site + 1 >= cache.state.nsites:
        raise IndexError("pair layout is outside the cached LETTA state.")
    if cache.state.site_neighborhood(layout.left_site) != layout.left_neighborhood:
        raise ValueError("pair layout does not match the cached LETTA state.")
    if (
        cache.state.site_neighborhood(layout.left_site + 1)
        != layout.right_neighborhood
    ):
        raise ValueError("pair layout does not match the cached LETTA state.")


class LETTAPairEnvironmentCache(LETTAEnvironmentCache):
    """Hamiltonian environments with an adjacent LETTA pair left active."""

    def prepare_pair_action(self, left, right, layout):
        """Bind pair environments and compatible MPO channels for one update.

        Scalar and batched factor solves share the channel slices. At most four
        batch shapes retain compiled local actions, all released after the pair.
        """
        _validate_layout(self, layout)
        site = layout.left_site
        bra = ((self.bra_virtual[site],)
               + tuple(self.bra_physical[p] for p in layout.merged_physical_sites)
               + (self.bra_virtual[site + 2],))
        ket = ((self.ket_virtual[site],)
               + tuple(self.ket_physical[p] for p in layout.merged_physical_sites)
               + (self.ket_virtual[site + 2],))
        first, second = self._group_labels(site)[2], self._group_labels(site + 1)[2]
        terms = []
        if self.use_sparse_mpo:
            by_middle = {}
            for middle, channel, operator in self.mpo.transitions[site + 1]:
                by_middle.setdefault(middle, []).append((channel, operator))
            for channel, middle, operator in self.mpo.transitions[site]:
                lhs, ll = self._select_channel(left, self.frontiers[site],
                                               self.mpo_virtual[site], channel)
                if lhs is None:
                    continue
                for channel, next_operator in by_middle.get(middle, ()):
                    rhs, rl = self._select_channel(right, self.frontiers[site + 2],
                                                   self.mpo_virtual[site + 2], channel)
                    if rhs is not None:
                        terms.append(([lhs, operator, next_operator, rhs],
                                      [tuple(ll), first[2:], second[2:], tuple(rl)]))
        else:
            lo, ll = _environment_operands(left, self.frontiers[site], 0)
            ro, rl = _environment_operands(right, self.frontiers[site + 2], 1)
            terms.append((lo + [self.mpo.factors[site], self.mpo.factors[site + 1]] + ro,
                          ll + [first, second] + rl))
        dtype = np.result_type(*self.state.tensors, *self.mpo.factors)
        size = int(np.prod(layout.merged_shape))

        @lru_cache(maxsize=4)
        def expressions(batch):
            shape = layout.merged_shape + (() if batch is None else (batch,))
            extra = () if batch is None else (-1,)
            result = []
            for fixed, labels in terms:
                operands = fixed + [np.empty(shape, dtype=dtype)]
                indices = labels + [ket + extra]
                used = {label for item in indices for label in item}
                for label, dimension in zip(bra + extra, shape):
                    if label not in used:
                        operands.append(np.ones(dimension))
                        indices.append((label,))
                result.append(_prepare_contraction(operands, indices, bra + extra, len(fixed)))
            return result

        def action(vector):
            vector = np.asarray(vector)
            if vector.ndim == 1 and vector.size == size:
                batch = None
            elif vector.ndim == 2 and vector.shape[0] == size:
                batch = vector.shape[1]
            else:
                raise ValueError("a pair action expects one vector or a column batch.")
            shape = layout.merged_shape + (() if batch is None else (batch,))
            value = vector.reshape(shape)
            result = np.zeros(shape, dtype=np.result_type(dtype, vector))
            for expression in expressions(batch):
                result += expression(value)
            return result.reshape(vector.shape)

        return action

    def effective_pair_action(self, left, right, layout, vector):
        _validate_layout(self, layout)
        if self.use_sparse_mpo:
            return self._sparse_effective_pair_action(
                left, right, layout, vector
            )
        site = layout.left_site
        vector = np.asarray(vector)
        if vector.ndim == 1:
            vector = vector.reshape(layout.merged_shape)
            batch_label = None
            output_shape = layout.merged_shape
        elif vector.ndim == 2 and vector.shape[0] == int(
            np.prod(layout.merged_shape)
        ):
            batch_label = -1
            output_shape = layout.merged_shape + (vector.shape[1],)
            vector = vector.reshape(output_shape)
        else:
            raise ValueError(
                "a pair action expects one vector or a column batch."
            )
        bra_output = (
            (self.bra_virtual[site],)
            + tuple(
                self.bra_physical[index]
                for index in layout.merged_physical_sites
            )
            + (self.bra_virtual[site + 2],)
        )
        ket_active = (
            (self.ket_virtual[site],)
            + tuple(
                self.ket_physical[index]
                for index in layout.merged_physical_sites
            )
            + (self.ket_virtual[site + 2],)
        )
        if batch_label is not None:
            bra_output = bra_output + (batch_label,)
            ket_active = ket_active + (batch_label,)
        first_operator = self._group_labels(site)[2]
        second_operator = self._group_labels(site + 1)[2]
        left_operands, left_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        right_operands, right_labels = _environment_operands(
            right, self.frontiers[site + 2], 1
        )
        operands = (
            left_operands
            + [self.mpo.factors[site], self.mpo.factors[site + 1]]
            + right_operands
            + [vector]
        )
        labels = (
            left_labels
            + [first_operator, second_operator]
            + right_labels
            + [ket_active]
        )
        used = {label for indices in labels for label in indices}
        for label, dimension in zip(bra_output, output_shape):
            if label not in used:
                operands.append(np.ones(dimension))
                labels.append((label,))
        result = _contract_operands(operands, labels, bra_output)
        if batch_label is None:
            return result.reshape(-1)
        return result.reshape(int(np.prod(layout.merged_shape)), vector.shape[-1])

    def _sparse_effective_pair_action(self, left, right, layout, vector):
        site = layout.left_site
        vector = np.asarray(vector)
        if vector.ndim == 1:
            vector = vector.reshape(layout.merged_shape)
            batch_label = None
            output_shape = layout.merged_shape
        elif vector.ndim == 2 and vector.shape[0] == int(
            np.prod(layout.merged_shape)
        ):
            batch_label = -1
            output_shape = layout.merged_shape + (vector.shape[1],)
            vector = vector.reshape(output_shape)
        else:
            raise ValueError(
                "a pair action expects one vector or a column batch."
            )
        bra_output = (
            (self.bra_virtual[site],)
            + tuple(
                self.bra_physical[index]
                for index in layout.merged_physical_sites
            )
            + (self.bra_virtual[site + 2],)
        )
        ket_active = (
            (self.ket_virtual[site],)
            + tuple(
                self.ket_physical[index]
                for index in layout.merged_physical_sites
            )
            + (self.ket_virtual[site + 2],)
        )
        if batch_label is not None:
            bra_output = bra_output + (batch_label,)
            ket_active = ket_active + (batch_label,)
        first_physical = self._group_labels(site)[2][2:]
        second_physical = self._group_labels(site + 1)[2][2:]
        result = np.zeros(
            output_shape,
            dtype=np.result_type(left, right, vector),
        )
        for (
            left_channel,
            middle_channel,
            first_operator,
        ) in self.mpo.transitions[site]:
            selected_left, left_labels = self._select_channel(
                left,
                self.frontiers[site],
                self.mpo_virtual[site],
                left_channel,
            )
            if selected_left is None:
                continue
            for (
                second_middle,
                right_channel,
                second_operator,
            ) in self.mpo.transitions[site + 1]:
                if second_middle != middle_channel:
                    continue
                selected_right, right_labels = self._select_channel(
                    right,
                    self.frontiers[site + 2],
                    self.mpo_virtual[site + 2],
                    right_channel,
                )
                if selected_right is None:
                    continue
                operands = [
                    selected_left,
                    first_operator,
                    second_operator,
                    selected_right,
                    vector,
                ]
                labels = [
                    tuple(left_labels),
                    first_physical,
                    second_physical,
                    tuple(right_labels),
                    ket_active,
                ]
                used = {label for indices in labels for label in indices}
                for label, dimension in zip(bra_output, output_shape):
                    if label not in used:
                        operands.append(np.ones(dimension))
                        labels.append((label,))
                result += _contract_operands(
                    operands, labels, bra_output
                )
        if batch_label is None:
            return result.reshape(-1)
        return result.reshape(int(np.prod(layout.merged_shape)), vector.shape[-1])

    def effective_pair_matrix(self, left, right, layout):
        dimension = int(np.prod(layout.merged_shape))
        identity = np.eye(
            dimension,
            dtype=np.result_type(
                left,
                right,
                *self.state.tensors,
                *self.mpo.factors,
            ),
        )
        matrix = np.column_stack(
            [
                self.effective_pair_action(
                    left, right, layout, identity[:, column]
                )
                for column in range(dimension)
            ]
        )
        return 0.5 * (matrix + matrix.conj().T)


class IdentityPairEnvironmentCache(IdentityEnvironmentCache):
    """Block-diagonal overlap metrics for one active LETTA pair."""

    def _reduced_pair_metric(self, left, right, layout):
        _validate_layout(self, layout)
        site = layout.left_site
        physical = tuple(
            self.physical[index] for index in layout.merged_physical_sites
        )
        output = (
            (self.bra_virtual[site],)
            + physical
            + (
                self.bra_virtual[site + 2],
                self.ket_virtual[site],
                self.ket_virtual[site + 2],
            )
        )
        left_operands, left_labels = _environment_operands(
            left, self.frontiers[site], 0
        )
        right_operands, right_labels = _environment_operands(
            right, self.frontiers[site + 2], 1
        )
        operands = left_operands + right_operands
        labels = left_labels + right_labels
        used = {label for indices in labels for label in indices}
        reduced_shape = (
            layout.merged_shape
            + (layout.merged_shape[0], layout.merged_shape[-1])
        )
        for label, dimension in zip(output, reduced_shape):
            if label not in used:
                operands.append(np.ones(dimension))
                labels.append((label,))
        return _contract_operands(operands, labels, output)

    def effective_pair_metric(self, left, right, layout):
        reduced = self._reduced_pair_metric(left, right, layout)
        physical_shape = layout.merged_shape[1:-1]
        flat_indices = np.arange(np.prod(layout.merged_shape)).reshape(
            layout.merged_shape
        )
        blocks = []
        indices = []
        for configuration in np.ndindex(*physical_shape):
            source = (slice(None),) + configuration + (slice(None),) * 3
            block = reduced[source].reshape(
                layout.merged_shape[0] * layout.merged_shape[-1],
                layout.merged_shape[0] * layout.merged_shape[-1],
            )
            index_set = flat_indices[
                (slice(None),) + configuration + (slice(None),)
            ].reshape(-1)
            blocks.append(block)
            indices.append(index_set)
        return BlockDiagonalMetric(
            int(np.prod(layout.merged_shape)), blocks, indices
        )

    def effective_pair_matrix(self, left, right, layout):
        return self.effective_pair_metric(left, right, layout).to_dense()

    def effective_pair_action(self, left, right, layout, vector):
        return self.effective_pair_metric(left, right, layout) @ vector
