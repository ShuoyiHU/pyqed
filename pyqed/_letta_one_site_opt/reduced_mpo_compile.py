"""Convert a scalar MPO to an explicitly spin-coupled operator MPS.

Only local MPO cores and polynomial virtual spaces are factorized. There is
no determinant-space Hamiltonian, wavefunction projection, or sector penalty.
The input builder's virtual spin labels are not assumed to be irreducible.
"""
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import svd

from pyqed.mps.su2 import SU2Irrep
from pyqed.mps.symmetry import Sector
from pyqed.mps.nonabelian.coupling import (
    clebsch_gordan, ordered_two_m_values,
)
from pyqed.mps.nonabelian.tensor import NonabelianTensor

from .reduced_symmetry import ReducedPhysicalBasis, _sector_irrep, SpinChargeSector


@lru_cache(maxsize=None)
def cg_tensor(left, physical, right):
    return np.array([[[clebsch_gordan(left, physical, right, ml, mp, mr)
                       for mr in ordered_two_m_values(right)]
                      for mp in ordered_two_m_values(physical)]
                     for ml in ordered_two_m_values(left)])


@dataclass(frozen=True)
class OperatorMultiplet:
    output_sector: object
    output_copy: int
    input_sector: object
    input_copy: int
    sector: Sector
    copy: int
    components: np.ndarray


def _operator_sector(output, input, irrep):
    """All conserved charge differences, with the spin tensor rank attached."""
    if isinstance(output, SpinChargeSector) and isinstance(input, SpinChargeSector):
        return Sector(('charge', 'su2'), (output.charge-input.charge, irrep))
    if not isinstance(output, Sector) or not isinstance(input, Sector) or output.labels != input.labels:
        raise TypeError('operator input/output sector labels must match')
    components = []
    for label, out, inp in zip(output.labels, output.components, input.components):
        if label == 'su2':
            components.append(irrep)
        elif label in {'pg', 'point_group', 'abelianpg'}:
            components.append(int(out)^int(inp))
        else:
            components.append(out-inp)
    return Sector(output.labels, tuple(components))


def operator_basis(physical_basis):
    """Orthonormal |out><in| basis coupled with the dual input irrep."""
    offsets, cursor = {}, 0
    for q, r in zip(physical_basis.sectors, physical_basis.multiplicities):
        offsets[q] = cursor
        cursor += r*_sector_irrep(q).dim
    grouped = defaultdict(list)
    for out in physical_basis.reduced_states:
        for inp in physical_basis.reduced_states:
            jo, ji = _sector_irrep(out.sector), _sector_irrep(inp.sector)
            for two_j in range(abs(jo.two_j-ji.two_j), jo.two_j+ji.two_j+1, 2):
                q = _operator_sector(out.sector, inp.sector, SU2Irrep(two_j))
                matrices = np.zeros((_sector_irrep(q).dim, cursor, cursor))
                for k, m in enumerate(ordered_two_m_values(_sector_irrep(q))):
                    for a, mo in enumerate(ordered_two_m_values(jo)):
                        for b, mi in enumerate(ordered_two_m_values(ji)):
                            matrices[k, offsets[out.sector]+out.copy*jo.dim+a,
                                     offsets[inp.sector]+inp.copy*ji.dim+b] = (
                                (-1.)**((ji.two_j-mi)//2)
                                * clebsch_gordan(jo, ji, _sector_irrep(q), mo, -mi, m))
                grouped[q].append((out, inp, matrices))
    channels, columns, sectors, multiplicities = [], [], [], []
    for q in sorted(grouped):
        sectors.append(q)
        multiplicities.append(len(grouped[q]))
        for copy, (out, inp, matrices) in enumerate(grouped[q]):
            channels.append(OperatorMultiplet(out.sector, out.copy, inp.sector,
                                             inp.copy, q, copy, matrices))
            columns.extend(matrices.reshape(_sector_irrep(q).dim, -1))
    basis = ReducedPhysicalBasis(tuple(str(q) for q in sectors), tuple(sectors),
                                 tuple(multiplicities))
    return basis, tuple(channels), np.array(columns).T


def _layout(multiplicities):
    offsets, cursor = {}, 0
    for q, r in multiplicities.items():
        offsets[q] = cursor
        cursor += r*_sector_irrep(q).dim
    return offsets, cursor


def _contract_operator_suffix(core, transfer, transform, d):
    """Exact local contraction, using sparse magnetic channel blocks if present.

    The output has W_left * d**2 * suffix_rank entries. In particular, no
    W_left * W_right * d**2 component core is required for RankCoupledMPO.
    This changes contraction order only, not a rank/truncation threshold.
    """
    from pyqed.mps.nonabelian.mpo import RankCoupledMPO
    if isinstance(core, RankCoupledMPO):
        if (core.phys_out_leg.total_dim != d or core.phys_in_leg.total_dim != d
                or core.right_dim != transfer.shape[0]):
            raise ValueError('MPO core does not match physical basis or suffix')
        values = np.zeros((core.left_dim, d, d, transfer.shape[1]),
                          dtype=np.result_type(core.dtype, transfer.dtype))
        left, right = core._left_slices(), core._right_slices()
        out, inp = core.phys_out_leg.slices(), core.phys_in_leg.slices()
        for qo, so in out.items():
            for qi, si in inp.items():
                for (l, r), block in core.reduced_block(qo, qi).items():
                    values[left[l], so, si, :] += np.einsum(
                        'aboi,br->aoir', block, transfer[right[r]])
        return np.einsum('aoir,oip->apr', values, transform.conj().reshape(d, d, -1))
    w = np.asarray(core.as_dense() if hasattr(core, 'as_dense') else core)
    if w.ndim != 4 or w.shape[2:] != (d, d) or w.shape[1] != transfer.shape[0]:
        raise ValueError('MPO core does not match physical basis or suffix')
    values = np.tensordot(w, transfer, axes=(1, 0))
    return np.einsum('aoir,oip->apr', values, transform.conj().reshape(d, d, -1))


@dataclass(frozen=True)
class SpinTensorMPO:
    """Reduced MPO cores in an orthonormal irreducible local operator basis."""
    sites: tuple
    physical_basis: ReducedPhysicalBasis
    operator_basis: ReducedPhysicalBasis
    channels: tuple
    local_transform: np.ndarray
    relative_reconstruction_error: float

    @classmethod
    def compile(cls, factors, physical_basis, *, tolerance=2e-13):
        if not 0 < tolerance < 1:
            raise ValueError('MPO compilation tolerance must be between zero and one')
        op_basis, channels, transform = operator_basis(physical_basis)
        # Contract each original core with the suffix transfer before expanding
        # the physical operator basis. Never retain the full dense input chain.
        cores = [None] * len(factors)
        transfer = np.ones((1, 1))
        for i in range(len(factors)-1, -1, -1):
            a = _contract_operator_suffix(factors[i], transfer, transform,
                                          physical_basis.dense_dim)
            if i:
                q, r = np.linalg.qr(a.reshape(a.shape[0], -1).T, mode='reduced')
                cores[i] = q.T.reshape(q.shape[1], a.shape[1], a.shape[2])
                transfer = r.T
            else:
                cores[0] = a
        if not cores or cores[0].shape[0] != 1:
            raise ValueError('scalar MPO must have unit open boundaries')
        scalar = _operator_sector(physical_basis.sectors[0], physical_basis.sectors[0], SU2Irrep(0))
        if np.linalg.norm(cores[0]) == 0:
            # Retain a scalar identity channel with zero amplitude at site 0.
            values = [np.trace(c.components[0]) for c in channels if c.sector == scalar]
            physical_qns = [q for q, r in zip(op_basis.sectors, op_basis.multiplicities)
                            for _ in range(r)]
            sites = tuple(NonabelianTensor(
                data={(scalar, scalar, scalar): np.asarray(values).reshape(1, -1, 1)*(i != 0)},
                qns=[[scalar], physical_qns, [scalar]], dirs=[-1, 1, 1],
                metadata={'physical_basis': 'fully_reduced_su2'}) for i in range(len(cores)))
            return cls(sites, physical_basis, op_basis, channels, transform, 0.)
        left_mult = {scalar: 1}
        op_mult = dict(zip(op_basis.sectors, op_basis.multiplicities))
        poff, _ = _layout(op_mult)
        sites, residuals = [], []
        for i, a in enumerate(cores):
            loff, _ = _layout(left_mult)
            groups = defaultdict(list)
            for ql, rl in left_mult.items():
                for qp, rp in op_mult.items():
                    part = a[loff[ql]:loff[ql]+rl*_sector_irrep(ql).dim,
                             poff[qp]:poff[qp]+rp*_sector_irrep(qp).dim].reshape(
                                 rl, _sector_irrep(ql).dim, rp, _sector_irrep(qp).dim, a.shape[2])
                    for qr in ql.fuse(qp):
                        c = cg_tensor(_sector_irrep(ql), _sector_irrep(qp), _sector_irrep(qr))
                        reduced = np.einsum('ampnb,mnz->apzb', part, c)
                        groups[qr].append(((ql, qp, qr), reduced))
            blocks, transfers, right_mult = {}, {}, {}
            norm = np.linalg.norm(a)
            for qr, entries in sorted(groups.items()):
                matrix = np.concatenate([v.reshape(-1, _sector_irrep(qr).dim*a.shape[2])
                                         for _, v in entries])
                if i == len(cores)-1:
                    # A scalar Hamiltonian has no nontrivial final irrep.
                    if qr != scalar:
                        continue
                    u = matrix
                    transfer = np.ones((1, 1, 1), dtype=a.dtype)
                    rank = 1
                else:
                    u, singular, vh = svd(matrix, full_matrices=False)
                    rank = int(np.count_nonzero(singular > tolerance*max(norm, 1e-300)))
                    if rank == 0:
                        continue
                    u = u[:, :rank]
                    transfer = (singular[:rank, None]*vh[:rank]).reshape(
                        rank, _sector_irrep(qr).dim, a.shape[2])
                right_mult[qr] = rank
                transfers[qr] = transfer
                start = 0
                for key, value in entries:
                    rl, rp = value.shape[:2]
                    blocks[key] = u[start:start+rl*rp].reshape(rl, rp, rank)
                    start += rl*rp
            if not right_mult:
                raise ValueError('zero MPO requires an explicit scalar zero representation')
            roff, size = _layout(right_mult)
            transfer = np.concatenate([transfers[q].reshape(-1, a.shape[2]) for q in right_mult])
            rebuilt = np.zeros((a.shape[0], a.shape[1], size), dtype=a.dtype)
            for (ql, qp, qr), block in blocks.items():
                expanded = np.einsum('apr,mnz->ampnrz', block,
                                    cg_tensor(_sector_irrep(ql), _sector_irrep(qp), _sector_irrep(qr))).reshape(
                    block.shape[0]*_sector_irrep(ql).dim, block.shape[1]*_sector_irrep(qp).dim,
                    block.shape[2]*_sector_irrep(qr).dim)
                rebuilt[loff[ql]:loff[ql]+expanded.shape[0],
                        poff[qp]:poff[qp]+expanded.shape[1],
                        roff[qr]:roff[qr]+expanded.shape[2]] = expanded
            error = np.linalg.norm(a-np.tensordot(rebuilt, transfer, axes=(2, 0)))
            relative = error/max(norm, 1e-300)
            residuals.append(float(relative))
            if relative > max(1e-10, 100*tolerance):
                raise ValueError(f'MPO is not a charge-conserving SU(2) scalar at site {i}: residual {relative:g}')
            sites.append(NonabelianTensor(data=blocks,
                qns=[[q for q, r in mult.items() for _ in range(r)]
                     for mult in (left_mult, op_mult, right_mult)], dirs=[-1, 1, 1],
                metadata={'physical_basis': 'fully_reduced_su2'}))
            if i+1 < len(cores):
                cores[i+1] = np.tensordot(transfer, cores[i+1], axes=(1, 0))
            left_mult = right_mult
        return cls(tuple(sites), physical_basis, op_basis, channels, transform,
                   max(residuals))

    def component_factors(self):
        """Explicit local magnetic view for validation, never sweep execution."""
        from .reduced_contraction import expand_reduced_mps_site
        d = self.physical_basis.dense_dim
        transform = self.local_transform.reshape(d, d, -1)
        return tuple(np.einsum('apb,oip->aboi', expand_reduced_mps_site(a), transform)
                     for a in self.sites)
