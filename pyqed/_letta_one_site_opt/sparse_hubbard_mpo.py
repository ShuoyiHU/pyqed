"""Sparse charge/SU(2) Hubbard MPO with explicit fermionic string channels.

Exact finite-state construction for spin-independent one-body terms and onsite
U. It uses local irreducible operator coefficients, not magnetic bond expansion
or chain QR/SVD. Channels are grouped by starting orbital (not a minimal-rank
factorization of the hopping matrix). Local 4x4 operator matrices only determine
fixed reduced coefficients; no determinant-space projection is used.
"""
from collections import Counter
import numpy as np
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from .reduced_symmetry import ReducedPhysicalBasis, _sector_irrep
from .reduced_mpo_compile import SpinTensorMPO, operator_basis, cg_tensor


def sparse_hubbard_mpo(h1, onsite_u, ecore=0.):
    basis = ReducedPhysicalBasis.spatial_orbital()
    op_basis, channels, transform = operator_basis(basis)
    scalar = next(q for q in op_basis.sectors if q[0] == 0 and _sector_irrep(q).two_j == 0)
    spinors = {charge: next(q for q in op_basis.sectors
                          if q[0] == charge and _sector_irrep(q).two_j == 1)
               for charge in (-1, 1)}
    mult = dict(zip(op_basis.sectors, op_basis.multiplicities))
    by_sector = {q: [c for c in channels if c.sector == q] for q in op_basis.sectors}
    parity = np.diag([1., -1., -1., 1.])
    up = np.zeros((4,4)); up[0,1] = up[2,3] = 1.
    down = np.zeros((4,4)); down[0,2] = 1.; down[1,3] = -1.
    annihilators = (up, down)
    def scalar_coeff(matrix):
        return np.array([np.vdot(c.components[0], matrix) for c in by_sector[scalar]])
    identity_coeff = scalar_coeff(np.eye(4))
    parity_coeff = scalar_coeff(parity)
    endpoints = {}
    for charge, q in spinors.items():
        other = spinors[-charge]
        cg = cg_tensor(_sector_irrep(q), _sector_irrep(other), _sector_irrep(scalar))[:,:,0]
        # c_i^dagger c_j = (c_i^dagger P_i) P_(i+1)... c_j;
        # c_j^dagger c_i = (P_i c_i) P_(i+1)... c_j^dagger.
        pairs = [(a.T@parity, a) if charge == 1 else (parity@a, a.T)
                 for a in annihilators]
        coefficients = np.zeros((mult[q], mult[other]))
        for p, first in enumerate(by_sector[q]):
            for t, last in enumerate(by_sector[other]):
                for f, g in pairs:
                    v = np.array([np.vdot(c, f) for c in first.components])
                    w = np.array([np.vdot(c, g) for c in last.components])
                    coefficients[p,t] += np.einsum('m,n,mn->', v, w, cg)
        # A spin-summed hopping family has exactly one copy channel. Factor
        # its tiny fixed coefficient matrix by a pivot, without a rank cutoff.
        p,t = np.unravel_index(np.argmax(abs(coefficients)), coefficients.shape)
        left = coefficients[:,t].copy()
        right = coefficients[p,:]/coefficients[p,t]
        if not np.allclose(np.outer(left,right), coefficients, rtol=2e-14, atol=2e-14):
            raise ValueError('Hubbard spinor endpoint factorization failed')
        # Validate the fixed two-orbital operator coefficients before use.
        rebuilt = np.zeros((4,4,4,4))
        for i, first in enumerate(by_sector[q]):
            for j, last in enumerate(by_sector[other]):
                rebuilt += coefficients[i,j]*np.einsum('mab,ncd,mn->abcd', first.components,last.components,cg)
        expected = sum(np.einsum('ab,cd->abcd', f,g) for f,g in pairs)
        if not np.allclose(rebuilt, expected, rtol=2e-14, atol=2e-14):
            raise ValueError('Hubbard spinor reconstruction failed')
        endpoints[charge] = (left,right)
    n = len(h1)
    def bond(cut):
        if cut == 0: return [(scalar,'start')]
        if cut == n: return [(scalar,'end')]
        live = [j for j in range(cut) if np.any(h1[j,cut:] != 0)]
        return [(scalar,'start'),(scalar,'end')]+[(spinors[c],(j,c)) for j in live for c in (-1,1)]
    sites=[]
    for site in range(n):
        left,right = bond(site),bond(site+1)
        counts_l,counts_r = Counter(q for q,_ in left),Counter(q for q,_ in right)
        def indices(labels):
            counts=Counter(); result={}
            for q,label in labels:
                result[label]=(q,counts[q]);counts[q]+=1
            return result
        li,ri = indices(left),indices(right);blocks={}
        def add(lab_l, qp, lab_r, coeff):
            if lab_l not in li or lab_r not in ri: return
            ql,l=li[lab_l];qr,r=ri[lab_r]
            if not np.any(coeff): return
            key=(ql,qp,qr)
            if key not in blocks: blocks[key]=np.zeros((counts_l[ql],mult[qp],counts_r[qr]))
            blocks[key][l,:,r] += coeff
        add('start',scalar,'start',identity_coeff)
        add('end',scalar,'end',identity_coeff)
        local=np.diag([0.,h1[site,site],h1[site,site],2*h1[site,site]+onsite_u[site]])
        if site == 0: local += ecore*np.eye(4)
        add('start',scalar,'end',scalar_coeff(local))
        for charge,q in spinors.items():
            opening,closing=endpoints[charge]
            add('start',q,(site,charge),opening)
            for origin in range(site):
                label=(origin,charge)
                add(label,scalar,label,parity_coeff)
                add(label,spinors[-charge],'end',h1[origin,site]*closing)
        if not blocks:
            blocks[(scalar,scalar,scalar)]=np.zeros((counts_l[scalar],mult[scalar],counts_r[scalar]))
        sites.append(NonabelianTensor(data=blocks,qns=[[q for q,_ in left],
            [q for q,r in mult.items() for _ in range(r)],[q for q,_ in right]],dirs=[-1,1,1],
            metadata={'physical_basis':'fully_reduced_su2','operator_construction':'sparse_hubbard'}))
    return SpinTensorMPO(tuple(sites),basis,op_basis,channels,transform,0.)


class HubbardComponentReference:
    """Lazy magnetic MPO core for explicit reference/blocked-basis consumers.

    Reduced sweeps use only its shape and the cached reduced MPO. No component
    arrays are allocated unless the consumer explicitly requests as_dense().
    """
    def __init__(self, mpo, site):
        self.mpo, self.site = mpo, site
        core = mpo.sites[site]
        left = sum(_sector_irrep(q).dim for q in core.qns[0])
        right = sum(_sector_irrep(q).dim for q in core.qns[2])
        self.component_shape = (left, right, mpo.physical_basis.dense_dim, mpo.physical_basis.dense_dim)

    def as_dense(self):
        from .reduced_contraction import expand_reduced_mps_site
        core = expand_reduced_mps_site(self.mpo.sites[self.site])
        d = self.mpo.physical_basis.dense_dim
        return np.einsum('apb,oip->aboi', core, self.mpo.local_transform.reshape(d,d,-1))
