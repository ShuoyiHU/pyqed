#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:48:18 2026

Quantum Chemitry DMRG

@author: Shuoyi Hu (hushuoyi@westlake.edu.cn)


"""


import numpy as np
import scipy.constants as const

from scipy.sparse.linalg import eigsh

import logging
import warnings

from pyqed import discretize, sort, dag, tensor
from pyqed.davidson import davidson

from pyqed import au2ev, au2angstrom

from pyqed.qchem.ci.fci import SpinOuterProduct, givenΛgetB
from pyqed.qchem.mcscf.casci import h1e_for_cas

from pyqed.qchem.jordan_wigner.spinful import SpinHalfFermionOperators

# from numba import vectorize, float64, jit
import time
from opt_einsum import contract

from collections import namedtuple
from scipy.sparse import identity, kron, csr_matrix, diags

from pyqed import Molecule
from pyqed.mps.mps import DMRG
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo


#  Fermionic Logic patch adding JW chain
def get_jw_term_robust(op_str_list, indices, factor):
    """
    Constructs a fermionic term with explicit Jordan-Wigner strings (sigma_z)
    and correct sign handling (parity).
    """
    # 1. Canonical Sort: Sort operators by site index
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
                swaps += 1

    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]

    final_indices = []
    final_ops_str = []
    parity = 0
    extra_sign = 1

    # 2. Insert sigma_z filling (Jordan-Wigner String)
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]

        # Fill gap between previous site and current site with Z
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")

        # 3. Handle Creation/Annihilation Phase
        # If we are applying 'a' and there are an odd number of operators to the right, flip sign
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1

        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1

    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)

# initial guess from hf but with added noise to prevenr stuck in hf product state, it happens sometimes
def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates an MPS guess based on filling the first N_elec spin-orbitals,
    but adds small noise to prevent the solver from getting stuck in the HF state.
    """
    d = 2
    mps_guess = []
    filled_count = 0

    for i in range(n_spin):
        vec = np.zeros((d, 1, 1))
        if filled_count < n_elec:
            vec[1, 0, 0] = 1.0; filled_count += 1
        else:
            vec[0, 0, 0] = 1.0

        # Add Noise
        vec += (np.random.rand(d, 1, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)

    return mps_guess




def graphic(sys_block, env_block, sys_label="l"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    assert sys_label in ("l", "r")
    graphic = ("=" * sys_block.length) + "**" + ("-" * env_block.length)
    if sys_label == "r":
        # The system should be on the right and the environment should be on
        # the left, so reverse the graphic.
        graphic = graphic[::-1]
    return graphic

# def infinite_system_algorithm(L, m):

#     initial_block = Block(length=1, basis_size=4, operator_dict={
#         "H": H1,
#         "Cu": ops['Cu'],
#         "Cd": ops['Cd'],
#         "Nu": ops['Nu'],
#         "Nd": ops['Nd']
#     })

#     block = initial_block
#     # Repeatedly enlarge the system by performing a single DMRG step, using a
#     # reflection of the current block as the environment.
#     while 2 * block.length < L:
#         print("L =", block.length * 2 + 2)
#         block, energy = single_dmrg_step(block, block, m=m)
#         print("E/L =", energy / (block.length * 2))



class QCDMRG:
    """
    ab initio DRMG quantum chemistry calculation
    """
    def __init__(self, mf, ncas, nelecas, D, init_guess='hf', m_warmup=None,\
                 spin=None, tol=1e-6):
        """
        DMRG sweeping algorithm directly using DVR set (without SCF calculations)

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        L : TYPE
            DESCRIPTION.
        D : TYPE, optional
            maximum bond dimension. The default is None.
        tol: float
            tolerance for energy convergence

        Returns
        -------
        None.

        """
        # assert(isinstance(mf, RHF1D))

        self.mf = mf

        self.d = 4 # local dimension for spacial orbital

        self.nsites = self.L = ncas

        # assert(mf.eri.shape == (self.L, self.L))


        self.D = self.m = D

        self.tol = tol # tolerance for energy convergence
        self.rigid_shift = 0

        if m_warmup is None:
            m_warmup = D
        self.m_warmup = m_warmup

        self.ncas = ncas # number of MOs in active space
        self.nelecas = nelecas

        ncore = mf.nelec//2 - self.nelecas//2 # core orbs
        assert(ncore >= 0)

        self.ncore = ncore

        if ncas > 20:
            warnings.warn('Active space with {} orbitals is probably too big.'.format(ncas))

        self.nstates = None
        # if nelecas is None:
        #     nelecas = mf.mol.nelec

        # if nelecas <= 2:
        #     print('Electrons < 2. Use CIS or CISD instead.')


        self.mo_core = None
        self.mo_cas = None

        if spin is None:
            spin = mf.mol.spin
        self.spin = spin
        self.shift = None
        self.ss = None

        self.mf = mf
        # self.chemical_potential = mu

        self.mol = mf.mol

        ###
        self.e_tot = None
        self.e_core = None # core energy
        self.ci = None # CI coefficients
        self.H = None


        self.hcore = self.h1e_cas = None # effective 1e CAS Hamiltonian including the influence of frozen orbitals
        self.eri_so = self.h2e_cas = None # spin-orbital ERI in the active space

        self.spin_purification = False

        # effective CAS Hamiltonian
        self.h1e = None
        self.h2e = None

        self.init_guess = init_guess

    def fix_nelec(self, shift):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{N} - N)^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    def fix_spin(self, shift, spin=0):
        """
        fix the number of electrons by energy penalty

        .. math::

            \mathcal{H} = H + \lambda (\hat{S}^2 - S(S+1))^2

        Parameters
        ----------
        shift : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # self.h1e += ...
        # self.eri += ...
        return

    def get_SO_matrix(self, spin_flip=False, H1=None, H2=None):
        """
        Given a rhf object get Spin-Orbit Matrices

        SF: bool
            spin-flip

        Returns
        -------
        H1: list of [h1e_a, h1e_b]
        H2: list of ERIs [[ERI_aa, ERI_ab], [ERI_ba, ERI_bb]]
        """
        # from pyscf import ao2mo

        mf = self.mf

        # molecular orbitals
        Ca, Cb = [self.mo_cas, ] * 2

        H, energy_core = h1e_for_cas(mf, ncas=self.ncas, ncore=self.ncore, \
                                     mo_coeff=self.mo_coeff)

        self.e_core = energy_core


        # S = (uhf_pyscf.mol).intor("int1e_ovlp")
        # eig, v = np.linalg.eigh(S)
        # A = (v) @ np.diag(eig**(-0.5)) @ np.linalg.inv(v)

        # H1e in AO
        # H = mf.get_hcore()
        # H = dag(Ca) @ H @ Ca

        # nmo = Ca.shape[1] # n

        eri = mf.eri  # (pq||rs) 1^* 1 2^* 2

        ### compute SO ERIs (MO)
        eri_aa = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Ca.conj(), Ca)

        # physicts notation <pq|rs>
        # eri_aa = contract('ip, jq, ij, ir, js -> pqrs', Ca.conj(), Ca.conj(), eri, Ca, Ca)

        # eri_aa -= eri_aa.swapaxes(1,3)

        eri_bb = eri_aa.copy()

        eri_ab = contract('ip, jq, ijkl, kr, ls -> pqrs', Ca.conj(), Ca, eri, Cb.conj(), Cb)
        eri_ba = contract('ip, jq, ijkl, kr, ls -> pqrs', Cb.conj(), Cb, eri, Ca.conj(), Ca)




        # eri_aa = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Ca, Ca),
        #                         compact=False)).reshape((n,n,n,n), order="C")
        # eri_aa -= eri_aa.swapaxes(1,3)

        # eri_bb = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # eri_bb -= eri_bb.swapaxes(1,3)

        # eri_ab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Ca, Cb, Cb),
        # compact=False)).reshape((n,n,n,n), order="C")
        # #eri_ba = (1.*eri_ab).swapaxes(0,3).swapaxes(1,2) ## !! caution depends on symmetry

        # eri_ba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Cb, Ca, Ca),
        # compact=False)).reshape((n,n,n,n), order="C")

        H2 = np.stack(( np.stack((eri_aa, eri_ab)), np.stack((eri_ba, eri_bb)) ))

        # H1 = np.asarray([np.einsum("AB, Ap, Bq -> pq", H, Ca, Ca),
                         # np.einsum("AB, Ap, Bq -> pq", H, Cb, Cb)])
        H1 = [H, H]

        if spin_flip:
            raise NotImplementedError('Spin-flip matrix elements not implemented yet')
        #     eri_abab = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_abba = (ao2mo.general( (uhf_pyscf)._eri , (Ca, Cb, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baab = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Ca, Cb),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     eri_baba = (ao2mo.general( (uhf_pyscf)._eri , (Cb, Ca, Cb, Ca),
        #     compact=False)).reshape((n,n,n,n), order="C")
        #     H2_SF = np.stack(( np.stack((eri_abab, eri_abba)), np.stack((eri_baab, eri_baba)) ))
        #     return H1, H2, H2_SF
        # else:
        #     return H1, H2
        return H1, H2

    def build(self):

        # 1. Extract Integrals & dims
        # mol = mf.mol
        mf = self.mf
        if self.ncore == 0:
            h1 = mf.get_hcore_mo()
            eri = mf.get_eri_mo(notation='chem') # (pq|rs)
        else:
            h1e, eri = self.get_SO_matrix()


        n_spatial = self.ncas

        nso = 2 * n_spatial
        print(f"  System: {n_spatial} spatial orbitals, {nso} spin-orbitals")

        # 2. Build Hamiltonian (Using Robust JW Builder)
        print("  Building Hamiltonian MPO...")
        ham_terms = []
        cutoff = 1e-10

        # --- One-Body Terms: h_pq a+_p a_q ---
        for p in range(n_spatial):
            for q in range(n_spatial):
                val = h1[p, q]
                if abs(val) > cutoff:
                    # Spin Up (Indices 2p, 2q)
                    ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))
                    # Spin Down (Indices 2p+1, 2q+1)
                    ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))

        # --- Two-Body Terms: 0.5 * (pq|rs) a+_p a+_r a_s a_q ---
        for p in range(n_spatial):
            for q in range(n_spatial):
                for r in range(n_spatial):
                    for s in range(n_spatial):
                        val = 0.5 * eri[p, q, r, s]
                        if abs(val) < cutoff: continue

                        # p,r creation; s,q annihilation

                        # Same Spin (Pauli Exclusion p!=r)
                        if p != r and s != q:
                            # Up-Up
                            ham_terms.append(get_jw_term_robust(
                                [r"a^\dagger", r"a^\dagger", "a", "a"],
                                [2*p, 2*r, 2*s, 2*q], val
                            ))
                            # Dn-Dn
                            ham_terms.append(get_jw_term_robust(
                                [r"a^\dagger", r"a^\dagger", "a", "a"],
                                [2*p+1, 2*r+1, 2*s+1, 2*q+1], val
                            ))

                        # Mixed Spin (No Pauli restriction on spatial indices)
                        # Up-Dn (p Up, r Dn, s Dn, q Up)
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p, 2*r+1, 2*s+1, 2*q], val
                        ))
                        # Dn-Up (p Dn, r Up, s Up, q Dn)
                        ham_terms.append(get_jw_term_robust(
                            [r"a^\dagger", r"a^\dagger", "a", "a"],
                            [2*p+1, 2*r, 2*s, 2*q+1], val
                        ))

        # 3. Generate MPO
        basis_sites = [BasisSimpleElectron(i) for i in range(nso)]
        model = Model(basis=basis_sites, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")

        # get it transposed for solver in PyQED: (L, R, P, P) -> (L, P, R, P)
        H = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        self.H = H

        return self

    def run(self):
        # if self.init_guess is None:
        #     logging.info('Building initial guess by iDMRG')
        #     # iDMRG

        # DMRG Parameters
        N_SWEEPS = 20
        Initial_guess_NOISE    = 1e-4

        # get mpo and mps initial guess
        # mpo_dmrg = qc_dmrg_mpo(mf)
        if self.init_guess == 'hf':
            mps0 = get_noisy_hf_guess(mol.nelec, 2*self.ncas, noise=Initial_guess_NOISE)


        t0 = time.time()

        # run dmrg!
        print(f"  Starting Sweeps (D={self.D})...")
        dmrg = DMRG(self.H, D=self.D, nsweeps=N_SWEEPS, init_guess=mps0)
        dmrg.run()

        # 6. Report result
        e_dmrg_total = dmrg.e_tot + self.mf.energy_nuc()


        print(f"  RHF Energy:         {mf.e_tot:.8f} Ha")
        print(f"  E(DMRG) =  {e_dmrg_total:.8f} Ha")
        print(f"  Correlation Energy = {e_dmrg_total - mf.e_tot:.8f} Ha")
        print(f"  Time:               {time.time()-t0:.2f} s")

        rdm1_list = dmrg.make_rdm()
        nelec_dmrg = sum(rho[1, 1] for rho in rdm1_list.values())
        print(f"  <N> (from 1-RDM):   {nelec_dmrg.real:.6f}")
        return dmrg


    def dump(self):
        pass




class DMRGSCF(QCDMRG):
    """
    optimize the orbitals
    """
    pass


if __name__=='__main__':

    from pyqed.qchem.mcscf.direct_ci import CASCI

    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)


    mol = Molecule(atom = [
        ['H' , (0. , 0. , 0)],
        ['Li' , (0. , 0. , 4)]])
    mol.basis = '631g'
    mol.build(driver='pyscf')

    mf = mol.RHF().run()

    mc = CASCI(mf, ncas=8, nelecas=4)
    mc.run()
    rdm1_cas = mc.make_rdm1(state_id=0, with_core=True)

    # The trace of the 1-RDM is the total number of electrons
    nelec_cas = np.trace(rdm1_cas)

    print(f"  <N> (from CASCI 1-RDM): {nelec_cas:.6f}") 



    dmrg = QCDMRG(mf, ncas=8, nelecas=4, D=20)
    dmrg.build().run()

    # conn refers to the connection operator, that is, the operator on the edge of
    # the block, on the interior of the chain.  We need to be able to represent S^z
    # and S^+ on that site in the current basis in order to grow the chain.
    # initial_block = Block(length=1, basis_size=model_d, operator_dict={
    #     "H": H1,
    #     "Cu": ops['Cu'],
    #     "Cd": ops['Cd'],
    #     "Nu": ops['Nu'],
    #     "Nd": ops['Nd']
    # })

    #infinite_system_algorithm(L=100, m=20)
    # finite_system_algorithm(L=nsites, m_warmup=10, m=10)