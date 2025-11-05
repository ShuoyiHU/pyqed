#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:25:21 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np

from pyqed import Molecule
from pyqed.qchem.hf.rhf import ao2mo
from pyqed.qchem.ci.cisd import overlap
from pyqed.qchem.ci import CASCI, spin_square

mol = Molecule(atom = [
['H' , (0. , 0. , 0)],
['Li' , (0. , 0. , 1)], ])

mol.basis = '631g'
mol.charge = 0

# mol.molecular_frame()
print(mol.atom_coords())

nstates = 5
# Rs = np.linspace(1,4,4)
# E = np.zeros((nstates, len(Rs)))



mol.build()

mf = mol.RHF()
mf.run()

ncas, nelecas = (4, 2)
casci = CASCI(mf, ncas, nelecas)

print('ncore = ', casci.ncore)

casci.run(nstates)
C = mf.mo_coeff

# print(C.shape)

# transition density matrix
# h1e = np.diag(mf.mo_occ//2)
# h1e = np.eye(mf.nmo)

# r = mol.moment_integral()


# h1e = ao2mo(h1e, C)

# print(h1e)

####################
# transition density matrix

# h1e = r[:,:,2]
# D = casci.make_tdm1(2, 0, h1e=h1e)

state_id = 2
dm1 = casci.make_rdm1(state_id)

print(dm1.shape)

# print(D)
# print(C @ D @ C.T)

# e = casci.contract_with_rdm2(casci.eri_so, 2)

# print(e)

dm2 = casci.make_rdm2(state_id)
print(dm2.shape)

print(dm2[:, :, 1, 1])
print(dm2[:, :, 3, 2])
# # print('pqqs -> ps')

# # eri = mf.get_eri('mo')

# print(casci.eri_so[0,1].shape)

# e = np.einsum('pqrs, pqrs ->', dm2, casci.eri_so[0,1])/2
# print(e)

# # e = np.einsum('pqrs, pqrs ->', D, eri)

# print(e)


# #################
print('\n-------------- PYSCF -------------\n')

from pyscf import mcscf, gto, fci, ao2mo
from pyscf.mcscf.casci import CASCI

mol = mol.topyscf()


mf = mol.RHF()
mf.kernel()
C = mf.mo_coeff

# ncas=4
# nelecas=2

mc = CASCI(mf, ncas, nelecas)
mc.fcisolver = fci.direct_spin0.FCI(mol)
mc.fcisolver.nroots = 3

mc.run()

dm1 = mc.fcisolver.make_rdm1(mc.ci[1], norb=ncas, nelec=nelecas)
print(dm1.shape)

D = mc.fcisolver.make_rdm2(mc.ci[1], norb=ncas, nelec=nelecas)
print(D.shape)

print(D[:, :, 1, 1])
print(D[:, :, 3, 2])
# eri = mol.intor('int2e', aosym='s1')

# eri_mo = ao2mo.full(eri, C)


# e = np.einsum('pqrs, pqrs ->', D, eri_mo)/2
# print(e)

# # print(mol.atom_coords('b'))
# mu = mol.intor('int1e_r')

# print(mu[2])

# D = mc.make_rdm1()

# print(C.shape, D.shape)
# print(C.T @ D @ C.conj())
# print(mc.nelecas)
# print(mc.ncas, mc.ci[0].shape)

# # transform density matrix in MO representation
# D = mc.fcisolver.trans_rdm1(mc.ci[1], mc.ci[0], mc.ncas, mc.nelecas)

# print(D.shape)

# for i in range(3):
#     print(np.trace(D @ mu[i].T))