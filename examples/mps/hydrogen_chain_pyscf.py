#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 16:02:43 2025

ab inito H chain calculation 

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.qchem.mol import atomic_chain
from pyqed import interval, discrete_cosine_transform_matrix, au2angstrom
import numpy as np

from pyscf.mcscf.casci import CASCI

natom = 8
ndim = natom - 2 # total nuclear dofs
l = 10/au2angstrom

z0 = np.linspace(-1, 1, natom, endpoint=True) * l/2
print(z0 * au2angstrom)

print('interatomic distance = ', (z0[1] - z0[0])*au2angstrom)


# print('number of electrons = ', mol.nelec)
###############################################################################

# elements = ['H'] * natom
T = discrete_cosine_transform_matrix(natom - 2) # remove the boundary atoms

# print(T)


N = 6
ds = np.linspace(-2, 2, N)


e_hf = np.zeros(N)
nstates = 3
e_casci = np.zeros((N, nstates))

i = 2
# print(T[:,i])

for n in range(N):

    q = np.zeros(natom-2) # collective coordinates
    q[i] = ds[n]


    z = z0.copy()
    z[1:natom-1] += T.T @ q

    print('geometry', z)

    mol = atomic_chain(natom, z)


    # HF
    mf = mol.topyscf().RHF()
    mf.build()
    mf.run()

    
    e_hf[n] = mf.e_tot

    # CASCI
    print(mf.mo_coeff.shape)

    mc = CASCI(mf, ncas=6, nelecas=6)
    mc.fcisolver.nroots = nstates
    mc.run()

    e_casci[n] = mc.e_tot



import ultraplot as plt
fig, ax = plt.subplots()

# ax.plot(ds, e_hf,'-o', label='HF')

for n in range(3):
    ax.plot(ds, e_casci[:,n],'-s', label='CASCI')
ax.legend()

# h1e = mf.hcore
# eri = mf.eri

# L = nsites = mf.nx
