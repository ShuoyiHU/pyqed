#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 23:40:10 2024


Exact diagonalization for quantum chemistry

@author: bingg
"""

from pyqed import RHF, FCI, Molecule
from pyqed.mps.fermion import SpinHalfFermionChain
import pyqed

atom= 'H 0, 0, -3.6; \
    H 0, 0, -1.2; \
    H 0, 0, 1.2; \
    H 0, 0, 3.6'

mol = Molecule(atom, basis='sto6g')


mol.build()


#print(mol.eri.shape)
mf = mol.RHF().run()
# print(mf.mo_occ)

# FCI(mf).run(3)

h1e = mf.get_hcore_mo()
eri = mf.get_eri_mo()


model = SpinHalfFermionChain(h1e, eri, [mol.nelec//2, mol.nelec//2])
model.run()

# model.make_rdm1()
# model = SpinHalfFermionChain(h1e, eri)