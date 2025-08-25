#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:54:28 2025

Exact diagonalization with PySCF input

@author: Bing Gu (gubing@westlake.edu.cn)
"""


from pyscf import gto, scf, dft, tddft, ao2mo
import pyscf
from pyqed.qchem import get_hcore_mo, get_eri_mo
# from pyqed.qchem.gto.rhf import RHF
from pyqed.qchem.mol import atomic_chain
import numpy as np 
from pyqed.mps.fermion import SpinHalfFermionChain
from opt_einsum import contract 

# mol = gto.Mole()
# mol.atom = [
#     ['H' , (0. , 0. , .917)],
#     ['H' , (0. , 0. , 0.)], ]
# mol.basis = '6311g'
# mol.build()

natom = 4
z = np.linspace(-4, 4, natom)
mol = atomic_chain(natom, z)
mol.basis = 'sto6g'
mol = mol.topyscf()
# mol.build()
mf = scf.RHF(mol).run()





def benchmark(mf):
    e, fcivec = pyscf.fci.FCI(mf).kernel(verbose=4)
    print(e)

Ca = mf.mo_coeff

n = Ca.shape[-1]
ne = mol.nelectron



print('number of electrons', ne)
print('number of orbs = ', n)


# extract hcore and eri in MO repr
h1e = mf.get_hcore()
h1e = contract('ia, ij, jb -> ab', Ca.conj(), h1e, Ca)

eri = mol.intor('int2e', aosym=8)

eri = ao2mo.incore.full(eri, Ca, compact=False).reshape(n,n,n,n)



model = SpinHalfFermionChain(h1e, eri, [ne//2, ne//2])
model.run()
print(model.e_tot + mf.energy_nuc())
print(model.X.shape)

