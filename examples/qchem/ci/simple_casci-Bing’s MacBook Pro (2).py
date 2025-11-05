#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:25:21 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import numpy as np

from pyqed import Molecule 
from pyqed.qchem import ao2mo
from pyqed.qchem.ci.cisd import overlap
from pyqed.qchem.ci import CASCI

mol = Molecule(atom = [
['H' , (0. , 0. , 0)],
['H' , (0. , 0. , 1)], ])

mol.basis = '631g'
mol.charge = 0

# mol.molecular_frame()
print(mol.atom_coords())

nstates = 2
# Rs = np.linspace(1,4,4)
# E = np.zeros((nstates, len(Rs)))



mol.build()

mf = mol.RHF()
mf.run()

C = mf.mo_coeff


ncas, nelecas = (4,2)
casci = CASCI(mf, ncas, nelecas)



casci.run(nstates)


# transition density matrix
h1e = np.diag(mf.mo_occ//2)
h1e = np.eye(mf.nmo)

r = mol.moment_integral()
h1e = r[:,:,2]

h1e = mf.ao2mo(h1e, C)

# print(h1e)

D = casci.make_rdm1(1, h1e=[h1e, h1e])
print(D)


#################
##pyscf

from pyscf.mcscf.casci import CASCI
mol = mol.topyscf()
mf = mol.RHF().run()
myci = CASCI(mf, ncas=4, nelecas=2).run(nstates=1)

print(mol.atom_coords('b'))
mu = mol.intor('int1e_r')

# print(mu[2])

D = myci.make_rdm1()
print(D)

print(np.trace(D @ mu[2].T))