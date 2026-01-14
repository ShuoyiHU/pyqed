#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 09:04:09 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.qchem import UCISD, Molecule, FCI

mol = Molecule(atom = [
    ['O', ( 0., 0.    , 0.   )],
    ['H', ( 0., -0.757, 0.587)],
    ['H', ( 0., 0.757 , 0.587)],])
mol.basis = 'sto3g'

mol.build()

mf = mol.RHF().run()

fci = FCI(mf).run()


####
myci = UCISD(mf)
myci.run()

print(myci.e_tot)


from pyscf import gto, scf
from pyscf.ci import CISD

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ['O', ( 0., 0.    , 0.   )],
    ['H', ( 0., -0.757, 0.587)],
    ['H', ( 0., 0.757 , 0.587)],]
mol.basis = 'sto3g'
mol.unit = 'b'
mol.build()
mf = scf.RHF(mol).run()
print(mf.e_tot)
myci = CISD(mf)

# eris = ccsd._make_eris_outcore(myci, mf.mo_coeff)
ecisd, civec = myci.kernel()
print(ecisd) 
# -0.048878084082066106)