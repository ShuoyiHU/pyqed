#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:37:25 2025

@author: bingg
"""

#!/usr/bin/env python
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci, mcscf
from pyscf import tools
from pyscf import symm

dist_range = numpy.arange(1.2,2.6,0.1)

energy_data = numpy.zeros((len(dist_range),3))
me_data = numpy.zeros((len(dist_range),5))
# for n, dist in enumerate(dist_range):

mol = gto.M(atom = [['Li', 0.0,  0.0,  0.0], ['H', 0.0,  0.0,  1]],
    basis = '631g',
    verbose = 0,
    unit = 'b',
    # symmetry = 1,
#    symmetry_subgroup = 'D2h',
)
mf = scf.RHF(mol)
mf.kernel()

C = mc_mo = mf.mo_coeff
#
# 3. Exited states.  In this example, B2u are bright states.
#
# Here, mc.ci[0] is the first excited state.

ncas, nelecas = 6,4 
mc = mcscf.CASCI(mf, 6, 4)
mc.fcisolver = fci.direct_spin0.FCI(mol)
mc.fcisolver.nroots = 4
mc.kernel(mc_mo)

print(mc.e_tot)

# energy_data[n,0] = dist
# energy_data[n,1:] = mc.e_tot[:2]
# print
# print "Energies:", mc.e_tot[:2]
# print
sys.stdout.flush()


D = mc.fcisolver.make_rdm1(mc.ci[0], norb=ncas, nelec=nelecas)

print(D)

#
# 4. transition density matrix and transition dipole
#
# Be careful with the gauge origin of the dipole integrals
#
charges = mol.atom_charges()
coords = mol.atom_coords()
nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
mol.set_common_orig_(nuc_charge_center)
dip_ints = mol.intor('cint1e_r_sph', comp=3)

# def makedip(ci_id1, ci_id2):
    # transform density matrix in MO representation
print(mc.ncas, mc.nelecas, mc.ci[0].shape)
t_dm1 = mc.fcisolver.trans_rdm1(mc.ci[1], mc.ci[0], mc.ncas, mc.nelecas)

print(t_dm1)


C = C[:,mc.ncore:mc.ncore+mc.ncas]
mu = mol.intor('int1e_r', comp=3)
mu = [C.T @ d @ C for d in mu]
print(mu[1].shape)

for i in range(3):
    print(numpy.trace(t_dm1 @ mu[i].T))