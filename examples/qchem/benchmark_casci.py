#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySCF vs PyQED CASCI(2,2) 3-state energy comparison
"""

import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyqed.qchem.mol import Molecule
from pyqed.qchem.ci import CASCI
from pyqed import au2angstrom

coords = np.array([
    [0.00000000, 0.00000000, 0.66796400],
    [0.92288300, 0.00000000, 1.24294900],
    [-0.92288300, 0.00000000, 1.24294900],
    [0.00000000, 0.00000000, -0.66796400],
    [0.54030916, 0.92288300, -0.86462045],
    [0.54030916, -0.92288300, -0.86462045],
])
atoms = ["C", "H", "H", "C", "H", "H"]

# print("=== Geometry (Angstrom) ===")
# for s, (x, y, z) in zip(atoms, coords):
#     print(f"{s:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
# print()

# print("===== PySCF CASCI(2,2) =====")

mol = gto.M(
    atom=[(a, c) for a, c in zip(atoms, coords)],
    basis="6-31g",
    unit="Angstrom",
    spin=0,
    verbose=1,
)

mf = scf.RHF(mol).run()
print(mf.e_tot)
print(mol.nelectron)
print('mo energy = ', mf.mo_energy)
ncas = 2
nelecas = (1, 1)
n_states = 3
weights = np.ones(n_states)/n_states

mc = mcscf.CASSCF(mf, ncas, nelecas).state_average_(weights)
mc.fcisolver.nroots = n_states
mc.fix_spin_(ss=0)
mc.verbose = 4
mc.run()

print(mc.e_states)
print(mc.ci)
print(mc.mo_coeff[mol.nelectron//2]-mf.mo_coeff[mol.nelectron//2])

# print(mc.mo_coeff[mol.nelectron//2], mc.mo_coeff[mol.nelectron//2+1])
# print(mf.mo_coeff[mol.nelectron//2], mf.mo_coeff[mol.nelectron//2+1])

# print("===== PyQED CASCI(2,2) =====")

# coords = coords /au2angstrom
# atom_pyqed = [[a, *r] for a, r in zip(atoms, coords)]
# mol_qed = Molecule(atom_pyqed, basis="6-31g")
# mol_qed.build()

# mf_qed = mol_qed.RHF()
# mf_qed.run()
# dm0 = mf_qed.make_rdm1()

# casci = CASCI(mf_qed, ncas, sum(nelecas))

# casci.run(4)

# E_qed = casci.e_tot

# print(casci.ci)



# print("\nPyQED CASCI Energies (Hartree):")
# for i, e in enumerate(E_qed):
#     print(f"State {i}: {e:15.8f}")