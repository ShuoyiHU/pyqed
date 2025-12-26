from pyscf import scf
from pyscf import gto
from pyscf import gto, scf, ao2mo, fci
import numpy as np

# # mol_h4 = gto.M(atom='H 0.495 0 0; H -0.495 0 0; H 0 0 0.495; H 0 0 -0.495', spin=0, basis="ccpvdz", unit="B") # (n+2 alpha, n beta) electrons
# mol_h4 = gto.M(atom='H 0.7 0.7 0; H -0.7 0.7 0; H 0.7 -0.7 0; H -0.7 -0.7 0', spin=0, basis="sto3g", unit="B") # (n+2 alpha, n beta) electrons
# # mol_h4 = gto.M(atom='H 0.8 0.8 0; H -0.8 0.8 0; H 0.8 -0.8 0; H -0.8 -0.8 0', spin=0, basis="sto6g", unit="B") # (n+2 alpha, n beta) electrons
# mol_h4 = gto.M(atom='H 0 0 3.6; H 0 0 0.91; H 0 0 -3.6; H 0 0 -0.91', spin=0, basis="ccpvqz", unit="B") # (n+2 alpha, n beta) electrons
# mol_h4 = gto.M(atom=
#             #    'H 0 0 11; ' \
#                'H 0 0 9; ' \
#                'H 0 0 7; ' \
#                'H 0 0 5; ' \
#                'H 0 0 3; ' \
#                'H 0 0 1; ' \
#             #    'H 0 0 -11; ' \
#                'H 0 0 -9; ' \
#                'H 0 0 -7; ' \
#                'H 0 0 -5; ' \
#                'H 0 0 -3; ' \
#                'H 0 0 -1; ', spin=0, basis="sto6g", unit="B") # (n+2 alpha, n beta) electrons
mol_h4 = gto.M(atom=
            #    'H 0 0.7 0.7; ' \
            #    'H 0 0.7 -0.7; ' \
            #    'H 0 -0.7 0.7; ' \
            #    'H 0 -0.7 -0.7; ', 

               # 'H 0 1 1; ' \
               # 'H 0 -1 -1; ' ,

            #    'H 0 1 1; ' \
            #    'H 0 1 -1; ' \
            #    'H 0 -1 1; ' \
            #    'H 0 -1 -1; ', 


               # 'H 0 0 0.2; ' \
               # 'H 0 0 -0.2; ' ,
               
               'H 0 0 1.889; ' \
               'H 0 0 -1.889; ' \
               'H 0 0 3.6; ' \
               'H 0 0 -3.6; ' ,

               # 'H 0 -0 0.91; ' \
               # 'H 0 -0 -0.91; ' \
               # 'H 0 0 3.6; ' \
               # 'H 0 0 -3.6; ' ,

               # 'H 0 0 0.7; ' \
               # 'H 0 0 -0.7; ' ,
               spin=0, basis="631g", unit="B") # (n+2 alpha, n beta) electrons
# mol_h4 = gto.M(atom='H 0 0 -0.7; H 0 0 0.7', spin=0, basis="ccpvqz", unit="B") # (n+2 alpha, n beta) electrons
# mol_h4 = gto.M(atom='be 0 0 0', spin=0, basis="ccpvdz", unit="B") # (n+2 alpha, n beta) electrons


# mol_h4 = gto.M(atom=
#             'H 0 0 -19; H 0 0 -17; H 0 0 -15; H 0 0 -13; H 0 0 -11; H 0 0 -9; H 0 0 -7; H 0 0 -5; H 0 0 -3; H 0 0 -1; H 0 0 1; H 0 0 3; H 0 0 5; H 0 0 7; H 0 0 9; H 0 0 11; H 0 0 13; H 0 0 15; H 0 0 17; H 0 0 19;', 
#             spin=0, basis="sto6g", unit="B")
# zs = np.linspace(-49,49,20)
# zs = np.linspace(-49,49, 20)
# # # # zs = np.linspace(-249,249, 100)
# # # zs = np.linspace(-49,49, 50)
# # # # # # print(zs)
# atom100 = '; '.join(f'H 0 0 {z:.6g}' for z in zs) + '; '

# mol_h4 = gto.M(atom=atom100, spin=0, basis="sto6g", unit="B")
# mol_h4 = gto.M(atom=atom100, spin=0, basis="631g", unit="B")
# mol_h4 = gto.M(atom=atom100, spin=0, basis="ccpvdz", unit="B")

# #cubic H8
# side = 4  # Adjustable side length of the cube in atomic units
# charges = [1.0] * 8  # 8 hydrogen atoms
# coords = [
#    [-side / 2, -side / 2, -side / 2],
#    [ side / 2, -side / 2, -side / 2],
#    [-side / 2,  side / 2, -side / 2],
#    [ side / 2,  side / 2, -side / 2],
#    [-side / 2, -side / 2,  side / 2],
#    [ side / 2, -side / 2,  side / 2],
#    [-side / 2,  side / 2,  side / 2],
#    [ side / 2,  side / 2,  side / 2]
# ]
# atom_str = '; '.join(f'H {x:.6g} {y:.6g} {z:.6g}' for x, y, z in coords)





# mol_h4 = gto.M(atom=atom100, spin=0, basis="631g**", unit="B")
# mol_h4 = gto.M(atom=atom100, spin=0, basis="sto6g", unit="B")
# mol_h4 = gto.M(atom=atom100, spin=0, basis="ccpvqz", unit="B")
# mol_h4 = gto.M(atom=atom100, spin=0, basis="631g**", unit="B")

rhf_h2o = scf.RHF(mol_h4)
e_h2o = rhf_h2o.kernel()

# # uhf_o2 = scf.UHF(mol_o2)
# # uhf_o2.kernel()
# # rohf_o2 = scf.ROHF(mol_o2)
# # rohf_o2.kernel()





# # Build a square H4 (side = 1.4 a0) in the xy-plane, centered at origin
# side = 1.4
# R_vals = np.linspace(0.5, 4.0, 15)
# energies_rhf = []
# energies_fci = []
# for R in R_vals:
#     print("current R", R)
#     coords = [
#         # (0.0, 0.0, -9.0),
#         # (0.0, 0.0, 9.0),
#         # (0.0, 0.0, -7.0),
#         # (0.0, 0.0, 7.0),
#         # (0.0, 0.0, -5.0),
#         # (0.0, 0.0, 5.0),
#         # (0.0, 0.0, -3.0),
#         # (0.0, 0.0, 3.0),
#         # (0.0, 0.0, -1.0),
#         # (0.0, 0.0, 1.0),
#         # (0.0, 0.0, -3.6),
#         # (0.0, 0.0, 3.6),
#         # (0.0, 0.0, -0.91),
#         # (0.0, 0.0, 0.91),
#         (0.0, 0.0, -R/2),
#         (0.0, 0.0, R/2),
#         # ( side/2,  side/2, 0.0),
#         # (-side/2,  side/2, 0.0),
#         # (-side/2, -side/2, 0.0),
#         # ( side/2, -side/2, 0.0),
#     ]
#     atom_str = '; '.join(f'H {x} {y} {z}' for x, y, z in coords)

#     # Use a small basis so FCI is affordable
#     mol = gto.M(atom=atom_str, unit='B', basis='631g**', spin=0)
#     mf = scf.RHF(mol).run()

#     norb = mf.mo_coeff.shape[1]
#     nelec = mol.nelectron

#     # One- and two-electron integrals in MO basis
#     h1_ao = mf.get_hcore()
#     h1_mo = mf.mo_coeff.T.dot(h1_ao).dot(mf.mo_coeff)
#     eri_mo_packed = ao2mo.full(mol, mf.mo_coeff)
#     eri_mo = ao2mo.restore(1, eri_mo_packed, norb)

#     # Full CI (FCI)
#     e_fci, ci_vector = fci.direct_spin0.kernel(h1_mo, eri_mo, norb, nelec)

#     print("RHF total energy:     ", mf.e_tot)
#     print("FCI electronic energy:", e_fci)
#     print("FCI total energy:     ", e_fci + mol.energy_nuc())
#     energies_rhf.append(mf.e_tot)
#     energies_fci.append(e_fci + mol.energy_nuc())


# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.plot(R_vals, energies_rhf, 'o--', label='rhf')
# plt.plot(R_vals, energies_fci, 's-', label='fci')

# # Reference H2 limit (approx -1.174 Eh at equilibrium)
# plt.axhline(-1.0, color='k', linestyle=':', label='H + H limit (-1.0)')

# plt.title(r"$H_2$ Dissociation Curve (Hybrid Gauss-DVR)")
# plt.xlabel("Bond Length R (Bohr)")
# plt.ylabel("Total Energy (Hartree)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Define the HeH+ molecule
# mol_heh = gto.M(atom='He 0 0 0; H 0 0 1.4632', charge=1, spin=0, basis='augccpvdz', unit='B')

# # Perform RHF calculation
# rhf_heh = scf.RHF(mol_heh)
# e_heh = rhf_heh.kernel()

# print("RHF total energy for HeH+: ", e_heh)