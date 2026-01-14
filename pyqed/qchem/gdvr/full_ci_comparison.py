from pyscf import gto, scf, fci


mol = gto.M(
    atom='H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6; H 0 0 8; H 0 0 10; H 0 0 12; H 0 0 14;', # H8 
    basis='631g', 
    unit='Bohr'
)

# Run RHF
mf = scf.RHF(mol).run()

# Run FCI
# For very small systems, this works:
e_fci, c_fci = fci.FCI(mf).kernel()
print(f"FCI Energy: {e_fci}")

