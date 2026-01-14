# import numpy as np
# import scipy.linalg
# import logging
# from pyscf import gto, scf, ao2mo, fci
# from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
# from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
# from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
# from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo
# from pyqed.mps.mps import DMRG


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)


# # PATCH for BasisSimpleElectron (Required for sigma_z)

# def patched_op_mat(self, op):
#     if not isinstance(op, Op):
#         op = Op(op, None)
#     op_symbol, op_factor = op.symbol, op.factor

#     mat = np.zeros((2, 2))
#     if op_symbol == r"a^\dagger":
#         mat[1, 0] = 1.
#     elif op_symbol == "a":
#         mat[0, 1] = 1.
#     elif op_symbol == r"a^\dagger a" or op_symbol == "n":
#         mat[1, 1] = 1.
#     elif op_symbol == "I":
#         mat = np.eye(2)
#     elif op_symbol == "sigma_z": 
#         # Z = |0><0| - |1><1|
#         mat[0, 0] = 1.
#         mat[1, 1] = -1.
#     else:
#         raise ValueError(f"op_symbol:{op_symbol} is not supported")
#     return mat * op_factor

# BasisSimpleElectron.op_mat = patched_op_mat


# # jordan-winger logic
# def get_jw_term_robust(op_str_list, indices, factor):
#     """
#     Constructs a valid Jordan-Wigner string for arbitrary N-body terms.
#     Handles: Normal Ordering, Parity-dependent Z-strings, and Commutation signs.
#     """
#     # 1. Sort indices (Normal Ordering)
#     chain = list(zip(indices, op_str_list))
#     n = len(chain)
#     swaps = 0
#     for i in range(n):
#         for j in range(0, n-i-1):
#             if chain[j][0] > chain[j+1][0]:
#                 chain[j], chain[j+1] = chain[j+1], chain[j]
#                 swaps += 1
    
#     sorted_indices = [x[0] for x in chain]
#     sorted_ops = [x[1] for x in chain]
    
#     final_indices = []
#     final_ops_str = []
    
#     # 2. Iterate to build string
#     # We maintain 'parity' to decide if we insert Zs in gaps.
#     parity = 0 
    
#     extra_sign = 1
    
#     for k in range(n):
#         site = sorted_indices[k]
#         op_sym = sorted_ops[k]
        
#         # --- A. GAP HANDLING ---
#         if k > 0:
#             prev_site = sorted_indices[k-1]
#             # If parity is ODD, we must fill the gap with Zs
#             if parity % 2 == 1:
#                 for z_site in range(prev_site + 1, site):
#                     final_indices.append(z_site)
#                     final_ops_str.append("sigma_z")
        
#         # --- B. LOCAL SIGN CORRECTION ---
#         # If an operator anti-commutes with Z (like 'a'), and there are an ODD 
#         # number of operators to the RIGHT, we pick up a minus sign.
#         # (Because the strings from the right act on us as Z)
        
#         ops_to_right = n - 1 - k
#         is_anticommuting = (op_sym == "a") # a^dagger commutes with Z (sigma+ Z = sigma+)
        
#         if is_anticommuting and (ops_to_right % 2 == 1):
#             extra_sign *= -1

#         # --- C. APPEND OPERATOR ---
#         final_indices.append(site)
#         final_ops_str.append(op_sym)
        
#         # Toggle parity for next gap
#         parity += 1
        
#     final_op_string = " ".join(final_ops_str)
    
#     # Total factor = Input * Swap Sign * Commutation Sign
#     final_factor = factor * ((-1) ** swaps) * extra_sign
    
#     return Op(final_op_string, final_indices, factor=final_factor)

# def generate_chem_terms(h1, h2, tol=1e-15):
#     terms = []
#     n_spatial = h1.shape[0]
    
#     # 1-Body
#     for p in range(n_spatial):
#         for q in range(n_spatial):
#             val = h1[p, q]
#             if abs(val) > tol:
#                 terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))
#                 terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))

#     # 2-Body
#     for p in range(n_spatial):
#         for q in range(n_spatial):
#             for r in range(n_spatial):
#                 for s in range(n_spatial):
#                     val = h2[p, q, r, s] 
#                     if abs(val) < tol: continue
#                     coef = 0.5 * val
#                     op_strs = [r"a^\dagger", r"a^\dagger", "a", "a"]
                    
#                     cases = []
#                     if p != r and q != s: # UpUp / DnDn
#                         cases.append([2*p, 2*r, 2*s, 2*q])
#                         cases.append([2*p+1, 2*r+1, 2*s+1, 2*q+1])
#                     # Mixed
#                     cases.append([2*p, 2*r+1, 2*s+1, 2*q])
#                     cases.append([2*p+1, 2*r, 2*s, 2*q+1])
                    
#                     for idx_list in cases:
#                         terms.append(get_jw_term_robust(op_strs, idx_list, coef))
#     return terms

# # main code for the test, with mpo from modified renormalized compared with full ci
# def run_exact_test():
#     # Setup
#     mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5; H 0 0 6; H 0 0 7;', unit='Bohr', basis='sto3g', verbose=0)
#     # mol = gto.M(atom='H 0 0 0; Li 0 0 1.4', unit='Bohr', basis='sto3g', verbose=0)
#     # mol = gto.M(atom='Li 0 0 0; Li 0 0 3', unit='Bohr', basis='sto3g', verbose=0)
#     mol.build()
#     mf = scf.RHF(mol)
#     mf.kernel()
    
#     h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
#     n_orb = h1.shape[0]
#     h2 = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n_orb,n_orb,n_orb,n_orb)
    
#     # Build
#     ham_terms = generate_chem_terms(h1, h2)
#     n_spin = 2 * n_orb
#     basis = [BasisSimpleElectron(i) for i in range(n_spin)]
#     mpo = Mpo(Model(basis=basis, ham_terms=ham_terms), algo="qr")
    
#     # 1. Shape Correction: Transpose MPO tensors for mps.py
#     # Change from (Left, Up, Down, Right) -> (Left, Right, Up, Down)
#     mpo_dmrg_format = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
    
#     # 2. Define Initial Guess (Hartree-Fock)
#     # MPS Tensors must be (Phys, Left, Right) -> (2, 1, 1)
#     d = 2
#     A_vac = np.zeros((d, 1, 1)); A_vac[0, 0, 0] = 1.0
#     A_occ = np.zeros((d, 1, 1)); A_occ[1, 0, 0] = 1.0
    
#     n_elec = mol.nelectron
#     mps_guess = [A_occ] * n_elec + [A_vac] * (n_spin - n_elec)
    
#     logger.info("Running DMRG with Transposed MPO...")

#     dmrg = DMRG(mpo_dmrg_format, D=20, nsweeps=8)
#     dmrg.init_guess = mps_guess
#     dmrg.run()
#     total_e_dmrg = dmrg.e_tot+mol.energy_nuc()
#     # Check by constructing dense matrix, large memory space taken, careful to run when dealing with large system.
#     # logger.info("Contracting...")
#     # # H_dense = mpo.to_dense(check_size=False)
#     # H_dense = mpo.to_dense_subspace(n_particles=8)
#     # e_mpo = scipy.linalg.eigvalsh(H_dense)[0] + mol.energy_nuc()
    
#     # Ref
#     my_fci = fci.FCI(mol, mf.mo_coeff)
#     e_fci, _ = my_fci.kernel()
    
#     logger.info("-" * 50)
#     logger.info(f"FCI Exact Energy: {e_fci:.12f}")
#     # logger.info(f"MPO Total Energy: {e_mpo:.12f}")
#     logger.info(f"DMRG Total Energy: {total_e_dmrg:.12f}")
#     logger.info("-" * 50)
    
#     # if abs(e_fci - e_mpo) < 1e-10:
#     #     logger.info("[SUCCESS] Perfect Match!")
#     # else:
#     #     logger.error(f"[FAIL] Diff: {abs(e_fci - e_mpo)}")

# if __name__ == "__main__":
#     run_exact_test()


import numpy as np
import scipy.linalg
import logging
from pyscf import gto, scf, ao2mo, fci

# --- Imports ---
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo


import pyqed.mps.mps as mps_lib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. CRITICAL PATCHES
# ==============================================================================

# [Patch A] Fix BasisSimpleElectron (Add sigma_z support)
def patched_op_mat(self, op):
    if not isinstance(op, Op):
        op = Op(op, None)
    op_symbol, op_factor = op.symbol, op.factor
    mat = np.zeros((2, 2))
    if op_symbol == r"a^\dagger": mat[1, 0] = 1.
    elif op_symbol == "a": mat[0, 1] = 1.
    elif op_symbol == r"a^\dagger a" or op_symbol == "n": mat[1, 1] = 1.
    elif op_symbol == "I": mat = np.eye(2)
    elif op_symbol == "sigma_z": mat[0, 0] = 1.; mat[1, 1] = -1.
    else: raise ValueError(f"op_symbol:{op_symbol} is not supported")
    return mat * op_factor
BasisSimpleElectron.op_mat = patched_op_mat

# [Patch B] Fix mps.py 'expect' bug (Library definition vs Call mismatch)
mps_lib.expect = mps_lib.expect_mps

# [Patch C] Fix Boundary Condition Mismatch (The cause of Sweep 0 Stop)
# MPO Gen puts Identity at [0]. mps.py expects it at [-1]. We force it to [0].
def patched_initial_F(W):
    F = np.zeros((W.shape[1], 1, 1))
    F[0] = 1.0  # <--- CHANGED FROM -1 TO 0
    return F
mps_lib.initial_F = patched_initial_F

# ==============================================================================
# 2. HELPER: NOISY INITIAL GUESS
# ==============================================================================
def get_noisy_hf_guess(n_elec, n_spin, noise=1e-2):
    """
    Creates a Hartree-Fock product state with small random noise
    to break symmetries and aid convergence.
    """
    d = 2
    mps_guess = []
    
    for i in range(n_spin):
        # 1. Create pure HF state for this site
        vec = np.zeros((d, 1, 1))
        if i < n_elec: # Occupied
            vec[1, 0, 0] = 1.0
        else: # Virtual
            vec[0, 0, 0] = 1.0
            
        # 2. Add random noise (Real-valued is fine for Hamiltonian)
        # We perturb the "other" state slightly
        perturbation = (np.random.rand(d, 1, 1) - 0.5) * noise
        vec += perturbation
        
        # 3. Re-normalize to ensure canonical property
        norm = np.linalg.norm(vec)
        vec /= norm
        
        mps_guess.append(vec)
        
    return mps_guess

# ==============================================================================
# 3. JORDAN-WIGNER LOGIC
# ==============================================================================
def get_jw_term_robust(op_str_list, indices, factor):
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
                swaps += 1
    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]
    final_indices = []
    final_ops_str = []
    parity = 0 
    extra_sign = 1
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1
        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1
    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)

def generate_chem_terms(h1, h2, tol=1e-15):
    terms = []
    n_spatial = h1.shape[0]
    for p in range(n_spatial):
        for q in range(n_spatial):
            val = h1[p, q]
            if abs(val) > tol:
                terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p, 2*q], val))
                terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*p+1, 2*q+1], val))
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    val = h2[p, q, r, s] 
                    if abs(val) < tol: continue
                    coef = 0.5 * val
                    op_strs = [r"a^\dagger", r"a^\dagger", "a", "a"]
                    cases = []
                    if p != r and q != s: # UpUp / DnDn
                        cases.append([2*p, 2*r, 2*s, 2*q])
                        cases.append([2*p+1, 2*r+1, 2*s+1, 2*q+1])
                    cases.append([2*p, 2*r+1, 2*s+1, 2*q])
                    cases.append([2*p+1, 2*r, 2*s, 2*q+1])
                    for idx_list in cases:
                        terms.append(get_jw_term_robust(op_strs, idx_list, coef))
    return terms

# ==============================================================================
# 4. MAIN TEST
# ==============================================================================
def run_exact_test():
    # Setup H8
    mol = gto.M(atom='H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3; H 0 0 4; H 0 0 5', unit='Bohr', basis='sto3g', verbose=0)
    # mol = gto.M(atom='H 0 0 0; H 0 0 2; H 0 0 4; H 0 0 6; H 0 0 8; H 0 0 10; H 0 0 12; H 0 0 14;', unit='Bohr', basis='ccpvtz', verbose=0)
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    n_orb = h1.shape[0]
    h2 = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n_orb,n_orb,n_orb,n_orb)
    
    logger.info("Generating Hamiltonian...")
    ham_terms = generate_chem_terms(h1, h2)
    n_spin = 2 * n_orb
    basis = [BasisSimpleElectron(i) for i in range(n_spin)]
    mpo = Mpo(Model(basis=basis, ham_terms=ham_terms), algo="qr")
    
    # --- TRANSPOSE MPO ---
    # Convert (L, U, D, R) -> (L, R, U, D) for mps.py
    mpo_dmrg_format = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
    
    # --- INITIAL GUESS (Noisy HF) ---
    logger.info("Generating Noisy Hartree-Fock Guess...")
    mps_guess = get_noisy_hf_guess(mol.nelectron, n_spin, noise=1e-4)
    
    logger.info("Running DMRG...")
    # Use D=50 for reasonable accuracy
    dmrg = mps_lib.DMRG(mpo_dmrg_format, D=20, nsweeps=5, init_guess=mps_guess) 
    dmrg.run()
    print(dmrg.make_rdm())
    print("------------------------------")
    print(dmrg.make_rdm2())
    e_dmrg = dmrg.e_tot + mol.energy_nuc()
    
    # my_fci = fci.FCI(mol, mf.mo_coeff)
    # e_fci, _ = my_fci.kernel()
    
    logger.info("-" * 50)
    # logger.info(f"FCI  Exact Energy: {e_fci:.12f}")
    logger.info(f"DMRG Total Energy: {e_dmrg:.12f}")
    logger.info("-" * 50)

if __name__ == "__main__":
    run_exact_test()