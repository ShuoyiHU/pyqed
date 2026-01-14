import numpy as np
import scipy.sparse
import scipy.linalg
import logging

from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSHO
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mpo_to_dense(mpo):
    """
    Contract the MPO tensors into a full dense matrix.
    Standard Renormalizer shape: (Bond_Left, Phys_Up, Phys_Down, Bond_Right)
    """
    accum = np.ones((1, 1, 1), dtype=mpo.dtype)
    for i, W in enumerate(mpo):
        # Contract Accum's last bond with W's first bond
        temp = np.tensordot(accum, W, axes=([-1], [0]))
        # Transpose to: (Rows_accum, Phys_Row, Cols_accum, Phys_Col, Bond_Right)
        temp = np.transpose(temp, (0, 2, 1, 3, 4))
        rows_old, phys_rows, cols_old, phys_cols, bond_right = temp.shape
        # Reshape to merge indices
        new_rows = rows_old * phys_rows
        new_cols = cols_old * phys_cols
        accum = temp.reshape(new_rows, new_cols, bond_right)
    return accum.squeeze()

def build_exact_bosonic_hamiltonian(model, ham_terms):
    """
    Builds Hamiltonian for Bosons using simple Kronecker products.
    Unlike Fermions, Bosons on different sites COMMUTE, so no Z-strings needed.
    """
    nsite = len(model.basis)
    # Calculate total Hilbert space dimension
    dim_list = [b.nbas for b in model.basis]
    total_dim = np.prod(dim_list)
    
    H_exact = scipy.sparse.csr_matrix((total_dim, total_dim), dtype=complex)
    
    # We use the model's own basis objects to generate local matrices.
    # This ensures consistency in normalization factors (like sqrt(n) vs n).
    
    def get_local_mat(site_idx, op_str):
        # Use the basis class to get the matrix (e.g. 3x3 for nbas=3)
        return scipy.sparse.csr_matrix(model.basis[site_idx].op_mat(op_str))

    for term in ham_terms:
        factor = term.factor
        indices = term.dofs
        ops = term.split_symbol
        
        # Start with Identity on all sites
        # We build the list of matrices to Kron: [M_0, M_1, ... M_N]
        op_list = []
        for i in range(nsite):
            op_list.append(scipy.sparse.eye(dim_list[i]))
            
        # Replace specific sites with their operators
        if len(indices) == 1:
            idx = indices[0]
            op_list[idx] = get_local_mat(idx, ops[0])
            
        elif len(indices) == 2:
            idx1, idx2 = indices
            # For Bosons, order of A_i * B_j doesn't matter if i != j
            # But we stick to the order in the operator string for correctness
            op_list[idx1] = get_local_mat(idx1, ops[0])
            
            # Handle case where both ops act on same site (rare in this format, but possible)
            if idx1 == idx2:
                # Multiply them locally first
                op_list[idx1] = get_local_mat(idx1, ops[0]) @ get_local_mat(idx1, ops[1])
            else:
                op_list[idx2] = get_local_mat(idx2, ops[1])
        
        # Perform Kronecker Product
        term_mat = op_list[0]
        for k in range(1, nsite):
            term_mat = scipy.sparse.kron(term_mat, op_list[k])
            
        H_exact += factor * term_mat
        
    return H_exact

def test_bosonic_chain():
    # ------------------------------------------------------------------
    # 1. Define Physics Parameters
    # ------------------------------------------------------------------
    nsite = 4         # Keep small because Hilbert space grows fast (nbas^N)
    nbas = 4          # Number of levels per oscillator (0, 1, 2, 3)
    omega = 1.0       # Oscillator frequency
    coupling = 0.2    # Coupling strength
    
    logger.info(f"--- Starting Test: Bosonic Chain (N={nsite}, Levels={nbas}) ---")
    logger.info(f"Total Dimension: {nbas}^{nsite} = {nbas**nsite}")

    # ------------------------------------------------------------------
    # 2. Build Hamiltonian Terms
    # ------------------------------------------------------------------
    ham_terms = []
    
    # On-site Energy: sum omega * b^dag b
    for i in range(nsite):
        ham_terms.append(Op(r"b^\dagger b", i, factor=omega))
    
    # Nearest Neighbor Coupling: J * (b^dag_i b_{i+1} + h.c.)
    for i in range(nsite - 1):
        ham_terms.append(Op(r"b^\dagger", i) * Op("b", i+1) * coupling)
        ham_terms.append(Op("b", i) * Op(r"b^\dagger", i+1) * coupling)

    # ------------------------------------------------------------------
    # 3. Generate MPO (Test Subject)
    # ------------------------------------------------------------------
    logger.info("Generating MPO via Light Automatic MPO...")
    
    # Use BasisSHO (Simple Harmonic Oscillator)
    # Note: 'dof' is just a label here
    basis = [BasisSHO(dof=i, omega=omega, nbas=nbas) for i in range(nsite)]
    model = Model(basis=basis, ham_terms=ham_terms)
    
    # Initialize light MPO
    mpo = Mpo(model, algo="qr")
    
    logger.info("Contracting MPO to Dense Matrix...")
    H_mpo_dense = mpo_to_dense(mpo)
    
    # ------------------------------------------------------------------
    # 4. Generate Exact Hamiltonian (Ground Truth)
    # ------------------------------------------------------------------
    logger.info("Generating Exact Bosonic Hamiltonian...")
    H_exact_sparse = build_exact_bosonic_hamiltonian(model, ham_terms)
    H_exact_dense = H_exact_sparse.toarray()

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    
    # Check 1: Frobenius Norm
    diff = np.linalg.norm(H_mpo_dense - H_exact_dense)
    logger.info(f"Frobenius Norm Difference: {diff:.2e}")
    
    if diff < 1e-10:
        logger.info("[SUCCESS] Matrices match element-wise.")
    else:
        logger.error("[FAIL] Matrix mismatch.")

    # Check 2: Spectra (Energy)
    logger.info("Diagonalizing...")
    eig_mpo = scipy.linalg.eigvalsh(H_mpo_dense)
    eig_exact = scipy.linalg.eigvalsh(H_exact_dense)
    
    # Check Ground State and First Excited State
    logger.info(f"Ground Energy (MPO):   {eig_mpo[0]:.10f}")
    logger.info(f"Ground Energy (Exact): {eig_exact[0]:.10f}")
    
    # Check Gap (physics check)
    gap_mpo = eig_mpo[1] - eig_mpo[0]
    gap_exact = eig_exact[1] - eig_exact[0]
    logger.info(f"Gap (MPO):   {gap_mpo:.10f}")
    logger.info(f"Gap (Exact): {gap_exact:.10f}")
    
    if np.abs(eig_mpo[0] - eig_exact[0]) < 1e-10:
        logger.info("[SUCCESS] Ground state energies match!")
    else:
        logger.error("[FAIL] Energy mismatch.")

if __name__ == "__main__":
    test_bosonic_chain()