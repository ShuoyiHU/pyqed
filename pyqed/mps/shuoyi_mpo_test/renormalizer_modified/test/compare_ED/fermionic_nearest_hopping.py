import numpy as np
import scipy.sparse
import scipy.linalg
import logging

# --- Imports from your path ---
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_exact_hamiltonian(nsite, ham_terms):
    """
    Builds the Ground Truth Fermionic Hamiltonian (String-on-Left).
    This matches the convention used by the Renormalizer MPO.
    """
    dim = 2 ** nsite
    H_exact = scipy.sparse.csr_matrix((dim, dim), dtype=complex)
    
    # Basis Definition: |0> = [1,0], |1> = [0,1]
    # Note: Renormalizer basis order is [Empty, Occupied]
    I = scipy.sparse.eye(2)
    a_mat = scipy.sparse.csr_matrix([[0, 1], [0, 0]])    # Annihilate |1> -> |0>
    adag_mat = scipy.sparse.csr_matrix([[0, 0], [1, 0]]) # Create |0> -> |1>
    n_mat = scipy.sparse.csr_matrix([[0, 0], [0, 1]])    # Number
    Z_mat = scipy.sparse.csr_matrix([[1, 0], [0, -1]])   # Pauli Z for Strings
    
    def get_fermionic_op(site_idx, op_type):
        """
        Constructs a global operator with Jordan-Wigner strings attached to the LEFT.
        Convention: O_i = Z_0 ... Z_{i-1} * local_op_i * I_{i+1} ...
        """
        op_list = []
        for k in range(nsite):
            if k < site_idx:
                op_list.append(Z_mat) # String Phase
            elif k == site_idx:
                if op_type == "a": op_list.append(a_mat)
                elif op_type == "adag": op_list.append(adag_mat)
                elif op_type == "n": op_list.append(n_mat)
            else:
                op_list.append(I)     # Identity
                
        # Kronecker product full chain
        full_op = op_list[0]
        for k in range(1, nsite):
            full_op = scipy.sparse.kron(full_op, op_list[k])
        return full_op

    for term in ham_terms:
        factor = term.factor
        indices = term.dofs
        ops = term.split_symbol
        
        # Build the term matrix
        if len(indices) == 1:
            # Clean symbol (handle "a^\dagger" vs "a")
            clean_sym = "adag" if "dagger" in ops[0] else ("n" if "a" in ops[0] and len(ops[0]) > 1 else "a")
            term_matrix = get_fermionic_op(indices[0], clean_sym)

        elif len(indices) == 2:
            idx1, idx2 = indices
            clean1 = "adag" if "dagger" in ops[0] else "a"
            clean2 = "adag" if "dagger" in ops[1] else "a"
            
            mat1 = get_fermionic_op(idx1, clean1)
            mat2 = get_fermionic_op(idx2, clean2)
            term_matrix = mat1 @ mat2
            
        H_exact += factor * term_matrix
        
    return H_exact.toarray()

def test_hopping_model():
    # ------------------------------------------------------------------
    # 1. Define Physics Parameters
    # ------------------------------------------------------------------
    nsite = 8   # System size
    t = 1.0     # Hopping strength
    
    logger.info(f"--- Starting Test: Fermionic Hopping Model (N={nsite}) ---")

    # ------------------------------------------------------------------
    # 2. Build Hamiltonian Terms
    # ------------------------------------------------------------------
    ham_terms = []
    
    # Nearest Neighbor Hopping: -t * (a^dag_i a_{i+1} + h.c.)
    # We define h.c. explicitly to ensure matrix is Hermitian
    for i in range(nsite - 1):
        ham_terms.append(Op(r"a^\dagger a", [i, i+1], factor=-t))
        ham_terms.append(Op(r"a^\dagger a", [i+1, i], factor=-t)) # h.c. term (adag_{i+1} a_i)
    
    # Add Random Potential
    np.random.seed(42)
    random_pot = np.random.rand(nsite) * 0.5
    for i in range(nsite):
         ham_terms.append(Op(r"a^\dagger a", i, factor=random_pot[i]))

    # ------------------------------------------------------------------
    # 3. Generate MPO 
    # ------------------------------------------------------------------
    logger.info("Generating MPO via Light Automatic MPO...")
    basis = [BasisSimpleElectron(i) for i in range(nsite)]
    model = Model(basis=basis, ham_terms=ham_terms)
    
    mpo = Mpo(model, algo="qr")
    
    logger.info("Calling mpo.to_dense() ...")
    try:
        H_mpo_dense = mpo.to_dense(check_size=True)
    except ValueError as e:
        logger.error(str(e))
        return

    # ------------------------------------------------------------------
    # 4. Generate Exact Hamiltonian (Ground Truth)
    # ------------------------------------------------------------------
    logger.info("Generating Exact Hamiltonian (Left-String Convention)...")
    H_exact_dense = build_exact_hamiltonian(nsite, ham_terms)

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    
    # Check 1: Frobenius Norm
    diff = np.linalg.norm(H_mpo_dense - H_exact_dense)
    logger.info(f"Frobenius Norm Difference: {diff:.2e}")
    
    if diff < 1e-10:
        logger.info("[SUCCESS] Matrices match element-wise.")
    else:
        logger.error("[FAIL] Matrix mismatch. Check string convention or basis order.")

    # Check 2: Ground State Energy
    logger.info("Diagonalizing to check Energies...")
    # eigvalsh is faster for Hermitian matrices
    E0_mpo = scipy.linalg.eigvalsh(H_mpo_dense)[0]
    E0_exact = scipy.linalg.eigvalsh(H_exact_dense)[0]
    
    logger.info(f"Ground Energy (MPO):   {E0_mpo:.10f}")
    logger.info(f"Ground Energy (Exact): {E0_exact:.10f}")
    
    if np.abs(E0_mpo - E0_exact) < 1e-10:
        logger.info("[SUCCESS] Ground state energies match.")
    else:
        logger.error("[FAIL] Energy mismatch.")

if __name__ == "__main__":
    test_hopping_model()