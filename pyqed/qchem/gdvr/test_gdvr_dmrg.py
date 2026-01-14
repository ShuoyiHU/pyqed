import torch
import numpy as np
import logging
from pyscf import lib

from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo

import pyqed.mps.mps as mps_lib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Local Fix of BasisSimpleElectron (Add sigma_z support for Jordan-Wigner)
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

# [Patch C] Fix Boundary Condition Mismatch (The "Sweep 0 Convergence" Bug)
# MPO Gen puts Identity at [0]. mps.py expects it at [-1]. We force it to [0].
def patched_initial_F(W):
    F = np.zeros((W.shape[1], 1, 1))
    F[0] = 1.0 
    return F
mps_lib.initial_F = patched_initial_F

# ==============================================================================
# 2. HELPER: NOISY INITIAL GUESS
# ==============================================================================
def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    """
    Creates a Hartree-Fock product state with small random noise
    to break symmetries and aid convergence.
    """
    d = 2
    mps_guess = []
    
    # Track current electrons to ensure we create exactly n_elec occupied sites
    filled_count = 0
    
    for i in range(n_spin):
        # 1. Create pure HF state for this site
        vec = np.zeros((d, 1, 1))
        
        # Simple filling: Fill first n_elec orbitals
        if filled_count < n_elec: 
            vec[1, 0, 0] = 1.0 # Occupied
            filled_count += 1
        else: 
            vec[0, 0, 0] = 1.0 # Empty
            
        # 2. Add random noise to break symmetry
        perturbation = (np.random.rand(d, 1, 1) - 0.5) * noise
        vec += perturbation
        
        # 3. Re-normalize
        norm = np.linalg.norm(vec)
        vec /= norm
        
        mps_guess.append(vec)
        
    return mps_guess

# ==============================================================================
# 3. MAIN SCRIPT
# ==============================================================================
def main():
    bridge_file = "h8_dmrg_input_nz_32_far.pt"
    # Load data safely
    try: 
        data = torch.load(bridge_file, weights_only=False)
    except TypeError: 
        data = torch.load(bridge_file)
    
    t_ij = data["t_ij"].numpy()
    J_matrix = data["V_coulomb"].numpy()
    Nz = data["Nz"]
    E_nuc = data["enuc"] if "enuc" in data else 0.0
    nelec = data["nelec"]
    n_spin = 2 * Nz

    logger.info(f"System: Nz={Nz} (Spin-Orbitals={n_spin}), Electrons={nelec}")

    # --- 1. Construct Hamiltonian Terms ---
    logger.info("Building Hamiltonian MPO...")
    ham_terms = []
    
    # Helpers
    def c_dag(i): return Op(r"a^\dagger", i)
    def c(i): return Op(r"a", i)
    def n(i): return Op(r"a^\dagger a", i)
    
    cutoff = 1e-10
    
    # A. Kinetic Energy (Hopping)
    rows, cols = np.nonzero(np.abs(t_ij) > cutoff)
    for i, j in zip(rows, cols):
        val = t_ij[i, j]
        # Spin Up
        ham_terms.append(c_dag(2*i)*c(2*j)*val)
        # Spin Down
        ham_terms.append(c_dag(2*i+1)*c(2*j+1)*val)
        
    # B. Coulomb Interaction (Density-Density)
    rows, cols = np.nonzero(np.abs(J_matrix) > cutoff)
    for i, k in zip(rows, cols):
        if i == k:
            # Diagonal (Hubbard U)
            val = J_matrix[i, k]
            ham_terms.append(n(2*i)*n(2*i+1)*val)
        else:
            # Off-Diagonal
            val = 0.5 * J_matrix[i, k]
            ham_terms.append(n(2*i)*n(2*k)*val)     # Up-Up
            ham_terms.append(n(2*i+1)*n(2*k+1)*val) # Dn-Dn
            ham_terms.append(n(2*i)*n(2*k+1)*val)   # Up-Dn
            ham_terms.append(n(2*i+1)*n(2*k)*val)   # Dn-Up

    # --- 2. Build MPO ---

    
    basis = [BasisSimpleElectron(i) for i in range(n_spin)]
    model = Model(basis=basis, ham_terms=ham_terms)
    
    # Use 'qr' for standard construction
    renom_mpo = Mpo(model, algo="qr")
    
    # --- 3. Format MPO for DMRG Solver ---
    # Renormalizer Output: (Left, Up, Down, Right)
    # MPS Lib Expects:     (Left, Right, Up, Down)
    mpo_dmrg_format = []
    for t in renom_mpo.matrices:
        # t is (L, U, D, R) -> Transpose to (L, R, U, D)
        mpo_dmrg_format.append(t.transpose(0, 3, 1, 2))

    # --- 4. Initialize MPS (Noisy HF Guess) ---
    logger.info("Initializing MPS...")
    mps_guess = get_noisy_hf_guess(nelec, n_spin, noise=1e-3)

    # --- 5. Run DMRG ---
    logger.info("Starting DMRG Optimization...")
    
    # Parameters
    bond_dim = 20
    sweeps = 20
    
    solver = mps_lib.DMRG(mpo_dmrg_format, D=bond_dim, nsweeps=sweeps, init_guess=mps_guess)
    solver.run()
    
    # --- 6. Results ---
    energy_dmrg = solver.e_tot
    total_energy = energy_dmrg + E_nuc
    
    logger.info("=" * 40)
    logger.info(f"Electronic Energy: {energy_dmrg:.10f} Ha")
    logger.info(f"Nuclear Repulsion: {E_nuc:.10f} Ha")
    logger.info(f"Total Energy:      {total_energy:.10f} Ha")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()