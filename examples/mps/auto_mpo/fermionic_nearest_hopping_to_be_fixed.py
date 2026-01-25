"""
Auto MPO Example: Fermionic Hopping Model (Tight-Binding)

Hamiltonian Definition:
-----------------------
The Hamiltonian represents spinless fermions hopping on a 1D chain with 
disordered on-site potentials.

    H = -t * Σ (c†_i * c_{i+1} + c†_{i+1} * c_i)  +  Σ (v_i * n_i)
             i                                     i

Where:
  - t       : Hopping amplitude
  - v_i     : Random on-site potential
  - c†, c   : Fermionic creation/annihilation operators (a^dag, a)
  - n_i     : Number operator (n)

Key Feature:
------------
The 'BasisSimpleElectron' has 'is_electron = True'. 
The MPO class will automatically insert 'sigma_z' gates (Jordan-Wigner strings)
to preserve fermionic anti-commutation relations.
"""

import logging
import numpy as np
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_fermionic_example():
    # ------------------------------------------------------------------
    # 1. Physics Parameters
    # ------------------------------------------------------------------
    nsite = 8       
    t = 1.0         
    
    # Use zero potential first to verify against exact solution
    # potentials = np.random.rand(nsite) * 0.5
    potentials = np.zeros(nsite) 
    
    logger.info(f"--- Building MPO for Fermionic Chain (N={nsite}) ---")

    # ------------------------------------------------------------------
    # 2. Build Hamiltonian (Manually Sorted)
    # ------------------------------------------------------------------
    ham_terms = []
    
    # Part A: Hopping
    for i in range(nsite - 1):
        # Forward: -t * c†_i c_{i+1}
        ham_terms.append(Op(r"a^\dagger", i) * Op("a", i+1) * (-t))
        
        # Backward: -t * c†_{i+1} c_i
        # Manually swap to site order (i, i+1) -> adds minus sign
        # -t * (- c_i c†_{i+1}) = +t * c_i c†_{i+1}
        ham_terms.append(Op("a", i) * Op(r"a^\dagger", i+1) * (t))
    
    # Part B: On-site Potential
    for i in range(nsite):
        if abs(potentials[i]) > 1e-12:
            ham_terms.append(Op("n", i, factor=potentials[i]))

    # ------------------------------------------------------------------
    # 3. Generate MPO
    # ------------------------------------------------------------------
    basis = [BasisSimpleElectron(dof=i) for i in range(nsite)]
    model = Model(basis=basis, ham_terms=ham_terms)
    
    # IMPORTANT: We transpose the MPO tensors for the solver immediately
    mpo = Mpo(model, algo="qr")
    
    # ------------------------------------------------------------------
    # 4. Verify Against Exact Solution (Tight Binding)
    # ------------------------------------------------------------------
    logger.info("Verifying against exact diagonalization...")
    
    # 1. Exact Eigenvalues for 1D chain with Open Boundary Conditions
    # E_k = -2t * cos(k * pi / (N+1))
    exact_energies = sorted([-2*t * np.cos(k * np.pi / (nsite + 1)) for k in range(1, nsite + 1)])
    exact_gs = sum(exact_energies[:nsite//2]) # Half-filling ground state
    
    # 2. MPO Dense Diagonalization
    H_dense = mpo.to_dense()
    mpo_evals = np.linalg.eigvalsh(H_dense)
    mpo_gs = mpo_evals[0]
    
    logger.info(f"Exact Half-Filling Energy: {exact_gs:.6f}")
    # Note: MPO GS is grand canonical (variable particle number), likely N=nsite/2 if chemical potential matches
    # But strictly, the lowest eigenvalue of H is for N_electrons = N_sites (all negative states filled)? 
    # Actually for simple hopping, GS is usually max filling if there's no chemical potential to penalize.
    # Let's check the lowest eigenvalue of the matrix directly:
    logger.info(f"MPO Matrix Min Energy:     {mpo_gs:.6f}")
    
    # For H = -t sum c+c, the spectrum is symmetric around 0.
    # The lowest energy state of the full Hilbert space is filling all negative energy levels.
    expected_matrix_min = sum([e for e in exact_energies if e < 0])
    
    logger.info(f"Expected Matrix Min:       {expected_matrix_min:.6f}")
    
    if np.isclose(mpo_gs, expected_matrix_min):
        logger.info(">>> SUCCESS: MPO matches exact solution.")
    else:
        logger.warning(">>> FAIL: Mismatch. Check operator ordering/signs.")

if __name__ == "__main__":
    run_fermionic_example()