import torch
import numpy as np
import logging
import scipy.sparse.linalg
import scipy.linalg

from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# 1. Block-Sparse SVD
def svd_qn_torch(matrix, qn_row, qn_col, noise=0.0):
    """
    Block-Sparse SVD with optional noise to escape local minima.
    """
    device = matrix.device
    real_dtype = matrix.real.dtype
    complex_dtype = matrix.dtype
    
    common_qns = np.intersect1d(qn_row.cpu().numpy(), qn_col.cpu().numpy())
    
    block_u, block_s, block_v = [], [], []
    qn_new_list = []
    u_indices, v_indices = [], []
    
    for q in common_qns:
        row_mask = (qn_row == q)
        col_mask = (qn_col == q)
        block = matrix[row_mask][:, col_mask]
        
        if block.numel() == 0: continue
            
        try:
            U, S, Vh = torch.linalg.svd(block, full_matrices=False)
        except RuntimeError:
            # Numerical stability fallback
            U, S, Vh = torch.linalg.svd(block + 1e-14 * torch.randn_like(block), full_matrices=False)
            
        block_u.append(U)
        block_s.append(S)
        block_v.append(Vh)
        qn_new_list.extend([q] * len(S))
        u_indices.append(torch.nonzero(row_mask).flatten())
        v_indices.append(torch.nonzero(col_mask).flatten())

    if not block_s:
        return (torch.zeros_like(matrix), torch.zeros(0, device=device, dtype=real_dtype), 
                torch.zeros_like(matrix), torch.zeros(0, device=device))

    # Reconstruct Sparse-Dense Tensors
    dim_bond = len(qn_new_list)
    U_full = torch.zeros(matrix.shape[0], dim_bond, dtype=complex_dtype, device=device)
    S_full = torch.tensor([s for block in block_s for s in block], dtype=real_dtype, device=device)
    Vh_full = torch.zeros(dim_bond, matrix.shape[1], dtype=complex_dtype, device=device)
    
    current = 0
    for i in range(len(block_s)):
        dim = len(block_s[i])
        U_full[u_indices[i][:, None], torch.arange(current, current+dim, device=device)] = block_u[i]
        Vh_full[torch.arange(current, current+dim, device=device)[:, None], v_indices[i]] = block_v[i]
        current += dim
        
    # Add Noise to prevent premature truncation of sectors
    if noise > 0:
        S_full = S_full + noise * torch.randn_like(S_full).abs()

    # Sort by singular value
    perm = torch.argsort(S_full, descending=True)
    return U_full[:, perm], S_full[perm], Vh_full[perm, :], torch.tensor(qn_new_list, device=device)[perm]



# 2. U(1) Symmetric MPS Class
class SymmetricMPS:
    def __init__(self, cores, qn_list, device='cpu'):
        self.cores = cores 
        self.qn = qn_list  
        self.device = device
        self.dtype = cores[0].dtype
        self.sigma_qn = torch.tensor([0, 1], device=device) 


    def canonicalise(self):
        """
        Orthogonalizes the MPS from Right to Left.
        Result: Right Canonical Form (Center is at Site 0).
        """
        # Iterate backwards from the last site down to site 1
        for i in range(len(self.cores) - 1, 0, -1):
            core = self.cores[i]
            B_L, P, B_R = core.shape
            
            # 1. Reshape for Right-Canonical SVD
            # We view the tensor as mapping: Left Bond -> (Physical + Right Bond)
            # Shape: (B_L, P * B_R)
            mat = core.reshape(B_L, P * B_R)
            
            # 2. Define Quantum Numbers
            # Conservation: Q_Left + Q_sigma = Q_Right
            # Therefore:    Q_Left = Q_Right - Q_sigma
            q_row = self.qn[i] # Current Left Bond QNs
            
            # Create target QNs for the columns (Physical + Right Bond)
            # This generates the "allowed" QNs for the left bond based on the right bond
            q_col = (self.qn[i+1].view(1, -1) - self.sigma_qn.view(-1, 1)).flatten()
            
            # 3. Perform SVD
            # M = U * S * Vh
            # U corresponds to the Left Bond indices
            # Vh corresponds to the (Physical + Right Bond) indices
            U, S, Vh, q_bond = svd_qn_torch(mat, q_row, q_col)
            
            # 4. Truncate (keep only non-zero singular values)
            mask = S > 1e-12
            U, S, Vh, q_bond = U[:, mask], S[mask], Vh[mask, :], q_bond[mask]
            
            # 5. Update the current site to be Right-Isometric (Vh)
            # Vh shape is (D_new, P*B_R) -> Reshape to (D_new, P, B_R)
            self.cores[i] = Vh.reshape(-1, P, B_R)
            
            # 6. Update the Quantum Numbers for the Left Bond (Bond i)
            # Note: q_bond returned by SVD matches the columns of U (and rows of Vh)
            self.qn[i] = q_bond 
            
            # 7. Absorb U and S into the LEFT neighbor (Site i-1)
            # core[i-1] has shape (Left_prev, P, Right_prev)
            # Right_prev matches rows of U.
            # New Core[i-1] = Old Core[i-1] @ (U * S)
            US = U @ torch.diag(S.to(self.dtype))
            prev_core = self.cores[i-1]
            
            # Contraction:
            # prev_core: (L_prev, P, R_prev)
            # US:        (R_prev, D_new)
            # Result:    (L_prev, P, D_new)
            new_prev = torch.tensordot(prev_core, US, dims=([2], [0]))
            self.cores[i-1] = new_prev

        return self

# 3. DMRG Solver
class SymmetricDMRG:
    def __init__(self, mps, mpo_cores):
        self.mps = mps
        self.mpo = mpo_cores
        self.L_env = [None] * len(mps.cores)
        self.R_env = [None] * len(mps.cores)
        self.build_initial_environments()
        
    def build_initial_environments(self):
        """Builds all R_env initially. L_env[0] is built."""
        device = self.mps.device
        dtype = self.mps.dtype
        
        # L boundary
        self.L_env[0] = torch.ones(1, 1, 1, device=device, dtype=dtype)
        # R boundary
        self.R_env[-1] = torch.ones(1, 1, 1, device=device, dtype=dtype)
        
        # Build R environments from right to left
        for i in range(len(self.mps.cores) - 1, 0, -1):
            self.R_env[i-1] = self.contract_env_block(self.R_env[i], self.mps.cores[i], self.mpo[i], 'right')

    def contract_env_block(self, Env, A, W, side):
        A_conj = A.conj()
        if side == 'left':
            T = torch.einsum('lwk, lpr -> wkpr', Env, A)
            Y = torch.einsum('wkpr, wpqv -> krqv', T, W)
            NewE = torch.einsum('krqv, kqs -> rvs', Y, A_conj)
        else: # right
            T = torch.einsum('rwk, lpr -> wklp', Env, A)
            Y = torch.einsum('wklp, vqpw -> klvq', T, W)
            NewE = torch.einsum('klvq, sqk -> lvs', Y, A_conj)
        return NewE

    def optimize_two_sites(self, i, direction, max_bond_dim, noise):
        """Optimizes sites i and i+1."""
        # 1. Form Theta Guess
        theta_guess = torch.tensordot(self.mps.cores[i], self.mps.cores[i+1], dims=([2], [0]))
        
        L = self.L_env[i]
        R = self.R_env[i+1]
        W1 = self.mpo[i]
        W2 = self.mpo[i+1]
        
        # 2. Mask for Valid U(1) Sector
        q_L = self.mps.qn[i].cpu().numpy()
        q_R = self.mps.qn[i+2].cpu().numpy()
        sigma = np.array([0, 1])
        Q_tensor = (q_L[:, None, None, None] + sigma[None, :, None, None] + 
                    sigma[None, None, :, None])
        
        mask = (Q_tensor == q_R[None, None, None, :])
        mask_t = torch.tensor(mask, device=self.mps.device)
        valid_indices = torch.nonzero(mask_t)
        
        if len(valid_indices) == 0: 
            return 0.0 # Should not happen if init is correct

        flat_guess = theta_guess[mask_t].detach().cpu().numpy()
        dim_sector = len(flat_guess)
        
        # 3. H_eff * v (Lanczos Matvec)
        def matvec(v):
            v_tensor = torch.zeros_like(theta_guess)
            v_tensor[mask_t] = torch.tensor(v, dtype=self.mps.dtype, device=self.mps.device)
            
            T1 = torch.tensordot(L, v_tensor, dims=([2],[0])) 
            T2 = torch.tensordot(T1, W1, dims=([1,2],[0,2])) 
            T3 = torch.tensordot(T2, W2, dims=([1,4],[2,0]))
            Out = torch.tensordot(T3, R, dims=([1,4],[0,1]))
            
            return Out[mask_t].detach().cpu().numpy()

        Op = scipy.sparse.linalg.LinearOperator((dim_sector, dim_sector), matvec=matvec, dtype=np.complex128)

        # 4. Davidson/Lanczos
        if dim_sector <= 10:
            dense_H = np.zeros((dim_sector, dim_sector), dtype=np.complex128)
            for k in range(dim_sector):
                v = np.zeros(dim_sector, dtype=np.complex128); v[k]=1.0
                dense_H[:, k] = matvec(v)
            vals, vecs = scipy.linalg.eigh(dense_H)
            ground_energy = vals[0]
            ground_vec = vecs[:, 0]
        else:
            # 'SA' = Smallest Algebraic (lowest energy)
            vals, vecs = scipy.sparse.linalg.eigsh(Op, k=1, v0=flat_guess, which='SA')
            ground_energy = vals[0]
            ground_vec = vecs[:, 0]
        
        # 5. Update Cores (SVD)
        theta_new = torch.zeros_like(theta_guess)
        theta_new[mask_t] = torch.tensor(ground_vec, dtype=self.mps.dtype, device=self.mps.device)
        
        mat = theta_new.view(theta_guess.shape[0] * 2, 2 * theta_guess.shape[3])
        q_row = (self.mps.qn[i].view(-1, 1) + self.mps.sigma_qn.view(1, -1)).flatten()
        q_col = (self.mps.qn[i+2].view(1, -1) - self.mps.sigma_qn.view(-1, 1)).flatten() 
        
        # Pass noise here
        U, S, Vh, q_bond = svd_qn_torch(mat, q_row, q_col, noise=noise)
        
        # Truncation
        # Keep states if S > 1e-12 OR if we are forced by noise/max_bond
        keep = S > 1e-12
        if max_bond_dim and keep.sum() > max_bond_dim:
             keep_idx = torch.argsort(S, descending=True)[:max_bond_dim]
             # Ensure we don't keep strictly zero garbage if noise is 0
             if noise == 0:
                 keep_idx = keep_idx[S[keep_idx] > 1e-12]
             U, S, Vh, q_bond = U[:, keep_idx], S[keep_idx], Vh[keep_idx, :], q_bond[keep_idx]
        else:
             U, S, Vh, q_bond = U[:, keep], S[keep], Vh[keep, :], q_bond[keep]

        # 6. Update MPS and Environments based on Direction
        if direction == 'right':
            # Moving Right: Core[i] becomes Left-Orthogonal (U), Core[i+1] absorbs S
            self.mps.cores[i] = U.reshape(theta_guess.shape[0], 2, -1)
            self.mps.qn[i+1] = q_bond
            self.mps.cores[i+1] = (torch.diag(S.to(self.mps.dtype)) @ Vh).reshape(-1, 2, theta_guess.shape[3])
            
            # Update Left Env for next step
            self.L_env[i+1] = self.contract_env_block(self.L_env[i], self.mps.cores[i], self.mpo[i], 'left')
            
        else: # direction == 'left'
            # Moving Left: Core[i+1] becomes Right-Orthogonal (Vh), Core[i] absorbs S
            self.mps.cores[i+1] = Vh.reshape(-1, 2, theta_guess.shape[3])
            self.mps.qn[i+1] = q_bond
            self.mps.cores[i] = (U @ torch.diag(S.to(self.mps.dtype))).reshape(theta_guess.shape[0], 2, -1)
            
            # Update Right Env for next step
            self.R_env[i] = self.contract_env_block(self.R_env[i+1], self.mps.cores[i+1], self.mpo[i+1], 'right')
            
        return ground_energy

    def run_sweep(self, max_bond_dim=20, noise=0.0):
        num_sites = len(self.mps.cores)
        
        # Sweep Left -> Right
        for i in range(num_sites - 1):
            E = self.optimize_two_sites(i, 'right', max_bond_dim, noise)
            
        # Sweep Right -> Left
        for i in range(num_sites - 2, -1, -1):
            E = self.optimize_two_sites(i, 'left', max_bond_dim, noise)
            
        return E


def main():
    bridge_file = "h4_dmrg_input_nz_16.pt"
    # Load data safely
    try: 
        data = torch.load(bridge_file, weights_only=False)
    except TypeError: 
        data = torch.load(bridge_file)
    
    t_ij = data["t_ij"].numpy()
    J_matrix = data["V_coulomb"].numpy()
    # K_matrix = data["V_exchange"].numpy() # IGNORED: Exchange is implicit in Full CI
    Nz = data["Nz"]
    E_nuc = data["enuc"] if "enuc" in data else 0.0
    nelec = data["nelec"]

    # 1. Full Hamiltonian Construction
    logger.info("Building Hamiltonian MPO...")
    ham_terms = []
    
    # Helpers for Renormalizer Ops
    def c_dag(i): return Op(r"a^\dagger", i)
    def c(i): return Op(r"a", i)
    def n(i): return Op(r"a^\dagger a", i)
    
    cutoff = 1e-10
    
    # --- A. Kinetic Energy / Hopping ---
    rows, cols = np.nonzero(np.abs(t_ij) > cutoff)
    for i, j in zip(rows, cols):
        val = t_ij[i, j]
        # Spin Up hopping
        ham_terms.append(c_dag(2*i)*c(2*j)*val)
        # Spin Down hopping
        ham_terms.append(c_dag(2*i+1)*c(2*j+1)*val)
        
    # --- B. Coulomb Interaction Only ---
    # In DVR basis, V_ijkl approx delta_ik delta_jl V_ij. 
    # Interaction is purely density-density: sum V_ij n_i n_j
    rows, cols = np.nonzero(np.abs(J_matrix) > cutoff)
    
    for i, k in zip(rows, cols):
        if i == k:
            # Diagonal (Hubbard U): On-site repulsion
            # Only possible between Up and Down due to Pauli exclusion
            val = J_matrix[i, k]
            ham_terms.append(n(2*i)*n(2*i+1)*val)
        else:
            # Off-Diagonal: Inter-site repulsion
            # Factor 0.5 is correct here because loop visits (i,k) and (k,i) separately
            val = 0.5 * J_matrix[i, k]
            
            # Add all spin combinations
            ham_terms.append(n(2*i)*n(2*k)*val)     # Up-Up
            ham_terms.append(n(2*i+1)*n(2*k+1)*val) # Down-Down
            ham_terms.append(n(2*i)*n(2*k+1)*val)   # Up-Down
            ham_terms.append(n(2*i+1)*n(2*k)*val)   # Down-Up

    # --- C. Exchange Interaction ---
    # DELETED: Do not add K_matrix terms.
    # In exact diagonalization (DMRG), exchange energy is not a Hamiltonian term;
    # it is a result of the antisymmetric wavefunction minimizing the Coulomb energy.
    
    # 2. Build Model & MPO
    basis = [BasisSimpleElectron(i) for i in range(2 * Nz)]
    model = Model(basis=basis, ham_terms=ham_terms)
    
    renom_mpo = Mpo(model)
    mpo_cores = []
    for t in renom_mpo:
        tn = t.to_dense() if hasattr(t,"to_dense") else np.array(t)
        mpo_cores.append(torch.from_numpy(tn).to(dtype=torch.complex128))

    # 3. Initialize MPS
    logger.info(f"Initializing Symmetric MPS for N_elec = {nelec}...")
    cores = []
    qn_list = [torch.tensor([0])]
    curr_n = 0
    
    # Fill orbitals from left to right until electrons run out
    for i in range(2 * Nz):
        C = torch.zeros(1, 2, 1, dtype=torch.complex128)
        if curr_n < nelec:
            C[0, 1, 0] = 1.0 # Occupied
            curr_n += 1
        else:
            C[0, 0, 0] = 1.0 # Empty
        cores.append(C)
        qn_list.append(torch.tensor([curr_n]))
    
    psi = SymmetricMPS(cores, qn_list)
    psi.canonicalise()
    
    # Check initialization
    final_n = psi.qn[-1].item()
    if final_n != nelec:
        logger.error(f"MPS Initialization Error: Created {final_n} electrons, expected {nelec}")
        return

    # 4. DMRG Loop
    solver = SymmetricDMRG(psi, mpo_cores)
    logger.info("Starting Sweeps...")
    
    # Robust Sweep Schedule:
    # 1. High Noise (1e-3) to break symmetries and find correct subspace
    # 2. Medium Noise (1e-4) to refine
    # 3. Zero Noise to converge energy
    noises = [1e-2]*2 + [1e-3]*2 + [1e-4]*4 + [1e-5]*4 + [0.0]*8
    noises = [1e-3]*4 + [1e-4]*4 + [1e-5]*4 + [0.0]*8
    noises = [0.0]*8
    
    for sweep, noise in enumerate(noises):
        E = solver.run_sweep(max_bond_dim=20, noise=noise)
        logger.info(f"Sweep {sweep+1:2d} (Noise={noise:.1e}): Total Energy = {E + E_nuc:.8f} Ha")
        
    logger.info(f"Final Total Energy: {E + E_nuc:.8f} Ha")

if __name__ == "__main__":
    main()