import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

# --- 1. Basic Setup & MPO Construction ---
def build_heisenberg_mpo(N, J=1.0):
    """
    Constructs the Heisenberg Hamiltonian as an MPO (Matrix Product Operator).
    The MPO 'W' tensors have shape (Left_Bond, Right_Bond, Physical_Row, Physical_Col).
    """
    # Pauli Matrices
    I = np.eye(2)
    Z = np.array([[0.5, 0], [0, -0.5]])
    Sp = np.array([[0, 1], [0, 0]])
    Sm = np.array([[0, 0], [1, 0]])
    Sz = Z

    # MPO tensor W construction (5x5 virtual dimension)
    # H = 0.5(S+S- + S-S+) + SzSz
    W = np.zeros((5, 5, 2, 2))
    
    # 1. Identity passthrough (Column 0 -> Column 0)
    W[0, 0] = I
    # 5. Identity passthrough (Row 4 -> Row 4)
    W[4, 4] = I
    
    # Interaction Terms
    # Left side (kinda like putting operators in the "pipeline")
    W[0, 1] = Sp      # Put S+ in pipe 1
    W[0, 2] = Sm      # Put S- in pipe 2
    W[0, 3] = Sz      # Put Sz in pipe 3
    
    # Right side (taking operators out of "pipeline" and completing interaction)
    W[1, 4] = 0.5*Sm  # Connect pipe 1 (S+) to 0.5*S-
    W[2, 4] = 0.5*Sp  # Connect pipe 2 (S-) to 0.5*S+
    W[3, 4] = Sz      # Connect pipe 3 (Sz) to Sz

    # Create list of W tensors for the chain
    mpo = [W] * N
    
    # Boundary conditions: First and Last tensors are vectors, not matrices
    # First site: only allow row 0 (start state)
    mpo[0] = mpo[0][0:1, :, :, :] 
    # Last site: only allow column 4 (end state - complete Hamiltonian)
    mpo[-1] = mpo[-1][:, 4:5, :, :]
    
    return mpo

# --- 2. MPS Initialization ---
def init_mps(N, bond_dim, phys_dim=2):
    """ Creates a random MPS with bond dimension 'bond_dim' """
    mps = []
    for i in range(N):
        # Shape: (Bond_Left, Bond_Right, Physical)
        # Note: We put Physical index last for easier reshaping later (standard in some libraries)
        # but here we'll stick to (L, R, Phys) or (L, Phys, R). 
        # Let's use (Bond_Left, Physical, Bond_Right) for standard "A[alpha, s, beta]"
        d_left = 1 if i == 0 else bond_dim
        d_right = 1 if i == N-1 else bond_dim
        
        A = np.random.rand(d_left, phys_dim, d_right)
        
        # Normalize slightly to avoid explosion
        mps.append(A / np.linalg.norm(A))
    return mps

# --- 3. The Heart of MPS: Contractions ---

def contract_env(tensor_L, tensor_A, tensor_W, tensor_A_conj):
    """
    Updates the Left Environment (L) by adding one site.
    Contraction: L_env + MPS_A + MPO_W + MPS_A* -> New_L_env
    """
    # A common contraction pattern. Using einsum for clarity.
    # L: (top_bond, mid_bond, bot_bond)
    # A: (bond_left, phys, bond_right)
    # W: (w_left, w_right, phys_up, phys_down)
    
    # 1. Contract L with A (bra)
    temp = np.tensordot(tensor_L, tensor_A, axes=(0, 0)) # sum over top_bond/bond_left
    
    # 2. Contract with W
    # temp has axes: (mid_bond, bot_bond, phys, bond_right)
    # W has axes: (w_left, w_right, phys_up, phys_down)
    # We sum: mid_bond==w_left AND phys==phys_up
    temp = np.tensordot(temp, tensor_W, axes=([0, 2], [0, 2]))
    
    # 3. Contract with A* (ket)
    # temp: (bot_bond, bond_right_bra, w_right, phys_down)
    # A_conj: (bond_left, phys, bond_right) -> bond_left matches bot_bond, phys matches phys_down
    new_L = np.tensordot(temp, tensor_A_conj, axes=([0, 3], [0, 1]))
    
    # new_L axes: (bond_right_bra, w_right, bond_right_ket)
    return new_L
def contract_env_R(tensor_R, tensor_A, tensor_W, tensor_A_conj):
    """ 
    Updates Right Environment.
    R: (bra, mpo, ket) -> a, b, c
    A: (left, phys, right) -> d, e, a (Right connects to R)
    W: (left, right, out, in) -> f, b, e, g (Right connects to R, Out connects to A)
    A_conj: (left, phys, right) -> h, g, c (Right connects to R, Phys connects to W In)
    
    Result: (d, f, h) -> (A_left_bra, W_left, A_left_ket)
    """
    return np.einsum('abc, dea, fbeg, hgc -> dfh', 
                     tensor_R, tensor_A, tensor_W, tensor_A_conj, 
                     optimize=True)
# --- 4. The Local Update (Lanczos + SVD) ---
def local_update(L, R, W1, W2, theta_guess):
    """
    Optimizes the two-site wavefunction 'theta' (shape: L_bond, p1, p2, R_bond)
    """
    dim_L = L.shape[0]
    dim_R = R.shape[0]
    phys_d = 2
    
    def matvec(v):
        theta = v.reshape(dim_L, phys_d, phys_d, dim_R)
        
        # Mapping for einsum:
        # L:  (l_bra, mpo_l, l_ket)      -> i, j, k
        # R:  (r_bra, mpo_r, r_ket)      -> l, m, n
        # W1: (mpo_l, mpo_mid, p1_out, p1_in) -> j, o, p, q
        # W2: (mpo_mid, mpo_r, p2_out, p2_in) -> o, m, r, s
        # theta: (l_ket, p1_in, p2_in, r_ket) -> k, q, s, n
        
        # Result: (l_bra, p1_out, p2_out, r_bra) -> i, p, r, l
        
        res = np.einsum('ijk, lmn, jopq, omrs, kqsn -> iprl', 
                        L, R, W1, W2, theta, optimize=True)
        return res.ravel()

    # Linear Operator wrapper
    dim = theta_guess.size
    print(dim)
    H_eff = LinearOperator((dim, dim), matvec=matvec)
    
    # Solve Eigenproblem
    energy, new_theta_flat = eigsh(H_eff, k=1, v0=theta_guess.ravel(), which='SA')
    new_theta = new_theta_flat[:, 0].reshape(dim_L, phys_d, phys_d, dim_R)
    
    return energy[0], new_theta

# --- 5. Main DMRG Loop ---

def run_dmrg_finite(N, max_sweeps=5, max_bond_dim=10):
    mpo = build_heisenberg_mpo(N)
    mps = init_mps(N, max_bond_dim)
    
    # Initialize Environment Storage
    # L_env[i] stores contraction of sites 0..i-1
    # R_env[i] stores contraction of sites i+1..N-1
    L_env = [np.array([[[1.]]])] * (N+1) # Boundary is scalar 1
    R_env = [np.array([[[1.]]])] * (N+1)
    
    # Pre-calculate Right Environments (Initial Sweep Right-to-Left essentially)
    print("Initializing Environments...")
    for i in range(N-1, 0, -1):
        R_env[i] = contract_env_R(R_env[i+1], mps[i], mpo[i], mps[i].conj())

    print(f"{'Sweep':<10} {'Energy':<20} {'Max Bond Dim'}")
    
    for sweep in range(max_sweeps):
        # --- Right Sweep (0 -> N-2) ---
        for i in range(N-1):
            # 1. Form Theta (Two-site tensor)
            # A[i] (L, p, mid) + A[i+1] (mid, p, R) -> Theta (L, p, p, R)
            theta = np.tensordot(mps[i], mps[i+1], axes=(2, 0))
            
            # 2. Optimize Theta
            E, theta = local_update(L_env[i], R_env[i+2], mpo[i], mpo[i+1], theta)
            
            # 3. SVD / Split Theta -> New A[i], New A[i+1]
            # Reshape theta for SVD: (L_bond * p) x (p * R_bond)
            dL, p1, p2, dR = theta.shape
            theta_matrix = theta.reshape(dL*p1, p2*dR)
            
            U, S, Vh = np.linalg.svd(theta_matrix, full_matrices=False)
            
            # Truncate
            keep = min(max_bond_dim, len(S))
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]
            
            # 4. Update MPS tensors
            # Keep orthogonality center on the RIGHT for the right sweep
            mps[i] = U.reshape(dL, p1, keep)
            
            # Absorb singular values into the next site
            mps[i+1] = np.dot(np.diag(S), Vh).reshape(keep, p2, dR)
            
            # 5. Update Left Environment
            L_env[i+1] = contract_env(L_env[i], mps[i], mpo[i], mps[i].conj())

        # --- Left Sweep (N-2 -> 0) ---
        for i in range(N-2, -1, -1):
            # Same logic, but move orthogonality to the LEFT
            theta = np.tensordot(mps[i], mps[i+1], axes=(2, 0))
            E, theta = local_update(L_env[i], R_env[i+2], mpo[i], mpo[i+1], theta)
            
            dL, p1, p2, dR = theta.shape
            theta_matrix = theta.reshape(dL*p1, p2*dR)
            U, S, Vh = np.linalg.svd(theta_matrix, full_matrices=False)
            
            keep = min(max_bond_dim, len(S))
            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]
            
            # Update MPS: Absorb S into LEFT site this time
            mps[i] = np.dot(U, np.diag(S)).reshape(dL, p1, keep)
            mps[i+1] = Vh.reshape(keep, p2, dR)
            
            # Update Right Environment
            R_env[i+1] = contract_env_R(R_env[i+2], mps[i+1], mpo[i+1], mps[i+1].conj())
            
        print(f"{sweep+1:<10} {E:<20.8f} {max_bond_dim}")


run_dmrg_finite(N=12, max_sweeps=4, max_bond_dim=20)