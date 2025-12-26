import numpy as np
import math
from numpy import linalg as LA
from scipy.linalg import svd, lstsq, solve

# ==========================================
# PART 1: Your Original Prony Solver
# ==========================================
def fit_t(t, expn, etal):
    """ Reconstruction helper """
    res = np.zeros_like(t, dtype=complex)
    for i in range(len(etal)):
        res += etal[i] * np.exp(-expn[i] * t)
    return res

def prony_decomposition(x, fft_ct, nexp, scale=None):
    """
    Decomposes signal into sum of exponentials using Hankel Matrix method.
    (Code adapted from Zi-Hao Chen)
    """
    # Ensure input is suitable size for Hankel (2N+1 points)
    # If even, drop the last point to make it odd
    if len(x) % 2 == 0:
        x = x[:-1]
        fft_ct = fft_ct[:-1]
        
    n = (len(x)-1)//2
    n_sample = n + 1 
    
    # Construct Hankel Matrix H
    h = np.real(fft_ct)
    H = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        H[i, :] = h[i:n_sample + i]
        
    # Eigen decomposition
    sing_vs, Q = LA.eigh(H)
    
    # Sort and filter roots
    phase_mat = np.diag([np.exp(-1j * np.angle(sing_v) / 2.0) for sing_v in sing_vs])
    vs = np.array([np.abs(sing_v) for sing_v in sing_vs])
    Qp = np.dot(Q, phase_mat)
    
    sort_array = np.argsort(vs)[::-1]
    vs = vs[sort_array]
    Qp = (Qp[:, sort_array])
    
    # Use requested number of exponentials (or max available)
    n_gamma = min(nexp, len(vs))
    
    # Root finding
    roots_poly = Qp[:, n_gamma][::-1]
    gamma = np.roots(roots_poly)
    
    # Sort roots by magnitude and take the stable ones
    gamma_new = gamma[np.argsort(np.abs(gamma))[:n_gamma]]
    
    # Convert roots to decay rates
    # gamma_discrete = exp(-lambda * dx)
    # log(gamma) = -lambda * dx
    # t_real represents log(gamma) scaled to the full interval
    t_real = 2 * n * np.log(gamma_new)
    
    # Solve for amplitudes (Linear Step)
    gamma_m = np.zeros((n_sample * 2 - 1, n_gamma), dtype=complex)
    for i in range(n_gamma):
        for j in range(n_sample * 2 - 1):
            gamma_m[j, i] = gamma_new[i]**j
            
    # Pseudoinverse to find coefficients
    try:
        omega_real = np.dot(LA.inv(np.dot(np.transpose(gamma_m), gamma_m)),
                            np.dot(np.transpose(gamma_m), np.transpose(h)))
    except:
        # Fallback for singular matrix
        omega_real = np.linalg.lstsq(gamma_m, h, rcond=None)[0]

    etal1 = omega_real
    expn1 = -t_real / scale
    
    return etal1, expn1

# ==========================================
# PART 2: The Tensor Fitter (Hybrid)
# ==========================================
def fit_tensor_with_prony_solver(W_tensor, n_modes=4):
    """
    Fits Tensor = sum_k gamma_k^d * A_k * B_k 
    Using Zi-Hao's Prony Solver for the decay rates.
    """
    d_max, n_orb, _, _, _ = W_tensor.shape
    
    # --- STAGE 1: Extract Representative Signal ---
    # We flatten the tensor trajectory to find the "Average Decay Shape"
    W_matrix = W_tensor[1:].reshape(d_max-1, -1)
    
    # SVD to get the principal time component
    # U[:, 0] is the dominant decay curve of the whole system
    U, S, _ = svd(W_matrix, full_matrices=False)
    signal_proxy = U[:, 0] * S[0]
    
    # Make sure signal is positive (standard decay)
    if np.mean(signal_proxy) < 0:
        signal_proxy *= -1

    # --- STAGE 2: Apply Your Prony Code ---
    # Construct grid x for Prony (Must correspond to distance d)
    # d goes from 1 to d_max-1.
    x_grid = np.arange(0, len(signal_proxy), dtype=float) 
    scale_val = x_grid[-1]
    
    # Call your solver
    # We ask for slightly more modes than needed to capture noise, then filter
    print(f"  ...calling Prony solver on proxy signal (len={len(x_grid)})...")
    coeffs, rates = prony_decomposition(x_grid, signal_proxy, nexp=n_modes+2, scale=scale_val)
    
    # Convert Physical Rates (expn) back to Discrete Steps (gamma)
    # Your code: val = coeff * exp(-rate * x)
    # My code:   val = coeff * gamma^d
    # Since x is integer steps here, gamma = exp(-rate)
    
    proposed_gammas = np.exp(-rates)
    
    # Filter for real, stable gammas (0 < g < 1)
    # Prony often returns complex conjugate pairs for oscillating data.
    # For Coulomb 1/r, we only want the pure real decaying parts.
    real_mask = (np.abs(np.imag(proposed_gammas)) < 1e-5) & \
                (np.real(proposed_gammas) > 0.0) & \
                (np.real(proposed_gammas) < 1.0)
                
    valid_gammas = np.real(proposed_gammas[real_mask])
    
    # Sort and truncate to requested n_modes
    valid_gammas = np.sort(valid_gammas)[::-1] # Largest (slowest decay) first
    if len(valid_gammas) > n_modes:
        valid_gammas = valid_gammas[:n_modes]
        
    print(f"  Prony found gammas: {np.round(valid_gammas, 4)}")
    
    # --- STAGE 3: Linear Projection (Tensor Reconstruction) ---
    # Now that we have the exact gammas, we solve for matrices A and B
    
    # Build Time Basis Matrix: [gamma_1^d, gamma_2^d, ...]
    # d corresponds to the slice index (1 to d_max-1)
    # Note: signal_proxy corresponded to W[1:], so time index 0 in proxy is d=1
    d_vals = np.arange(1, d_max)
    Time_Basis = np.zeros((len(d_vals), len(valid_gammas)))
    
    for k, g in enumerate(valid_gammas):
        Time_Basis[:, k] = g ** d_vals

    # Solve: Time_Basis * M_stacked = W_matrix
    # Ridge regularization (alpha) helps if gammas are very close
    alpha = 1e-12
    LHS = Time_Basis.T @ Time_Basis + alpha * np.eye(len(valid_gammas))
    RHS = Time_Basis.T @ W_matrix
    M_stacked = solve(LHS, RHS)
    
    # --- STAGE 4: Decompose Spatial Matrices ---
    fitted_terms = []
    n_sq = n_orb * n_orb
    
    for k, g in enumerate(valid_gammas):
        spatial_map = M_stacked[k].reshape(n_sq, n_sq)
        u_map, s_map, vt_map = svd(spatial_map, full_matrices=False)
        
        # Rank-1 Approx A*B
        A_final = u_map[:, 0].reshape(n_orb, n_orb) * np.sqrt(s_map[0])
        B_final = vt_map[0, :].reshape(n_orb, n_orb) * np.sqrt(s_map[0])
        
        fitted_terms.append({'gamma': g, 'A': A_final, 'B': B_final})
        
    return fitted_terms

# ==========================================
# PART 3: Verification
# ==========================================
def generate_mock_tensor(d_max, n_orb):
    """ Truth = 1/r + 1/r^3 """
    np.random.seed(42)
    A1, B1 = np.random.rand(n_orb, n_orb), np.random.rand(n_orb, n_orb)
    A2, B2 = np.random.randn(n_orb, n_orb)*0.5, np.random.randn(n_orb, n_orb)*0.5
    W = np.zeros((d_max, n_orb, n_orb, n_orb, n_orb))
    for d in range(1, d_max):
        dist = float(d)
        W[d] = (1/dist)*np.einsum('mn,ls->mnls', A1, B1) + \
               (1/dist**3)*np.einsum('mn,ls->mnls', A2, B2)
    return W

if __name__ == "__main__":
    D_MAX = 100    # Prony needs enough points to form the Hankel matrix
    N_ORB = 2
    N_MODES = 30
    
    print("--- Generating Data ---")
    W_truth = generate_mock_tensor(D_MAX, N_ORB)
    
    print("\n--- Running Tensor Fit (Prony-Powered) ---")
    results = fit_tensor_with_prony_solver(W_truth, n_modes=N_MODES)
    
    print("\n--- Validation ---")
    test_d = 5
    W_recon = np.zeros_like(W_truth[test_d])
    for term in results:
        W_recon += (term['gamma']**test_d) * np.einsum('mn,ls->mnls', term['A'], term['B'])
        
    err = np.linalg.norm(W_truth[test_d] - W_recon) / np.linalg.norm(W_truth[test_d])
    print(f"Relative Error at d={test_d}: {err:.4e}")