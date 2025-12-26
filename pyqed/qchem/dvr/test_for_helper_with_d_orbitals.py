import numpy as np
import matplotlib.pyplot as plt
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals import (
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    _V_en_prony_general,
    eri_2d_cartesian_with_p,
    make_xy_spd_primitive_basis,
    PrimitiveLabel
)

# =================================================================
#  1. Grid & Basis Evaluation Helpers
# =================================================================

def eval_gto_on_grid(alpha, center, label, X, Y):
    """
    Evaluate a single primitive GTO on a 2D grid (X, Y).
    """
    xc, yc = center
    dx = X - xc
    dy = Y - yc
    r2 = dx**2 + dy**2
    
    # parse label
    if isinstance(label, dict):
        kind = label['kind']
    else:
        kind = label.kind

    pre = np.exp(-alpha * r2)

    if kind == '2d-s':   return pre
    if kind == '2d-px':  return dx * pre
    if kind == '2d-py':  return dy * pre
    if kind == '2d-dx2': return (dx**2) * pre
    if kind == '2d-dy2': return (dy**2) * pre
    if kind == '2d-dxy': return (dx*dy) * pre
    return np.zeros_like(pre)

def compute_laplacian_fd(Z, dx, dy):
    """
    Compute 2D Laplacian using 5-point Finite Difference stencil.
    """
    # Z shape (Ny, Nx)
    lap = np.zeros_like(Z)
    
    # Central difference: f(x+h) - 2f(x) + f(x-h)
    d2x = (Z[1:-1, 2:] - 2*Z[1:-1, 1:-1] + Z[1:-1, :-2]) / (dx**2)
    d2y = (Z[2:, 1:-1] - 2*Z[1:-1, 1:-1] + Z[:-2, 1:-1]) / (dy**2)
    
    # Combine on interior points
    lap[1:-1, 1:-1] = d2x + d2y
    return lap

def numerical_eri_gaussian_kernel(grid_x, grid_y, phi_a, phi_b, phi_c, phi_d, gamma_val):
    """
    Numerically integrate <ab | exp(-gamma r12^2) | cd>.
    Uses a simplified separable approach:
    I = Integral[ rho_AB(r1) * Integral[ exp(-gamma(r1-r2)^2) * rho_CD(r2) dr2 ] dr1 ]
    This is an O(N^4) grid operation, so we keep grids moderate.
    """
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    dA = dx * dy
    
    X, Y = np.meshgrid(grid_x, grid_y)
    rho_AB = phi_a * phi_b
    rho_CD = phi_c * phi_d
    
    # This double integral is a convolution.
    # To verify correctness without massive cost, let's implement the convolution
    # explicitly but optimize.
    # Or simpler: If gamma is small, the kernel is broad.
    # Let's just do the raw sum.
    
    # Flatten
    r_AB = rho_AB.ravel()
    r_CD = rho_CD.ravel()
    coords = np.column_stack((X.ravel(), Y.ravel()))
    N = len(r_AB)
    
    # Compute kernel matrix K_uv = exp(-gamma |r_u - r_v|^2)
    # This will be N^2. If grid is 50x50, N=2500, N^2 = 6e6 (fine).
    
    # Distance matrix squared
    # (x_u - x_v)^2 + (y_u - y_v)^2
    # Use broadcasting
    # x col: (N,1), x row: (1,N)
    x_vec = coords[:, 0]
    y_vec = coords[:, 1]
    
    dx_mat = x_vec[:, None] - x_vec[None, :]
    dy_mat = y_vec[:, None] - y_vec[None, :]
    dist_sq = dx_mat**2 + dy_mat**2
    
    K = np.exp(-gamma_val * dist_sq)
    
    # Integral ~ sum_u sum_v r_AB[u] * K[u,v] * r_CD[v] * dA * dA
    
    # First contraction: V_pot[u] = sum_v K[u,v] * r_CD[v] * dA
    V_pot = K @ (r_CD * dA)
    
    # Second contraction: sum_u r_AB[u] * V_pot[u] * dA
    val = np.dot(r_AB, V_pot) * dA
    
    return val

# =================================================================
#  2. Main Test Runner
# =================================================================

def run_tests():
    print("="*60)
    print("  NUMERICAL FINITE-DIFFERENCE / QUADRATURE TEST (S, P, D)")
    print("="*60)
    
    # -----------------------------------------
    # Setup System: 2 Centers, s, p, d orbitals
    # -----------------------------------------
    # Place centers slightly off-grid to avoid divide-by-zero artifacts (though Gaussians are safe)
    c1 = np.array([0.1, 0.2])
    c2 = np.array([1.2, -0.5]) # Distance ~ 1.5
    
    # Exponents
    a1 = 1.5
    a2 = 0.8
    
    # Define a mixed basis set manually
    # 0: s on c1
    # 1: px on c1
    # 2: dx2 on c1
    # 3: s on c2
    # 4: dxy on c2
    
    alphas = np.array([a1, a1, a1, a2, a2])
    centers = np.array([c1, c1, c1, c2, c2])
    labels = [
        PrimitiveLabel('2d-s', 2, (0,0,0)),
        PrimitiveLabel('2d-px', 2, (1,0,0)),
        PrimitiveLabel('2d-dx2', 2, (2,0,0)), # d orbital test
        PrimitiveLabel('2d-s', 2, (0,0,0)),
        PrimitiveLabel('2d-dxy', 2, (1,1,0))  # d orbital test
    ]
    
    # -----------------------------------------
    # Setup Grid for Numerical Integration
    # -----------------------------------------
    # Need high resolution for Kinetic Energy FD stability
    L = 7.0 # Box size from -L to L
    Npts = 120 
    x_grid = np.linspace(-4.0, 5.0, Npts) # Cover both centers comfortably
    y_grid = np.linspace(-4.0, 4.0, Npts)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    X, Y = np.meshgrid(x_grid, y_grid)
    
    print(f"Grid: {Npts}x{Npts}, dx={dx:.4f}, dy={dy:.4f}")
    print("Evaluating Basis on Grid...")
    
    # Eval all basis functions
    phis = []
    for i in range(len(labels)):
        phis.append(eval_gto_on_grid(alphas[i], centers[i], labels[i], X, Y))
    phis = np.array(phis) # (N_basis, Ny, Nx)
    
    # =================================================================
    # Test 1: OVERLAP (S)
    # =================================================================
    print("\n[Test 1] Overlap Integrals (S)")
    S_ana = overlap_2d_cartesian(alphas, centers, labels)
    
    # Numerical S_ij = sum phi_i * phi_j * dA
    max_err_S = 0.0
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            S_num = np.sum(phis[i] * phis[j]) * dx * dy
            err = abs(S_num - S_ana[i,j])
            max_err_S = max(max_err_S, err)
            label_str = f"<{labels[i].kind}|{labels[j].kind}>"
            print(f"  {label_str:15s}: Ana={S_ana[i,j]:.6f}  Num={S_num:.6f}  Diff={err:.2e}")
            
    if max_err_S < 1e-4: print(">> OVERLAP TEST PASSED")
    else: print(">> OVERLAP TEST FAILED / LOW PRECISION")

    # =================================================================
    # Test 2: KINETIC ENERGY (T)
    # T = <i | -0.5 Laplacian | j>
    # =================================================================
    print("\n[Test 2] Kinetic Energy (T) via FD Laplacian")
    T_ana = kinetic_2d_cartesian(alphas, centers, labels)
    
    max_err_T = 0.0
    # Precompute laplacians
    laps = []
    for i in range(len(labels)):
        laps.append(compute_laplacian_fd(phis[i], dx, dy))
        
    # Inner region mask to avoid boundary errors in FD
    # (Not strictly necessary if box is large enough and func -> 0)
    
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            # T_num = sum phi_i * (-0.5 * lap_j)
            # Symmetrize numerical error: 0.5 * (<i|T|j> + <j|T|i>)
            val1 = np.sum(phis[i] * (-0.5 * laps[j])) * dx * dy
            val2 = np.sum(phis[j] * (-0.5 * laps[i])) * dx * dy
            T_num = 0.5 * (val1 + val2)
            
            err = abs(T_num - T_ana[i,j])
            max_err_T = max(max_err_T, err)
            label_str = f"<{labels[i].kind}|T|{labels[j].kind}>"
            print(f"  {label_str:15s}: Ana={T_ana[i,j]:.6f}  Num={T_num:.6f}  Diff={err:.2e}")

    if max_err_T < 5e-3: print(">> KINETIC TEST PASSED (FD Error Accepted)")
    else: print(">> KINETIC TEST FAILED (Check Resolution?)")

    # =================================================================
    # Test 3: NUCLEAR ATTRACTION (V_en) - Prony Branch
    # Use dz > 0 to trigger generalized Prony logic
    # =================================================================
    print("\n[Test 3] Nuclear Attraction (V_en) - Prony Logic")
    dz_val = 1.0
    nuc_pos = np.array([0.5, 0.0]) # Arbitrary nucleus position
    
    # Analytical
    # Note: the helper function returns matrix for *all* nuclei sum.
    # We hack it to just return the matrix for one nucleus interaction (Z=1)
    # Actually `_V_en_prony_general` returns the raw interaction matrix (positive).
    V_ana = _V_en_prony_general(alphas, centers, labels, nuc_pos, dz_val)
    
    # Numerical Potential
    # V(r) = exp( ... ) in Prony expansion
    # Wait, the analytical function computes Sum( weight_k * Integral( ... ) )
    # We should verify the full sum.
    # The potential being integrated is: Sum_k w_k * exp( -gamma_k * (r-N)^2 )
    # where gamma_k = xi / dz^2, w_k = eta / dz
    
    # Construct numerical potential grid
    from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals import ETAS, XIS
    invz = 1.0 / dz_val
    V_grid = np.zeros_like(X)
    
    # Build Prony Potential on Grid
    for eta, xi in zip(ETAS, XIS):
        gam_p = xi * (invz**2)
        w_p = eta * invz
        
        # exp(-gam_p * |r - nuc|^2)
        dist2 = (X - nuc_pos[0])**2 + (Y - nuc_pos[1])**2
        V_grid += w_p * np.exp(-gam_p * dist2)
        
    # Note: This V_grid approximates 1/sqrt(r^2 + z^2).
    # Let's compare integrals.
    
    max_err_V = 0.0
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            V_num = np.sum(phis[i] * V_grid * phis[j]) * dx * dy
            err = abs(V_num - V_ana[i,j])
            max_err_V = max(max_err_V, err)
            label_str = f"<{labels[i].kind}|V|{labels[j].kind}>"
            print(f"  {label_str:15s}: Ana={V_ana[i,j]:.6f}  Num={V_num:.6f}  Diff={err:.2e}")
            
    if max_err_V < 1e-4: print(">> POTENTIAL TEST PASSED")
    else: print(">> POTENTIAL TEST FAILED")

    # =================================================================
    # Test 4: ERI Check (Gaussian Geminal)
    # Check <ii | exp(-g r12^2) | jj>
    # This validates the 4-center contraction logic
    # =================================================================
    print("\n[Test 4] ERI Kernel (Gaussian Geminal)")
    
    # We need to call `eri_2d_cartesian_with_p` with specific parameters
    # to isolate a single Gaussian interaction term.
    # We can temporarily mock ETAS=[1.0], XIS=[gamma_val * dz^2] ?
    # Easier: Just implement the analytical formula for one Gaussian here using the helper's internal logic?
    # No, that defeats the purpose of testing the helper.
    
    # Strategy: The helper computes Sum( w_k * (ii| G_k |jj) ).
    # If we set delta_z such that invz=1, then gamma_k = XIS[k].
    # Let's check the total sum result against numerical integration of the TOTAL prony kernel.
    
    dz_eri = 1.0
    invz_eri = 1.0
    
    # Build 2-body kernel K(r1, r2) on grid is too big (N^4).
    # We use the `numerical_eri_gaussian_kernel` which does the convolution.
    # But we need to sum over all prony terms.
    
    # Pick a specific pair to test to save time (e.g. dx2 on A, dxy on B)
    idx_a = 2 # dx2 on c1
    idx_b = 4 # dxy on c2
    
    # Analytical Result (Standard function call)
    # It returns full tensor.
    eri_tensor_ana = eri_2d_cartesian_with_p(alphas, centers, labels, dz_eri)
    val_ana = eri_tensor_ana[idx_a, idx_a, idx_b, idx_b] # (aa|bb) Coulomb
    
    # Numerical Result
    # Sum over Prony components
    val_num_total = 0.0
    
    # Reduce grid size for convolution speed
    coarse_step = 2
    sx = slice(0, Npts, coarse_step)
    sy = slice(0, Npts, coarse_step)
    X_c = X[sx, sy]; Y_c = Y[sx, sy]
    gx_c = x_grid[::coarse_step]
    gy_c = y_grid[::coarse_step]
    phi_a_c = phis[idx_a][sx, sy]
    phi_b_c = phis[idx_b][sx, sy] # Actually we want (aa|bb) -> density a*a and b*b
    rho_A = phi_a_c * phi_a_c
    rho_B = phi_b_c * phi_b_c
    
    print(f"  Computing numerical ERI convolution on {len(gx_c)}x{len(gy_c)} grid...")
    
    for eta, xi in zip(ETAS, XIS):
        gamma_p = xi * (invz_eri**2)
        weight_p = eta * invz_eri
        
        # Single kernel integral
        # Using simpler specialized convolution function for density-density
        # <rho_A | G | rho_B>
        # Re-using logic from 'numerical_eri_gaussian_kernel' but adapted
        
        # Flatten
        rA = rho_A.ravel()
        rB = rho_B.ravel()
        coords = np.column_stack((X_c.ravel(), Y_c.ravel()))
        
        # Dist matrix (condensed)
        # To save memory, do blocked? Or just N~3600 is fine.
        x_vec = coords[:,0]
        y_vec = coords[:,1]
        d2 = (x_vec[:,None] - x_vec[None,:])**2 + (y_vec[:,None] - y_vec[None,:])**2
        
        K = np.exp(-gamma_p * d2)
        
        dA = (gx_c[1]-gx_c[0]) * (gy_c[1]-gy_c[0])
        term = weight_p * (rA @ K @ rB) * dA * dA
        val_num_total += term
        
    print(f"  ERI <{labels[idx_a].kind}{labels[idx_a].kind} | V | {labels[idx_b].kind}{labels[idx_b].kind}>")
    print(f"  Ana: {val_ana:.6f}")
    print(f"  Num: {val_num_total:.6f}")
    
    err_eri = abs(val_ana - val_num_total)
    if err_eri < 5e-3: 
        print(">> ERI TEST PASSED")
    else:
        print(">> ERI TEST FAILED (Could be grid resolution)")

if __name__ == "__main__":
    run_tests()