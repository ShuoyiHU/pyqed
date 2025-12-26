import numpy as np
import scipy.linalg as la
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
# Import the new analytical integrals
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_linear import (
    make_xy_spd_primitive_basis,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    V_en_analytical_linear,
    eri_2d_analytical_linear,
    build_h1_nm_analytical
)
# Attempt to import NewtonHelper from the package structure used in original files
try:
    from pyqed.qchem.dvr.newton_helper import CollocatedERIOp, NewtonHelper
except ImportError:
    # Fallback or placeholder if running in an environment without the full package
    print("Warning: pyqed.qchem.dvr.newton_helper not found. Newton method will fail if dependencies are missing.")
    CollocatedERIOp = None
    NewtonHelper = None

# ------------------------
#  Basic molecule holder
# ------------------------
class Molecule:
    def __init__(self, charges, coords, nelec=None):
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        if nelec is None:
            self.nelec = int(round(float(np.sum(self.charges))))
        else:
            self.nelec = int(nelec)

    def to_tuples(self):
        return [(float(Z), float(x), float(y), float(z))
                for (Z, (x, y, z)) in zip(self.charges, self.coords)]
    
    def nuclear_repulsion_energy(self):
        E = 0.0
        Z = self.charges
        R = self.coords
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                dR = R[i] - R[j]
                E += Z[i] * Z[j] / float(np.linalg.norm(dR))
        return E

# -----------------------------
#  DVR 1D
# -----------------------------
def sine_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    j = np.arange(1, N + 1)
    z = zmin + j * L / (N + 1)
    n = np.arange(1, N + 1)
    U = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * np.outer(j, n) / (N + 1))
    lam = 0.5 * (np.pi * n / L) ** 2       
    Tz = (U * lam) @ U.T                   
    dz = L / (N + 1)
    return z, Tz, dz

def Exponential_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    dz = L / N
    z = zmin + np.arange(N) * dz
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    arg = np.pi * n_diff / N
    Tz = np.zeros((N, N))
    mask = n_diff != 0
    sign = (-1.0)**n_diff[mask]
    sin_sq = np.sin(arg[mask])**2
    if N % 2 == 0:
        Tz[mask] = 2.0 * sign / sin_sq
        diag_val = (N**2 + 2.0) / 3.0
    else:
        cos_val = np.cos(arg[mask])
        Tz[mask] = 2.0 * sign * cos_val / sin_sq
        diag_val = (N**2 - 1.0) / 3.0
    np.fill_diagonal(Tz, diag_val)
    Tz *= 0.5 * (np.pi / L)**2
    return z, Tz, dz

def sinc_dvr_1d(zmin, zmax, N):
    L = zmax - zmin
    dz = L / (N + 1) 
    j = np.arange(1, N + 1)
    z = zmin + j * dz
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    Tz = np.zeros((N, N))
    diag_val = np.pi**2 / (6.0 * dz**2)
    np.fill_diagonal(Tz, diag_val)
    mask = n_diff != 0
    sign = (-1.0)**n_diff[mask]
    denom = (n_diff[mask] * dz)**2
    Tz[mask] = sign / denom
    return z, Tz, dz

# ---------------------------------------------------------
#  Slicing and Orthogonalization
# ---------------------------------------------------------
def _s_orthonormalizer(S, eps_rel=1e-12):
    S = 0.5 * (S + S.T)                
    w, U = la.eigh(S)
    if w.size == 0: raise ValueError("Empty overlap.")
    wmax = float(np.max(w))
    keep = w > (eps_rel * wmax)
    r = int(np.count_nonzero(keep))
    X = U[:, keep] / np.sqrt(w[keep])
    return X, r, w

def slice_eigens_xy(z_grid, S_prim, T_prim, V_en_func, M=1):
    Nz = len(z_grid)
    X, r, wS = _s_orthonormalizer(S_prim, eps_rel=1e-12)
    if M > r: M = r

    E_slices = np.zeros((Nz, M), float)
    C_list   = [np.zeros((S_prim.shape[0], M), float) for _ in range(Nz)]

    for k, zk in enumerate(z_grid):
        Vz = V_en_func(float(zk))        
        Hk = T_prim + Vz                 
        Hk_ = X.T @ Hk @ X               
        if M == 1:
            w, V_ = la.eigh(Hk_, subset_by_index=[0, 0])
            Vk = V_[:, :1]
        else:
            w, V_ = la.eigh(Hk_, subset_by_index=[0, M-1])
            Vk = V_[:, :M]
        Uk = X @ Vk
        E_slices[k, :len(w[:M])] = w[:M]
        C_list[k] = Uk

    return E_slices, C_list

def _psd_project_small(M):
    M = 0.5 * (M + M.T)
    w, V = la.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T

# ----------------------------------------------------
#  Precompute ERI (Analytical)
# ----------------------------------------------------
def precompute_eri_analytical_JK_psd(
    alphas, centers, labels, z_grid, C_list, M,
    max_offset=None, auto_cut=False, cut_eps=1e-8, verbose=True
):
    Nz = z_grid.size
    if Nz > 1: dz = float(abs(z_grid[1] - z_grid[0]))
    else: dz = 0.0

    if max_offset is None: max_offset = Nz - 1

    ERI_J = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]

    eri_by_h = {}
    norm0 = None
    h_max = max_offset
    
    t1 = time.time()
    for h in range(0, max_offset + 1):
        delta_z = float(h * dz)
        
        # CALL NEW ANALYTICAL INTEGRAL
        # Note: 'centers' are ignored inside, assumed 0.0
        eri_ao = eri_2d_analytical_linear(alphas, centers, labels, delta_z)

        eri_by_h[h] = eri_ao

        if auto_cut:
            nh = float(np.linalg.norm(eri_ao.reshape(-1)))
            if h == 0: norm0 = max(nh, 1e-16)
            elif norm0 and (nh / norm0 < cut_eps):
                h_max = h
                break

    # Contraction to MOs
    for m in range(Nz):
        C_m = np.asarray(C_list[m], float)  
        for n in range(Nz):
            h = abs(n - m)
            if h > h_max: continue

            C_n = np.asarray(C_list[n], float)
            eri_ao = eri_by_h[h]

            Jabcd = np.einsum("pqrs,pa,qb,rc,sd->abcd", eri_ao, C_m, C_m, C_n, C_n, optimize=True)  
            EmnJ = _psd_project_small(Jabcd.reshape(M * M, M * M))

            Kabcd = np.einsum("pqrs,pa,rc,qb,sd->abcd", eri_ao, C_m, C_n, C_m, C_n, optimize=True)
            EmnK = 0.5 * (Kabcd.reshape(M * M, M * M) + Kabcd.reshape(M * M, M * M).T)

            ERI_J[m][n] = EmnJ
            ERI_K[m][n] = EmnK

    if verbose:
        print(f"[Analytical-ERI] Built J/K blocks (M={M}, h_max={h_max}) in {time.time()-t1:.2f}s")
    return ERI_J, ERI_K

def fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=1.0):
    N = Nz * M
    P4 = P.reshape(Nz, M, Nz, M)
    Ddiag = [P4[b, :, b, :].copy() for b in range(Nz)] 

    F2e = np.zeros((N, N), dtype=float)

    for m in range(Nz):
        J_mm = np.zeros((M, M), float)
        for n in range(Nz):
            EmnJ = ERI_J[m][n]
            # Ensure EmnJ is 2D array
            if np.isscalar(EmnJ) or np.array(EmnJ).ndim == 0:
                 EmnJ = np.array(EmnJ).reshape(M, M)
            jvec = EmnJ @ Ddiag[n].reshape(M * M)
            J_mm += jvec.reshape(M, M)
        im0 = m * M
        F2e[im0:im0+M, im0:im0+M] += J_mm

    for m in range(Nz):
        im0, im1 = m*M, (m+1)*M
        for n in range(Nz):
            in0, in1 = n*M, (n+1)*M
            EmnK = ERI_K[m][n]
            if np.isscalar(EmnK) or np.array(EmnK).ndim == 0:
                 EmnK = np.array(EmnK).reshape(M, M)
            BK   = P4[m, :, n, :]                   
            kvec = EmnK @ BK.reshape(M * M)
            K_mn = kvec.reshape(M, M)
            F2e[im0:im1, in0:in1] -= 0.5 * k_scale * K_mn
    return F2e

# ----------------------------------------------------
#  Builder Function (Linear Analytical)
# ----------------------------------------------------

def build_linear_analytical(
    mol: Molecule,
    Lz=18.0, Nz=121, M=1,
    s_exps=None, p_exps=None, d_exps=None,
    verbose=True, dvr_method='sine'
):
    """
    Method-II builder utilizing exact analytical integrals for linear chains.
    """
    t0 = time.time()
    if dvr_method == 'sine':
        z, Kz, dz = sine_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'exp':
        z, Kz, dz = Exponential_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'sinc':
        z, Kz, dz = sinc_dvr_1d(-Lz, Lz, Nz)
    
    nuclei = mol.to_tuples()
    
    alphas, centers_raw, labels = make_xy_spd_primitive_basis(
        nuclei, s_exps, p_exps, d_exps
    )
    # Force centers to zero for analytical mode
    centers = np.zeros_like(centers_raw) 

    if verbose:
        print(f"[Linear Analytical] Basis size: {len(alphas)}")
        print(f"[Linear Analytical] Forced all primitive centers to (0,0).")

    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)

    def V_en_func(zk: float):
        return V_en_analytical_linear(alphas, centers, labels, nuclei, zk)

    E_slices, C_list = slice_eigens_xy(z, S_prim, T_prim, V_en_func, M=M)

    size = Nz * M
    S_slice = np.zeros((Nz, Nz, M, M), float)
    for k in range(Nz):
        Ck = C_list[k]
        for m in range(Nz):
            Cm = C_list[m]
            S_slice[k, m] = Ck.T @ (S_prim @ Cm)

    Hcore = np.einsum('km,kmab->kamb', Kz, S_slice, optimize=True).reshape(size, size)
    Hcore += np.diag(E_slices.reshape(-1))

    ERI_J, ERI_K = precompute_eri_analytical_JK_psd(
        alphas, centers, labels, z, C_list, M,
        max_offset=None, auto_cut=True, cut_eps=1e-9, verbose=verbose
    )

    return Hcore, z, dz, ERI_J, ERI_K, E_slices, C_list, S_prim, T_prim, alphas, centers, labels

# ----------------------------------------------------
#  SCF & DIIS
# ----------------------------------------------------

class PulayCDIIS:
    def __init__(self, space=6, reg=1e-12):
        self.space = int(space)
        self.reg   = float(reg)
        self.F_hist = []   
        self.E_hist = []   

    def push(self, F, P):
        E = F @ P - P @ F
        self.F_hist.append(F.copy())
        self.E_hist.append(E.ravel().copy())
        if len(self.F_hist) > self.space:
            self.F_hist = self.F_hist[-self.space:]
            self.E_hist = self.E_hist[-self.space:]

    def mix(self):
        m = len(self.F_hist)
        if m < 2: return self.F_hist[-1]
        R = np.vstack(self.E_hist)            
        G = R @ R.T                           
        B = np.empty((m + 1, m + 1), dtype=float)
        B[:m, :m] = G
        B[:m,  m] = -1.0
        B[ m, :m] = -1.0
        B[ m,  m] =  0.0
        rhs = np.zeros(m + 1, dtype=float)
        rhs[m] = -1.0
        B[:m, :m] += self.reg * np.eye(m)
        try:
            coeff = la.solve(B, rhs)[:m]
        except la.LinAlgError:
            return self.F_hist[-1]
        Fmix = np.zeros_like(self.F_hist[0])
        for c, Fi in zip(coeff, self.F_hist):
            Fmix += c * Fi
        return Fmix

def scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz, M, nelec, Enuc=0.0,
                    conv=1e-7, max_iter=50, verbose=True,
                    damp=0.20, diis_start=3, diis_space=8,
                    level_shift=0.5, shift_decay=0.75, k_ramp_iters=8):
    N = Nz * M
    nocc = nelec // 2
    I = np.eye(N)

    eps, Cmo = la.eigh(Hcore)
    Cocc = Cmo[:, :nocc]
    P = 2.0 * (Cocc @ Cocc.T)

    E_last = np.inf
    P_prev = P.copy()
    P_prev2 = None

    diis = PulayCDIIS(space=diis_space)
    beta = float(level_shift)

    def total_energy(Puse, Fuse):
        E_el = 0.5 * np.sum((Hcore + Fuse) * Puse)
        return E_el + Enuc

    for it in range(1, max_iter + 1):
        t0 = time.time()
        k_scale = 1.0 if k_ramp_iters <= 1 else min(1.0, it / float(k_ramp_iters))

        F2e = fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=k_scale)
        F = Hcore + F2e

        diis.push(F, P)
        if it >= diis_start:
            F = diis.mix()
        if beta > 0.0:
            Q = I - 0.5 * P
            F = F + beta * Q

        eps, Cmo = la.eigh(F)
        Cocc = Cmo[:, :nocc]
        P_new = 2.0 * (Cocc @ Cocc.T)

        if damp > 0.0:
            P_new = (1.0 - damp) * P_new + damp * P

        if P_prev2 is not None:
            d2 = la.norm(P_new - P_prev2, ord='fro')
            d1 = la.norm(P_new - P_prev,  ord='fro')
            if d2 < 1e-6 and d1 > 5e-5:
                P_new = 0.5 * (P_new + P_prev)

        Etot = total_energy(P_new, F)
        dE = abs(Etot - E_last)

        R = F @ P_new - P_new @ F
        rnorm = la.norm(R, ord='fro')

        if verbose:
            print(f"SCF {it:3d}: E = {Etot: .10f}  dE={dE:.2e}  ||[F,P]||={rnorm:.2e}")

        if dE < conv and rnorm < 1e-5:
            P = P_new
            E_last = Etot
            break

        if it > 1 and rnorm > 1.5 * la.norm(F @ P - P @ F, ord='fro'):
            damp = min(0.5, 1.25 * damp + 0.02)
            beta = min(2.0, beta * 1.25)
        else:
            beta = max(0.0, beta * shift_decay)

        P_prev2 = P_prev
        P_prev  = P
        P       = P_new
        E_last  = Etot

    info = {"iter": it, "dE": dE, "rnorm": rnorm}
    return Etot, eps, Cmo, P, info

# ----------------------------------------------------
#  Newton Helper Functions
# ----------------------------------------------------

def s_norm(v, S):
    return float(np.sqrt(max(1e-32, v.T @ (S @ v))))

def bound_step_S(delta_dict, active, S, max_norm):
    out = {}
    scale = 1.0
    for n in active:
        nrm = s_norm(delta_dict[n], S)
        if nrm > max_norm:
            scale = min(scale, max_norm / nrm)
    if scale < 1.0:
        for n in active:
            out[n] = delta_dict[n] * scale
        return out, scale
    else:
        return delta_dict, 1.0

def g_dot_delta(g_full, delta_dict, active):
    val = 0.0
    for n in active:
        val += float(np.dot(g_full[n].ravel(), delta_dict[n].ravel()))
    return val

def select_active_slices(nh, d_stack, P_slice, mode="topk_grad", topk=1):
    Nz = d_stack.shape[0]
    if mode == "center":
        return [Nz // 2]
    elif mode == "topk_grad":
        g = nh.gradient(d_stack, P_slice)
        norms = np.linalg.norm(g, axis=1)
        idx = np.argsort(norms)[::-1][:max(1, topk)]
        return sorted(idx.tolist())
    else:
        raise ValueError("mode not supported. use 'center' or 'topk_grad'.")

def eri_JK_from_kernels_M1(C_list, K_h, Kx_h):
    Nz = len(C_list)
    d = [C_list[m][:, 0] for m in range(Nz)]
    v_mm = [np.kron(dm, dm) for dm in d]

    ERI_J = [[np.zeros((1,1)) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((1,1)) for _ in range(Nz)] for _ in range(Nz)]

    for h in range(Nz):
        Nh = Nz - h
        if Nh <= 0: break
        V_right = np.column_stack([v_mm[nn] for nn in range(h, Nz)])  
        WJ = K_h[h] @ V_right
        for m in range(Nh):
            nn = m + h
            val_J = float(v_mm[m].T @ WJ[:, m])
            ERI_J[m][nn] = np.array([[val_J]])
            ERI_J[nn][m] = ERI_J[m][nn]
            v_mn = np.kron(d[m], d[nn])
            w_mn = Kx_h[h] @ v_mn
            val_K = float(v_mn.T @ w_mn)
            ERI_K[m][nn] = np.array([[val_K]])
            ERI_K[nn][m] = ERI_K[m][nn]

    for m in range(Nz):
        for n in range(Nz):
            if ERI_J[m][n][0,0] < 0.0:
                ERI_J[m][n][0,0] = 0.0
    return ERI_J, ERI_K

def rebuild_Hcore_from_d(
    d_stack, z, Kz, S_prim, T_prim,
    alphas, centers, labels, nuclei,
):
    Nz, N = d_stack.shape

    S_scalar = np.zeros((Nz, Nz), float)
    for n in range(Nz):
        dn = d_stack[n]
        S_scalar[n, n] = float(dn.T @ (S_prim @ dn))
        for m in range(n + 1, Nz):
            dm = d_stack[m]
            val = float(dn.T @ (S_prim @ dm))
            S_scalar[n, m] = val
            S_scalar[m, n] = val

    e_local = np.zeros(Nz, float)
    for n in range(Nz):
        # Using Analytical V_en
        Vz = V_en_analytical_linear(alphas, centers, labels, nuclei, float(z[n])) 
        e_local[n] = float(d_stack[n].T @ ((T_prim + Vz) @ d_stack[n]))

    Hcore = (Kz * S_scalar).astype(float)
    Hcore += np.diag(e_local)
    return Hcore

def evaluate_trial_step(step, d_stack, C_list, active,
                        S_prim, z, Kz, T_prim, alphas, centers, labels, nuclei,
                        Nz, M, nelec, Enuc, K_h, Kx_h,
                        SHORT_SCF_MAXITER, delta_dict):
    trial_d = d_stack.copy()
    NewtonHelper.update_inplace(trial_d, delta_dict, S_prim, active, step=step)

    trial_C_list = [C_list[n].copy() for n in range(Nz)]
    for n in active:
        trial_C_list[n][:, 0] = trial_d[n]

    Hcore_try = rebuild_Hcore_from_d(
        trial_d, z, Kz, S_prim, T_prim,
        alphas, centers, labels, nuclei
    )

    ERI_J_try, ERI_K_try = eri_JK_from_kernels_M1(trial_C_list, K_h, Kx_h)

    Etot_try, _, _, P_try, _ = scf_rhf_method2(
        Hcore_try, ERI_J_try, ERI_K_try, Nz, M,
        nelec=nelec, Enuc=Enuc,
        conv=3e-7, max_iter=SHORT_SCF_MAXITER, verbose=False
    )
    return Etot_try, trial_d, trial_C_list, P_try

if __name__ == "__main__":
    # Test on H4 linear chain
    coords = np.array([
        [0.0, 0.0, -3.6],
        [0.0, 0.0, -0.91],
        [0.0, 0.0,  0.91],
        [0.0, 0.0,  3.6]
    ])
    charges = [1.0, 1.0, 1.0, 1.0]
    mol = Molecule(charges, coords)
    NELEC = mol.nelec
    Enuc = mol.nuclear_repulsion_energy()
    
    # STO-3G for Hydrogen (s-type only for analytical check)
    s_exps = np.array([3.425250016, 0.6239134896, 0.1688561568])
    p_exps = []
    d_exps = []

    print("=== Testing Linear Analytical Hybrid Gauss-DVR with Newton ===")
    
    Nz_val = 31  # Define Nz here to avoid hardcoding

    # 1. Build Initial Basis and SCF
    (Hcore, z, dz, ERI_J, ERI_K, 
     E_slices, C_list, S_prim, T_prim, 
     alphas, centers, labels) = build_linear_analytical(
        mol, Lz=8.0, Nz=Nz_val, M=1,
        s_exps=s_exps, p_exps=p_exps, d_exps=d_exps
    )
    
    print(f"[Debug] Calculated Nuclear Repulsion: {Enuc:.6f} Eh")

    # Initial SCF
    Etot, eps, Cmo, P, info = scf_rhf_method2(
        Hcore, ERI_J, ERI_K, Nz_val, 1, NELEC, Enuc=Enuc,
        max_iter=100, verbose=True
    )
    print(f"[SCF 0] E = {Etot:.8f} Eh")
    E_history = [Etot]

    # 2. Setup Newton Helper (Requires K_h and Kx_h Kernels)
    #    We must generate these kernels analytically.
    if NewtonHelper is not None:
        print("\n[Newton] Setting up Newton Helper with Analytical Kernels...")
        n_ao = len(alphas)
        K_h = []
        Kx_h = []

        # Re-generate kernels for all h separations
        for h in range(Nz_val):
            dz_val = float(h * dz)
            eri_tensor = eri_2d_analytical_linear(alphas, centers, labels, delta_z=dz_val)
            
            K_mat = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
            K_h.append(K_mat)
            
            eri_perm = eri_tensor.transpose(0, 2, 1, 3)
            Kx_mat = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
            Kx_h.append(Kx_mat)

        # Build h1_nm operator for Newton
        # Need z_grid and a closure for V_en
        _, Kz_mat, _ = sine_dvr_1d(-8.0, 8.0, Nz_val)
        def V_en_func_newton(zk): 
            return V_en_analytical_linear(alphas, centers, labels, mol.to_tuples(), zk)
        
        h1_nm = build_h1_nm_analytical(Kz_mat, S_prim, T_prim, z, V_en_func_newton)

        ERIop = CollocatedERIOp.from_kernels(
            N=S_prim.shape[0], Nz=Nz_val, dz=dz, K_h=K_h, Kx_h=Kx_h
        )
        nh = NewtonHelper(h1_nm, S_prim, ERIop)
        
        # Prepare d_stack (normalized coefficients)
        d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz_val)])
        for n in range(Nz_val):
            dn = d_stack[n]
            d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))
        
        # 3. Newton Alternation Loop
        ALT_CYCLES = 5
        for cyc in range(1, ALT_CYCLES + 1):
            P_slice = P.reshape(Nz_val, 1, Nz_val, 1)[:, 0, :, 0].copy()
            active = select_active_slices(nh, d_stack, P_slice, mode="topk_grad", topk=Nz_val)
            
            # Newton Step
            g_full = nh.gradient(d_stack, P_slice)
            delta_dict, lam, info_kkt = nh.kkt_step(d_stack, P_slice, S_prim, active=active, ridge=0.0)
            
            # Ensure descent direction
            if g_dot_delta(g_full, delta_dict, active) > 0.0:
                for n in active: delta_dict[n] *= -1.0
            
            # Trust region scaling
            delta_dict, scaled = bound_step_S(delta_dict, active, S_prim, 0.10)
            
            # Line Search
            STEP_LIST = [100, 50, 20, 10, 5, 2, 1.0, 0.5, 0.1]
            best = (np.inf, None, None, None, None)
            
            for step in STEP_LIST:
                E_try, d_try, C_try, P_try = evaluate_trial_step(
                    step, d_stack, C_list, active,
                    S_prim, z, Kz_mat, T_prim, alphas, centers, labels, mol.to_tuples(),
                    Nz_val, 1, NELEC, Enuc, K_h, Kx_h, 40, delta_dict
                )
                if E_try < best[0]:
                    best = (E_try, step, d_try, C_try, P_try)
            
            # Update State
            if best[0] < E_history[-1] - 1e-8:
                d_stack, C_list, Etot, P = best[2], best[3], best[0], best[4]
                print(f"[Cycle {cyc}] Newton Success: Step={best[1]} E={Etot:.8f}")
            else:
                print(f"[Cycle {cyc}] Newton: No improvement.")

            # Re-run full SCF to tighten
            Hcore = rebuild_Hcore_from_d(d_stack, z, Kz_mat, S_prim, T_prim, alphas, centers, labels, mol.to_tuples())
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
            Etot, _, _, P, _ = scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz_val, 1, NELEC, Enuc, max_iter=50, verbose=False)
            E_history.append(Etot)
            print(f"[Cycle {cyc}] Post-SCF E={Etot:.8f}")

    print(f"Final Energy: {E_history[-1]:.8f} Eh")