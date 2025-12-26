import numpy as np
import scipy.linalg as la
from scipy.special import i0e as I0e
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import os

from pyqed.qchem.dvr.newton_helper import CollocatedERIOp, NewtonHelper

from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals import (
    make_xy_spd_primitive_basis,  
    V_en_sp_total_at_z,           
    # _permute_K_ikjl,
    build_h1_nm,
    # PrimitiveLabel,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
)

# ------------------------
#  Basic molecule holder
# ------------------------
class Molecule:
    def __init__(self, charges, coords, nelec=None):
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        assert self.charges.shape[0] == self.coords.shape[0]

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
#  Sine-DVR along z (shared)
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
    """
    Constructs a Periodic (Exponential) DVR grid and Kinetic Energy Matrix.
    
    Equivalent to the ExponentialDVR class, but in a functional form.
    
    Parameters
    ----------
    zmin : float
        Start of the interval.
    zmax : float
        End of the interval.
    N : int
        Number of grid points.

    Returns
    -------
    z : ndarray
        Grid points (N,).
    Tz : ndarray
        Kinetic energy matrix (N, N).
    dz : float
        Grid spacing.
    """
    L = zmax - zmin
    dz = L / N
    
    # Grid generation: 0 to N-1
    # Note: In periodic DVR, the point at zmax is equivalent to zmin, 
    # so we exclude zmax to strictly maintain N unique points.
    z = zmin + np.arange(N) * dz
    
    # ---------------------------------------------------------
    # Kinetic Energy Matrix Construction
    # Based on Colbert & Miller, J. Chem. Phys. 96, 1982 (1992)
    # ---------------------------------------------------------
    
    # 1. Create index difference matrix (i - j)
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    
    # 2. Calculate argument for trig functions
    arg = np.pi * n_diff / N
    
    # 3. Initialize T matrix
    Tz = np.zeros((N, N))
    
    # 4. Handle Off-Diagonal Elements (where n_diff != 0)
    # We use a mask to avoid division by zero on the diagonal
    mask = n_diff != 0
    
    # The prefactor (-1)^(i-j)
    sign = (-1.0)**n_diff[mask]
    sin_sq = np.sin(arg[mask])**2
    
    # Logic derived from ExponentialDVR class source
    if N % 2 == 0:
        # Even case
        Tz[mask] = 2.0 * sign / sin_sq
        diag_val = (N**2 + 2.0) / 3.0
    else:
        # Odd case
        cos_val = np.cos(arg[mask])
        Tz[mask] = 2.0 * sign * cos_val / sin_sq
        diag_val = (N**2 - 1.0) / 3.0

    # 5. Fill Diagonal
    np.fill_diagonal(Tz, diag_val)
    prefactor = 0.5 * (np.pi / L)**2
    Tz *= prefactor
    
    return z, Tz, dz


def sinc_dvr_1d(zmin, zmax, N):
    """
    Constructs a Standard Sinc DVR (Colbert-Miller) grid and Kinetic Energy Matrix.
    
    Standard DVR for the infinite line (vanishing boundary conditions at +/- infinity).
    Often used for scattering or soft potentials (e.g. Harmonic Oscillator).
    
    Ref: Colbert & Miller, J. Chem. Phys. 96, 1982 (1992) Eqs 2.6
    """
    L = zmax - zmin
    # Using same grid density convention as Sine DVR (interior points)
    # so that dz matches typical non-periodic grids
    dz = L / (N + 1) 
    j = np.arange(1, N + 1)
    z = zmin + j * dz
    
    # Matrix Construction
    i = np.arange(N)[:, np.newaxis]
    j = np.arange(N)[np.newaxis, :]
    n_diff = i - j
    
    Tz = np.zeros((N, N))
    
    # Diagonal element: T_ii = (hbar^2 / 2m) * (pi^2 / 3 dx^2)
    # With hbar=1, m=1:
    diag_val = np.pi**2 / (6.0 * dz**2)
    np.fill_diagonal(Tz, diag_val)
    
    # Off-diagonal: T_ij = (hbar^2 / 2m) * (-1)^(i-j) * (2 / (i-j)^2 dx^2)
    # With hbar=1, m=1:
    mask = n_diff != 0
    sign = (-1.0)**n_diff[mask]
    denom = (n_diff[mask] * dz)**2
    
    Tz[mask] = sign / denom
    
    return z, Tz, dz
# ==========================================
#  B) S-metric orthonormalization
# ==========================================
def _s_orthonormalizer(S, eps_rel=1e-12):
    S = 0.5 * (S + S.T)                
    w, U = la.eigh(S)
    if w.size == 0:
        raise ValueError("Empty overlap matrix.")
    wmax = float(np.max(w))
    if wmax <= 0.0:
        raise ValueError("S_prim is non-positive (all eigenvalues <= 0).")
    keep = w > (eps_rel * wmax)
    r = int(np.count_nonzero(keep))
    if r == 0:
        raise ValueError("S_prim has zero numerical rank under the given threshold.")
    X = U[:, keep] / np.sqrt(w[keep])
    return X, r, w


def slice_eigens_xy(z_grid, S_prim, T_prim, V_en_of_z, M=1):
    Nz = len(z_grid)
    M  = int(M)

    X, r, wS = _s_orthonormalizer(S_prim, eps_rel=1e-12)
    if M > r:
        warnings.warn(f"M={M} exceeds rank(S)={r}; reducing M to {r}.")
        M = r

    E_slices = np.zeros((Nz, M), float)
    C_list   = [np.zeros((S_prim.shape[0], M), float) for _ in range(Nz)]

    for k, zk in enumerate(z_grid):
        Vz = V_en_of_z(float(zk))        
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


# ----------------------------------------------------
#  Helpers for J/K and PSD
# ----------------------------------------------------
def _psd_project_small(M):
    M = 0.5 * (M + M.T)
    w, V = la.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T


def precompute_eri_method2_JK_psd(
    alphas,
    centers,
    labels,
    z_grid,
    C_list,
    M,
    max_offset=None,
    auto_cut=False,
    cut_eps=1e-8,
    verbose=True,
):
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)
    z_grid = np.asarray(z_grid, float)
    Nz = z_grid.size
    N_ao = alphas.size

    if Nz == 0:
        raise ValueError("precompute_eri_method2_JK_psd: empty z_grid")

    if Nz > 1:
        dz = float(abs(z_grid[1] - z_grid[0]))
    else:
        dz = 0.0

    if max_offset is None or max_offset > (Nz - 1):
        max_offset = Nz - 1

    ERI_J = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]

    eri_by_h = {}
    norm0 = None
    h_max = max_offset
    t1 = time.time()
    for h in range(0, max_offset + 1):

        delta_z = abs(h * dz)
        if verbose:
            print(f"[psd-ERI] building AO ERI for h={h:2d}, Δz={delta_z: .6f}, time spent = {time.time() - t1}")

        # CHANGED: This now calls the generalized ERI function in the SPD helper
        eri_ao = eri_2d_cartesian_with_p(
            alphas,
            centers,
            labels,
            delta_z,
        )  

        eri_by_h[h] = eri_ao

        if auto_cut:
            nh = float(np.linalg.norm(eri_ao.reshape(-1)))
            if h == 0:
                norm0 = max(nh, 1e-16)
            else:
                if norm0 is None:
                    norm0 = max(nh, 1e-16)
                ratio = nh / norm0
                if verbose:
                    print(f"           ||ERI(h={h})||/||ERI(h=0)|| = {ratio:.2e}")
                if ratio < cut_eps:
                    if verbose:
                        print(f"[psd-ERI] auto-cut: stopping at h={h} (ratio<{cut_eps:.1e})")
                    h_max = h
                    break

    for m in range(Nz):
        C_m = np.asarray(C_list[m], float)  
        for n in range(Nz):
            h = abs(n - m)
            if h > h_max:
                continue

            C_n = np.asarray(C_list[n], float)
            eri_ao = eri_by_h[h]

            Jabcd = np.einsum(
                "pqrs,pa,qb,rc,sd->abcd",
                eri_ao,
                C_m, C_m, C_n, C_n,
                optimize=True,
            )  
            EmnJ = Jabcd.reshape(M * M, M * M)
            EmnJ = _psd_project_small(EmnJ)

            Kabcd = np.einsum(
                "pqrs,pa,rc,qb,sd->abcd",
                eri_ao,
                C_m, C_n, C_m, C_n,
                optimize=True,
            )
            EmnK = Kabcd.reshape(M * M, M * M)
            EmnK = 0.5 * (EmnK + EmnK.T)  

            ERI_J[m][n] = EmnJ
            ERI_K[m][n] = EmnK

    if verbose:
        print(f"[psd-ERI] built J/K blocks for Nz={Nz}, M={M}, h_max={h_max}")

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
            EmnJ = np.asarray(EmnJ)
            if EmnJ.ndim == 0:
                EmnJ = EmnJ.reshape(1, 1)
            jvec = EmnJ @ Ddiag[n].reshape(M * M)
            J_mm += jvec.reshape(M, M)
        im0 = m * M
        im1 = im0 + M
        F2e[im0:im1, im0:im1] += J_mm

    for m in range(Nz):
        im0 = m * M
        im1 = im0 + M
        for n in range(Nz):
            EmnK = ERI_K[m][n]
            EmnK = np.asarray(EmnK)
            if EmnK.ndim == 0:
                EmnK = EmnK.reshape(1, 1)
            BK   = P4[m, :, n, :]                   
            kvec = EmnK @ BK.reshape(M * M)
            K_mn = kvec.reshape(M, M)
            in0 = n * M
            in1 = in0 + M
            F2e[im0:im1, in0:in1] -= 0.5 * k_scale * K_mn

    return F2e


# ----------------------------------------------------
#  Build Method II (core + ERIs) and SCF
# ----------------------------------------------------
def build_method2(
    mol: Molecule,
    Lz=18.0,
    Nz=121,
    M=1,
    s_exps=None,
    p_exps=None,
    d_exps=None,  # CHANGED: Added d_exps argument
    max_offset=None,
    auto_cut=False,
    cut_eps=1e-6,
    verbose=True,
    dvr_method = 'sine'
):
    """
    Method-II builder using full s+p+d 2D AO basis.
    """
    t0 = time.time()
    if dvr_method == 'sine':
        z, Kz, dz = sine_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'exp':
        z, Kz, dz = Exponential_dvr_1d(-Lz, Lz, Nz)
    elif dvr_method == 'sinc':
        z, Kz, dz = sinc_dvr_1d(-Lz, Lz, Nz) 
    elif dvr_method == 'Legender':
        pass
    else:
        raise NotImplementedError('use sine or exp for not')
    nuclei = mol.to_tuples()

    print(f"\n[DEBUG] Grid Info for Nz={Nz}, Lz={Lz}:")
    print(f"  > Grid Spacing (dz): {dz:.6f}")
    print(f"  > First 3 points: {z[:3]}")
    print(f"  > Center point (index {len(z)//2}): {z[len(z)//2]}")
    
    if s_exps is None:
        raise ValueError("Please provide s_exps.")
    if p_exps is None: p_exps = np.array([], float)
    if d_exps is None: d_exps = np.array([], float) # Handle D

    # CHANGED: Calls make_xy_spd_primitive_basis
    alphas, centers, labels = make_xy_spd_primitive_basis(
        nuclei,
        exps_s=np.asarray(s_exps, float),
        exps_p=np.asarray(p_exps, float),
        exps_d=np.asarray(d_exps, float),
    )
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)

    def V_en_of_z(zk: float) -> np.ndarray:
        return V_en_sp_total_at_z(alphas, centers, labels, nuclei, zk)

    E_slices, C_list = slice_eigens_xy(
        z_grid=z,
        S_prim=S_prim,
        T_prim=T_prim,
        V_en_of_z=V_en_of_z,
        M=M,
    )

    Nz = len(z)
    S_slice = np.zeros((Nz, Nz, M, M), float)
    for k in range(Nz):
        Ck = C_list[k]
        for m in range(Nz):
            Cm = C_list[m]
            S_slice[k, m] = Ck.T @ (S_prim @ Cm)

    size = Nz * M
    Hcore = np.einsum('km,kmab->kamb', Kz, S_slice, optimize=True).reshape(size, size)
    Hcore += np.diag(E_slices.reshape(-1))

    if verbose:
        print(f"[Method2] Built Hcore {Hcore.shape} in {time.time()-t0:.2f}s")

    ERI_J, ERI_K = precompute_eri_method2_JK_psd(
        alphas, centers, labels, z, C_list,
        M=M, max_offset=max_offset,
        auto_cut=auto_cut, cut_eps=cut_eps, verbose=verbose
    )

    shapes = {"Nz": Nz, "M": M, "n_ao2d": len(alphas), "size": size, "dz": dz}
    return Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes


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
        if m < 2:
            return self.F_hist[-1]
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
                    damp=0.20,
                    diis_start=3, diis_space=8,
                    level_shift=0.5,
                    shift_decay=0.75,
                    k_ramp_iters=8):
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
            print(f"SCF {it:3d}: E = {Etot: .10f}  dE={dE:.2e}  ||[F,P]||={rnorm:.2e}  "
                  f"k={k_scale:.2f}  β={beta:.2f}  damp={damp:.2f}  dt={time.time()-t0:.2f}s")

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

    info = {"iter": it, "dE": dE, "rnorm": rnorm, "damp": damp, "level_shift": beta, "k_scale": k_scale}
    return Etot, eps, Cmo, P, info


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
    n  = C_list[0].shape[0]
    n2 = n * n

    d = [C_list[m][:, 0] for m in range(Nz)]
    v_mm = [np.kron(dm, dm) for dm in d]

    ERI_J = [[0.0 for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[0.0 for _ in range(Nz)] for _ in range(Nz)]

    for h in range(Nz):
        Nh = Nz - h
        if Nh <= 0:
            break
        V_right = np.column_stack([v_mm[nn] for nn in range(h, Nz)])  
        WJ = K_h[h] @ V_right
        for m in range(Nh):
            nn = m + h
            ERI_J[m][nn] = float(v_mm[m].T @ WJ[:, m])
            ERI_J[nn][m] = ERI_J[m][nn]
            v_mn = np.kron(d[m], d[nn])
            w_mn = Kx_h[h] @ v_mn
            ERI_K[m][nn] = float(v_mn.T @ w_mn)
            ERI_K[nn][m] = ERI_K[m][nn]

    for m in range(Nz):
        for n in range(Nz):
            if ERI_J[m][n] < 0.0:
                ERI_J[m][n] = 0.0
    return ERI_J, ERI_K


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


def rebuild_Hcore_from_d(
    d_stack,
    z,
    Kz,
    S_prim,
    T_prim,
    alphas,
    centers,
    labels,
    nuclei,
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
        Vz = V_en_sp_total_at_z(
            alphas,
            centers,
            labels,
            nuclei,
            float(z[n])
        )  
        e_local[n] = float(d_stack[n].T @ ((T_prim + Vz) @ d_stack[n]))

    Hcore = (Kz * S_scalar).astype(float)
    Hcore += np.diag(e_local)
    return Hcore


# ====================================================
#  Helper to save SCF results (AO/MO/Energy/Info)
# ====================================================
def save_scf_snapshot(run_folder, run_label, Nz, M, cycle, Etot, C_list, Cmo, P, eps, info,
                      alphas, centers, labels, z_grid, Lz):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    fname = f"scf_res_{run_label}_Nz{Nz}_M{M}_cyc{cycle}_{now_str}.npz"
    full_path = os.path.join(run_folder, fname)
    
    try:
        C_list_arr = np.stack(C_list, axis=0)
    except ValueError:
        C_list_arr = np.array(C_list, dtype=object)

    labels_serialized = []
    for l in labels:
        labels_serialized.append({
            'kind': l.kind,
            'dim': l.dim,
            'l': l.l,
            'role': l.role
        })
    
    np.savez_compressed(
        full_path,
        Etot=Etot,
        C_list=C_list_arr,  
        Cmo=Cmo,            
        P=P,                
        eps=eps,            
        info=info,          
        cycle=cycle,
        timestamp=now_str,
        alphas=alphas,
        centers=centers,
        labels_serialized=labels_serialized, 
        z_grid=z_grid,
        Lz=Lz,
        Nz=Nz,
        M=M
    )
    print(f"[Snapshot] Saved SCF data to {full_path}")


if __name__ == "__main__":
    stime = time.time()

    # molecular information
    charges = np.array([1.0, 1.0, 1.0, 1.0], float)
    # charges = np.array([1.0, 2.0], float)
    coords  = np.array([
    #     [0.0, 0.0,  0.7316 ],
    #     [0.0, 0.0,  -0.7316],
        [0.0, 0.0,  3.6 ],
        [0.0, 0.0,  0.91],
        [0.0, 0.0, -3.6 ],
        [0.0, 0.0, -0.91],
    #     # [0.0, 0.7,  0.7 ],
    #     # [0.0, -0.7,  0.7],
    #     # [0.0, 0.7, -0.7 ],
    #     # [0.0, -0.7, -0.7],
    ], float)
    # coords = np.linspace(-49, 49, 20, dtype=float)
    # coords = np.linspace(-19, 19, 20, dtype=float)
    # charges = np.ones_like(coords, dtype=float)
    # coords = np.stack([np.zeros_like(coords), np.zeros_like(coords), coords], axis=1)
    mol = Molecule(charges, coords, nelec=4)
    NELEC = mol.nelec
    Enuc  = mol.nuclear_repulsion_energy()
    print("=== Newton alternation test (s+p+d 2D basis) ===")
    print(f"nelec = {NELEC}")
    print(f"Enuc  = {Enuc:.10f} Eh")

    from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import (
        STO6_EXPS_H,
        Exp_631g_ss_H,
    )

    # s_exps = np.array([18.73113696, 2.825394365, 0.6401216923, 0.1612777588, 38.42163400, 5.778030000, 1.241774000, 0.2979640000], float) 
    s_exps = Exp_631g_ss_H.copy()
    p_exps = np.array([], float)  
    d_exps = np.array([], float) 

    # ------------------------
    #  DVR / slice parameters
    # ------------------------
    Nz_list = [63]   
    M_list  = [1]    
    LZ      = 12.5

    Nz_compare      = Nz_list[0]
    LZ_compare      = LZ
    M_list_compare  = [1, 2, 3, 4, 5, 6]

    REF_LINES = {
        # # H4 square
        # "STO-6G": -1.47441227867887,
        # "aug-ccpvdz": -1.75816708017562,

        # # H4 chain
        # "STO-6G": -1.8988478912704,
        # # "6-31G**": -2.02686367786042,
        # "cc-PVDZ": -2.02986485121015,

        # # # HeH+
        # "STO-6G": -2.873759439153611,
        # "6-31G**": -2.92470569991682,
        # "cc-PVDZ": -2.92411813977193,

        # # H20 from np.linspace(-49,49, 20)
        "STO-6G": -10.403664,
        "6-31G": -10.670717,
        "cc-PVDZ": -10.697370,
    }

    # -------------------------
    #  Newton / SCF parameters
    # -------------------------
    ALT_CYCLES              = 10
    NEWTON_STEPS_PER_CYCLE  = 1
    ACTIVE_MODE             = "topk_grad"   
    ACTIVE_TOPK             = 127          
    NEWTON_RIDGE            = 0.0
    SCF_MONO_TOL            = 1e-8
    SHORT_SCF_MAXITER       = 60
    VERBOSE                 = True
    DVR_METHOD              = 'sine'
    summary         = []
    E_history       = None
    E_newton_final  = None

    batch_folder = f"results_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
        print(f"\n[Main] Created results folder: {batch_folder}")

    for Nz in Nz_list:
        for M in M_list:
            print(f"\n==== Analytic-Newton Alternation: Nz={Nz}, M={M}, Lz={LZ} ====")

            Hcore, z, dz, E_slices, C_list, _ERI_J0, _ERI_K0, shapes = build_method2(
                mol,
                Lz=LZ,
                Nz=Nz,
                M=M,
                s_exps=s_exps,
                p_exps=p_exps,
                d_exps=d_exps, 
                max_offset=None,
                auto_cut=False,
                verbose=VERBOSE,
                dvr_method=DVR_METHOD,
            )

            nuclei = mol.to_tuples()

            alphas, centers, labels = make_xy_spd_primitive_basis(
                nuclei,
                exps_s=s_exps,
                exps_p=p_exps,
                exps_d=d_exps,
            )
            S_prim = overlap_2d_cartesian(alphas, centers, labels)
            T_prim = kinetic_2d_cartesian(alphas, centers, labels)
            if DVR_METHOD == 'sine':
                z_chk, Kz, dz_chk = sine_dvr_1d(-LZ, LZ, Nz)
            elif DVR_METHOD == 'exp':
                z_chk, Kz, dz_chk = Exponential_dvr_1d(-LZ, LZ, Nz)
            elif DVR_METHOD == 'sinc':
                z_chk, Kz, dz_chk = sinc_dvr_1d(-LZ, LZ, Nz)
            print("[Debug] DVR z grid:", z_chk)

            n_ao = len(alphas)
            K_h = []
            Kx_h = []

            for h in range(Nz):
                dz_val = h * dz
                # This generalized ERI function handles s/p/d automatically
                eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
                
                K_mat = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
                K_h.append(K_mat)
                
                eri_perm = eri_tensor.transpose(0, 2, 1, 3)
                Kx_mat = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
                Kx_h.append(Kx_mat)

            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)

            Etot, eps, Cmo, P, info = scf_rhf_method2(
                Hcore, ERI_J, ERI_K, Nz, M,
                nelec=NELEC, Enuc=Enuc,
                conv=1e-7, max_iter=100, verbose=VERBOSE,
            )
            print(f"[SCF 0] E = {Etot:.12f} Eh  (iters={info['iter']})")
            E_history = [Etot]

            save_scf_snapshot(
                run_folder=batch_folder, run_label="init",
                Nz=Nz, M=M, cycle=0,
                Etot=Etot, C_list=C_list, Cmo=Cmo, P=P, eps=eps, info=info,
                alphas=alphas, centers=centers, labels=labels, z_grid=z, Lz=LZ
            )

            h1_nm = build_h1_nm(
                Kz,
                S_prim,
                T_prim,
                z,
                lambda zz: V_en_sp_total_at_z(
                    alphas,
                    centers,
                    labels,
                    nuclei,
                    zz,
                ),
            )

            ERIop = CollocatedERIOp.from_kernels(
                N=S_prim.shape[0],
                Nz=Nz,
                dz=dz,
                K_h=K_h,
                Kx_h=Kx_h,
            )
            nh = NewtonHelper(h1_nm, S_prim, ERIop)
            
            d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz)])
            for n in range(Nz):
                dn = d_stack[n]
                d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

            for cyc in range(1, ALT_CYCLES + 1):
                P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()

                active = select_active_slices(
                    nh,
                    d_stack,
                    P_slice,
                    mode=ACTIVE_MODE,
                    topk=ACTIVE_TOPK,
                )
                if VERBOSE:
                    print(f"[Cycle {cyc}] active slices = {active}")

                for st in range(NEWTON_STEPS_PER_CYCLE):
                    g_full = nh.gradient(d_stack, P_slice)
                    delta_dict, lam, info_kkt = nh.kkt_step(
                        d_stack,
                        P_slice,
                        S_prim,
                        active=active,
                        ridge=NEWTON_RIDGE,
                    )

                    if g_dot_delta(g_full, delta_dict, active) > 0.0:
                        for n in active:
                            delta_dict[n] *= -1.0

                    MAX_STEP_S_NORM = 0.10
                    delta_dict, scaled = bound_step_S(
                        delta_dict,
                        active,
                        S_prim,
                        MAX_STEP_S_NORM,
                    )
                    if VERBOSE and scaled < 1.0:
                        print(f"  [Newton {st+1}] trust-region scale = {scaled:.3f}")

                    STEP_LIST = [
                        100, 65, 50, 20, 15, 8, 5, 3, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10,
                        0.05, 0.02, 0.01,
                    ]
                    tried = {}
                    best = (np.inf, None, None, None, None)  

                    for step in STEP_LIST:
                        if step in tried:
                            continue
                        E_try, d_try, C_try, P_try = evaluate_trial_step(
                            step, d_stack, C_list, active,
                            S_prim, z, Kz, T_prim, alphas, centers, labels, nuclei,
                            Nz, M, NELEC, Enuc, K_h, Kx_h,
                            SHORT_SCF_MAXITER, delta_dict,
                        )
                        tried[step] = E_try
                        if VERBOSE:
                            print(
                                f"  [Newton {st+1}] step={step:g}  "
                                f"E_try={E_try:.12f}  ΔE={E_try - E_history[-1]:+.3e}"
                            )
                        if E_try < best[0]:
                            best = (E_try, step, d_try, C_try, P_try)

                    if best[0] < E_history[-1] - SCF_MONO_TOL:
                        d_stack = best[2]
                        C_list  = best[3]
                        Etot    = best[0]
                        P       = best[4]
                    else:
                        if VERBOSE:
                            print("  [Newton] no monotone step accepted; keeping previous d_stack")

                Hcore = rebuild_Hcore_from_d(
                    d_stack, z, Kz, S_prim, T_prim,
                    alphas, centers, labels, nuclei,
                )
                ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
                Etot, eps, Cmo, P, info = scf_rhf_method2(
                    Hcore, ERI_J, ERI_K, Nz, M,
                    nelec=NELEC, Enuc=Enuc,
                    conv=1e-7, max_iter=100, verbose=VERBOSE,
                )
                E_history.append(Etot)
                print(
                    f"[SCF {cyc}] E = {Etot:.12f} Eh  (iters={info['iter']})  "
                    f"ΔE={Etot - E_history[-2]:+.3e}"
                )

                save_scf_snapshot(
                    run_folder=batch_folder, run_label="newton",
                    Nz=Nz, M=M, cycle=cyc,
                    Etot=Etot, C_list=C_list, Cmo=Cmo, P=P, eps=eps, info=info,
                    alphas=alphas, centers=centers, labels=labels, z_grid=z, Lz=LZ
                )

            print(f"== Energies (Nz={Nz}, M={M}) ==")
            for i, E in enumerate(E_history):
                print(f"  Stage {i:2d}: {E:.12f} Eh")
            summary.append((Nz, M, E_history[0], E_history[-1]))
            E_newton_final = E_history[-1]

            mpl.rcParams.update(
                {"mathtext.fontset": "dejavusans", "font.family": "dejavusans"}
            )
            fig1, ax1 = plt.subplots(figsize=(6.0, 4.0))
            cycles = np.arange(len(E_history))
            ax1.plot(
                cycles,
                E_history,
                marker='o',
                label=f"Newton alternation (Nz={Nz}, M={M})",
            )
            for name, val in REF_LINES.items():
                if val is not None:
                    ax1.axhline(val, ls='--', label=name)
            ax1.set_xlabel("Cycle # (Newton block + SCF)")
            ax1.set_ylabel("RHF total energy (Eh)")
            ax1.set_title(r"Convergence (M=1): $E$ vs cycles")
            ax1.grid(True)
            ax1.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(batch_folder, "convergence_m1_cycles.png"),
                dpi=2000,
                bbox_inches='tight',
                pad_inches=0.04,
            )

    print("\nSummary (Nz, M, E_initial, E_final, ΔE_total):")
    for Nz, M, Ei, Ef in summary:
        print(f"  Nz={Nz:3d}  M={M}  Ei={Ei:.12f}  Ef={Ef:.12f}  ΔE={Ef - Ei:+.6e}")

