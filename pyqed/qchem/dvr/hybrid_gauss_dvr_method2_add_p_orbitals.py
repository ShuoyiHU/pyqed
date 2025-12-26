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
# from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import *
from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import (
    # make_xy_primitives_on_nuclei,
    # V_en_prim_total_at_z,
    V_en_sp_total_at_z,
    # pair_params, _permute_K_ikjl,
    build_h1_nm,
    # PrimitiveLabel,
    make_xy_sp_primitive_basis,
    overlap_2d_cartesian,
    kinetic_2d_cartesian,
    eri_2d_cartesian_with_p,
)


# ------------------------
#  Basic molecule holder
# ------------------------
class Molecule:
    def __init__(self, charges, coords, nelec=None):
        """
        charges : array-like (N_nuc,)
        coords  : array-like (N_nuc,3)
        nelec   : total number of electrons. If None, assume neutral molecule
                  with nelec = sum(Z).
        """
        self.charges = np.asarray(charges, float).reshape(-1)
        self.coords  = np.asarray(coords,  float).reshape(-1, 3)
        assert self.charges.shape[0] == self.coords.shape[0]

        if nelec is None:
            self.nelec = int(round(float(np.sum(self.charges))))
        else:
            self.nelec = int(nelec)

    def to_tuples(self):
        """Return list of (Z, x, y, z) tuples for integrals."""
        return [(float(Z), float(x), float(y), float(z))
                for (Z, (x, y, z)) in zip(self.charges, self.coords)]

    def nuclear_repulsion_energy(self):
        """Simple Coulomb repulsion Σ_{i<j} Z_i Z_j / |R_i - R_j|."""
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
    """
    Standard sine-DVR on [zmin,zmax] with N interior points.

    Returns z (N,), kinetic Tz (N,N), dz spacing.
    """
    L = zmax - zmin
    j = np.arange(1, N + 1)
    z = zmin + j * L / (N + 1)
    n = np.arange(1, N + 1)
    U = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * np.outer(j, n) / (N + 1))
    lam = 0.5 * (np.pi * n / L) ** 2       # -1/2 d^2/dz^2 eigenvalues
    Tz = (U * lam) @ U.T                   # DVR kinetic
    dz = L / (N + 1)
    return z, Tz, dz


# ==========================================
#  B) S-metric orthonormalization
# ==========================================
def _s_orthonormalizer(S, eps_rel=1e-12):
    """
    Compute X such that X^T S X = I on the numerically stable subspace.

    Returns:
      X : (n, r)       with r = rank(S) above threshold
      r : int          numerical rank
      w : eigenvalues of S (for diagnostics)
    """
    S = 0.5 * (S + S.T)                # symmetrize
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



def slice_eigens_xy(
    z_grid,
    S_prim,
    T_prim,
    V_en_of_z,
    M=1,
):
    """
    Solve (T_prim + V_en(z)) c = E S_prim c per z.

    V_en_of_z: callable, V_en_of_z(z_k) -> (N_AO, N_AO) in the full AO basis.
    """
    Nz = len(z_grid)
    M  = int(M)

    X, r, wS = _s_orthonormalizer(S_prim, eps_rel=1e-12)
    if M > r:
        warnings.warn(f"M={M} exceeds rank(S)={r}; reducing M to {r}.")
        M = r

    E_slices = np.zeros((Nz, M), float)
    C_list   = [np.zeros((S_prim.shape[0], M), float) for _ in range(Nz)]

    for k, zk in enumerate(z_grid):
        Vz = V_en_of_z(float(zk))        # full AO V_en
        Hk = T_prim + Vz                 # AO 1e Hamiltonian on slice
        Hk_ = X.T @ Hk @ X               # orthonormalized
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
    """Nearest PSD in Frobenius norm for a small (M^2×M^2) block."""
    M = 0.5 * (M + M.T)
    w, V = la.eigh(M)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T



#  ERI calculation for MO, calling eri_2d_cartesian_with_p for ERI for AO as a bridge
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
    """
    Build Coulomb / exchange blocks in the slice basis:

        ERI_J[m][n] : (M^2, M^2) with elements J[(ab),(cd)] = (ma mb | n c n d)
        ERI_K[m][n] : (M^2, M^2) with elements K[(ab),(cd)] = (ma n c | m b n d)

    Uses full AO ERIs from eri_2d_cartesian_psss (s + p_x + p_y) for each
    Δz = |z_m - z_n|. We exploit that sine-DVR grid is equally spaced, so
    Δz depends only on h = |m-n| and cache AO ERIs by h.
    """
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

    # ERI_J[m][n], ERI_K[m][n] — each (M^2, M^2)
    ERI_J = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]
    ERI_K = [[np.zeros((M * M, M * M), float) for _ in range(Nz)] for _ in range(Nz)]

    # Cache AO ERIs for each offset h = |m-n|
    eri_by_h = {}
    norm0 = None
    h_max = max_offset
    t1 = time.time()
    for h in range(0, max_offset + 1):

        delta_z = abs(h * dz)
        if verbose:
            print(f"[psd-ERI] building AO ERI for h={h:2d}, Δz={delta_z: .6f}, time spent = {time.time() - t1}")

        eri_ao = eri_2d_cartesian_with_p(
            alphas,
            centers,
            labels,
            delta_z,
        )  # shape: (N_ao, N_ao, N_ao, N_ao), chemist order (p q | r s)

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

    # Contract AO ERIs into slice-basis J/K blocks
    for m in range(Nz):
        C_m = np.asarray(C_list[m], float)  # (N_ao, M)
        for n in range(Nz):
            h = abs(n - m)
            if h > h_max:
                # leave zeros (weak-coupling tail)
                continue

            C_n = np.asarray(C_list[n], float)
            eri_ao = eri_by_h[h]

            # Coulomb: (ma mb | n c n d)
            Jabcd = np.einsum(
                "pqrs,pa,qb,rc,sd->abcd",
                eri_ao,
                C_m, C_m, C_n, C_n,
                optimize=True,
            )  # (M,M,M,M)
            EmnJ = Jabcd.reshape(M * M, M * M)
            EmnJ = _psd_project_small(EmnJ)

            # Exchange: (ma n c | m b n d)
            Kabcd = np.einsum(
                "pqrs,pa,rc,qb,sd->abcd",
                eri_ao,
                C_m, C_n, C_m, C_n,
                optimize=True,
            )
            EmnK = Kabcd.reshape(M * M, M * M)
            EmnK = 0.5 * (EmnK + EmnK.T)  # symmetrize

            ERI_J[m][n] = EmnJ
            ERI_K[m][n] = EmnK

    if verbose:
        print(f"[psd-ERI] built J/K blocks for Nz={Nz}, M={M}, h_max={h_max}")

    return ERI_J, ERI_K

# ----------------------------------------------------
#  Two-electron Fock from J/K blocks (slice basis)
# ----------------------------------------------------
def fock_2e_slice_collocated(P, ERI_J, ERI_K, Nz, M, k_scale=1.0):
    """
    P: density in slice basis (Nz*M, Nz*M).
    ERI_J[m][n] ~ (ab|cd) Coulomb blocks
    ERI_K[m][n] ~ (ac|bd) Exchange blocks
    k_scale: 0..1 factor on exchange (adiabatic turn-on).
    """
    N = Nz * M
    P4 = P.reshape(Nz, M, Nz, M)
    Ddiag = [P4[b, :, b, :].copy() for b in range(Nz)]  # (M,M)

    F2e = np.zeros((N, N), dtype=float)

    # Coulomb J: contributes to (m,m) only
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

    # Exchange K: fills (m,n), scaled by k_scale
    for m in range(Nz):
        im0 = m * M
        im1 = im0 + M
        for n in range(Nz):
            EmnK = ERI_K[m][n]
            EmnK = np.asarray(EmnK)
            if EmnK.ndim == 0:
                EmnK = EmnK.reshape(1, 1)
            BK   = P4[m, :, n, :]                   # P_{(m,c),(n,d)}
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
    max_offset=None,
    auto_cut=False,
    cut_eps=1e-6,
    verbose=True,
):
    """
    Method-II builder using full s+px+py 2D AO basis.
    """
    t0 = time.time()
    z, Kz, dz = sine_dvr_1d(-Lz, Lz, Nz)
    nuclei = mol.to_tuples()

    # --- AO primitives: s + p_x + p_y with separate exponents ---
    if s_exps is None:
        raise ValueError("Please provide the s basis coefficient by specifying a s_exps numpy array, which is a must for calculation.")
    if p_exps is None:
        p_exps = np.array([], float)

    alphas, centers, labels = make_xy_sp_primitive_basis(
        nuclei,
        exps_s=np.asarray(s_exps, float),
        exps_p=np.asarray(p_exps, float),
    )
    alphas = np.asarray(alphas, float)
    centers = np.asarray(centers, float)

    # One-electron overlap and kinetic in full AO basis
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)

    # Analytic full s+p electron–nuclear matrix on each slice
    def V_en_of_z(zk: float) -> np.ndarray:
        return V_en_sp_total_at_z(alphas, centers, labels, nuclei, zk)

    # Slice eigenstates (keep M lowest per z)
    E_slices, C_list = slice_eigens_xy(
        z_grid=z,
        S_prim=S_prim,
        T_prim=T_prim,
        V_en_of_z=V_en_of_z,
        M=M,
    )

    # Inter-slice overlap S_{km}^{ab} = C_k^T S_prim C_m
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

    # J/K from your existing p-ready ERI
    ERI_J, ERI_K = precompute_eri_method2_JK_psd(
        alphas, centers, labels, z, C_list,
        M=M, max_offset=max_offset,
        auto_cut=auto_cut, cut_eps=cut_eps, verbose=verbose
    )

    shapes = {"Nz": Nz, "M": M, "n_ao2d": len(alphas), "size": size, "dz": dz}
    return Hcore, z, dz, E_slices, C_list, ERI_J, ERI_K, shapes


class PulayCDIIS:
    """Pulay/DIIS mixing on F using commutator residuals E = [F, P]."""
    def __init__(self, space=6, reg=1e-12):
        self.space = int(space)
        self.reg   = float(reg)
        self.F_hist = []   # list of F copies
        self.E_hist = []   # list of flattened error vectors

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
        R = np.vstack(self.E_hist)            # (m, n)
        G = R @ R.T                           # Gram matrix
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
    """
    Dense RHF in the (orthonormal) slice basis with damping, CDIIS (currently off),
    level shift, and exchange ramping. Return (Etot, eps, Cmo, P, info).
    """
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

        # DIIS and level shift are available but currently commented out, like your original:
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


# ----------------------------------------------------
#  Small helpers used in Newton alternation
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
    """
    M=1 version of ERI builder using precomputed AO-pair kernels K_h, Kx_h.
    """
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
        V_right = np.column_stack([v_mm[nn] for nn in range(h, Nz)])  # (n2, Nh)
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
    """
    Trial Newton step (M=1 case) on only 'active' slices.
    Returns (Etot_try, trial_d, trial_C_list, P_try).
    """
    trial_d = d_stack.copy()
    NewtonHelper.update_inplace(trial_d, delta_dict, S_prim, active, step=step)

    trial_C_list = [C_list[n].copy() for n in range(Nz)]
    for n in active:
        trial_C_list[n][:, 0] = trial_d[n]

    # Hcore with full s+p potential
    Hcore_try = rebuild_Hcore_from_d(
        trial_d, z, Kz, S_prim, T_prim,
        alphas, centers, labels, nuclei
    )

    # J/K from kernels for the trial contracted orbitals
    ERI_J_try, ERI_K_try = eri_JK_from_kernels_M1(trial_C_list, K_h, Kx_h)

    Etot_try, _, _, P_try, _ = scf_rhf_method2(
        Hcore_try, ERI_J_try, ERI_K_try, Nz, M,
        nelec=nelec, Enuc=Enuc,
        conv=3e-7, max_iter=SHORT_SCF_MAXITER, verbose=False
    )
    return Etot_try, trial_d, trial_C_list, P_try


def quad_refine(alpha1, E1, alpha2, E2, alpha3, E3):
    """
    Quadratic refinement helper (not used in the main flow right now,
    but kept for possible future line-search refinement).
    """
    denom = (alpha1 - alpha2) * (alpha1 - alpha3) * (alpha2 - alpha3)
    if abs(denom) < 1e-16:
        return None
    a = (alpha3 * (E2 - E1) + alpha2 * (E1 - E3) + alpha1 * (E3 - E2)) / denom
    b = (alpha3 * alpha3 * (E1 - E2) +
         alpha2 * alpha2 * (E3 - E1) +
         alpha1 * alpha1 * (E2 - E3)) / denom
    if abs(a) < 1e-16:
        return None
    alpha_star = -b / (2.0 * a)
    lo, hi = min(alpha1, alpha3), max(alpha1, alpha3)
    if alpha_star <= lo or alpha_star >= hi:
        return None
    return alpha_star


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
    """
    Rebuild Hcore in the M=1 case from the current contracted vectors d_stack.

    Uses the full s + p_x + p_y electron–nuclear potential V_en_sp_total_at_z
    in the primitive 2D AO basis.
    """
    Nz, N = d_stack.shape

    # Inter-slice overlap in the contracted (M=1) slice basis
    S_scalar = np.zeros((Nz, Nz), float)
    for n in range(Nz):
        dn = d_stack[n]
        S_scalar[n, n] = float(dn.T @ (S_prim @ dn))
        for m in range(n + 1, Nz):
            dm = d_stack[m]
            val = float(dn.T @ (S_prim @ dm))
            S_scalar[n, m] = val
            S_scalar[m, n] = val

    # Local transverse energies per slice from T + V_en(s+p)
    e_local = np.zeros(Nz, float)
    for n in range(Nz):
        Vz = V_en_sp_total_at_z(
            alphas,
            centers,
            labels,
            nuclei,
            float(z[n])
        )  # (N_ao, N_ao)
        e_local[n] = float(d_stack[n].T @ ((T_prim + Vz) @ d_stack[n]))

    # Hcore_{nm} = Kz_{nm} <φ_n|φ_m> + δ_{nm} e_local(n)
    Hcore = (Kz * S_scalar).astype(float)
    Hcore += np.diag(e_local)
    return Hcore


# ====================================================
#  Helper to save SCF results (AO/MO/Energy/Info)
# ====================================================
def save_scf_snapshot(run_folder, run_label, Nz, M, cycle, Etot, C_list, Cmo, P, eps, info,
                      alphas, centers, labels, z_grid, Lz):
    """
    Saves a snapshot of the current SCF calculation to a .npz file.
    Includes basis info (alphas, centers, labels) so density can be reconstructed.
    """
    # High-precision timestamp
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Filename construction
    fname = f"scf_res_{run_label}_Nz{Nz}_M{M}_cyc{cycle}_{now_str}.npz"
    full_path = os.path.join(run_folder, fname)
    
    # Convert C_list to array
    try:
        C_list_arr = np.stack(C_list, axis=0)
    except ValueError:
        C_list_arr = np.array(C_list, dtype=object)

    # Convert complex label objects to serializable dicts
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
        C_list=C_list_arr,  # Slice contraction coefficients (Effective AOs)
        Cmo=Cmo,            # MO coefficients
        P=P,                # Density matrix
        eps=eps,            # Orbital energies
        info=info,          # Convergence info
        cycle=cycle,
        timestamp=now_str,
        # Basis info for reconstruction:
        alphas=alphas,
        centers=centers,
        labels_serialized=labels_serialized, # stored as array of dicts/objects
        z_grid=z_grid,
        Lz=Lz,
        Nz=Nz,
        M=M
    )
    print(f"[Snapshot] Saved SCF data to {full_path}")


# main: Newton alternation (M=1) + Final E vs M
# currently with s, p orbital available
if __name__ == "__main__":
    stime = time.time()

    # molecular information
    charges = np.array([1.0, 1.0, 1.0, 1.0], float)
    coords  = np.array([
        # [0.0,  0.7,  0.7],
        # [0.0,  0.7, -0.7],
        # [0.0, -0.7,  0.7],
        # [0.0, -0.7, -0.7],
        [0.0, 0.0,  3.6 ],
        [0.0, 0.0,  0.91],
        [0.0, 0.0, -3.6 ],
        [0.0, 0.0, -0.91],
    ], float)

    mol = Molecule(charges, coords, nelec=None)
    NELEC = mol.nelec
    Enuc  = mol.nuclear_repulsion_energy()
    print("=== Newton alternation test (s+p 2D basis) ===")
    print(f"nelec = {NELEC}")
    print(f"Enuc  = {Enuc:.10f} Eh")

    # basis information
    from pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_p_orbitals import (
        STO6_EXPS_H,
        Exp_631g_ss_H,
    )

    s_exps = Exp_631g_ss_H.copy()
    p_exps = np.array([], float)  

    # ------------------------
    #  DVR / slice parameters
    # ------------------------
    Nz_list = [31]   # number of DVR grid used
    M_list  = [1]    # Newton alternation currently only work for M=1
    LZ      = 8.0

    # For the "final E vs M" scan
    Nz_compare      = Nz_list[0]
    LZ_compare      = LZ
    M_list_compare  = [1, 2, 3, 4, 5, 6]

    # Reference lines
    REF_LINES = {
        # H4 square
        # "STO-6G": -1.47441227867887,
        # "aug-ccpvdz": -1.75816708017562,

        # H4 chain
        "STO-6G": -1.8988478912704,
        # "6-31G**": -2.02686367786042,
        "cc-PVDZ": -2.02986485121015,
    }

    # -------------------------
    #  Newton / SCF parameters
    # -------------------------
    ALT_CYCLES              = 10
    NEWTON_STEPS_PER_CYCLE  = 1
    ACTIVE_MODE             = "topk_grad"   # "center" | "topk_grad"
    ACTIVE_TOPK             = 127          # when ACTIVE_MODE == "topk_grad"
    NEWTON_RIDGE            = 0.0
    SCF_MONO_TOL            = 1e-8
    SHORT_SCF_MAXITER       = 60
    VERBOSE                 = True

    summary         = []
    E_history       = None
    E_newton_final  = None

    # Create a folder for this batch run
    batch_folder = f"results_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(batch_folder):
        os.makedirs(batch_folder)
        print(f"\n[Main] Created results folder: {batch_folder}")

    # Newton alternation for M = 1 re-optimization

    for Nz in Nz_list:
        for M in M_list:
            print(f"\n==== Analytic-Newton Alternation: Nz={Nz}, M={M}, Lz={LZ} ====")

            # Initial Method-II build with p-ready ERIs (s+p AO basis)
            Hcore, z, dz, E_slices, C_list, _ERI_J0, _ERI_K0, shapes = build_method2(
                mol,
                Lz=LZ,
                Nz=Nz,
                M=M,
                s_exps=s_exps,
                p_exps=p_exps,
                max_offset=None,
                auto_cut=False,
                verbose=VERBOSE,
            )

            nuclei = mol.to_tuples()

            # AO primitives and 1e integrals (s+p), consistent with build_method2
            alphas, centers, labels = make_xy_sp_primitive_basis(
                nuclei,
                exps_s=s_exps,
                exps_p=p_exps,
            )
            S_prim = overlap_2d_cartesian(alphas, centers, labels)
            T_prim = kinetic_2d_cartesian(alphas, centers, labels)

            # DVR kinetic along z (for h1_nm / Hcore rebuild)
            z_chk, Kz, dz_chk = sine_dvr_1d(-LZ, LZ, Nz)
            print("[Debug] DVR z grid:", z_chk)

            n_ao = len(alphas)
            K_h = []
            Kx_h = []

            for h in range(Nz):
                dz_val = h * dz
                # 1. Calculate full tensor (N,N,N,N)
                eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
                
                # 2. Reshape to Matrix (N^2, N^2) for Coulomb K_h
                K_mat = eri_tensor.reshape(n_ao * n_ao, n_ao * n_ao)
                K_h.append(K_mat)
                
                # 3. Permute and Reshape for Exchange Kx_h
                # (p q | r s) -> (p r | q s)
                eri_perm = eri_tensor.transpose(0, 2, 1, 3)
                Kx_mat = eri_perm.reshape(n_ao * n_ao, n_ao * n_ao)
                Kx_h.append(Kx_mat)

            # J/K from kernels (M=1)
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)

            # Initial SCF in the slice basis
            Etot, eps, Cmo, P, info = scf_rhf_method2(
                Hcore, ERI_J, ERI_K, Nz, M,
                nelec=NELEC, Enuc=Enuc,
                conv=1e-7, max_iter=100, verbose=VERBOSE,
            )
            print(f"[SCF 0] E = {Etot:.12f} Eh  (iters={info['iter']})")
            E_history = [Etot]

            # --- SAVE SNAPSHOT (Initial) ---
            save_scf_snapshot(
                run_folder=batch_folder, run_label="init",
                Nz=Nz, M=M, cycle=0,
                Etot=Etot, C_list=C_list, Cmo=Cmo, P=P, eps=eps, info=info,
                alphas=alphas, centers=centers, labels=labels, z_grid=z, Lz=LZ
            )
            # -------------------------------

            # h1_{n,m} in primitive basis for NewtonHelper (s+p version)
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

            
            # # DEBUGGING stuff: dump all basic integrals to disk
            # # ... (commented out sections kept as is) ...

            # Current contracted vectors (M=1), S-normalized
            d_stack = np.vstack([C_list[n][:, 0].copy() for n in range(Nz)])
            for n in range(Nz):
                dn = d_stack[n]
                d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

            # ===== Alternating Newton + SCF =====
            for cyc in range(1, ALT_CYCLES + 1):
                # Density in slice picture for NewtonHelper
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

                # Newton steps with monotone acceptance
                for st in range(NEWTON_STEPS_PER_CYCLE):
                    g_full = nh.gradient(d_stack, P_slice)
                    delta_dict, lam, info_kkt = nh.kkt_step(
                        d_stack,
                        P_slice,
                        S_prim,
                        active=active,
                        ridge=NEWTON_RIDGE,
                    )

                    # Ensure descent direction
                    if g_dot_delta(g_full, delta_dict, active) > 0.0:
                        for n in active:
                            delta_dict[n] *= -1.0

                    # Trust-region bound in S-norm
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
                        3, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10,
                        0.05, 0.02, 0.01,
                        -3, -1.5, -1.0, -0.75, -0.5, -0.35, -0.25, -0.15,
                        -0.10, -0.05, -0.02, -0.01,
                    ]
                    tried = {}
                    best = (np.inf, None, None, None, None)  # (E, step, d_try, C_try, P_try)

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

                # Full SCF after Newton step(s)
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

                # --- SAVE SNAPSHOT (Newton Cycle) ---
                save_scf_snapshot(
                    run_folder=batch_folder, run_label="newton",
                    Nz=Nz, M=M, cycle=cyc,
                    Etot=Etot, C_list=C_list, Cmo=Cmo, P=P, eps=eps, info=info,
                    alphas=alphas, centers=centers, labels=labels, z_grid=z, Lz=LZ
                )
                # ------------------------------------

            print(f"== Energies (Nz={Nz}, M={M}) ==")
            for i, E in enumerate(E_history):
                print(f"  Stage {i:2d}: {E:.12f} Eh")
            summary.append((Nz, M, E_history[0], E_history[-1]))
            E_newton_final = E_history[-1]

            # ---- Fig.1: Energy vs Cycle (M=1 Newton alternation) ----
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
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.04,
            )

    print("\nSummary (Nz, M, E_initial, E_final, ΔE_total):")
    for Nz, M, Ei, Ef in summary:
        print(f"  Nz={Nz:3d}  M={M}  Ei={Ei:.12f}  Ef={Ef:.12f}  ΔE={Ef - Ei:+.6e}")
    # # ==========================================================
    # #  PART 2: multi-M (pure SCF, no Newton) — final E vs M
    # # ==========================================================
    # Enuc = mol.nuclear_repulsion_energy()
    # M_vals          = []
    # E_final_vs_M    = []

    # for M in M_list_compare:
    #     print(f"\n[Final-E vs M] Building Nz={Nz_compare}, M={M} ...")
    #     Hcore, z, dz, Esl, C_list, ERI_J, ERI_K, shapes = build_method2(
    #         mol,
    #         Lz=LZ_compare,
    #         Nz=Nz_compare,
    #         M=M,
    #         s_exps=s_exps,
    #         p_exps=p_exps,
    #         max_offset=None,
    #         auto_cut=False,
    #         verbose=False,
    #     )
    #     Etot, eps, Cmo, P, info = scf_rhf_method2(
    #         Hcore, ERI_J, ERI_K, Nz_compare, M,
    #         nelec=NELEC, Enuc=Enuc,
    #         conv=1e-7, max_iter=60, verbose=True,
    #     )
    #     print(f"[Final-E vs M] M={M}: final E={Etot:.12f} Eh (iters={info['iter']})")
    #     M_vals.append(M)
    #     E_final_vs_M.append(Etot)

    # fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
    # ax2.plot(
    #     M_vals,
    #     E_final_vs_M,
    #     marker='o',
    #     linestyle='-',
    #     label=f"Pure SCF (Nz={Nz_compare})",
    # )
    # if E_newton_final is not None:
    #     ax2.axhline(
    #         E_newton_final,
    #         ls='--',
    #         label=f"Newton (M=1 reopt), E={E_newton_final:.6f}",
    #     )
    #     if 1 in M_vals:
    #         idx = M_vals.index(1)
    #         ax2.scatter(
    #             [1],
    #             [E_final_vs_M[idx]],
    #             marker='x',
    #             s=80,
    #             label="Pure SCF M=1",
    #         )

    # for name, val in REF_LINES.items():
    #     if val is not None:
    #         ax2.axhline(val, ls=':', label=name)

    # ax2.set_xlabel("M (number of contracted Gaussians per z-slice)")
    # ax2.set_ylabel("Final RHF total energy (Eh)")
    # ax2.set_title(
    #     r"Final energy vs $M$ (fixed $N_z$) — comparison with Newton $M{=}1$"
    # )
    # ax2.grid(True)
    # ax2.legend()
    # mpl.rcParams.update(
    #     {"mathtext.fontset": "dejavusans", "font.family": "dejavusans"}
    # )
    # plt.tight_layout()
    # plt.savefig(
    #     "final_energy_vs_M_fixed_Nz.png",
    #     dpi=600,
    #     bbox_inches='tight',
    #     pad_inches=0.04,
    # )

    # print(f"\nTotal wall time: {time.time() - stime:.2f} s")
