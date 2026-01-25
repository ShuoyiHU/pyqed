import numpy as np
import time
import logging
import scipy.linalg
from typing import List, Tuple
from collections import namedtuple

# --- Imports ---
from pyqed.qchem.dvr.hybrid_gauss_dvr_method_sweep import (
    Molecule, build_method2, make_xy_spd_primitive_basis, 
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,sine_dvr_1d,
    rebuild_Hcore_from_d, eri_JK_from_kernels_M1,
    build_h1_nm, V_en_sp_total_at_z, CollocatedERIOp
)
from pyqed.mps.autompo.model import Model
from pyqed.mps.autompo.Operator import Op
from pyqed.mps.autompo.basis import BasisSimpleElectron
from pyqed.mps.autompo.light_automatic_mpo import Mpo
import pyqed.mps.autompo.light_automatic_mpo as lampo
import pyqed.mps.mps as mps_lib
import pyqed.qchem.gdvr.macro_dmrg_scf_sweep.gdvr_dmrg_scf as gdvr_dmrg_scf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

OpTuple = namedtuple("OpTuple", ["symbol", "qn", "factor"])

# ==============================================================================
#  PATCH: Fixed-Rank QR (Safety Mechanism)
# ==============================================================================
TARGET_MPO_BOND_DIM = 100 

def fixed_decompose_qr(term_row, term_col, non_red, in_ops_list, factor, primary_ops, algo, k=1):
    # Standard QR
    non_red.data = factor[non_red.data - 1]
    gamma = non_red.todense()
    if gamma.shape[1] != 1:
        q, r, p = scipy.linalg.qr(gamma, mode="economic", pivoting=True)
    else:
        q = gamma; r = np.array([1]).reshape(1, 1); p = np.array([0])
        
    # Force Fixed Rank (Padding)
    rank = TARGET_MPO_BOND_DIM
    limit = min(rank, q.shape[1])
    
    out_ops = [[] for _ in range(rank)]
    atol = 1e-10
    
    # Fill Q
    for i, j in zip(*np.where(np.abs(q[:, :limit]) > atol)):
        symbol = term_row[i]
        qn = lampo._compute_qn(in_ops_list, symbol, primary_ops, k)
        out_op = OpTuple(symbol, qn, factor=q[i, j])
        out_ops[j].append(out_op)
        
    # Fill R (Padded)
    r_padded = np.zeros((rank, r.shape[1]), dtype=r.dtype)
    rows_to_copy = min(rank, r.shape[0])
    r_padded[:rows_to_copy, :] = r[:rows_to_copy, :]
    r2 = r_padded[:, np.argsort(p)]
    
    idx1, idx2 = np.where(np.abs(r2) > 1e-15)
    new_factor = r2[(idx1, idx2)]
    col_syms = [term_col[i] for i in idx2]
    
    if len(idx1) > 0:
        new_table = np.concatenate([idx1.reshape(-1, 1), col_syms], axis=1)
    else:
        new_table = np.zeros((0, len(term_col[0]) + 1), dtype=int)
        
    return out_ops, new_table, new_factor

# Apply Patch
lampo._decompose_qr = fixed_decompose_qr

# ==============================================================================
#  MONKEY PATCH: Operator Matrices
# ==============================================================================
def patched_op_mat(self, op):
    if not isinstance(op, Op): op = Op(op, None)
    op_symbol, op_factor = op.symbol, op.factor
    mat = np.zeros((2, 2))
    if op_symbol == r"a^\dagger": mat[1, 0] = 1.
    elif op_symbol == "a": mat[0, 1] = 1.
    elif op_symbol == "n" or op_symbol == r"a^\dagger a": mat[1, 1] = 1.
    elif op_symbol == "I": mat = np.eye(2)
    elif op_symbol == "sigma_z": mat[0, 0] = 1.; mat[1, 1] = -1.
    else: raise ValueError(f"op_symbol:{op_symbol} is not supported")
    return mat * op_factor
BasisSimpleElectron.op_mat = patched_op_mat

# ==============================================================================
#  HELPER: MPO Construction
# ==============================================================================
def build_mpo_from_integrals(Hcore, V_coul, Nz, bond_dim):
    n_spin = 2 * Nz
    cutoff = 1e-12
    ham_terms = []
    rows, cols = np.nonzero(np.abs(Hcore) > cutoff)
    for i, j in zip(rows, cols):
        val = Hcore[i, j]
        ham_terms.append(Op(r"a^\dagger a", [2*i, 2*j], val, qn=[1, -1]))
        ham_terms.append(Op(r"a^\dagger a", [2*i+1, 2*j+1], val, qn=[1, -1]))
    rows, cols = np.nonzero(np.abs(V_coul) > cutoff)
    for i, k in zip(rows, cols):
        val = V_coul[i, k]
        if i == k:
            ham_terms.append(Op(r"n n", [2*i, 2*i+1], val, qn=[0, 0]))
        else:
            val *= 0.5 
            ham_terms.append(Op(r"n n", [2*i, 2*k], val, qn=[0, 0]))
            ham_terms.append(Op(r"n n", [2*i+1, 2*k+1], val, qn=[0, 0]))
            ham_terms.append(Op(r"n n", [2*i, 2*k+1], val, qn=[0, 0]))
            ham_terms.append(Op(r"n n", [2*i+1, 2*k], val, qn=[0, 0]))
    basis = [BasisSimpleElectron(i) for i in range(n_spin)]
    model = Model(basis=basis, ham_terms=ham_terms)
    mpo = Mpo(model, algo="qr")
    return [w.transpose(0, 3, 1, 2) for w in mpo.matrices], model

# ==============================================================================
#  MAIN DRIVER
# ==============================================================================
class FixedRankInterleavedSolver:
    def __init__(self, mol, Lz, Nz, basis_cfg):
        self.mol = mol; self.Lz = Lz; self.Nz = Nz; self.basis_cfg = basis_cfg
        self.Enuc = mol.nuclear_repulsion_energy()
        self.d_stack = None
        self._setup_primitive_basis()

    def _setup_primitive_basis(self):
        print("-> Setting up Primitive Basis & Kernels...")
        self.Hcore_init, self.z, self.dz, _, self.C_list, _, _, _ = build_method2(
            self.mol, Lz=self.Lz, Nz=self.Nz, M=1, s_exps=self.basis_cfg.get('s'), verbose=False, dvr_method='sine'
        )
        nuclei = self.mol.to_tuples()
        self.alphas, self.centers, self.labels = make_xy_spd_primitive_basis(
            nuclei, self.basis_cfg.get('s'), self.basis_cfg.get('p', []), self.basis_cfg.get('d', [])
        )
        self.S_prim = overlap_2d_cartesian(self.alphas, self.centers, self.labels)
        self.T_prim = kinetic_2d_cartesian(self.alphas, self.centers, self.labels)
        self.n_ao_2d = len(self.alphas)
        self.K_h = []; self.Kx_h = []
        n2 = self.n_ao_2d**2
        for h in range(self.Nz):
            dz_val = h * self.dz
            eri_tensor = eri_2d_cartesian_with_p(self.alphas, self.centers, self.labels, delta_z=dz_val)
            self.K_h.append(eri_tensor.reshape(n2, n2))
            self.Kx_h.append(eri_tensor.transpose(0, 2, 1, 3).reshape(n2, n2))
        self.ERIop = CollocatedERIOp.from_kernels(N=self.n_ao_2d, Nz=self.Nz, dz=self.dz, K_h=self.K_h, Kx_h=self.Kx_h)
        _, self.Kz_grid, _ = sine_dvr_1d(-self.Lz, self.Lz, self.Nz)
        self.h1_nm_func = build_h1_nm(
            self.Kz_grid, self.S_prim, self.T_prim, self.z, 
            lambda zz: V_en_sp_total_at_z(self.alphas, self.centers, self.labels, nuclei, zz)
        )
        self.d_stack = np.vstack([self.C_list[n][:, 0] for n in range(self.Nz)])

    def _get_noisy_guess(self, noise=1e-3):
        n_spin = 2 * self.Nz
        d = 2; mps_guess = []
        filled = 0
        for i in range(n_spin):
            vec = np.zeros((d, 1, 1))
            if filled < self.mol.nelec: vec[1, 0, 0] = 1.0; filled += 1
            else: vec[0, 0, 0] = 1.0
            vec += (np.random.rand(d, 1, 1) - 0.5) * noise
            vec /= np.linalg.norm(vec)
            mps_guess.append(vec)
        return mps_guess
    
    # --------------------------------------------------------------------------
    #  CORRECTED REBUILDER (Fixed Off-By-One Error)
    # --------------------------------------------------------------------------
    def rebuild_stacks_for_step_i(self, i, mps, mpo):
        # 
        
        # E (Left Environment) for bond (i, i+1)
        # Must contain contractions of sites 0, 1, ..., i-1.
        # Top element E[-1] is "Environment Left of i".
        E = [mps_lib.initial_E(mpo[0])]
        
        # FIX: range(i) stops at i-1. 
        # For i=0 (first site), range(0) is empty -> E=[Vac]. Correct.
        for k in range(i):
            E_next = mps_lib.contract_from_left(mpo[k], mps[k], E[-1], mps[k])
            E.append(E_next)
            
        # F (Right Environment) for bond (i, i+1)
        # Must contain contractions of sites N-1, ..., i+2.
        # Top element F[-1] is "Environment Right of i+1".
        F = [mps_lib.initial_F(mpo[-1])]
        N = len(mps)
        
        # range(N-1, i+1, -1) stops at i+2.
        # For i=0, stops at 2. Contracts N-1...2. F[-1] is Env(2..N). Correct.
        for k in range(N - 1, i + 1, -1):
            F_next = mps_lib.contract_from_right(mpo[k], mps[k], F[-1], mps[k])
            F.append(F_next)
             
        return E, F

    def run(self, max_sweeps=20, dmrg_bond_dim=20, trust_radius=0.1, warmup_sweeps=5):
        print("="*60)
        print(f"Fixed-Rank Interleaved GDVR")
        print("="*60)
        
        # 1. Init MPO
        Hcore = rebuild_Hcore_from_d(self.d_stack, self.z, self.Kz_grid, self.S_prim, self.T_prim, 
                                     self.alphas, self.centers, self.labels, self.mol.to_tuples())
        C_list = [self.d_stack[n].reshape(-1, 1) for n in range(self.Nz)]
        V_coul, _ = eri_JK_from_kernels_M1(C_list, self.K_h, self.Kx_h)
        mpo_tensors, _ = build_mpo_from_integrals(Hcore, np.array(V_coul), self.Nz, dmrg_bond_dim)
        mps_tensors = self._get_noisy_guess()

        # Warmup
        print(f"\n[Phase 1] Warmup ({warmup_sweeps} sweeps)...")
        solver = mps_lib.DMRG(mpo_tensors, D=dmrg_bond_dim, nsweeps=warmup_sweeps, init_guess=mps_tensors)
        solver.run()
        mps_tensors = [t.copy() for t in solver.ground_state.Bs]
        print(f"  -> Warmup Energy: {solver.e_tot + self.Enuc:.8f} Ha")
        
        # Initialize Envs for Step i=0
        E, F = self.rebuild_stacks_for_step_i(0, mps_tensors, mpo_tensors)

        print(f"\n[Phase 2] Starting Loop...")
        Energy = solver.e_tot
        
        for sweep in range(max_sweeps):
            print(f"\n[Sweep {sweep+1}/{max_sweeps}]")
            n_sites = len(mps_tensors)

            # --- Forward Sweep ---
            for i in range(n_sites - 1):
                # 1. Optimize MPS Bond
                Energy, mps_tensors[i], mps_tensors[i+1], trunc, states = mps_lib.optimize_two_sites(
                    mps_tensors[i], mps_tensors[i+1],
                    mpo_tensors[i], mpo_tensors[i+1],
                    E[-1], F[-1], dmrg_bond_dim, 'right'
                )
                print(f"Sweep {sweep} (->) Sites {i},{i+1:<2} Energy {Energy + self.Enuc:.8f}")

                # 2. Optimize AO
                rebuild_needed = False
                if i % 2 == 0:
                    k_slice = i // 2
                    solver.ground_state.Bs = mps_tensors 
                    P, D = gdvr_dmrg_scf.extract_rdms_for_helper(solver, self.Nz, verbose=False)
                    nh = gdvr_dmrg_scf.DMRGNewtonHelper(self.h1_nm_func, self.S_prim, self.ERIop, D)
                    delta_d, lam, g_n = nh.kkt_step_slice(k_slice, self.d_stack, P, self.S_prim, ridge=0.5)
                    
                    snorm = np.sqrt(delta_d @ self.S_prim @ delta_d)
                    if snorm > trust_radius: delta_d *= (trust_radius / snorm)
                    
                    d_new = self.d_stack[k_slice] + delta_d
                    d_new /= np.linalg.norm(self.S_prim @ d_new)
                    if d_new @ self.S_prim @ self.d_stack[k_slice] < 0: d_new *= -1.0
                    self.d_stack[k_slice] = d_new
                    rebuild_needed = True

                if rebuild_needed:
                    Hcore = rebuild_Hcore_from_d(self.d_stack, self.z, self.Kz_grid, self.S_prim, self.T_prim, 
                                                 self.alphas, self.centers, self.labels, self.mol.to_tuples())
                    C_list = [self.d_stack[n].reshape(-1, 1) for n in range(self.Nz)]
                    V_coul, _ = eri_JK_from_kernels_M1(C_list, self.K_h, self.Kx_h)
                    mpo_tensors, _ = build_mpo_from_integrals(Hcore, np.array(V_coul), self.Nz, dmrg_bond_dim)
                    
                    # Rebuild Envs for NEXT step (i+1)
                    # We pass 'i+1' because the next iteration of loop will be i+1
                    E, F = self.rebuild_stacks_for_step_i(i + 1, mps_tensors, mpo_tensors)
                
                else:
                    # Zipper
                    E.append(mps_lib.contract_from_left(mpo_tensors[i], mps_tensors[i], E[-1], mps_tensors[i]))
                    if len(F) > 0: F.pop()

            # --- Backward Sweep ---
            F = [mps_lib.initial_F(mpo_tensors[-1])]
            E.pop()
            
            for i in range(n_sites - 2, -1, -1):
                Energy, mps_tensors[i], mps_tensors[i+1], trunc, states = mps_lib.optimize_two_sites(
                    mps_tensors[i], mps_tensors[i+1],
                    mpo_tensors[i], mpo_tensors[i+1],
                    E[-1], F[-1], dmrg_bond_dim, 'left'
                )
                print(f"Sweep {sweep} (<-) Sites {i},{i+1:<2} Energy {Energy + self.Enuc:.8f}")
                F.append(mps_lib.contract_from_right(mpo_tensors[i+1], mps_tensors[i+1], F[-1], mps_tensors[i+1]))
                E.pop()
            
            E = [mps_lib.initial_E(mpo_tensors[0])]
            F.pop()

        return Energy + self.Enuc

if __name__ == "__main__":
    charges = [1.0, 1.0, 1.0, 1.0]
    coords = [[0.0, 0.0, 0.91], [0.0, 0.0, -0.91], [0.0, 0.0, -3.6], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4)
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    
    solver = FixedRankInterleavedSolver(mol, Lz=8.0, Nz=32, basis_cfg=basis_cfg)
    solver.run(max_sweeps=10, dmrg_bond_dim=20, trust_radius=0.1, warmup_sweeps=5)