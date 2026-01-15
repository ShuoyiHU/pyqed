import numpy as np
import torch
import logging
import time
from pyqed.qchem.dvr.hybrid_gauss_dvr_method_sweep import (
    Molecule, build_method2, make_xy_spd_primitive_basis, 
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,
    scf_rhf_method2, sine_dvr_1d, eri_JK_from_kernels_M1,
    build_h1_nm, V_en_sp_total_at_z, CollocatedERIOp, rebuild_Hcore_from_d,
    SweepNewtonHelper, sweep_optimize_driver
)
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.model import Model
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.Operator import Op
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.basis import BasisSimpleElectron
from pyqed.mps.shuoyi_mpo_test.renormalizer_modified.light_automatic_mpo import Mpo
import pyqed.mps.mps as mps_lib

# helper for the post dmrg AO optimization
import gdvr_dmrg_scf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PATCHES ---
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

def patched_initial_F(W):
    F = np.zeros((W.shape[1], 1, 1))
    F[0] = 1.0 
    return F
mps_lib.initial_F = patched_initial_F

def get_jw_term_robust(op_str_list, indices, factor):
    chain = list(zip(indices, op_str_list))
    n = len(chain)
    swaps = 0
    for i in range(n):
        for j in range(0, n-i-1):
            if chain[j][0] > chain[j+1][0]:
                chain[j], chain[j+1] = chain[j+1], chain[j]
                swaps += 1
    sorted_indices = [x[0] for x in chain]
    sorted_ops = [x[1] for x in chain]
    final_indices = []
    final_ops_str = []
    parity = 0 
    extra_sign = 1
    for k in range(n):
        site = sorted_indices[k]
        op_sym = sorted_ops[k]
        if k > 0:
            prev_site = sorted_indices[k-1]
            if parity % 2 == 1:
                for z_site in range(prev_site + 1, site):
                    final_indices.append(z_site)
                    final_ops_str.append("sigma_z")
        ops_to_right = n - 1 - k
        if (op_sym == "a") and (ops_to_right % 2 == 1):
            extra_sign *= -1
        final_indices.append(site)
        final_ops_str.append(op_sym)
        parity += 1
    final_op_string = " ".join(final_ops_str)
    return Op(final_op_string, final_indices, factor=factor * ((-1) ** swaps) * extra_sign)

def get_noisy_hf_guess(n_elec, n_spin, noise=1e-3):
    d = 2; mps_guess = []
    filled_count = 0
    for i in range(n_spin):
        vec = np.zeros((d, 1, 1))
        if filled_count < n_elec: 
            vec[1, 0, 0] = 1.0; filled_count += 1
        else: 
            vec[0, 0, 0] = 1.0
        vec += (np.random.rand(d, 1, 1) - 0.5) * noise
        vec /= np.linalg.norm(vec)
        mps_guess.append(vec)
    return mps_guess

def align_orbital_phases(d_old, d_new, S_prim):
    Nz = d_old.shape[0]
    min_overlap = 1.0
    for n in range(Nz):
        overlap = float(d_old[n].T @ S_prim @ d_new[n])
        if overlap < 0:
            d_new[n] *= -1.0
            overlap = -overlap
        min_overlap = min(min_overlap, overlap)
    return d_new, min_overlap

# --- MAIN LOOP ---
def run_gdvr_dmrg_loop(
    mol, Lz, Nz, basis_cfg,
    pre_opt_cycles=10,      
    dmrg_cycles=3,          
    dmrg_bond_dim=20,
    dmrg_sweeps=10,
    post_dmrg_opt_cycles=5
):
    print("="*60)
    print(f"GDVR-DMRG Self-Consistent Loop (Final Robust)")
    print(f"System: {mol.nelec} electrons, Nz={Nz}, Lz={Lz}")
    print("="*60)

    # --- Phase A: Initial HF ---
    s_exps = basis_cfg.get('s'); p_exps = basis_cfg.get('p', []); d_exps = basis_cfg.get('d', [])
    Hcore, z, dz, E_slices, C_list, _, _, _ = build_method2(
        mol, Lz=Lz, Nz=Nz, M=1, s_exps=s_exps, p_exps=p_exps, d_exps=d_exps, 
        verbose=False, dvr_method='sine'
    )
    
    nuclei = mol.to_tuples()
    alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, p_exps, d_exps)
    S_prim = overlap_2d_cartesian(alphas, centers, labels)
    T_prim = kinetic_2d_cartesian(alphas, centers, labels)
    n_ao_2d = len(alphas)
    
    K_h = []; Kx_h = []
    for h in range(Nz):
        dz_val = h * dz
        eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
        n2 = n_ao_2d * n_ao_2d
        K_h.append(eri_tensor.reshape(n2, n2))
        Kx_h.append(eri_tensor.transpose(0, 2, 1, 3).reshape(n2, n2))

    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
    Enuc = mol.nuclear_repulsion_energy()
    Etot, _, Cmo, P, _ = scf_rhf_method2(Hcore, ERI_J, ERI_K, Nz, 1, mol.nelec, Enuc, verbose=False)
    print(f"  -> Initial HF Energy: {Etot:.8f} Ha")
    
    _, Kz_grid, _ = sine_dvr_1d(-Lz, Lz, Nz)
    ERIop = CollocatedERIOp.from_kernels(N=n_ao_2d, Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
    h1_nm_func = build_h1_nm(Kz_grid, S_prim, T_prim, z, 
                             lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz))
    
    d_stack = np.vstack([C_list[n][:, 0] for n in range(Nz)])

    # --- Phase A.5: Pre-Optimization ---
    if pre_opt_cycles > 0:
        print(f"\n[Phase A.5] Pre-optimizing AOs (HF level)...")
        nh_sweep = SweepNewtonHelper(h1_nm_func, S_prim, ERIop)
        for pcyc in range(pre_opt_cycles):
            P_slice = P.reshape(Nz, 1, Nz, 1)[:, 0, :, 0].copy()
            d_stack = sweep_optimize_driver(
                nh_sweep, d_stack, P_slice, S_prim,
                n_cycles=5, ridge=0.5, trust_step=1.0, trust_radius=2.0, verbose=False
            )
            Hcore_curr = rebuild_Hcore_from_d(d_stack, z, Kz_grid, S_prim, T_prim, alphas, centers, labels, nuclei)
            C_list_curr = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
            ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
            Etot, _, Cmo, P, _ = scf_rhf_method2(Hcore_curr, ERI_J, ERI_K, Nz, 1, mol.nelec, Enuc, verbose=False)
            if (pcyc + 1) % 2 == 0: print(f"   Cycle {pcyc+1}: HF Energy = {Etot:.8f} Ha")

    # --- Phase B: Self-Consistent Loop ---
    total_energy_history = []
    last_mps_tensors = None 
    d_stack_old = d_stack.copy()
    
    for cycle in range(dmrg_cycles):
        print(f"\n[Macro Cycle {cycle+1}/{dmrg_cycles}]")
        
        # ALIGN PHASES
        d_stack, match_quality = align_orbital_phases(d_stack_old, d_stack, S_prim)
        d_stack_old = d_stack.copy()
        
        if match_quality < 0.5:
            print(f"  [Warning] Orbitals changed significantly (Min Overlap={match_quality:.4f}). Disabling Warm Start.")
            last_mps_tensors = None
        
        # 1. Rebuild Hamiltonian
        print("  1. Rebuilding Hamiltonian...")
        Hcore_curr = rebuild_Hcore_from_d(d_stack, z, Kz_grid, S_prim, T_prim, alphas, centers, labels, nuclei)
        C_list_curr = [d_stack[n].reshape(-1, 1) for n in range(Nz)]
        V_coul, V_exch = eri_JK_from_kernels_M1(C_list_curr, K_h, Kx_h)
        V_coul = np.array(V_coul) 
        
        # 2. Construct MPO
        print("  2. Constructing MPO...")
        ham_terms = []
        n_spin = 2 * Nz
        cutoff = 1e-10
        
        rows, cols = np.nonzero(np.abs(Hcore_curr) > cutoff)
        for i, j in zip(rows, cols):
            val = Hcore_curr[i, j]
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i, 2*j], val))
            ham_terms.append(get_jw_term_robust([r"a^\dagger", "a"], [2*i+1, 2*j+1], val))
            
        rows, cols = np.nonzero(np.abs(V_coul) > cutoff)
        for i, k in zip(rows, cols):
            if i == k: 
                val = V_coul[i, k]
                ham_terms.append(Op("n", 2*i) * Op("n", 2*i+1) * val)
            else: 
                val = 0.5 * V_coul[i, k]
                ham_terms.append(Op("n", 2*i) * Op("n", 2*k) * val)     
                ham_terms.append(Op("n", 2*i+1) * Op("n", 2*k+1) * val) 
                ham_terms.append(Op("n", 2*i) * Op("n", 2*k+1) * val)   
                ham_terms.append(Op("n", 2*i+1) * Op("n", 2*k) * val)   
                
        basis = [BasisSimpleElectron(i) for i in range(n_spin)]
        model = Model(basis=basis, ham_terms=ham_terms)
        mpo = Mpo(model, algo="qr")
        mpo_dmrg = [w.transpose(0, 3, 1, 2) for w in mpo.matrices]
        
        # 3. Run DMRG
        print(f"  3. Running DMRG (D={dmrg_bond_dim})...")
        
        if last_mps_tensors is None:
            print("     -> Using fresh Noisy HF guess")
            mps_guess = get_noisy_hf_guess(mol.nelec, n_spin, noise=1e-3)
        else:
            print(f"     -> Warm starting from previous MPS")
            mps_guess = [t.copy() for t in last_mps_tensors]
        
        solver = mps_lib.DMRG(mpo_dmrg, D=dmrg_bond_dim, nsweeps=dmrg_sweeps, init_guess=mps_guess)
        solver.run()
        
        # Recalc Energy to avoid bug
        try:
            psi_tensors = solver.ground_state.Bs
            e_elec_recalc = mps_lib.expect_mps(psi_tensors, solver.H, psi_tensors)
            e_dmrg = np.real(e_elec_recalc) + Enuc
        except:
            e_dmrg = solver.e_tot + Enuc

        last_mps_tensors = solver.ground_state.Bs
        total_energy_history.append(e_dmrg)
        print(f"     -> Final Cycle Energy: {e_dmrg:.8f} Ha")
        
        # 4. Post-DMRG AO Optimization

        if cycle < dmrg_cycles - 1: 
            print("  4. Re-optimizing AOs using DMRG 1-RDM (with Exchange)...")
            
            d_stack = gdvr_dmrg_scf.dmrg_ao_optimization_step(
                mol, d_stack, None, S_prim, ERIop, h1_nm_func, 
                z, Kz_grid, T_prim, alphas, centers, labels, K_h, Kx_h, 
                solver=solver,
                Enuc=Enuc,
                n_cycles=post_dmrg_opt_cycles,
                verbose=True
            )

    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    for i, e in enumerate(total_energy_history):
        print(f"Cycle {i+1}: {e:.8f} Ha")

if __name__ == "__main__":
    charges = [1.0, 1.0, 1.0, 1.0]
    coords = [[0.0, 0.0, 0.91], [0.0, 0.0, -0.91], [0.0, 0.0, -3.6], [0.0, 0.0, 3.6]]
    mol = Molecule(charges, coords, nelec=4)
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS}
    
    run_gdvr_dmrg_loop(
        mol, Lz=8.0, Nz=32, basis_cfg=basis_cfg,
        pre_opt_cycles=10,    
        dmrg_cycles=10,         
        dmrg_bond_dim=20,
        dmrg_sweeps=20,
        post_dmrg_opt_cycles=10 
    )