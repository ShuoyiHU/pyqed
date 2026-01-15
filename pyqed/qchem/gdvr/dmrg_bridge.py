import numpy as np
import torch
import time
import os
from pyscf import gto, scf
PYSCF_AVAILABLE = True

from pyqed.qchem.dvr.hybrid_gauss_dvr_method_sweep import (
    Molecule, build_method2, make_xy_spd_primitive_basis, 
    rebuild_Hcore_from_d, 
    sweep_optimize_driver, CollocatedERIOp, SweepNewtonHelper,
    build_h1_nm, V_en_sp_total_at_z, 
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,
    scf_rhf_method2, sine_dvr_1d, eri_JK_from_kernels_M1
)
import pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_extrapolate as hg_int

class DMRGInputGenerator:
    def __init__(self, molecule, Lz, Nz, M=1):
        self.mol = molecule
        self.Lz = Lz
        self.Nz = Nz
        self.M = M
        self.final_hf_energy = None
        self.t_ij = None
        self.V_coulomb = None
        self.V_exchange = None
        
    def optimize_aos(self, basis_config, alt_cycles=15, sweep_iter=10):
        """
        Runs the full Alternating Optimization (Sweep <-> SCF).
        """
        print(f"--- Starting AO Optimization (Nz={self.Nz}, Lz={self.Lz}) ---")
        
        # 0. Clear Cache
        hg_int._VEN_SPLINES = {}
        hg_int._ERI_SPLINES = None
        
        # 1. Initial Build
        s_exps = basis_config.get('s')
        p_exps = basis_config.get('p', [])
        d_exps = basis_config.get('d', [])
        
        (Hcore, z, dz, E_slices, C_list, _, _, _) = build_method2(
            self.mol, Lz=self.Lz, Nz=self.Nz, M=self.M,
            s_exps=s_exps, p_exps=p_exps, d_exps=d_exps, 
            verbose=False, dvr_method='sine'
        )
        
        # 2. Setup Primitives
        nuclei = self.mol.to_tuples()
        alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, p_exps, d_exps)
        S_prim = overlap_2d_cartesian(alphas, centers, labels)
        T_prim = kinetic_2d_cartesian(alphas, centers, labels)
        n_ao_2d = len(alphas)

        print("Precomputing 2D Integral Kernels...")
        K_h = []
        Kx_h = []
        for h in range(self.Nz):
            dz_val = h * dz
            eri_tensor = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=dz_val)
            n2 = n_ao_2d * n_ao_2d
            # Standard
            K_h.append(eri_tensor.reshape(n2, n2))
            # Transposed
            eri_perm = eri_tensor.transpose(0, 2, 1, 3)
            Kx_h.append(eri_perm.reshape(n2, n2))

        # 3. Initial SCF (Seed Density)
        print("Running Initial SCF...")
        ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)
        Enuc = self.mol.nuclear_repulsion_energy()
        
        Etot, eps, Cmo, P, info = scf_rhf_method2(
            Hcore, ERI_J, ERI_K, self.Nz, self.M,
            nelec=self.mol.nelec, Enuc=Enuc,
            conv=1e-6, max_iter=50, verbose=False
        )
        print(f"  [Init SCF] E = {Etot:.8f} Ha")

        # 4. Alternating Optimization Loop
        
        # Prepare Helper Data
        _, Kz_grid, _ = sine_dvr_1d(-self.Lz, self.Lz, self.Nz)
        ERIop = CollocatedERIOp.from_kernels(N=n_ao_2d, Nz=self.Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
        
        # Stack C_list
        d_stack = np.vstack([C_list[n][:, 0] for n in range(self.Nz)])
        # Ensure normalization
        for n in range(self.Nz):
            dn = d_stack[n]
            d_stack[n] = dn / np.sqrt(float(dn.T @ (S_prim @ dn)))

        E_history = [Etot]

        print(f"Starting {alt_cycles} Alternating Cycles...")
        for cyc in range(1, alt_cycles + 1):
            
            # A. Prepare Newton Helper with CURRENT orbitals/grid
            #    (Recalculate V_eff if needed, but build_h1_nm is static wrt density)
            h1_nm_func = build_h1_nm(
                Kz_grid, # Real Kz
                S_prim, T_prim, z, 
                lambda zz: V_en_sp_total_at_z(alphas, centers, labels, nuclei, zz)
            )
            nh = SweepNewtonHelper(h1_nm_func, S_prim, ERIop)
            
            # B. Extract Density Slice from current P
            P_slice = P.reshape(self.Nz, self.M, self.Nz, self.M)[:, 0, :, 0].copy()
            
            # C. Run Sweep (Optimize d_stack)
            d_stack = sweep_optimize_driver(
                nh, d_stack, P_slice, S_prim,
                n_cycles=sweep_iter, 
                ridge=0.5, trust_step=1.0, trust_radius=2.0, # params, larger give faster convergence speed, while might harm monotonic decrement. TODO: adaptive update of radius based on (E_new - E_previous)/(\Delta E_predicted)
                verbose=False
            )
            
            # D. Update C_list
            self.C_list_opt = [d_stack[n].reshape(-1, 1) for n in range(self.Nz)]
            
            # E. Rebuild Integrals (Hcore, J, K) based on NEW orbitals
            Hcore_new = rebuild_Hcore_from_d(
                d_stack, z, Kz_grid, S_prim, T_prim, 
                alphas, centers, labels, nuclei
            )
            ERI_J, ERI_K = eri_JK_from_kernels_M1(self.C_list_opt, K_h, Kx_h)
            
            # F. Run SCF (Optimize P)
            Etot, eps, Cmo, P, info = scf_rhf_method2(
                Hcore_new, ERI_J, ERI_K, self.Nz, self.M,
                nelec=self.mol.nelec, Enuc=Enuc,
                conv=1e-7, max_iter=60, verbose=False
            )
            
            print(f"  [Cycle {cyc}] E = {Etot:.8f} Ha  (dE = {Etot - E_history[-1]:.2e})")
            
            if abs(Etot - E_history[-1]) < 1e-7:
                print("  Converged.")
                break
            E_history.append(Etot)

        # 5. Store Final Results
        self.final_hf_energy = Etot
        self.t_ij = Hcore_new
        self.V_coulomb = np.array(ERI_J)
        self.V_exchange = np.array(ERI_K)
        self.final_Cmo = Cmo  # Store final MO coefficients
        self.enuc = Enuc
        
    def run_pyscf_benchmark(self, basis='sto-3g'):
        if not PYSCF_AVAILABLE: return
        print(f"\n--- PySCF Benchmark ({basis}) ---")
        atoms = [[int(Z), self.mol.coords[i]] for i, Z in enumerate(self.mol.charges)]
        pmol = gto.M(atom=atoms, basis=basis, unit='Bohr', verbose=0)
        mf = scf.RHF(pmol)
        e_pyscf = mf.kernel()
        print(f"  => PySCF Energy:    {e_pyscf:.8f} Ha")
        print(f"  => Ours (Hybrid):   {self.final_hf_energy:.8f} Ha")
        print(f"  => Difference:      {self.final_hf_energy - e_pyscf:.8f} Ha")

    def save_to_file(self, filename="dmrg_input.pt"):
        data = {
            "t_ij": torch.tensor(self.t_ij),
            "V_coulomb": torch.tensor(self.V_coulomb),
            "V_exchange": torch.tensor(self.V_exchange),
            "nelec": self.mol.nelec,
            "Nz": self.Nz,
            "hf_energy": self.final_hf_energy,
            "C_mo": torch.tensor(self.final_Cmo),
            "enuc": self.enuc
        }
        torch.save(data, filename)
        print(f"Shape of t_ij: {self.t_ij.shape}")
        print(f"Shape of V_coulomb: {self.V_coulomb.shape}")
        print(f"Shape of V_exchange: {self.V_exchange.shape}")
        print(f"Saved inputs to {filename}")

if __name__ == "__main__":
    # Test using H2
    charges = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # coords = [[0.0, 0.0, 0.91], [0.0, 0.0, -0.91], [0.0, 0.0, -3.6], [0.0, 0.0, 3.6]]
    coords = [[0.0, 0.0, -3.5],[0.0, 0.0, -2.5],[0.0, 0.0, -1.5],[0.0, 0.0, -0.5],[0.0, 0.0, 0.5],[0.0, 0.0, 1.5],[0.0, 0.0, 2.5],[0.0, 0.0, 3.5]]
    coords = [[0.0, 0.0, -1],[0.0, 0.0, -3],[0.0, 0.0, -5],[0.0, 0.0, -7],[0.0, 0.0, 1],[0.0, 0.0, 3],[0.0, 0.0, 5],[0.0, 0.0, 7]]
    mol = Molecule(charges, coords, nelec=8)
    
    # uncontracted 631g (without p)
    S_EXPS = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
    basis_cfg = {'s': S_EXPS, 'p': [], 'd': []}
    
    # Run Generator
    gen = DMRGInputGenerator(mol, Lz=10.0, Nz=32) 
    gen.optimize_aos(basis_cfg, alt_cycles=10, sweep_iter=10)
    
    gen.run_pyscf_benchmark('sto-3g') 
    gen.save_to_file("h8_dmrg_input_nz_32_far.pt")
