import time
import numpy as np
import matplotlib.pyplot as plt

# Import your modules
import pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_extrapolate as hg_int
from pyqed.qchem.dvr.hybrid_gauss_dvr_method_sweep import (
    Molecule, build_method2, make_xy_spd_primitive_basis,
    overlap_2d_cartesian, kinetic_2d_cartesian, eri_2d_cartesian_with_p,
    build_h1_nm, CollocatedERIOp, sweep_optimize_driver, SweepNewtonHelper
)

# 1. Instrumented Helper to Isolate "Build" Time
class TimerHelper(SweepNewtonHelper):
    def __init__(self, h1_nm, S_prim, eri_op):
        super().__init__(h1_nm, S_prim, eri_op)
        self.t_build = 0.0

    def kkt_step_slice(self, n, d_stack, P_slice, S_prim, ridge=0.0):
        # Measure ONLY the Gradient + Hessian Construction
        t0 = time.time()
        self.get_gradient_slice_onthefly(n, d_stack, P_slice)
        self.get_diagonal_hessian_block_sparse(n, d_stack, P_slice)
        self.t_build += (time.time() - t0)
        
        # Return dummy values to keep the driver running
        N = d_stack.shape[1]
        return np.zeros(N), 0.0, np.zeros(N)

def run_proof_of_scaling():
    print(f"{'='*60}")
    print(f"{'PROOF OF N^4 SCALING (S-Orbitals Only)':^60}")
    print(f"{'='*60}")
    print(f"{'N_AO':<6} | {'Build(s)':<10} | {'Ratio':<8} | {'N^4 Theory':<10}")
    print("-" * 60)

    # Test N_AO from 10 to 60.
    # We use 'n' s-orbitals on 1 atom to get exactly N_AO = n.
    cases = [10, 20, 30, 40, 50, 60]
    Nz = 20  # Reduced Nz to speed up the loop (math/overhead ratio is independent of Nz)
    
    times = []
    base_time = None
    base_n = None

    for n_s in cases:
        # Clear Integral Cache
        hg_int._VEN_SPLINES = {}
        hg_int._ERI_SPLINES = None

        # 1. Setup System (S-Orbitals only -> Fast Integrals)
        s_exps = [1.0 + i*0.1 for i in range(n_s)]
        mol = Molecule([1.0], [[0.,0.,0.]], nelec=2) # 1 Atom
        
        # 2. Build Basis & Operators (Quietly)
        nuclei = mol.to_tuples()
        alphas, centers, labels = make_xy_spd_primitive_basis(nuclei, s_exps, [], [])
        N_AO = len(alphas)
        z_grid = np.linspace(-2, 2, Nz); dz = z_grid[1]-z_grid[0]
        
        S_prim = overlap_2d_cartesian(alphas, centers, labels)
        T_prim = kinetic_2d_cartesian(alphas, centers, labels)
        
        # 3. Precompute Kernels (Fast for S-orbitals)
        K_h = []
        Kx_h = []
        for h in range(Nz):
            eri = eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=h*dz)
            K_h.append(eri.reshape(N_AO**2, N_AO**2))
            Kx_h.append(eri.transpose(0, 2, 1, 3).reshape(N_AO**2, N_AO**2))

        ERIop = CollocatedERIOp.from_kernels(N=N_AO, Nz=Nz, dz=dz, K_h=K_h, Kx_h=Kx_h)
        h1_nm = np.zeros((Nz, Nz, N_AO, N_AO), dtype=float)

        # 4. Measure
        nh = TimerHelper(h1_nm, S_prim, ERIop)
        d_stack = np.random.rand(Nz, N_AO)
        P_slice = np.eye(Nz)
        
        sweep_optimize_driver(nh, d_stack, P_slice, S_prim, n_cycles=1, verbose=False)
        
        # 5. Analyze
        t_curr = nh.t_build
        if base_time is None:
            base_time = t_curr
            base_n = N_AO
            ratio = 1.0
            theory = 1.0
        else:
            ratio = t_curr / base_time
            theory = (N_AO / base_n)**4
            
        print(f"{N_AO:<6} | {t_curr:<10.4f} | {ratio:<8.2f} | {theory:<10.1f}")
        times.append(t_curr)

    return cases, times

if __name__ == "__main__":
    n_vals, t_vals = run_proof_of_scaling()
    
    # Optional Plotting
    try:
        plt.figure()
        plt.plot(n_vals, t_vals, 'o-', label='Measured Time')
        
        # Fit A + B*N^4
        # Use last point to estimate B
        B = (t_vals[-1] - t_vals[0]) / (n_vals[-1]**4 - n_vals[0]**4)
        A = t_vals[0] - B * n_vals[0]**4
        y_fit = [A + B * n**4 for n in n_vals]
        
        plt.plot(n_vals, y_fit, '--', label=f'Fit: {A:.2f} + {B:.2e} * N^4')
        plt.xlabel('N_AO'); plt.ylabel('Build Time (s)')
        plt.title('Method 2 Scaling Proof')
        plt.legend()
        plt.show()
    except:
        pass