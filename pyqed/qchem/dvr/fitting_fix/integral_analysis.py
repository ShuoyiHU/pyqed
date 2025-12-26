# # import numpy as np
# # import matplotlib.pyplot as plt
# # import sys
# # import os

# # # --- Import your module ---
# # # Ensure current directory is in path so we can import the local file
# # sys.path.insert(0, os.getcwd())

# # try:
# #     import pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals as dvr_int
# # except ImportError:
# #     # Fallback if running directly in the folder with the file
# #     import hybrid_gauss_dvr_integrals_add_d_orbitals as dvr_int

# # # ==========================================
# # # 1. Setup Basis and Grid
# # # ==========================================

# # # Standard H exponents (from your configuration)
# # s_exps = [18.73113696, 2.825394365, 0.6401216923, 0.1612777588]
# # p_exps = [1.0] # Add dummy p to see behavior
# # d_exps = [1.0] # Add dummy d to see behavior

# # # Dummy geometry (Nucleus at origin)
# # nuclei = [(1.0, 0.0, 0.0, 0.0)] # Charge 1, at (0,0,0)

# # # Generate primitives
# # alphas, centers, labels = dvr_int.make_xy_spd_primitive_basis(
# #     nuclei, 
# #     exps_s=np.array(s_exps),
# #     exps_p=np.array(p_exps),
# #     exps_d=np.array(d_exps)
# # )

# # print(f"Basis generated: {len(alphas)} functions")
# # for i, l in enumerate(labels):
# #     print(f"  {i}: {l.kind}")

# # # Define dz range (Logarithmic to zoom in on 0)
# # # From 1e-5 to 0.1
# # dz_vals = np.logspace(-5, -1, 100)

# # # ==========================================
# # # 2. Calculate V_en Trends
# # # ==========================================
# # print("\n--- Calculating V_en Trends ---")

# # # Storage for diagonal elements of specific orbitals
# # # We'll pick one S, one P, one D to track
# # idx_s = [i for i, l in enumerate(labels) if 's' in l.kind][0]
# # idx_p = [i for i, l in enumerate(labels) if 'p' in l.kind][0]
# # idx_d = [i for i, l in enumerate(labels) if 'd' in l.kind][0]

# # ven_prony_s = []
# # ven_prony_p = []
# # ven_prony_d = []

# # # Compute Prony Curve (Force usage by calling internal function or ensuring dz > cutoff)
# # # Note: We access the internal _V_en_prony_general directly to see the curve
# # nuc_xy = np.array([0.0, 0.0])

# # for z in dz_vals:
# #     # We only care about the interaction with the nucleus at (0,0)
# #     # The function signature: _V_en_prony_general(alphas, centers, labels, nuc_xy, dz_abs)
# #     mat = dvr_int._V_en_prony_general(alphas, centers, labels, nuc_xy, z)
    
# #     # Value is -Z * Integral (Nucleus charge 1.0)
# #     # The function returns the integral part. Total V = -Z * Integral
# #     val_s = -1.0 * mat[idx_s, idx_s]
# #     val_p = -1.0 * mat[idx_p, idx_p]
# #     val_d = -1.0 * mat[idx_d, idx_d]
    
# #     ven_prony_s.append(val_s)
# #     ven_prony_p.append(val_p)
# #     ven_prony_d.append(val_d)

# # # Compute Analytical Exact Limit (dz=0)
# # mat_exact = dvr_int._V_en_exact_dz0_general(alphas, centers, labels, nuc_xy)
# # exact_s = -1.0 * mat_exact[idx_s, idx_s]
# # exact_p = -1.0 * mat_exact[idx_p, idx_p]
# # exact_d = -1.0 * mat_exact[idx_d, idx_d]

# # # ==========================================
# # # 3. Calculate ERI Trends
# # # ==========================================
# # print("--- Calculating ERI Trends ---")
# # # We will look at diagonal element (ii|ii) (Self-repulsion at distance dz)

# # eri_prony_s = []
# # eri_prony_p = []

# # eri_exact_s = []
# # eri_exact_p = []

# # for z in dz_vals:
# #     # 1. Force Prony: Call eri with dz_tol = 0.0
# #     # The function logic: if abs(delta_z) < dz_tol: Analytical else: Prony
# #     # So dz_tol=0.0 forces Prony
# #     tensor_prony = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=z, dz_tol=0.0)
# #     eri_prony_s.append(tensor_prony[idx_s, idx_s, idx_s, idx_s])
# #     eri_prony_p.append(tensor_prony[idx_p, idx_p, idx_p, idx_p])

# #     # 2. Force Analytical: Call eri with huge dz_tol (so it thinks z is "small enough")
# #     # This evaluates the Hermite/Bessel expansion at distance z
# #     tensor_exact = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=z, dz_tol=10.0)
# #     eri_exact_s.append(tensor_exact[idx_s, idx_s, idx_s, idx_s])
# #     eri_exact_p.append(tensor_exact[idx_p, idx_p, idx_p, idx_p])


# # # ==========================================
# # # 4. Plotting
# # # ==========================================
# # fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# # # --- Plot V_en ---
# # ax = axes[0]
# # ax.plot(dz_vals, ven_prony_s, label='Prony (S)', color='blue')
# # ax.plot(dz_vals, ven_prony_p, label='Prony (P)', color='red')
# # ax.plot(dz_vals, ven_prony_d, label='Prony (D)', color='green')

# # # Add Exact Targets (dz=0)
# # ax.scatter([0], [exact_s], color='blue', marker='*', s=150, label='Exact z=0 (S)', zorder=10)
# # ax.scatter([0], [exact_p], color='red', marker='*', s=150, label='Exact z=0 (P)', zorder=10)
# # ax.scatter([0], [exact_d], color='green', marker='*', s=150, label='Exact z=0 (D)', zorder=10)

# # ax.set_title("Nuclear Attraction $V_{en}$ vs $\Delta z$")
# # ax.set_xlabel("$\Delta z$ (Bohr)")
# # ax.set_ylabel("Energy (Hartree)")
# # ax.set_xscale('log')
# # ax.axvline(0.01, color='k', linestyle='--', alpha=0.5, label='Current Cutoff (0.01)')
# # ax.legend()
# # ax.grid(True, which="both", ls="-")

# # # --- Plot ERI (S) ---
# # ax = axes[1]
# # ax.plot(dz_vals, eri_prony_s, label='Prony (SS|SS)', color='blue')
# # ax.plot(dz_vals, eri_exact_s, label='Analytical (SS|SS)', color='cyan', linestyle='--')
# # ax.set_title("ERI (s-orbitals) vs $\Delta z$")
# # ax.set_xlabel("$\Delta z$ (Bohr)")
# # ax.set_xscale('log')
# # ax.axvline(0.01, color='k', linestyle='--', alpha=0.5)
# # ax.legend()
# # ax.grid(True, which="both", ls="-")

# # # --- Plot ERI (P) ---
# # ax = axes[2]
# # ax.plot(dz_vals, eri_prony_p, label='Prony (PP|PP)', color='red')
# # ax.plot(dz_vals, eri_exact_p, label='Analytical (PP|PP)', color='orange', linestyle='--')
# # ax.set_title("ERI (p-orbitals) vs $\Delta z$")
# # ax.set_xlabel("$\Delta z$ (Bohr)")
# # ax.set_xscale('log')
# # ax.axvline(0.01, color='k', linestyle='--', alpha=0.5)
# # ax.legend()
# # ax.grid(True, which="both", ls="-")

# # plt.tight_layout()
# # plt.savefig("Ven_ERI_Trend_Analysis.png", dpi=150)
# # print("Plot saved to Ven_ERI_Trend_Analysis.png")
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os

# # --- Import your module ---
# sys.path.insert(0, os.getcwd())

# try:
#     import pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_add_d_orbitals as dvr_int
# except ImportError:
#     import hybrid_gauss_dvr_integrals_add_d_orbitals as dvr_int

# # ==========================================
# # 1. Setup Basis
# # ==========================================

# # User's exponents
# s_exps = [30.825394365, 0.6401216923, 0.1612777588]
# nuclei = [(1.0, 0.0, 0.0, 0.0)]

# alphas, centers, labels = dvr_int.make_xy_spd_primitive_basis(
#     nuclei, 
#     exps_s=np.array(s_exps),
#     exps_p=[],
#     exps_d=[]
# )

# # Identify Indices
# # Index 0: Tightest (30.8)
# # Index 1: Medium (0.64)
# # Index 2: Diffuse (0.16)
# idx_tight = 0
# idx_diff  = 1 

# print(f"Analyzing Integrals for:")
# print(f"  Tight   (idx {idx_tight}): exp={alphas[idx_tight]:.4f}")
# print(f"  Diffuse (idx {idx_diff}): exp={alphas[idx_diff]:.4f}")

# # Define dz range (Logarithmic)
# dz_vals = np.logspace(-7, -2, 500) # 1e-7 to 0.1

# # ==========================================
# # 2. Data Collection
# # ==========================================

# # Storage
# ven_data = {
#     'tight': [], 'diff': [], 'cross': []
# }
# eri_data = {
#     'tight': [], 'diff': [], 'cross': []
# }
# # Analytical Targets (at dz=0)
# ven_exact = {}
# eri_exact = {}

# # --- A. Compute Exact Limits at z=0 ---
# nuc_xy = np.array([0.0, 0.0])
# mat_ven_ex = dvr_int._V_en_exact_dz0_general(alphas, centers, labels, nuc_xy)
# ven_exact['tight'] = -1.0 * mat_ven_ex[idx_tight, idx_tight]
# ven_exact['diff']  = -1.0 * mat_ven_ex[idx_diff, idx_diff]
# ven_exact['cross'] = -1.0 * mat_ven_ex[idx_tight, idx_diff]

# # For ERI exact, we force the analytical branch with huge tolerance
# eri_tensor_ex = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=0.0, dz_tol=10.0)
# eri_exact['tight'] = eri_tensor_ex[idx_tight, idx_tight, idx_tight, idx_tight]
# eri_exact['diff']  = eri_tensor_ex[idx_diff, idx_diff, idx_diff, idx_diff]
# eri_exact['cross'] = eri_tensor_ex[idx_tight, idx_tight, idx_diff, idx_diff]

# print("\nExact Limits (z=0):")
# print(f"  Ven Tight: {ven_exact['tight']:.6f}")
# print(f"  Ven Cross: {ven_exact['cross']:.6f}")
# print(f"  ERI Cross: {eri_exact['cross']:.6f}")

# # --- B. Compute Trends (Prony) ---
# print("\nScanning dz values...")
# for z in dz_vals:
#     # 1. Ven (Force Prony)
#     mat_v = dvr_int._V_en_prony_general(alphas, centers, labels, nuc_xy, z)
#     ven_data['tight'].append(-1.0 * mat_v[idx_tight, idx_tight])
#     ven_data['diff'].append(-1.0 * mat_v[idx_diff, idx_diff])
#     ven_data['cross'].append(-1.0 * mat_v[idx_tight, idx_diff])
    
#     # 2. ERI (Force Prony with dz_tol=0.0)
#     # We only compute the specific elements we need to save time? 
#     # Actually calculating the full tensor 100 times is slow for big bases, 
#     # but for 3 functions it's instant.
#     ten_e = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=z, dz_tol=0.0)
#     eri_data['tight'].append(ten_e[idx_tight, idx_tight, idx_tight, idx_tight])
#     eri_data['diff'].append(ten_e[idx_diff, idx_diff, idx_diff, idx_diff])
#     eri_data['cross'].append(ten_e[idx_tight, idx_tight, idx_diff, idx_diff])

# # ==========================================
# # 3. Plotting
# # ==========================================
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle(f"Integral Stability Analysis: Tight (30.8) vs Diffuse (0.64)", fontsize=16)

# # Helper to plot
# def plot_subplot(ax, data_key, title, ylabel, is_eri=False):
#     x = dz_vals
#     if is_eri:
#         y_prony = eri_data[data_key]
#         y_exact = eri_exact[data_key]
#     else:
#         y_prony = ven_data[data_key]
#         y_exact = ven_exact[data_key]
        
#     ax.plot(x, y_prony, 'b-', label='Prony')
    
#     # Plot Exact Target
#     ax.axhline(y_exact, color='c', linestyle=':', linewidth=2, label='Exact (z=0)')
#     ax.scatter([x[0]], [y_exact], color='c', marker='*', s=100, zorder=10)
    
#     ax.axvline(0.01, color='k', linestyle='--', alpha=0.5, label='Cutoff (0.01)')
#     # ax.set_xscale('log')
#     ax.set_title(title)
#     ax.set_xlabel("$\Delta z$ (Bohr)")
#     if ylabel: ax.set_ylabel(ylabel)
#     ax.grid(True, which="both", alpha=0.3)
#     ax.legend()

# # --- Row 1: Ven ---
# plot_subplot(axes[0,0], 'tight', r"$V_{en}$ Diagonal (Tight)", "Energy (Ha)")
# plot_subplot(axes[0,1], 'diff',  r"$V_{en}$ Diagonal (Diffuse)", "")
# plot_subplot(axes[0,2], 'cross', r"$V_{en}$ Off-Diagonal (Tight-Diffuse)", "")

# # --- Row 2: ERI ---
# plot_subplot(axes[1,0], 'tight', r"ERI (Tight-Tight)", "Energy (Ha)", is_eri=True)
# plot_subplot(axes[1,1], 'diff',  r"ERI (Diff-Diff)", "", is_eri=True)
# plot_subplot(axes[1,2], 'cross', r"ERI (Tight-Tight | Diff-Diff)", "", is_eri=True)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig("multiorbital_integral_analysis.png", dpi=150)
# print("Plot saved to multiorbital_integral_analysis.png")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- Import your module ---
sys.path.insert(0, os.getcwd())

try:
    # We try to import the specific file where you put the Interpolator class
    import pyqed.qchem.dvr.hybrid_gauss_dvr_integrals_extrapolate as dvr_int
except ImportError:
    print("Error: Could not import hybrid_gauss_dvr_integrals_add_d_orbitals.py")
    sys.exit(1)

# ==========================================
# 1. Setup Basis
# ==========================================

# User's exponents
s_exps = [30.825394365, 0.6401216923, 0.1612777588]
nuclei = [(1.0, 0.0, 0.0, 0.0)]

alphas, centers, labels = dvr_int.make_xy_spd_primitive_basis(
    nuclei, 
    exps_s=np.array(s_exps),
    exps_p=[],
    exps_d=[]
)

# Identify Indices
idx_tight = 0
idx_diff  = 1 

print(f"Analyzing Integrals for:")
print(f"  Tight   (idx {idx_tight}): exp={alphas[idx_tight]:.4f}")
print(f"  Diffuse (idx {idx_diff}): exp={alphas[idx_diff]:.4f}")

# Define dz range (Logarithmic)
dz_vals = np.logspace(-7, 0, 1000) # 1e-7 to 0.1

# ==========================================
# 2. Initialize Interpolator (New Strategy)
# ==========================================
print("\nInitializing VenInterpolator...")
# This builds the spline table for dz < 0.05
ven_interp = dvr_int.VenInterpolator(alphas, centers, labels, nuclei, dz_cut=0.5, n_pts=500)

# ==========================================
# 3. Data Collection
# ==========================================

# Storage
ven_data = {
    'tight': [], 'diff': [], 'cross': []
}
ven_interp_data = {
    'tight': [], 'diff': [], 'cross': []
}
eri_data = {
    'tight': [], 'diff': [], 'cross': []
}

# Analytical Targets (at dz=0)
nuc_xy = np.array([0.0, 0.0])
mat_ven_ex = dvr_int._V_en_exact_dz0_general(alphas, centers, labels, nuc_xy)
ven_exact = {
    'tight': -1.0 * mat_ven_ex[idx_tight, idx_tight],
    'diff':  -1.0 * mat_ven_ex[idx_diff, idx_diff],
    'cross': -1.0 * mat_ven_ex[idx_tight, idx_diff]
}

# ERI Exact
eri_tensor_ex = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=0.0, dz_tol=10.0)
eri_exact = {
    'tight': eri_tensor_ex[idx_tight, idx_tight, idx_tight, idx_tight],
    'diff':  eri_tensor_ex[idx_diff, idx_diff, idx_diff, idx_diff],
    'cross': eri_tensor_ex[idx_tight, idx_tight, idx_diff, idx_diff]
}

print("\nScanning dz values...")
for z in dz_vals:
    # 1. Raw Prony (Unstable / Deep Well)
    mat_v_prony = dvr_int._V_en_prony_general(alphas, centers, labels, nuc_xy, z)
    ven_data['tight'].append(-1.0 * mat_v_prony[idx_tight, idx_tight])
    ven_data['diff'].append(-1.0 * mat_v_prony[idx_diff, idx_diff])
    ven_data['cross'].append(-1.0 * mat_v_prony[idx_tight, idx_diff])
    
    # 2. New Interpolated Method (Smooth / Physical)
    # We use the interpolator instance we created
    # Note: VenInterpolator returns the full matrix V, including the -Z factor
    mat_v_interp = ven_interp(z)
    ven_interp_data['tight'].append(mat_v_interp[idx_tight, idx_tight])
    ven_interp_data['diff'].append(mat_v_interp[idx_diff, idx_diff])
    ven_interp_data['cross'].append(mat_v_interp[idx_tight, idx_diff])
    
    # 3. ERI (Standard)
    # We use the standard function, assuming it has the update or is stable
    ten_e = dvr_int.eri_2d_cartesian_with_p(alphas, centers, labels, delta_z=z, dz_tol=0.01)
    eri_data['tight'].append(ten_e[idx_tight, idx_tight, idx_tight, idx_tight])
    eri_data['diff'].append(ten_e[idx_diff, idx_diff, idx_diff, idx_diff])
    eri_data['cross'].append(ten_e[idx_tight, idx_tight, idx_diff, idx_diff])

# ==========================================
# 4. Plotting
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Integral Analysis: Raw Prony vs. Quadrature Spline (Cutoff=0.05)", fontsize=16)

def plot_subplot(ax, data_key, title, ylabel, is_eri=False):
    x = dz_vals
    
    if is_eri:
        # For ERI we just compare to exact target
        y_vals = eri_data[data_key]
        y_ex = eri_exact[data_key]
        ax.plot(x, y_vals, 'b-', label='Calculated')
        ax.axhline(y_ex, color='c', linestyle=':', linewidth=2, label='Exact (z=0)')
    else:
        # For Ven we compare Raw vs Interpolated
        y_raw = ven_data[data_key]
        y_interp = ven_interp_data[data_key]
        y_ex = ven_exact[data_key]
        
        ax.plot(x, y_raw, 'b--', alpha=0.5, label='Raw Prony (Unstable)')
        ax.plot(x, y_interp, 'r-', linewidth=2, label='Spline Interpolation')
        ax.axhline(y_ex, color='c', linestyle=':', linewidth=2, label='Exact Limit (z=0)')
    
    ax.axvline(0.05, color='k', linestyle='--', alpha=0.5, label='Cutoff (0.05)')
    ax.set_xscale('log')
    ax.set_title(title)
    ax.set_xlabel("$\Delta z$ (Bohr)")
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

# --- Row 1: Ven ---
plot_subplot(axes[0,0], 'tight', r"$V_{en}$ Diagonal (Tight)", "Energy (Ha)")
plot_subplot(axes[0,1], 'diff',  r"$V_{en}$ Diagonal (Diffuse)", "")
plot_subplot(axes[0,2], 'cross', r"$V_{en}$ Off-Diagonal (Tight-Diffuse)", "")

# --- Row 2: ERI ---
plot_subplot(axes[1,0], 'tight', r"ERI (Tight-Tight)", "Energy (Ha)", is_eri=True)
plot_subplot(axes[1,1], 'diff',  r"ERI (Diff-Diff)", "", is_eri=True)
plot_subplot(axes[1,2], 'cross', r"ERI (Tight-Tight | Diff-Diff)", "", is_eri=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("multiorbital_integral_spline_check.png", dpi=150)
print("Plot saved to multiorbital_integral_spline_check.png")
plt.show()