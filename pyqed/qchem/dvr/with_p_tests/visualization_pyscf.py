import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft

def get_rho_z_profile(mol, dm, nz=200, z_range=(-10, 10), box_width=10.0, n_transverse=50):
    """
    Calculate the integrated electron density rho(z) = integral rho(x,y,z) dx dy.
    """
    # 1. Define Z-grid
    z_vals = np.linspace(z_range[0], z_range[1], nz)
    rho_z = np.zeros(nz)
    
    # 2. Define Transverse Grid (x,y) for integration
    # We use a simple grid integration for the slice
    x = np.linspace(-box_width/2, box_width/2, n_transverse)
    y = np.linspace(-box_width/2, box_width/2, n_transverse)
    X, Y = np.meshgrid(x, y)
    weights = (x[1] - x[0]) * (y[1] - y[0]) # dA
    
    # Flatten transverse coords
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    n_t = len(X_flat)
    
    print(f"Integrating slices... ({nz} slices, {n_t} points per slice)")
    
    for i, z in enumerate(z_vals):
        # Build coords (N, 3) for this slice
        coords = np.zeros((n_t, 3))
        coords[:, 0] = X_flat
        coords[:, 1] = Y_flat
        coords[:, 2] = z
        
        # Evaluate rho on this slice
        # values is array of (N_points,)
        ao_value = dft.numint.eval_ao(mol, coords)
        rho_slice = dft.numint.eval_rho(mol, ao_value, dm)
        
        # Integrate
        rho_z[i] = np.sum(rho_slice) * weights
        
    return z_vals, rho_z

# def run_comparison():
#     # Define H4 Chain Geometry (Same as your DVR test)
#     # Coordinates in Angstrom (PySCF default) or Bohr. 
#     # Your code likely uses Bohr (atomic units). PySCF uses Angstrom by default 
#     # unless unit='Bohr' is specified.
#     atom_str = """
#     H 0.0 0.0  3.6
#     H 0.0 0.0  0.91
#     H 0.0 0.0 -3.6
#     H 0.0 0.0 -0.91
#     """
    
#     bases = ['sto-3g', '6-31g', 'cc-pvdz']
#     results = {}
    
#     plt.figure(figsize=(10, 6))
    
#     for basis in bases:
#         print(f"\nRunning PySCF RHF with {basis}...")
#         mol = gto.M(atom=atom_str, basis=basis, unit='Bohr', charge=0, spin=0, verbose=0)
#         mf = scf.RHF(mol)
#         e_tot = mf.kernel()
#         print(f"  E_tot = {e_tot:.6f} Eh")
        
#         # Calculate Z-Profile
#         z_vals, rho_z = get_rho_z_profile(mol, mf.make_rdm1(), nz=300, z_range=(-8, 8))
        
#         results[basis] = (z_vals, rho_z, e_tot)
#         plt.plot(z_vals, rho_z, label=f"{basis} (E={e_tot:.4f})")

#     # Add Nuclear Markers
#     # Parse manually or use mol object
#     z_nucs = [3.6, 0.91, -3.6, -0.91]
#     for z in z_nucs:
#         plt.axvline(z, color='k', linestyle=':', alpha=0.3)
#         plt.text(z, 0.05, 'H', ha='center', va='bottom', color='k')

#     plt.title("3D Electron Density Integrated over Transverse Plane: $\\rho(z) = \\int \\rho(x,y,z) dx dy$")
#     plt.xlabel("z (Bohr)")
#     plt.ylabel("Integrated Density (electrons/bohr)")
#     plt.legend()
#     plt.grid(True, alpha=0.5)
#     plt.tight_layout()
    
#     filename = "pyscf_z_profile_comparison.png"
#     plt.savefig(filename, dpi=150)
#     print(f"\nPlot saved to {filename}")
#     plt.show()

# if __name__ == "__main__":
#     run_comparison()

def run_comparison():
    # Define HeH+ Geometry
    # Coordinates in Angstrom (PySCF default) or Bohr.
    # zs = np.linspace(-19,19, 20)
    # # # # zs = np.linspace(-249,249, 100)
    # # # zs = np.linspace(-49,49, 50)
    # # # # # # print(zs)
    # atom_str = '; '.join(f'H 0 0 {z:.6g}' for z in zs) + '; '
    atom_str = """
    H 0.0 0.0 -0.7316
    H  0.0 0.0 0.7316
    """
    
    bases = ['sto-6g', '6-31g', 'cc-pvdz']
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for basis in bases:
        print(f"\nRunning PySCF RHF with {basis}...")
        mol = gto.M(atom=atom_str, basis=basis, unit='A', spin=0, verbose=0)
        mf = scf.RHF(mol)
        e_tot = mf.kernel()
        print(f"  E_tot = {e_tot:.6f} Eh")
        
        # Calculate Z-Profile
        z_vals, rho_z = get_rho_z_profile(mol, mf.make_rdm1(), nz=800, z_range=(-6,6))
        
        results[basis] = (z_vals, rho_z, e_tot)
        plt.plot(z_vals, rho_z, label=f"{basis} (E={e_tot:.4f})")

    # # Add Nuclear Markers
    # z_nucs = np.linspace(-19,19, 10)
    # for z in z_nucs:
    #     plt.axvline(z, color='k', linestyle=':', alpha=0.3)
    #     plt.text(z, 0.05, 'H', ha='center', va='bottom', color='k')

    plt.title("3D Electron Density Integrated over Transverse Plane: $\\rho(z) = \\int \\rho(x,y,z) dx dy$")
    plt.xlabel("z (Bohr)")
    plt.ylabel("Integrated Density (electrons/bohr)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    filename = "pyscf_z_profile_H20.png"
    plt.savefig(filename, dpi=1000)
    print(f"\nPlot saved to {filename}")
    plt.show()

if __name__ == "__main__":
    run_comparison()
