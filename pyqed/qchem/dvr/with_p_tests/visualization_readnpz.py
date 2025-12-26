import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

# =================================================================================
#  Plotter for Density from Hybrid Gauss-DVR SCF Runs
#  Reconstructs P(x,y,z) from:
#    1. Sine-DVR basis along Z: u_k(z)
#    2. Contracted Gaussian basis along XY: phi_mu(x,y; z_k)
# =================================================================================

def sine_dvr_basis_val(z_eval, z_min, z_max, N, k_index_0based):
    """
    Evaluate the k-th sine-DVR basis function at z_eval.
    """
    L = z_max - z_min
    j = k_index_0based + 1 # Grid index (1..N)
    
    val = 0.0
    pre_u = np.sqrt(2.0 / (N + 1))
    pre_f = np.sqrt(2.0 / L)
    
    n = np.arange(1, N + 1)
    
    U_kn = pre_u * np.sin(np.pi * j * n / (N + 1))
    
    if isinstance(z_eval, np.ndarray):
        arg = np.outer(n, np.pi * (z_eval - z_min) / L)
        f_n = pre_f * np.sin(arg)
        val = U_kn @ f_n
    else:
        arg = n * np.pi * (z_eval - z_min) / L
        f_n = pre_f * np.sin(arg)
        val = np.dot(U_kn, f_n)
        
    return val

def eval_primitive_xy(alphas, centers, labels, x, y):
    """
    Evaluate all primitive gaussians at (x,y).
    Updated to support d-orbitals.
    """
    vals = []
    
    for i in range(len(alphas)):
        a = alphas[i]
        xc, yc = centers[i]
        l_data = labels[i]
        
        # Handle both dict (from npz) and object (direct)
        if isinstance(l_data, dict):
            kind = l_data['kind']
        else:
            kind = l_data.kind

        dx = x - xc
        dy = y - yc
        r2 = dx**2 + dy**2
        
        gauss = np.exp(-a * r2)
        
        if kind == '2d-s':
            vals.append(gauss)
        elif kind == '2d-px':
            vals.append(dx * gauss)
        elif kind == '2d-py':
            vals.append(dy * gauss)
        # --- D Orbitals ---
        elif kind == '2d-dx2':
            vals.append((dx**2) * gauss)
        elif kind == '2d-dy2':
            vals.append((dy**2) * gauss)
        elif kind == '2d-dxy':
            vals.append((dx * dy) * gauss)
        else:
            vals.append(np.zeros_like(gauss))
            
    return np.array(vals)

def reconstruct_density_1d_z(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, z_evals, fixed_x=0.0, fixed_y=0.0):
    chi_vals = eval_primitive_xy(alphas, centers, labels, fixed_x, fixed_y)
    
    phi_vals = []
    for k in range(Nz):
        phi_vals.append(C_list[k].T @ chi_vals) 
    
    phi_vals = np.array(phi_vals) # (Nz, M)

    zmin = -Lz
    zmax = Lz
    
    theta_vals = np.zeros((Nz, len(z_evals)))
    for k in range(Nz):
        theta_vals[k, :] = sine_dvr_basis_val(z_evals, zmin, zmax, Nz, k)
        
    rho_z = np.zeros_like(z_evals)
    
    W = theta_vals[:, None, :] * phi_vals[:, :, None] 
    
    N_total = Nz * M
    W_flat = W.reshape(N_total, len(z_evals))
    
    temp = P @ W_flat 
    rho_z = np.sum(W_flat * temp, axis=0) 
    
    return rho_z

def reconstruct_density_1d_x(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals, fixed_y=0.0, fixed_z=0.0):
    zmin = -Lz
    zmax = Lz
    
    theta_vals = np.zeros(Nz)
    for k in range(Nz):
        theta_vals[k] = sine_dvr_basis_val(fixed_z, zmin, zmax, Nz, k)
        
    chi_vals = eval_primitive_xy(alphas, centers, labels, x_evals, np.full_like(x_evals, fixed_y)) 
    
    phi_vals = np.zeros((Nz, M, len(x_evals)))
    for k in range(Nz):
        phi_vals[k] = C_list[k].T @ chi_vals
        
    W = theta_vals[:, None, None] * phi_vals
    W_flat = W.reshape(Nz * M, len(x_evals))
    
    temp = P @ W_flat
    rho_x = np.sum(W_flat * temp, axis=0)
    
    return rho_x

def reconstruct_density_1d_y(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, y_evals, fixed_x=0.0, fixed_z=0.0):
    """
    Reconstruct density along Y axis at fixed x,z.
    """
    zmin = -Lz
    zmax = Lz
    
    # 1. Theta at fixed z
    theta_vals = np.zeros(Nz)
    for k in range(Nz):
        theta_vals[k] = sine_dvr_basis_val(fixed_z, zmin, zmax, Nz, k)
        
    # 2. Primitives at (fixed_x, y_evals)
    chi_vals = eval_primitive_xy(alphas, centers, labels, np.full_like(y_evals, fixed_x), y_evals) 
    
    # 3. Contract
    phi_vals = np.zeros((Nz, M, len(y_evals)))
    for k in range(Nz):
        phi_vals[k] = C_list[k].T @ chi_vals
        
    W = theta_vals[:, None, None] * phi_vals
    W_flat = W.reshape(Nz * M, len(y_evals))
    
    temp = P @ W_flat
    rho_y = np.sum(W_flat * temp, axis=0)
    
    return rho_y

def reconstruct_density_2d_xy(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals, y_evals, fixed_z=0.0):
    zmin = -Lz
    zmax = Lz
    
    theta_vals = np.zeros(Nz)
    for k in range(Nz):
        theta_vals[k] = sine_dvr_basis_val(fixed_z, zmin, zmax, Nz, k)
        
    X, Y = np.meshgrid(x_evals, y_evals) 
    shape_2d = X.shape
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    
    chi_vals = eval_primitive_xy(alphas, centers, labels, X_flat, Y_flat)
    
    W_flat_list = []
    for k in range(Nz):
        phi_k = C_list[k].T @ chi_vals
        w_k = theta_vals[k] * phi_k
        W_flat_list.append(w_k)
        
    W_flat = np.vstack(W_flat_list) 
    
    temp = P @ W_flat
    rho_flat = np.sum(W_flat * temp, axis=0)
    
    return rho_flat.reshape(shape_2d)

def reconstruct_density_2d_xz(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals, z_evals, fixed_y=0.0):
    X, Z = np.meshgrid(x_evals, z_evals) 
    shape_2d = X.shape
    X_flat = X.ravel()
    Z_flat = Z.ravel()
    
    zmin = -Lz
    zmax = Lz

    theta_unique = np.zeros((Nz, len(z_evals)))
    for k in range(Nz):
        theta_unique[k, :] = sine_dvr_basis_val(z_evals, zmin, zmax, Nz, k)
    
    chi_unique = eval_primitive_xy(alphas, centers, labels, x_evals, np.full_like(x_evals, fixed_y))
    
    phi_x = np.zeros((Nz, M, len(x_evals)))
    for k in range(Nz):
        phi_x[k] = C_list[k].T @ chi_unique
        
    rho_img = np.zeros(shape_2d)
    
    for i_z, z_val in enumerate(z_evals):
        th = theta_unique[:, i_z]
        W_slice = (th[:, None, None] * phi_x).reshape(Nz*M, len(x_evals))
        temp = P @ W_slice
        rho_row = np.sum(W_slice * temp, axis=0)
        rho_img[i_z, :] = rho_row
        
    return rho_img

def reconstruct_density_2d_yz(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, y_evals, z_evals, fixed_x=0.0):
    """
    Reconstruct density in YZ plane at fixed x.
    """
    Y, Z = np.meshgrid(y_evals, z_evals) 
    shape_2d = Y.shape
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    
    zmin = -Lz
    zmax = Lz

    theta_unique = np.zeros((Nz, len(z_evals)))
    for k in range(Nz):
        theta_unique[k, :] = sine_dvr_basis_val(z_evals, zmin, zmax, Nz, k)
    
    # Primitives at fixed x, varying y
    chi_unique = eval_primitive_xy(alphas, centers, labels, np.full_like(y_evals, fixed_x), y_evals)
    
    phi_y = np.zeros((Nz, M, len(y_evals)))
    for k in range(Nz):
        phi_y[k] = C_list[k].T @ chi_unique
        
    rho_img = np.zeros(shape_2d)
    
    for i_z, z_val in enumerate(z_evals):
        th = theta_unique[:, i_z]
        W_slice = (th[:, None, None] * phi_y).reshape(Nz*M, len(y_evals))
        temp = P @ W_slice
        rho_row = np.sum(W_slice * temp, axis=0)
        rho_img[i_z, :] = rho_row
        
    return rho_img

def parse_cut_string(s):
    if '=' not in s:
        return None, None
    parts = s.split('=')
    return parts[0].strip().lower(), float(parts[1])

def run_plotter():
    print("=== Density Plotter for Hybrid Gauss-DVR ===")
    folder_path = input("Enter the folder path containing .npz files: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at: {os.path.abspath(folder_path)}")
        return

    files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
    if not files:
        print(f"No .npz files found in: {os.path.abspath(folder_path)}")
        return
    
    print(f"Found {len(files)} files in {os.path.abspath(folder_path)}")
    
    plot_type = input("Enter plot type ('1D' or '2D'): ").strip().upper()
    axis_info = input("Enter cut info (e.g. 'z' for 1D z-plot, 'z=0' for 2D xy-plot): ").strip().lower()
    
    mode = None
    fixed_val = 0.0
    
    if plot_type == '1D':
        if '=' in axis_info:
             print("Warning: for 1D, usually just specify the axis variable (x, y, or z).")
             var, val = parse_cut_string(axis_info)
             axis_info = var
        
        if axis_info == 'z':
            mode = '1D_Z'
        elif axis_info == 'x':
            mode = '1D_X'
        elif axis_info == 'y':
            mode = '1D_Y'
        else:
            print("Unknown 1D axis. Defaulting to Z.")
            mode = '1D_Z'
            
    elif plot_type == '2D':
        var, val = parse_cut_string(axis_info)
        if var is None:
            print("Error: For 2D, please specify slice, e.g., 'z=0'.")
            return
        fixed_val = val
        if var == 'z':
            mode = '2D_XY'
        elif var == 'y':
            mode = '2D_XZ'
        elif var == 'x':
            mode = '2D_YZ'
        else:
            print("Unknown slice axis.")
            return

    print(f"Processing mode: {mode}, Fixed Val (if any): {fixed_val}")

    for fname in files:
        print(f"Processing {os.path.basename(fname)}...")
        data = np.load(fname, allow_pickle=True)
        
        P = data['P']
        C_list = data['C_list']
        alphas = data['alphas']
        centers = data['centers']
        labels = data['labels_serialized'] 
        z_grid = data['z_grid']
        Lz = float(data['Lz'])
        Nz = int(data['Nz'])
        M = int(data['M'])
        cycle = data['cycle']
        
        if C_list.dtype == object:
            C_list = np.stack(C_list)
            
        # Construct safe output filename
        base_name = os.path.splitext(fname)[0]
        outname = None # Initialize to prevent UnboundLocalError

        if mode == '1D_Z':
            z_evals = np.linspace(-Lz, Lz, 300)
            rho = reconstruct_density_1d_z(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, z_evals)
            
            plt.figure()
            plt.plot(z_evals, rho)
            # plt.scatter(z_evals, rho)
            plt.title(f"Density along Z (Cycle {cycle})")
            plt.xlabel("z (bohr)")
            plt.ylabel("Density")
            outname = base_name + "_dens1D_z.png"
            plt.savefig(outname)
            plt.close()
            
        elif mode == '1D_X':
            x_evals = np.linspace(-6.0, 6.0, 300) 
            rho = reconstruct_density_1d_x(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals)
            
            plt.figure()
            plt.plot(x_evals, rho)
            plt.title(f"Density along X (Cycle {cycle})")
            plt.xlabel("x (bohr)")
            plt.ylabel("Density")
            outname = base_name + "_dens1D_x.png"
            plt.savefig(outname)
            plt.close()

        elif mode == '1D_Y':
            y_evals = np.linspace(-6.0, 6.0, 300) 
            rho = reconstruct_density_1d_y(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, y_evals)
            
            plt.figure()
            plt.plot(y_evals, rho)
            plt.title(f"Density along Y (Cycle {cycle})")
            plt.xlabel("y (bohr)")
            plt.ylabel("Density")
            outname = base_name + "_dens1D_y.png"
            plt.savefig(outname)
            plt.close()
            
        elif mode == '2D_XY':
            Npts = 100
            x_evals = np.linspace(-5.0, 5.0, Npts)
            y_evals = np.linspace(-5.0, 5.0, Npts)
            rho = reconstruct_density_2d_xy(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals, y_evals, fixed_z=fixed_val)
            
            plt.figure()
            plt.imshow(rho, extent=[-5,5,-5,5], origin='lower', cmap='inferno')
            plt.colorbar(label='Density')
            plt.title(f"Density XY at z={fixed_val} (Cycle {cycle})")
            plt.xlabel("x")
            plt.ylabel("y")
            outname = base_name + f"_dens2D_xy_z{fixed_val:.1f}.png"
            plt.savefig(outname)
            plt.close()
            
        elif mode == '2D_XZ':
            Nx = 100
            Nz_pts = 150
            x_evals = np.linspace(-5.0, 5.0, Nx)
            z_evals = np.linspace(-Lz, Lz, Nz_pts)
            rho = reconstruct_density_2d_xz(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, x_evals, z_evals, fixed_y=fixed_val)
            
            plt.figure()
            plt.imshow(rho, extent=[-5,5,-Lz,Lz], origin='lower', cmap='inferno', aspect='auto')
            plt.colorbar(label='Density')
            plt.title(f"Density XZ at y={fixed_val} (Cycle {cycle})")
            plt.xlabel("x")
            plt.ylabel("z")
            outname = base_name + f"_dens2D_xz_y{fixed_val:.1f}.png"
            plt.savefig(outname)
            plt.close()

        elif mode == '2D_YZ':
            Ny = 100
            Nz_pts = 150
            y_evals = np.linspace(-5.0, 5.0, Ny)
            z_evals = np.linspace(-Lz, Lz, Nz_pts)
            rho = reconstruct_density_2d_yz(P, C_list, z_grid, Lz, Nz, M, alphas, centers, labels, y_evals, z_evals, fixed_x=fixed_val)
            
            plt.figure()
            plt.imshow(rho, extent=[-5,5,-Lz,Lz], origin='lower', cmap='inferno', aspect='auto')
            plt.colorbar(label='Density')
            plt.title(f"Density YZ at x={fixed_val} (Cycle {cycle})")
            plt.xlabel("y")
            plt.ylabel("z")
            outname = base_name + f"_dens2D_yz_x{fixed_val:.1f}.png"
            plt.savefig(outname, dpi=2000)
            plt.close()
            
        # Print absolute path to assist user
        if outname:
            print(f"Saved plot to: {os.path.abspath(outname)}")
        else:
            print(f"Warning: No plot generated for mode {mode}")

if __name__ == "__main__":
    run_plotter()