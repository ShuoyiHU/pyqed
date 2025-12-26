import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import cv2 # <-- Added for video creation
import glob # <-- Added to easily find files
import re   # <-- Added to parse filenames reliably

# Import the classes from your saved file
from pyqed.floquet.floquet import TightBinding, FloquetBloch

def plot_quasienergy_map(k, omega, E_min, E_max, E_steps):
    """
    Generates and plots the quasienergy map for a 1D tight-binding model.

    Args:
        k (float): The crystal momentum in the first Brillouin zone.
        omega (float): The driving frequency.
        E_min (float): The minimum driving strength.
        E_max (float): The maximum driving strength.
        E_steps (int): The number of steps for the driving strength.
    """
    # 1. Set up the Tight-Binding Model
    coords = [[0], [0.7]]
    lattice_constant = [1.0]
    relative_hopping = [1.5, 1.0] 
    tb_model = TightBinding(coords, 
                            lattice_constant=lattice_constant, 
                            relative_Hopping=relative_hopping)

    # 2. Define the range of the external driving strength E
    E_values = np.linspace(E_min, E_max, E_steps)

    # 3. Set up the paths for data and figures
    base_path = f"/Volumes/Shuoyi's SSD/quasienergy_E_map_for_fix_k"
    figure_dir = os.path.join(base_path, f"k={k:.4f}_omega={omega:.4f}")
    data_dir = os.path.join(figure_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    floquet_model = tb_model.Floquet(
        omegad=omega,
        E0=E_values,
        nt=11,
        polarization=[1],
        data_path=data_dir
    )

    # 4. Run the calculation for the given k-point
    k_point = [k]
    floquet_model.run(k=k_point)

    # 5. Load the results
    all_quasienergies = []
    for E in E_values:
        filename = os.path.join(data_dir, f"band_E{E:.6f}.h5")
        with h5py.File(filename, 'r') as f:
            quasienergies = f['band_energy'][:]
            all_quasienergies.append(quasienergies[0])

    quasienergy_array = np.array(all_quasienergies)

    # 6. Plot and save the quasienergy map
    fig = plt.figure(figsize=(10, 6))
    for i in range(quasienergy_array.shape[1]):
        plt.plot(E_values, quasienergy_array[:, i], linestyle='-', marker='.', markersize=3)
    
    plt.xlabel('Driving Strength (E)')
    plt.ylabel('Quasienergy')
    plt.title(f'Quasienergy Map for k={k:.4f}, $\omega$={omega:.4f}')
    plt.grid(True)
    
    # Save the plot
    output_png_path = os.path.join(figure_dir, f'quasienergy_map_k={k:.4f}_omega={omega:.4f}.png')
    plt.savefig(output_png_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory

# -----------------------------------------------------------------------------
# NEW VIDEO GENERATION FUNCTION
# -----------------------------------------------------------------------------
def create_video_from_plots(base_storage_path, omega, output_filename="quasienergy_evolution.mp4", fps=30):
    """
    Scans a directory for plot images, sorts them by k-value, and creates a video.

    Args:
        base_storage_path (str): The root directory where the k-dependent folders are.
        omega (float): The omega value used to find the correct folders.
        output_filename (str): The name of the output video file.
        fps (int): Frames per second for the video.
    """
    print(f"🎬 Assembling video for omega = {omega:.4f}...")
    
    # 1. Find all relevant plot images
    search_pattern = os.path.join(base_storage_path, f'k=*_omega={omega:.4f}', '*.png')
    image_files = glob.glob(search_pattern)

    if not image_files:
        print(f"Error: No image files found at '{search_pattern}'. Cannot create video.")
        return

    # 2. Extract k-values and sort the files
    # We use a regular expression to reliably extract the floating point k-value
    def extract_k_value(filepath):
        match = re.search(r'k=([0-9]+\.[0-9]+)', os.path.basename(filepath))
        return float(match.group(1)) if match else -1

    sorted_image_files = sorted(image_files, key=extract_k_value)

    # 3. Get image dimensions from the first image
    frame = cv2.imread(sorted_image_files[0])
    height, width, layers = frame.shape
    size = (width, height)

    # 4. Initialize the video writer
    # The 'mp4v' codec is a good choice for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    output_path = os.path.join(base_storage_path, output_filename)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    # 5. Write each image frame to the video
    print(f"Found {len(sorted_image_files)} frames. Writing to video...")
    for filename in sorted_image_files:
        frame = cv2.imread(filename)
        video_writer.write(frame)

    # 6. Finalize the video
    video_writer.release()
    print(f"✅ Video saved successfully to '{output_path}'")


if __name__ == '__main__':
    # --- User-defined parameters for the simulation ---
    k_grid = np.linspace(0, np.pi, 101) # A finer grid for a smoother video
    omega_drive = 10.0      
    E_min_drive = 0.0      
    E_max_drive = 200.0     
    E_num_steps = 500    
    
    # --- Run the simulation loop to generate all plots ---
    for i, k_point in enumerate(k_grid):
        print(f"--- Processing k-point {i+1}/{len(k_grid)}: k = {k_point:.4f} ---")
        # plot_quasienergy_map(k_point, omega_drive, E_min_drive, E_max_drive, E_num_steps)
    
    print("\n--- All plots have been generated. ---\n")

    # --- Call the function to create the video ---
    base_path = "/Volumes/Shuoyi's SSD/quasienergy_E_map_for_fix_k"
    create_video_from_plots(
        base_storage_path=base_path,
        omega=omega_drive,
        output_filename=f"quasienergy_map_omega={omega_drive:.2f}.mp4",
        fps=20 # Adjust frames per second as desired
    )