import os
import numpy as np
import matplotlib.pyplot as plt
import re

def calculate_ipr(filepath):
    """
    Calculates the Inverse Participation Ratio (IPR) from a data file.
    The file should contain two columns: Site Index and Probability.
    The IPR is calculated as the sum of the squares of the probabilities.
    """
    try:
        # Load the probability data from the second column of the text file
        # The first row is skipped as it's a header.
        probabilities = np.loadtxt(filepath, usecols=1, skiprows=1)

        # The IPR formula is sum(|psi|^4). Since the input data is |psi|^2,
        # we just need to square each probability value and sum them up.
        ipr = np.sum(probabilities**2)
        return ipr
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None

def main():
    """
    Main function to find data files, calculate IPR for each,
    and plot the results as a linear heatmap.
    """
    # --- Configuration ---
    # Assumes your data files are in a subfolder named 'data'
    folder_path = '/Users/shuoyihu/Documents/GitHub/pyqed/MacBook_local_data/open_boundary_condition_calculation/coords_0.0_0.75_t1_0.1286_t2_0.0919/floquet_data_Gomez_Leon_test_b=0.75/E_max_0.5_omega_0.025_OBC_cells_20_nt_101/2_Edge_State_Plots'

    # --- Data Processing ---
    results = []
    
    # Regex to find files starting with 'edge_state_1_E_' and capture the energy value
    file_pattern = re.compile(r'edge_state_1_E_([\d\.]+)_distribution\.txt')

    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' was not found.")
        print("Please make sure your data files are in a subfolder named 'data'.")
        return

    print(f"Searching for files in '{folder_path}'...")
    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)
        if match:
            # Extract the energy value (E) from the filename
            energy_str = match.group(1)
            energy = float(energy_str)
            
            filepath = os.path.join(folder_path, filename)
            
            # Calculate the IPR for the file
            ipr = calculate_ipr(filepath)
            
            if ipr is not None:
                results.append({'energy': energy, 'ipr': ipr})

    if not results:
        print("No matching data files were found. Please check the filenames and folder path.")
        return
        
    print(f"Found and processed {len(results)} data files.")

    # Sort the results by energy for a clean plot
    results.sort(key=lambda x: x['energy'])
    
    energies = [r['energy'] for r in results]
    ipr_values = [r['ipr'] for r in results]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Reshape the IPR values into a 2D array for the heatmap
    # The array will have 1 row and as many columns as there are data points.
    ipr_reshaped = np.array(ipr_values).reshape(1, -1)
    for i in range(len(ipr_reshaped[0])):
        print(f"{ipr_reshaped[0][i]}")
    
    # Use imshow to create the heatmap. The aspect ratio is set to 'auto'
    # to make the heatmap stretch across the plot area.
    heatmap = ax.imshow(ipr_reshaped, cmap='viridis', aspect='auto', 
                        extent=[min(energies), max(energies), 0, 1])

    # --- Aesthetics and Labels ---
    ax.set_yticks([]) # Hide the y-axis ticks as they are not meaningful here
    ax.set_title('IPR of Edge State vs. Energy (E)', fontsize=16)
    ax.set_xlabel('Energy (E)', fontsize=12)
    
    # Add a color bar to show the IPR values corresponding to the colors
    cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Inverse Participation Ratio (IPR)', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
