import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# Define the lists for the axes
list1 = np.linspace(0, 1, 1001)
list2 = np.linspace(0, 20, 1001)
c=0.09
# Create an empty 2D array to store the heatmap data
heatmap_data = np.zeros((len(list1), len(list2)))

# Iterate over the lists, calculate the parameters, and fill the heatmap
for i, x in enumerate(list1):
    for j, y in enumerate(list2):
        # Calculate the two bessel function values
        param1_val = 2.25*(np.abs(jv(0, x * y + 1j*y*c)))**2
        param2_val = (np.abs(jv(0, (1 - x) * y- 1j*y*c)))**2

        # Compare the values and populate the heatmap
        if param1_val < param2_val:
            heatmap_data[j, i] = 1
        else:
            heatmap_data[j, i] = 0

# Create the heatmap plot
plt.figure(figsize=(10, 8))
# Using extent to set the axes scales correctly and origin='lower'
plt.imshow(heatmap_data, extent=[list1.min(), list1.max(), list2.min(), list2.max()], origin='lower', aspect='auto', cmap='viridis')

# Add a color bar
cbar = plt.colorbar(ticks=[0, 1])
cbar.set_ticklabels(['$J_0((1-x)y) \geq J_0(xy)$', '$J_0(xy) > J_0((1-x)y)$'])


# Add labels and a title
plt.xlabel("x")
plt.ylabel("y")
plt.title("Heatmap Comparison of $J_0(xy)$ and $J_0((1-x)y)$")

# Save the plot to a file
plt.savefig(f"bessel_comparison_heatmap_with_c_{c}.png")

print("Heatmap saved as bessel_comparison_heatmap.png")
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import jv

# # Define the lists for the axes
# list1 = np.linspace(0, 1, 1001)
# list2 = np.linspace(0, 20, 1001)

# # Create an empty 2D array to store the heatmap data
# heatmap_data = np.zeros((len(list1), len(list2)))

# # Iterate over the lists, calculate the parameters, and fill the heatmap
# for i, x in enumerate(list1):
#     for j, y in enumerate(list2):
#         # Calculate the two bessel function values
#         param1_val = 2.25*(np.abs(jv(0, x * y)))**2
#         param2_val = (np.abs(jv(0, (1 - x) * y)))**2

#         # Compare the values and populate the heatmap
#         if param1_val < param2_val:
#             heatmap_data[j, i] = 1
#         else:
#             heatmap_data[j, i] = 0

# # Create the heatmap plot
# plt.figure(figsize=(10, 8))
# # Using extent to set the axes scales correctly and origin='lower'
# plt.imshow(heatmap_data, extent=[list1.min(), list1.max(), list2.min(), list2.max()], origin='lower', aspect='auto', cmap='viridis')

# # Add a color bar
# cbar = plt.colorbar(ticks=[0, 1])
# cbar.set_ticklabels(['$J_0((1-x)y) \geq J_0(xy)$', '$J_0(xy) > J_0((1-x)y)$'])


# # Add labels and a title
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Heatmap Comparison of $J_0(xy)$ and $J_0((1-x)y)$")

# # Save the plot to a file
# plt.savefig("bessel_comparison_heatmap.png")

# print("Heatmap saved as bessel_comparison_heatmap.png")