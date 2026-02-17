import os
import pandas as pd
import numpy as np

"""
Extract each kernel from the original CSV files and normalize to [0, 1].
You should update the paths below to match your local data layout.
"""

# Input folder containing raw CSV files from the QPI dataset
input_folder = './data/raw/64X64 vp coords/'

# Output folder where per-kernel .npy files (64x64) will be saved
output_folder_obs = './data/kernel/all/'
if not os.path.exists(output_folder_obs):
    os.makedirs(output_folder_obs)

max_min_values = []


def normalize_data(data):
    """Min-max normalize a 1D array to [0, 1]."""
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)


# Iterate over all CSV files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):  # Only process CSV files
        file_path = os.path.join(input_folder, filename)

        # Read CSV file
        try:
            df = pd.read_csv(file_path, header=None)
            # Use the third row (index 2) and first 64*64 entries as the kernel
            selected_row = df.iloc[2, :64 * 64].values.flatten()
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

        # Normalize kernel data
        normalized_data = normalize_data(selected_row)

        # Build new filename using the first two tokens of the original filename
        new_filename0 = os.path.splitext(filename)[0]
        new_filename1 = new_filename0.split()
        new_filename = " ".join(new_filename1[:2])
        new_filename = f"{new_filename}nom_2d"
        new_file_path = os.path.join(output_folder_obs, new_filename)

        # Reshape to 2D 64x64 kernel
        reshaped_data = np.reshape(normalized_data, (64, 64))

        # Save as .npy
        np.save(new_file_path, reshaped_data)

print('All files processed.')
