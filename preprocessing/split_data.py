import os
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Extract, for each kernel, all corresponding observations and activation vectors
from the raw CSV files and save them to per-kernel folders.
Update the paths below to match your local data layout.
"""

# Input folder containing raw CSV files from the QPI dataset
input_folder = './data/raw/64X64 vp coords/'

# Output base folders
output_folder_obs_csv = './data/observation/csv/'
output_folder_obs_npy = './data/observation/npy/'
output_folder_act = './data/activation/csv/'


def count_lines(filename, encoding='utf-8'):
    """Return the number of lines in a text file."""
    with open(filename, 'r', encoding=encoding) as f:
        return sum(1 for _ in f)


file_counter = 0

# Iterate over all files in the input folder
for filename in tqdm(os.listdir(input_folder)):
    file_counter += 1
    max_min_values = []

    if filename.endswith('.csv'):  # Only process CSV files
        new_filename0 = os.path.splitext(filename)[0]
        # Use the first two tokens of the original filename as kernel identifier
        new_filename1 = new_filename0.split()
        new_filename = " ".join(new_filename1[:2])

        output_dir_obs_csv = os.path.join(output_folder_obs_csv, new_filename)
        output_dir_obs_npy = os.path.join(output_folder_obs_npy, new_filename)
        output_dir_act = os.path.join(output_folder_act, new_filename)

        if not os.path.exists(output_dir_obs_csv):
            os.makedirs(output_dir_obs_csv, exist_ok=True)
            os.makedirs(output_dir_obs_npy, exist_ok=True)
            os.makedirs(output_dir_act, exist_ok=True)

        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path, header=None)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

        num_rows = count_lines(file_path)

        # For each row, extract observation (first 4096 values reshaped to 64x64)
        # and activation (last 40 values)
        for i in range(1, num_rows - 2):
            front = df.iloc[i + 2, :4096].values.flatten()
            last = df.iloc[i + 2, -40:].values.flatten()

            front_filename_csv = f"{new_filename} {i}observation.csv"
            front_filename_npy = f"{new_filename} {i}observation"
            last_filename_csv = f"{new_filename} {i}activation.csv"

            new_file_path_f_csv = os.path.join(output_dir_obs_csv, front_filename_csv)
            new_file_path_f_npy = os.path.join(output_dir_obs_npy, front_filename_npy)
            new_file_path_l_csv = os.path.join(output_dir_act, last_filename_csv)

            # Reshape observation to 64x64
            reshaped_data = np.reshape(front, (64, 64))
            reshaped_df = pd.DataFrame(reshaped_data)
            reshaped_last = pd.DataFrame(last)

            # Save observation (npy + csv) and activation (csv)
            np.save(new_file_path_f_npy, reshaped_data)
            reshaped_df.to_csv(new_file_path_f_csv, index=False, header=False)
            reshaped_last.to_csv(new_file_path_l_csv, index=False, header=False)

            max_value = front.max()
            min_value = front.min()
            max_min_values.append([filename, max_value, min_value])

        print(f"File {filename} processed.")

        max_min_df = pd.DataFrame(max_min_values, columns=['Filename', 'Max_Value', 'Min_Value'])
        summary_dir = './data/observation/summary/'
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        summary = f"{new_filename} obs_summary.csv"
        # max_min_df.to_csv(os.path.join(summary_dir, summary), index=False)

print('All files processed.')