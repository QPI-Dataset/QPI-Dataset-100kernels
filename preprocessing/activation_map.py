import os
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Generate 64x64 activation maps for every activation CSV of each kernel.
Each (x, y) coordinate pair is mapped into a 2D grid.
Update the paths below to match your local data layout.
"""

# Root folder that contains one subfolder per kernel, each with activation CSV files
main_folder = "./data/activation/csv/"

# Placeholder for optional statistics (not currently used)
results = [[0 for _ in range(100)] for _ in range(500)]
sub_folder_index = 1

# Iterate over kernel subfolders
for sub_folder_name in tqdm(os.listdir(main_folder)):
    sub_folder_path = os.path.join(main_folder, sub_folder_name)

    output_dir_obs_npy = f"./data/activation/map/{sub_folder_name}/"
    os.makedirs(output_dir_obs_npy, exist_ok=True)

    # Ensure the subfolder exists
    if os.path.isdir(sub_folder_path):
        file_names = [name for name in os.listdir(sub_folder_path)]
        # Loop over all activation CSVs for this kernel
        for file_index in range(1, len(file_names) + 1):
            file_name = f"{sub_folder_name} {file_index}activation.csv"
            file_path = os.path.join(sub_folder_path, file_name)

            if os.path.isfile(file_path):
                # Read CSV file
                try:
                    data = pd.read_csv(file_path, header=None)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue

                # Convert activation list to (x, y) coordinate pairs
                coordinates = []
                for i in range(0, len(data) - 1, 2):
                    x = data.iloc[i].item()
                    y = data.iloc[i + 1].item()
                    coordinates.append((x, y))

                # Create 64x64 grid; 1 indicates activated bin
                grid = np.zeros((64, 64))
                count_in_range = 0

                # Map coordinates into grid
                for x, y in coordinates:
                    # Only keep coordinates within [-100, 100] range
                    if -100 <= x <= 100 and -100 <= y <= 100:
                        grid_x = int((x + 200) / (400 / 64))
                        grid_y = int((y + 200) / (400 / 64))

                        # Ensure indices are in [0, 63]
                        if 0 <= grid_x < 64 and 0 <= grid_y < 64:
                            # (x, y) matches observation coordinate convention;
                            # (y, x) would correspond to transposed coordinates
                            grid[grid_x][grid_y] = 1
                            count_in_range += 1

                # Save as .npy activation map
                front_filename_npy = f"{sub_folder_name} {file_index}activation map"
                new_file_path_f_npy = os.path.join(output_dir_obs_npy, front_filename_npy)
                grid_df = pd.DataFrame(grid)
                np.save(new_file_path_f_npy, grid)

        print(f"Kernel number{sub_folder_index} done")
    sub_folder_index += 1

result_df = pd.DataFrame(results)

print('All files processed.')