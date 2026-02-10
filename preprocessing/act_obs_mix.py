import os
import numpy as np
from tqdm import tqdm

"""
Merge normalized observations and activation maps into 2-channel inputs.
Channel 0: normalized observation
Channel 1: activation map
Update the paths below to match your local data layout.
"""

# Root folders
main_folder1 = "D:/桌面/STIR/64X64 kernels/data_noise/observation_nom/"
main_folder2 = "D:/桌面/STIR/64X64 kernels/data_noise/activation_map/data_npy/"
output_folder = 'D:/桌面/STIR/64X64 kernels/data_noise/channel2_mixed_data/'

# Iterate over all kernel subfolders
for sub_folder_name in tqdm(os.listdir(main_folder1)):
    sub_folder_path1 = os.path.join(main_folder1, sub_folder_name)
    sub_folder_path2 = os.path.join(main_folder2, sub_folder_name)
    sub_folder_path3 = os.path.join(output_folder, sub_folder_name)

    os.makedirs(sub_folder_path3, exist_ok=True)

    # Ensure the observation subfolder exists
    if os.path.isdir(sub_folder_path1):
        file_names = [name for name in os.listdir(sub_folder_path1)]

        # For each observation, load paired activation map and stack to 2 channels
        for file_index in range(1, len(file_names) + 1):
            file_name1 = f"{sub_folder_name} {file_index}observation_nom.npy"
            file_name2 = f"{sub_folder_name} {file_index}activation map.npy"
            file_name3 = f"{sub_folder_name} {file_index}channel2 mixed"
            file_path1 = os.path.join(sub_folder_path1, file_name1)
            file_path2 = os.path.join(sub_folder_path2, file_name2)
            file_path3 = os.path.join(sub_folder_path3, file_name3)

            if os.path.isfile(file_path1):
                # Load observation and activation map
                array1 = np.load(file_path1)
                array2 = np.load(file_path2)

                # Merge into (2, 64, 64)
                combined_array = np.stack((array1, array2), axis=0)

                # Save merged 2-channel npy
                np.save(file_path3, combined_array)

print('All files processed.')
