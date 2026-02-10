import os
import numpy as np
from tqdm import tqdm
import shutil
import math

"""
Split 2-channel mixed data for each kernel into train and test sets.
By default, ~20% of samples are copied to the test set and the rest to the train set.
Update the paths below to match your local data layout.
"""

main_folder = "D:/桌面/STIR/64X64 kernels/data_noise/channel2_mixed_data/"
train = "D:/桌面/STIR/64X64 kernels/data_noise/channel2_mixed_data_train&test/train/"
test = "D:/桌面/STIR/64X64 kernels/data_noise/channel2_mixed_data_train&test/test/"

# Iterate over kernel subfolders
for sub_folder_index in tqdm(range(1, 101)):
    sub_folder_name = f"Kernel number{sub_folder_index}"
    sub_folder_path = os.path.join(main_folder, sub_folder_name)

    # Only split kernels that are included in the stage-1 training set
    kernel_name = f"Kernel number{sub_folder_index}nom_2d.npy"
    kernel_path = os.path.join("D:/桌面/STIR/64X64 kernels/data_noise/kernel/train/", kernel_name)
    if os.path.isfile(kernel_path) and os.path.isdir(sub_folder_path):
        source_folder = sub_folder_path
        folder_train = os.path.join(train, sub_folder_name)
        folder_test = os.path.join(test, sub_folder_name)
        os.makedirs(folder_train, exist_ok=True)
        os.makedirs(folder_test, exist_ok=True)

        npy_files = [f for f in os.listdir(source_folder) if f.endswith('.npy')]

        # Choose 20% of files for test set
        num_to_test = math.ceil(len(npy_files) * 0.2)

        np.random.shuffle(npy_files)

        for file in npy_files[:num_to_test]:
            shutil.copy(os.path.join(source_folder, file), folder_test)

        for file in npy_files[num_to_test:]:
            shutil.copy(os.path.join(source_folder, file), folder_train)

print('All files split into train and test.')
