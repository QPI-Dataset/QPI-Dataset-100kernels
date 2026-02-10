import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""Visualization of observations and/or activation maps per kernel."""

# Activation summary: per-observation activation counts or labels
act_summary = pd.read_csv("D:/桌面/STIR/experimental_data/64X64/result_with_labels.csv", header=None)

# Root folder containing per-kernel observation or activation-map .npy files
main_folder = "D:/桌面/STIR/64X64 kernels stage2/data_observation/observation_nom/"

# Iterate over all kernel subfolders
sub_folder_index = 0
for sub_folder_name in os.listdir(main_folder):
    sub_folder_index += 1
    sub_folder_path = os.path.join(main_folder, sub_folder_name)
    output_dir_obs_npy = f"D:/桌面/STIR/64X64 kernels stage2/figure/observation_figure/Kernel number{sub_folder_index}/"

    if os.path.isdir(sub_folder_path):
        # For each kernel, plot up to 500 activation maps / observations
        for file_index in range(1, 501, 500):
            fig, axes = plt.subplots(5, 10, figsize=(20, 10))  # 5 rows x 10 columns
            axes = axes.flatten()
            i = 0
            for j in range(500):
                file_name = f"{sub_folder_name} {file_index + j}activation map.npy"
                # Alternatively: file_name = f"{sub_folder_name} {file_index + j}observation_nom.npy"
                file_path = os.path.join(sub_folder_path, file_name)
                if os.path.isfile(file_path):
                    if i > 49:
                        plt.tight_layout()
                        plt.show()
                        plt.close(fig)
                        i = 0

                    act_num = act_summary.iloc[file_index + j - 1, sub_folder_index - 1]

                    data = np.load(file_path)
                    axes[i].imshow(data, cmap='coolwarm')
                    axes[i].axis('off')
                    axes[i].set_title(f"activation amount {act_num}", fontsize=8)
                    i += 1

            plt.tight_layout()

            print(f"{sub_folder_name}")
            output_path = os.path.join(output_dir_obs_npy, f"{sub_folder_name} {file_index // 50 + 1}.png")
            # plt.savefig(output_path)
            plt.show()
            plt.close(fig)

        print(f"{sub_folder_name} done")

print('All observation / activation figures generated.')