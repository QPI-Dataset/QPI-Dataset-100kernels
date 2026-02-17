import numpy as np
import matplotlib.pyplot as plt
import os

"""Visualize all 100 kernels in a 10x10 grid."""


def main(main_folder: str) -> None:
    """Load kernel .npy files from `main_folder` and visualize them in a grid."""
    if os.path.isdir(main_folder):
        fig, axes = plt.subplots(10, 10, figsize=(20, 10))
        axes = axes.flatten()

        for j in range(1, 101):
            try:
                file_name = f"Kernel number{j} nom_2d.npy"
                file_path = os.path.join(main_folder, file_name)
                data = np.load(file_path)
            except Exception:
                file_name = f"Kernel number{j}nom_2d.npy"
                file_path = os.path.join(main_folder, file_name)
                data = np.load(file_path)

            axes[j - 1].imshow(data, cmap='coolwarm')
            axes[j - 1].axis('off')
            axes[j - 1].set_title(f"Kernel number{j}", fontsize=10)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print('All kernels visualized.')


if __name__ == '__main__':
    folder_path = "./data/kernel/all/"
    main(folder_path)