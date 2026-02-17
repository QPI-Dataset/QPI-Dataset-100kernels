import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from torch import nn

"""Kernel reconstruction visualization for the stage-1 model."""


class CustomDataset(Dataset):
    def __init__(self, data_dir, kernel_file, nfold_file):
        """
        Dataset that loads a single kernel .npy file and its n-fold label.
        """
        self.data_dir = data_dir
        self.folds = []
        self.fold_array = np.load(nfold_file)
        self.file_list = []

        file_path = os.path.join(data_dir, kernel_file)
        if os.path.isfile(file_path):
            kernel_name = kernel_file[:-10]
            kernel_number = int(kernel_name.split('number')[-1])
            self.folds.append(self.fold_array[kernel_number - 1])
            self.file_list.append(file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image = np.load(file_path)
        nfold = self.folds[idx]
        # Convert to single-channel tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return nfold, image

# class CustomDataset(Dataset):
#     def __init__(self, data_dir, file_name):
#         self.data_dir = data_dir
#         self.file_name = file_name
#         self.file_list = [os.path.join(data_dir, file_name)]
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         file_path = self.file_list[idx]
#         image = np.load(file_path)
#         image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Convert to single-channel tensor
#         return image

def cal_symmetry_loss0(batch_n_fold, batch_kernel, batch_rec, middle_size=40):
    """
    Legacy symmetry loss implementation using OpenCV rotation (not used in training).
    Kept here only for debugging / comparison.
    """
    error_sum = 0
    batch_rec1 = batch_rec
    batch_rec = batch_rec.detach().cpu().numpy()
    batch_n_fold = batch_n_fold.detach().cpu().numpy()

    for i in range(batch_kernel.shape[0]):
        kernel = batch_kernel[i]
        rec = batch_rec[i]
        n_fold = batch_n_fold[i]
        if n_fold != 1:
            angle_dict = {2: 180, 3: 120, 4: 90, 6: 60}
            rotation_matrix = cv2.getRotationMatrix2D((32, 32), angle_dict[n_fold], 1)
            rotated_pixels = cv2.warpAffine(
                rec[0],
                rotation_matrix,
                (64, 64),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            rotated_pixels = torch.tensor(
                rotated_pixels, requires_grad=True, device=batch_rec1.device
            )
            start_x = (64 - middle_size) // 2
            start_y = (64 - middle_size) // 2
            middle_pixels = kernel[0][start_y:start_y + middle_size, start_x:start_x + middle_size]
            rotated_middle_pixels = rotated_pixels[start_y + 1:1 + start_y + middle_size,
                                                   start_x:start_x + middle_size]
            # mse over middle region
            error = torch.mean((kernel[0] - rotated_pixels) ** 2)
        else:
            error = 0
        error_sum += error
    return error_sum

def get_rotation_matrix(batch_size, angle_deg, center, device):
    """Build an affine rotation matrix tensor of shape [batch_size, 2, 3]."""
    theta = torch.tensor(angle_deg * 3.1415926 / 180).to(device)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    affine_matrix = torch.zeros(batch_size, 2, 3).to(device)
    affine_matrix[:, 0, 0] = cos
    affine_matrix[:, 0, 1] = -sin
    affine_matrix[:, 1, 0] = sin
    affine_matrix[:, 1, 1] = cos
    # Centered rotation; translation is set to zero here
    tx = (1 - cos) * center[0] + sin * center[1]
    ty = -sin * center[0] + (1 - cos) * center[1]
    affine_matrix[:, 0, 2] = 0  # tx / (center[0] * 2)
    affine_matrix[:, 1, 2] = 0  # ty / (center[1] * 2)
    return affine_matrix


def cal_symmetry_loss(batch_n_fold, batch_kernel, batch_rec, middle_size=40):
    """
    Compute symmetry-consistency loss for reconstructed kernels (debug/visualization version).
    Returns cropped rotated and ground-truth patches for inspection.
    """
    device = batch_kernel.device
    B, C, H, W = batch_kernel.shape

    for n_fold in [2, 3, 4, 6]:
        idxs = (batch_n_fold == n_fold).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        rec_sub = batch_rec[idxs]
        kernel_sub = batch_kernel[idxs]
        angle = {2: 180, 3: 120, 4: 90, 6: 60}[n_fold]

        # Build rotation affine matrix
        affine_matrix = get_rotation_matrix(len(idxs), angle_deg=angle, center=(W / 2, H / 2), device=device)
        print((W / 2, H / 2))
        print(affine_matrix)

        # Sample rotated images
        grid = F.affine_grid(affine_matrix, rec_sub.size(), align_corners=True)
        rotated = F.grid_sample(rec_sub, grid, mode='bilinear', padding_mode='border', align_corners=False)

        start_x = (W - middle_size) // 2
        start_y = (H - middle_size) // 2
        end_x = start_x + middle_size
        end_y = start_y + middle_size
        rotated_middle = rotated[:, :, start_y:end_y, start_x:end_x]
        kernel_middle = kernel_sub[:, :, start_y:end_y, start_x:end_x]

    return rotated_middle, kernel_middle

def visualize_images(save, max_loss, rec_loss, file_name, figure_dir, image1, image2):
    """Visualize original vs reconstructed kernel with simple error metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(image1, cmap='coolwarm', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='coolwarm', vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    file_name_ = file_name[:-10]
    plt.text(32, 65, f"MAE:  {rec_loss:.6f}", ha='center', va='top', fontsize=12, color='blue')
    plt.text(32, 70, f"Max abs error:  {max_loss:.6f}", ha='center', va='top', fontsize=12, color='blue')
    figure_path = os.path.join(figure_dir, file_name_)
    if save == 't':
        plt.savefig(figure_path)
    else:
        plt.show()
    plt.close(fig)

train_test = 'train'
beta = 1e-06

# Model paths / configuration
model_path = './data/models/stage1'

model_ori_path = "./data/models/pretrained_vae"  # madebyollin/sdxl-vae-fp16-fix
data_dir = f'./data/kernel/{train_test}/'
figure_dir = f'./data/figures/stage1/{train_test}/'
nfold_dir = "./data/kernel_latents/n_fold_noise.npy"
fig_save = 't'

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

device = "cuda" if torch.cuda.is_available() and fig_save == 't' else "cpu"
vae = AutoencoderKL.from_pretrained(model_path, in_channels=1, out_channels=1
                                    ,ignore_mismatched_sizes=True,low_cpu_mem_usage=False)
vae.requires_grad_(False)

vae.eval()
vae.to(device)

all_files = os.listdir(data_dir)
npy_files = [file for file in all_files if file.endswith(".npy")]

results = [[0 for _ in range(4)] for _ in range(len(npy_files))]
i = 0
mapping = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}

for file_index in tqdm(range(1, 101)):
    random_npy_file = f"Kernel number{file_index}nom_2d.npy"

    file_path = os.path.join(data_dir, random_npy_file)
    if os.path.isfile(file_path):
        original_file = np.load(file_path)

        dataset = CustomDataset(data_dir,random_npy_file, nfold_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # Image reconstruction
        for batch_idx, (fold_num, batch) in enumerate(dataloader):
            fold_num = fold_num.to(device)
            batch = batch.to(device)

            # Encode and decode
            latents = vae.encode(batch).latent_dist
            # latents_sample = vae.encode(batch).latent_dist.sample()
            latents_sample = latents.sample()
            reconstructed = vae.decode(latents_sample).sample

            rec_loss = torch.sqrt(F.mse_loss(batch, reconstructed))

            mse = F.mse_loss(batch, reconstructed)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(batch - reconstructed))
            max = torch.max(torch.abs(batch - reconstructed))*2
            # print(rec_loss)
            # print(reconstructed[0][0].shape)
            result_mse = mse * 4
            result_rmse = rmse * 2
            result_mae = mae * 2

            results[i - 1][1] = result_mae.item()
            results[i - 1][2] = result_mse.item()
            results[i - 1][0] = file_index
            results[i - 1][3] = result_rmse.item()
            i += 1
            # print(f"{relative_error :.4f}")

            reconstructed_cpu = reconstructed.detach().cpu().numpy()
            rec = reconstructed_cpu[0][0]
            visualize_images(fig_save, max, result_mae,  random_npy_file, figure_dir, original_file, rec)
    else:
        pass


