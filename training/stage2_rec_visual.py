import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

"""Observation & activation reconstruction visualization for the stage-2 model."""


class CustomDataset(Dataset):
    """Dataset that loads a single 2-channel mixed observation .npy file."""

    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        self.file_list = [os.path.join(data_dir, file_name)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image = np.load(file_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image


def fourier_transform(img):
    """
    Compute the 2D Fourier transform magnitude of an image.
    """
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    magnitude = np.log(1 + np.abs(F_shifted))
    return magnitude


def visualize_images(save, file_name, figure_dir, image1, image2, image3, image4):
    """
    Visualize ground-truth kernel, observation, activation map, and inferred kernel.
    Layout:
      [Observation] [Activation map] [Inference] [Ground truth]
    """
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))

    axes[3].imshow(image1, cmap='coolwarm', vmin=0, vmax=1)
    axes[3].set_title("Ground Truth")
    axes[3].axis('off')

    axes[0].imshow(image2, cmap='coolwarm', vmin=0, vmax=1)
    axes[0].set_title("Observation")
    axes[0].axis('off')

    axes[1].imshow(image3, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Activation Map")
    axes[1].axis('off')

    axes[2].imshow(image4, cmap='coolwarm', vmin=0, vmax=1)
    axes[2].set_title("Inference")
    axes[2].axis('off')

    plt.tight_layout()
    plt.suptitle(file_name, fontsize=11)
    figure_path = os.path.join(figure_dir, file_name)
    if save == 't':
        plt.savefig(figure_path)
    else:
        plt.show()
    plt.close(fig)


# Model loading / configuration
model_path = '/data/coding/stage1/kl_beta1e-06_alpha0.7_g0.97/vae_kl_time1'

fig_save = 'f'
device = "cuda" if torch.cuda.is_available() and fig_save == 't' else "cpu"
save_dir = f'/data/coding/figure/observation_comparison'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

vae = AutoencoderKL.from_pretrained(
    model_path, in_channels=1, out_channels=1, ignore_mismatched_sizes=True, low_cpu_mem_usage=False
)  # stage1 model used for decoding
vae.requires_grad_(False)
vae.eval()
vae.to(device)

main_folder = "/data/coding/channel2_mixed_data/"  # _train&test/test/
model_path_stage2 = f"/data/coding/stage2/kl/time1"

model_original_path = "/data/coding/vae_test2"  # madebyollin/sdxl-vae-fp16-fix

vae2 = AutoencoderKL.from_pretrained(
    model_path_stage2, in_channels=2, out_channels=1, ignore_mismatched_sizes=True, low_cpu_mem_usage=False
)  # stage2 model used for encoding
vae2.requires_grad_(False)
vae2.eval()
vae2.to(device)

results = []
i = 0
j=0
original_file_dir = "/data/coding/kernel/test/"  # use test split here

# Iterate over all kernel subfolders
k_num = 0
for sub_folder_name in tqdm(os.listdir(main_folder)):
    k_num = int(sub_folder_name[13:])
    inference = []

    data_dir = os.path.join(main_folder,sub_folder_name)
    original_file_path = os.path.join(original_file_dir, f"Kernel number{k_num}nom_2d.npy")
    if os.path.isfile(original_file_path):
        j+=1

        file_amount = 0   # Per kernel
        a = [name for name in os.listdir(data_dir)]

        for file_index in range(1, 500 + 1):

            random_npy_file = f"{sub_folder_name} {file_index}channel2 mixed.npy"

            try:
                original_kernel_file = f"/data/coding/data_noise/kernel00/Kernel number{k_num} nom_2d.npy"
                kernel = np.load(original_kernel_file)

            except Exception as e:
                original_kernel_file = f"/data/coding/all_kernel/Kernel number{k_num}nom_2d.npy"
                kernel = np.load(original_kernel_file)
            kernel_data = torch.tensor(kernel).to(device)

            file_path = os.path.join(data_dir, random_npy_file)
            if os.path.isfile(file_path):
                original_file = np.load(file_path)
                dataset = CustomDataset(data_dir, random_npy_file)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

                # Reconstruction
                for batch_idx, batch in enumerate(dataloader):
                    batch = batch.to(device)
                    latents_sample = vae2.encode(batch[0]).latent_dist.sample()

                    reconstructed = vae.decode(latents_sample).sample
                    mse = F.mse_loss(kernel_data, reconstructed[0][0])
                    rmse = torch.sqrt(mse)
                    mae = torch.mean(torch.abs(kernel_data - reconstructed[0][0]))
                    result_mse = mse * 4
                    result_rmse = rmse * 2
                    result_mae = mae * 2

                    # Log error statistics for this sample
                    statics = [k_num,result_mae.item(),result_mse.item(),result_rmse.item()]
                    results.append(statics)
                    i += 1

                    reconstructed_cpu = reconstructed.detach().cpu().numpy()
                rec = reconstructed_cpu[0][0]
                # Visualization: kernel, observation (channel 0), activation map (channel 1), reconstruction
                visualize_images(fig_save,f"{sub_folder_name} {file_index}observation", save_dir, kernel, original_file[0], original_file[1],rec)
                file_amount += 1
                if file_amount >= 60:
                    break
            else:
                pass

results1 = np.array(results)
summary_max = [999,np.max(results1[:, 1]),np.max(results1[:,2]),np.max(results1[:,3])]
summary_mean = [999,np.mean(results1[:,1]),np.mean(results1[:,2]),np.mean(results1[:,3])]
results.append(summary_max)
results.append(summary_mean)
result_df = pd.DataFrame(results)

result_df.columns = [f"kernel_num", "mae", "mse", "rmse"]

result_df.to_csv(f"/data/coding/ERRORS/error.csv", index=False, header=True)

