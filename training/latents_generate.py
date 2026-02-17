import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

"""Generate mean/logvar latent labels for the 100 kernels using a trained stage-1 VAE."""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class CustomDataset(Dataset):
    """Simple dataset that loads a single kernel .npy file."""

    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        self.file_list = [os.path.join(data_dir, file_name)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image = np.load(file_path)
        # Convert to single-channel tensor
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image

def visualize_images(file_name, figure_dir, image1, image2):
    """Optional helper to visualize original vs reconstructed kernel."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(image1, cmap='coolwarm', vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(image2, cmap='coolwarm', vmin=0, vmax=1)
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.show()


# Load model
data_dir = './data/kernel/all/'
model_path = './data/models/stage1'
# ntype_dir = "./data/kernel_latents/6_type_kernels.npy"
# n_type = np.load(ntype_dir)


vae = AutoencoderKL.from_pretrained(model_path, in_channels=1, out_channels=1
                                    ,ignore_mismatched_sizes=True,low_cpu_mem_usage=False)
vae.requires_grad_(False)
vae.eval()
vae.to(device)

latent_list = []
kernel_list = []
i=0
for file_index in tqdm(range(1,101)):

    random_npy_file = f"Kernel number{file_index}nom_2d.npy"
    # print(f"The file chosen：{random_npy_file}")

    file_path = os.path.join(data_dir, random_npy_file)
    original_file = np.load(file_path)
    # if np.isin([file_index], n_type)[0]:

    # original_file_path = os.path.join(original_file_dir, f"Kernel number{file_index}nom_2d.npy")

    dataset = CustomDataset(data_dir, random_npy_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    i+=1
    # Image encoding to generate latents
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        latents = vae.encode(batch).latent_dist
        kernel_latents = torch.stack([latents.mean[0], latents.logvar[0]], dim=0)
        latent_list.append(kernel_latents)
        # latent_list.append(batch[0])  # Generate kernels
        # break
        # print(latents.mean.shape)   # torch.Size([1, 4, 8, 8]) First dimension is batch_size, need to change to (4,8,8)
        # print(latents.logvar.shape)   # torch.Size([1, 4, 8, 8])
        # print(latents.shape)   # AttributeError: 'DiagonalGaussianDistribution' object has no attribute 'shape'

stacked_kernel_latents = torch.stack(latent_list, dim=0)
stacked_kernel_latents_cpu = stacked_kernel_latents.cpu().numpy()

print(stacked_kernel_latents_cpu.shape)
print(i,'  files')
dir = './data/kernel_latents/'
if not os.path.exists(dir):
    os.makedirs(dir)
np.save(os.path.join(dir,'100kernel latent vectors.npy'), stacked_kernel_latents_cpu)
# np.save('./data/kernel_latents/100kernels.npy', stacked_kernel_latents_cpu)


# Plotting
# rec = reconstructed_cpu[0][0]
# visualize_images(random_npy_file, figure_dir, original_file, rec)