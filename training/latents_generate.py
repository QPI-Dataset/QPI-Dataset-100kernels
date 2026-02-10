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

# Load model
data_dir = '/data/coding/all_kernel/'
model_path = ''

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

stacked_kernel_latents = torch.stack(latent_list, dim=0)
stacked_kernel_latents_cpu = stacked_kernel_latents.cpu().numpy()

dir = ''
if not os.path.exists(dir):
    os.makedirs(dir)
np.save(os.path.join(dir,'100kernel latent vectors.npy'), stacked_kernel_latents_cpu)

