## STIR: Symmetry-Aware Two-Stage Inverse Reconstruction

This repository contains code for a two-stage reconstruction framework on the QPI (Quantitative Phase Imaging) kernel dataset.
The pipeline includes:

- **Data preprocessing** from the original QPI dataset
- **Stage-1** symmetry-aware VAE training on kernel images
- **Stage-2** encoder training to align observation features with stage-1 latents
- Optional **visualization and analysis scripts**

The original data are provided in the QPI dataset release:
[`QPI-Dataset-100kernels`](https://github.com/QPI-Dataset/QPI-Dataset-100kernels/releases/tag/v1.0.0).

---

### Directory Structure

```
preprocessing/
  split_data_kernel.py    # Extract and normalize kernel images from raw CSV
  split_data.py           # Extract observation and activation data
  obs_data_nom.py         # Normalize observation data
  activation_map.py       # Generate activation maps
  act_obs_mix.py          # Merge into 2-channel inputs (obs + activation)
  nfold_generate.py       # Generate symmetry-fold labels
training/
  train_test_file_copy.py # Split training and test sets
  stage1.py               # Stage-1: VAE with symmetry loss on kernels
  stage1_rec_visual.py    # Stage-1 reconstruction visualization
  latents_generate.py     # Generate latent vectors from stage-1
  stage2.py               # Stage-2: encoder from observations to latents
  stage2_rec_visual.py    # Stage-2 reconstruction visualization
  stage2-1step.py         # 1-step baseline model
plotting/
  visual_kernel.py        # Visualize all kernels
  visual_obs.py           # Visualize observations
  tsne.py                 # t-SNE of latent vectors
  denoise.py              # Denoising algorithm comparison
```

---

### 1. Quick Start

This section gives a minimal end-to-end pipeline from raw QPI data to trained models and evaluation.

#### 1.1 Prepare environment

Install dependencies (Python 3.8+, PyTorch with CUDA, Diffusers, etc.):

```bash
conda create -y -n stir_env python=3.10
conda activate stir_env

pip install numpy pandas tqdm matplotlib scikit-learn pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors
```

#### 1.2 Download QPI dataset

```bash
mkdir -p data/raw
wget -O data/raw/qpi.tar.gz \
  https://github.com/QPI-Dataset/QPI-Dataset-100kernels/archive/refs/tags/v1.0.0.tar.gz
tar -xzf data/raw/qpi.tar.gz -C data/raw --strip-components=1
```

The raw CSV files should appear under `data/raw/` (look for a folder like `64X64 vp coords/`).

#### 1.3 Download pretrained VAE

```bash
python3 -c "
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')
vae.save_pretrained('./data/models/pretrained_vae')
print('Pretrained VAE saved.')
"
```

#### 1.4 Create data directories

```bash
mkdir -p data/kernel/{all,train,test}
mkdir -p data/observation/{csv,npy,nom}
mkdir -p data/activation/{csv,map}
mkdir -p data/channel2_mixed/{all,train,test}
mkdir -p data/kernel_latents
mkdir -p data/models/{pretrained_vae,stage1,stage2}
mkdir -p data/loss/{stage1,stage2}
mkdir -p data/figures data/errors
```

#### 1.5 Run preprocessing

Edit the path variables at the top of each script in `preprocessing/` to match your data layout, then run in order:

```bash
python preprocessing/split_data_kernel.py   # Extract/normalize kernels
python preprocessing/split_data.py           # Extract observations and activations
python preprocessing/obs_data_nom.py         # Normalize observations
python preprocessing/activation_map.py       # Generate activation maps
python preprocessing/act_obs_mix.py          # Merge 2-channel inputs
python preprocessing/nfold_generate.py       # Generate n-fold symmetry labels
```

#### 1.6 Create train/test split

```bash
python training/train_test_file_copy.py
```

This creates 80/20 train/test splits for both kernels and 2-channel observation data.

#### 1.7 Train Stage-1 (Kernel VAE with symmetry loss)

```python
import sys
sys.path.insert(0, './training')
from diffusers import AutoencoderKL
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pretrained 3-channel VAE weights for proper initialization
vae_3ch = AutoencoderKL.from_pretrained('./data/models/pretrained_vae')
conv_in_w3 = vae_3ch.encoder.conv_in.weight.data.clone()
conv_out_w3 = vae_3ch.decoder.conv_out.weight.data.clone()
conv_out_b3 = vae_3ch.decoder.conv_out.bias.data.clone()
del vae_3ch

# Load VAE with 1-channel config
vae = AutoencoderKL.from_pretrained(
    './data/models/pretrained_vae', in_channels=1, out_channels=1,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=False
)

# IMPORTANT: Initialize mismatched layers from pretrained weights.
# Without this, random init causes NaN loss from epoch 1.
vae.encoder.conv_in.weight.data = conv_in_w3.mean(dim=1, keepdim=True)
vae.decoder.conv_out.weight.data = conv_out_w3.mean(dim=0, keepdim=True)
vae.decoder.conv_out.bias.data = conv_out_b3[:1]

vae.config.in_channels = 1
vae.config.out_channels = 1
vae.config.sample_size = 64
vae.to(device)

from stage1 import run
run(
    vae,
    fold_file='./data/kernel_latents/n_fold_noise.npy',
    train_data_dir='./data/kernel/train',
    test_data_dir='./data/kernel/test',
    model_save_path='./data/models/stage1',
    loss_dir='./data/loss/stage1/',
    alpha=0.7, beta=1e-6,
    num_epochs=300, batch_size=8,
    initial_lr=1e-4, step_size=10, gamma=0.97,
)
```

#### 1.8 Generate latent vectors

```python
import os, numpy as np, torch
from diffusers import AutoencoderKL
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = AutoencoderKL.from_pretrained(
    './data/models/stage1', in_channels=1, out_channels=1,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=False
)
vae.requires_grad_(False); vae.eval(); vae.to(device)

latent_list = []
for i in tqdm(range(1, 101), desc='Encoding kernels'):
    fpath = os.path.join('./data/kernel/all', f'Kernel number{i}nom_2d.npy')
    if not os.path.isfile(fpath): continue
    image = np.load(fpath)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode(tensor).latent_dist
        latent_list.append(torch.stack([latents.mean[0], latents.logvar[0]], dim=0))

stacked = torch.stack(latent_list, dim=0).cpu().numpy()
np.save('./data/kernel_latents/100kernel_latent_vectors.npy', stacked)
print(f'Latent vectors saved: {stacked.shape}')
```

#### 1.9 Train Stage-2 (Observation-to-Latent Encoder)

```python
import sys
sys.path.insert(0, './training')
from diffusers import AutoencoderKL
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vae_3ch = AutoencoderKL.from_pretrained('./data/models/pretrained_vae')
conv_in_w3 = vae_3ch.encoder.conv_in.weight.data.clone()
conv_out_w3 = vae_3ch.decoder.conv_out.weight.data.clone()
conv_out_b3 = vae_3ch.decoder.conv_out.bias.data.clone()
del vae_3ch

vae = AutoencoderKL.from_pretrained(
    './data/models/pretrained_vae', in_channels=2, out_channels=1,
    ignore_mismatched_sizes=True, low_cpu_mem_usage=False
)

# Initialize mismatched layers (2-channel input from 3-channel pretrained)
vae.encoder.conv_in.weight.data = conv_in_w3[:, :2, :, :] * (3.0 / 2.0)
vae.decoder.conv_out.weight.data = conv_out_w3.mean(dim=0, keepdim=True)
vae.decoder.conv_out.bias.data = conv_out_b3[:1]

vae.config.in_channels = 2
vae.config.out_channels = 1
vae.config.sample_size = 64
vae.to(device)

from stage2 import run
run(
    fold_file='./data/kernel_latents/n_fold_noise.npy',
    train_data_length=250,
    vae=vae,
    train_data_dir='./data/channel2_mixed/train',
    test_data_dir='./data/channel2_mixed/test',
    latents_file='./data/kernel_latents/100kernel_latent_vectors.npy',
    model_save_path='./data/models/stage2',
    loss_dir='./data/loss/stage2/',
    num_epochs=50, batch_size=8,
    initial_lr=5e-5, step_size=5, gamma=0.97,
    kernel_dir='./data/kernel/train',
)
```

#### 1.10 Evaluate and visualize

```bash
python training/stage1_rec_visual.py   # Stage-1 reconstruction
python training/stage2_rec_visual.py   # Stage-2 end-to-end reconstruction

python plotting/visual_kernel.py       # Kernel grid display
python plotting/visual_obs.py          # Observation display
python plotting/tsne.py                # t-SNE of latent vectors
python plotting/denoise.py             # Denoising comparison
```

Make sure paths in each script match your directory layout.

---

### 2. Architecture

#### Two-Stage Pipeline

```
Raw CSV -> Preprocessing -> Stage-1 (Kernel VAE) -> Latent Generation -> Stage-2 (Obs->Latent) -> Reconstruction
```

**Stage-1:** Trains an `AutoencoderKL` (from `diffusers`) on single-channel 64x64 kernel images.
Loss = MSE_reconstruction + alpha * symmetry_loss + beta * KL_divergence.
The symmetry loss enforces rotational invariance using `F.affine_grid`/`F.grid_sample` based on each kernel's n-fold class (1, 2, 3, 4, or 6-fold).

**Latent Generation:** Encodes all 100 kernels through the trained stage-1 encoder, producing a `(100, 2, 4, 8, 8)` tensor of (mean, logvar) pairs.

**Stage-2:** Trains another `AutoencoderKL` encoder (2 input channels: observation + activation map) to predict stage-1's latent vectors. Loss = MSE(mean_pred, mean_target) + MSE(logvar_pred, logvar_target). At inference, stage-2 encoder output is decoded by stage-1's decoder.

#### Key Hyperparameters

| Parameter | Stage-1 | Stage-2 |
|-----------|---------|---------|
| Learning rate | 1e-4 | 5e-5 |
| Epochs | 300 | 50 |
| Batch size | 8 | 8 |
| LR step size | 10 | 5 |
| LR gamma | 0.97 | 0.97 |
| alpha (symmetry weight) | 0.7 | -- |
| beta (KL weight) | 1e-6 | -- |

#### Pretrained VAE Backbone

The backbone is `madebyollin/sdxl-vae-fp16-fix` from HuggingFace. When adapting from 3-channel to 1-channel or 2-channel, the mismatched layers (`encoder.conv_in`, `decoder.conv_out`) must be initialized from the pretrained weights by averaging across channels. **Without this step, training produces NaN loss from epoch 1** because the default random initialization is incompatible with the pretrained internal layers.

---

### 3. Data Organization

```
data/
  raw/                          # Raw QPI CSV files
  kernel/
    all/                        # All 100 kernels: Kernel number{X}nom_2d.npy (64x64)
    train/                      # 80% train split
    test/                       # 20% test split
  observation/
    csv/Kernel number{X}/       # Per-kernel observation CSV files
    npy/Kernel number{X}/       # Per-kernel observation NPY files
    nom/Kernel number{X}/       # Normalized observations
  activation/
    csv/Kernel number{X}/       # Per-kernel activation CSV files
    map/Kernel number{X}/       # Per-kernel activation map NPY files
  channel2_mixed/
    all/Kernel number{X}/       # 2-channel stacked files (obs + activation), shape (2,64,64)
    train/Kernel number{X}/     # Train split
    test/Kernel number{X}/      # Test split (held-out obs from train kernels)
  kernel_latents/
    n_fold_noise.npy            # N-fold symmetry labels, shape (100,)
    100kernel_latent_vectors.npy # Stage-1 latent vectors, shape (100,2,4,8,8)
  models/
    pretrained_vae/             # Downloaded HuggingFace VAE
    stage1/                     # Trained stage-1 model
    stage2/                     # Trained stage-2 model
  loss/
    stage1/                     # Stage-1 loss logs
    stage2/                     # Stage-2 loss logs
  errors/                       # Reconstruction error CSVs
  figures/                      # Visualization outputs
```

---

### 4. Important Notes

- All Python scripts use **hardcoded paths** at the top of each file. Edit these path variables before running.
- Both `stage1.py` and `stage2.py` use `matplotlib.use('Agg')` for headless rendering on GPU nodes.
- Kernel numbering is **1-indexed** (1-100) and must stay consistent across filenames, nfold labels, and latent arrays.
- Per-kernel folder names follow the pattern `Kernel number{X}` -- stage-2 dataset loading depends on this naming.
- No formal test suite exists. Validation is done via train/test loss tracking and visual inspection.
- CUDA GPU is required for training.
- Gradient clipping (`max_norm=1.0`) is applied in both training scripts for numerical stability.
