## STIR: Symmetry-Aware Two-Stage Inverse Reconstruction

This repository contains code for a two-stage reconstruction framework on the QPI (Quantitative Phase Imaging) kernel dataset.  
The pipeline includes:

- **Data preprocessing** from the original QPI dataset
- **Stage-1** symmetry-aware VAE training on kernel images
- **Stage-2** encoder training to align observation features with stage‑1 latents
- Optional **visualization and analysis scripts**

The original data are provided in the QPI dataset release:  
[`QPI-Dataset-100kernels`](https://github.com/QPI-Dataset/QPI-Dataset-100kernels/releases/tag/v1.0.0).

---

### Directory Structure

- `preprocessing/`
  - `split_data_kernel.py`: extract and normalize kernel images from raw CSV files
  - `split_data.py`: extract observation and activation data for each kernel
  - `activation_map.py`: plot activation maps
  - `obs_data_nom.py`: normalize observation data
  - `act_obs_mix.py`: merge two-channel data
  - `nfold_generate.py`: generate symmetry-fold labels for each kernel
- `training/`
  - `train_test_file_copy.py`: split training and test sets by copying files
  - `stage1.py`: stage‑1 model training (VAE with symmetry loss on kernels)
  - `stage1_rec_visual.py`: stage‑1 inference / reconstruction visualization
  - `latents_generate.py`: generate kernel latent vectors for stage‑2
  - `stage2.py`: stage‑2 model training (encoder from observations to latents)
  - `stage2_rec_visual.py`: stage‑2 inference / reconstruction visualization
  - `stage2-1step.py`: 1‑step baseline model training
- `plotting/`
  - `visual_kernel.py`: visualize all kernels
  - `visual_obs.py`: visualize observations
  - `tsne.py`: t‑SNE visualization of latent vectors
  - `denoise.py`: visualize denoised observations

---

### 1. Quick Start

This section gives a minimal end-to-end pipeline from raw QPI data to trained models and evaluation.

1. **Prepare environment**
   - Install dependencies (Python 3.8+, PyTorch, Diffusers, etc.):

   ```bash
   pip install numpy pandas tqdm matplotlib torch torchvision diffusers scikit-learn pillow
   ```

2. **Download QPI dataset**
   - Download `QPI-Dataset-100kernels` from  
     [`QPI-Dataset-100kernels v1.0.0`](https://github.com/QPI-Dataset/QPI-Dataset-100kernels/releases/tag/v1.0.0)
   - Unpack to a folder, e.g. `/data/QPI-Dataset-100kernels/64X64_vp_coords/`

3. **Preprocess data (kernels, observations, activations)**
   - Edit paths in `preprocessing/split_data_kernel.py` and run:

   ```bash
   python preprocessing/split_data_kernel.py
   ```

   - Edit paths in `preprocessing/split_data.py` and run:

   ```bash
   python preprocessing/split_data.py
   ```

   - (Optional) Normalize observations:

   ```bash
   python preprocessing/obs_data_nom.py
   ```

   - Generate activation maps:

   ```bash
   python preprocessing/activation_map.py
   ```

   - Merge observations and activation maps into 2-channel inputs:

   ```bash
   python preprocessing/act_obs_mix.py
   ```

   - Generate n-fold symmetry labels:

   ```bash
   python preprocessing/nfold_generate.py
   ```

4. **Create train/test splits**

   - Edit paths in `training/train_test_file_copy.py` and run:

   ```bash
   python training/train_test_file_copy.py
   ```

   - After this step you should have kernel and 2-channel data under train/test directories such as:
     - `/data/coding/kernel/train/`, `/data/coding/kernel/test/`
     - `/data/coding/channel2_mixed_data_train&test/train/`, `/data/coding/channel2_mixed_data_train&test/test/`

5. **Train stage‑1 model (kernel VAE with symmetry loss)**

   - In `training/stage1.py`, set:
     - `train_data_dir`, `test_data_dir`
     - `fold_file` (output of `nfold_generate.py`)
     - `model_ori_path` (pretrained VAE weights)
     - `model_save_path` and `loss_dir`
   - Then run:

   ```bash
   python training/stage1.py
   ```

6. **Generate kernel latent vectors**

   - In `training/latents_generate.py`, set:
     - `data_dir` (all kernel .npy files, typically train + test)
     - `model_path` (trained stage‑1 model)
     - output directory/path for latent vectors
   - Run:

   ```bash
   python training/latents_generate.py
   ```

7. **Train stage‑2 encoder (2‑channel → latent)**

   - In `training/stage2.py`, set:
     - `train_data_dir`, `test_data_dir` (2‑channel mixed data train/test)
     - `latents_file` (kernel latent vectors from previous step)
     - `fold_file`, `model_ori_path`, `model_save_path`, `loss_dir`
   - Then run:

   ```bash
   python training/stage2.py
   ```

8. **(Optional) Train 1‑step baseline**

   - Configure `training/stage2-1step.py` similarly, then:

   ```bash
   python training/stage2-1step.py
   ```

9. **Evaluate and visualize**

   - Stage‑1 reconstruction:

   ```bash
   python training/stage1_rec_visual.py
   ```

   - Stage‑2 reconstruction from 2‑channel inputs:

   ```bash
   python training/stage2_rec_visual.py
   ```

   - Visualize kernels / observations / latent t‑SNE / denoising:

   ```bash
   python plotting/visual_kernel.py
   python plotting/visual_obs.py
   python plotting/tsne.py
   python plotting/denoise.py
   ```

   Make sure paths in each script match your directory layout (Linux paths like `/data/...`).

---

### 2. Environment Setup

**Dependencies (main):**

- Python 3.8+
- `numpy`
- `pandas`
- `tqdm`
- `matplotlib`
- `torch`, `torchvision`
- `diffusers` (for `AutoencoderKL` backbone)

Example installation:

```bash
pip install numpy pandas tqdm matplotlib torch torchvision diffusers
```

Make sure you have a CUDA-capable GPU and a proper PyTorch CUDA build installed for efficient training.

---

### 3. Data Preparation

#### 3.1 Download QPI Dataset

1. Download the QPI dataset release from  
   [`QPI-Dataset-100kernels v1.0.0`](https://github.com/QPI-Dataset/QPI-Dataset-100kernels/releases/tag/v1.0.0).

2. Unpack the archive to a folder, for example:

```bash
/data/QPI-Dataset-100kernels/
```

The raw CSV files (kernel + observation + activation) should be under a directory similar to:

```bash
/data/QPI-Dataset-100kernels/64X64_vp_coords/
```

Adjust the paths in preprocessing scripts to point to your actual location.

---

### 4. Preprocessing Pipeline

All preprocessing scripts are located in `preprocessing/`.  
By default, many of them contain hard-coded Windows-style paths (e.g., `D:/桌面/...`); you should **edit these paths** to your own data locations before running on Linux.

#### 4.1 Extract and Normalize Kernels

Script: `preprocessing/split_data_kernel.py`

- **Input**: raw CSV files containing multiple kernels and observations per file (from the QPI dataset)
- **Output**: one normalized `64 × 64` kernel image per file in `.npy` format

Key variables to edit:

- `input_folder`: directory where raw CSV files are stored (e.g. `/data/QPI-Dataset-100kernels/64X64_vp_coords/`)
- `output_folder_obs`: directory where per-kernel `.npy` files will be saved

Run:

```bash
python preprocessing/split_data_kernel.py
```

This produces files like:

```text
<output_folder_obs>/
  Kernel number1nom_2d.npy
  Kernel number2nom_2d.npy
  ...
```

Each file is a `64 × 64` normalized kernel.

#### 4.2 Extract Observations and Activations

Script: `preprocessing/split_data.py`

- **Input**: same raw CSV folder as above
- **Output**:
  - Observation images (both `.csv` and `.npy`)
  - Activation vectors (CSV)

Key variables to edit:

- `input_folder`
- `output_folder_obs_csv`
- `output_folder_obs_npy`
- `output_folder_act`

Run:

```bash
python preprocessing/split_data.py
```

This creates a structure like:

```text
<output_folder_obs_csv>/
  Kernel number1/
    Kernel number1 1observation.csv
    Kernel number1 2observation.csv
    ...
<output_folder_obs_npy>/
  Kernel number1/
    Kernel number1 1observation.npy
    ...
<output_folder_act>/
  Kernel number1/
    Kernel number1 1activation.csv
    ...
```

Each observation corresponds to one measurement/viewpoint for a given kernel.

#### 4.3 (Optional) Observation Normalization

Script: `preprocessing/obs_data_nom.py`

- Normalizes observation data across the dataset.

Edit input/output paths to point to your observation `.npy` or `.csv` directories, then run:

```bash
python preprocessing/obs_data_nom.py
```

#### 4.4 (Optional) Merge Multi‑Channel Data

Script: `preprocessing/act_obs_mix.py`

- Combines observation and activation (e.g., 2‑channel inputs) if needed.

Edit paths and run:

```bash
python preprocessing/act_obs_mix.py
```

#### 4.5 Generate Symmetry Fold Labels

Script: `preprocessing/nfold_generate.py`

- Generates an `nfold` label for each kernel (e.g., 1, 2, 3, 4, 6‑fold symmetry).
- Output is a `.npy` array used in stage‑1 and stage‑2 training.

Run:

```bash
python preprocessing/nfold_generate.py
```

This should output a file like:

```text
/data/coding/kernel/nfold.npy   # example path; adjust to your setup
```

Make sure the path to this file matches `fold_file` arguments in `stage1.py` and `stage2.py`.

---

### 5. Train/Test Split for Kernels

Script: `training/train_test_file_copy.py`

- Copies kernel and observation data into train/test folders according to your split strategy.

Typical usage:

1. Set:
   - Source directories for kernels / observations
   - Destination directories like `/data/coding/kernel/train/` and `/data/coding/kernel/test/`
2. Run:

```bash
python training/train_test_file_copy.py
```

After this step you should have something like:

```text
/data/coding/kernel/train/
  Kernel number1nom_2d.npy
  Kernel number2nom_2d.npy
  ...
/data/coding/kernel/test/
  Kernel numberXnom_2d.npy
  ...
```

and similar structures for observations if the script is configured that way.

---

### 6. Stage‑1: Symmetry‑Aware VAE on Kernels

Stage‑1 trains a VAE (`diffusers.AutoencoderKL`) on kernels with an additional **symmetry loss** that enforces rotational invariance based on the kernel’s `nfold` value.

Main script: `training/stage1.py`

#### 6.1 Configure Paths and Hyperparameters

Inside `stage1.py`:

- `fold_file`: path to the `nfold` label array (e.g. `/data/coding/kernel/nfold.npy`)
- `train_data_dir`: directory containing training kernel `.npy` files (e.g. `/data/coding/kernel/train/`)
- `test_data_dir`: directory containing test kernel `.npy` files
- `model_save_path`: directory to save trained VAE weights
- `loss_dir`: directory to save loss logs

Hyperparameters in `run(...)`:

- `alpha`: weight for symmetry loss term
- `beta`: weight for KL loss term
- `num_epochs`, `batch_size`, `initial_lr`, `step_size`, `gamma`

The critical training function signature is:

```python
def run(
    vae,
    fold_file,
    train_data_dir,
    test_data_dir,
    model_save_path,
    loss_dir,
    alpha=0.5,
    beta=0.001,
    num_epochs=10,
    batch_size=8,
    initial_lr=1e-4,
    step_size=50,
    gamma=0.5,
):
    ...
```

You need to construct a `vae = AutoencoderKL.from_pretrained(...)` or a custom instance and then call `run`.

#### 6.2 Example Training Command

After editing `stage1.py` so that the `if __name__ == "__main__":` section (or your own main block) properly initializes the model and paths, run:

```bash
python training/stage1.py
```

This will:

- Train the stage‑1 model
- Save model checkpoints in `model_save_path`
- Log train/test losses into text files under `loss_dir`

#### 6.3 Stage‑1 Reconstruction Visualization

Script: `training/stage1_rec_visual.py`

- Loads a trained stage‑1 model
- Runs reconstruction on kernels and visualizes/saves results

Configure:

- Model checkpoint path
- Input kernel directory
- Output image directory

Run:

```bash
python training/stage1_rec_visual.py
```

---

### 7. Latent Generation for Stage‑2

Script: `training/latents_generate.py`

- Uses the trained stage‑1 VAE to extract latent vectors for each kernel.
- Outputs a latent tensor array such as shape `(100, 2, 4, 8, 8)` saved to a `.npy` file.

You should configure:

- Path to trained stage‑1 model
- Kernel data directory for which to generate latents
- Output `latents_file` path (e.g. `/data/coding/kernel/kernel_latents.npy`)

Run:

```bash
python training/latents_generate.py
```

This latent file is required by `stage2.py` as `label_file` / `latents_file`.

---

### 8. Stage‑2: Encoder Training from Observations to Latent Space

Stage‑2 trains an encoder on observation data (or mixed observation+activation) to match the latents of stage‑1.

Main script: `training/stage2.py`

#### 8.1 Dataset and Inputs

`CustomNpyDataset` reads:

- `root_dir`: root directory of observation `.npy` files, organized per kernel (e.g. `/data/coding/obs/train/Kernel number1/…`)
- `label_file`: stage‑1 latent `.npy` file (e.g. `/data/coding/kernel/kernel_latents.npy`)
- `fold_file`: `nfold` label file
- `len_train`: number of samples per kernel in the training set
- `len_test`: number of samples per kernel in the test set (computed as ~0.25 × `len_train`)

The `run(...)` function:

```python
def run(
    fold_file,
    train_data_length,
    vae,
    train_data_dir,
    test_data_dir,
    latents_file,
    model_save_path,
    loss_dir,
    num_epochs=10,
    batch_size=8,
    initial_lr=1e-4,
    step_size=50,
    gamma=0.5,
):
    ...
```

Core behavior:

- Loads observation data from `train_data_dir` / `test_data_dir`
- Loads latent labels from `latents_file`
- Trains the encoder so that its latents `(mean, logvar)` match the precomputed stage‑1 latents

#### 8.2 Configure Paths and Parameters

Edit `stage2.py` to set:

- `fold_file`: path to `nfold` labels
- `train_data_dir`: path to observation `.npy` train data
- `test_data_dir`: path to observation `.npy` test data
- `latents_file`: stage‑1 kernel latent `.npy` file
- `model_save_path`: where to save stage‑2 encoder weights
- `loss_dir`: where to save train/test losses
- `train_data_length`: number of observations per kernel to use (e.g. `50`, `100`)

Initialize a VAE or encoder compatible with the latent shape (`2*4` channels, `8 × 8` spatial), and call `run(...)`.

#### 8.3 Run Stage‑2 Training

After paths and model are set:

```bash
python training/stage2.py
```

This will train the encoder and log losses into text files in `loss_dir`.

#### 8.4 Stage‑2 Reconstruction Visualization

Script: `training/stage2_rec_visual.py`

- Uses the trained stage‑2 encoder to reconstruct images from observations via the stage‑1 decoder pipeline.
- Configure input observation directories, encoder/decoder checkpoints, and output paths, then run:

```bash
python training/stage2_rec_visual.py
```

---

### 9. 1‑Step Baseline

Script: `training/stage2-1step.py`

- Trains a 1‑step baseline model directly from observation to reconstruction without a two‑stage latent alignment.

Usage is similar to `stage2.py`: configure paths and hyperparameters, then:

```bash
python training/stage2-1step.py
```

---

### 10. Visualization and Analysis

All in `plotting/`:

- `visual_kernel.py`: visualize kernel `.npy` images
- `visual_obs.py`: visualize observation `.npy` or `.csv` images
- `tsne.py`: t‑SNE plots of latent vectors from stage‑1 or stage‑2
- `denoise.py`: apply and visualize denoising on observations

For each script:

1. Set input directories and output figure paths.
2. Run, for example:

```bash
python plotting/visual_kernel.py
python plotting/visual_obs.py
python plotting/tsne.py
python plotting/denoise.py
```

---

### 11. Notes and Tips

- Many scripts currently use **absolute Windows paths** (`D:/...`). On Linux, modify them to your own absolute paths such as `/data/...`.
- Make sure the **kernel indexing** is consistent:
  - Filenames like `Kernel numberXnom_2d.npy` are expected by `stage1.py` and `stage2.py`.
  - `nfold` label arrays and latent files are indexed by kernel number (1–100).
- When changing directory structures, keep the **per-kernel folder names** (`Kernel numberX`) so that `stage2.py` can correctly align observations and latent labels.

