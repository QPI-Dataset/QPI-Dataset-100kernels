import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""Stage-1 model training: Symmetry-aware VAE on kernel images"""

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, nfold_file):
        self.data_dir = data_dir
        self.folds = []
        self.fold_array = np.load(nfold_file)
        self.file_list = []
        # self.file_list = [os.path.join(data_dir, file) for file in os.listdir(data_dir)
        #                   if os.path.isfile(os.path.join(data_dir, file))]
        for kernel_file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, kernel_file)
            if os.path.isfile(file_path):
                kernel_name = kernel_file[:-10]
                kernel_number = int(kernel_name.split('number')[-1])
                self.folds.append(self.fold_array[kernel_number-1])
                self.file_list.append(file_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        image = np.load(file_path)
        nfold = self.folds[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Convert to single-channel tensor
        return nfold, image


def get_rotation_matrix(batch_size, angle_deg, center, device):
    theta = torch.tensor(angle_deg * 3.1415926 / 180).to(device)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # Build affine matrix: [batch_size, 2, 3]
    affine_matrix = torch.zeros(batch_size, 2, 3).to(device)
    affine_matrix[:, 0, 0] = cos
    affine_matrix[:, 0, 1] = -sin
    affine_matrix[:, 1, 0] = sin
    affine_matrix[:, 1, 1] = cos
    # Rotation center is at image center
    # tx = (1 - cos) * center[0] + sin * center[1]
    # ty = -sin * center[0] + (1 - cos) * center[1]
    affine_matrix[:, 0, 2] = 0
    affine_matrix[:, 1, 2] = 0
    return affine_matrix


def cal_symmetry_loss(batch_n_fold, batch_kernel, batch_rec, middle_size=40):
    device = batch_kernel.device
    B, C, H, W = batch_kernel.shape
    loss = 0
    start_x = (W - middle_size) // 2
    start_y = (H - middle_size) // 2
    end_x = start_x + middle_size
    end_y = start_y + middle_size

    for n_fold in [1, 2, 3, 4, 6]:
        idxs = (batch_n_fold == n_fold).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        rec_sub = batch_rec[idxs]
        kernel_sub = batch_kernel[idxs]
        kernel_middle = kernel_sub[:, :, start_y:end_y, start_x:end_x]
        angles = {1: [90,180], 2: [180], 3: [120], 4: [90], 6: [60,120,180]}[n_fold]
        affine_matrixs = []
        subloss = 0
        for angle in angles:
            # Generate rotation affine matrix
            if angle == 180:
                affine_matrixs.append(get_rotation_matrix(len(idxs), angle_deg=angle, center=(W / 2, H / 2), device=device))
            else:
                affine_matrixs.append(get_rotation_matrix(len(idxs), angle_deg=angle, center=(W / 2, H / 2), device=device))
                affine_matrixs.append(get_rotation_matrix(len(idxs), angle_deg=-angle, center=(W / 2, H / 2), device=device))

        for affine_matrix in affine_matrixs:
            # Generate sampling grid
            grid = F.affine_grid(affine_matrix, rec_sub.size(), align_corners=False)
            rotated = F.grid_sample(rec_sub, grid, mode='bilinear', padding_mode='border', align_corners=False)
            rotated_middle = rotated[:, :, start_y:end_y, start_x:end_x]
            subloss += F.mse_loss(rotated_middle, kernel_middle)
        loss += subloss/len(affine_matrixs)

    return loss

# Training and testing function
def run(vae, fold_file, train_data_dir, test_data_dir, model_save_path, loss_dir,alpha,beta,
        num_epochs, batch_size, initial_lr, step_size, gamma):

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    # Data loaders
    train_dataset = CustomDataset(train_data_dir,fold_file)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomDataset(test_data_dir,fold_file)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Training and testing loop
    train_loss_list = []
    test_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        # Training phase
        vae.train()
        epoch_train_loss = 0

        for batch_idx, (fold_num, batch) in enumerate(train_dataloader):
            fold_num = fold_num.to(device)
            batch = batch.to(device)

            optimizer.zero_grad()

            # Encode and decode
            latents = vae.encode(batch).latent_dist
            reconstructed = vae.decode(latents.sample()).sample

            # Calculate reconstruction loss
            rec_loss = F.mse_loss(batch, reconstructed)
            symmetry_loss = cal_symmetry_loss(fold_num,batch,reconstructed).to(device)
            loss = rec_loss + symmetry_loss * alpha

            mu = latents.mean
            logvar = latents.logvar
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = loss + kl_loss * beta
            # Backward propagation and optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)
        train_loss_list.append(epoch_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}")

        # Testing phase
        vae.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_idx, (fold_num, batch) in enumerate(test_dataloader):
                fold_num = fold_num.to(device)
                batch = batch.to(device)

                # Encode and decode
                latents = vae.encode(batch).latent_dist
                reconstructed = vae.decode(latents.sample()).sample

                # Calculate reconstruction loss
                rec_loss = F.mse_loss(batch, reconstructed)
                symmetry_loss = cal_symmetry_loss(fold_num, batch, reconstructed).to(device)
                loss = rec_loss + symmetry_loss * alpha

                mu = latents.mean
                logvar = latents.logvar
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = loss + kl_loss * beta
                epoch_test_loss += loss.item()

        epoch_test_loss /= len(test_dataloader)
        test_loss_list.append(epoch_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {epoch_test_loss:.6f}")

        scheduler.step()
        # Save training and testing losses
        with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_train_loss}||     ")
        with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_test_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_test_loss}||      ")
        if (epoch+1) % 5 == 0:
            with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_train_loss.txt'), 'a') as f:
                f.write(f"\n")
            with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_test_loss.txt'), 'a') as f:
                f.write(f"\n")

    with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_train_loss.txt'), 'a') as f:
        f.write(f"-----------------------kl_time1_beta{beta}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")
    with open(os.path.join(loss_dir, f'train1_kl_beta{beta}_alpha{alpha}_test_loss.txt'), 'a') as f:
        f.write(f"-----------------------kl_time1_beta{beta}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")

    # Save model
    vae.save_pretrained(model_save_path)

    loss_path = os.path.join(loss_dir, f"train_kl_time1_beta{beta}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    train_loss_list_cpu = torch.Tensor(train_loss_list).cpu().numpy()
    np.save(loss_path, train_loss_list_cpu)
    loss_path1 = os.path.join(loss_dir, f"test_kl_time1_beta{beta}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    test_loss_list_cpu = torch.Tensor(test_loss_list).cpu().numpy()
    np.save(loss_path1, test_loss_list_cpu)

    save_pth = os.path.join(loss_dir, f"train1_time1_a{alpha}.png")
    x = np.arange(len(train_loss_list_cpu))
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(x, train_loss_list_cpu, label='Train loss', color='blue', marker='o')  # Plot first array
    plt.plot(x, test_loss_list_cpu, label='Test loss', color='red', marker='x')  # Plot second array
    plt.legend()
    plt.title(f"loss visualization - train1_time1_a{alpha}")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(save_pth)
    print('Training completed')


# Main program
if __name__ == "__main__":

    alpha = 0.7
    beta = 1e-6

    epoch = 300
    batch_size = 8
    initial_lr = 1e-4
    step_size = 10
    gamma = 0.97
    # Data paths
    train_data_dir = '/data/coding/kernel/train'
    test_data_dir = '/data/coding/kernel/test'

    model_ori_path = "/data/coding/vae_test2"  # madebyollin/sdxl-vae-fp16-fix
    model_path = f'/data/coding/stage1/kl_beta{beta}_alpha{alpha}_g{gamma}/vae_kl_time1'
    model_save_path = f'/data/coding/stage1/kl_beta{beta}_alpha{alpha}_g{gamma}/vae_kl_time1'

    loss_dir = f'/data/coding/stage1/kl_beta{beta}_alpha{alpha}_g{gamma}/loss/'
    nfold_dir = "/data/coding/kernel_latents/n_fold_noise.npy"

    url = "D:/桌面/domain_adaptation/STIR/vae_non_pretrained"#/vae-ft-kl-840000-ema-pruned.safetensors"
    # print(diffusers.__version__)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    # Load pretrained VAE model
    try:
        # vae = AutoencoderKL.from_single_file(url, in_channels=1, out_channels=1,ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
        vae = AutoencoderKL.from_pretrained(model_ori_path, in_channels=1, out_channels=1,
                                            ignore_mismatched_sizes=True, low_cpu_mem_usage=False,from_tf=True)
        print("Model loaded successfully!")
        vae.config.in_channels = 1
        vae.config.out_channels = 1
        vae.config.sample_size = 64
        vae.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    run(vae, nfold_dir, train_data_dir, test_data_dir, model_save_path, loss_dir, alpha=alpha,beta=beta,
        num_epochs=epoch, batch_size=batch_size,initial_lr=initial_lr, step_size=step_size,gamma=gamma)