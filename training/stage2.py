import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
from tqdm import tqdm
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""Stage-2 training: encoder from observations to latent space"""

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define dataset class
class CustomNpyDataset(Dataset):
    def __init__(self, root_dir, label_file, fold_file, len_train, len_test, is_train=True, kernel_dir=None):
        """
        :param root_dir: Root directory of dataset
        :param label_file: Path to .npy file storing labels - i.e., latent vectors
        :param kernel_dir: Path to kernel train directory for filtering valid kernels
        """
        self.data_paths = []
        self.labels = []
        self.kernels = []
        self.is_train = is_train
        self.len_train = len_train
        self.len_test = len_test
        self.train_flag = 0
        self.test_flag = 0

        # Load label array
        self.label_array = np.load(label_file)  # Shape: (100, 2, 4, 8, 8)
        self.fold_array = np.load(fold_file)
        self.folds = []

        self.kernel_dir = kernel_dir

        # Traverse folders to get data paths and labels
        if self.is_train:
                train_test_dir = root_dir

                for folder_name in sorted(os.listdir(train_test_dir)):
                    original_file_dir = self.kernel_dir
                    folder_path = os.path.join(train_test_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Get current folder index
                        folder_index = int(folder_name.split('number')[-1])
                        # Get label corresponding to current folder
                        folder_label = self.label_array[folder_index - 1]
                        fold = self.fold_array[folder_index - 1]
                        original_file_path = os.path.join(original_file_dir, f"Kernel number{folder_index}nom_2d.npy")
                        if os.path.isfile(original_file_path):     # Ensure kernel number is in stage1 training set and satisfies specific fold
                            # self.kernels.append(folder_index)
                            # self.folds.append(fold)
                            flag_data_length = 0

                            for file_name in os.listdir(folder_path):
                                if file_name.endswith('.npy'):
                                    file_path = os.path.join(folder_path, file_name)
                                    self.data_paths.append(file_path)
                                    self.labels.append(folder_label)
                                    self.train_flag +=1
                                    flag_data_length += 1
                                    if flag_data_length >= self.len_train:
                                        break

        else:
            # for act_num in sorted(os.listdir(root_dir)):
            #     train_test_dir = os.path.join(root_dir, f"{act_num}/test")
            train_test_dir = root_dir

            for folder_name in sorted(os.listdir(train_test_dir)):
                    flag_data_length = 0
                    original_file_dir = self.kernel_dir
                    folder_path = os.path.join(train_test_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Get current folder index
                        folder_index = int(folder_name.split('number')[-1])
                        # Get label corresponding to current folder
                        folder_label = self.label_array[folder_index - 1]
                        fold = self.fold_array[folder_index - 1]
                        original_file_path = os.path.join(original_file_dir, f"Kernel number{folder_index}nom_2d.npy")
                        if os.path.isfile(original_file_path):     # Ensure kernel number is in stage1 training set and fold meets requirements
                            self.kernels.append(folder_index)

                            for file_name in os.listdir(folder_path):
                                if file_name.endswith('.npy'):
                                    file_path = os.path.join(folder_path, file_name)
                                    self.data_paths.append(file_path)
                                    self.labels.append(folder_label)
                                    self.test_flag +=1
                                    flag_data_length += 1
                                    if flag_data_length >= self.len_test:
                                        break
        # print(self.folds)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load data
        data_path = self.data_paths[idx]
        data = np.load(data_path)

        # Get label
        label = self.labels[idx]

        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return data, label


# Training and testing function
def run( fold_file, train_data_length, vae, train_data_dir, test_data_dir, latents_file, model_save_path, loss_dir,
        num_epochs=10, batch_size=8, initial_lr=1e-4, step_size=50, gamma=0.5, kernel_dir=None):
    # Create save directory
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    kernel_latents_file = latents_file
    # Data loaders
    test_data_length = math.floor(train_data_length * 0.25)
    train_dataset = CustomNpyDataset(train_data_dir, kernel_latents_file,fold_file, train_data_length, test_data_length,
                                     is_train=True, kernel_dir=kernel_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataset.train_flag)

    test_dataset = CustomNpyDataset(test_data_dir, kernel_latents_file,fold_file, train_data_length, test_data_length,
                                    is_train=False, kernel_dir=kernel_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(test_dataset.test_flag)

    print('data load finished')

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

        for batch_idx, (batch, label_latents) in enumerate(train_dataloader):
            batch = batch.to(device)

            # reshaped_tensor = label_latents.to(device)
            reshaped_tensor = label_latents.view(label_latents.shape[0], 2 * 4, 8, 8)  # Shape becomes (8, 8, 8, 8)
            mean_label, logvar_label = torch.chunk(reshaped_tensor, 2, dim=1)
            mean_label = mean_label.to(device)
            logvar_label = logvar_label.to(device)

            optimizer.zero_grad()

            # Encode
            latents = vae.encode(batch).latent_dist
            # rec = vae.decode(latents).sample
            mean = latents.mean
            logvar = latents.logvar

            # Calculate reconstruction loss
            loss = torch.mean((mean - mean_label) ** 2) + torch.mean((logvar - logvar_label) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            # break

        epoch_train_loss /= len(train_dataloader)
        train_loss_list.append(epoch_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}")

        # Testing phase
        vae.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_idx, (batch, label_latents) in enumerate(test_dataloader):
                batch = batch.to(device)

                # reshaped_tensor = label_latents.to(device)
                reshaped_tensor = label_latents.view(label_latents.shape[0], 2 * 4, 8, 8)  # Shape becomes (8, 8, 8, 8)
                mean_label, logvar_label = torch.chunk(reshaped_tensor, 2, dim=1)
                mean_label = mean_label.to(device)
                logvar_label = logvar_label.to(device)

                # Encode
                latents = vae.encode(batch).latent_dist
                mean = latents.mean
                logvar = latents.logvar

                # Calculate reconstruction loss
                loss = torch.mean((mean - mean_label) ** 2) + torch.mean((logvar - logvar_label) ** 2)

                epoch_test_loss += loss.item()
                # break

        epoch_test_loss /= len(test_dataloader)
        test_loss_list.append(epoch_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {epoch_test_loss:.6f}")

        scheduler.step()
        # break
        # Save training and testing losses
        with open(os.path.join(loss_dir, f'noise_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_train_loss}||     ")
        with open(os.path.join(loss_dir, f'noise_test_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_test_loss}||      ")
        if (epoch + 1) % 5 == 0:
            with open(os.path.join(loss_dir, f'noise_train_loss.txt'), 'a') as f:
                f.write(f"\n")
            with open(os.path.join(loss_dir, f'noise_test_loss.txt'), 'a') as f:
                f.write(f"\n")

    with open(os.path.join(loss_dir, f'noise_train_loss.txt'), 'a') as f:
        f.write(f"-----------------------"
                f"time1_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")
    with open(os.path.join(loss_dir, f'noise_test_loss.txt'), 'a') as f:
        f.write(f"-----------------------"
                f"time1_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")

    vae.save_pretrained(model_save_path)

    loss_path = os.path.join(loss_dir, f"train_time1_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    train_loss_list_cpu = torch.Tensor(train_loss_list).cpu().numpy()
    np.save(loss_path, train_loss_list_cpu)
    loss_path1 = os.path.join(loss_dir, f"test_time1_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    test_loss_list_cpu = torch.Tensor(test_loss_list).cpu().numpy()
    np.save(loss_path1, test_loss_list_cpu)

    save_pth = os.path.join(loss_dir, f"train1_noise_time1.png")
    x = np.arange(len(train_loss_list_cpu))
    # Plot two arrays
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(x, train_loss_list_cpu, label='Train loss', color='blue', marker='o')  # Plot first array
    plt.plot(x, test_loss_list_cpu, label='Test loss', color='red', marker='x')  # Plot second array
    plt.legend()
    # Add title and axis labels
    plt.title(f"loss visualization - train1_noise_time1")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # Show grid
    plt.grid(True)
    # Display image
    plt.savefig(save_pth)

    print('-----------------Training completed------------------')



# Main program
if __name__ == "__main__":

    train_length = 250

    # Data paths
    train_data_dir = "./data/channel2_mixed/train"
    test_data_dir = "./data/channel2_mixed/test"

    latents_file = "./data/kernel_latents/100kernel_latent_vectors.npy"

    model_ori_path = "./data/models/pretrained_vae"  # madebyollin/sdxl-vae-fp16-fix
    model_save_path = "./data/models/stage2"

    loss_dir = './data/loss/stage2/'
    nfold_dir = "./data/kernel_latents/n_fold_noise.npy"

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    # Load pretrained VAE model
    try:
        vae = AutoencoderKL.from_pretrained(model_ori_path, in_channels=2, out_channels=1,
                                            ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
        print("Model loaded successfully!")
        vae.config.in_channels = 2
        vae.config.out_channels = 1
        vae.config.sample_size = 64
        vae.to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    batch_size = 8
    len_epoch = 50
    step_size = 5
    initial_lr = 5e-5
    gamma = 0.97

    run(nfold_dir, train_length, vae, train_data_dir, test_data_dir, latents_file, model_save_path, loss_dir,
        num_epochs=len_epoch, batch_size=batch_size, initial_lr=initial_lr, step_size=step_size, gamma=gamma)
