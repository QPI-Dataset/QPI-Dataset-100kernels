import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
import os
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

"""One-step baseline: train an encoder directly from 2-channel inputs to kernel images."""

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomNpyDataset(Dataset):
    """
    Dataset for 1-step baseline.

    root_dir: root directory containing per-kernel folders of 2-channel .npy files
    label_file: path to .npy file storing target kernel tensors (e.g., latent-like representation)
    fold_file: path to n-fold label file (kept for consistency with stage-2 API)
    len_train / len_test: number of samples per kernel for train / test
    """

    def __init__(self, root_dir, label_file, fold_file, len_train, len_test, is_train=True):
        self.data_paths = []
        self.labels = []
        self.kernels = []
        self.is_train = is_train
        self.len_train = len_train
        self.len_test = len_test
        self.train_flag = 0
        self.test_flag = 0

        # Load kernel label array, e.g. shape (100, 1, 64, 64) or similar
        self.label_array = np.load(label_file)
        self.fold_array = np.load(fold_file)
        self.folds = []

        # Traverse folders to gather data paths and labels
        if self.is_train:
            # for act_num in sorted(os.listdir(root_dir)):
            #     train_test_dir = os.path.join(root_dir, f"{act_num}/train")
            train_test_dir = root_dir

            for folder_name in sorted(os.listdir(train_test_dir)):
                    original_file_dir = "/data/coding/kernel/train/"
                    folder_path = os.path.join(train_test_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Get kernel index from folder name
                        folder_index = int(folder_name.split('number')[-1])
                        # Get corresponding kernel label tensor
                        folder_label = self.label_array[folder_index - 1]
                        fold = self.fold_array[folder_index - 1]
                        original_file_path = os.path.join(original_file_dir, f"Kernel number{folder_index}nom_2d.npy")
                        if os.path.isfile(original_file_path):  # ensure this kernel is in the stage-1 train set
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
                    original_file_dir = "/data/coding/kernel/train/"
                    folder_path = os.path.join(train_test_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Get kernel index from folder name
                        folder_index = int(folder_name.split('number')[-1])
                        # Get corresponding kernel label tensor
                        folder_label = self.label_array[folder_index - 1]
                        fold = self.fold_array[folder_index - 1]
                        original_file_path = os.path.join(original_file_dir, f"Kernel number{folder_index}nom_2d.npy")
                        if os.path.isfile(original_file_path):  # ensure kernel is included and fold is valid
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
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load input data
        data_path = self.data_paths[idx]
        data = np.load(data_path)

        # Get corresponding label
        label = self.labels[idx]

        # Convert to tensors
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)  # keep label shape

        return data, label


def run(
    fold_file,
    train_data_length,
    vae,
    train_data_dir,
    test_data_dir,
    latents_file,
    model_save_path,
    loss_dir,
    num_epochs,
    batch_size,
    initial_lr,
    step_size,
    gamma,
    beta,
):
    """
    Train and evaluate the 1-step baseline encoder.
    The VAE is trained to reconstruct the target kernel tensor directly from 2-channel inputs.
    """
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    kernel_latents_file = latents_file
    # Data loaders
    test_data_length = math.floor(train_data_length * 0.25)
    train_dataset = CustomNpyDataset(train_data_dir, kernel_latents_file,fold_file, train_data_length, test_data_length,
                                     is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataset.train_flag)

    test_dataset = CustomNpyDataset(test_data_dir, kernel_latents_file,fold_file, train_data_length, test_data_length,
                                    is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(test_dataset.test_flag)

    print('Data loading finished.')

    # Optimizer / scheduler
    optimizer = torch.optim.Adam(vae.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Training and evaluation loop
    train_loss_list = []
    test_loss_list = []
    for epoch in tqdm(range(num_epochs)):
        # Training phase
        vae.train()
        epoch_train_loss = 0

        for batch_idx, (batch, label_latents) in enumerate(train_dataloader):
            batch = batch.to(device)
            reshaped_tensor = label_latents.to(device)

            optimizer.zero_grad()

            # Encode and decode
            latents = vae.encode(batch).latent_dist
            rec = vae.decode(latents.sample()).sample
            loss = torch.mean((rec - reshaped_tensor) ** 2)

            mu = latents.mean
            logvar = latents.logvar
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_loss * beta

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)
        train_loss_list.append(epoch_train_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.6f}")

        # Evaluation phase
        vae.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_idx, (batch, label_latents) in enumerate(test_dataloader):
                batch = batch.to(device)
                reshaped_tensor = label_latents.to(device)

                # Encode and decode
                latents = vae.encode(batch).latent_dist
                rec = vae.decode(latents.sample()).sample
                loss = torch.mean((rec - reshaped_tensor) ** 2)

                mu = latents.mean
                logvar = latents.logvar
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss += kl_loss * beta

                epoch_test_loss += loss.item()

        epoch_test_loss /= len(test_dataloader)
        test_loss_list.append(epoch_test_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {epoch_test_loss:.6f}")

        scheduler.step()

        # Save training and test losses to text files
        with open(os.path.join(loss_dir, f'1step_train_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_train_loss}||     ")
        with open(os.path.join(loss_dir, f'1step_test_loss.txt'), 'a') as f:
            f.write(f"{epoch + 1}:  {epoch_test_loss}||      ")
        if (epoch + 1) % 5 == 0:
            with open(os.path.join(loss_dir, f'1step_train_loss.txt'), 'a') as f:
                f.write(f"\n")
            with open(os.path.join(loss_dir, f'1step_test_loss.txt'), 'a') as f:
                f.write(f"\n")

    with open(os.path.join(loss_dir, f'1step_train_loss.txt'), 'a') as f:
        f.write(f"-----------------------"
                f"t_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")
    with open(os.path.join(loss_dir, f'1step_test_loss.txt'), 'a') as f:
        f.write(f"-----------------------"
                f"t_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}---------------------------\n")

    vae.save_pretrained(model_save_path)

    loss_path = os.path.join(loss_dir, f"train_t_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    train_loss_list_cpu = torch.Tensor(train_loss_list).cpu().numpy()
    np.save(loss_path, train_loss_list_cpu)
    loss_path1 = os.path.join(loss_dir, f"test_t_len{train_data_length}_e{num_epochs}_i{initial_lr}_s{step_size}_g{gamma}")
    test_loss_list_cpu = torch.Tensor(test_loss_list).cpu().numpy()
    np.save(loss_path1, test_loss_list_cpu)

    save_pth = os.path.join(loss_dir, f"train1_1step_t.png")
    x = np.arange(len(train_loss_list_cpu))

    plt.figure(figsize=(10, 6))
    plt.plot(x, train_loss_list_cpu, label='Train loss', color='blue', marker='o')
    plt.plot(x, test_loss_list_cpu, label='Test loss', color='red', marker='x')
    plt.legend()
    plt.title(f"loss visualization - train1_1step_t")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(save_pth)

    print('----------------- Training finished ------------------')


if __name__ == "__main__":

    train_length = 250

    # Data paths
    train_data_dir = f"/data/coding/channel2_mixed_data_train&test/train"
    test_data_dir = f"/data/coding/channel2_mixed_data_train&test/test"

    latents_file = "/data/coding/kernel_latents/train1/100kernels.npy"

    model_ori_path = "/data/coding/vae_test2"  # madebyollin/sdxl-vae-fp16-fix
    model_path = f"/data/coding/1step_train1/time1"
    model_save_path = f"/data/coding/1step_train1/t"

    loss_dir = '/data/coding/1step_train1/loss/'
    nfold_dir = "/data/coding/kernel_latents/n_fold_noise.npy"

    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    # Load pre-trained VAE model as initialization
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
    beta = 1e-6

    run(nfold_dir, train_length, vae, train_data_dir, test_data_dir, latents_file, model_save_path, loss_dir,
        num_epochs=len_epoch, batch_size=batch_size, initial_lr=initial_lr, step_size=step_size, gamma=gamma,beta=beta)
