import os
import numpy as np
from tqdm import tqdm

SRC_DIR = 'D:/桌面/STIR/64X64 kernels/data_noise_free/observation/data_npy'   # Source folder
DST_DIR = 'D:/桌面/STIR/64X64 kernels/data_noise_free/observation_nom' # Output folder for normalized results
ROWS, COLS = 64, 64

os.makedirs(DST_DIR, exist_ok=True)

def min_max_normalize(arr):
    arr = arr.astype(np.float64)
    min_, max_ = arr.min(), arr.max()
    if max_ == min_:
        return np.zeros_like(arr)
    return (arr - min_) / (max_ - min_)

def process_one_file(folder, name):
    src_path = os.path.join(os.path.join(SRC_DIR,folder),name)
    os.makedirs(os.path.join(DST_DIR,folder), exist_ok=True)

    try:
        data = np.load(src_path)
    except Exception as e:
        print(f'[ERROR] Failed to read {os.path.basename(src_path)}: {e}')
        return

    if data.shape != (ROWS, COLS):
        print(f'[SKIP] Size mismatch {os.path.basename(src_path)}: {data.shape} ≠ ({ROWS},{COLS})')
        return

    norm_arr = min_max_normalize(data)

    base_name = os.path.basename(src_path)
    name_only, _ = os.path.splitext(base_name)
    dst_name = name_only + '_nom.npy'
    dst_path = os.path.join(os.path.join(DST_DIR,folder), dst_name)

    np.save(dst_path, norm_arr)

def main():
    if not os.path.isdir(SRC_DIR):
        print('Source folder not found')
        return

    for folder in tqdm(os.listdir(SRC_DIR)):
        for file_name in os.listdir(os.path.join(SRC_DIR,folder)):
            if file_name.lower().endswith('.npy'):
                process_one_file(folder, file_name)


if __name__ == '__main__':
    main()