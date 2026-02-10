import numpy as np

"""
Manually define n-fold rotational symmetry labels for each of the 100 kernels.
The mapping dictionary `a` uses:
  key   = n-fold symmetry (1, 2, 3, 4, 6)
  value = list of kernel indices (1-based) that belong to this symmetry class.
The result is saved as a (100,) integer array, where index i corresponds to
kernel number (i + 1).
Update the output path to your desired location.
"""

symmetry_map = {
    1: [1, 2, 4, 5, 7, 8, 82, 83, 86, 85, 89, 88],
    2: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27,
        91, 92, 93, 94, 95, 96, 97, 98, 99],
    3: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 100],
    4: [3, 6, 9, 28, 29, 30, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 84, 87, 90],
    6: [46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63],
}

nfold_labels = [0 for _ in range(100)]
for nfold, kernel_indices in symmetry_map.items():
    for idx in kernel_indices:
        nfold_labels[idx - 1] = nfold

nfold_labels = np.array(nfold_labels, dtype=int)

output_path = "D:/桌面/STIR/64X64 kernels stage2/kernel_latents/n_fold_noise.npy"
np.save(output_path, nfold_labels)

print('n-fold labels generated and saved.')