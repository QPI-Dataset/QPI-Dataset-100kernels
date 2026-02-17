import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

"""t-SNE visualization of kernel latent vectors with embedded kernel images."""

# Paths to latent vectors and kernel images
# latents_file = "./data/kernel_latents/80kernel_latent_vectors.npy"
latents_file = "./data/kernel_latents/100kernel_latent_vectors.npy"
kernels_file = "./data/kernel_latents/100kernels.npy"
nfold_dir = "./data/kernel_latents/n_fold_noise.npy"
ntype_dir = "./data/kernel_latents/6_type.npy"

# Load data
kernel_latents = np.load(latents_file)
kernels = np.load(kernels_file)  # shape: (N, 1, H, W) or similar

# Extract mean vectors and flatten
mean = kernel_latents[:, 0]
x = mean.reshape(mean.shape[0], -1)

# t-SNE projection to 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(x)

# Normalize coordinates into [padding, 1 - padding] to keep a margin
padding = 0.05
X_min = X_tsne.min(axis=0)
X_max = X_tsne.max(axis=0)
X_range = X_max - X_min

X_norm = (X_tsne - X_min) / X_range
X_norm = X_norm * (1 - 2 * padding) + padding

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(X_norm)):
    image = kernels[i][0]

    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    colored_image = cm.coolwarm(image)

    # Drop alpha channel
    colored_image = (colored_image[..., :3] * 255).astype(np.uint8)

    im = OffsetImage(colored_image, zoom=0.5)
    ab = AnnotationBbox(im, X_norm[i], frameon=False)
    ax.add_artist(ab)

ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
# plt.savefig("./data/figures/tsne_visualization_new.png", dpi=100, bbox_inches='tight')

plt.show()
