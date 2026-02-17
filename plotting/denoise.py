import os
import json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from tqdm import tqdm

"""Apply and visualize multiple denoising algorithms on observations."""

import matplotlib.cm as cm


def to_colormap(im, cmap_name="viridis"):
    """Convert a grayscale image to an RGB image using a Matplotlib colormap."""
    arr = np.array(im)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(arr / 255.0) * 255).astype(np.uint8)
    rgb = Image.fromarray(rgba[:, :, :3], mode="RGB")
    return rgb

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_image(path: str) -> Image.Image:
    """Load a .npy or image file as a grayscale PIL Image."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)  # (64, 64)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.open(path).convert("L")
    return img

def denoise_gaussian(img, s):
    """Gaussian blur denoising with strength coefficient s in [0, 1]."""
    radius = 0.5 + 2.5 * s
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": round(radius, 2)}


def denoise_median(img, s):
    """Median filter denoising with strength coefficient s in [0, 1]."""
    k = int(3 + round(s * 6))  # 3~9
    if k % 2 == 0:
        k += 1
    return img.filter(ImageFilter.MedianFilter(size=k)), {"kernel_size": k}


def denoise_box(img, s):
    """Box blur denoising with strength coefficient s in [0, 1]."""
    radius = 0.5 + 3.5 * s
    return img.filter(ImageFilter.BoxBlur(radius=radius)), {"radius": round(radius, 2)}


algorithms = {
    "Gaussian": denoise_gaussian,
    "Median":   denoise_median,
    "Box":      denoise_box,
}

def make_grid(tiles, strengths_labels, out_path, tile_size=128):
    """Compose a grid image with all algorithms and strengths (not used in final version)."""
    algos_order = []
    for algo, _, _ in tiles:
        if algo not in algos_order:
            algos_order.append(algo)

    cols = len(strengths_labels)
    rows = len(algos_order)

    grid_w = cols * tile_size
    grid_h = rows * (tile_size + 28)  # Reserve space for title
    canvas = Image.new("RGB", (grid_w, grid_h), 255)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for r, algo in enumerate(algos_order):
        for c, strength in enumerate(strengths_labels):
            for a, s, im in tiles:
                if a == algo and s == strength:
                    # imr = im.resize((tile_size, tile_size))
                    imr = to_colormap(im, "plasma").resize((tile_size, tile_size))

                    x = c * tile_size
                    y = r * (tile_size + 28)

                    draw.rectangle([x, y, x + tile_size, y + 28], fill=230)
                    draw.text((x + 4, y + 6), f"{algo} | {s}", fill=0, font=font)
                    canvas.paste(imr, (x, y + 28))
    canvas.save(out_path)

import matplotlib.pyplot as plt

def make_grid_matplotlib(img, tiles, out_path):
    """Make a comparison grid (original vs light/heavy denoise) using Matplotlib."""
    algos = list(tiles.keys())
    cols = 3  # Original + Light + Heavy
    rows = len(algos)

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

    if rows == 1:
        axes = [axes]

    for r, algo in enumerate(algos):
        # Original image
        axes[r][0].imshow(np.array(img), cmap="coolwarm")
        axes[r][0].set_title("Original", fontsize=10)
        axes[r][0].axis("off")

        # Light denoising
        axes[r][1].imshow(np.array(tiles[algo]["light"]), cmap="coolwarm")
        axes[r][1].set_title(f"{algo} | Strength coefficient=0.3", fontsize=10)
        axes[r][1].axis("off")

        # Heavy denoising
        axes[r][2].imshow(np.array(tiles[algo]["heavy"]), cmap="coolwarm")
        axes[r][2].set_title(f"{algo} | Strength coefficient=0.7", fontsize=10)
        axes[r][2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(file_title, input_file, out_dir):
    ensure_dir(out_dir)
    img = load_image(input_file)

    params_log = {}
    tiles_dict = {}

    for algo_name, func in algorithms.items():
        params_log[algo_name] = {}
        tiles_dict[algo_name] = {}

        # Light denoising
        out_img_light, params_light = func(img, strengths[0])
        # out_img_light.save(os.path.join(out_dir, f"{algo_name}_light.png"))
        params_log[algo_name][f"light"] = params_light
        tiles_dict[algo_name][f"light"] = out_img_light

        # Heavy denoising
        out_img_heavy, params_heavy = func(img, strengths[1])
        # out_img_heavy.save(os.path.join(out_dir, f"{algo_name}_heavy.png"))
        params_log[algo_name][f"heavy"] = params_heavy
        tiles_dict[algo_name][f"heavy"] = out_img_heavy

    grid_path = os.path.join(out_dir, f"{file_title} denoise_grid.png")
    make_grid_matplotlib(img, tiles_dict, grid_path)

root_dir = './data/observation/nom'
out_root_path = "./data/figures/denoise"  # output directory
strengths = [0.3, 0.5, 0.7]             # denoising strengths


if __name__ == "__main__":
    for sub_folder in tqdm(os.listdir(root_dir)):
        sub_folder_path = os.path.join(root_dir, sub_folder)
        out_path = os.path.join(out_root_path, sub_folder)

        for file in os.listdir(sub_folder_path):
            if file.endswith('.npy'):
                file_name = file[:-16]

                input_file_path = os.path.join(sub_folder_path, file)
                try:
                    main(file_name, input_file_path, out_path)
                except Exception:
                    # Skip files that fail to load or process
                    pass
