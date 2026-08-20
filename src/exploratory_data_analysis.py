import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

PROCESSED_TRAIN = Path("data/processed/train/images")
METADATA_PATH = Path("data/raw/nuclei_dataset/metadata.csv")
OUT_DIR = Path("outputs/eda")


def sample_grid_by_density():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(METADATA_PATH)
    meta = meta[meta["split"] == "train"]

    # grab one example image per density regime so the grid actually shows variety
    densities = meta["density"].unique()
    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 4))

    for ax, density in zip(axes, densities):
        row = meta[meta["density"] == density].iloc[0]
        img_path = PROCESSED_TRAIN / f"{row['image_id']}.png"
        img = Image.open(img_path)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{density}\n(n={row['n_objects']})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sample_grid.png", dpi=150)
    plt.close()
    print("saved sample grid, one image per density regime")


def intensity_histogram():
    all_pixels = []
    for img_path in PROCESSED_TRAIN.glob("*.png"):
        arr = np.array(Image.open(img_path))
        all_pixels.append(arr.ravel())
    all_pixels = np.concatenate(all_pixels)

    plt.figure(figsize=(8, 5))
    plt.hist(all_pixels, bins=50, color="steelblue")
    plt.title("Pixel intensity distribution across training images")
    plt.xlabel("Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.savefig(OUT_DIR / "intensity_histogram.png", dpi=150)
    plt.close()
    print("saved intensity histogram")


def print_dataset_stats():
    meta = pd.read_csv(METADATA_PATH)
    print("images per split:")
    print(meta["split"].value_counts())
    print("\nobject count range:", meta["n_objects"].min(), "-", meta["n_objects"].max())
    print("density regimes:", meta["density"].unique().tolist())


def run_eda():
    print_dataset_stats()
    sample_grid_by_density()
    intensity_histogram()


if __name__ == "__main__":
    run_eda()