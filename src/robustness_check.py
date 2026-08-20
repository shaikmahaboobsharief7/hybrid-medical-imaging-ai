import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import measure, morphology
import matplotlib.pyplot as plt

from src.unet_architecture import UNet
from src.unet_training_and_evaluation import get_device

CLEAN_IMAGES = Path("data/processed/test/images")
CORRUPTED_IMAGES = Path("data/raw/nuclei_dataset/test_corrupted/images")
WEIGHTS_PATH = Path("outputs/task3_unet_segmentation/unet_weights.pth")
OUT_DIR = Path("outputs/task4_hybrid_pipeline/robustness")

PAIRS = ["test_000", "test_004"]
CORRUPTIONS = ["blur", "lowcontrast"]


def load_model(device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model


def predict_mask(model, device, img_path: Path):
    img = Image.open(img_path).convert("L").resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    mask = (pred.squeeze().cpu().numpy() > 0.5)
    mask = morphology.remove_small_objects(mask, min_size=15)
    return img, mask


def count_objects(mask):
    labels = measure.label(mask)
    return labels.max()


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = load_model(device)

    rows = []
    fig, axes = plt.subplots(len(PAIRS), 1 + len(CORRUPTIONS), figsize=(12, 4 * len(PAIRS)))

    for row_idx, image_id in enumerate(PAIRS):
        clean_path = CLEAN_IMAGES / f"{image_id}.png"
        clean_img, clean_mask = predict_mask(model, device, clean_path)
        clean_count = count_objects(clean_mask)

        axes[row_idx, 0].imshow(clean_mask, cmap="gray")
        axes[row_idx, 0].set_title(f"{image_id} clean\n(n={clean_count}, area={clean_mask.sum()}px)")
        axes[row_idx, 0].axis("off")

        rows.append({"image_id": image_id, "condition": "clean", "n_objects": clean_count})

        for col_idx, corruption in enumerate(CORRUPTIONS, start=1):
            corrupt_path = CORRUPTED_IMAGES / f"{image_id}_{corruption}.png"
            if not corrupt_path.exists():
                continue

            corrupt_img, corrupt_mask = predict_mask(model, device, corrupt_path)
            corrupt_count = count_objects(corrupt_mask)

            axes[row_idx, col_idx].imshow(corrupt_mask, cmap="gray")
            axes[row_idx, col_idx].set_title(f"{image_id} {corruption}\n(n={corrupt_count}, area={corrupt_mask.sum()}px)")
            axes[row_idx, col_idx].axis("off")

            rows.append({"image_id": image_id, "condition": corruption, "n_objects": corrupt_count})

    plt.tight_layout()
    plt.savefig(OUT_DIR / "corruption_mask_comparison.png", dpi=150)
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "corruption_counts.csv", index=False)
    print(df)
    print(f"\nsaved comparison figure and csv to {OUT_DIR}")


if __name__ == "__main__":
    run()