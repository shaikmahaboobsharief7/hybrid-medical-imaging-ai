import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from skimage import transform, measure, morphology
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from src.unet_architecture import UNet
from src.unet_training_and_evaluation import get_device

TEST_IMAGES = Path("data/processed/test/images")
UNET_WEIGHTS = Path("outputs/task3_unet_segmentation/unet_weights.pth")
MEDSAM_CKPT = Path("models/medsam_vit_b.pth")
OUT_DIR = Path("outputs/task3_unet_segmentation/foundation_model_comparison")

SAMPLE_IMAGES = ["test_000", "test_002", "test_004"]


def load_unet(device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(UNET_WEIGHTS, map_location=device))
    model.eval()
    return model


def unet_predict(model, device, img_path: Path):
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
    mask = (pred.squeeze().cpu().numpy() > 0.5)
    mask = morphology.remove_small_objects(mask, min_size=15)
    return img, mask


def load_medsam(device):
    sam = sam_model_registry["vit_b"](checkpoint=None)
    state_dict = torch.load(str(MEDSAM_CKPT), map_location="cpu")
    sam.load_state_dict(state_dict)
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def medsam_predict(predictor, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    predictor.set_image(img_np)

    # medsam needs a bounding box prompt, we give it the whole image
    # since it has no automatic "find everything" mode like u-net does
    h, w = img_np.shape[:2]
    box = np.array([0, 0, w, h])

    mask, score, _ = predictor.predict(box=box, multimask_output=False)
    return img_np, mask[0]


def count_objects(mask):
    labels = measure.label(mask)
    return labels.max()


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    unet = load_unet(device)
    medsam_predictor = load_medsam(device)

    rows = []
    fig, axes = plt.subplots(len(SAMPLE_IMAGES), 3, figsize=(10, 3.3 * len(SAMPLE_IMAGES)))

    for row_idx, image_id in enumerate(SAMPLE_IMAGES):
        img_path = TEST_IMAGES / f"{image_id}.png"

        raw_img, unet_mask = unet_predict(unet, device, img_path)
        unet_count = count_objects(unet_mask)

        _, medsam_mask = medsam_predict(medsam_predictor, img_path)
        medsam_coverage = 100 * medsam_mask.sum() / medsam_mask.size

        axes[row_idx, 0].imshow(raw_img, cmap="gray")
        axes[row_idx, 0].set_title(f"{image_id} input")

        axes[row_idx, 1].imshow(unet_mask, cmap="gray")
        axes[row_idx, 1].set_title(f"U-Net (n={unet_count})")

        axes[row_idx, 2].imshow(medsam_mask, cmap="gray")
        axes[row_idx, 2].set_title(f"MedSAM ({medsam_coverage:.1f}% coverage)")
        axes[row_idx, 2].add_patch(plt.Rectangle(
            (0, 0), medsam_mask.shape[1] - 1, medsam_mask.shape[0] - 1,
            fill=False, edgecolor="gray", linewidth=1.5
        ))

        for ax in axes[row_idx]:
            ax.axis("off")

        rows.append({
            "image_id": image_id,
            "unet_n_objects": unet_count,
            "medsam_coverage_percent": round(medsam_coverage, 1),
        })

    plt.tight_layout()
    plt.savefig(OUT_DIR / "unet_vs_medsam.png", dpi=150)
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "unet_vs_medsam_counts.csv", index=False)
    print(df)
    print(f"saved comparison to {OUT_DIR}")


if __name__ == "__main__":
    run()