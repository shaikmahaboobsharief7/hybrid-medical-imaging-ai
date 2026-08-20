import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.unet_architecture import UNet

def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum()
    dice = (2 * intersection + eps) / (pred.sum() + target.sum() + eps)
    return 1 - dice


def get_loss_fn(loss_type: str):
    bce = nn.BCEWithLogitsLoss()

    if loss_type == "bce":
        return lambda pred, target: bce(pred, target)
    elif loss_type == "dice":
        return lambda pred, target: dice_loss(pred, target)
    elif loss_type == "bce_dice":
        return lambda pred, target: bce(pred, target) + dice_loss(pred, target)
    else:
        raise ValueError(f"unknown loss type: {loss_type}")

TRAIN_IMAGES = Path("data/processed/train/images")
TRAIN_MASKS = Path("data/processed/train/masks")
VAL_IMAGES = Path("data/processed/val/images")
VAL_MASKS = Path("data/processed/val/masks")
OUT_DIR = Path("outputs/task3_unet_segmentation")

EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 1e-3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NucleiDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path):
        self.image_paths = sorted(images_dir.glob("*.png"))
        self.masks_dir = masks_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def train(loss_type="bce"):
    device = get_device()
    print(f"training on {device} with {loss_type} loss")

    train_ds = NucleiDataset(TRAIN_IMAGES, TRAIN_MASKS)
    val_ds = NucleiDataset(VAL_IMAGES, VAL_MASKS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = get_loss_fn(loss_type)

    train_losses = []
    val_dice_scores = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        dice_total = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = torch.sigmoid(model(imgs))
                dice_total += dice_score(preds, masks).item()

        avg_dice = dice_total / len(val_loader)
        val_dice_scores.append(avg_dice)

        print(f"[{loss_type}] epoch {epoch + 1}/{EPOCHS}, loss: {avg_loss:.4f}, val dice: {avg_dice:.4f}")

    run_out_dir = OUT_DIR / f"loss_{loss_type}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_out_dir / "unet_weights.pth")

    return model, device, train_losses, val_dice_scores, val_ds, run_out_dir


def plot_curves(train_losses, val_dice_scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")

    axes[1].plot(val_dice_scores, color="green")
    axes[1].set_title("Validation Dice score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_dice_curves.png", dpi=150)
    plt.close()
    print("saved loss and dice curves")


def evaluate_final_metrics(model, device, val_ds):
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model.eval()

    dice_total, iou_total = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            dice_total += dice_score(preds, masks).item()
            iou_total += iou_score(preds, masks).item()

    mean_dice = dice_total / len(val_loader)
    mean_iou = iou_total / len(val_loader)
    print(f"final mean dice: {mean_dice:.4f}, mean iou: {mean_iou:.4f}")
    return mean_dice, mean_iou


def show_sample_predictions(model, device, val_ds, n_samples=3):
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))

    model.eval()
    with torch.no_grad():
        for i in range(n_samples):
            img, mask = val_ds[i]
            img_batch = img.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(img_batch))
            pred = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)

            axes[i, 0].imshow(img.squeeze(), cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 1].imshow(mask.squeeze(), cmap="gray")
            axes[i, 1].set_title("Ground truth")
            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("Prediction")

            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sample_predictions.png", dpi=150)
    plt.close()
    print("saved sample prediction panels")


def run():
    model, device, train_losses, val_dice_scores, val_ds = train()
    plot_curves(train_losses, val_dice_scores)
    evaluate_final_metrics(model, device, val_ds)
    show_sample_predictions(model, device, val_ds)


if __name__ == "__main__":
    run()