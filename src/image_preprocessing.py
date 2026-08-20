from pathlib import Path
from PIL import Image
import shutil

RAW_ROOT = Path("data/raw/nuclei_dataset")
PROCESSED_ROOT = Path("data/processed")
TARGET_SIZE = (256, 256)

SPLITS = ["train", "val", "test", "test_corrupted"]


def preprocess_image(img_path: Path, target_size=TARGET_SIZE) -> Image.Image:
    img = Image.open(img_path).convert("L")
    if img.size != target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return img


def process_split(split: str):
    split_raw = RAW_ROOT / split
    split_out = PROCESSED_ROOT / split

    if not split_raw.exists():
        print(f"skipping {split}, folder not found")
        return 0

    images_raw = split_raw / "images"
    images_out = split_out / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_raw.glob("*.png")):
        processed = preprocess_image(img_path)
        processed.save(images_out / img_path.name)
        count += 1
    for subfolder in ["masks", "labels"]:
        src_dir = split_raw / subfolder
        if src_dir.exists():
            dst_dir = split_out / subfolder
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.glob("*.png"):
                shutil.copy2(f, dst_dir / f.name)

    print(f"{split}: processed {count} images")
    return count


def preprocess_dataset():
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in SPLITS:
        total += process_split(split)
    print(f"done, {total} images processed in total")


if __name__ == "__main__":
    preprocess_dataset()