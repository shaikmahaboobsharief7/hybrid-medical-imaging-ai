import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io, filters, morphology, measure
import ollama
import json

IMAGE_DIR = Path("data/processed/train/images")
METADATA_PATH = Path("data/raw/nuclei_dataset/metadata.csv")
OUT_DIR = Path("outputs/task2_classical_features")

TEXT_MODEL = "llama3.1"


def segment_image(img_path: Path):
    img = io.imread(img_path)

    # otsu picks the threshold automatically based on the image histogram
    thresh = filters.threshold_otsu(img)
    binary = img > thresh

    # clean up small noise specks and fill tiny holes
    cleaned = morphology.remove_small_objects(binary, min_size=15)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=15)

    # label connected regions so each nucleus gets its own id
    labels = measure.label(cleaned)
    return img, labels


def extract_features(img, labels):
    props = measure.regionprops_table(
        labels,
        intensity_image=img,
        properties=["label", "area", "eccentricity", "solidity", "mean_intensity"],
    )
    return pd.DataFrame(props)


def summarize_features(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "No objects were detected in this image."

    n_objects = len(df)
    mean_area = df["area"].mean()
    mean_eccentricity = df["eccentricity"].mean()
    mean_solidity = df["solidity"].mean()
    mean_intensity = df["mean_intensity"].mean()

    summary = (
        f"The image contains {n_objects} detected objects. "
        f"Average object area is {mean_area:.1f} pixels. "
        f"Average eccentricity is {mean_eccentricity:.2f} (0 is circular, 1 is elongated). "
        f"Average solidity is {mean_solidity:.2f} (how filled-in vs irregular the shape is). "
        f"Average mean intensity is {mean_intensity:.1f}."
    )
    return summary


def ask_llm_for_interpretation(summary_text: str) -> str:
    prompt = f"""You are assisting with an educational biomedical image analysis exercise.
You are given a numbers-only summary of objects detected in a microscopy image.
You have not seen the image itself, only these statistics.

Summary:
{summary_text}

Write one short paragraph describing what this suggests about the sample.
Then return a JSON object on a new line with exactly these fields:

{{
  "n_objects": integer,
  "density_class": "sparse, normal, dense, or clustered",
  "shape_regularity": "regular, irregular, or mixed",
  "quality_flag": "good, borderline, or poor"
}}

If you are unsure about any field, use your best judgement based on the numbers given.
"""
    response = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def pick_representative_image():
    meta = pd.read_csv(METADATA_PATH)
    row = meta[(meta["split"] == "train") & (meta["density"] == "normal")].iloc[0]
    return IMAGE_DIR / f"{row['image_id']}.png", row


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, row = pick_representative_image()
    print(f"using {image_path.name} (ground truth: {row['density']}, n_objects: {row['n_objects']})")

    img, labels = segment_image(image_path)
    df = extract_features(img, labels)

    # save the feature table so it's available as raw evidence in the report
    df.to_csv(OUT_DIR / "regionprops_table.csv", index=False)
    print(f"detected {len(df)} objects with otsu + connected components")

    summary = summarize_features(df)
    print("asking llm to interpret the numbers")
    llm_response = ask_llm_for_interpretation(summary)

    result = {
        "image": image_path.name,
        "ground_truth_n_objects": int(row["n_objects"]),
        "ground_truth_density": row["density"],
        "detected_n_objects": len(df),
        "numeric_summary": summary,
        "llm_response": llm_response,
    }

    with open(OUT_DIR / "classical_llm_interpretation.json", "w") as f:
        json.dump(result, f, indent=2)

    print("saved results to outputs/task2_classical_features")


if __name__ == "__main__":
    run()