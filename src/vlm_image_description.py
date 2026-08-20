import ollama
import json
from pathlib import Path
import pandas as pd

IMAGE_DIR = Path("data/processed/train/images")
METADATA_PATH = Path("data/raw/nuclei_dataset/metadata.csv")
OUT_DIR = Path("outputs/task1_vlm_descriptions")

NAIVE_PROMPT = "What do you see in this image?"

ENGINEERED_PROMPT = """You are assisting with an educational biomedical image analysis exercise.
You are looking at a fluorescence microscopy image showing stained cell nuclei.

Your role is strictly descriptive, not diagnostic. Do not attempt to diagnose disease,
identify pathology, or make clinical judgements. This image is for research and teaching only.

Describe what you observe and return your answer as a JSON object with exactly these fields:

{
  "modality": "the imaging modality shown",
  "tissue_type": "the type of tissue or sample shown, or 'uncertain' if not clear",
  "notable_features": "a short description of visible features, e.g. clustering, brightness, shape variation",
  "image_quality": "your assessment of image quality, e.g. sharp, noisy, low contrast",
  "approx_density_estimate": "your rough estimate of nucleus density: sparse, normal, dense, or clustered"
}

If you are not confident about any field, write "uncertain" rather than guessing.
Return only the JSON object, nothing else.
"""


def pick_representative_image():
    meta = pd.read_csv(METADATA_PATH)
    row = meta[(meta["split"] == "train") & (meta["density"] == "normal")].iloc[0]
    return IMAGE_DIR / f"{row['image_id']}.png", row


def ask_vlm(image_path: Path, prompt: str) -> str:
    response = ollama.chat(
        model="llava",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
    )
    return response["message"]["content"]


def try_parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, row = pick_representative_image()
    print(f"using {image_path.name} as the representative image (ground truth density: {row['density']}, n_objects: {row['n_objects']})")

    results = {}

    print("asking naive prompt")
    naive_response = ask_vlm(image_path, NAIVE_PROMPT)
    results["naive_prompt"] = {"prompt": NAIVE_PROMPT, "response": naive_response}

    print("asking engineered prompt, run 1")
    run1 = ask_vlm(image_path, ENGINEERED_PROMPT)
    results["engineered_run_1"] = {"prompt": ENGINEERED_PROMPT, "response": run1, "parsed_json": try_parse_json(run1)}

    print("asking engineered prompt, run 2 (checking consistency)")
    run2 = ask_vlm(image_path, ENGINEERED_PROMPT)
    results["engineered_run_2"] = {"prompt": ENGINEERED_PROMPT, "response": run2, "parsed_json": try_parse_json(run2)}

    out_path = OUT_DIR / "vlm_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"saved results to {out_path}")


if __name__ == "__main__":
    run()