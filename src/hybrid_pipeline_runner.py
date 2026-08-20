import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image
from skimage import measure, morphology
import ollama

from src.unet_architecture import UNet
from src.unet_training_and_evaluation import get_device

TEST_IMAGES = Path("data/processed/test/images")
WEIGHTS_PATH = Path("outputs/task3_unet_segmentation/unet_weights.pth")
OUT_DIR = Path("outputs/task4_hybrid_pipeline")

TEXT_MODEL = "llama3.1"


def load_model(device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model


def predict_mask(model, device, img_path: Path):
    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    pred_mask = (pred.squeeze().cpu().numpy() > 0.5)
    pred_mask = morphology.remove_small_objects(pred_mask, min_size=15)
    return img, pred_mask


def extract_features(img, mask):
    labels = measure.label(mask)
    props = measure.regionprops_table(labels, intensity_image=img, properties=["label", "area"])
    return pd.DataFrame(props)


def classify_density(n_objects: int) -> str:
    if n_objects < 15:
        return "sparse"
    elif n_objects < 40:
        return "normal"
    else:
        return "dense"


def ask_llm_for_record(image_id: str, n_objects: int, mean_area: float) -> dict:
    density_guess = classify_density(n_objects)

    prompt = f"""You are assisting with an educational biomedical image analysis exercise.
                You are given numbers-only statistics from a segmented microscopy image, not the image itself.

                image_id: {image_id}
                n_objects: {n_objects}
                mean_area: {mean_area:.1f} pixels
                density_class (estimated): {density_guess}

                Write one short paragraph narrative describing what this suggests about the sample.
                Then on a new line return a JSON object with exactly these fields:

                {{
                "image_id": "{image_id}",
                "n_objects": {n_objects},
                "mean_area": {mean_area:.1f},
                "density_class": "{density_guess}",
                "quality_flag": "good, borderline, or poor, based on whether the numbers look sensible"
                }}
                """
    response = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def parse_json_from_response(text: str):
    start = text.rfind("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = load_model(device)

    all_records = []
    narratives = {}

    for img_path in sorted(TEST_IMAGES.glob("*.png")):
        image_id = img_path.stem
        print(f"processing {image_id}")

        img, mask = predict_mask(model, device, img_path)
        df = extract_features(img, mask)

        n_objects = len(df)
        mean_area = df["area"].mean() if n_objects > 0 else 0.0

        llm_response = ask_llm_for_record(image_id, n_objects, mean_area)
        parsed = parse_json_from_response(llm_response)

        narratives[image_id] = llm_response

        if parsed:
            all_records.append(parsed)
        else:
            all_records.append({
                "image_id": image_id,
                "n_objects": n_objects,
                "mean_area": round(mean_area, 1),
                "density_class": classify_density(n_objects),
                "quality_flag": "unparsed_llm_response",
            })

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(OUT_DIR / "hybrid_pipeline_results.csv", index=False)

    with open(OUT_DIR / "narratives.json", "w") as f:
        json.dump(narratives, f, indent=2)

    print(f"done, processed {len(all_records)} test images")
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    run()