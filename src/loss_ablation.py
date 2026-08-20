import pandas as pd
from pathlib import Path

from src.unet_training_and_evaluation import train, evaluate_final_metrics

OUT_DIR = Path("outputs/task3_unet_segmentation/loss_ablation")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for loss_type in ["bce", "dice", "bce_dice"]:
        model, device, train_losses, val_dice_scores, val_ds, run_out_dir = train(loss_type=loss_type)
        mean_dice, mean_iou = evaluate_final_metrics(model, device, val_ds)
        results.append({"loss_type": loss_type, "final_val_dice": mean_dice, "final_val_iou": mean_iou})

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "loss_ablation_results.csv", index=False)
    print(df)
    print(f"saved comparison to {OUT_DIR}")


if __name__ == "__main__":
    run()