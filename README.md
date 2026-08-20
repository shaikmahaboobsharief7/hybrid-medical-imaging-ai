# Hybrid Medical Imaging AI

A pipeline combining a vision-language model, classical image processing, and a U-Net to analyse fluorescence microscopy nuclei images.

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull llava
ollama pull llama3.1
```

## Run the pipeline (in order)

```bash
python src/image_preprocessing.py
python src/exploratory_data_analysis.py
python src/vlm_image_description.py
python src/classical_feature_extraction.py
python -m src.unet_training_and_evaluation
python -m src.hybrid_pipeline_runner
```

## Optional extensions

```bash
python -m src.robustness_check
python -m src.loss_ablation
python -m src.foundation_model_comparison
```

## Outputs

All results are saved under `outputs/`, organised by task.