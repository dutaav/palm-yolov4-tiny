# Modal Training

YOLOv4-tiny training pipeline running on Modal (L4 GPU for training and evaluation, CPU for build and upload steps), uploading results to HuggingFace Hub.

## Setup (one-time)

### 1. Install Modal CLI

```bash
pip install modal
modal setup
```

### 2. Create Secrets on Modal Dashboard

Go to https://modal.com/secrets and create:

**Secret name: `roboflow`**
- Key: `ROBOFLOW_API_KEY`
- Value: API key from https://app.roboflow.com (Settings -> Roboflow API)

**Secret name: `huggingface`**
- Key: `HF_TOKEN`
- Value: Token from https://huggingface.co/settings/tokens (Write permission)

### 3. Create a HuggingFace model repository

- Go to https://huggingface.co/new
- Repository name: `palm-yolov4-tiny`
- Visibility: Public
- Full name: `dutaav/palm-yolov4-tiny`

Edit `app.py` and set:

```python
HF_REPO_DEFAULT = "dutaav/palm-yolov4-tiny"
```

## Run

### Full pipeline

```bash
cd modal_training
modal run app.py
```

Estimated time on the configured GPUs (L4 for all stages):
- Build darknet: ~5 minutes (one-time)
- Dataset download: ~3 minutes (one-time)
- Model 1 + Model 2 (parallel on L4): ~60-90 minutes
- GA (5 generations x 10 parallel evaluations on L4): ~25-40 minutes
- Model 3 + Model 4 (parallel on L4): ~60-90 minutes
- Evaluation + upload: ~5 minutes

Total: approximately 2.5-3.5 hours. Estimated cost: $3-5.

### Custom runs

Skip GA and use a predefined learning rate:

```bash
modal run app.py --skip-ga --ga-lr-override 0.007465
```

Custom HF repository:

```bash
modal run app.py --hf-repo myusername/palm-yolov4
```

Longer GA fitness iterations:

```bash
modal run app.py --fitness-iters 3000 --n-generations 5 --n-population 10
```

### Individual steps

```bash
modal run app.py::build_darknet
modal run app.py::prepare_dataset
modal run app.py::upload_to_hf --repo-id dutaav/palm-yolov4-tiny
```

### Force rebuild

```bash
modal run app.py::build_darknet --force
```

## Output

After completion, all artifacts are uploaded to HuggingFace:

```
HF_REPO/
├── weights/
│   ├── model1_baseline_best.weights
│   ├── model2_es_best.weights
│   ├── model3_ga_best.weights
│   └── model4_es_ga_best.weights
├── configs/*.cfg
├── logs/*.log
├── artifacts/
│   ├── metrics.json
│   ├── ga_history.json
│   └── training_info.json
└── dataset_info.json
```

Continue to `colab_inference/` for visualization in Google Colab.
