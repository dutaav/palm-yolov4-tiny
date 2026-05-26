# Palm Oil FFB Ripeness Detection - YOLOv4-tiny

Replication of Salim and Suharjito (2023) with a split workflow:

- **Training** on **Modal** (A100-40GB GPUs, parallel) -> uploads artifacts to **HuggingFace Hub**
- **Inference and visualization** on **Google Colab** (T4 free tier) -> loads from HF Hub

## Structure

```
yolov4-tiny-hpo-ffb-maturity/
├── modal_training/              training pipeline (Modal H100)
│   ├── app.py                   4 models + GA + HF upload
│   ├── darknet_utils.py         cfg/log/eval parsers
│   ├── ga.py                    Genetic Algorithm
│   └── README.md
│
├── colab_inference/             one-click reproducibility notebook
│   ├── build_notebook.py        generator
│   └── PalmYOLOv4_Inference.ipynb
│
└── scripts/                     local helpers (run after training)
    ├── plot_convergence.py      loss + mAP convergence plots from logs
    ├── run_inference.py         OpenCV DNN inference (no darknet compile)
    └── pull_sample_images.py    fetch a small test-image subset from Roboflow
```

## Workflow

### 1. Train on Modal

```bash
cd modal_training
pip install modal
modal setup
modal run app.py
```

See `modal_training/README.md` for secret setup.

### 2. Inference on Colab

1. Upload `colab_inference/PalmYOLOv4_Inference.ipynb` to Google Colab
2. Set `HF_REPO = "dutaav/yolov4-tiny-hpo-ffb-maturity"` in the first cell
3. Set `ROBOFLOW_API_KEY` for test image downloads
4. Runtime -> Change runtime type -> GPU (T4)
5. Run all cells

Outputs: evaluation tables, plots, bounding box samples, exported as a zip.

## Performance

| Stage | Colab T4 (free) | Modal (A100-40GB) |
|-------|-----------------|-----------------|
| Darknet compilation | Each notebook run | Once, cached in volume |
| Train Model 1+2 | Sequential ~3h | Parallel on A100-40GB ~25-40 min |
| GA (50 fitness evals) | Sequential ~5h | 10 parallel A100-40GB containers ~15-25 min |
| Train Model 3+4 | Sequential ~3h | Parallel on A100-40GB ~25-40 min |
| **Total** | **10+ hours** | **~2.5-3.5 hours** |

GA fitness evaluations run in parallel via `Modal.Function.map()` - 10 individuals = 10 simultaneous containers per generation.

## Editing the Colab notebook

The notebook is generated from `build_notebook.py`. To modify:

```bash
cd colab_inference
python3 build_notebook.py
```

## Local scripts

After training has uploaded artifacts to HF, pull them locally with
`huggingface-cli download dutaav/yolov4-tiny-hpo-ffb-maturity --include 'runs/h100-hankai/*' --local-dir run_artifacts/`.
Then:

```bash
# Generate convergence plots (loss and mAP vs iteration) for the article.
python scripts/plot_convergence.py --run-tag h100-hankai

# Fetch a small subset of test images from Roboflow (avoids full dataset download).
ROBOFLOW_API_KEY=xxx python scripts/pull_sample_images.py --split test --count 30

# Run bbox inference locally on the sampled images (OpenCV DNN, CPU is enough).
python scripts/run_inference.py --images-dir run_artifacts/sample_images_roboflow --limit 30
```

Outputs land in `run_artifacts/{run_tag}/plots/` and `run_artifacts/{run_tag}/detections/`.

## Results

Dataset: Roboflow `tugas-akhir-pybma/palm-ripeness-detection` v5 (6 classes, train/valid/test = 11614/1657/835).

Test mAP@0.50:

| Model | mAP | Precision | Recall | F1 | Best LR |
|-------|-----|-----------|--------|-----|---------|
| YOLOv4-tiny baseline      | 87.99% | 0.591 | 0.842 | 0.695 | 0.00261 |
| YOLOv4-tiny + ES          | 85.94% | 0.603 | 0.861 | 0.709 | 0.00261 |
| YOLOv4-tiny + GA          | **89.75%** | 0.626 | 0.891 | 0.735 | **0.007003** |
| YOLOv4-tiny + ES + GA     | 86.23% | 0.624 | 0.847 | 0.718 | 0.007003 |

GA found a learning rate of `0.007003` (paper Salim & Suharjito 2023: `0.007465`, consistent range). GA alone improves test mAP by +1.76pp over baseline. Early stopping with patience=5 triggered too aggressively on the GA-tuned LR (stopped at iter 4258), see `run_artifacts/h100-hankai/plots/`.
