# Palm Oil FFB Ripeness Detection - YOLOv4-tiny

Replication of Salim and Suharjito (2023) with a split workflow:

- **Training** on **Modal** (L4 GPUs, parallel) -> uploads artifacts to **HuggingFace Hub**
- **Inference and visualization** on **Google Colab** (T4 free tier) -> loads from HF Hub

## Structure

```
palm-yolov4-tiny/
├── modal_training/
│   ├── app.py
│   ├── darknet_utils.py
│   ├── ga.py
│   └── README.md
│
└── colab_inference/
    ├── build_notebook.py
    └── PalmYOLOv4_Inference.ipynb
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
2. Set `HF_REPO = "dutaav/palm-yolov4-tiny"` in the first cell
3. Set `ROBOFLOW_API_KEY` for test image downloads
4. Runtime -> Change runtime type -> GPU (T4)
5. Run all cells

Outputs: evaluation tables, plots, bounding box samples, exported as a zip.

## Performance

| Stage | Colab T4 (free) | Modal (L4) |
|-------|-----------------|-----------------|
| Darknet compilation | Each notebook run | Once, cached in volume |
| Train Model 1+2 | Sequential ~3h | Parallel on L4 ~60-90 min |
| GA (50 fitness evals) | Sequential ~5h | 10 parallel L4 containers ~25-40 min |
| Train Model 3+4 | Sequential ~3h | Parallel on L4 ~60-90 min |
| **Total** | **10+ hours** | **~2.5-3.5 hours** |

GA fitness evaluations run in parallel via `Modal.Function.map()` - 10 individuals = 10 simultaneous containers per generation.

## Editing the Colab notebook

The notebook is generated from `build_notebook.py`. To modify:

```bash
cd colab_inference
python3 build_notebook.py
```
