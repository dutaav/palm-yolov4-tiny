from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    ("model1_baseline", "YOLOv4-tiny baseline"),
    ("model2_es", "YOLOv4-tiny + ES"),
    ("model3_ga", "YOLOv4-tiny + GA"),
    ("model4_es_ga", "YOLOv4-tiny + ES + GA"),
]

CLASS_COLORS = [
    (255, 56, 56),
    (255, 159, 56),
    (255, 235, 56),
    (102, 204, 0),
    (52, 152, 219),
    (155, 89, 182),
]


def load_class_names(run_dir: Path) -> list[str]:
    info = json.loads((run_dir / "dataset_info.json").read_text())
    names = info["class_names"]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("class_names must be list[str]")
    return names


def build_net(cfg_path: Path, weights_path: Path) -> cv2.dnn.Net:
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect(
    net: cv2.dnn.Net,
    image: np.ndarray,
    input_size: int = 416,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
) -> list[tuple[int, float, int, int, int, int]]:
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    boxes: list[list[int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for out in outputs:
        for det in out:
            scores = det[5:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < conf_thresh:
                continue
            cx, cy, bw, bh = det[0:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            ww = int(bw * w)
            hh = int(bh * h)
            boxes.append([x, y, ww, hh])
            confidences.append(conf)
            class_ids.append(cls)

    if not boxes:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    if len(idxs) == 0:
        return []

    flat = idxs.flatten() if hasattr(idxs, "flatten") else idxs
    result: list[tuple[int, float, int, int, int, int]] = []
    for i in flat:
        i = int(i)
        x, y, ww, hh = boxes[i]
        result.append((class_ids[i], confidences[i], x, y, x + ww, y + hh))
    return result


def draw_detections(
    image: np.ndarray,
    detections: list[tuple[int, float, int, int, int, int]],
    class_names: list[str],
) -> np.ndarray:
    out = image.copy()
    for cls, conf, x1, y1, x2, y2 in detections:
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - bl - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def collect_images(images_dir: Path, limit: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        raise FileNotFoundError(f"no images found under {images_dir}")
    if limit is not None and limit < len(imgs):
        random.seed(42)
        imgs = random.sample(imgs, limit)
    return imgs


def download_test_split(target_dir: Path) -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set in env")
    from roboflow import Roboflow

    target_dir.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("tugas-akhir-pybma").project("palm-ripeness-detection")
    dataset = project.version(5).download("darknet", location=str(target_dir))
    test_dir = Path(dataset.location) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"expected {test_dir} after download")
    return test_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="h100-hankai")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="folder containing test images (jpg/png)")
    parser.add_argument("--download-test", action="store_true",
                        help="download dataset v5 from Roboflow into run_artifacts/dataset/")
    parser.add_argument("--model", choices=[m[0] for m in MODELS] + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=6, help="number of images to run")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    args = parser.parse_args()

    run_dir = ROOT / "run_artifacts" / args.run_tag
    class_names = load_class_names(run_dir)
    print(f"[info] classes ({len(class_names)}): {class_names}")

    if args.images_dir is not None:
        images_dir = args.images_dir
    elif args.download_test:
        images_dir = download_test_split(ROOT / "run_artifacts" / "dataset")
    else:
        default = ROOT / "run_artifacts" / "dataset" / "test"
        if default.exists():
            images_dir = default
        else:
            raise SystemExit(
                "no images source. pass --images-dir PATH or --download-test "
                "(needs ROBOFLOW_API_KEY in env)"
            )

    imgs = collect_images(images_dir, args.limit)
    print(f"[info] running on {len(imgs)} images from {images_dir}")

    models_to_run = MODELS if args.model == "all" else [m for m in MODELS if m[0] == args.model]

    out_root = run_dir / "detections"
    out_root.mkdir(parents=True, exist_ok=True)

    for name, label in models_to_run:
        cfg_path = run_dir / "configs" / f"{name}.cfg"
        weights_path = run_dir / "weights" / f"{name}_best.weights"
        if not cfg_path.exists() or not weights_path.exists():
            print(f"[skip] {name}: missing cfg or weights")
            continue
        print(f"\n[model] {name} ({label})")
        net = build_net(cfg_path, weights_path)
        model_out = out_root / name
        model_out.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [warn] cannot read {img_path}")
                continue
            dets = detect(net, img, conf_thresh=args.conf, nms_thresh=args.nms)
            drawn = draw_detections(img, dets, class_names)
            out_path = model_out / img_path.name
            cv2.imwrite(str(out_path), drawn)
            classes_found = sorted({class_names[d[0]] for d in dets})
            print(f"  {img_path.name}: {len(dets)} boxes -> {classes_found}")

    print(f"\n[done] outputs in {out_root}")


if __name__ == "__main__":
    main()
