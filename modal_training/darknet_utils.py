from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_RE_ITER = re.compile(r"^\s*(\d+):")
_RE_LOSS = re.compile(r"(\d+\.\d+)\s+avg")
_RE_MAP = re.compile(
    r"mean\s+average\s+precision\s*\(mAP@0\.50\)\s*=\s*([\d.]+)",
    re.IGNORECASE,
)
_RE_TP = re.compile(r"TP\s*=\s*(\d+)")
_RE_FP = re.compile(r"FP\s*=\s*(\d+)")
_RE_FN = re.compile(r"FN\s*=\s*(\d+)")
_RE_PREC = re.compile(r"precision\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_REC = re.compile(r"recall\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_F1 = re.compile(r"F1[-\s]?score\s*=\s*([\d.]+)", re.IGNORECASE)
_RE_IOU = re.compile(r"average\s+IoU\s*=\s*([\d.]+)", re.IGNORECASE)


@dataclass(slots=True)
class EvalMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou: float = 0.0
    mAP: float = 0.0

    def to_dict(self) -> dict:
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "Precision": self.precision, "Recall": self.recall,
            "F1": self.f1, "IoU": self.iou, "mAP": self.mAP,
        }


def write_yolov4_tiny_cfg(
    template_path: str | Path,
    output_path: str | Path,
    num_classes: int,
    learning_rate: float = 0.00261,
    max_batches: int | None = None,
    subdivisions: int = 32,
    batch: int = 64,
) -> None:
    template_path = Path(template_path)
    output_path = Path(output_path)

    if max_batches is None:
        max_batches = max(num_classes * 2000, 6000)

    steps_1 = int(max_batches * 0.8)
    steps_2 = int(max_batches * 0.9)
    filters = (num_classes + 5) * 3

    with template_path.open("r") as f:
        lines = f.readlines()

    out: list[str] = []
    in_net = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[net]"):
            in_net = True
            out.append(line)
            continue
        if stripped.startswith("[") and not stripped.startswith("[net]"):
            in_net = False

        if in_net:
            if stripped.startswith("batch="):
                out.append(f"batch={batch}\n")
            elif stripped.startswith("subdivisions="):
                out.append(f"subdivisions={subdivisions}\n")
            elif stripped.startswith("max_batches"):
                out.append(f"max_batches={max_batches}\n")
            elif stripped.startswith("steps="):
                out.append(f"steps={steps_1},{steps_2}\n")
            elif stripped.startswith("learning_rate="):
                out.append(f"learning_rate={learning_rate}\n")
            else:
                out.append(line)
        else:
            out.append(line)

    yolo_indices = [i for i, line in enumerate(out) if line.strip() == "[yolo]"]

    for yolo_idx in yolo_indices:
        j = yolo_idx + 1
        while j < len(out) and not out[j].strip().startswith("["):
            if out[j].strip().startswith("classes="):
                out[j] = f"classes={num_classes}\n"
            j += 1

        k = yolo_idx - 1
        while k >= 0 and out[k].strip() != "[convolutional]":
            k -= 1
        if k >= 0:
            m = k + 1
            while m < yolo_idx and not out[m].strip().startswith("["):
                if out[m].strip().startswith("filters="):
                    out[m] = f"filters={filters}\n"
                    break
                m += 1

    with output_path.open("w") as f:
        f.writelines(out)


def stream_log_lines(file_or_proc) -> Iterator[str]:
    if hasattr(file_or_proc, "readline"):
        for line in file_or_proc:
            yield line
    else:
        with open(file_or_proc, "r") as f:
            for line in f:
                yield line


def parse_iteration_loss(line: str) -> tuple[int | None, float | None]:
    iter_m = _RE_ITER.match(line)
    loss_m = _RE_LOSS.search(line)
    return (
        int(iter_m.group(1)) if iter_m else None,
        float(loss_m.group(1)) if loss_m else None,
    )


def parse_map(line: str) -> float | None:
    m = _RE_MAP.search(line)
    return float(m.group(1)) if m else None


def parse_eval_output(text: str) -> EvalMetrics:
    em = EvalMetrics()
    for line in text.splitlines():
        if (m := _RE_TP.search(line)): em.tp = int(m.group(1))
        if (m := _RE_FP.search(line)): em.fp = int(m.group(1))
        if (m := _RE_FN.search(line)): em.fn = int(m.group(1))
        if (m := _RE_PREC.search(line)): em.precision = float(m.group(1))
        if (m := _RE_REC.search(line)): em.recall = float(m.group(1))
        if (m := _RE_F1.search(line)): em.f1 = float(m.group(1))
        if (m := _RE_IOU.search(line)): em.iou = float(m.group(1)) / 100.0
        if (m := _RE_MAP.search(line)): em.mAP = float(m.group(1))

    if em.f1 == 0 and em.precision > 0 and em.recall > 0:
        em.f1 = 2 * em.precision * em.recall / (em.precision + em.recall)
    return em


def write_obj_data(
    path: str | Path,
    num_classes: int,
    train_txt: str,
    valid_txt: str,
    names_file: str,
    backup_dir: str,
) -> None:
    Path(path).write_text(
        f"classes = {num_classes}\n"
        f"train = {train_txt}\n"
        f"valid = {valid_txt}\n"
        f"names = {names_file}\n"
        f"backup = {backup_dir}\n"
    )


def generate_file_list(split_dir: str | Path, output_path: str | Path) -> int:
    split_dir = Path(split_dir)
    exts = (".jpg", ".jpeg", ".png")
    imgs = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in exts)
    with open(output_path, "w") as f:
        f.write("\n".join(str(p) for p in imgs))
    return len(imgs)
