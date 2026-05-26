from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modal_training"))

from darknet_utils import parse_iteration_loss, parse_map, _RE_ANSI


MODELS = [
    ("model1_baseline", "YOLOv4-tiny baseline"),
    ("model2_es", "YOLOv4-tiny + ES"),
    ("model3_ga", "YOLOv4-tiny + GA"),
    ("model4_es_ga", "YOLOv4-tiny + ES + GA"),
]

COLORS = {
    "model1_baseline": "#1f77b4",
    "model2_es": "#ff7f0e",
    "model3_ga": "#2ca02c",
    "model4_es_ga": "#d62728",
}


@dataclass(slots=True)
class TrainCurve:
    iters: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    map_iters: list[int] = field(default_factory=list)
    map_values: list[float] = field(default_factory=list)


def parse_log(log_path: Path) -> TrainCurve:
    curve = TrainCurve()
    last_iter = 0
    pending_map_calc = False

    with log_path.open("r", errors="replace") as f:
        raw = f.read()

    clean = _RE_ANSI.sub("", raw)

    for line in clean.splitlines():
        it, loss = parse_iteration_loss(line)
        if it is not None and loss is not None:
            curve.iters.append(it)
            curve.losses.append(loss)
            last_iter = it
            continue

        if "Calculating mAP" in line:
            pending_map_calc = True
            continue

        m = parse_map(line)
        if m is not None and pending_map_calc:
            curve.map_iters.append(last_iter)
            curve.map_values.append(m)
            pending_map_calc = False

    return curve


def smooth(values: list[float], window: int = 50) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    acc = 0.0
    from collections import deque
    buf: deque[float] = deque(maxlen=window)
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf))
    return out


def plot_loss_per_model(curves: dict[str, TrainCurve], out_dir: Path) -> None:
    for name, label in MODELS:
        c = curves[name]
        if not c.iters:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(c.iters, c.losses, color=COLORS[name], alpha=0.25, linewidth=0.6, label="raw")
        ax.plot(c.iters, smooth(c.losses, 100), color=COLORS[name], linewidth=1.8, label="smoothed (window=100)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Average Loss")
        ax.set_title(f"Training Loss Convergence: {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_{name}.png", dpi=150)
        plt.close(fig)


def plot_loss_combined(curves: dict[str, TrainCurve], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, label in MODELS:
        c = curves[name]
        if not c.iters:
            continue
        ax.plot(c.iters, smooth(c.losses, 100), color=COLORS[name], linewidth=1.6, label=label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Loss (smoothed, window=100)")
    ax.set_title("Training Loss Convergence Comparison")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_combined.png", dpi=150)
    plt.close(fig)


def plot_map_combined(curves: dict[str, TrainCurve], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, label in MODELS:
        c = curves[name]
        if not c.map_iters:
            continue
        ax.plot(
            c.map_iters,
            [v * 100 for v in c.map_values],
            color=COLORS[name],
            linewidth=1.6,
            marker="o",
            markersize=3,
            label=label,
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mAP@0.50 (%)")
    ax.set_title("Validation mAP During Training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "map_combined.png", dpi=150)
    plt.close(fig)


def plot_map_per_model(curves: dict[str, TrainCurve], out_dir: Path) -> None:
    for name, label in MODELS:
        c = curves[name]
        if not c.map_iters:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            c.map_iters,
            [v * 100 for v in c.map_values],
            color=COLORS[name],
            linewidth=1.6,
            marker="o",
            markersize=4,
        )
        best_idx = max(range(len(c.map_values)), key=lambda i: c.map_values[i])
        ax.axhline(c.map_values[best_idx] * 100, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"best={c.map_values[best_idx] * 100:.2f}% @ iter {c.map_iters[best_idx]}",
            xy=(c.map_iters[best_idx], c.map_values[best_idx] * 100),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=9,
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("mAP@0.50 (%)")
        ax.set_title(f"Validation mAP: {label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"map_{name}.png", dpi=150)
        plt.close(fig)


def dump_summary(curves: dict[str, TrainCurve], out_dir: Path) -> None:
    summary = {}
    for name, _ in MODELS:
        c = curves[name]
        if not c.iters:
            continue
        best_idx = max(range(len(c.map_values)), key=lambda i: c.map_values[i]) if c.map_values else None
        summary[name] = {
            "num_iter_points": len(c.iters),
            "last_iter": c.iters[-1],
            "final_loss": c.losses[-1],
            "min_loss": min(c.losses),
            "num_map_evals": len(c.map_values),
            "best_map_training": c.map_values[best_idx] if best_idx is not None else None,
            "best_map_iter": c.map_iters[best_idx] if best_idx is not None else None,
        }
    (out_dir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="h100-hankai")
    args = parser.parse_args()

    run_dir = ROOT / "run_artifacts" / args.run_tag
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[str, TrainCurve] = {}
    for name, _ in MODELS:
        log_path = logs_dir / f"{name}.log"
        if not log_path.exists():
            print(f"[skip] {log_path} not found")
            continue
        print(f"[parse] {log_path.name}")
        curves[name] = parse_log(log_path)
        c = curves[name]
        print(f"  iters={len(c.iters)} last_iter={c.iters[-1] if c.iters else 0} "
              f"map_evals={len(c.map_values)} best_map={max(c.map_values) if c.map_values else 0:.4f}")

    print(f"\n[plot] writing into {out_dir}")
    plot_loss_per_model(curves, out_dir)
    plot_loss_combined(curves, out_dir)
    plot_map_per_model(curves, out_dir)
    plot_map_combined(curves, out_dir)
    dump_summary(curves, out_dir)
    print("[done]")


if __name__ == "__main__":
    main()
