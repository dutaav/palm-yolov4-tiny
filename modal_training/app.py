from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal


app = modal.App("palm-yolov4-tiny")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .run_commands(
        "ln -snf /usr/share/zoneinfo/$TZ /etc/localtime",
        "echo $TZ > /etc/timezone",
    )
    .apt_install(
        "git", "wget", "build-essential",
        "libopencv-dev", "python3-opencv",
        "pkg-config",
    )
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "roboflow==1.1.50",
        "huggingface_hub==0.26.2",
    )
    .add_local_python_source("darknet_utils", "ga")
)

volume = modal.Volume.from_name("palm-yolov4-vol", create_if_missing=True)

VOL = "/vol"
DARKNET_DIR = f"{VOL}/darknet"
DARKNET_BIN = f"{DARKNET_DIR}/darknet"
DATASET_DIR = f"{VOL}/dataset"
DATA_DIR = f"{VOL}/data"
CFG_DIR = f"{VOL}/cfg"
WEIGHTS_DIR = f"{VOL}/weights"
LOGS_DIR = f"{VOL}/logs"
ARTIFACTS_DIR = f"{VOL}/artifacts"
PRETRAINED = f"{VOL}/yolov4-tiny.conv.29"

ROBOFLOW_WORKSPACE = "tugas-akhir-pybma"
ROBOFLOW_PROJECT = "palm-ripeness-detection"
ROBOFLOW_VERSION = 5

HF_REPO_DEFAULT = "dutaav/palm-yolov4-tiny"


@app.function(
    image=image,
    volumes={VOL: volume},
    timeout=1800,
)
def build_darknet(force: bool = False) -> str:
    if not force and Path(DARKNET_BIN).exists():
        print(f"[build_darknet] Already built at {DARKNET_BIN}, skipping.")
        return DARKNET_BIN

    Path(VOL).mkdir(parents=True, exist_ok=True)

    if Path(DARKNET_DIR).exists():
        shutil.rmtree(DARKNET_DIR)

    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/AlexeyAB/darknet.git", DARKNET_DIR],
        check=True,
    )

    makefile = Path(DARKNET_DIR) / "Makefile"
    txt = makefile.read_text()
    for old, new in [
        ("GPU=0", "GPU=1"),
        ("CUDNN=0", "CUDNN=1"),
        ("CUDNN_HALF=0", "CUDNN_HALF=1"),
        ("OPENCV=0", "OPENCV=1"),
        ("LIBSO=0", "LIBSO=1"),
    ]:
        txt = txt.replace(old, new)
    makefile.write_text(txt)

    subprocess.run(
        ["make", "-j", str(os.cpu_count() or 4)],
        cwd=DARKNET_DIR, check=True,
    )

    if not Path(PRETRAINED).exists():
        subprocess.run(
            ["wget", "-q",
             "https://github.com/AlexeyAB/darknet/releases/download/"
             "darknet_yolo_v4_pre/yolov4-tiny.conv.29",
             "-O", PRETRAINED],
            check=True,
        )

    volume.commit()
    print(f"[build_darknet] Done. Binary: {DARKNET_BIN}")
    return DARKNET_BIN


@app.function(
    image=image,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("roboflow")],
    timeout=1800,
    retries=modal.Retries(max_retries=5, backoff_coefficient=2.0, initial_delay=10.0),
)
def prepare_dataset(force: bool = False) -> dict:
    from roboflow import Roboflow
    from darknet_utils import write_obj_data, generate_file_list

    info_path = Path(VOL) / "dataset_info.json"
    if not force and info_path.exists():
        info = json.loads(info_path.read_text())
        print(f"[prepare_dataset] Cached: {info}")
        return info

    api_key = os.environ["ROBOFLOW_API_KEY"]
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    os.chdir(DATASET_DIR)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("darknet")

    ds_path = Path(dataset.location)
    print(f"[prepare_dataset] Downloaded to {ds_path}")

    labels_file = None
    for candidate in ["train/_darknet.labels", "_darknet.labels", "obj.names"]:
        p = ds_path / candidate
        if p.exists():
            labels_file = p
            break
    if labels_file is None:
        for root, _, files in os.walk(ds_path):
            for f in files:
                if f.endswith(".labels"):
                    labels_file = Path(root) / f
                    break
            if labels_file:
                break

    assert labels_file is not None, "Could not find class labels file"
    class_names = [
        ln.strip() for ln in labels_file.read_text().splitlines() if ln.strip()
    ]
    num_classes = len(class_names)

    names_path = Path(DATA_DIR) / "obj.names"
    names_path.write_text("\n".join(class_names) + "\n")

    train_txt = Path(DATA_DIR) / "train.txt"
    valid_txt = Path(DATA_DIR) / "valid.txt"
    test_txt = Path(DATA_DIR) / "test.txt"

    n_train = generate_file_list(ds_path / "train", train_txt)
    n_valid = generate_file_list(ds_path / "valid", valid_txt)
    n_test = generate_file_list(ds_path / "test", test_txt)

    backup_dir = f"{VOL}/backup"
    Path(backup_dir).mkdir(parents=True, exist_ok=True)

    write_obj_data(
        Path(DATA_DIR) / "obj.data",
        num_classes, str(train_txt), str(valid_txt), str(names_path), backup_dir,
    )
    write_obj_data(
        Path(DATA_DIR) / "obj_test.data",
        num_classes, str(train_txt), str(test_txt), str(names_path), backup_dir,
    )

    info = {
        "num_classes": num_classes,
        "class_names": class_names,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "dataset_path": str(ds_path),
    }
    info_path.write_text(json.dumps(info, indent=2))
    volume.commit()
    print(f"[prepare_dataset] {info}")
    return info


def _run_darknet_train(
    cfg_path: str,
    log_path: str,
    early_stopping: bool = False,
    patience: int = 5,
    data_path: str | None = None,
) -> tuple[float, int]:
    from darknet_utils import parse_iteration_loss, parse_map

    data_path = data_path or f"{DATA_DIR}/obj.data"

    cmd = [
        DARKNET_BIN, "detector", "train",
        data_path, cfg_path, PRETRAINED,
        "-dont_show", "-map", "-mjpeg_port", "-1",
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=DARKNET_DIR,
    )

    best_map = 0.0
    best_iter = 0
    no_improve = 0
    current_iter = 0

    log_f = open(log_path, "w")
    try:
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()

            it, _loss = parse_iteration_loss(line)
            if it is not None:
                current_iter = it

            mAP = parse_map(line)
            if mAP is not None:
                if mAP > best_map:
                    best_map = mAP
                    best_iter = current_iter
                    no_improve = 0
                    print(f"  iter {current_iter}: mAP={mAP:.4f} [BEST]")
                else:
                    no_improve += 1
                    print(f"  iter {current_iter}: mAP={mAP:.4f} [no improve {no_improve}/{patience}]")

                if early_stopping and no_improve >= patience:
                    print(f"  Early stopping at iter {current_iter} (best mAP={best_map:.4f} @ {best_iter})")
                    proc.terminate()
                    break
    finally:
        log_f.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    return best_map, best_iter


@app.function(
    image=image,
    volumes={VOL: volume},
    gpu="H100",
    timeout=4 * 60 * 60,
)
def train_one_model(
    name: str,
    learning_rate: float,
    use_early_stopping: bool = False,
    patience: int = 5,
    max_batches: int | None = None,
) -> dict:
    from darknet_utils import write_yolov4_tiny_cfg

    volume.reload()

    info = json.loads(Path(f"{VOL}/dataset_info.json").read_text())
    num_classes = info["num_classes"]

    Path(CFG_DIR).mkdir(parents=True, exist_ok=True)
    Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    cfg_path = f"{CFG_DIR}/{name}.cfg"
    log_path = f"{LOGS_DIR}/{name}.log"

    write_yolov4_tiny_cfg(
        template_path=f"{DARKNET_DIR}/cfg/yolov4-tiny.cfg",
        output_path=cfg_path,
        num_classes=num_classes,
        learning_rate=learning_rate,
        max_batches=max_batches,
    )

    print(f"\n[train:{name}] LR={learning_rate}, ES={use_early_stopping}, max_batches={max_batches}")
    t0 = time.time()
    best_map, best_iter = _run_darknet_train(
        cfg_path=cfg_path,
        log_path=log_path,
        early_stopping=use_early_stopping,
        patience=patience,
    )
    elapsed = time.time() - t0
    print(f"[train:{name}] done in {elapsed:.1f}s, best mAP={best_map:.4f} @ iter {best_iter}")

    backup_best = Path(f"{VOL}/backup/{name}_best.weights")
    if not backup_best.exists():
        backup_best = Path(f"{VOL}/backup/{name}_last.weights")
    final_weights = Path(WEIGHTS_DIR) / f"{name}_best.weights"
    shutil.copy(backup_best, final_weights)

    volume.commit()

    return {
        "name": name,
        "learning_rate": learning_rate,
        "early_stopping": use_early_stopping,
        "best_mAP_training": best_map,
        "best_iteration": best_iter,
        "elapsed_sec": elapsed,
        "weights_path": str(final_weights),
        "cfg_path": cfg_path,
        "log_path": log_path,
    }


@app.function(
    image=image,
    volumes={VOL: volume},
    gpu="H100",
    timeout=60 * 60,
)
def ga_fitness_eval(learning_rate: float, fitness_iters: int = 2000) -> float:
    from darknet_utils import write_yolov4_tiny_cfg, parse_eval_output

    volume.reload()
    info = json.loads(Path(f"{VOL}/dataset_info.json").read_text())
    num_classes = info["num_classes"]

    tag = f"ga_eval_{learning_rate:.6f}".replace(".", "_")
    cfg_path = f"{CFG_DIR}/{tag}.cfg"
    Path(CFG_DIR).mkdir(parents=True, exist_ok=True)

    write_yolov4_tiny_cfg(
        template_path=f"{DARKNET_DIR}/cfg/yolov4-tiny.cfg",
        output_path=cfg_path,
        num_classes=num_classes,
        learning_rate=learning_rate,
        max_batches=fitness_iters,
    )

    print(f"[ga_fit] LR={learning_rate:.6f}, iters={fitness_iters}")
    t0 = time.time()

    subprocess.run(
        [DARKNET_BIN, "detector", "train",
         f"{DATA_DIR}/obj.data", cfg_path, PRETRAINED,
         "-dont_show", "-mjpeg_port", "-1"],
        cwd=DARKNET_DIR, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    weights_path = f"{VOL}/backup/{tag}_last.weights"
    result = subprocess.run(
        [DARKNET_BIN, "detector", "map",
         f"{DATA_DIR}/obj.data", cfg_path, weights_path,
         "-dont_show"],
        cwd=DARKNET_DIR, capture_output=True, text=True,
    )
    metrics = parse_eval_output(result.stdout)

    Path(weights_path).unlink(missing_ok=True)
    Path(f"{VOL}/backup/{tag}_final.weights").unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"[ga_fit] LR={learning_rate:.6f} result mAP={metrics.mAP:.4f} ({elapsed:.0f}s)")
    return metrics.mAP


@app.function(
    image=image,
    volumes={VOL: volume},
    gpu="H100",
    timeout=30 * 60,
)
def evaluate_on_test(name: str, cfg_path: str, weights_path: str) -> dict:
    from darknet_utils import parse_eval_output

    volume.reload()
    out = {}
    for split, data_file in [
        ("valid", f"{DATA_DIR}/obj.data"),
        ("test", f"{DATA_DIR}/obj_test.data"),
    ]:
        result = subprocess.run(
            [DARKNET_BIN, "detector", "map",
             data_file, cfg_path, weights_path, "-dont_show"],
            cwd=DARKNET_DIR, capture_output=True, text=True,
        )
        out[split] = parse_eval_output(result.stdout).to_dict()
        Path(f"{LOGS_DIR}/{name}_eval_{split}.txt").write_text(result.stdout)

    print(f"[eval:{name}] test mAP={out['test']['mAP']:.4f}")
    volume.commit()
    return {"name": name, **out}


@app.function(image=image, volumes={VOL: volume}, timeout=300)
def save_artifacts(
    metrics: list, ga_result: dict, training_info: list,
) -> None:
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{ARTIFACTS_DIR}/metrics.json").write_text(json.dumps(metrics, indent=2))
    Path(f"{ARTIFACTS_DIR}/ga_history.json").write_text(json.dumps(ga_result, indent=2))
    Path(f"{ARTIFACTS_DIR}/training_info.json").write_text(json.dumps(training_info, indent=2))
    volume.commit()
    print("[save_artifacts] Written metrics.json, ga_history.json, training_info.json")


@app.function(
    image=image,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=30 * 60,
)
def upload_to_hf(repo_id: str) -> str:
    from huggingface_hub import HfApi, create_repo

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    create_repo(repo_id=repo_id, token=token, exist_ok=True, repo_type="model")

    for subdir, path_in_repo in [
        (WEIGHTS_DIR, "weights"),
        (CFG_DIR, "configs"),
        (LOGS_DIR, "logs"),
        (ARTIFACTS_DIR, "artifacts"),
    ]:
        if Path(subdir).exists():
            ignore = ["ga_eval_*"] if subdir == CFG_DIR else None
            api.upload_folder(
                folder_path=subdir,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                ignore_patterns=ignore,
            )

    api.upload_file(
        path_or_fileobj=f"{VOL}/dataset_info.json",
        path_in_repo="dataset_info.json",
        repo_id=repo_id,
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[hf] Uploaded to {url}")
    return url


@app.local_entrypoint()
def main(
    hf_repo: str = HF_REPO_DEFAULT,
    skip_ga: bool = False,
    ga_lr_override: float = 0.0,
    fitness_iters: int = 2000,
    n_generations: int = 5,
    n_population: int = 10,
):
    print(">>> Step 1: Build Darknet")
    build_darknet.remote()

    print("\n>>> Step 2: Prepare dataset")
    info = prepare_dataset.remote()
    print(f"   classes={info['num_classes']}, "
          f"train/val/test={info['n_train']}/{info['n_valid']}/{info['n_test']}")

    print("\n>>> Step 3: Train Model 1 (baseline) + Model 2 (ES) in parallel")
    futures = [
        train_one_model.spawn("model1_baseline", learning_rate=0.00261,
                              use_early_stopping=False),
        train_one_model.spawn("model2_es", learning_rate=0.00261,
                              use_early_stopping=True, patience=5),
    ]
    model1_info = futures[0].get()
    model2_info = futures[1].get()

    if skip_ga or ga_lr_override > 0:
        best_lr = ga_lr_override if ga_lr_override > 0 else 0.007465
        ga_result = {"best_lr": best_lr, "best_map": 0.0, "history": [],
                     "note": "GA skipped, used override"}
        print(f"\n>>> Skipping GA, using LR={best_lr}")
    else:
        print(f"\n>>> Step 4: GA ({n_generations} generations x {n_population} parallel evaluations)")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ga import run_genetic_algorithm, GAConfig

        ga_cfg = GAConfig(
            n_population=n_population,
            n_generations=n_generations,
            fitness_iters=fitness_iters,
        )

        def parallel_fitness(lrs: list[float]) -> list[float]:
            return list(ga_fitness_eval.map(
                lrs,
                kwargs={"fitness_iters": fitness_iters},
            ))

        result = run_genetic_algorithm(parallel_fitness, ga_cfg)
        best_lr = result.best_lr
        ga_result = result.to_dict()

    print(f"\n>>> Step 5: Train Model 3 (GA LR) + Model 4 (ES + GA LR) in parallel")
    futures = [
        train_one_model.spawn("model3_ga", learning_rate=best_lr,
                              use_early_stopping=False),
        train_one_model.spawn("model4_es_ga", learning_rate=best_lr,
                              use_early_stopping=True, patience=5),
    ]
    model3_info = futures[0].get()
    model4_info = futures[1].get()

    print("\n>>> Step 6: Evaluate all 4 models on validation + test sets")
    eval_futures = [
        evaluate_on_test.spawn(m["name"], m["cfg_path"], m["weights_path"])
        for m in [model1_info, model2_info, model3_info, model4_info]
    ]
    eval_results = [f.get() for f in eval_futures]

    print("\n>>> Step 7: Save artifacts + upload to HuggingFace")
    save_artifacts.remote(
        eval_results, ga_result,
        [model1_info, model2_info, model3_info, model4_info],
    )

    url = upload_to_hf.remote(hf_repo)

    print("\n" + "=" * 80)
    print("FINAL RESULTS (TEST SET)")
    print("=" * 80)
    print(f"{'Model':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'IoU':>7} {'F1':>7} {'mAP':>8}")
    print("-" * 80)
    for r in eval_results:
        t = r["test"]
        print(f"{r['name']:<22} {t['TP']:>5} {t['FP']:>5} {t['FN']:>5} "
              f"{t['Precision']:>7.4f} {t['Recall']:>7.4f} "
              f"{t['IoU']*100:>6.2f}% {t['F1']:>7.4f} {t['mAP']*100:>7.2f}%")

    print(f"\nGA Best LR: {best_lr:.6f}")
    print(f"HF repo: {url}")
