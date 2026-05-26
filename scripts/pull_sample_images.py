from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = "tugas-akhir-pybma"
PROJECT = "palm-ripeness-detection"
VERSION = 5


def fetch_image_list(
    api_key: str,
    split: str,
    limit: int,
    offset: int = 0,
    class_name: str | None = None,
) -> list[dict]:
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/search"
    payload: dict = {
        "api_key": api_key,
        "fields": ["id", "name", "split", "labels", "owner"],
        "split": split,
        "limit": limit,
        "offset": offset,
    }
    if class_name:
        payload["class"] = class_name
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "results" not in data:
        raise RuntimeError(f"unexpected response: {data}")
    return data["results"]


def fetch_image_bytes(api_key: str, image_id: str) -> bytes:
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/images/{image_id}"
    r = requests.get(url, params={"api_key": api_key}, timeout=60)
    r.raise_for_status()
    info = r.json()
    img_url = info.get("image", {}).get("urls", {}).get("original")
    if not img_url:
        raise RuntimeError(f"no original url in {info}")
    blob = requests.get(img_url, timeout=120)
    blob.raise_for_status()
    return blob.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--count", type=int, default=30,
                        help="total images to sample (spread across split)")
    parser.add_argument("--split-size", type=int, default=835,
                        help="approx total images in the split")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "run_artifacts" / "sample_images_roboflow")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set")

    args.out.mkdir(parents=True, exist_ok=True)

    stride = max(1, args.split_size // args.count)
    print(f"[info] sampling {args.count} from {args.split} (stride={stride})")
    total = 0
    for i in range(args.count):
        offset = i * stride
        try:
            items = fetch_image_list(api_key, args.split, limit=1, offset=offset)
        except Exception as e:
            print(f"  [warn] offset {offset}: {e}")
            continue
        if not items:
            continue
        it = items[0]
        image_id = it.get("id")
        name = it.get("name", f"{image_id}.jpg")
        if not image_id:
            continue
        try:
            blob = fetch_image_bytes(api_key, image_id)
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            continue
        safe_name = name.replace("/", "_")
        out_path = args.out / safe_name
        out_path.write_bytes(blob)
        total += 1
        print(f"  [{offset:4d}] saved {safe_name}")

    print(f"\n[done] saved {total} unique images to {args.out}")


if __name__ == "__main__":
    main()
