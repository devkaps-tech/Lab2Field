"""Download a dataset and extract the tomato classes into data/raw/<dataset>_tomato/<label>/.

Usage:  python -m src.data.extract plantvillage
        python -m src.data.extract plantdoc

What it does:
1. Downloads the source into data/downloads/<dataset>/ (skipped if already there).
2. Finds the folders listed in params.yaml -> data.<dataset>.class_map.
3. Checks every image opens, drops broken files, drops exact duplicates.
4. Copies images into data/raw/<dataset>_tomato/<label>/, named by content hash.
5. Writes a report to reports/extract_<dataset>.json.

The download folder is NOT tracked by DVC. The output folder is.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOWNLOADS = Path("data/downloads")
RAW = Path("data/raw")
REPORTS = Path("reports")


def norm(name: str) -> str:
    """Compare folder names ignoring case and extra spaces."""
    return " ".join(name.lower().split())


def download(source: dict, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"[download] {dest} already exists, skipping download")
        return
    dest.mkdir(parents=True, exist_ok=True)

    if source["type"] == "kaggle":
        from kaggle.api.kaggle_api_extended import KaggleApi  # needs your Kaggle API token

        api = KaggleApi()
        api.authenticate()
        print(f"[download] Kaggle dataset {source['id']} -> {dest}")
        api.dataset_download_files(source["id"], path=str(dest), unzip=True, quiet=False)
    elif source["type"] == "git":
        print(f"[download] git clone {source['id']} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", source["id"], str(dest)], check=True)
        shutil.rmtree(dest / ".git", ignore_errors=True)
    else:
        sys.exit(f"Unknown source type: {source['type']}")


def find_class_dirs(root: Path, class_map: dict, subdir: str | None) -> dict[str, list[Path]]:
    """Return {label: [folders]} for every folder whose name is in class_map."""
    lookup = {norm(k): v for k, v in class_map.items()}
    found = defaultdict(list)
    unmatched_tomato = set()

    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        rel_parts = [norm(p) for p in d.relative_to(root).parts[:-1]]
        if subdir is not None and norm(subdir) not in rel_parts:
            continue
        key = norm(d.name)
        if key in lookup:
            found[lookup[key]].append(d)
        elif "tomato" in key:
            unmatched_tomato.add(d.name)

    missing = sorted(set(lookup.values()) - set(found))
    if missing:
        print(f"[warning] no folder found for: {missing}")
    if unmatched_tomato:
        print(f"[warning] tomato folders not in class_map (ignored): {sorted(unmatched_tomato)}")
    for label, dirs in found.items():
        if len(dirs) > 1:
            print(f"[info] {label}: combining {len(dirs)} folders -> {[str(x.relative_to(root)) for x in dirs]}")
    return found


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def extract(dataset: str) -> None:
    params = yaml.safe_load(open("params.yaml"))["data"][dataset]
    dl_dir = DOWNLOADS / dataset
    out_dir = RAW / f"{dataset}_tomato"

    download(params["source"], dl_dir)
    class_dirs = find_class_dirs(dl_dir, params["class_map"], params.get("subdir"))
    if not class_dirs:
        sys.exit("No matching class folders found. Check class_map and subdir in params.yaml.")

    if out_dir.exists():
        shutil.rmtree(out_dir)  # always rebuild from scratch so the output is reproducible

    counts, broken, dupes = Counter(), Counter(), Counter()
    seen = {}  # content hash -> label
    conflicts = []

    for label, dirs in sorted(class_dirs.items()):
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for d in dirs:
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTS:
                    continue
                if not is_valid_image(f):
                    broken[label] += 1
                    continue
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h in seen:
                    dupes[label] += 1
                    if seen[h] != label:
                        conflicts.append((h, seen[h], label))
                    continue
                seen[h] = label
                shutil.copy2(f, out_dir / label / f"{h}{f.suffix.lower()}")
                counts[label] += 1

    report = {
        "dataset": dataset,
        "total_images": sum(counts.values()),
        "per_class": dict(sorted(counts.items())),
        "broken_skipped": dict(broken),
        "duplicates_skipped": dict(dupes),
        "same_image_in_two_classes": len(conflicts),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"extract_{dataset}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"plantvillage", "plantdoc"}:
        sys.exit("Usage: python -m src.data.extract [plantvillage|plantdoc]")
    extract(sys.argv[1])
