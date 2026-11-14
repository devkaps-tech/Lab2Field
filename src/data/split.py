"""Build manifest CSVs (image_path, label) instead of copying images again.

Usage:  python -m src.data.split lab     -> data/processed/lab/{train,val,test}.csv
        python -m src.data.split field   -> data/processed/field/replay.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

REPORTS = Path("reports")


def list_images(root: Path) -> pd.DataFrame:
    rows = [
        {"image_path": str(f.as_posix()), "label": f.parent.name}
        for f in sorted(root.glob("*/*"))
        if f.is_file()
    ]
    if not rows:
        sys.exit(f"No images found in {root}. Run the extract stage first.")
    return pd.DataFrame(rows)


def write_report(name: str, frames: dict[str, pd.DataFrame]) -> None:
    report = {
        split: {"total": len(df), "per_class": df["label"].value_counts().sort_index().to_dict()}
        for split, df in frames.items()
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def split_lab() -> None:
    p = yaml.safe_load(open("params.yaml"))["split"]
    df = list_images(Path("data/raw/plantvillage_tomato"))

    # First take the test set, then split the rest into train and val.
    # stratify keeps the class proportions the same in every split.
    rest, test = train_test_split(
        df, test_size=p["test_size"], stratify=df["label"], random_state=p["seed"]
    )
    val_share = p["val_size"] / (1 - p["test_size"])
    train, val = train_test_split(
        rest, test_size=val_share, stratify=rest["label"], random_state=p["seed"]
    )

    out = Path("data/processed/lab")
    out.mkdir(parents=True, exist_ok=True)
    frames = {"train": train, "val": val, "test": test}
    for name, frame in frames.items():
        frame.sort_values("image_path").to_csv(out / f"{name}.csv", index=False)
    write_report("split_lab", frames)


def prepare_field() -> None:
    df = list_images(Path("data/raw/plantdoc_tomato"))
    out = Path("data/processed/field")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "replay.csv", index=False)
    write_report("prepare_field", {"replay": df})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"lab", "field"}:
        sys.exit("Usage: python -m src.data.split [lab|field]")
    split_lab() if sys.argv[1] == "lab" else prepare_field()
