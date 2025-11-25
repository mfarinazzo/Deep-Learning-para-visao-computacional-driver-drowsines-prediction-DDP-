from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


BASE = Path(__file__).resolve().parents[1]
RAW4_DIR = BASE / "raw" / "4" / "classification_frames"
PROCESSED_DIR = BASE / "data" / "processed"

# Map dataset-4 driver_state labels to our global classes
RAW4_STATE_TO_TARGET: Dict[str, str] = {
    "alert": "SafeDriving",
    "yawning": "SleepyDriving",
    "microsleep": "SleepyDriving",
}

TARGET_CLASSES = sorted(set(RAW4_STATE_TO_TARGET.values()))  # ["SafeDriving", "SleepyDriving"]


def load_all_annotations() -> Dict[str, dict]:
    ann_path = RAW4_DIR / "annotations_all.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations_all.json not found at {ann_path}")
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_src_path(rel_key: str) -> Path:
    # keys look like "./classification_frames/P1043xxx_720/frameNNN.jpg"
    prefix = "./classification_frames/"
    if rel_key.startswith(prefix):
        tail = rel_key[len(prefix) :]
    else:
        # Fallback: sometimes keys may already be relative from classification_frames
        tail = rel_key.lstrip("./")
        if tail.startswith("raw/4/classification_frames/"):
            tail = tail.split("classification_frames/", 1)[1]
    return RAW4_DIR / tail


def collect_grouped() -> Dict[str, List[Path]]:
    ann = load_all_annotations()
    grouped: Dict[str, List[Path]] = defaultdict(list)
    skipped = 0

    for k, v in ann.items():
        state = v.get("driver_state")
        if state not in RAW4_STATE_TO_TARGET:
            skipped += 1
            continue
        target = RAW4_STATE_TO_TARGET[state]
        src = resolve_src_path(k)
        if src.exists():
            grouped[target].append(src)
        else:
            skipped += 1

    # Ensure keys exist for all targets
    for cls in TARGET_CLASSES:
        grouped.setdefault(cls, [])

    return grouped


def split_indices(n: int, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_train = int(0.70 * n)
    n_valid = int(0.15 * n)
    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train : n_train + n_valid]
    test_idx = idxs[n_train + n_valid :]
    return train_idx, test_idx, valid_idx


def make_output_dirs() -> None:
    for split in ("train", "test", "valid"):
        for cls in TARGET_CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink(missing_ok=True)
        elif p.is_dir():
            shutil.rmtree(p)


def purge_targets() -> None:
    # Only purge the two target classes to avoid impacting other datasets/classes
    for split in ("train", "test", "valid"):
        for cls in TARGET_CLASSES:
            clear_dir(PROCESSED_DIR / split / cls)


def _unique_destination(dst_dir: Path, filename: str) -> Path:
    dst = dst_dir / filename
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}__{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def copy_files(items: List[Path], dst_dir: Path) -> Tuple[int, int]:
    copied = 0
    missing = 0
    for src in items:
        if not src.exists():
            missing += 1
            continue
        
        # ALTERAÇÃO: Prefixo raw4_
        new_name = f"raw4_{src.name}"
        
        dst = _unique_destination(dst_dir, new_name)
        shutil.copy2(src, dst)
        copied += 1
    return copied, missing


def run(seed: int, dry_run: bool, purge_target: bool) -> None:
    grouped = collect_grouped()

    # Prepare split buckets
    per_split: Dict[str, Dict[str, List[Path]]] = {
        "train": {cls: [] for cls in TARGET_CLASSES},
        "test": {cls: [] for cls in TARGET_CLASSES},
        "valid": {cls: [] for cls in TARGET_CLASSES},
    }

    # Split each target class independently to maintain class balance
    for cls, items in grouped.items():
        n = len(items)
        train_idx, test_idx, valid_idx = split_indices(n, seed)
        per_split["train"][cls] = [items[i] for i in train_idx]
        per_split["test"][cls] = [items[i] for i in test_idx]
        per_split["valid"][cls] = [items[i] for i in valid_idx]

    # Report counts
    print("raw/4 -> processed (70/15/15) preview")
    for split in ("train", "test", "valid"):
        total = sum(len(per_split[split][cls]) for cls in TARGET_CLASSES)
        print(f"- {split}: {total} files")
        for cls in TARGET_CLASSES:
            print(f"    {cls}: {len(per_split[split][cls])}")

    if dry_run:
        print("Dry-run enabled. No files will be copied.")
        return

    make_output_dirs()
    if purge_target:
        purge_targets()

    # Copy
    for split in ("train", "test", "valid"):
        for cls in TARGET_CLASSES:
            dst_dir = PROCESSED_DIR / split / cls
            copied, missing = copy_files(per_split[split][cls], dst_dir)
            print(f"Copied {copied} to {split}/{cls} (missing: {missing})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest raw/4 classification_frames into data/processed with 70/15/15 split")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    p.add_argument("--dry-run", action="store_true", help="Only show counts, do not copy")
    p.add_argument("--purge-target", action="store_true", help="Purge target class dirs before copying")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(seed=args.seed, dry_run=args.dry_run, purge_target=args.purge_target)
