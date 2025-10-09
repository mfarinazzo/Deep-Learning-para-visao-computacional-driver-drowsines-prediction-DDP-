#!/usr/bin/env python3
"""
Ingestor for raw/5 dataset into unified data/processed layout.

Expected source structure:
  raw/5/
    train/
      *.jpg
      _annotations.txt
      _classes.txt  # optional, informative only
    valid/
      ...
    test/
      ...

Annotation file format (one per line):
  <image_filename>.jpg x1,y1,x2,y2,label

We only need the final numeric label (0..5). Mapping to global classes:
  0 -> DangerousDriving
  1 -> Distracted
  2 -> Object          # Drinking
  3 -> SafeDriving
  4 -> SleepyDriving
  5 -> SleepyDriving   # Yawn

Copies the images into:
  data/processed/{train|valid|test}/{DangerousDriving|Distracted|Object|SafeDriving|SleepyDriving}

Features:
- Dry run (counts only)
- Collision-safe copying (suffixes if needed)
- Optional manifest-based purge (deletes files created by previous runs of this script)

Usage examples:
  python src/ingest_raw5_to_processed.py --dry-run
  python src/ingest_raw5_to_processed.py --apply
  python src/ingest_raw5_to_processed.py --apply --purge-first
  python src/ingest_raw5_to_processed.py --only-splits train valid --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW5_ROOT = PROJECT_ROOT / "raw" / "5"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "reports" / "ingest_raw5_manifest.json"


LABEL_TO_CLASS = {
    0: "DangerousDriving",
    1: "Distracted",
    2: "Object",  # Drinking
    3: "SafeDriving",
    4: "SleepyDriving",
    5: "SleepyDriving",  # Yawn
}

# Global class list
ALL_CLASSES = ["DangerousDriving", "Distracted", "Object", "SafeDriving", "SleepyDriving"]


@dataclass(frozen=True)
class Sample:
    split: str
    src: Path
    target_class: str


def parse_annotations_file(txt_path: Path) -> Dict[str, int]:
    """Parse Roboflow-style _annotations.txt into a mapping: filename -> label_int.

    Each line is expected as: "<file> x1,y1,x2,y2,label". We use only the last integer.
    If a filename appears multiple times, the last label wins (and a conflict is counted elsewhere).
    """
    mapping: Dict[str, int] = {}
    if not txt_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {txt_path}")
    with txt_path.open("r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"Warning: malformed line {i} in {txt_path.name}: '{line}'", file=sys.stderr)
                continue
            fname = parts[0]
            tail = parts[-1]
            comps = tail.split(",")
            if not comps:
                print(f"Warning: cannot parse bbox/label on line {i} in {txt_path.name}", file=sys.stderr)
                continue
            label_str = comps[-1]
            try:
                label = int(label_str)
            except ValueError:
                print(f"Warning: non-integer label on line {i} in {txt_path.name}: '{label_str}'", file=sys.stderr)
                continue
            mapping[fname] = label
    return mapping


def collision_safe_dest(dest_dir: Path, filename: str, suffix_hint: str = "raw5") -> Path:
    """Return a destination path that doesn't overwrite existing files by adding a suffix if needed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    base = Path(filename).name
    stem = Path(base).stem
    ext = Path(base).suffix
    candidate = dest_dir / base
    if not candidate.exists():
        return candidate
    # Add incrementing suffix
    n = 1
    while True:
        candidate = dest_dir / f"{stem}__{suffix_hint}_{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def collect_samples_for_split(split: str, only_existing: bool = True) -> Tuple[List[Sample], Dict[str, int]]:
    """Collect samples for a given split using the _annotations.txt file.

    Returns (samples, conflict_counts) where conflict_counts contains the number of times
    the same filename was assigned multiple distinct labels (best-effort detection).
    """
    split_dir = RAW5_ROOT / split
    ann_path = split_dir / "_annotations.txt"
    labels_by_file = parse_annotations_file(ann_path)

    # Detect conflicts by checking duplicates in file (we only retained last label)
    # Basic heuristic: if there are duplicate filenames in annotations file with differing labels,
    # they will manifest only if we track seen labels. We'll do a second pass to count conflicts.
    conflicts = 0
    seen_labels: Dict[str, int] = {}
    with ann_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fname = parts[0]
            comps = parts[-1].split(",")
            if not comps:
                continue
            try:
                label = int(comps[-1])
            except ValueError:
                continue
            if fname in seen_labels and seen_labels[fname] != label:
                conflicts += 1
            seen_labels[fname] = label

    samples: List[Sample] = []
    missing = 0
    for fname, label in labels_by_file.items():
        src_path = split_dir / fname
        if only_existing and not src_path.exists():
            missing += 1
            continue
        target_class = LABEL_TO_CLASS.get(label)
        if target_class is None:
            print(f"Warning: unknown label {label} for {fname} in split {split}", file=sys.stderr)
            continue
        samples.append(Sample(split=split, src=src_path, target_class=target_class))

    if missing:
        print(f"Note: {missing} annotated files not found on disk in split '{split}' (skipped)")
    return samples, {"conflicts": conflicts}


def load_manifest() -> List[str]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("files", []) or []
    except Exception:
        return []


def save_manifest(files: Iterable[Path]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    rels = [str(p.relative_to(PROCESSED_ROOT)) for p in files]
    payload = {
        "script": Path(__file__).name,
        "root": str(PROCESSED_ROOT),
        "files": rels,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def purge_from_manifest(dry_run: bool = False) -> int:
    rels = load_manifest()
    if not rels:
        print("No manifest entries to purge.")
        return 0
    deleted = 0
    for rel in rels:
        path = PROCESSED_ROOT / rel
        if path.exists():
            if dry_run:
                print(f"DRY-RUN purge: would delete {path}")
            else:
                try:
                    path.unlink()
                except Exception as e:
                    print(f"Warning: failed to delete {path}: {e}")
                else:
                    deleted += 1
    if not dry_run:
        try:
            MANIFEST_PATH.unlink()
        except Exception:
            pass
    return deleted


def copy_samples(samples: List[Sample], dry_run: bool = True) -> Tuple[int, List[Path]]:
    copied = 0
    created: List[Path] = []
    for s in samples:
        dest_dir = PROCESSED_ROOT / s.split / s.target_class
        dest_path = collision_safe_dest(dest_dir, s.src.name, suffix_hint="raw5")
        if dry_run:
            # Don't actually copy, but count and show a few examples
            copied += 1
            if copied <= 5:
                print(f"DRY-RUN copy: {s.src} -> {dest_path}")
            continue
        copy2(s.src, dest_path)
        created.append(dest_path)
        copied += 1
        if copied % 1000 == 0:
            print(f"Copied {copied} files so far...")
    return copied, created


def main(argv: List[str]):
    parser = argparse.ArgumentParser(description="Ingest raw/5 dataset into data/processed")
    parser.add_argument("--apply", action="store_true", help="Perform the copy (otherwise dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Alias for not using --apply")
    parser.add_argument("--purge-first", action="store_true", help="Purge files from previous manifest before ingesting")
    parser.add_argument("--only-splits", nargs="*", choices=["train", "valid", "test"], help="Restrict to specific splits")
    parser.add_argument("--verify-all", action="store_true", help="Verify processed counts vs expected from raw/1..5 (70/15/15 for 1-4; provided splits for 5)")
    args = parser.parse_args(argv)

    if args.verify_all:
        return verify_all()

    do_dry_run = (not args.apply) or args.dry_run
    splits = args.only_splits or ["train", "valid", "test"]

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source root:  {RAW5_ROOT}")
    print(f"Target root:  {PROCESSED_ROOT}")
    print(f"Mode: {'DRY-RUN' if do_dry_run else 'APPLY'} | Splits: {', '.join(splits)}")

    if args.purge_first:
        deleted = purge_from_manifest(dry_run=do_dry_run)
        print(("DRY-RUN purge would delete" if do_dry_run else "Purged") + f" {deleted} files from previous manifest")

    all_samples: List[Sample] = []
    per_split_counts: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in splits}
    conflict_total = 0

    for split in splits:
        samples, conflict_counts = collect_samples_for_split(split)
        conflict_total += conflict_counts.get("conflicts", 0)
        all_samples.extend(samples)
        for s in samples:
            per_split_counts[split][s.target_class] += 1

    # Report counts
    print("\nPlanned copy counts per split/class:")
    grand_total = 0
    classes = sorted(set(LABEL_TO_CLASS.values()))
    for split in splits:
        subtotal = sum(per_split_counts[split].values())
        grand_total += subtotal
        print(f"  {split}:")
        for cls in classes:
            print(f"    {cls:15s}: {per_split_counts[split].get(cls, 0):6d}")
        print(f"    {'TOTAL':15s}: {subtotal:6d}")
    print(f"Grand total planned: {grand_total}")
    if conflict_total:
        print(f"Note: detected {conflict_total} duplicate filename label conflicts across annotations (last label wins)")

    # Execute copy
    copied = 0
    created_paths: List[Path] = []
    if all_samples:
        copied, created_paths = copy_samples(all_samples, dry_run=do_dry_run)
    print(("DRY-RUN would copy" if do_dry_run else "Copied") + f" {copied} files.")

    # Save manifest only on successful apply
    if not do_dry_run and created_paths:
        save_manifest(created_paths)
        print(f"Manifest written: {MANIFEST_PATH}")

    print("Done.")
    

# ------------------ Verification across raw/1..5 ------------------

# Helpers reused for verification
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def _list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if _is_image(p)]

def _counts_70_15_15(n: int) -> Tuple[int, int, int]:
    n_train = int(0.70 * n)
    n_test = int(0.15 * n)
    n_valid = n - n_train - n_test
    return n_train, n_test, n_valid

def _verify_raw1(expected: Dict[str, Dict[str, int]]):
    # raw/1 mapping matches split_raw1_to_processed.py
    mapping = {
        "safe_driving": "SafeDriving",
        "turning": "SafeDriving",
        "talking_phone": "DangerousDriving",
        "texting_phone": "DangerousDriving",
    }
    raw1_root = PROJECT_ROOT / "raw" / "1" / "Multi-Class Driver Behavior Image Dataset"
    per_class_counts: Dict[str, int] = {"SafeDriving": 0, "DangerousDriving": 0}
    for src_folder, target in mapping.items():
        imgs = _list_images(raw1_root / src_folder)
        per_class_counts[target] += len(imgs)
    for cls, n in per_class_counts.items():
        tr, te, va = _counts_70_15_15(n)
        expected["train"][cls] += tr
        expected["test"][cls] += te
        expected["valid"][cls] += va

def _verify_raw2(expected: Dict[str, Dict[str, int]]):
    # raw/2 mapping matches ingest_raw2_to_processed.py
    raw2_map = {
        "c0": "SafeDriving",
        "c1": "DangerousDriving",
        "c2": "DangerousDriving",
        "c3": "DangerousDriving",
        "c4": "DangerousDriving",
        "c5": "Distracted",
        "c6": "Object",
        "c7": "Distracted",
        "c8": "Distracted",
        "c9": "Distracted",
    }
    raw2_train_root = PROJECT_ROOT / "raw" / "2" / "imgs" / "train"
    per_class_counts: Dict[str, int] = {c: 0 for c in ALL_CLASSES}
    if raw2_train_root.exists():
        for d in sorted(p for p in raw2_train_root.iterdir() if p.is_dir()):
            tgt = raw2_map.get(d.name)
            if not tgt:
                continue
            per_class_counts[tgt] += len(_list_images(d))
    # Apply 70/15/15 per class
    for cls, n in per_class_counts.items():
        if n == 0:
            continue
        tr, te, va = _counts_70_15_15(n)
        expected["train"][cls] += tr
        expected["test"][cls] += te
        expected["valid"][cls] += va

def _verify_raw3(expected: Dict[str, Dict[str, int]]):
    # raw/3 mapping matches ingest_raw3_to_processed.py
    mapping = {
        "Active Subjects": "SafeDriving",
        "Fatigue Subjects": "SleepyDriving",
    }
    raw3_root = PROJECT_ROOT / "raw" / "3" / "0 FaceImages"
    per_class_counts: Dict[str, int] = {"SafeDriving": 0, "SleepyDriving": 0}
    for src_folder, target in mapping.items():
        per_class_counts[target] += len(_list_images(raw3_root / src_folder))
    for cls, n in per_class_counts.items():
        tr, te, va = _counts_70_15_15(n)
        expected["train"][cls] += tr
        expected["test"][cls] += te
        expected["valid"][cls] += va

def _verify_raw4(expected: Dict[str, Dict[str, int]]):
    # raw/4 mapping matches ingest_raw4_to_processed.py
    raw4_map = {
        "alert": "SafeDriving",
        "yawning": "SleepyDriving",
        "microsleep": "SleepyDriving",
    }
    ann_path = PROJECT_ROOT / "raw" / "4" / "classification_frames" / "annotations_all.json"
    if not ann_path.exists():
        print(f"Warning: {ann_path} not found. Skipping raw/4 in verification.")
        return
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: failed to read {ann_path}: {e}. Skipping raw/4 in verification.")
        return
    per_class_counts: Dict[str, int] = {"SafeDriving": 0, "SleepyDriving": 0}
    for _, v in data.items():
        state = v.get("driver_state")
        tgt = raw4_map.get(state)
        if tgt:
            per_class_counts[tgt] += 1
    for cls, n in per_class_counts.items():
        tr, te, va = _counts_70_15_15(n)
        expected["train"][cls] += tr
        expected["test"][cls] += te
        expected["valid"][cls] += va

def _verify_raw5(expected: Dict[str, Dict[str, int]]):
    # For raw/5 we honor provided splits; reuse parsing here
    for split in ("train", "valid", "test"):
        samples, _ = collect_samples_for_split(split)
        for s in samples:
            expected[split][s.target_class] += 1

def _actual_processed_counts() -> Dict[str, Dict[str, int]]:
    actual: Dict[str, Dict[str, int]] = {"train": {c: 0 for c in ALL_CLASSES},
                                          "valid": {c: 0 for c in ALL_CLASSES},
                                          "test": {c: 0 for c in ALL_CLASSES}}
    for split in ("train", "valid", "test"):
        for cls in ALL_CLASSES:
            dirp = PROCESSED_ROOT / split / cls
            if dirp.exists():
                actual[split][cls] = len([p for p in dirp.iterdir() if _is_image(p)])
    return actual

def verify_all() -> int:
    print("Verifying totals across raw/1..5 vs data/processed (70/15/15 for raws 1-4; raw/5 uses provided splits)...")
    expected: Dict[str, Dict[str, int]] = {
        "train": {c: 0 for c in ALL_CLASSES},
        "valid": {c: 0 for c in ALL_CLASSES},
        "test": {c: 0 for c in ALL_CLASSES},
    }
    _verify_raw1(expected)
    _verify_raw2(expected)
    _verify_raw3(expected)
    _verify_raw4(expected)
    _verify_raw5(expected)

    actual = _actual_processed_counts()

    # Summaries
    def _sum_counts(d: Dict[str, Dict[str, int]]) -> int:
        return sum(v for split in d.values() for v in split.values())

    print("\nExpected vs Actual per split/class:")
    ok = True
    for split in ("train", "valid", "test"):
        print(f"  {split}:")
        for cls in ALL_CLASSES:
            e = expected[split][cls]
            a = actual[split][cls]
            status = "OK" if e == a else "MISMATCH"
            print(f"    {cls:15s}: expected={e:6d} | actual={a:6d}  {status}")
            if e != a:
                ok = False
        e_total = sum(expected[split].values())
        a_total = sum(actual[split].values())
        status = "OK" if e_total == a_total else "MISMATCH"
        print(f"    {'TOTAL':15s}: expected={e_total:6d} | actual={a_total:6d}  {status}")
        if e_total != a_total:
            ok = False

    e_grand = _sum_counts(expected)
    a_grand = _sum_counts(actual)
    print(f"\nGrand totals: expected={e_grand} | actual={a_grand} | {'OK' if e_grand == a_grand else 'MISMATCH'}")

    print("\nResult:")
    if ok:
        print("PASS: processed structure matches expected totals from raws 1..5.")
        return 0
    else:
        print("FAIL: discrepancies found. See details above.")
        return 1

if __name__ == "__main__":
    main(sys.argv[1:])
