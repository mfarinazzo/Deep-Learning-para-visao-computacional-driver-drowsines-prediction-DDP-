#!/usr/bin/env python3
"""
Generic verification for data/processed.

Compares actual counts under data/processed/{train,valid,test}/{Class}
against expected counts derived from raw/1..5 with the agreed mappings:

- raw/1 (Multi-Class Driver Behavior Image Dataset):
    safe_driving, turning -> SafeDriving
    talking_phone, texting_phone -> DangerousDriving
    ignore other_activities
    split 70/15/15 per class (seeded shuffle)

- raw/2 (StateFarm-like c0..c9):
    c0 -> SafeDriving
    c1,c2,c3,c4 -> DangerousDriving
    c5,c7,c8,c9 -> Distracted
    c6 -> Object
    split 70/15/15 per class (seeded shuffle)

- raw/3 (0 FaceImages):
    Active Subjects -> SafeDriving
    Fatigue Subjects -> SleepyDriving
    split 70/15/15 per class (seeded shuffle)

- raw/4 (classification_frames + annotations_all.json):
    alert -> SafeDriving
    yawning, microsleep -> SleepyDriving
    split 70/15/15 per class (seeded shuffle)

- raw/5 (Roboflow-style):
    Provided splits train/valid/test with numeric class ids:
    0->DangerousDriving, 1->Distracted, 2->Object, 3->SafeDriving, 4->SleepyDriving, 5->SleepyDriving
    For raw/5 we DO NOT resplit; we respect provided splits.

Notes:
- Small +/-1 differences can occur due to floor/ceil when splitting; we use floor for train and valid, remainder to test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_ROOT = PROJECT_ROOT / "raw"

SPLITS = ("train", "valid", "test")
CLASSES = ("DangerousDriving", "Distracted", "Object", "SafeDriving", "SleepyDriving")


def count_processed(splits: Iterable[str], classes: Iterable[str]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in classes} for s in splits}
    for s in splits:
        for c in classes:
            d = PROCESSED_ROOT / s / c
            if not d.exists():
                continue
            counts[s][c] = sum(1 for p in d.iterdir() if p.is_file())
    return counts


def split_counts(n: int) -> Tuple[int, int, int]:
    train = floor(0.7 * n)
    valid = floor(0.15 * n)
    test = n - train - valid
    return train, valid, test


def assign_split(names: List[str], seed: int = 42) -> Dict[str, int]:
    names = list(names)
    rnd = random.Random(seed)
    rnd.shuffle(names)
    n = len(names)
    n_tr, n_va, n_te = split_counts(n)
    idx = {"train": set(names[:n_tr]), "valid": set(names[n_tr:n_tr + n_va]), "test": set(names[n_tr + n_va:])}
    # Return a mapping name->split index 0/1/2
    inv = {}
    for nm in names[:n_tr]:
        inv[nm] = 0
    for nm in names[n_tr:n_tr + n_va]:
        inv[nm] = 1
    for nm in names[n_tr + n_va:]:
        inv[nm] = 2
    return inv


def expected_from_raw1(seed: int) -> Dict[str, Dict[str, int]]:
    base = RAW_ROOT / "1" / "Multi-Class Driver Behavior Image Dataset"
    mapping = {
        "safe_driving": "SafeDriving",
        "turning": "SafeDriving",
        "talking_phone": "DangerousDriving",
        "texting_phone": "DangerousDriving",
    }
    per_class_names = defaultdict(list)  # class -> [filenames]
    for sub, cls in mapping.items():
        d = base / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                per_class_names[cls].append(p.name)
    exp = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for cls, names in per_class_names.items():
        inv = assign_split(names, seed)
        counts = [0, 0, 0]
        for v in inv.values():
            counts[v] += 1
        exp["train"][cls] += counts[0]
        exp["valid"][cls] += counts[1]
        exp["test"][cls] += counts[2]
    return exp


def expected_from_raw2(seed: int) -> Dict[str, Dict[str, int]]:
    # Map c0..c9 to global classes
    label_map = {
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
    csv_path = RAW_ROOT / "2" / "driver_imgs_list.csv"
    per_class_names = defaultdict(list)
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = label_map.get(row.get("classname") or row.get("class"))
                img = row.get("img")
                if cls and img:
                    per_class_names[cls].append(img)
    else:
        # Fallback: enumerate directories
        base = RAW_ROOT / "2" / "imgs" / "train"
        if base.exists():
            for cdir in base.iterdir():
                if not cdir.is_dir():
                    continue
                cls = label_map.get(cdir.name)
                if not cls:
                    continue
                for p in cdir.iterdir():
                    if p.is_file():
                        per_class_names[cls].append(p.name)
    exp = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for cls, names in per_class_names.items():
        inv = assign_split(names, seed)
        counts = [0, 0, 0]
        for v in inv.values():
            counts[v] += 1
        exp["train"][cls] += counts[0]
        exp["valid"][cls] += counts[1]
        exp["test"][cls] += counts[2]
    return exp


def expected_from_raw3(seed: int) -> Dict[str, Dict[str, int]]:
    base = RAW_ROOT / "3" / "0 FaceImages"
    mapping = {
        "Active Subjects": "SafeDriving",
        "Fatigue Subjects": "SleepyDriving",
    }
    per_class_names = defaultdict(list)
    for sub, cls in mapping.items():
        d = base / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                per_class_names[cls].append(p.name)
    exp = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for cls, names in per_class_names.items():
        inv = assign_split(names, seed)
        counts = [0, 0, 0]
        for v in inv.values():
            counts[v] += 1
        exp["train"][cls] += counts[0]
        exp["valid"][cls] += counts[1]
        exp["test"][cls] += counts[2]
    return exp


def expected_from_raw4(seed: int) -> Dict[str, Dict[str, int]]:
    base = RAW_ROOT / "4" / "classification_frames"
    ann = base / "annotations_all.json"
    per_class_names = defaultdict(list)
    if ann.exists():
        data = json.loads(ann.read_text(encoding="utf-8"))
        for k, v in data.items():
            state = (v.get("driver_state") or v.get("label") or "").lower()
            if state == "alert":
                cls = "SafeDriving"
            elif state in ("yawning", "microsleep"):
                cls = "SleepyDriving"
            else:
                continue
            fname = Path(k).name
            per_class_names[cls].append(fname)
    else:
        # Fallback: cannot infer labels, so skip
        pass
    exp = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for cls, names in per_class_names.items():
        inv = assign_split(names, seed)
        counts = [0, 0, 0]
        for v in inv.values():
            counts[v] += 1
        exp["train"][cls] += counts[0]
        exp["valid"][cls] += counts[1]
        exp["test"][cls] += counts[2]
    return exp


def expected_from_raw5() -> Dict[str, Dict[str, int]]:
    base = RAW_ROOT / "5"
    id_to_class = {
        0: "DangerousDriving",
        1: "Distracted",
        2: "Object",
        3: "SafeDriving",
        4: "SleepyDriving",
        5: "SleepyDriving",
    }
    exp = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for split in SPLITS:
        ann = base / split / "_annotations.txt"
        if not ann.exists():
            continue
        for line in ann.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(parts[-1])
            except Exception:
                continue
            cls = id_to_class.get(cls_id)
            if cls:
                exp[split][cls] += 1
    return exp


def sum_dicts(dicts: List[Dict[str, Dict[str, int]]]) -> Dict[str, Dict[str, int]]:
    out = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    for d in dicts:
        for s in SPLITS:
            for c in CLASSES:
                out[s][c] += d.get(s, {}).get(c, 0)
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify data/processed counts against expectations from raw datasets")
    ap.add_argument("--seed", type=int, default=42, help="Seed used for 70/15/15 splits on raws 1-4")
    ap.add_argument("--include-raws", nargs="*", type=int, choices=[1,2,3,4,5], default=[1,2,3,4,5], help="Which raw datasets to include in expected counts")
    ap.add_argument("--splits", nargs="*", choices=SPLITS, default=list(SPLITS))
    ap.add_argument("--classes", nargs="*", choices=CLASSES, default=list(CLASSES))
    args = ap.parse_args(argv)

    splits = args.splits
    classes = args.classes

    actual = count_processed(splits, classes)

    exp_parts: List[Dict[str, Dict[str, int]]] = []
    if 1 in args.include_raws:
        exp_parts.append(expected_from_raw1(args.seed))
    if 2 in args.include_raws:
        exp_parts.append(expected_from_raw2(args.seed))
    if 3 in args.include_raws:
        exp_parts.append(expected_from_raw3(args.seed))
    if 4 in args.include_raws:
        exp_parts.append(expected_from_raw4(args.seed))
    if 5 in args.include_raws:
        exp_parts.append(expected_from_raw5())

    expected = sum_dicts(exp_parts)

    # Print summary and deltas
    total_act = 0
    total_exp = 0
    print("Split/Class counts (Actual vs Expected | Delta):")
    for s in splits:
        print(f"\n[{s}]")
        for c in classes:
            a = actual[s][c]
            e = expected[s][c]
            d = a - e
            total_act += a
            total_exp += e
            print(f"  {c:16s}: {a:6d} vs {e:6d} | {d:+5d}")

    print("\nTotals:")
    print(f"  Actual:   {total_act}")
    print(f"  Expected: {total_exp}")
    print(f"  Delta:    {total_act - total_exp:+d}")

    ok = actual == expected
    if ok:
        print("\nPASS: Actual counts match expected counts exactly.")
        return 0
    else:
        print("\nMISMATCH: Some counts differ. This can happen if ingestion was re-run with a different seed or partial additions.")
        return 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
