#!/usr/bin/env python3
"""
Ingestor for raw/5 dataset into unified data/processed layout.
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
                continue
            fname = parts[0]
            tail = parts[-1]
            comps = tail.split(",")
            if not comps:
                continue
            label_str = comps[-1]
            try:
                label = int(label_str)
            except ValueError:
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
    split_dir = RAW5_ROOT / split
    ann_path = split_dir / "_annotations.txt"
    labels_by_file = parse_annotations_file(ann_path)

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
                except Exception:
                    pass
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
        
        # ALTERAÇÃO: Prefixo raw5_
        new_name = f"raw5_{s.src.name}"
        
        dest_path = collision_safe_dest(dest_dir, new_name, suffix_hint="raw5")
        
        if dry_run:
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
    parser.add_argument("--verify-all", action="store_true", help="Verify processed counts vs expected from raw/1..5")
    args = parser.parse_args(argv)

    if args.verify_all:
        return verify_all()

    do_dry_run = (not args.apply) or args.dry_run
    splits = args.only_splits or ["train", "valid", "test"]

    print(f"Mode: {'DRY-RUN' if do_dry_run else 'APPLY'} | Splits: {', '.join(splits)}")

    if args.purge_first:
        deleted = purge_from_manifest(dry_run=do_dry_run)
        print(f"Purged {deleted} files from previous manifest")

    all_samples: List[Sample] = []
    per_split_counts: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in splits}
    
    for split in splits:
        samples, _ = collect_samples_for_split(split)
        all_samples.extend(samples)
        for s in samples:
            per_split_counts[split][s.target_class] += 1

    # Report counts
    grand_total = 0
    classes = sorted(set(LABEL_TO_CLASS.values()))
    for split in splits:
        subtotal = sum(per_split_counts[split].values())
        grand_total += subtotal
        print(f"  {split}:")
        for cls in classes:
            print(f"    {cls:15s}: {per_split_counts[split].get(cls, 0):6d}")
        print(f"    {'TOTAL':15s}: {subtotal:6d}")

    # Execute copy
    copied = 0
    created_paths: List[Path] = []
    if all_samples:
        copied, created_paths = copy_samples(all_samples, dry_run=do_dry_run)
    print(("DRY-RUN would copy" if do_dry_run else "Copied") + f" {copied} files.")

    if not do_dry_run and created_paths:
        save_manifest(created_paths)
        print(f"Manifest written: {MANIFEST_PATH}")

    print("Done.")

# ------------------ Verification (Simplificada para manter compatibilidade) ------------------

def verify_all() -> int:
    print("Verification skipped in this streamlined version.")
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
