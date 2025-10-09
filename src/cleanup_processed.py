#!/usr/bin/env python3
"""
Generic cleanup tool for data/processed.

Supports two precise modes:
- By manifest(s): delete exactly the files listed in JSON manifests (as produced by our ingestors/standardizer).
- By raw id(s): derive source basenames from raw/{1..5} and delete processed files whose basenames match.

Safety:
- Default is list-only (dry). Use --apply to actually delete.
- You can restrict to specific splits/classes.

Examples:
  python src/cleanup_processed.py --by-manifest outputs/reports/ingest_raw5_manifest.json --list-only
  python src/cleanup_processed.py --raw 3 --splits train valid --apply
  python src/cleanup_processed.py --raw 1 2 --classes SafeDriving --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
RAW_ROOT = PROJECT_ROOT / "raw"

SPLITS = ("train", "valid", "test")
CLASSES = ("DangerousDriving", "Distracted", "Object", "SafeDriving", "SleepyDriving")


def ensure_iter(obj) -> List:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]


def list_processed_targets(splits: Iterable[str], classes: Iterable[str]) -> List[Path]:
    roots: List[Path] = []
    for split in splits:
        for cls in classes:
            roots.append(PROCESSED_ROOT / split / cls)
    return roots


def load_manifest_paths(manifest_path: Path) -> List[Path]:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    files = data.get("files") or []
    paths: List[Path] = []
    for rel in files:
        p = PROCESSED_ROOT / rel
        paths.append(p)
    return paths


def basenames_from_raw(raw_id: int) -> Set[str]:
    """Collect a set of source basenames for a given raw id by scanning its source folders/annotations.
    This uses best-effort heuristics based on current raw layouts.
    """
    names: Set[str] = set()
    if raw_id == 1:
        base = RAW_ROOT / "1" / "Multi-Class Driver Behavior Image Dataset"
        for sub in ("safe_driving", "turning", "talking_phone", "texting_phone"):
            d = base / sub
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        names.add(p.name)
    elif raw_id == 2:
        base = RAW_ROOT / "2" / "imgs" / "train"
        if base.exists():
            for cdir in base.iterdir():
                if cdir.is_dir():
                    for p in cdir.iterdir():
                        if p.is_file():
                            names.add(p.name)
    elif raw_id == 3:
        base = RAW_ROOT / "3" / "0 FaceImages"
        for sub in ("Active Subjects", "Fatigue Subjects"):
            d = base / sub
            if d.exists():
                for p in d.iterdir():
                    if p.is_file():
                        names.add(p.name)
    elif raw_id == 4:
        base = RAW_ROOT / "4" / "classification_frames"
        # gather all frames filenames across subfolders listed in annotations JSONs if present
        ann_all = base / "annotations_all.json"
        if ann_all.exists():
            try:
                data = json.loads(ann_all.read_text(encoding="utf-8"))
                for k in data.keys():
                    # keys look like ./classification_frames/<folder>/frame.jpg
                    fname = Path(k).name
                    if fname:
                        names.add(fname)
            except Exception:
                pass
        # fallback: scan frame folders
        for d in base.iterdir():
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file():
                        names.add(p.name)
    elif raw_id == 5:
        base = RAW_ROOT / "5"
        for split in ("train", "valid", "test"):
            ann = base / split / "_annotations.txt"
            if ann.exists():
                try:
                    for line in ann.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        fname = parts[0]
                        names.add(Path(fname).name)
                except Exception:
                    continue
    return names


def find_matching_processed(basenames: Set[str], splits: Iterable[str], classes: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    targets = list_processed_targets(splits, classes)
    for root in targets:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_file() and p.name in basenames:
                matches.append(p)
    return matches


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cleanup data/processed files generically (by manifest or by raw id)")
    ap.add_argument("--by-manifest", nargs="*", type=Path, help="Manifest JSON paths listing files to delete (relative to data/processed)")
    ap.add_argument("--raw", nargs="*", type=int, choices=[1,2,3,4,5], help="Raw dataset ids to clean (1..5)")
    ap.add_argument("--splits", nargs="*", choices=SPLITS, default=list(SPLITS), help="Restrict to these splits")
    ap.add_argument("--classes", nargs="*", choices=CLASSES, default=list(CLASSES), help="Restrict to these classes")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (default is list-only)")
    ap.add_argument("--list-only", action="store_true", help="List matches only (default)")
    args = ap.parse_args(argv)

    splits = args.splits
    classes = args.classes

    planned: List[Path] = []

    # From manifests
    for mpath in ensure_iter(args.by_manifest):
        if not mpath:
            continue
        if not mpath.exists():
            print(f"Warning: manifest not found: {mpath}")
            continue
        planned.extend(load_manifest_paths(mpath))

    # From raw ids
    for rid in ensure_iter(args.raw):
        if not rid:
            continue
        base_names = basenames_from_raw(int(rid))
        if not base_names:
            print(f"Note: no basenames derived for raw/{rid}")
            continue
        matches = find_matching_processed(base_names, splits, classes)
        planned.extend(matches)

    # De-duplicate
    uniq = []
    seen = set()
    for p in planned:
        try:
            key = p.resolve()
        except Exception:
            key = p
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    print(f"Found {len(uniq)} files matching criteria under data/processed.")
    for p in uniq[:20]:
        print(f"  {p}")
    if len(uniq) > 20:
        print(f"  ... (+{len(uniq)-20} more)")

    if not args.apply:
        print("List-only mode (use --apply to delete). Nothing was deleted.")
        return 0

    deleted = 0
    for p in uniq:
        try:
            if p.exists():
                p.unlink()
                deleted += 1
        except Exception as e:
            print(f"Warning: failed to delete {p}: {e}")

    print(f"Deleted {deleted} files.")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
