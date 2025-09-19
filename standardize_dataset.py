#!/usr/bin/env python3
"""
Standardize dataset file names and image sizes into a clean, uniform structure.

Inputs (default):
- src root:  data/processed
- dst root:  data/standardized

Output layout:
  data/standardized/{train|valid|test}/{DangerousDriving|Distracted|Object|SafeDriving|SleepyDriving}/
    Class_Split_000001.jpg
    Class_Split_000002.jpg
    ...

Features:
- Deterministic renaming with per-(split,class) counters
- Optional resize to a fixed WxH with modes: fit (letterbox), fill (cover+center crop), stretch
- Converts to JPEG by default (quality configurable)
- Skip-existing mode for resumable runs
- Dry-run preview and manifest CSV/JSON with mappings

Example:
  python src/standardize_dataset.py --dry-run
  python src/standardize_dataset.py --size 224 --mode fit --convert jpg --quality 95
  python src/standardize_dataset.py --size 256x256 --mode fill --convert jpg --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = PROJECT_ROOT / "data" / "processed"
DEFAULT_DST = PROJECT_ROOT / "data" / "standardized"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SPLITS = ("train", "valid", "test")
CLASSES = ("DangerousDriving", "Distracted", "Object", "SafeDriving", "SleepyDriving")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PlanItem:
    split: str
    cls: str
    src: Path
    dst: Path


def parse_size(s: str) -> Tuple[int, int]:
    if "x" in s.lower():
        w, h = s.lower().split("x", 1)
        return int(w), int(h)
    v = int(s)
    return v, v


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def iter_dataset(src_root: Path, splits: Iterable[str], classes: Iterable[str]) -> Dict[Tuple[str, str], List[Path]]:
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for split in splits:
        for cls in classes:
            d = src_root / split / cls
            files = [p for p in d.iterdir()] if d.exists() else []
            files = [p for p in files if is_image_file(p)]
            grouped[(split, cls)] = sorted(files, key=lambda p: p.name.lower())
    return grouped


def make_plan(src_root: Path, dst_root: Path, size: Optional[Tuple[int, int]], mode: str, convert: str, skip_existing: bool) -> List[PlanItem]:
    grouped = iter_dataset(src_root, SPLITS, CLASSES)
    plan: List[PlanItem] = []
    counters: Dict[Tuple[str, str], int] = { (s,c): 0 for s in SPLITS for c in CLASSES }

    ext = ".jpg" if convert.lower() == "jpg" else None  # None -> keep original
    for (split, cls), files in grouped.items():
        for src in files:
            counters[(split, cls)] += 1
            idx = counters[(split, cls)]
            base = f"{cls}_{split}_{idx:06d}"
            suffix = ext or src.suffix.lower()
            dst = dst_root / split / cls / f"{base}{suffix}"
            if skip_existing and dst.exists():
                continue
            plan.append(PlanItem(split=split, cls=cls, src=src, dst=dst))
    return plan


def resize_and_save(src: Path, dst: Path, size: Tuple[int, int], mode: str, convert: str, quality: int) -> None:
    # Import PIL lazily to allow dry-run without Pillow installed
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow (PIL) is required for resizing. Please install: pip install pillow") from e

    ensure_dir(dst.parent)
    with Image.open(src) as im:
        im = im.convert("RGB")
        target_w, target_h = size
        if mode == "stretch":
            resized = im.resize((target_w, target_h), Image.BILINEAR)
        else:
            # Preserve aspect ratio
            src_w, src_h = im.size
            scale_fit = min(target_w / src_w, target_h / src_h)
            scale_fill = max(target_w / src_w, target_h / src_h)
            if mode == "fit":
                new_w, new_h = int(src_w * scale_fit), int(src_h * scale_fit)
                im_resized = im.resize((new_w, new_h), Image.BILINEAR)
                # letterbox
                bg = Image.new("RGB", (target_w, target_h), (0, 0, 0))
                offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
                bg.paste(im_resized, offset)
                resized = bg
            elif mode == "fill":
                new_w, new_h = int(src_w * scale_fill), int(src_h * scale_fill)
                im_resized = im.resize((new_w, new_h), Image.BILINEAR)
                # center crop
                left = (new_w - target_w) // 2
                top = (new_h - target_h) // 2
                right = left + target_w
                bottom = top + target_h
                resized = im_resized.crop((left, top, right, bottom))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # Save
        if convert.lower() == "jpg":
            resized.save(dst.with_suffix(".jpg"), format="JPEG", quality=quality, optimize=True)
        else:
            # keep original extension
            resized.save(dst)


def copy_only(src: Path, dst: Path, convert: str, quality: int) -> None:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow (PIL) is required for conversion. Please install: pip install pillow") from e
    ensure_dir(dst.parent)
    with Image.open(src) as im:
        im = im.convert("RGB")
        if convert.lower() == "jpg":
            im.save(dst.with_suffix(".jpg"), format="JPEG", quality=quality, optimize=True)
        else:
            im.save(dst)


def write_reports(plan: List[PlanItem], dst_root: Path, size: Optional[Tuple[int, int]], mode: str, convert: str) -> None:
    ensure_dir(REPORTS_DIR)
    # CSV index
    csv_path = REPORTS_DIR / "standardize_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "dst_rel", "src_rel"]) 
        for it in plan:
            w.writerow([
                it.split,
                it.cls,
                str(it.dst.relative_to(dst_root)),
                str(it.src.relative_to(PROJECT_ROOT)),
            ])
    # JSON manifest
    json_path = REPORTS_DIR / "standardize_manifest.json"
    manifest = {
        "src_root": str(DEFAULT_SRC),
        "dst_root": str(dst_root),
        "count": len(plan),
        "size": None if size is None else {"width": size[0], "height": size[1]},
        "mode": mode,
        "convert": convert,
        "items": [
            {
                "split": it.split,
                "class": it.cls,
                "dst": str(it.dst.relative_to(dst_root)),
                "src": str(it.src.relative_to(PROJECT_ROOT)),
            }
            for it in plan
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Rename and resize images into a standardized dataset")
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC, help=f"Source root (default: {DEFAULT_SRC})")
    p.add_argument("--dst-root", type=Path, default=DEFAULT_DST, help=f"Destination root (default: {DEFAULT_DST})")
    p.add_argument("--size", type=str, default=None, help="Target size (e.g., 224 or 224x224). Omit to keep original size.")
    p.add_argument("--mode", choices=["fit", "fill", "stretch"], default="fit", help="Resize strategy")
    p.add_argument("--convert", choices=["jpg", "keep"], default="jpg", help="Convert to JPEG or keep original extension")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    p.add_argument("--skip-existing", action="store_true", help="Skip if destination file already exists")
    p.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    args = p.parse_args(argv)

    size = parse_size(args.size) if args.size else None
    # Build plan
    plan = make_plan(args.src_root, args.dst_root, size=size, mode=args.mode, convert=args.convert, skip_existing=args.skip_existing)

    # Report
    by_key: Dict[Tuple[str, str], int] = {}
    for it in plan:
        by_key[(it.split, it.cls)] = by_key.get((it.split, it.cls), 0) + 1
    print("Planned files per split/class:")
    total = 0
    for split in SPLITS:
        print(f"  {split}:")
        for cls in CLASSES:
            n = by_key.get((split, cls), 0)
            print(f"    {cls:15s}: {n:6d}")
            total += n
    print(f"Total planned: {total}")

    if args.dry_run:
        print("Dry-run: no files will be written.")
        return 0

    # Execute
    processed = 0
    for it in plan:
        if size is None:
            copy_only(it.src, it.dst, convert=args.convert, quality=args.quality)
        else:
            resize_and_save(it.src, it.dst, size=size, mode=args.mode, convert=args.convert, quality=args.quality)
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} files...")

    write_reports(plan, args.dst_root, size=size, mode=args.mode, convert=args.convert)
    print(f"Done. Wrote {processed} files to {args.dst_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
