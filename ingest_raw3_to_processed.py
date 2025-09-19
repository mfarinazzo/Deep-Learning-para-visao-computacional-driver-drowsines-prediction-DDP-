from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# raw/3 source folders -> target labels
RAW3_SOURCE_TO_TARGET: Dict[str, str] = {
    "Active Subjects": "SafeDriving",
    "Fatigue Subjects": "SleepyDriving",
}

RATIOS = (0.70, 0.15, 0.15)  # train, test, valid


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if is_image(p)]


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dest_dir / filename
    k = 1
    while candidate.exists():
        candidate = dest_dir / f"{base}__{k}{suffix}"
        k += 1
    return candidate


def copy_files(files: Iterable[Path], dest_dir: Path, dry_run: bool = False, skip_existing: bool = False) -> int:
    ensure_dir(dest_dir)
    count = 0
    for src in files:
        dst = dest_dir / src.name if skip_existing else _unique_destination(dest_dir, src.name)
        if skip_existing and dst.exists():
            continue
        if not dry_run:
            shutil.copy2(src, dst)
        count += 1
    return count


def train_test_valid_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if not (abs(sum(ratios) - 1.0) < 1e-6):
        raise ValueError("Ratios must sum to 1.0")
    n_train = int(n * ratios[0])
    n_test = int(n * ratios[1])
    n_valid = n - n_train - n_test
    return n_train, n_test, n_valid


def split_indices(n: int, seed: int, ratios: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train, n_test, _ = train_test_valid_counts(n, ratios)
    train_idx = idxs[:n_train]
    test_idx = idxs[n_train : n_train + n_test]
    valid_idx = idxs[n_train + n_test :]
    return train_idx, test_idx, valid_idx


def collect_raw3_grouped(raw3_faces_root: Path) -> Dict[str, List[Path]]:
    """Collect images grouped by target class from raw/3/0 FaceImages."""
    grouped: Dict[str, List[Path]] = {}
    for source_name, target_name in RAW3_SOURCE_TO_TARGET.items():
        src_dir = raw3_faces_root / source_name
        imgs = list_images(src_dir)
        if not imgs:
            continue
        grouped.setdefault(target_name, []).extend(imgs)
    return grouped


def make_output_dirs(processed_root: Path, targets: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    out = {"train": {}, "test": {}, "valid": {}}
    for split in out.keys():
        for cls in targets:
            path = processed_root / split / cls
            ensure_dir(path)
            out[split][cls] = path
    return out


def run(workspace_root: Path, seed: int = 42, dry_run: bool = False, purge_target: bool = False, only_classes: List[str] | None = None, skip_existing: bool = False) -> None:
    raw3_faces_root = workspace_root / "raw" / "3" / "0 FaceImages"
    processed_root = workspace_root / "data" / "processed"

    if not raw3_faces_root.exists():
        raise FileNotFoundError(f"Raw/3 faces root not found at: {raw3_faces_root}")
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found at: {processed_root}")

    grouped = collect_raw3_grouped(raw3_faces_root)
    if not grouped:
        raise RuntimeError("No images found in raw/3/0 FaceImages. Check paths and contents.")

    targets = sorted(grouped.keys())
    if only_classes:
        targets = [t for t in targets if t in set(only_classes)]
    out_dirs = make_output_dirs(processed_root, targets)

    if purge_target:
        for split in ("train", "test", "valid"):
            for cls in targets:
                clear_dir(out_dirs[split][cls])

    summary = {}
    for cls in targets:
        files = grouped.get(cls, [])
        n = len(files)
        tr_idx, te_idx, va_idx = split_indices(n, seed=seed, ratios=RATIOS)
        train_files = [files[i] for i in tr_idx]
        test_files = [files[i] for i in te_idx]
        valid_files = [files[i] for i in va_idx]

        copy_files(train_files, out_dirs["train"][cls], dry_run=dry_run, skip_existing=skip_existing)
        copy_files(test_files, out_dirs["test"][cls], dry_run=dry_run, skip_existing=skip_existing)
        copy_files(valid_files, out_dirs["valid"][cls], dry_run=dry_run, skip_existing=skip_existing)

        summary[cls] = {
            "total": n,
            "train": len(train_files),
            "test": len(test_files),
            "valid": len(valid_files),
        }

    print("Split raw/3 (Active->SafeDriving, Fatigue->SleepyDriving)" + (" (dry-run)" if dry_run else ""))
    for cls in targets:
        s = summary[cls]
        print(f"{cls}: total={s['total']} -> train={s['train']}, test={s['test']}, valid={s['valid']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split raw/3/0 FaceImages into data/processed train/test/valid with mapping: "
            "'Active Subjects' -> SafeDriving, 'Fatigue Subjects' -> SleepyDriving"
        )
    )
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root, help=f"Workspace root (default: {default_root})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy files, only print what would happen")
    parser.add_argument(
        "--purge-target",
        action="store_true",
        help="Clear target class dirs (SafeDriving, SleepyDriving) before copying",
    )
    parser.add_argument(
        "--only-classes",
        nargs="*",
        choices=["SafeDriving", "SleepyDriving"],
        help="Restrict copying to specific target classes",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip copy if a file with the same name already exists in destination (idempotent)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(workspace_root=args.root, seed=args.seed, dry_run=args.dry_run, purge_target=args.purge_target, only_classes=args.only_classes, skip_existing=args.skip_existing)
