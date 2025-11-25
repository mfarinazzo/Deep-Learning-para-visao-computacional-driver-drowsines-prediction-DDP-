from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Source -> Target mapping
SOURCE_TO_TARGET: Dict[str, str] = {
    "safe_driving": "SafeDriving",
    "turning": "SafeDriving",
    "talking_phone": "DangerousDriving",
    "texting_phone": "DangerousDriving",
}

RATIOS = (0.70, 0.15, 0.15)  # train, test, valid


@dataclass
class SplitCounts:
    train: int
    test: int
    valid: int


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def list_images(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        return []
    return [p for p in dir_path.iterdir() if is_image(p)]


def train_test_valid_counts(n: int, ratios: Tuple[float, float, float]) -> SplitCounts:
    if not (abs(sum(ratios) - 1.0) < 1e-6):
        raise ValueError("Ratios must sum to 1.0")
    n_train = int(n * ratios[0])
    n_test = int(n * ratios[1])
    n_valid = n - n_train - n_test
    return SplitCounts(n_train, n_test, n_valid)


def split_indices(n: int, seed: int, ratios: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    counts = train_test_valid_counts(n, ratios)
    train_idx = idxs[: counts.train]
    test_idx = idxs[counts.train : counts.train + counts.test]
    valid_idx = idxs[counts.train + counts.test :]
    return train_idx, test_idx, valid_idx


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except Exception:
                    pass
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)


def collect_by_target(raw_root: Path) -> Dict[str, List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for source_name, target_name in SOURCE_TO_TARGET.items():
        source_dir = raw_root / source_name
        imgs = list_images(source_dir)
        if not imgs:
            continue
        grouped.setdefault(target_name, []).extend(imgs)
    return grouped


def _unique_destination(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding destination path by appending numeric suffix if needed."""
    base = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dest_dir / filename
    k = 1
    while candidate.exists():
        candidate = dest_dir / f"{base}__{k}{suffix}"
        k += 1
    return candidate


def copy_files(files: Iterable[Path], dest_dir: Path, dry_run: bool = False, skip_existing: bool = False, only_colliding: bool = False) -> None:
    ensure_dir(dest_dir)
    for src in files:
        # Prefixo raw1_
        new_name = f"raw1_{src.name}"
        base_dst = dest_dir / new_name
        
        if only_colliding:
            if not base_dst.exists():
                continue
            dst = _unique_destination(dest_dir, new_name)
        else:
            dst = base_dst if skip_existing else _unique_destination(dest_dir, new_name)
            if skip_existing and dst.exists():
                continue
        
        if dry_run:
            continue
        shutil.copy2(src, dst)


def make_output_dirs(processed_root: Path, targets: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    out = {"train": {}, "test": {}, "valid": {}}
    for split in out.keys():
        for cls in targets:
            path = processed_root / split / cls
            ensure_dir(path)
            out[split][cls] = path
    return out


def split_and_copy(
    workspace_root: Path,
    seed: int = 42,
    dry_run: bool = False,
    purge_target: bool = False,
    only_classes: List[str] | None = None,
    skip_existing: bool = False,
    only_colliding: bool = False,
):
    raw_root = workspace_root / "raw" / "1" / "Multi-Class Driver Behavior Image Dataset"
    processed_root = workspace_root / "data" / "processed"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_root}")
    
    # ALTERAÇÃO: Cria a pasta processed se não existir, em vez de dar erro
    processed_root.mkdir(parents=True, exist_ok=True)

    grouped = collect_by_target(raw_root)
    if not grouped:
        raise RuntimeError("No images found for the specified categories.")

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
        
        copy_files(train_files, out_dirs["train"][cls], dry_run=dry_run, skip_existing=skip_existing, only_colliding=only_colliding)
        copy_files(test_files, out_dirs["test"][cls], dry_run=dry_run, skip_existing=skip_existing, only_colliding=only_colliding)
        copy_files(valid_files, out_dirs["valid"][cls], dry_run=dry_run, skip_existing=skip_existing, only_colliding=only_colliding)

        summary[cls] = {
            "total": n,
            "train": len(train_files),
            "test": len(test_files),
            "valid": len(valid_files),
        }

    print("Split complete" + (" (dry-run)" if dry_run else ""))
    for cls in targets:
        s = summary[cls]
        print(f"{cls}: total={s['total']} -> train={s['train']}, test={s['test']}, valid={s['valid']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw/1 into data/processed with raw1_ prefix")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--purge-target", action="store_true")
    parser.add_argument("--only-classes", nargs="*", choices=["DangerousDriving", "SafeDriving"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only-colliding", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_and_copy(
        workspace_root=args.root,
        seed=args.seed,
        dry_run=args.dry_run,
        purge_target=args.purge_target,
        only_classes=args.only_classes,
        skip_existing=args.skip_existing,
        only_colliding=args.only_colliding,
    )
