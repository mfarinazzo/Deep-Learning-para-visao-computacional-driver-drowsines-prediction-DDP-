from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable  # use current env python


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    print("Command:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        proc.terminate()
        raise
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Step failed ({title}) with exit code {ret}")


def delete_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run end-to-end dataset pipeline: download -> ingest -> verify -> standardize")
    ap.add_argument("--skip-download", action="store_true", help="Skip Kaggle downloads")
    ap.add_argument("--download-mode", choices=["cli", "api"], default="api", help="Downloader mode: cli or api (api shows progress)")
    ap.add_argument("--rebuild", action="store_true", help="Delete data/processed and data/standardized before ingest")
    ap.add_argument("--seed", type=int, default=42, help="Seed for 70/15/15 splits on raws 1-4")
    ap.add_argument("--std-size", type=int, default=224, help="Standardized image size (square)")
    ap.add_argument("--std-mode", choices=["fit", "fill", "stretch"], default="fit", help="Resize strategy")
    ap.add_argument("--std-convert", choices=["jpg", "png", "keep"], default="jpg", help="Output format for standardized set")
    args = ap.parse_args(argv)

    if not args.skip_download:
        run_step([PY, str(ROOT / "src" / "download_kaggle_datasets.py"), "--apply", "--mode", args.download_mode], title="Download raw datasets from Kaggle")

    if args.rebuild:
        delete_dir(ROOT / "data" / "processed")
        delete_dir(ROOT / "data" / "standardized")

    # Ingest raws 1..4 (70/15/15) and raw/5 (provided splits)
    run_step([PY, str(ROOT / "src" / "ingest_raw1_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/1 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw2_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/2 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw3_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/3 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw4_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/4 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw5_to_processed.py")], title="Ingest raw/5 -> data/processed")

    # Verify totals
    run_step([PY, str(ROOT / "src" / "verify_processed_totals.py")], title="Verify processed totals")

    # Standardize dataset
    std_args = [
        PY,
        str(ROOT / "src" / "standardize_dataset.py"),
        "--size",
        str(args.std_size),
        "--mode",
        args.std_mode,
    ]
    if args.std_convert != "keep":
        std_args += ["--convert", args.std_convert]
    run_step(std_args, title="Standardize dataset -> data/standardized")

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
