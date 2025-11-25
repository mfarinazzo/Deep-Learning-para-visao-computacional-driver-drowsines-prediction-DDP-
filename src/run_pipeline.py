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
        print(f"Deleting directory: {path}")
        shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run end-to-end dataset pipeline")
    ap.add_argument("--skip-download", action="store_true", help="Skip Kaggle downloads")
    ap.add_argument("--download-mode", choices=["cli", "api"], default="api", help="Downloader mode")
    ap.add_argument("--rebuild", action="store_true", help="Delete data/processed and data/standardized before ingest")
    ap.add_argument("--seed", type=int, default=42, help="Seed for splits")
    ap.add_argument("--std-size", type=int, default=224, help="Standardized image size")
    ap.add_argument("--std-mode", choices=["fit", "fill", "stretch"], default="fit", help="Resize strategy")
    ap.add_argument("--std-convert", choices=["jpg", "png", "keep"], default="jpg", help="Output format")
    args = ap.parse_args(argv)

    if not args.skip_download:
        run_step([PY, str(ROOT / "src" / "download_kaggle_datasets.py"), "--apply", "--mode", args.download_mode], title="Download raw datasets from Kaggle")

    if args.rebuild:
        delete_dir(ROOT / "data" / "processed")
        delete_dir(ROOT / "data" / "standardized")

    # Ingest raws 1..5
    run_step([PY, str(ROOT / "src" / "ingest_raw1_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/1 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw2_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/2 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw3_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/3 -> data/processed")
    run_step([PY, str(ROOT / "src" / "ingest_raw4_to_processed.py"), "--seed", str(args.seed)], title="Ingest raw/4 -> data/processed")
    
    # CORREÇÃO: Adicionado --apply para o raw5 sair do modo dry-run
    run_step([PY, str(ROOT / "src" / "ingest_raw5_to_processed.py"), "--apply"], title="Ingest raw/5 -> data/processed")

    # Verify totals
    run_step([PY, str(ROOT / "src" / "verify_processed_totals.py")], title="Verify processed totals")

    # Standardize dataset
    std_args = [
        PY,
        str(ROOT / "src" / "standardize_dataset.py"),
        "--size", str(args.std_size),
        "--mode", args.std_mode,
    ]
    if args.std_convert != "keep":
        std_args += ["--convert", args.std_convert]
    run_step(std_args, title="Standardize dataset -> data/standardized")

    # Cleanup processed (Limpeza do lixo)
    print("\n=== Cleaning up intermediate data ===")
    delete_dir(ROOT / "data" / "processed")
    print("Deleted data/processed to save space. Only data/standardized and raw/ remain.")

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
