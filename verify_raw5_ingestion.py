#!/usr/bin/env python3
"""
Verify raw/5 ingestion by comparing expected counts from annotations to the
files copied (from the manifest) into data/processed.

Checks performed:
- Re-parse raw/5 annotations using the same logic as the ingestor to compute
  expected counts per split/class (skips missing files like ingestor).
- Load outputs/reports/ingest_raw5_manifest.json and group created files per
  split/class.
- Compare expected vs actual counts; report PASS/FAIL.
- Ensure every manifest file exists on disk.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import sys as _sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "reports" / "ingest_raw5_manifest.json"


def load_manifest() -> list[Path]:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        sys.exit(2)
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    files = data.get("files")
    if not isinstance(files, list):
        print("ERROR: Manifest 'files' missing or not a list")
        sys.exit(2)
    return [PROCESSED_ROOT / Path(rel) for rel in files]


def main() -> None:
    # Ensure project root is on sys.path so we can import the 'src' package
    if str(PROJECT_ROOT) not in _sys.path:
        _sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import src.ingest_raw5_to_processed as r5
    except Exception as e:
        print(f"ERROR: failed to import ingestor module: {e}")
        sys.exit(2)

    splits = ["train", "valid", "test"]

    # Expected counts from annotations
    expected_counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in splits}
    expected_total = 0
    for split in splits:
        samples, _ = r5.collect_samples_for_split(split)
        for s in samples:
            expected_counts[split][s.target_class] += 1
            expected_total += 1

    # Actual counts from manifest
    manifest_paths = load_manifest()
    missing_files = [p for p in manifest_paths if not p.exists()]
    actual_counts: dict[str, dict[str, int]] = {s: defaultdict(int) for s in splits}
    for p in manifest_paths:
        try:
            split = p.relative_to(PROCESSED_ROOT).parts[0]
            cls = p.relative_to(PROCESSED_ROOT).parts[1]
        except Exception:
            # Not in expected split/class path shape
            continue
        if split in actual_counts:
            actual_counts[split][cls] += 1

    # Report comparison
    classes = sorted({c for split in splits for c in expected_counts[split].keys()} | {c for split in splits for c in actual_counts[split].keys()})
    print("Expected vs Actual counts per split/class:")
    all_ok = True
    for split in splits:
        print(f"  {split}:")
        for cls in classes:
            e = expected_counts[split].get(cls, 0)
            a = actual_counts[split].get(cls, 0)
            flag = "OK" if e == a else "MISMATCH"
            print(f"    {cls:15s}: expected={e:6d} | actual={a:6d}  {flag}")
            if e != a:
                all_ok = False
        e_total = sum(expected_counts[split].values())
        a_total = sum(actual_counts[split].values())
        flag = "OK" if e_total == a_total else "MISMATCH"
        print(f"    {'TOTAL':15s}: expected={e_total:6d} | actual={a_total:6d}  {flag}")
        if e_total != a_total:
            all_ok = False

    print(f"\nGrand totals: expected={expected_total} | actual={len(manifest_paths)} | {'OK' if expected_total == len(manifest_paths) else 'MISMATCH'}")
    if expected_total != len(manifest_paths):
        all_ok = False

    if missing_files:
        print(f"WARNING: {len(missing_files)} manifest files are missing on disk (should be 0)")
        for p in missing_files[:10]:
            print(f"  missing: {p}")
        all_ok = False

    print("\nResult:")
    if all_ok:
        print("PASS: raw/5 ingestion matches annotations and files exist.")
        sys.exit(0)
    else:
        print("FAIL: Discrepancies found. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
