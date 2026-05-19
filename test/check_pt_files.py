"""
check_pt_files.py
=================
Scans all .pt clip files under a root directory and reports:
  - Corrupted / unloadable files
  - Files with a different shape from the majority

Usage:
    python check_pt_files.py --data_dir /hadatasets/EPIC-KITCHENS_rgb_crops
    python check_pt_files.py --data_dir /hadatasets/EPIC-KITCHENS_rgb_crops --num_workers 8
"""

import os
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm


def find_pt_files(data_dir: str) -> list:
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".pt"):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def check_file(path: str) -> tuple:
    """Returns (path, shape_or_None, error_or_None)"""
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            return (path, None, f"not a Tensor, got {type(tensor).__name__}")
        return (path, tuple(tensor.shape), None)
    except Exception as e:
        return (path, None, str(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True, help="Root of .pt clip files")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    print(f"\nScanning {args.data_dir} ...")
    paths = find_pt_files(args.data_dir)
    print(f"Found {len(paths):,} .pt files  |  workers={args.num_workers}\n")

    if not paths:
        print("No .pt files found. Check --data_dir.")
        return

    corrupted     = []
    good          = []
    shape_counter = Counter()

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(check_file, p): p for p in paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="checking"):
            path, shape, err = fut.result()
            if err:
                corrupted.append((path, err))
            else:
                good.append((path, shape))
                shape_counter[shape] += 1

    majority_shape = shape_counter.most_common(1)[0][0] if shape_counter else None
    wrong_shape    = [(p, s) for p, s in good if s != majority_shape]

    # ── Report ────────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"RESULTS  ({len(paths):,} files checked)")
    print(sep)
    print(f"  OK            : {len(good):,}")
    print(f"  Corrupted     : {len(corrupted):,}")
    print(f"  Wrong shape   : {len(wrong_shape):,}  (vs majority {majority_shape})")

    print(f"\n── Shape frequency ───────────────────────────────────────────────")
    for shape, cnt in shape_counter.most_common():
        flag = "  <- MAJORITY" if shape == majority_shape else ""
        print(f"  {str(shape):<30}  {cnt:>6,} files{flag}")

    if corrupted:
        print(f"\n{sep}")
        print(f"CORRUPTED FILES ({len(corrupted)})")
        print(sep)
        for path, err in corrupted:
            print(f"  {path}")
            print(f"  ERROR: {err}")

    if wrong_shape:
        print(f"\n{sep}")
        print(f"WRONG SHAPE FILES ({len(wrong_shape)})  -- majority is {majority_shape}")
        print(sep)
        for path, shape in wrong_shape:
            print(f"  {shape}  ->  {path}")

    if not corrupted and not wrong_shape:
        print("\nAll files are consistent -- no issues found.")

    # Save bad files list
    bad = [(p, "CORRUPT") for p, _ in corrupted] + [(p, "WRONG_SHAPE") for p, _ in wrong_shape]
    if bad:
        out_path = os.path.join(args.data_dir, "bad_pt_files.txt")
        with open(out_path, "w") as f:
            for path, tag in bad:
                f.write(f"{tag}\t{path}\n")
        print(f"\nBad file list saved -> {out_path}")

    print(sep)


if __name__ == "__main__":
    main()