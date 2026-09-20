"""
extract_signal_multiscale.py

Phase 0.3 - Multi-scale signal representation.

Extends your original extract_signal.py: instead of committing to
200 bp / mean-signal up front, compute several bin sizes and several
summary statistics per bin, so Phase 0 analysis can decide which
representation is stable before Phase 1 modeling locks one in.

Run this on chr1 only first (CHROM_SUBSET below) to compare
representations quickly, before scaling to the full genome.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pybigtools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "multiscale_signal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bin sizes to test. Add 50_000 later for broad marks if 10 kb looks
# too fine-grained for e.g. H3K9me3/H3K27me3.
BIN_SIZES = [200, 1_000, 5_000, 10_000]
# Set to None to run the full genome once you've picked a
# representation; keep restricted to chr1 while you're comparing.
CHROM_SUBSET = ["chr1"]
def bin_statistics(values, background_threshold=0.0):
    """Compute all summary statistics for one bin's raw values."""
    if len(values) == 0:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "frac_above_bg": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "frac_above_bg": float((values > background_threshold).mean()),
    }
def extract_multiscale(bigwig_file, bin_size, output_dir, chrom_subset=None):
    output_file = output_dir / f"{bigwig_file.stem}_{bin_size}bp_signal.parquet"
    if output_file.exists():
        print(f"Skipping {output_file.name} (exists)")
        return
    print(f"\nOpening {bigwig_file.name} at {bin_size} bp resolution.")
    bw = pybigtools.open(str(bigwig_file))
    chroms = chrom_subset if chrom_subset else (
        [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    )
    rows = []
    bin_id = 0
    for chrom in chroms:
        if chrom not in bw.chroms():
            continue
        chrom_length = bw.chroms()[chrom]
        for start in range(0, chrom_length, bin_size):
            end = min(start + bin_size, chrom_length)
            values = bw.values(chrom=chrom, start=start, end=end, fillna=0)
            stats = bin_statistics(values)
            rows.append({
                "bin_id": bin_id,
                "chromosome": chrom,
                "start": start,
                "end": end,
                **stats,
            })
            bin_id += 1
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, index=False)
    print(f"Saved {len(df)} bins to {output_file}")
    bw.close()
if __name__ == "__main__":
    bigwigs = sorted(DATA_DIR.glob("*.bigWig"))
    print(f"Found {len(bigwigs)} bigWig files.")
    jobs = [
        (bw_file, bin_size, OUTPUT_DIR, CHROM_SUBSET)
        for bw_file in bigwigs
        for bin_size in BIN_SIZES
    ]
    print(f"Total (file, bin_size) jobs: {len(jobs)}")
    NUM_WORKERS = 3
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(executor.map(
            extract_multiscale,
            [j[0] for j in jobs],
            [j[1] for j in jobs],
            [j[2] for j in jobs],
            [j[3] for j in jobs],
        ))

    print("\nDone. Next: run compare_representations.py to pick the "
          "primary bin size + statistic before Phase 1 modeling.")