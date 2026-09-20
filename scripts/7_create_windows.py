# 7

#packages
from pathlib import Path #for specifying the paths much easier
import pandas as pd #for dataframe creation and modification
import numpy as np
from concurrent.futures import ProcessPoolExecutor #parallel processing

#directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
BINARY_DIR = PROJECT_ROOT / "data" / "processed" / "binary" #the path to the data
WINDOW_DIR = PROJECT_ROOT / "data" / "processed" / "windows" #the path to where the output file would
WINDOW_DIR.mkdir(exist_ok = True) #creates the directory if not available. if exists, no error

# FUNCTION
def create_windows(binary_file, output_dir):
    output_window = output_dir / f"{binary_file.stem[:-7]}_window.parquet"
    read_file = pd.read_parquet(binary_file)
    windows_list = []
    for chromosome in read_file["chromosome"].unique(): #processing one chromosome number at a time
        chrom_df = read_file[read_file["chromosome"] == chromosome].reset_index(drop = True)
        groups = np.arange(len(chrom_df)) // 50
        windows = chrom_df.groupby(groups).agg({
            "start": "first",
            "end": "last",
            "binary": "max"
            }).reset_index(drop = True)
        windows.insert(0, "chromosome", chromosome)
        windows_list.append(windows)
    window_df = pd.concat(windows_list, ignore_index=True)
    window_df.to_parquet(
        output_window,
        index = False
    )

if __name__ == "__main__":
    binary_files = sorted(
        BINARY_DIR.glob("*_binary.parquet")
    )
    print(f"Found {len(binary_files)} binary files.")
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(
            executor.map(
                create_windows,
                binary_files,
                [WINDOW_DIR] * len(binary_files)
            )
        )
    print("Phase 7 is over!")