'''
binarize the siganls of each chromosome in each signal file
'''
#packages
from pathlib import Path #for specifying the paths much easier
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor #parallel processing


PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
SIGNAL_DIR = PROJECT_ROOT / "data" / "processed" / "signals" #BINARY data directory
BINARY_OUTPUT_DIR = PROJECT_ROOT / "data"/ "processed" / "binary" #BINARY output directory
BINARY_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok= True
)

def binarize_signal(signal_file, binary_output_dir):
    """
    converting signal to binary files based on 95% percentile threshold.
    The thresholds from all chromosomes and files are saved in the results folder (thresholds.csv)
    inputs:
    - signal_file: the signal parquet file
    - binary_output_dir: the directory where the binary file to be saved in
    """
    #signal_file = SIGNAL_DIR / signal_file
    print(f"Opening {signal_file.name} file.\n")
    output_binary = binary_output_dir / f"{signal_file.stem[:-7]}_binary.parquet" #output BINARY file directory (with file name) 
    read_file = pd.read_parquet(path = signal_file) #opening the file
    no_zeros = read_file[read_file["signal"] > 0] #filter out the rows with signal 0
    if no_zeros.empty:
        threshold = 0
    else:
        threshold = np.percentile(no_zeros["signal"], 95)
    read_file["binary"] = (read_file["signal"] >= threshold).astype(int) #adding the binary status to the tables.
    read_file = read_file.drop(columns=["signal"])
    read_file.to_parquet(path=output_binary,
                         index=False)
    return threshold

if __name__ == "__main__":
    signal_files = sorted(SIGNAL_DIR.glob("*_signal.parquet"))
    print(f"Found {len(signal_files)} signal parquet files.")
    with ProcessPoolExecutor(max_workers = 3) as executor:
        thresholds = list(
            executor.map(
            binarize_signal,
            signal_files,
            [BINARY_OUTPUT_DIR] * len(signal_files)
            )
        )

    threshold_df = pd.DataFrame({
        "signal_file" : [file.stem[:-7] for file in signal_files],
        "threshold" : thresholds
    })
    threshold_df.to_csv(
        PROJECT_ROOT / "results" / "thresholds.csv",
        index = False
    )
    print("\nThresholds saved")