#packages
from pathlib import Path #for specifying the paths much easier
import pandas as pd
import sys #for importing config
import os #for importing config


#importing CELL_TYPES from config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
sys.path.append(os.path.abspath(PROJECT_ROOT / "config")) #since the file is in a different directory
from config import CELL_TYPES

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "windows"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
OUTPUT_DIR.mkdir(parents=True,
                 exist_ok=True)

for cell in CELL_TYPES:
    print(cell)
    for f in range(0,len(list(INPUT_DIR.glob(f"{cell}*")))):
        mark = list(INPUT_DIR.glob(f"{cell}*"))[f].stem.split("_")[1]
        file = pd.read_parquet(list(INPUT_DIR.glob(f"{cell}*"))[f])
        file = file.rename(columns={"binary":mark})
        if f == 0:
            merged_files = file.copy()
        else:
            merged_files = pd.merge(left = merged_files,
                                    right=file,
                                    how = "left",
                                    on = ["chromosome", "start", "end"])
    merged_files.to_parquet(path=OUTPUT_DIR / f"{cell}_window_matrix.parquet",
                            index=False)