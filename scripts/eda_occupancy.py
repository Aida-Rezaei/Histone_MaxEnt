# 8
# Occupancy distribution
# how many histone marks are present in each 10 kb genomic window.
# For your 7 marks, each window can have 0-7 marks.

#import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda" / "occupancy"
FIGURE_DIR= PROJECT_ROOT / "figures" / "eda" / "occupancy"
RESULTS_DIR.mkdir(parents=True,
                  exist_ok=True)
FIGURE_DIR.mkdir(parents= True,
                 exist_ok=True)

for f in INPUT_DIR.iterdir():
    print(f.name[:-22])
    occ = pd.read_parquet(f).drop(columns=["chromosome", "start", "end"]).sum(axis=1).value_counts().sort_index().reset_index()
    occ.columns = ["occupancy", "window_count"]
    print(occ)
    print()
    #saving the occupancy file
    occ.to_csv(path_or_buf = RESULTS_DIR / f"{f.name[:-22]}_occupancy.csv")
    #plot
    plt.bar(occ["occupancy"], occ["window_count"])
    plt.xlabel("Number of Histone Markers per Window (Occupancy)")
    plt.ylabel("Number of Windows")
    plt.title(f"{f.name[:-22]} Occupancy Barplot")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{f.name[:-22]}_occupancy.png", 
                dpi = 300)
    plt.close()

