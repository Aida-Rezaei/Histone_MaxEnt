# 8

#import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda" / "frequencies"
FIGURE_DIR= PROJECT_ROOT / "figures" / "eda" / "frequencies"
RESULTS_DIR.mkdir(parents=True,
                  exist_ok=True)
FIGURE_DIR.mkdir(parents= True,
                 exist_ok=True)

for f in INPUT_DIR.iterdir():
    print(f.name[:-22])
    df = pd.read_parquet(f).drop(columns=["chromosome", "start", "end"]).mean().reset_index()
    df.columns = ["mark", "frequency"]
    print(df)
    df.to_csv(path_or_buf=RESULTS_DIR / f"{f.name[:-22]}_mark_frequencies.csv")
    df.plot.bar(x = "mark", y = "frequency")
    plt.title(f"{f.name[:-22]} Histone Mark Frequencies")
    plt.xlabel("Histone Mark")
    plt.ylabel("Frequency")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{f.name[:-22]}_mark_frequencies.png",
                dpi = 300)