# 8
# Chromosome differences

#import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda" / "chromosome_means"
FIGURE_DIR= PROJECT_ROOT / "figures" / "eda" / "chromosome_means"
RESULTS_DIR.mkdir(parents=True,
                  exist_ok=True)
FIGURE_DIR.mkdir(parents= True,
                 exist_ok=True)

for f in INPUT_DIR.iterdir():
    print(f.name[:-22])
    chr = pd.read_parquet(f).drop(columns=["start", "end"]).groupby(by = "chromosome").mean()
    chromosome_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chr = chr.reindex(chromosome_order)
    print(chr)
    print()
    #saving the csv file
    chr.to_csv(path_or_buf= RESULTS_DIR / f"{f.name[:-22]}_chromosome_means.csv")
    #figure
    plt.figure(figsize=(8, 7))
    sns.heatmap(data=chr,
                cmap="coolwarm",
                annot=True,
                fmt=".3f",
                vmin=0,
                vmax = 1,
                cbar_kws= {"label": "Chromosome Means"})
    plt.title(f"{f.name[:-22]}Chromosome Means")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{f.name[:-22]}_chromosome_means.png")
    plt.close()