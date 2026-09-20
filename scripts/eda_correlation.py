# 8

#import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda" / "correlation"
FIGURE_DIR= PROJECT_ROOT / "figures" / "eda" / "correlation"
RESULTS_DIR.mkdir(parents=True,
                  exist_ok=True)
FIGURE_DIR.mkdir(parents= True,
                 exist_ok=True)

for f in INPUT_DIR.iterdir():
    print(f.name[:-22])
    corre = pd.read_parquet(f).drop(columns=["chromosome", "start", "end"]).corr()
    print(corre)
    #saving the matrix to csv files
    corre.to_csv(path_or_buf=RESULTS_DIR / f"{f.name[:-22]}_correlation.csv")
    #correlation matrix image
    sns.heatmap(data=corre,
                cmap="coolwarm",
                annot=True,
                fmt=".3f",
                vmin = -1,
                vmax= 1,
                cbar_kws= {"label": "Pearson Correlation (r)"})
    plt.title(f"{f.name[:-22]} Correlation Heatmap")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{f.name[:-22]}_correlation.png",
                dpi = 300)
    plt.close()
