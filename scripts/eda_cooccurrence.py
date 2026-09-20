# 8
# co-occurrance: two or more things happening, existing, or appearing 
# together at the same time or in the same place
#import packages
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent #the root directory
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
RESULTS_DIR = PROJECT_ROOT / "results" / "eda" / "cooccurrence"
FIGURE_DIR= PROJECT_ROOT / "figures" / "eda" / "cooccurrence"
RESULTS_DIR.mkdir(parents=True,
                  exist_ok=True)
FIGURE_DIR.mkdir(parents= True,
                 exist_ok=True)

for f in INPUT_DIR.iterdir():
    print(f.name[:-22])
    file = pd.read_parquet(f).drop(columns=["chromosome", "start", "end"])
    co = np.log10(file.T @ file) #log10(number of co-occurring 10-kb windows)
    print(co)
    #saving the co-occurence matrix in a csv file
    co.to_csv(path_or_buf=RESULTS_DIR / f"{f.name[:-22]}_cooccurrence.csv")
    #saving the figure
    sns.heatmap(data=co,
                annot=True,
                cmap= "coolwarm",
                fmt= ".3f",
                cbar_kws= {"label": "Co-occurancy rate (log10)"})
    plt.title(f"{f.name[:-22]} Co-occurance Heatmap")
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{f.name[:-22]}_cooccurrence.png",
                dpi = 300)
    plt.close()