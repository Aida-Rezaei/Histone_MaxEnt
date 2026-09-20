#10: done: all #9 files are there!
# 11
# Compare Maximum Entropy Couplings Across Five Cell Types

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAXENT_DIR = (
    PROJECT_ROOT
    / "results"
    / "maxent"
)


# ============================================================
# CELL TYPES
# ============================================================

CELL_TYPES = [
    "H1",
    "GM12878",
    "K562",
    "HepG2",
    "IMR90"
]


# ============================================================
# HISTONE MARKS
# ============================================================

MARKS = [
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3"
]


# ============================================================
# LOAD ALL COUPLING MATRICES
# ============================================================

all_J = {}

for cell_type in CELL_TYPES:

    file = (
        MAXENT_DIR
        / cell_type
        / f"{cell_type}_coupling_matrix.csv"
    )

    print(
        f"Reading {cell_type}: {file}"
    )

    J = pd.read_csv(
        file,
        index_col=0
    )

    all_J[cell_type] = J


# ============================================================
# CREATE THE 21 UNIQUE PAIRS
# ============================================================

pairs = [
    (MARKS[i], MARKS[j])
    for i in range(len(MARKS))
    for j in range(i + 1, len(MARKS))
]


# ============================================================
# CREATE CROSS-CELL-TYPE TABLE
# ============================================================

rows = []

for mark_a, mark_b in pairs:

    row = {
        "mark_a": mark_a,
        "mark_b": mark_b
    }

    for cell_type in CELL_TYPES:

        row[cell_type] = (
            all_J[cell_type]
            .loc[mark_a, mark_b]
        )

    rows.append(row)


coupling_df = pd.DataFrame(
    rows
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

values = coupling_df[
    CELL_TYPES
]


coupling_df["mean_J"] = (
    values.mean(axis=1)
)


coupling_df["sd_J"] = (
    values.std(axis=1)
)


coupling_df["min_J"] = (
    values.min(axis=1)
)


coupling_df["max_J"] = (
    values.max(axis=1)
)


coupling_df["range_J"] = (
    coupling_df["max_J"]
    - coupling_df["min_J"]
)


# ============================================================
# SIGN CONSISTENCY
# ============================================================

coupling_df["positive_count"] = (
    values.gt(0).sum(axis=1)
)


coupling_df["negative_count"] = (
    values.lt(0).sum(axis=1)
)


coupling_df["sign_consistent"] = (
    (
        coupling_df["positive_count"] == len(CELL_TYPES)
    )
    |
    (
        coupling_df["negative_count"] == len(CELL_TYPES)
    )
)


# ============================================================
# SAVE COUPLING TABLE
# ============================================================

coupling_df.to_csv(
    MAXENT_DIR
    / "cross_cell_type_couplings_summary.csv",
    index=False
)


print("\nCROSS-CELL-TYPE COUPLING TABLE")
print(
    coupling_df.to_string(index=False)
)


# ============================================================
# PAIRWISE CELL-TYPE DISTANCES
#
# Use only the 21 unique couplings.
# ============================================================

distance_rows = []


for i, cell_a in enumerate(CELL_TYPES):

    for cell_b in CELL_TYPES[i + 1:]:

        vector_a = np.array([
            all_J[cell_a].loc[
                mark_a,
                mark_b
            ]
            for mark_a, mark_b in pairs
        ])

        vector_b = np.array([
            all_J[cell_b].loc[
                mark_a,
                mark_b
            ]
            for mark_a, mark_b in pairs
        ])


        euclidean_distance = np.linalg.norm(
            vector_a - vector_b
        )


        correlation = np.corrcoef(
            vector_a,
            vector_b
        )[0, 1]


        distance_rows.append({
            "cell_type_1": cell_a,
            "cell_type_2": cell_b,
            "euclidean_distance": euclidean_distance,
            "coupling_correlation": correlation
        })


distance_df = pd.DataFrame(
    distance_rows
)


# ============================================================
# SAVE DISTANCES
# ============================================================

distance_df.to_csv(
    MAXENT_DIR
    / "J_matrix_distances.csv",
    index=False
)


print("\nCELL-TYPE COUPLING DISTANCES")
print(
    distance_df.to_string(index=False)
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 11 COUPLING COMPARISON COMPLETE")
print("=" * 70)