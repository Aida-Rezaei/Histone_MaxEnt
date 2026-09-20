# ============================================================
# PHASE 12 — SEQUENCE / GENOMIC CONTEXT CONTROL
# Epigenenomic Project
#
# One cohesive script for the complete Phase 12 analysis.
#
# Main analysis:
#   1. Load 5 cell-type histone-state matrices
#   2. Build genomic-context features
#   3. Hold out chr20
#   4. Fit logistic regression for each histone mark
#   5. Predict held-out chr20
#   6. Residualize histone states
#   7. Compute controlled pairwise dependencies
#   8. Compare controlled dependencies with Phase 9 J couplings
#   9. Optional random-forest sensitivity analysis
#  10. Generate summary tables and figures
#
# IMPORTANT:
# Controlled residual correlations are NOT identical to Ising J.
# They should therefore be called:
#
#     residual correlation
#     controlled dependency
#     sequence-controlled association
#
# and NOT "controlled J" unless a separate maximum-entropy
# inference is performed on the residualized system.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

RESULTS_DIR = PROJECT_ROOT / "results" / "sequence_control"

FEATURE_DIR = RESULTS_DIR / "genomic_features"
MODEL_DIR = RESULTS_DIR / "logistic_models"
RESIDUAL_DIR = RESULTS_DIR / "residuals"
CONTROLLED_DIR = RESULTS_DIR / "controlled_dependencies"
FIGURE_DIR = RESULTS_DIR / "figures"

for directory in [
    RESULTS_DIR,
    FEATURE_DIR,
    MODEL_DIR,
    RESIDUAL_DIR,
    CONTROLLED_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PROJECT PARAMETERS
# ============================================================

CELL_TYPES = [
    "H1",
    "GM12878",
    "K562",
    "HepG2",
    "IMR90",
]

MARKS = [
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
]

VALIDATION_CHROMS = {"chr20"}

WINDOW_SIZE = 10_000

# Logistic regression regularization.
# C=1.0 is the standard baseline used in the Phase 12 plan.
LOGISTIC_C = 1.0

MAX_ITER = 1000

# Optional random forest.
# Keep FALSE for the fast preprint V1 pipeline.
RUN_RANDOM_FOREST = False

# Optional replication timing.
# If a BigWig exists, it will be used.
USE_REPLICATION_TIMING = False

RANDOM_SEED = 42


# ============================================================
# 3. REFERENCE FILES
# ============================================================

REFERENCE_FASTA = REFERENCE_DIR / "hg38.fa"

CPG_ISLAND_BED = REFERENCE_DIR / "cpgIslandExt.hg38.bed"

TSS_BED = REFERENCE_DIR / "tss.hg38.bed"

REPEAT_BED = REFERENCE_DIR / "rmsk.hg38.bed"

REPLICATION_BIGWIG = (
    REFERENCE_DIR / "replication_timing.hg38.bw"
)


# ============================================================
# 4. REQUIRED FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
    "gc_content",
    "cpg_density",
    "cpg_island_overlap",
    "log_dist_to_tss",
    "repeat_density",
]

if USE_REPLICATION_TIMING and REPLICATION_BIGWIG.exists():
    FEATURE_COLS.append("repli_timing")


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_chromosome_name(x):
    """
    Normalize chromosome names to chr1, chr2, ..., chrX, etc.
    """
    x = str(x)

    if not x.startswith("chr"):
        x = "chr" + x

    return x


def find_column(df, candidates):
    """
    Find the first matching column name.
    """

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def identify_coordinate_columns(df):
    """
    Automatically identify chromosome/start/end columns.
    """

    chrom_col = find_column(
        df,
        [
            "chrom",
            "chr",
            "chromosome",
            "seqname",
        ],
    )

    start_col = find_column(
        df,
        [
            "start",
            "window_start",
            "bin_start",
        ],
    )

    end_col = find_column(
        df,
        [
            "end",
            "window_end",
            "bin_end",
        ],
    )

    if chrom_col is None:
        raise ValueError(
            "Could not identify chromosome column."
        )

    if start_col is None:
        raise ValueError(
            "Could not identify start column."
        )

    if end_col is None:
        raise ValueError(
            "Could not identify end column."
        )

    return chrom_col, start_col, end_col


def load_table(path):
    """
    Load CSV or Parquet.
    """

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file format: {path}"
    )


# ============================================================
# 6. FIND WINDOW MATRIX
# ============================================================

def find_window_file(cell_type):
    """
    Try to locate the existing window matrix automatically.

    Exact names are tried first.
    """

    candidates = [
        PROCESSED_DIR / f"{cell_type}_window_matrix.csv",
        PROCESSED_DIR / f"{cell_type}_window_matrix.parquet",
        PROCESSED_DIR / f"{cell_type}_windows.csv",
        PROCESSED_DIR / f"{cell_type}_windows.parquet",
        PROCESSED_DIR / f"{cell_type}.csv",
        PROCESSED_DIR / f"{cell_type}.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Recursive fallback.
    possible = []

    for path in PROCESSED_DIR.rglob("*"):
        if path.suffix.lower() not in [".csv", ".parquet"]:
            continue

        name = path.name.lower()
        cell = cell_type.lower()

        if cell not in name:
            continue

        # FIX: Ensure it matches the full matrix file name pattern 
        # and ignore individual chromosome/mark files like '_binary' or '_H3K...'
        if (
            name.endswith(f"{cell}_window_matrix.parquet")
            or name.endswith(f"{cell}_window_matrix.csv")
            or name == f"{cell}_windows.parquet"
            or name == f"{cell}_windows.csv"
            or cell == name.rsplit(".", 1)[0]
        ):
            possible.append(path)

    if len(possible) == 1:
        return possible[0]

    if len(possible) == 0:
        raise FileNotFoundError(
            f"\nNo window matrix found for {cell_type}.\n"
            f"Expected something like:\n"
            f"  {PROCESSED_DIR / f'{cell_type}_window_matrix.csv'}\n"
        )

    raise RuntimeError(
        f"\nMultiple possible files found for {cell_type}:\n"
        + "\n".join(str(p) for p in possible)
        + "\n\nPlease rename the desired window matrix "
          "to an unambiguous filename."
    )



# ============================================================
# 7. LOAD AND STANDARDIZE WINDOW MATRIX
# ============================================================

def load_cell_matrix(cell_type):

    path = find_window_file(cell_type)

    print(f"\nLoading {cell_type}")
    print(f"File: {path}")

    df = load_table(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    chrom_col, start_col, end_col = (
        identify_coordinate_columns(df)
    )

    rename_map = {
        chrom_col: "chrom",
        start_col: "start",
        end_col: "end",
    }

    df = df.rename(columns=rename_map)

    missing_marks = [
        mark
        for mark in MARKS
        if mark not in df.columns
    ]

    if missing_marks:

        # Try case-insensitive matching.
        lower_map = {
            str(c).lower(): c
            for c in df.columns
        }

        for mark in missing_marks:
            source = lower_map.get(mark.lower())

            if source is not None:
                df[mark] = df[source]

    missing_marks = [
        mark
        for mark in MARKS
        if mark not in df.columns
    ]

    if missing_marks:
        raise ValueError(
            f"{cell_type}: missing histone mark columns:\n"
            f"{missing_marks}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    df["chrom"] = df["chrom"].map(
        normalize_chromosome_name
    )

    df["start"] = pd.to_numeric(
        df["start"],
        errors="coerce",
    )

    df["end"] = pd.to_numeric(
        df["end"],
        errors="coerce",
    )

    if df[
        ["chrom", "start", "end"]
    ].isna().any().any():

        raise ValueError(
            f"{cell_type}: invalid genomic coordinates."
        )

    for mark in MARKS:

        df[mark] = pd.to_numeric(
            df[mark],
            errors="coerce",
        )

        if df[mark].isna().any():
            raise ValueError(
                f"{cell_type}: missing values in {mark}."
            )

        unique_values = set(
            df[mark].unique()
        )

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"{cell_type}: {mark} is not binary.\n"
                f"Observed values: "
                f"{sorted(unique_values)[:20]}"
            )

        df[mark] = df[mark].astype(np.int8)

    df = df[
        [
            "chrom",
            "start",
            "end",
            *MARKS,
        ]
    ].copy()

    df = df.sort_values(
        ["chrom", "start", "end"]
    ).reset_index(drop=True)

    print(
        f"{cell_type}: "
        f"{len(df):,} valid windows"
    )

    return df


# ============================================================
# 8. VERIFY THAT ALL CELL TYPES SHARE THE SAME WINDOWS
# ============================================================

def verify_shared_windows(data):

    print_header(
        "VERIFYING SHARED GENOMIC WINDOWS"
    )

    reference = data[CELL_TYPES[0]][
        ["chrom", "start", "end"]
    ].copy()

    for cell in CELL_TYPES[1:]:

        coords = data[cell][
            ["chrom", "start", "end"]
        ].copy()

        if len(coords) != len(reference):

            raise ValueError(
                f"{cell} has {len(coords):,} windows "
                f"but {CELL_TYPES[0]} has "
                f"{len(reference):,}."
            )

        same = (
            coords.reset_index(drop=True)
            .equals(
                reference.reset_index(drop=True)
            )
        )

        if not same:

            raise ValueError(
                f"Genomic windows differ between "
                f"{CELL_TYPES[0]} and {cell}."
            )

        print(
            f"{cell}: windows match "
            f"{CELL_TYPES[0]}"
        )

    print(
        f"\nAll {len(CELL_TYPES)} cell types share "
        f"{len(reference):,} windows."
    )


# ============================================================
# 9. LOAD PYFAIDX
# ============================================================

def load_fasta():

    if not REFERENCE_FASTA.exists():

        raise FileNotFoundError(
            "\nMissing reference FASTA:\n"
            f"{REFERENCE_FASTA}\n\n"
            "Place the hg38 FASTA there before "
            "running Phase 12."
        )

    try:
        from pyfaidx import Fasta

    except ImportError:

        raise ImportError(
            "\npyfaidx is required.\n\n"
            "Install with:\n"
            "pip install pyfaidx"
        )

    print(
        f"\nOpening reference FASTA:\n"
        f"{REFERENCE_FASTA}"
    )

    return Fasta(
        str(REFERENCE_FASTA),
        as_raw=True,
        sequence_always_upper=True,
    )


# ============================================================
# 10. FAST SEQUENCE FEATURES
# ============================================================

def compute_sequence_features(windows):

    print_header(
        "COMPUTING GC CONTENT AND CpG DENSITY"
    )

    genome = load_fasta()

    result = windows[
        ["chrom", "start", "end"]
    ].copy()

    result["gc_content"] = np.nan
    result["cpg_density"] = np.nan

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT build chromosome-wide prefix arrays.
    #
    # chr1 is ~249 million bases, so an int64 cumulative
    # array alone requires ~2 GB.
    #
    # Instead, process a limited number of windows at a time.
    # --------------------------------------------------------

    WINDOWS_PER_CHUNK = 5000

    for chrom, chrom_group in result.groupby(
        "chrom",
        sort=False,
    ):

        chrom_group = chrom_group.sort_values(
            ["start", "end"]
        )

        print(
            f"Processing {chrom}: "
            f"{len(chrom_group):,} windows"
        )

        try:
            chrom_length = len(genome[chrom])

        except Exception as exc:

            warnings.warn(
                f"Could not determine length of {chrom}: {exc}"
            )

            continue

        # ----------------------------------------------------
        # Process chromosome in manageable chunks.
        # ----------------------------------------------------

        for chunk_number, chunk_start in enumerate(
            range(
                0,
                len(chrom_group),
                WINDOWS_PER_CHUNK,
            ),
            start=1,
        ):

            chunk = chrom_group.iloc[
                chunk_start:
                chunk_start + WINDOWS_PER_CHUNK
            ]

            starts = (
                chunk["start"]
                .astype(np.int64)
                .to_numpy()
            )

            ends = (
                chunk["end"]
                .astype(np.int64)
                .to_numpy()
            )

            # Clamp coordinates to chromosome bounds.
            starts = np.clip(
                starts,
                0,
                chrom_length,
            )

            ends = np.clip(
                ends,
                0,
                chrom_length,
            )

            # Skip pathological intervals.
            valid_interval = ends > starts

            if not np.any(valid_interval):
                continue

            # ------------------------------------------------
            # Load only the sequence needed for this chunk.
            # ------------------------------------------------

            sequence_start = int(
                starts.min()
            )

            sequence_end = int(
                ends.max()
            )

            try:

                sequence = genome[
                    chrom
                ][
                    sequence_start:
                    sequence_end
                ]

            except Exception as exc:

                warnings.warn(
                    f"Could not extract "
                    f"{chrom}:{sequence_start}-{sequence_end}: "
                    f"{exc}"
                )

                continue

            sequence = str(
                sequence
            ).upper()

            arr = np.frombuffer(
                sequence.encode("ascii"),
                dtype=np.uint8,
            )

            n = len(arr)

            if n == 0:
                continue

            # ------------------------------------------------
            # Base masks.
            # ------------------------------------------------

            is_valid = (
                (arr == ord("A"))
                | (arr == ord("C"))
                | (arr == ord("G"))
                | (arr == ord("T"))
            )

            is_gc = (
                (arr == ord("G"))
                | (arr == ord("C"))
            )

            # ------------------------------------------------
            # Local prefix sums only.
            #
            # int32 is sufficient here because a chunk is
            # much smaller than 2.1 billion bases.
            # ------------------------------------------------

            valid_cs = np.empty(
                n + 1,
                dtype=np.int32,
            )

            valid_cs[0] = 0

            np.cumsum(
                is_valid,
                dtype=np.int32,
                out=valid_cs[1:],
            )

            gc_cs = np.empty(
                n + 1,
                dtype=np.int32,
            )

            gc_cs[0] = 0

            np.cumsum(
                is_gc,
                dtype=np.int32,
                out=gc_cs[1:],
            )

            # ------------------------------------------------
            # CpG prefix sum.
            #
            # cpg[i] means bases i and i+1 are C/G.
            # ------------------------------------------------

            if n >= 2:

                cpg = (
                    (arr[:-1] == ord("C"))
                    & (arr[1:] == ord("G"))
                )

                cpg_cs = np.empty(
                    n,
                    dtype=np.int32,
                )

                cpg_cs[0] = 0

                np.cumsum(
                    cpg,
                    dtype=np.int32,
                    out=cpg_cs[1:],
                )

            else:

                cpg_cs = np.zeros(
                    1,
                    dtype=np.int32,
                )

            # ------------------------------------------------
            # Convert genomic coordinates to local chunk
            # coordinates.
            # ------------------------------------------------

            local_starts = (
                starts
                - sequence_start
            )

            local_ends = (
                ends
                - sequence_start
            )

            local_starts = np.clip(
                local_starts,
                0,
                n,
            )

            local_ends = np.clip(
                local_ends,
                0,
                n,
            )

            lengths = np.maximum(
                local_ends - local_starts,
                1,
            )

            # ------------------------------------------------
            # Count valid bases and GC bases.
            # ------------------------------------------------

            valid_counts = (
                valid_cs[local_ends]
                - valid_cs[local_starts]
            )

            gc_counts = (
                gc_cs[local_ends]
                - gc_cs[local_starts]
            )

            # ------------------------------------------------
            # CpG count.
            #
            # For interval [start, end), CpG starts can occur
            # at positions start ... end-2.
            # ------------------------------------------------

            cpg_counts = np.zeros(
                len(chunk),
                dtype=np.int32,
            )

            cpg_cap = max(
                n - 1,
                0,
            )

            cpg_starts = np.clip(
                local_starts,
                0,
                cpg_cap,
            )

            cpg_ends = np.clip(
                local_ends - 1,
                0,
                cpg_cap,
            )

            has_cpg_space = (
                local_ends - local_starts >= 2
            )

            if np.any(has_cpg_space):

                cpg_counts[
                    has_cpg_space
                ] = (
                    cpg_cs[
                        cpg_ends[
                            has_cpg_space
                        ]
                    ]
                    -
                    cpg_cs[
                        cpg_starts[
                            has_cpg_space
                        ]
                    ]
                )

            # ------------------------------------------------
            # GC fraction.
            # ------------------------------------------------

            gc_values = np.divide(
                gc_counts,
                np.maximum(
                    valid_counts,
                    1,
                ),
                dtype=float,
            )

            # ------------------------------------------------
            # CpGs per kb.
            # ------------------------------------------------

            cpg_values = (
                cpg_counts
                / lengths
                * 1000.0
            )

            # ------------------------------------------------
            # Store results.
            # ------------------------------------------------

            result.loc[
                chunk.index,
                "gc_content",
            ] = gc_values

            result.loc[
                chunk.index,
                "cpg_density",
            ] = cpg_values

            if (
                chunk_number == 1
                or chunk_number % 10 == 0
                or (
                    chunk_start
                    + WINDOWS_PER_CHUNK
                    >= len(chrom_group)
                )
            ):

                processed = min(
                    chunk_start
                    + WINDOWS_PER_CHUNK,
                    len(chrom_group),
                )

                print(
                    f"  chunk {chunk_number}: "
                    f"{processed:,}/"
                    f"{len(chrom_group):,} windows"
                )

            # ------------------------------------------------
            # Explicitly release chunk-sized arrays.
            # ------------------------------------------------

            del (
                sequence,
                arr,
                is_valid,
                is_gc,
                valid_cs,
                gc_cs,
                cpg_cs,
                local_starts,
                local_ends,
                valid_counts,
                gc_counts,
                cpg_counts,
                gc_values,
                cpg_values,
            )

    genome.close()

    missing = result[
        ["gc_content", "cpg_density"]
    ].isna().sum()

    if missing.any():

        print(
            "\nMissing sequence features:"
        )

        print(
            missing
        )

    print(
        "\nSequence feature computation complete."
    )

    return result


# ============================================================
# 11. LOAD BED ANNOTATION
# ============================================================

def load_bed(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing annotation:\n{path}"
        )

    print(
        f"Loading annotation:\n{path}"
    )

    # BED-like files can sometimes contain comments.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
    )

    df.columns = [
        "chrom",
        "start",
        "end",
    ]

    df["chrom"] = df["chrom"].map(
        normalize_chromosome_name
    )

    df["start"] = pd.to_numeric(
        df["start"],
        errors="coerce",
    )

    df["end"] = pd.to_numeric(
        df["end"],
        errors="coerce",
    )

    df = df.dropna()

    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)

    df = df[
        (df["end"] > df["start"])
    ].copy()

    return df


# ============================================================
# 12. MERGE OVERLAPPING INTERVALS
# ============================================================

def merge_intervals(intervals):

    if len(intervals) == 0:
        return intervals

    intervals = intervals.sort_values(
        ["chrom", "start", "end"]
    )

    merged = []

    current_chrom = None
    current_start = None
    current_end = None

    for row in intervals.itertuples(
        index=False
    ):

        chrom = row.chrom
        start = int(row.start)
        end = int(row.end)

        if current_chrom is None:

            current_chrom = chrom
            current_start = start
            current_end = end
            continue

        if chrom != current_chrom:

            merged.append(
                (
                    current_chrom,
                    current_start,
                    current_end,
                )
            )

            current_chrom = chrom
            current_start = start
            current_end = end

            continue

        if start <= current_end:

            current_end = max(
                current_end,
                end,
            )

        else:

            merged.append(
                (
                    current_chrom,
                    current_start,
                    current_end,
                )
            )

            current_start = start
            current_end = end

    merged.append(
        (
            current_chrom,
            current_start,
            current_end,
        )
    )

    return pd.DataFrame(
        merged,
        columns=[
            "chrom",
            "start",
            "end",
        ],
    )


# ============================================================
# 13. FAST INTERVAL OVERLAP
# ============================================================

def calculate_overlap_fraction(
    windows,
    annotation,
):

    annotation = merge_intervals(
        annotation
    )

    output = np.zeros(
        len(windows),
        dtype=float,
    )

    # Process chromosome by chromosome.
    for chrom, window_group in windows.groupby(
        "chrom",
        sort=False,
    ):

        ann = annotation[
            annotation["chrom"] == chrom
        ]

        if len(ann) == 0:
            continue

        ann_starts = ann[
            "start"
        ].to_numpy(dtype=np.int64)

        ann_ends = ann[
            "end"
        ].to_numpy(dtype=np.int64)

        ann_index = 0

        for idx, row in window_group.iterrows():

            ws = int(row["start"])
            we = int(row["end"])

            while (
                ann_index < len(ann_ends)
                and ann_ends[ann_index] <= ws
            ):
                ann_index += 1

            j = ann_index
            overlap = 0

            while (
                j < len(ann_starts)
                and ann_starts[j] < we
            ):

                overlap += max(
                    0,
                    min(
                        we,
                        ann_ends[j],
                    )
                    - max(
                        ws,
                        ann_starts[j],
                    ),
                )

                j += 1

            window_length = max(
                we - ws,
                1,
            )

            output[idx] = (
                overlap / window_length
            )

    return output


# ============================================================
# 14. CpG ISLAND OVERLAP
# ============================================================

def add_cpg_island_feature(features):

    print_header(
        "COMPUTING CpG-ISLAND OVERLAP"
    )

    cpg = load_bed(
        CPG_ISLAND_BED
    )

    features["cpg_island_overlap"] = (
        calculate_overlap_fraction(
            features,
            cpg,
        )
    )

    print(
        "CpG-island overlap complete."
    )

    return features


# ============================================================
# 15. REPEAT DENSITY
# ============================================================

def add_repeat_feature(features):

    print_header(
        "COMPUTING REPEAT DENSITY"
    )

    repeats = load_bed(
        REPEAT_BED
    )

    features["repeat_density"] = (
        calculate_overlap_fraction(
            features,
            repeats,
        )
    )

    print(
        "Repeat density complete."
    )

    return features


# ============================================================
# 16. DISTANCE TO NEAREST TSS
# ============================================================

def add_tss_distance(features):

    print_header(
        "COMPUTING DISTANCE TO NEAREST TSS"
    )

    tss = load_bed(
        TSS_BED
    )

    tss_positions = {}

    for chrom, group in tss.groupby(
        "chrom",
        sort=False,
    ):

        positions = np.sort(
            (
                (
                    group["start"].to_numpy(
                        dtype=np.int64
                    )
                    +
                    group["end"].to_numpy(
                        dtype=np.int64
                    )
                )
                // 2
            )
        )

        tss_positions[chrom] = positions

    distances = np.zeros(
        len(features),
        dtype=np.float64,
    )

    for chrom, group in features.groupby(
        "chrom",
        sort=False,
    ):

        positions = tss_positions.get(
            chrom
        )

        if positions is None or len(positions) == 0:

            distances[
                group.index
            ] = np.nan

            continue

        mids = (
            (
                group["start"].to_numpy(
                    dtype=np.int64
                )
                +
                group["end"].to_numpy(
                    dtype=np.int64
                )
            )
            // 2
        )

        insertion = np.searchsorted(
            positions,
            mids,
            side="left",
        )

        insertion = np.clip(
            insertion,
            0,
            len(positions) - 1,
        )

        right_dist = np.abs(
            positions[insertion]
            - mids
        )

        left_index = np.maximum(
            insertion - 1,
            0,
        )

        left_dist = np.abs(
            positions[left_index]
            - mids
        )

        distances[
            group.index
        ] = np.minimum(
            left_dist,
            right_dist,
        )

    # Log transform.
    #
    # +1 avoids log(0).
    features["log_dist_to_tss"] = np.log1p(
        distances
    )

    print(
        "TSS distance complete."
    )

    return features


# ============================================================
# 17. OPTIONAL REPLICATION TIMING
# ============================================================

def add_replication_timing(features):

    if not USE_REPLICATION_TIMING:
        return features

    if not REPLICATION_BIGWIG.exists():

        print(
            "\nReplication timing BigWig not found."
        )

        print(
            "Continuing without replication timing."
        )

        return features

    try:
        import pyBigWig

    except ImportError:

        print(
            "\npyBigWig is not installed."
        )

        print(
            "Continuing without replication timing."
        )

        return features

    print_header(
        "COMPUTING REPLICATION TIMING"
    )

    bw = pyBigWig.open(
        str(REPLICATION_BIGWIG)
    )

    values = np.full(
        len(features),
        np.nan,
        dtype=float,
    )

    for idx, row in features.iterrows():

        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])

        try:

            result = bw.stats(
                chrom,
                start,
                end,
                type="mean",
            )

            if result is not None:
                if len(result) > 0:
                    values[idx] = result[0]

        except Exception:
            pass

    bw.close()

    features["repli_timing"] = values

    print(
        "Replication timing complete."
    )

    return features


# ============================================================
# 18. BUILD ALL GENOMIC FEATURES
# ============================================================

def build_genomic_features(windows):

    feature_path = (
        FEATURE_DIR
        / "all_windows_genomic_features.csv"
    )

    if feature_path.exists():

        print(
            "\nExisting genomic feature table found."
        )

        print(
            f"Loading: {feature_path}"
        )

        features = pd.read_csv(
            feature_path
        )

        return features

    print_header(
        "BUILDING GENOMIC FEATURE TABLE"
    )

    features = windows[
        ["chrom", "start", "end"]
    ].copy()

    features = compute_sequence_features(
        features
    )

    features = add_cpg_island_feature(
        features
    )

    features = add_tss_distance(
        features
    )

    features = add_repeat_feature(
        features
    )

    features = add_replication_timing(
        features
    )

    features.to_csv(
        feature_path,
        index=False,
    )

    print(
        f"\nSaved genomic features:\n"
        f"{feature_path}"
    )

    return features


# ============================================================
# 19. FEATURE QUALITY CONTROL
# ============================================================

def feature_quality_control(features):

    print_header(
        "FEATURE QUALITY CONTROL"
    )

    print(
        "\nFeature columns:"
    )

    for col in FEATURE_COLS:
        if col not in features.columns:
            raise ValueError(
                f"Required feature missing: {col}"
            )

        values = features[col]

        print(
            f"{col:25s}"
            f"min={values.min():.5g}  "
            f"median={values.median():.5g}  "
            f"max={values.max():.5g}  "
            f"missing={values.isna().sum():,}"
        )

    missing = features[
        FEATURE_COLS
    ].isna().any(axis=1)

    print(
        f"\nRows with complete features: "
        f"{(~missing).sum():,}"
    )

    print(
        f"Rows with missing features: "
        f"{missing.sum():,}"
    )

    return ~missing


# ============================================================
# 20. LOAD PHASE 9 COUPLINGS
# ============================================================

def load_phase9_couplings():

    print_header(
        "LOADING PHASE 9 COUPLINGS"
    )

    # Search likely locations.
    candidates = [
        RESULTS_DIR.parent.parent
        / "maxent"
        / "couplings.csv",

        PROJECT_ROOT
        / "results"
        / "maxent"
        / "couplings.csv",

        PROJECT_ROOT
        / "results"
        / "phase9"
        / "couplings.csv",

        PROJECT_ROOT
        / "results"
        / "maxent_models"
        / "couplings.csv",

        PROJECT_ROOT
        / "results"
        / "couplings.csv",
    ]

    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    if existing:

        path = existing[0]

        print(
            f"Loading Phase 9 couplings:\n"
            f"{path}"
        )

        return pd.read_csv(path)

    # Recursive fallback.
    possible = []

    for path in (
        PROJECT_ROOT / "results"
    ).rglob("*.csv"):

        name = path.name.lower()

        if (
            "coupling" in name
            and (
                "maxent" in str(path).lower()
                or "phase9" in str(path).lower()
            )
        ):

            possible.append(path)

    if len(possible) == 1:

        print(
            f"Loading Phase 9 couplings:\n"
            f"{possible[0]}"
        )

        return pd.read_csv(
            possible[0]
        )

    print(
        "\nWARNING:"
    )

    print(
        "Could not automatically find Phase 9 "
        "coupling table."
    )

    print(
        "Controlled analysis will still run."
    )

    print(
        "The full-J comparison will be skipped."
    )

    return None


# ============================================================
# 21. EXTRACT J VALUES
# ============================================================

def build_j_dictionary(j_table):

    if j_table is None:
        return None

    print(
        "\nDetected Phase 9 coupling columns:"
    )

    print(
        list(j_table.columns)
    )

    # Expected flexible format:
    #
    # mark_i, mark_j, H1, GM12878, K562, HepG2, IMR90
    #
    # Or:
    #
    # pair, H1, GM12878, ...

    lower = {
        str(c).lower(): c
        for c in j_table.columns
    }

    cell_columns = {}

    for cell in CELL_TYPES:

        if cell.lower() in lower:

            cell_columns[cell] = lower[
                cell.lower()
            ]

    if len(cell_columns) != len(CELL_TYPES):

        print(
            "\nCould not identify all cell-type "
            "columns automatically."
        )

        return None

    mark_i_col = None
    mark_j_col = None

    for name in [
        "mark_i",
        "mark1",
        "mark_a",
        "i",
    ]:

        if name in lower:
            mark_i_col = lower[name]
            break

    for name in [
        "mark_j",
        "mark2",
        "mark_b",
        "j",
    ]:

        if name in lower:
            mark_j_col = lower[name]
            break

    if mark_i_col is None or mark_j_col is None:

        # Try pair column.
        pair_col = None

        for name in [
            "pair",
            "interaction",
            "coupling",
        ]:

            if name in lower:
                pair_col = lower[name]
                break

        if pair_col is None:

            print(
                "\nCould not identify coupling-pair "
                "columns."
            )

            return None

        rows = []

        for _, row in j_table.iterrows():

            pair = str(
                row[pair_col]
            )

            separator = None

            for sep in [
                " - ",
                "-",
                "_",
                "|",
                ":",
            ]:

                if sep in pair:

                    separator = sep
                    break

            if separator is None:
                continue

            a, b = pair.split(
                separator,
                1,
            )

            rows.append(
                {
                    "mark_i": a.strip(),
                    "mark_j": b.strip(),
                    **{
                        cell: row[col]
                        for cell, col
                        in cell_columns.items()
                    },
                }
            )

        j_table = pd.DataFrame(rows)

        mark_i_col = "mark_i"
        mark_j_col = "mark_j"

    j_dict = {}

    for cell in CELL_TYPES:

        j_dict[cell] = {}

        col = cell_columns.get(
            cell,
            cell,
        )

        for _, row in j_table.iterrows():

            a = str(
                row[mark_i_col]
            ).strip()

            b = str(
                row[mark_j_col]
            ).strip()

            if a not in MARKS or b not in MARKS:
                continue

            value = float(
                row[col]
            )

            key = tuple(
                sorted([a, b])
            )

            j_dict[cell][key] = value

    return j_dict


# ============================================================
# 22. FIT LOGISTIC MODELS
# ============================================================

def fit_logistic_models(
    cell_type,
    df,
    features,
    complete_feature_mask,
):

    print_header(
        f"LOGISTIC SEQUENCE MODELS — {cell_type}"
    )

    coordinates = df[
        ["chrom", "start", "end"]
    ].reset_index(drop=True)

    sigma = df[
        MARKS
    ].to_numpy(
        dtype=np.float64
    )

    feature_matrix = features[
        FEATURE_COLS
    ].to_numpy(
        dtype=np.float64
    )

    # Remove incomplete-feature rows.
    usable = complete_feature_mask.copy()

    validation = (
        coordinates["chrom"]
        .isin(VALIDATION_CHROMS)
        .to_numpy()
    )

    training = (
        ~validation
        & usable
    )

    validation = (
        validation
        & usable
    )

    X_train = feature_matrix[
        training
    ]

    X_val = feature_matrix[
        validation
    ]

    print(
        f"Training windows: "
        f"{training.sum():,}"
    )

    print(
        f"Validation windows: "
        f"{validation.sum():,}"
    )

    results = []

    predictions = coordinates[
        validation
    ].copy()

    for mark_idx, mark in enumerate(
        MARKS
    ):

        print(
            f"\nFitting {mark}"
        )

        y = sigma[
            :,
            mark_idx
        ]

        y_train = y[
            training
        ]

        y_val = y[
            validation
        ]

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "logistic",
                    LogisticRegression(
                        C=LOGISTIC_C,
                        max_iter=MAX_ITER,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        p_val = model.predict_proba(
            X_val
        )[:, 1]

        residual = (
            y_val
            - p_val
        )

        predictions[
            f"{mark}_observed"
        ] = y_val

        predictions[
            f"{mark}_predicted"
        ] = p_val

        predictions[
            f"{mark}_residual"
        ] = residual

        # ----------------------------------------------------
        # Held-out metrics
        # ----------------------------------------------------

        if len(np.unique(y_val)) == 2:

            auc = roc_auc_score(
                y_val,
                p_val,
            )

        else:

            auc = np.nan

        ap = average_precision_score(
            y_val,
            p_val,
        )

        ll = log_loss(
            y_val,
            p_val,
            labels=[0, 1],
        )

        brier = brier_score_loss(
            y_val,
            p_val,
        )

        prevalence = y_val.mean()

        results.append(
            {
                "cell_type": cell_type,
                "mark": mark,
                "training_n": int(
                    training.sum()
                ),
                "validation_n": int(
                    validation.sum()
                ),
                "prevalence_validation": prevalence,
                "auc": auc,
                "average_precision": ap,
                "log_loss": ll,
                "brier_score": brier,
            }
        )

        # ----------------------------------------------------
        # Save coefficients
        # ----------------------------------------------------

        scaler = model.named_steps[
            "scaler"
        ]

        logistic = model.named_steps[
            "logistic"
        ]

        coefficient = (
            logistic.coef_[0]
        )

        coefficient_df = pd.DataFrame(
            {
                "feature": FEATURE_COLS,
                "coefficient_standardized": coefficient,
            }
        )

        coefficient_df[
            "cell_type"
        ] = cell_type

        coefficient_df[
            "mark"
        ] = mark

        coefficient_df[
            "intercept"
        ] = logistic.intercept_[0]

        coefficient_df[
            "feature_mean_training"
        ] = scaler.mean_

        coefficient_df[
            "feature_sd_training"
        ] = scaler.scale_

        coefficient_path = (
            MODEL_DIR
            / f"{cell_type}_{mark}_coefficients.csv"
        )

        coefficient_df.to_csv(
            coefficient_path,
            index=False,
        )

    # --------------------------------------------------------
    # Save held-out predictions/residuals
    # --------------------------------------------------------

    prediction_path = (
        RESIDUAL_DIR
        / f"{cell_type}_chr20_predictions_residuals.csv"
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nSaved validation predictions:\n"
        f"{prediction_path}"
    )

    metrics_df = pd.DataFrame(
        results
    )

    metrics_path = (
        MODEL_DIR
        / f"{cell_type}_logistic_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"Saved logistic metrics:\n"
        f"{metrics_path}"
    )

    return (
        metrics_df,
        predictions,
    )


# ============================================================
# 23. CONTROLLED PAIRWISE DEPENDENCIES
# ============================================================

def compute_controlled_dependencies(
    cell_type,
    predictions,
):

    print_header(
        f"CONTROLLED PAIRWISE DEPENDENCIES — {cell_type}"
    )

    residual_columns = [
        f"{mark}_residual"
        for mark in MARKS
    ]

    residuals = predictions[
        residual_columns
    ].to_numpy(
        dtype=float
    )

    residuals = np.nan_to_num(
        residuals,
        nan=0.0,
    )

    # Pearson residual correlation.
    corr = np.corrcoef(
        residuals,
        rowvar=False,
    )

    # Residual covariance.
    covariance = np.cov(
        residuals,
        rowvar=False,
        ddof=1,
    )

    rows = []

    for i, j in combinations(
        range(len(MARKS)),
        2,
    ):

        rows.append(
            {
                "cell_type": cell_type,
                "mark_i": MARKS[i],
                "mark_j": MARKS[j],
                "residual_correlation": corr[i, j],
                "residual_covariance": covariance[i, j],
            }
        )

    result = pd.DataFrame(
        rows
    )

    path = (
        CONTROLLED_DIR
        / f"{cell_type}_controlled_dependencies.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved controlled dependencies:\n"
        f"{path}"
    )

    # Full matrix.
    corr_matrix = pd.DataFrame(
        corr,
        index=MARKS,
        columns=MARKS,
    )

    corr_matrix.to_csv(
        CONTROLLED_DIR
        / f"{cell_type}_controlled_correlation_matrix.csv"
    )

    return result, corr_matrix


# ============================================================
# 24. COMPARE FULL J WITH CONTROLLED DEPENDENCIES
# ============================================================

def compare_full_vs_controlled(
    controlled,
    j_dict,
):

    if j_dict is None:

        print(
            "\nSkipping full-J comparison."
        )

        return None

    print_header(
        "FULL MAXENT J VS SEQUENCE-CONTROLLED DEPENDENCIES"
    )

    rows = []

    for cell in CELL_TYPES:

        cell_controlled = controlled[
            controlled["cell_type"] == cell
        ]

        for _, row in cell_controlled.iterrows():

            key = tuple(
                sorted(
                    [
                        row["mark_i"],
                        row["mark_j"],
                    ]
                )
            )

            j_value = (
                j_dict
                .get(cell, {})
                .get(key, np.nan)
            )

            controlled_value = (
                row[
                    "residual_correlation"
                ]
            )

            if np.isnan(j_value):

                sign_agreement = np.nan

            else:

                sign_agreement = (
                    np.sign(j_value)
                    == np.sign(
                        controlled_value
                    )
                )

            rows.append(
                {
                    "cell_type": cell,
                    "mark_i": row["mark_i"],
                    "mark_j": row["mark_j"],
                    "J_full": j_value,
                    "controlled_residual_correlation":
                        controlled_value,
                    "controlled_residual_covariance":
                        row[
                            "residual_covariance"
                        ],
                    "same_sign":
                        sign_agreement,
                }
            )

    comparison = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Per-cell correlations
    # --------------------------------------------------------

    summary_rows = []

    for cell in CELL_TYPES:

        sub = comparison[
            comparison["cell_type"] == cell
        ].dropna(
            subset=[
                "J_full",
                "controlled_residual_correlation",
            ]
        )

        if len(sub) >= 3:

            pearson = np.corrcoef(
                sub["J_full"].to_numpy(),
                sub[
                    "controlled_residual_correlation"
                ].to_numpy(),
            )[0, 1]

            from scipy.stats import spearmanr

            spearman = spearmanr(
                sub["J_full"],
                sub[
                    "controlled_residual_correlation"
                ],
            ).statistic

            sign_fraction = (
                sub["same_sign"]
                .astype(float)
                .mean()
            )

        else:

            pearson = np.nan
            spearman = np.nan
            sign_fraction = np.nan

        summary_rows.append(
            {
                "cell_type": cell,
                "n_pairs": len(sub),
                "pearson_J_vs_controlled":
                    pearson,
                "spearman_J_vs_controlled":
                    spearman,
                "fraction_same_sign":
                    sign_fraction,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    comparison.to_csv(
        CONTROLLED_DIR
        / "full_vs_controlled_all_pairs.csv",
        index=False,
    )

    summary.to_csv(
        CONTROLLED_DIR
        / "full_vs_controlled_summary.csv",
        index=False,
    )

    print(
        "\nFull vs controlled summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return comparison


# ============================================================
# 25. CROSS-CELL CONTROLLED MATRIX
# ============================================================

def build_cross_cell_controlled_table(
    controlled
):

    print_header(
        "CROSS-CELL-TYPE CONTROLLED DEPENDENCIES"
    )

    table = controlled.copy()

    table["pair"] = (
        table["mark_i"]
        + " - "
        + table["mark_j"]
    )

    wide = table.pivot(
        index="pair",
        columns="cell_type",
        values="residual_correlation",
    )

    # Keep expected order.
    wide = wide.reindex(
        columns=CELL_TYPES
    )

    path = (
        CONTROLLED_DIR
        / "cross_cell_controlled_dependencies.csv"
    )

    wide.to_csv(
        path
    )

    print(
        f"Saved:\n{path}"
    )

    return wide


# ============================================================
# 26. PLOT LOGISTIC PERFORMANCE
# ============================================================

def plot_logistic_performance(
    metrics_all
):

    print(
        "\nCreating logistic performance figure..."
    )

    auc_table = metrics_all.pivot(
        index="mark",
        columns="cell_type",
        values="auc",
    )

    plt.figure(
        figsize=(10, 6)
    )

    for cell in CELL_TYPES:

        plt.plot(
            MARKS,
            auc_table[cell],
            marker="o",
            label=cell,
        )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Held-out ROC AUC"
    )

    plt.xlabel(
        "Histone mark"
    )

    plt.title(
        "Sequence / Genomic-Context Prediction"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FIGURE_DIR
        / "phase12_logistic_auc.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# 27. PLOT CONTROLLED DEPENDENCIES
# ============================================================

def plot_controlled_heatmaps(
    controlled
):

    print(
        "\nCreating controlled dependency heatmaps..."
    )

    for cell in CELL_TYPES:

        sub = controlled[
            controlled["cell_type"] == cell
        ]

        matrix = np.zeros(
            (len(MARKS), len(MARKS))
        )

        for _, row in sub.iterrows():

            i = MARKS.index(
                row["mark_i"]
            )

            j = MARKS.index(
                row["mark_j"]
            )

            value = row[
                "residual_correlation"
            ]

            matrix[i, j] = value
            matrix[j, i] = value

        np.fill_diagonal(
            matrix,
            0.0,
        )

        plt.figure(
            figsize=(8, 7)
        )

        vmax = np.max(
            np.abs(matrix)
        )

        if vmax == 0:
            vmax = 1

        plt.imshow(
            matrix,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )

        plt.xticks(
            range(len(MARKS)),
            MARKS,
            rotation=45,
            ha="right",
        )

        plt.yticks(
            range(len(MARKS)),
            MARKS,
        )

        plt.colorbar(
            label="Residual correlation"
        )

        plt.title(
            f"{cell} — Sequence-Controlled Dependencies"
        )

        plt.tight_layout()

        path = (
            FIGURE_DIR
            / f"{cell}_controlled_heatmap.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved: {path}"
        )


# ============================================================
# 28. PLOT FULL J VS CONTROLLED
# ============================================================

def plot_full_vs_controlled(
    comparison
):

    if comparison is None:
        return

    print(
        "\nCreating full-J vs controlled plots..."
    )

    for cell in CELL_TYPES:

        sub = comparison[
            comparison["cell_type"] == cell
        ].dropna(
            subset=[
                "J_full",
                "controlled_residual_correlation",
            ]
        )

        if len(sub) < 3:
            continue

        plt.figure(
            figsize=(7, 6)
        )

        plt.scatter(
            sub["J_full"],
            sub[
                "controlled_residual_correlation"
            ],
        )

        plt.axhline(
            0,
            linewidth=1,
        )

        plt.axvline(
            0,
            linewidth=1,
        )

        plt.xlabel(
            "Full MaxEnt coupling J"
        )

        plt.ylabel(
            "Sequence-controlled residual correlation"
        )

        plt.title(
            f"{cell}: Full vs Sequence-Controlled"
        )

        plt.tight_layout()

        path = (
            FIGURE_DIR
            / f"{cell}_full_vs_controlled.png"
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved: {path}"
        )


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

def write_phase12_summary(
    metrics_all,
    controlled,
    comparison,
):

    print_header(
        "WRITING PHASE 12 SUMMARY"
    )

    summary_path = (
        RESULTS_DIR
        / "PHASE12_SUMMARY.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 12 — SEQUENCE / GENOMIC CONTEXT CONTROL\n"
        )

        f.write(
            "Epigenenomic Project\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "ANALYSIS DESIGN\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Cell types: {', '.join(CELL_TYPES)}\n"
        )

        f.write(
            f"Histone marks: {', '.join(MARKS)}\n"
        )

        f.write(
            f"Validation chromosome(s): "
            f"{', '.join(sorted(VALIDATION_CHROMS))}\n"
        )

        f.write(
            "Sequence/context features:\n"
        )

        for feature in FEATURE_COLS:
            f.write(
                f"  - {feature}\n"
            )

        f.write(
            "\n"
        )

        # ----------------------------------------------------
        # Logistic metrics
        # ----------------------------------------------------

        f.write(
            "HELD-OUT LOGISTIC PREDICTION\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell in CELL_TYPES:

            sub = metrics_all[
                metrics_all["cell_type"]
                == cell
            ]

            f.write(
                f"\n{cell}\n"
            )

            for _, row in sub.iterrows():

                f.write(
                    f"  {row['mark']}: "
                    f"AUC={row['auc']:.4f}, "
                    f"AP={row['average_precision']:.4f}, "
                    f"logloss={row['log_loss']:.4f}, "
                    f"Brier={row['brier_score']:.4f}\n"
                )

        # ----------------------------------------------------
        # Controlled dependencies
        # ----------------------------------------------------

        f.write(
            "\n"
        )

        f.write(
            "CONTROLLED PAIRWISE DEPENDENCIES\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell in CELL_TYPES:

            sub = controlled[
                controlled["cell_type"]
                == cell
            ]

            f.write(
                f"\n{cell}\n"
            )

            sub = sub.sort_values(
                "residual_correlation",
                key=lambda x: np.abs(x),
                ascending=False,
            )

            for _, row in sub.head(10).iterrows():

                f.write(
                    f"  "
                    f"{row['mark_i']} - "
                    f"{row['mark_j']}: "
                    f"r={row['residual_correlation']:.5f}\n"
                )

        # ----------------------------------------------------
        # Full vs controlled
        # ----------------------------------------------------

        if comparison is not None:

            f.write(
                "\n"
            )

            f.write(
                "FULL MAXENT J VS CONTROLLED DEPENDENCY\n"
            )

            f.write(
                "-" * 70
                + "\n"
            )

            summary = (
                pd.read_csv(
                    CONTROLLED_DIR
                    / "full_vs_controlled_summary.csv"
                )
            )

            for _, row in summary.iterrows():

                f.write(
                    f"{row['cell_type']}: "
                    f"Pearson="
                    f"{row['pearson_J_vs_controlled']:.4f}, "
                    f"Spearman="
                    f"{row['spearman_J_vs_controlled']:.4f}, "
                    f"same-sign fraction="
                    f"{row['fraction_same_sign']:.4f}\n"
                )

        # ----------------------------------------------------
        # Interpretation note
        # ----------------------------------------------------

        f.write(
            "\n"
        )

        f.write(
            "INTERPRETATION NOTE\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "The logistic models estimate the probability of each "
            "histone mark from genomic-context features.\n"
        )

        f.write(
            "Residuals are defined as observed binary state minus "
            "held-out predicted probability.\n"
        )

        f.write(
            "Residual correlations quantify pairwise dependency "
            "remaining after genomic-context prediction.\n"
        )

        f.write(
            "Residual correlations are not numerically equivalent "
            "to pairwise maximum-entropy J parameters.\n"
        )

        f.write(
            "Therefore they should not be reported as controlled "
            "J values unless a separate controlled MaxEnt model "
            "is subsequently fitted.\n"
        )

    print(
        f"\nSaved final summary:\n"
        f"{summary_path}"
    )


# ============================================================
# 30. MAIN PIPELINE
# ============================================================

def main():

    print_header(
        "PHASE 12 — COMPLETE SEQUENCE CONTROL PIPELINE"
    )

    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nOutput directory:\n{RESULTS_DIR}"
    )

    # --------------------------------------------------------
    # STEP 1
    # Load all five cell types.
    # --------------------------------------------------------

    print_header(
        "STEP 1 — LOADING CELL-TYPE MATRICES"
    )

    data = {}

    for cell in CELL_TYPES:

        data[cell] = load_cell_matrix(
            cell
        )

    # --------------------------------------------------------
    # STEP 2
    # Verify shared windows.
    # --------------------------------------------------------

    verify_shared_windows(
        data
    )

    # --------------------------------------------------------
    # STEP 3
    # Common genomic coordinates.
    # --------------------------------------------------------

    windows = data[
        CELL_TYPES[0]
    ][
        ["chrom", "start", "end"]
    ].copy()

    # --------------------------------------------------------
    # STEP 4
    # Build genomic features once.
    # --------------------------------------------------------

    features = build_genomic_features(
        windows
    )

    # --------------------------------------------------------
    # STEP 5
    # QC features.
    # --------------------------------------------------------

    complete_feature_mask = (
        feature_quality_control(
            features
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # Load Phase 9 J values.
    # --------------------------------------------------------

    j_table = load_phase9_couplings()

    j_dict = build_j_dictionary(
        j_table
    )

    # --------------------------------------------------------
    # STEP 7
    # Logistic models for every cell type.
    # --------------------------------------------------------

    all_metrics = []
    all_controlled = []

    predictions_by_cell = {}

    for cell in CELL_TYPES:

        metrics, predictions = (
            fit_logistic_models(
                cell_type=cell,
                df=data[cell],
                features=features,
                complete_feature_mask=
                    complete_feature_mask,
            )
        )

        all_metrics.append(
            metrics
        )

        predictions_by_cell[cell] = (
            predictions
        )

        controlled, _ = (
            compute_controlled_dependencies(
                cell_type=cell,
                predictions=predictions,
            )
        )

        all_controlled.append(
            controlled
        )

    # --------------------------------------------------------
    # STEP 8
    # Combine logistic metrics.
    # --------------------------------------------------------

    metrics_all = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    metrics_path = (
        MODEL_DIR
        / "ALL_CELL_TYPES_LOGISTIC_METRICS.csv"
    )

    metrics_all.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"\nSaved combined logistic metrics:\n"
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # STEP 9
    # Combine controlled dependencies.
    # --------------------------------------------------------

    controlled_all = pd.concat(
        all_controlled,
        ignore_index=True,
    )

    controlled_path = (
        CONTROLLED_DIR
        / "ALL_CELL_TYPES_CONTROLLED_DEPENDENCIES.csv"
    )

    controlled_all.to_csv(
        controlled_path,
        index=False,
    )

    print(
        f"\nSaved combined controlled dependencies:\n"
        f"{controlled_path}"
    )

    # --------------------------------------------------------
    # STEP 10
    # Cross-cell table.
    # --------------------------------------------------------

    build_cross_cell_controlled_table(
        controlled_all
    )

    # --------------------------------------------------------
    # STEP 11
    # Compare full MaxEnt J with controlled dependency.
    # --------------------------------------------------------

    comparison = (
        compare_full_vs_controlled(
            controlled=controlled_all,
            j_dict=j_dict,
        )
    )

    # --------------------------------------------------------
    # STEP 12
    # Figures.
    # --------------------------------------------------------

    plot_logistic_performance(
        metrics_all
    )

    plot_controlled_heatmaps(
        controlled_all
    )

    plot_full_vs_controlled(
        comparison
    )

    # --------------------------------------------------------
    # STEP 13
    # Final summary.
    # --------------------------------------------------------

    write_phase12_summary(
        metrics_all=metrics_all,
        controlled=controlled_all,
        comparison=comparison,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print_header(
        "PHASE 12 COMPLETE"
    )

    print(
        "\nMain output directory:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nGenerated:"
    )

    print(
        "  genomic_features/"
    )

    print(
        "  logistic_models/"
    )

    print(
        "  residuals/"
    )

    print(
        "  controlled_dependencies/"
    )

    print(
        "  figures/"
    )

    print(
        "  PHASE12_SUMMARY.txt"
    )

    print(
        "\nPHASE 12 SEQUENCE CONTROL COMPLETE."
    )
if __name__ == "__main__":
    main()