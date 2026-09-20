"""
PHASE 14 — FINAL QC + PREPRINT V1 FIGURE PACKAGE
Epigenenomic Project

Purpose
-------
Final computational quality control after Phases 9–13.

This script DOES NOT refit the models.

It:
1. Verifies all Phase 12 outputs.
2. Verifies all Phase 13 outputs.
3. Reads the EXACT Phase 12/13 CSV schemas.
4. Rebuilds publication figures directly from those outputs.
5. Refuses to save a figure if its plotted data are empty/non-finite.
6. Checks the Phase 12 CpG-island feature.
7. Reconnects Phase 12 controlled dependencies to Phase 9 J
   when a compatible Phase 9 coupling table exists.
8. Produces final tables and a final QC report.

No new biological inference is introduced here.

Outputs
-------
results/
    final_qc/
        figures/
        tables/
        FINAL_QC_REPORT.txt
        PHASE14_FILE_AUDIT.csv
        PHASE14_FIGURE_AUDIT.csv
"""

# ============================================================
# 0. IMPORTS
# ============================================================

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

PHASE12_DIR = RESULTS_DIR / "sequence_control"
PHASE12_FEATURE_DIR = PHASE12_DIR / "genomic_features"
PHASE12_MODEL_DIR = PHASE12_DIR / "logistic_models"
PHASE12_RESIDUAL_DIR = PHASE12_DIR / "residuals"
PHASE12_CONTROLLED_DIR = PHASE12_DIR / "controlled_dependencies"
PHASE12_FIGURE_DIR = PHASE12_DIR / "figures"

PHASE13_DIR = RESULTS_DIR / "validation"
PHASE13_BOOTSTRAP_DIR = PHASE13_DIR / "bootstrap"
PHASE13_CV_DIR = PHASE13_DIR / "cross_validation"
PHASE13_THRESHOLD_DIR = PHASE13_DIR / "threshold_sensitivity"
PHASE13_WINDOW_DIR = PHASE13_DIR / "window_size_sensitivity"
PHASE13_FIGURE_DIR = PHASE13_DIR / "figures"

FINAL_DIR = RESULTS_DIR / "final_qc"
FINAL_FIGURE_DIR = FINAL_DIR / "figures"
FINAL_TABLE_DIR = FINAL_DIR / "tables"

for directory in [
    FINAL_DIR,
    FINAL_FIGURE_DIR,
    FINAL_TABLE_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


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

PAIRS = [
    (a, b)
    for a, b in combinations(MARKS, 2)
]

PAIR_NAMES = [
    f"{a}__{b}"
    for a, b in PAIRS
]


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path, description):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMISSING REQUIRED FILE\n"
            f"{description}\n"
            f"{path}\n"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"\nFILE IS EMPTY\n"
            f"{path}\n"
        )

    return path


def require_columns(df, columns, path):
    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMISSING REQUIRED COLUMNS\n"
            f"File: {path}\n"
            f"Missing: {missing}\n"
            f"Available: {list(df.columns)}"
        )


def finite_series(df, column, path):
    values = pd.to_numeric(
        df[column],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if len(values) == 0:
        raise RuntimeError(
            f"{path}: column {column} contains zero rows."
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{path}: column {column} contains "
            f"NaN or infinite values."
        )

    return values


def save_figure_checked(
    path,
    required_n,
    arrays,
):
    """
    Save figure only after checking that the plotted data
    contain at least required_n finite observations.
    """

    counts = []

    for name, values in arrays.items():

        values = np.asarray(
            values,
            dtype=float,
        ).ravel()

        finite = np.isfinite(values)

        n = int(
            finite.sum()
        )

        counts.append(
            (
                name,
                n,
            )
        )

        if n < required_n:
            raise RuntimeError(
                f"\nFIGURE DATA CHECK FAILED\n"
                f"Figure: {path.name}\n"
                f"Series: {name}\n"
                f"Finite observations: {n}\n"
                f"Required: {required_n}"
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    if not path.exists():
        raise RuntimeError(
            f"Figure was not created: {path}"
        )

    if path.stat().st_size < 5000:
        raise RuntimeError(
            f"Figure appears suspiciously small: {path}"
        )

    return counts


# ============================================================
# 4. FILE AUDIT
# ============================================================

def audit_phase12_phase13_files():

    header(
        "PHASE 14 — AUDITING PHASE 12 AND PHASE 13"
    )

    rows = []

    required_phase12 = {
        "Phase12_summary":
            PHASE12_DIR / "PHASE12_SUMMARY.txt",

        "Phase12_features":
            PHASE12_FEATURE_DIR /
            "all_windows_genomic_features.csv",

        "Phase12_metrics":
            PHASE12_MODEL_DIR /
            "ALL_CELL_TYPES_LOGISTIC_METRICS.csv",

        "Phase12_controlled":
            PHASE12_CONTROLLED_DIR /
            "ALL_CELL_TYPES_CONTROLLED_DEPENDENCIES.csv",
    }

    required_phase13 = {
        "Phase13_summary":
            PHASE13_DIR / "PHASE13_SUMMARY.txt",
    }

    for cell in CELL_TYPES:

        required_phase12[
            f"{cell}_logistic_metrics"
        ] = (
            PHASE12_MODEL_DIR /
            f"{cell}_logistic_metrics.csv"
        )

        required_phase12[
            f"{cell}_controlled"
        ] = (
            PHASE12_CONTROLLED_DIR /
            f"{cell}_controlled_dependencies.csv"
        )

        required_phase12[
            f"{cell}_predictions"
        ] = (
            PHASE12_RESIDUAL_DIR /
            f"{cell}_chr20_predictions_residuals.csv"
        )

        required_phase13[
            f"{cell}_bootstrap"
        ] = (
            PHASE13_BOOTSTRAP_DIR /
            f"{cell}_bootstrap_couplings.csv"
        )

        required_phase13[
            f"{cell}_bootstrap_stability"
        ] = (
            PHASE13_BOOTSTRAP_DIR /
            f"{cell}_bootstrap_stability.csv"
        )

        required_phase13[
            f"{cell}_cv"
        ] = (
            PHASE13_CV_DIR /
            f"{cell}_chromosome_cv.csv"
        )

        required_phase13[
            f"{cell}_threshold"
        ] = (
            PHASE13_THRESHOLD_DIR /
            f"{cell}_coupling_threshold_sensitivity.csv"
        )

        required_phase13[
            f"{cell}_window_size"
        ] = (
            PHASE13_WINDOW_DIR /
            f"{cell}_window_size_summary.csv"
        )

    for label, path in {
        **required_phase12,
        **required_phase13,
    }.items():

        exists = path.exists()
        size = (
            path.stat().st_size
            if exists
            else 0
        )

        status = (
            "PASS"
            if exists and size > 0
            else "FAIL"
        )

        print(
            f"{status:5s} "
            f"{label:35s} "
            f"{path}"
        )

        rows.append(
            {
                "phase": (
                    "12"
                    if label.startswith("Phase12")
                    else "13"
                ),
                "label": label,
                "path": str(path),
                "exists": exists,
                "size_bytes": size,
                "status": status,
            }
        )

    audit = pd.DataFrame(rows)

    audit.to_csv(
        FINAL_DIR /
        "PHASE14_FILE_AUDIT.csv",
        index=False,
    )

    failed = audit[
        audit["status"] != "PASS"
    ]

    if len(failed) > 0:

        raise RuntimeError(
            "\nPhase 14 stopped because required "
            "Phase 12/13 outputs are missing.\n"
            + "\n".join(
                failed["path"].tolist()
            )
        )

    print(
        "\nAll required Phase 12/13 files exist."
    )

    return audit


# ============================================================
# 5. LOAD PHASE 12 DATA
# ============================================================

def load_phase12():

    header(
        "LOADING EXACT PHASE 12 OUTPUTS"
    )

    feature_path = (
        PHASE12_FEATURE_DIR /
        "all_windows_genomic_features.csv"
    )

    metrics_path = (
        PHASE12_MODEL_DIR /
        "ALL_CELL_TYPES_LOGISTIC_METRICS.csv"
    )

    controlled_path = (
        PHASE12_CONTROLLED_DIR /
        "ALL_CELL_TYPES_CONTROLLED_DEPENDENCIES.csv"
    )

    features = pd.read_csv(
        feature_path
    )

    metrics = pd.read_csv(
        metrics_path
    )

    controlled = pd.read_csv(
        controlled_path
    )

    require_columns(
        features,
        [
            "chrom",
            "start",
            "end",
            "gc_content",
            "cpg_density",
            "cpg_island_overlap",
            "log_dist_to_tss",
            "repeat_density",
        ],
        feature_path,
    )

    require_columns(
        metrics,
        [
            "cell_type",
            "mark",
            "training_n",
            "validation_n",
            "prevalence_validation",
            "auc",
            "average_precision",
            "log_loss",
            "brier_score",
        ],
        metrics_path,
    )

    require_columns(
        controlled,
        [
            "cell_type",
            "mark_i",
            "mark_j",
            "residual_correlation",
            "residual_covariance",
        ],
        controlled_path,
    )

    print(
        f"Features: {len(features):,} rows"
    )

    print(
        f"Logistic metrics: "
        f"{len(metrics):,} rows"
    )

    print(
        f"Controlled dependencies: "
        f"{len(controlled):,} rows"
    )

    return (
        features,
        metrics,
        controlled,
    )


# ============================================================
# 6. PHASE 12 FIGURE 1
# LOGISTIC PREDICTION PERFORMANCE
# ============================================================

def figure_phase12_logistic(
    metrics,
):

    header(
        "FIGURE — PHASE 12 LOGISTIC PERFORMANCE"
    )

    df = metrics.copy()

    auc = (
        df.groupby("cell_type")["auc"]
        .mean()
        .reindex(CELL_TYPES)
    )

    ap = (
        df.groupby("cell_type")["average_precision"]
        .mean()
        .reindex(CELL_TYPES)
    )

    if auc.isna().any():
        raise RuntimeError(
            "Phase 12 AUC data contain missing "
            "cell-type means."
        )

    if ap.isna().any():
        raise RuntimeError(
            "Phase 12 AP data contain missing "
            "cell-type means."
        )

    x = np.arange(
        len(CELL_TYPES)
    )

    width = 0.38

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        auc.to_numpy(),
        width,
        label="Mean AUC",
    )

    plt.bar(
        x + width / 2,
        ap.to_numpy(),
        width,
        label="Mean average precision",
    )

    plt.xticks(
        x,
        CELL_TYPES,
    )

    plt.ylabel(
        "Held-out prediction performance"
    )

    plt.ylim(
        0,
        1,
    )

    plt.title(
        "Phase 12: genomic-context prediction "
        "of histone-mark states"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_1_Phase12_logistic_performance.png"
    )

    save_figure_checked(
        path,
        required_n=5,
        arrays={
            "AUC": auc.to_numpy(),
            "average_precision": ap.to_numpy(),
        },
    )

    print(
        f"PASS: {path}"
    )


# ============================================================
# 7. PHASE 12 FIGURE 2
# CONTROLLED DEPENDENCY HEATMAPS
# ============================================================

def make_symmetric_matrix(
    df,
    value_column,
    cell,
):

    sub = df[
        df["cell_type"] == cell
    ].copy()

    matrix = np.zeros(
        (
            len(MARKS),
            len(MARKS),
        ),
        dtype=float,
    )

    for _, row in sub.iterrows():

        a = row["mark_i"]
        b = row["mark_j"]

        if (
            a not in MARKS
            or
            b not in MARKS
        ):
            continue

        i = MARKS.index(a)
        j = MARKS.index(b)

        value = float(
            row[value_column]
        )

        matrix[i, j] = value
        matrix[j, i] = value

    return matrix


def figure_phase12_controlled_heatmaps(
    controlled,
):

    header(
        "FIGURE — PHASE 12 CONTROLLED DEPENDENCY HEATMAPS"
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5),
    )

    all_values = []

    matrices = {}

    for cell in CELL_TYPES:

        matrix = make_symmetric_matrix(
            controlled,
            "residual_correlation",
            cell,
        )

        matrices[cell] = matrix

        upper = matrix[
            np.triu_indices(
                len(MARKS),
                k=1,
            )
        ]

        if len(upper) != 21:
            raise RuntimeError(
                f"{cell}: expected 21 pairwise "
                f"dependencies."
            )

        if not np.isfinite(
            upper
        ).all():
            raise RuntimeError(
                f"{cell}: controlled dependency "
                f"matrix contains non-finite values."
            )

        all_values.extend(
            upper.tolist()
        )

    vmax = max(
        abs(v)
        for v in all_values
    )

    if vmax <= 0:
        raise RuntimeError(
            "All controlled dependencies are zero."
        )

    for ax, cell in zip(
        axes,
        CELL_TYPES,
    ):

        matrix = matrices[cell]

        im = ax.imshow(
            matrix,
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set_title(
            cell
        )

        ax.set_xticks(
            range(len(MARKS))
        )

        ax.set_yticks(
            range(len(MARKS))
        )

        ax.set_xticklabels(
            MARKS,
            rotation=90,
            fontsize=8,
        )

        ax.set_yticklabels(
            MARKS,
            fontsize=8,
        )

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.75,
        label="Residual correlation",
    )

    fig.suptitle(
        "Phase 12: sequence-controlled pairwise dependencies"
    )

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_2_Phase12_controlled_heatmaps.png"
    )

    save_figure_checked(
        path,
        required_n=21,
        arrays={
            "all_21_pairs_across_cells":
                np.asarray(all_values)
        },
    )

    print(
        f"PASS: {path}"
    )

    return matrices


# ============================================================
# 8. PHASE 12 FIGURE 3
# FULL VS CONTROLLED
# ============================================================

def find_phase9_coupling_files():

    header(
        "SEARCHING FOR PHASE 9 COUPLING OUTPUT"
    )

    candidates = []

    roots = [
        RESULTS_DIR,
        RESULTS_DIR / "maxent",
        RESULTS_DIR / "phase9",
        RESULTS_DIR / "maxent_models",
    ]

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*.csv"
        ):

            name = path.name.lower()

            if (
                "coupl" in name
                or
                "j_matrix" in name
                or
                "parameters" in name
            ):
                candidates.append(
                    path
                )

    candidates = sorted(
        set(candidates)
    )

    for path in candidates:
        print(
            f"Candidate: {path}"
        )

    return candidates


def load_phase9_couplings():

    candidates = (
        find_phase9_coupling_files()
    )

    if not candidates:
        print(
            "\nWARNING: No Phase 9 coupling CSV "
            "was found automatically."
        )
        return None

    # --------------------------------------------------------
    # Look for a table with all five cell types.
    # --------------------------------------------------------

    for path in candidates:

        try:
            df = pd.read_csv(
                path
            )
        except Exception:
            continue

        lower = {
            str(c).lower(): c
            for c in df.columns
        }

        cells_present = [
            cell
            for cell in CELL_TYPES
            if cell.lower() in lower
        ]

        if len(cells_present) == 5:

            print(
                f"\nUsing Phase 9 coupling table:"
            )

            print(
                path
            )

            return df

    print(
        "\nWARNING: Coupling CSVs were found, "
        "but none contains all five cell types "
        "in a directly usable table."
    )

    return None


# ============================================================
# 9. NORMALIZE PHASE 9 COUPLING TABLE
# ============================================================

def normalize_phase9_table(
    df,
):

    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    cell_columns = {}

    for cell in CELL_TYPES:

        key = cell.lower()

        if key in lower:
            cell_columns[cell] = (
                lower[key]
            )

    if len(cell_columns) != 5:
        return None

    mark_i = None
    mark_j = None

    for name in [
        "mark_i",
        "mark1",
        "mark_a",
    ]:

        if name in lower:
            mark_i = lower[name]
            break

    for name in [
        "mark_j",
        "mark2",
        "mark_b",
    ]:

        if name in lower:
            mark_j = lower[name]
            break

    # --------------------------------------------------------
    # Pair column fallback.
    # --------------------------------------------------------

    if mark_i is None or mark_j is None:

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
            return None

        rows = []

        for _, row in df.iterrows():

            pair = str(
                row[pair_col]
            )

            separator = None

            for sep in [
                "__",
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

            a = a.strip()
            b = b.strip()

            if (
                a not in MARKS
                or
                b not in MARKS
            ):
                continue

            item = {
                "mark_i": a,
                "mark_j": b,
            }

            for cell in CELL_TYPES:
                item[cell] = row[
                    cell_columns[cell]
                ]

            rows.append(
                item
            )

        df = pd.DataFrame(
            rows
        )

        mark_i = "mark_i"
        mark_j = "mark_j"

    else:

        df = df.rename(
            columns={
                mark_i: "mark_i",
                mark_j: "mark_j",
            }
        )

    rows = []

    for _, row in df.iterrows():

        a = str(
            row["mark_i"]
        ).strip()

        b = str(
            row["mark_j"]
        ).strip()

        if (
            a not in MARKS
            or
            b not in MARKS
        ):
            continue

        for cell in CELL_TYPES:

            value = pd.to_numeric(
                row[cell],
                errors="coerce",
            )

            rows.append(
                {
                    "cell_type": cell,
                    "mark_i": a,
                    "mark_j": b,
                    "J_full": value,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        return None

    result["pair"] = (
        result["mark_i"]
        + "__"
        + result["mark_j"]
    )

    return result


# ============================================================
# 10. FULL VS CONTROLLED TABLE
# ============================================================

def build_full_vs_controlled(
    controlled,
    phase9,
):

    if phase9 is None:
        print(
            "\nFull-vs-controlled comparison skipped."
        )
        return None

    full = normalize_phase9_table(
        phase9
    )

    if full is None:
        print(
            "\nPhase 9 table could not be normalized."
        )
        return None

    merged = controlled.copy()

    merged["pair"] = (
        merged["mark_i"]
        + "__"
        + merged["mark_j"]
    )

    merged = merged.merge(
        full[
            [
                "cell_type",
                "pair",
                "J_full",
            ]
        ],
        on=[
            "cell_type",
            "pair",
        ],
        how="left",
    )

    merged[
        "absolute_difference"
    ] = (
        merged["J_full"]
        -
        merged[
            "residual_correlation"
        ]
    ).abs()

    merged[
        "same_sign"
    ] = (
        np.sign(
            merged["J_full"]
        )
        ==
        np.sign(
            merged[
                "residual_correlation"
            ]
        )
    )

    output = (
        FINAL_TABLE_DIR /
        "Phase12_full_vs_controlled.csv"
    )

    merged.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved: {output}"
    )

    return merged


# ============================================================
# 11. FULL VS CONTROLLED FIGURE
# ============================================================

def figure_full_vs_controlled(
    comparison,
):

    if comparison is None:
        print(
            "\nSkipping full-vs-controlled figure "
            "because Phase 9 J was not available."
        )
        return

    valid = comparison[
        comparison["J_full"].notna()
        &
        comparison[
            "residual_correlation"
        ].notna()
    ].copy()

    if len(valid) == 0:
        raise RuntimeError(
            "Full-vs-controlled table contains "
            "zero usable observations."
        )

    plt.figure(
        figsize=(9, 7)
    )

    for cell in CELL_TYPES:

        sub = valid[
            valid["cell_type"] == cell
        ]

        if len(sub) == 0:
            continue

        plt.scatter(
            sub["J_full"],
            sub[
                "residual_correlation"
            ],
            label=cell,
            alpha=0.8,
        )

    xmin = valid["J_full"].min()
    xmax = valid["J_full"].max()

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Full MaxEnt coupling J"
    )

    plt.ylabel(
        "Sequence-controlled residual correlation"
    )

    plt.title(
        "Full MaxEnt versus sequence-controlled dependency"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_3_Phase12_full_vs_controlled.png"
    )

    save_figure_checked(
        path,
        required_n=10,
        arrays={
            "J_full":
                valid["J_full"].to_numpy(),
            "controlled":
                valid[
                    "residual_correlation"
                ].to_numpy(),
        },
    )

    print(
        f"PASS: {path}"
    )


# ============================================================
# 12. CpG QC
# ============================================================

def cpg_quality_control(
    features,
):

    header(
        "PHASE 12 CpG-ISLAND QUALITY CONTROL"
    )

    values = finite_series(
        features,
        "cpg_island_overlap",
        "all_windows_genomic_features.csv",
    )

    n_nonzero = int(
        np.sum(
            values > 0
        )
    )

    print(
        f"Total windows: {len(values):,}"
    )

    print(
        f"Non-zero CpG-island overlaps: "
        f"{n_nonzero:,}"
    )

    print(
        f"Maximum overlap: "
        f"{values.max():.8g}"
    )

    print(
        f"Median overlap: "
        f"{np.median(values):.8g}"
    )

    if n_nonzero == 0:

        print()
        print(
            "WARNING:"
        )

        print(
            "The Phase 12 CpG-island-overlap "
            "feature is exactly zero for every window."
        )

        print(
            "This feature must NOT be interpreted "
            "as informative in Preprint V1."
        )

        status = "WARNING_ALL_ZERO"

    else:

        print(
            "CpG-island feature contains "
            "non-zero values."
        )

        status = "PASS"

    qc = pd.DataFrame(
        [
            {
                "feature":
                    "cpg_island_overlap",
                "n_windows":
                    len(values),
                "n_nonzero":
                    n_nonzero,
                "median":
                    np.median(values),
                "max":
                    values.max(),
                "status":
                    status,
            }
        ]
    )

    qc.to_csv(
        FINAL_TABLE_DIR /
        "Phase12_feature_QC.csv",
        index=False,
    )

    return status


# ============================================================
# 13. PHASE 13 BOOTSTRAP FIGURE
# ============================================================

def figure_phase13_bootstrap():

    header(
        "FIGURE — PHASE 13 BOOTSTRAP STABILITY"
    )

    combined = []

    for cell in CELL_TYPES:

        path = (
            PHASE13_BOOTSTRAP_DIR /
            f"{cell}_bootstrap_couplings.csv"
        )

        df = pd.read_csv(
            path
        )

        require_columns(
            df,
            [
                "cell_type",
                "pair",
                "J_original",
                "bootstrap_mean",
                "bootstrap_sd",
                "CI_2.5",
                "CI_50",
                "CI_97.5",
                "CI_excludes_zero",
            ],
            path,
        )

        if len(df) != 21:
            raise RuntimeError(
                f"{cell}: expected 21 bootstrap "
                f"coupling rows; got {len(df)}."
            )

        combined.append(
            df
        )

    all_bootstrap = pd.concat(
        combined,
        ignore_index=True,
    )

    all_bootstrap.to_csv(
        FINAL_TABLE_DIR /
        "Phase13_bootstrap_couplings_all_cells.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Publication figure: all 21 couplings for each cell.
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 20),
        sharex=True,
    )

    for ax, cell in zip(
        axes,
        CELL_TYPES,
    ):

        df = all_bootstrap[
            all_bootstrap["cell_type"] == cell
        ].copy()

        df = df.sort_values(
            "J_original"
        ).reset_index(
            drop=True
        )

        y = np.arange(
            len(df)
        )

        lower = (
            df["J_original"]
            -
            df["CI_2.5"]
        ).to_numpy()

        upper = (
            df["CI_97.5"]
            -
            df["J_original"]
        ).to_numpy()

        ax.errorbar(
            df["J_original"],
            y,
            xerr=[
                lower,
                upper,
            ],
            fmt="o",
        )

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
        )

        ax.set_yticks(
            y
        )

        ax.set_yticklabels(
            df["pair"],
            fontsize=7,
        )

        ax.set_title(
            cell
        )

        ax.set_xlabel(
            "Estimated coupling J"
        )

    fig.suptitle(
        "Phase 13: bootstrap stability of inferred couplings",
        y=0.995,
    )

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_4_Phase13_bootstrap_stability.png"
    )

    save_figure_checked(
        path,
        required_n=105,
        arrays={
            "J_original":
                all_bootstrap[
                    "J_original"
                ].to_numpy(),
            "CI_2.5":
                all_bootstrap[
                    "CI_2.5"
                ].to_numpy(),
            "CI_97.5":
                all_bootstrap[
                    "CI_97.5"
                ].to_numpy(),
        },
    )

    print(
        f"PASS: {path}"
    )


# ============================================================
# 14. PHASE 13 CROSS-VALIDATION FIGURE
# ============================================================

def figure_phase13_cv():

    header(
        "FIGURE — PHASE 13 CHROMOSOME CROSS-VALIDATION"
    )

    combined = []

    for cell in CELL_TYPES:

        path = (
            PHASE13_CV_DIR /
            f"{cell}_chromosome_cv.csv"
        )

        df = pd.read_csv(
            path
        )

        require_columns(
            df,
            [
                "cell_type",
                "fold",
                "n_train",
                "n_validation",
                "train_avg_pseudolikelihood",
                "validation_avg_pseudolikelihood",
                "mean_abs_J",
                "max_abs_J",
                "optimization_success",
            ],
            path,
        )

        if len(df) != 5:
            raise RuntimeError(
                f"{cell}: expected 5 CV folds; "
                f"got {len(df)}."
            )

        combined.append(
            df
        )

    cv = pd.concat(
        combined,
        ignore_index=True,
    )

    cv.to_csv(
        FINAL_TABLE_DIR /
        "Phase13_chromosome_CV_all_cells.csv",
        index=False,
    )

    data = [
        cv.loc[
            cv["cell_type"] == cell,
            "validation_avg_pseudolikelihood",
        ].to_numpy()
        for cell in CELL_TYPES
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.boxplot(
        data,
        tick_labels=CELL_TYPES,
    )

    plt.ylabel(
        "Held-out average pseudo-log-likelihood"
    )

    plt.xlabel(
        "Cell type"
    )

    plt.title(
        "Phase 13: chromosome-based cross-validation"
    )

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_5_Phase13_chromosome_CV.png"
    )

    save_figure_checked(
        path,
        required_n=25,
        arrays={
            "validation_pseudolikelihood":
                cv[
                    "validation_avg_pseudolikelihood"
                ].to_numpy()
        },
    )

    print(
        f"PASS: {path}"
    )


# ============================================================
# 15. PHASE 13 WINDOW-SIZE FIGURE
# ============================================================

def figure_phase13_window_size():

    header(
        "FIGURE — PHASE 13 WINDOW-SIZE SENSITIVITY"
    )

    combined = []

    for cell in CELL_TYPES:

        path = (
            PHASE13_WINDOW_DIR /
            f"{cell}_window_size_summary.csv"
        )

        df = pd.read_csv(
            path
        )

        require_columns(
            df,
            [
                "cell_type",
                "window_size_kb",
                "n_windows",
                "pseudo_likelihood",
                "mean_abs_J",
                "max_abs_J",
                "optimization_success",
            ],
            path,
        )

        if len(df) != 4:
            raise RuntimeError(
                f"{cell}: expected four window sizes; "
                f"got {len(df)}."
            )

        combined.append(
            df
        )

    windows = pd.concat(
        combined,
        ignore_index=True,
    )

    windows.to_csv(
        FINAL_TABLE_DIR /
        "Phase13_window_size_all_cells.csv",
        index=False,
    )

    plt.figure(
        figsize=(9, 6)
    )

    for cell in CELL_TYPES:

        sub = windows[
            windows["cell_type"] == cell
        ].sort_values(
            "window_size_kb"
        )

        plt.plot(
            sub["window_size_kb"],
            sub["mean_abs_J"],
            marker="o",
            label=cell,
        )

    plt.xlabel(
        "Window size (kb)"
    )

    plt.ylabel(
        "Mean |J|"
    )

    plt.title(
        "Phase 13: coupling sensitivity to genomic scale"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FINAL_FIGURE_DIR /
        "Figure_6_Phase13_window_size_sensitivity.png"
    )

    save_figure_checked(
        path,
        required_n=20,
        arrays={
            "mean_abs_J":
                windows[
                    "mean_abs_J"
                ].to_numpy()
        },
    )

    print(
        f"PASS: {path}"
    )


# ============================================================
# 16. PHASE 13 THRESHOLD TABLE
# ============================================================

def build_threshold_table():

    header(
        "PHASE 13 THRESHOLD-SENSITIVITY QC"
    )

    combined = []

    for cell in CELL_TYPES:

        path = (
            PHASE13_THRESHOLD_DIR /
            f"{cell}_coupling_threshold_sensitivity.csv"
        )

        df = pd.read_csv(
            path
        )

        require_columns(
            df,
            [
                "cell_type",
                "absolute_J_threshold",
                "n_selected",
                "fraction_selected",
                "mean_abs_J_selected",
            ],
            path,
        )

        combined.append(
            df
        )

    result = pd.concat(
        combined,
        ignore_index=True,
    )

    result.to_csv(
        FINAL_TABLE_DIR /
        "Phase13_threshold_sensitivity_all_cells.csv",
        index=False,
    )

    print(
        f"Rows: {len(result):,}"
    )

    return result


# ============================================================
# 17. FINAL SUMMARY TABLE
# ============================================================

def build_final_summary(
    metrics,
    controlled,
):

    rows = []

    for cell in CELL_TYPES:

        m = metrics[
            metrics["cell_type"] == cell
        ]

        c = controlled[
            controlled["cell_type"] == cell
        ]

        rows.append(
            {
                "cell_type":
                    cell,

                "n_logistic_models":
                    len(m),

                "mean_AUC":
                    m["auc"].mean(),

                "mean_average_precision":
                    m["average_precision"].mean(),

                "mean_log_loss":
                    m["log_loss"].mean(),

                "mean_Brier":
                    m["brier_score"].mean(),

                "n_controlled_pairs":
                    len(c),

                "mean_abs_controlled_correlation":
                    np.mean(
                        np.abs(
                            c[
                                "residual_correlation"
                            ]
                        )
                    ),

                "max_abs_controlled_correlation":
                    np.max(
                        np.abs(
                            c[
                                "residual_correlation"
                            ]
                        )
                    ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    path = (
        FINAL_TABLE_DIR /
        "Phase14_final_summary_by_cell_type.csv"
    )

    summary.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )

    return summary


# ============================================================
# 18. WRITE FINAL QC REPORT
# ============================================================

def write_final_report(
    audit,
    cpg_status,
    comparison,
):

    header(
        "WRITING FINAL PHASE 14 REPORT"
    )

    report_path = (
        FINAL_DIR /
        "FINAL_QC_REPORT.txt"
    )

    figure_files = sorted(
        FINAL_FIGURE_DIR.glob(
            "*.png"
        )
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 14 — FINAL QC REPORT\n"
        )

        f.write(
            "Epigenenomic Project\n"
        )

        f.write(
            "=" * 78 +
            "\n\n"
        )

        f.write(
            "STATUS\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        f.write(
            "Phase 12 outputs audited: PASS\n"
        )

        f.write(
            "Phase 13 outputs audited: PASS\n"
        )

        f.write(
            f"CpG-island feature status: "
            f"{cpg_status}\n"
        )

        f.write(
            "Figures generated from real Phase 12/13 "
            "output tables: PASS\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "CELL TYPES\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        for cell in CELL_TYPES:
            f.write(
                f"- {cell}\n"
            )

        f.write(
            "\nHISTONE MARKS\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        for mark in MARKS:
            f.write(
                f"- {mark}\n"
            )

        f.write(
            "\nFINAL FIGURES\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        for path in figure_files:

            f.write(
                f"- {path.name}\n"
            )

        f.write(
            f"\nNumber of final figures: "
            f"{len(figure_files)}\n"
        )

        f.write(
            "\nPHASE 12 / PHASE 13 INTERPRETATION\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        f.write(
            "Phase 12 controlled dependencies are "
            "residual correlations after logistic "
            "prediction from genomic-context features. "
            "They are not themselves Ising J parameters.\n"
        )

        f.write(
            "Phase 13 bootstrap results represent "
            "window-resampling stability.\n"
        )

        f.write(
            "Phase 13 chromosome cross-validation "
            "assesses held-out genomic chromosome robustness.\n"
        )

        f.write(
            "Phase 13 window-size analysis is a "
            "resolution sensitivity analysis based on "
            "aggregation of existing 10-kb windows.\n"
        )

        f.write(
            "\n"
        )

        if cpg_status == "WARNING_ALL_ZERO":

            f.write(
                "IMPORTANT QC WARNING\n"
            )

            f.write(
                "-" * 78 +
                "\n"
            )

            f.write(
                "The CpG-island-overlap feature was zero "
                "for all windows in the existing Phase 12 "
                "feature table. It should not be presented "
                "as an informative predictor in Preprint V1 "
                "unless independently repaired and rerun.\n"
            )

            f.write(
                "\n"
            )

        if comparison is None:

            f.write(
                "PHASE 9 LINK\n"
            )

            f.write(
                "-" * 78 +
                "\n"
            )

            f.write(
                "The automatic Phase 9 coupling-table "
                "connection was not available during this "
                "QC run. Full-J versus controlled comparison "
                "was therefore not generated.\n"
            )

        else:

            f.write(
                "PHASE 9 LINK\n"
            )

            f.write(
                "-" * 78 +
                "\n"
            )

            f.write(
                "Phase 9 full MaxEnt coupling values were "
                "successfully connected to Phase 12 controlled "
                "dependencies.\n"
            )

        f.write(
            "\n"
        )

        f.write(
            "PREPRINT V1 COMPUTATIONAL STATUS\n"
        )

        f.write(
            "-" * 78 +
            "\n"
        )

        f.write(
            "Phases 9–13 are computationally complete. "
            "Phase 14 performs final QC and publication "
            "packaging rather than introducing another "
            "modeling phase.\n"
        )

    print(
        f"Saved: {report_path}"
    )

    return report_path


# ============================================================
# 19. MAIN
# ============================================================

def main():

    header(
        "PHASE 14 — FINAL QC + PREPRINT V1 PACKAGE"
    )

    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nFinal output:\n{FINAL_DIR}"
    )

    # --------------------------------------------------------
    # STEP 1
    # Audit exact Phase 12/13 outputs.
    # --------------------------------------------------------

    audit = (
        audit_phase12_phase13_files()
    )

    # --------------------------------------------------------
    # STEP 2
    # Load exact Phase 12 tables.
    # --------------------------------------------------------

    (
        features,
        metrics,
        controlled,
    ) = load_phase12()

    # --------------------------------------------------------
    # STEP 3
    # CpG QC.
    # --------------------------------------------------------

    cpg_status = (
        cpg_quality_control(
            features
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # Phase 12 figures.
    # --------------------------------------------------------

    figure_phase12_logistic(
        metrics
    )

    matrices = (
        figure_phase12_controlled_heatmaps(
            controlled
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # Connect Phase 9 J.
    # --------------------------------------------------------

    phase9 = (
        load_phase9_couplings()
    )

    comparison = (
        build_full_vs_controlled(
            controlled,
            phase9,
        )
    )

    figure_full_vs_controlled(
        comparison
    )

    # --------------------------------------------------------
    # STEP 6
    # Phase 13 figures.
    # --------------------------------------------------------

    figure_phase13_bootstrap()

    figure_phase13_cv()

    figure_phase13_window_size()

    # --------------------------------------------------------
    # STEP 7
    # Phase 13 threshold table.
    # --------------------------------------------------------

    build_threshold_table()

    # --------------------------------------------------------
    # STEP 8
    # Final summary table.
    # --------------------------------------------------------

    build_final_summary(
        metrics,
        controlled,
    )

    # --------------------------------------------------------
    # STEP 9
    # Final report.
    # --------------------------------------------------------

    write_final_report(
        audit,
        cpg_status,
        comparison,
    )

    # --------------------------------------------------------
    # STEP 10
    # Final figure audit.
    # --------------------------------------------------------

    header(
        "PHASE 14 FIGURE AUDIT"
    )

    figure_rows = []

    for path in sorted(
        FINAL_FIGURE_DIR.glob(
            "*.png"
        )
    ):

        size = path.stat().st_size

        status = (
            "PASS"
            if size >= 5000
            else "FAIL"
        )

        print(
            f"{status:5s} "
            f"{path.name:55s} "
            f"{size:,} bytes"
        )

        figure_rows.append(
            {
                "figure":
                    path.name,
                "size_bytes":
                    size,
                "status":
                    status,
            }
        )

    figure_audit = pd.DataFrame(
        figure_rows
    )

    figure_audit.to_csv(
        FINAL_DIR /
        "PHASE14_FIGURE_AUDIT.csv",
        index=False,
    )

    if (
        len(figure_audit) == 0
    ):
        raise RuntimeError(
            "PHASE 14 FAILED: "
            "no final figures were generated."
        )

    if (
        not (
            figure_audit["status"]
            == "PASS"
        ).all()
    ):
        raise RuntimeError(
            "PHASE 14 FAILED: "
            "one or more final figures failed QC."
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    header(
        "PHASE 14 COMPLETE"
    )

    print(
        "\nFinal figures:"
    )

    for path in sorted(
        FINAL_FIGURE_DIR.glob(
            "*.png"
        )
    ):
        print(
            f"  {path}"
        )

    print(
        "\nFinal tables:"
    )

    for path in sorted(
        FINAL_TABLE_DIR.glob(
            "*.csv"
        )
    ):
        print(
            f"  {path}"
        )

    print(
        "\nFinal report:"
    )

    print(
        FINAL_DIR /
        "FINAL_QC_REPORT.txt"
    )

    print(
        "\nPHASE 14 FINAL QC COMPLETE."
    )

    print(
        "Computational analysis is now ready "
        "for Preprint V1 assembly."
    )


if __name__ == "__main__":
    main()