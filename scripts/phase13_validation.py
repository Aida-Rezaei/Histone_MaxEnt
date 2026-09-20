"""
phase13_validation.py

PHASE 13 — MODEL ROBUSTNESS AND VALIDATION

Goal
----
Demonstrate that the pairwise maximum-entropy model is robust to:

1. Bootstrap resampling of genomic windows
2. Chromosome-based cross-validation
3. Coupling-threshold sensitivity
4. Window-size sensitivity using coarser windows constructed
   from the existing 10-kb genomic windows

IMPORTANT
---------
The pseudo-likelihood objective and analytical gradient are
implemented consistently.

The pseudo-likelihood is averaged over:

    number of windows × number of marks

Therefore the analytical gradient uses the same normalization.

For each coupling J_ij, both conditional likelihood terms
are included:

    sigma_i | sigma_-i
    sigma_j | sigma_-j

This is essential because the model uses symmetric J.

Outputs
-------
results/
    validation/
        bootstrap/
        cross_validation/
        threshold_sensitivity/
        window_size_sensitivity/
        figures/
        PHASE13_SUMMARY.txt
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WINDOW_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "window_matrices"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation"
)

BOOTSTRAP_DIR = OUTPUT_DIR / "bootstrap"
CV_DIR = OUTPUT_DIR / "cross_validation"
THRESHOLD_DIR = OUTPUT_DIR / "threshold_sensitivity"
WINDOW_SIZE_DIR = OUTPUT_DIR / "window_size_sensitivity"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [
    OUTPUT_DIR,
    BOOTSTRAP_DIR,
    CV_DIR,
    THRESHOLD_DIR,
    WINDOW_SIZE_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
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

M = len(MARKS)

# Same regularization used in Phase 9
LAMBDA_J = 0.0001
LAMBDA_H = 0.0001

# Bootstrap
N_BOOTSTRAPS = 50

# Cross-validation
N_CV_FOLDS = 5

# Coupling thresholds
COUPLING_THRESHOLDS = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]

# Coarser windows constructed from 10-kb windows
WINDOW_SIZES_KB = [
    10,
    20,
    50,
    100,
]

# Optimizer
MAX_ITER = 2000
MAX_LINE_SEARCH = 50

FTOL = 1e-12
GTOL = 1e-8

RANDOM_SEED = 20260919

RNG = np.random.default_rng(RANDOM_SEED)


# ============================================================
# CONFIGURATION SPACE
# ============================================================

CONFIGS = np.array(
    list(itertools.product([0, 1], repeat=M)),
    dtype=float,
)

UPPER_I, UPPER_J = np.triu_indices(
    M,
    k=1,
)

N_COUPLINGS = len(UPPER_I)

PAIR_NAMES = [
    f"{MARKS[i]}__{MARKS[j]}"
    for i, j in zip(UPPER_I, UPPER_J)
]


# ============================================================
# UTILITY
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DATA LOADING
# ============================================================

def load_cell_matrix(cell_type):

    path = (
        WINDOW_DIR
        / f"{cell_type}_window_matrix.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n{path}"
        )

    print()
    print(f"Loading {cell_type}")
    print(f"File: {path}")

    df = pd.read_parquet(path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    missing = [
        mark
        for mark in MARKS
        if mark not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{cell_type}: missing mark columns: "
            f"{missing}"
        )

    coordinate_candidates = [
        ["chrom", "start", "end"],
        ["chromosome", "start", "end"],
    ]

    coord_cols = None

    for candidate in coordinate_candidates:

        if all(
            c in df.columns
            for c in candidate
        ):

            coord_cols = candidate
            break

    if coord_cols is None:

        raise ValueError(
            f"{cell_type}: could not identify "
            "chrom/start/end columns."
        )

    chrom_col, start_col, end_col = coord_cols

    df = df.copy()

    df["chrom"] = (
        df[chrom_col]
        .astype(str)
    )

    df["start"] = (
        df[start_col]
        .astype(int)
    )

    df["end"] = (
        df[end_col]
        .astype(int)
    )

    X = df[
        MARKS
    ].to_numpy(
        dtype=float
    )

    unique_values = np.unique(X)

    if not np.all(
        np.isin(
            unique_values,
            [0, 1]
        )
    ):

        raise ValueError(
            f"{cell_type}: mark matrix contains "
            f"values other than 0/1: "
            f"{unique_values[:20]}"
        )

    print(
        f"{cell_type}: "
        f"{len(df):,} valid windows"
    )

    return df, X


# ============================================================
# PARAMETER HANDLING
# ============================================================

def unpack_params(params):

    h = params[:M]

    J = np.zeros(
        (M, M),
        dtype=float,
    )

    J[
        UPPER_I,
        UPPER_J
    ] = params[M:]

    J[
        UPPER_J,
        UPPER_I
    ] = params[M:]

    return h, J


def extract_J(params):

    _, J = unpack_params(params)

    return J


def upper_J_vector(J):

    return J[
        UPPER_I,
        UPPER_J
    ]


# ============================================================
# CONDITIONAL PROBABILITIES
# ============================================================

def conditional_probability(
    params,
    X,
):

    h, J = unpack_params(params)

    fields = (
        h[None, :]
        +
        X @ J.T
    )

    return expit(fields)


# ============================================================
# PSEUDO-LIKELIHOOD OBJECTIVE
# ============================================================

def pseudo_likelihood_value(
    params,
    X,
    lambda_J=LAMBDA_J,
    lambda_h=LAMBDA_H,
    weights=None,
):
    """
    Average negative pseudo-log-likelihood.

    IMPORTANT:
    The likelihood is averaged over BOTH:

        windows × marks

    This matches the normalization used by Phase 9.
    """

    h, J = unpack_params(params)

    probabilities = conditional_probability(
        params,
        X,
    )

    eps = 1e-12

    pll = (
        X
        * np.log(
            np.clip(
                probabilities,
                eps,
                1 - eps,
            )
        )
        +
        (1 - X)
        * np.log(
            np.clip(
                1 - probabilities,
                eps,
                1 - eps,
            )
        )
    )

    if weights is None:

        loss = -pll.mean()

    else:

        weights = np.asarray(
            weights,
            dtype=float,
        )

        weighted_sum = np.sum(
            weights[:, None] * pll
        )

        normalization = (
            np.sum(weights)
            * M
        )

        loss = (
            -weighted_sum
            / normalization
        )

    loss += (
        lambda_J
        * np.sum(
            params[M:] ** 2
        )
    )

    loss += (
        lambda_h
        * np.sum(
            h ** 2
        )
    )

    return float(loss)


# ============================================================
# ANALYTICAL GRADIENT
# ============================================================

def pseudo_likelihood_gradient(
    params,
    X,
    lambda_J=LAMBDA_J,
    lambda_h=LAMBDA_H,
    weights=None,
):
    """
    Analytical gradient of the average negative
    pseudo-likelihood.

    CRITICAL CORRECTION
    -------------------
    The objective averages over n_windows * M.

    Therefore the likelihood gradient must use
    the same normalization.

    For J_ij, both conditional terms contribute:

        residual_i * X_j

    and

        residual_j * X_i

    because J_ij = J_ji.
    """

    h, J = unpack_params(params)

    probabilities = conditional_probability(
        params,
        X,
    )

    residual = (
        probabilities
        - X
    )

    n = X.shape[0]

    if weights is None:

        # Average over windows AND marks.
        normalization = (
            n * M
        )

        grad_h = (
            residual.sum(axis=0)
            / normalization
        )

        grad_J_matrix = (
            residual.T @ X
            +
            X.T @ residual
        ) / normalization

    else:

        weights = np.asarray(
            weights,
            dtype=float,
        )

        weighted_residual = (
            residual
            * weights[:, None]
        )

        normalization = (
            np.sum(weights)
            * M
        )

        grad_h = (
            weighted_residual.sum(axis=0)
            / normalization
        )

        grad_J_matrix = (
            weighted_residual.T @ X
            +
            X.T @ weighted_residual
        ) / normalization

    grad_h += (
        2
        * lambda_h
        * h
    )

    grad_J = (
        grad_J_matrix[
            UPPER_I,
            UPPER_J
        ]
        +
        2
        * lambda_J
        * params[M:]
    )

    return np.concatenate(
        [
            grad_h,
            grad_J,
        ]
    )


# ============================================================
# GRADIENT CHECK
# ============================================================

def check_gradient():

    print_header(
        "NUMERICAL GRADIENT CHECK"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    X_test = rng.integers(
        0,
        2,
        size=(300, M),
    ).astype(float)

    params = rng.normal(
        0,
        0.2,
        size=M + N_COUPLINGS,
    )

    analytical = (
        pseudo_likelihood_gradient(
            params,
            X_test,
        )
    )

    numerical = np.zeros_like(
        params
    )

    epsilon = 1e-6

    for k in range(
        len(params)
    ):

        p_plus = params.copy()
        p_minus = params.copy()

        p_plus[k] += epsilon
        p_minus[k] -= epsilon

        numerical[k] = (
            pseudo_likelihood_value(
                p_plus,
                X_test,
            )
            -
            pseudo_likelihood_value(
                p_minus,
                X_test,
            )
        ) / (
            2 * epsilon
        )

    max_error = np.max(
        np.abs(
            analytical
            - numerical
        )
    )

    mean_error = np.mean(
        np.abs(
            analytical
            - numerical
        )
    )

    print(
        f"Maximum absolute gradient error: "
        f"{max_error:.3e}"
    )

    print(
        f"Mean absolute gradient error: "
        f"{mean_error:.3e}"
    )

    if max_error > 1e-5:

        raise RuntimeError(
            "Gradient check FAILED. "
            "Do not continue to Phase 13."
        )

    print(
        "Gradient check PASSED."
    )


# ============================================================
# INITIAL PARAMETERS
# ============================================================

def independent_initialization(X):

    p = X.mean(
        axis=0
    )

    p = np.clip(
        p,
        1e-6,
        1 - 1e-6,
    )

    h = np.log(
        p
        /
        (1 - p)
    )

    J = np.zeros(
        N_COUPLINGS
    )

    return np.concatenate(
        [
            h,
            J,
        ]
    )


# ============================================================
# ROBUST MODEL FITTING
# ============================================================

def fit_maxent(
    X,
    initial_params=None,
    weights=None,
):
    """
    Fit the regularized pairwise maximum-entropy model.

    Uses L-BFGS-B with the corrected analytical gradient.

    If L-BFGS-B reports failure, a second optimization
    attempt is made using the same objective and gradient
    from the best available parameters.

    A failed optimization is never silently accepted.
    """

    n_params = (
        M
        +
        N_COUPLINGS
    )

    if initial_params is None:

        initial_params = (
            independent_initialization(
                X
            )
        )

    initial_params = np.asarray(
        initial_params,
        dtype=float,
    )

    result = minimize(
        pseudo_likelihood_value,
        initial_params,
        jac=pseudo_likelihood_gradient,
        args=(
            X,
            LAMBDA_J,
            LAMBDA_H,
            weights,
        ),
        method="L-BFGS-B",
        options={
            "maxiter": MAX_ITER,
            "maxls": MAX_LINE_SEARCH,
            "ftol": FTOL,
            "gtol": GTOL,
        },
    )

    if result.success:

        return result.x, result

    warnings.warn(
        "L-BFGS-B did not converge: "
        f"{result.message}. "
        "Attempting a second fit from the current parameters."
    )

    second_result = minimize(
        pseudo_likelihood_value,
        result.x,
        jac=pseudo_likelihood_gradient,
        args=(
            X,
            LAMBDA_J,
            LAMBDA_H,
            weights,
        ),
        method="L-BFGS-B",
        options={
            "maxiter": MAX_ITER * 2,
            "maxls": MAX_LINE_SEARCH * 2,
            "ftol": FTOL,
            "gtol": GTOL,
        },
    )

    if second_result.success:

        return (
            second_result.x,
            second_result,
        )

    raise RuntimeError(
        "Maximum-entropy optimization failed.\n"
        f"First optimizer message: {result.message}\n"
        f"Second optimizer message: "
        f"{second_result.message}"
    )


# ============================================================
# MODEL SCORING
# ============================================================

def average_pseudolikelihood(
    X,
    params,
):

    probabilities = (
        conditional_probability(
            params,
            X,
        )
    )

    eps = 1e-12

    pll = (
        X
        * np.log(
            np.clip(
                probabilities,
                eps,
                1 - eps,
            )
        )
        +
        (1 - X)
        * np.log(
            np.clip(
                1 - probabilities,
                eps,
                1 - eps,
            )
        )
    )

    return float(
        pll.mean()
    )


def mark_auc(
    X,
    params,
):

    try:

        from sklearn.metrics import (
            roc_auc_score
        )

    except ImportError:

        return np.full(
            M,
            np.nan,
        )

    probabilities = (
        conditional_probability(
            params,
            X,
        )
    )

    aucs = []

    for i in range(M):

        if len(
            np.unique(
                X[:, i]
            )
        ) < 2:

            aucs.append(
                np.nan
            )

        else:

            aucs.append(
                roc_auc_score(
                    X[:, i],
                    probabilities[:, i],
                )
            )

    return np.array(
        aucs
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_cell_type(
    cell_type,
    X,
    n_bootstraps=N_BOOTSTRAPS,
):

    print_header(
        f"BOOTSTRAP ROBUSTNESS — {cell_type}"
    )

    n = X.shape[0]

    print(
        "Fitting original model..."
    )

    params_original, result_original = (
        fit_maxent(X)
    )

    J_original = extract_J(
        params_original
    )

    print(
        f"Original optimization success: "
        f"{result_original.success}"
    )

    print(
        f"Original final objective: "
        f"{result_original.fun:.8f}"
    )

    bootstrap_J = np.zeros(
        (
            n_bootstraps,
            N_COUPLINGS,
        )
    )

    bootstrap_h = np.zeros(
        (
            n_bootstraps,
            M,
        )
    )

    bootstrap_pl = np.zeros(
        n_bootstraps
    )

    original_J_vector = (
        upper_J_vector(
            J_original
        )
    )

    current_initial = (
        params_original.copy()
    )

    for b in range(
        n_bootstraps
    ):

        if b % 5 == 0:

            print(
                f"Bootstrap "
                f"{b + 1}/{n_bootstraps}"
            )

        # ----------------------------------------------------
        # Bootstrap represented as multinomial weights.
        #
        # This is mathematically equivalent to resampling
        # n windows with replacement, but avoids allocating
        # another n x 7 matrix.
        # ----------------------------------------------------

        counts = RNG.multinomial(
            n,
            np.full(
                n,
                1.0 / n,
            ),
        )

        params_boot, result_boot = (
            fit_maxent(
                X,
                initial_params=current_initial,
                weights=counts,
            )
        )

        current_initial = (
            params_boot.copy()
        )

        J_boot = extract_J(
            params_boot
        )

        bootstrap_J[b] = (
            upper_J_vector(
                J_boot
            )
        )

        bootstrap_h[b] = (
            params_boot[:M]
        )

        bootstrap_pl[b] = (
            pseudo_likelihood_value(
                params_boot,
                X,
                LAMBDA_J,
                LAMBDA_H,
                counts,
            )
        )

        if not result_boot.success:

            raise RuntimeError(
                f"{cell_type}: bootstrap "
                f"{b + 1} failed."
            )

    # --------------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------------

    rows = []

    for pair_idx, pair_name in enumerate(
        PAIR_NAMES
    ):

        values = (
            bootstrap_J[
                :,
                pair_idx
            ]
        )

        q025, q50, q975 = (
            np.percentile(
                values,
                [2.5, 50, 97.5],
            )
        )

        original = (
            original_J_vector[
                pair_idx
            ]
        )

        rows.append(
            {
                "cell_type": cell_type,
                "pair": pair_name,
                "J_original": original,
                "bootstrap_mean": np.mean(
                    values
                ),
                "bootstrap_sd": np.std(
                    values,
                    ddof=1,
                ),
                "CI_2.5": q025,
                "CI_50": q50,
                "CI_97.5": q975,
                "CI_excludes_zero": (
                    q025 > 0
                    or q975 < 0
                ),
            }
        )

    bootstrap_df = pd.DataFrame(
        rows
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_DIR
        / f"{cell_type}_bootstrap_couplings.csv",
        index=False,
    )

    # Raw bootstrap J
    bootstrap_raw = pd.DataFrame(
        bootstrap_J,
        columns=PAIR_NAMES,
    )

    bootstrap_raw.to_csv(
        BOOTSTRAP_DIR
        / f"{cell_type}_bootstrap_raw.csv",
        index=False,
    )

    # Bootstrap parameter distances
    bootstrap_distances = np.sqrt(
        np.sum(
            (
                bootstrap_J
                -
                original_J_vector[
                    None,
                    :
                ]
            ) ** 2,
            axis=1,
        )
    )

    stability = pd.DataFrame(
        [
            {
                "cell_type": cell_type,
                "n_bootstraps": n_bootstraps,
                "median_J_distance": np.median(
                    bootstrap_distances
                ),
                "mean_J_distance": np.mean(
                    bootstrap_distances
                ),
                "95pct_J_distance": np.percentile(
                    bootstrap_distances,
                    95,
                ),
                "original_pseudolikelihood": (
                    average_pseudolikelihood(
                        X,
                        params_original,
                    )
                ),
                "bootstrap_mean_pseudolikelihood": (
                    np.mean(
                        bootstrap_pl
                    )
                ),
            }
        ]
    )

    stability.to_csv(
        BOOTSTRAP_DIR
        / f"{cell_type}_bootstrap_stability.csv",
        index=False,
    )

    return (
        params_original,
        bootstrap_df,
        bootstrap_J,
    )


# ============================================================
# CHROMOSOME FOLDS
# ============================================================

def chromosome_sort_key(
    chrom
):

    chrom = (
        str(chrom)
        .replace(
            "chr",
            "",
        )
    )

    if chrom.isdigit():

        return int(
            chrom
        )

    if chrom == "X":

        return 23

    if chrom == "Y":

        return 24

    return 100


def create_chromosome_folds(
    chromosomes
):

    counts = (
        pd.Series(
            chromosomes
        )
        .value_counts()
        .sort_values(
            ascending=False
        )
    )

    fold_load = np.zeros(
        N_CV_FOLDS,
        dtype=int,
    )

    fold_chromosomes = [
        []
        for _ in range(
            N_CV_FOLDS
        )
    ]

    for chrom, count in counts.items():

        fold_idx = np.argmin(
            fold_load
        )

        fold_chromosomes[
            fold_idx
        ].append(
            chrom
        )

        fold_load[
            fold_idx
        ] += count

    return fold_chromosomes


# ============================================================
# CROSS-VALIDATION
# ============================================================

def cross_validate_cell_type(
    cell_type,
    df,
    X,
):

    print_header(
        f"CHROMOSOME CROSS-VALIDATION — {cell_type}"
    )

    chromosomes = (
        df["chrom"].to_numpy()
    )

    folds = (
        create_chromosome_folds(
            chromosomes
        )
    )

    rows = []

    for fold_id, validation_chroms in enumerate(
        folds,
        start=1,
    ):

        validation_mask = np.isin(
            chromosomes,
            validation_chroms,
        )

        train_mask = (
            ~validation_mask
        )

        X_train = X[
            train_mask
        ]

        X_val = X[
            validation_mask
        ]

        print()
        print(
            f"Fold {fold_id}/"
            f"{N_CV_FOLDS}"
        )

        print(
            "Validation chromosomes:",
            ", ".join(
                sorted(
                    validation_chroms,
                    key=chromosome_sort_key,
                )
            ),
        )

        print(
            f"Training windows: "
            f"{len(X_train):,}"
        )

        print(
            f"Validation windows: "
            f"{len(X_val):,}"
        )

        params, result = fit_maxent(
            X_train
        )

        train_pl = (
            average_pseudolikelihood(
                X_train,
                params,
            )
        )

        validation_pl = (
            average_pseudolikelihood(
                X_val,
                params,
            )
        )

        J = extract_J(
            params
        )

        rows.append(
            {
                "cell_type": cell_type,
                "fold": fold_id,
                "validation_chromosomes":
                    ";".join(
                        validation_chroms
                    ),
                "n_train": len(
                    X_train
                ),
                "n_validation": len(
                    X_val
                ),
                "train_avg_pseudolikelihood":
                    train_pl,
                "validation_avg_pseudolikelihood":
                    validation_pl,
                "mean_abs_J":
                    np.mean(
                        np.abs(
                            upper_J_vector(
                                J
                            )
                        )
                    ),
                "max_abs_J":
                    np.max(
                        np.abs(
                            upper_J_vector(
                                J
                            )
                        )
                    ),
                "optimization_success":
                    result.success,
            }
        )

    cv_df = pd.DataFrame(
        rows
    )

    cv_df.to_csv(
        CV_DIR
        / f"{cell_type}_chromosome_cv.csv",
        index=False,
    )

    return cv_df


# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

def threshold_sensitivity(
    cell_type,
    params,
):

    print_header(
        f"COUPLING THRESHOLD SENSITIVITY — {cell_type}"
    )

    J = extract_J(
        params
    )

    values = (
        upper_J_vector(
            J
        )
    )

    rows = []

    for threshold in (
        COUPLING_THRESHOLDS
    ):

        selected = (
            np.abs(values)
            >= threshold
        )

        rows.append(
            {
                "cell_type": cell_type,
                "absolute_J_threshold":
                    threshold,
                "n_selected":
                    int(
                        selected.sum()
                    ),
                "fraction_selected":
                    selected.mean(),
                "mean_abs_J_selected":
                    (
                        np.mean(
                            np.abs(
                                values[
                                    selected
                                ]
                            )
                        )
                        if selected.any()
                        else np.nan
                    ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        THRESHOLD_DIR
        / f"{cell_type}_coupling_threshold_sensitivity.csv",
        index=False,
    )

    pair_rows = []

    for pair, value in zip(
        PAIR_NAMES,
        values,
    ):

        row = {
            "cell_type":
                cell_type,
            "pair":
                pair,
            "J":
                value,
        }

        for threshold in (
            COUPLING_THRESHOLDS
        ):

            row[
                f"selected_absJ_ge_{threshold}"
            ] = (
                abs(value)
                >= threshold
            )

        pair_rows.append(
            row
        )

    pd.DataFrame(
        pair_rows
    ).to_csv(
        THRESHOLD_DIR
        / f"{cell_type}_coupling_threshold_pairs.csv",
        index=False,
    )

    return result_df


# ============================================================
# WINDOW SIZE AGGREGATION
# ============================================================

def aggregate_windows(
    df,
    X,
    target_window_kb,
):

    base_size = 10

    if (
        target_window_kb
        % base_size
        != 0
    ):

        raise ValueError(
            "Target window must be "
            "a multiple of 10 kb."
        )

    factor = (
        target_window_kb
        // base_size
    )

    temp = df[
        [
            "chrom",
            "start",
            "end",
        ]
    ].copy()

    temp[
        "row_index"
    ] = np.arange(
        len(temp)
    )

    window_bp = (
        target_window_kb
        * 1000
    )

    temp[
        "group_start"
    ] = (
        temp["start"]
        // window_bp
    ) * window_bp

    states = pd.DataFrame(
        X,
        columns=MARKS,
    )

    temp = pd.concat(
        [
            temp,
            states,
        ],
        axis=1,
    )

    grouped = []

    for (
        chrom,
        group_start,
    ), group in temp.groupby(
        [
            "chrom",
            "group_start",
        ],
        sort=False,
    ):

        if len(group) != factor:

            continue

        starts = np.sort(
            group[
                "start"
            ].to_numpy()
        )

        expected = np.arange(
            starts[0],
            starts[0]
            + factor * 10000,
            10000,
        )

        if not np.array_equal(
            starts,
            expected,
        ):

            continue

        row = {
            "chrom": chrom,
            "start": int(
                group_start
            ),
            "end": int(
                group_start
                + window_bp
            ),
        }

        for mark in MARKS:

            fraction_present = (
                group[
                    mark
                ].mean()
            )

            row[mark] = int(
                fraction_present
                >= 0.5
            )

        grouped.append(
            row
        )

    if not grouped:

        raise RuntimeError(
            f"No complete "
            f"{target_window_kb}-kb "
            "windows could be constructed."
        )

    result_df = pd.DataFrame(
        grouped
    )

    result_X = (
        result_df[
            MARKS
        ].to_numpy(
            dtype=float
        )
    )

    return (
        result_df,
        result_X,
    )


# ============================================================
# WINDOW SIZE SENSITIVITY
# ============================================================

def window_size_sensitivity(
    cell_type,
    df,
    X,
):

    print_header(
        f"WINDOW-SIZE SENSITIVITY — {cell_type}"
    )

    rows = []

    for window_kb in (
        WINDOW_SIZES_KB
    ):

        print()
        print(
            f"Window size: "
            f"{window_kb} kb"
        )

        if window_kb == 10:

            current_df = df.copy()
            current_X = X.copy()

        else:

            (
                current_df,
                current_X,
            ) = aggregate_windows(
                df,
                X,
                window_kb,
            )

        print(
            f"Windows: "
            f"{len(current_X):,}"
        )

        params, result = fit_maxent(
            current_X
        )

        J = extract_J(
            params
        )

        rows.append(
            {
                "cell_type": cell_type,
                "window_size_kb":
                    window_kb,
                "n_windows":
                    len(current_X),
                "pseudo_likelihood":
                    average_pseudolikelihood(
                        current_X,
                        params,
                    ),
                "mean_abs_J":
                    np.mean(
                        np.abs(
                            upper_J_vector(
                                J
                            )
                        )
                    ),
                "max_abs_J":
                    np.max(
                        np.abs(
                            upper_J_vector(
                                J
                            )
                        )
                    ),
                "optimization_success":
                    result.success,
            }
        )

        J_df = pd.DataFrame(
            J,
            index=MARKS,
            columns=MARKS,
        )

        J_df.to_csv(
            WINDOW_SIZE_DIR
            / (
                f"{cell_type}_"
                f"{window_kb}kb_J.csv"
            )
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        WINDOW_SIZE_DIR
        / f"{cell_type}_window_size_summary.csv",
        index=False,
    )

    return result_df


# ============================================================
# WINDOW SIZE CORRELATIONS
# ============================================================

def calculate_window_correlations(
    cell_type,
):

    records = []

    vectors = {}

    for window_kb in (
        WINDOW_SIZES_KB
    ):

        path = (
            WINDOW_SIZE_DIR
            / (
                f"{cell_type}_"
                f"{window_kb}kb_J.csv"
            )
        )

        if not path.exists():

            continue

        J_df = pd.read_csv(
            path,
            index_col=0,
        )

        J = J_df.to_numpy(
            dtype=float
        )

        vectors[
            window_kb
        ] = upper_J_vector(
            J
        )

    sizes = list(
        vectors.keys()
    )

    for i, size1 in enumerate(
        sizes
    ):

        for size2 in sizes[
            i + 1:
        ]:

            v1 = vectors[
                size1
            ]

            v2 = vectors[
                size2
            ]

            if (
                np.std(v1) == 0
                or
                np.std(v2) == 0
            ):

                corr = np.nan

            else:

                corr = np.corrcoef(
                    v1,
                    v2,
                )[0, 1]

            distance = np.linalg.norm(
                v1 - v2
            )

            records.append(
                {
                    "cell_type":
                        cell_type,
                    "window_size_1_kb":
                        size1,
                    "window_size_2_kb":
                        size2,
                    "J_vector_correlation":
                        corr,
                    "J_vector_euclidean_distance":
                        distance,
                }
            )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        WINDOW_SIZE_DIR
        / f"{cell_type}_window_size_correlations.csv",
        index=False,
    )

    return result


# ============================================================
# SEARCH FOR 200-BP DATA
# ============================================================

def find_200bp_binary_candidates():

    candidate_roots = [
        PROJECT_ROOT
        / "data"
        / "processed",

        PROJECT_ROOT
        / "results",
    ]

    patterns = [
        "*200bp*.parquet",
        "*200bp*.csv",
        "*200_bp*.parquet",
        "*200_bp*.csv",
        "*bin_matrix*.parquet",
        "*bin_matrix*.csv",
        "*binarized*.parquet",
        "*binarized*.csv",
    ]

    found = []

    for root in candidate_roots:

        if not root.exists():

            continue

        for pattern in patterns:

            found.extend(
                root.rglob(
                    pattern
                )
            )

    unique = sorted(
        set(
            p.resolve()
            for p in found
        )
    )

    return unique


def optional_200bp_threshold_analysis():

    print_header(
        "OPTIONAL 200-BP BINARIZATION THRESHOLD ANALYSIS"
    )

    candidates = (
        find_200bp_binary_candidates()
    )

    if not candidates:

        print(
            "No compatible 200-bp "
            "binary matrices found."
        )

        print(
            "Skipping original "
            "binarization-threshold sensitivity."
        )

        report = pd.DataFrame(
            [
                {
                    "status": "SKIPPED",
                    "reason":
                        "No compatible "
                        "200-bp binary "
                        "matrix files "
                        "were found.",
                }
            ]
        )

        report.to_csv(
            THRESHOLD_DIR
            / "200bp_threshold_analysis_status.csv",
            index=False,
        )

        return None

    print(
        "Potential 200-bp files found:"
    )

    for path in candidates:

        print(
            f"  {path}"
        )

    print()
    print(
        "Detected files require schema "
        "verification before use."
    )

    report = pd.DataFrame(
        {
            "candidate_file":
                [
                    str(p)
                    for p in candidates
                ],
            "status":
                "DETECTED_REQUIRES_SCHEMA_CHECK",
        }
    )

    report.to_csv(
        THRESHOLD_DIR
        / "200bp_threshold_analysis_status.csv",
        index=False,
    )

    return candidates


# ============================================================
# FIGURES
# ============================================================

def make_bootstrap_figure(
    bootstrap_results
):

    for cell_type, df in (
        bootstrap_results.items()
    ):

        plot_df = df.sort_values(
            "J_original"
        )

        y = np.arange(
            len(plot_df)
        )

        plt.figure(
            figsize=(10, 8)
        )

        lower_error = (
            plot_df["J_original"]
            -
            plot_df["CI_2.5"]
        )

        upper_error = (
            plot_df["CI_97.5"]
            -
            plot_df["J_original"]
        )

        plt.errorbar(
            plot_df["J_original"],
            y,
            xerr=[
                lower_error,
                upper_error,
            ],
            fmt="o",
        )

        plt.axvline(
            0,
            linestyle="--",
        )

        plt.yticks(
            y,
            plot_df["pair"],
        )

        plt.xlabel(
            "Estimated coupling J"
        )

        plt.ylabel(
            "Histone-mark pair"
        )

        plt.title(
            f"{cell_type}: "
            "bootstrap coupling stability"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR
            / (
                f"{cell_type}_"
                "bootstrap_couplings.png"
            ),
            dpi=200,
        )

        plt.close()


def make_cv_figure(
    cv_results
):

    records = []

    for cell_type, df in (
        cv_results.items()
    ):

        for _, row in df.iterrows():

            records.append(
                {
                    "cell_type":
                        cell_type,
                    "fold":
                        row["fold"],
                    "validation_pseudolikelihood":
                        row[
                            "validation_avg_pseudolikelihood"
                        ],
                }
            )

    df = pd.DataFrame(
        records
    )

    plt.figure(
        figsize=(10, 6)
    )

    positions = np.arange(
        len(CELL_TYPES)
    )

    data = [
        df.loc[
            df["cell_type"] == cell,
            "validation_pseudolikelihood",
        ].to_numpy()
        for cell in CELL_TYPES
    ]

    plt.boxplot(
        data,
        positions=positions,
    )

    plt.xticks(
        positions,
        CELL_TYPES,
    )

    plt.ylabel(
        "Held-out average "
        "pseudo-log-likelihood"
    )

    plt.title(
        "Chromosome-based "
        "cross-validation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "chromosome_cross_validation.png",
        dpi=200,
    )

    plt.close()


def make_window_size_figure(
    window_results
):

    plt.figure(
        figsize=(9, 6)
    )

    for cell_type, df in (
        window_results.items()
    ):

        plt.plot(
            df["window_size_kb"],
            df["mean_abs_J"],
            marker="o",
            label=cell_type,
        )

    plt.xlabel(
        "Window size (kb)"
    )

    plt.ylabel(
        "Mean |J|"
    )

    plt.title(
        "Maximum-entropy coupling "
        "sensitivity to genomic scale"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "window_size_sensitivity.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    bootstrap_results,
    cv_results,
    threshold_results,
    window_results,
):

    summary_path = (
        OUTPUT_DIR
        / "PHASE13_SUMMARY.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 13 — MODEL VALIDATION SUMMARY\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "PROJECT\n"
        )

        f.write(
            "Histone Maximum Entropy Project\n\n"
        )

        f.write(
            "MODEL\n"
        )

        f.write(
            "Pairwise maximum entropy / "
            "Ising-like model\n"
        )

        f.write(
            f"Marks: {M}\n"
        )

        f.write(
            f"Pairwise couplings: "
            f"{N_COUPLINGS}\n"
        )

        f.write(
            f"Lambda_J: {LAMBDA_J}\n"
        )

        f.write(
            f"Lambda_h: {LAMBDA_H}\n\n"
        )

        f.write(
            "1. BOOTSTRAP ROBUSTNESS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell_type, df in (
            bootstrap_results.items()
        ):

            stable = int(
                df[
                    "CI_excludes_zero"
                ].sum()
            )

            total = len(df)

            f.write(
                f"{cell_type}: "
                f"{stable}/{total} "
                "couplings had bootstrap "
                "95% intervals excluding zero.\n"
            )

        f.write("\n")

        f.write(
            "2. CHROMOSOME CROSS-VALIDATION\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell_type, df in (
            cv_results.items()
        ):

            mean_pl = df[
                "validation_avg_pseudolikelihood"
            ].mean()

            sd_pl = df[
                "validation_avg_pseudolikelihood"
            ].std()

            f.write(
                f"{cell_type}: "
                f"mean held-out "
                f"pseudo-likelihood = "
                f"{mean_pl:.6f} "
                f"(SD {sd_pl:.6f})\n"
            )

        f.write("\n")

        f.write(
            "3. COUPLING THRESHOLD SENSITIVITY\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell_type, df in (
            threshold_results.items()
        ):

            f.write(
                f"\n{cell_type}\n"
            )

            for _, row in df.iterrows():

                f.write(
                    f"  |J| >= "
                    f"{row['absolute_J_threshold']:.2f}: "
                    f"{int(row['n_selected'])} "
                    "couplings\n"
                )

        f.write("\n")

        f.write(
            "4. WINDOW-SIZE SENSITIVITY\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for cell_type, df in (
            window_results.items()
        ):

            f.write(
                f"\n{cell_type}\n"
            )

            for _, row in df.iterrows():

                f.write(
                    f"  "
                    f"{int(row['window_size_kb'])} kb: "
                    f"{int(row['n_windows']):,} "
                    f"windows, mean |J| = "
                    f"{row['mean_abs_J']:.6f}\n"
                )

        f.write("\n")

        f.write(
            "5. INTERPRETATION GUIDANCE\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            "Bootstrap intervals quantify parameter "
            "stability under genomic-window resampling.\n\n"
        )

        f.write(
            "Chromosome cross-validation tests whether "
            "conditional predictions generalize to "
            "held-out chromosomes.\n\n"
        )

        f.write(
            "Coupling-threshold sensitivity shows how "
            "the set of large-magnitude couplings "
            "changes as the descriptive |J| threshold "
            "changes. These thresholds are not "
            "statistical significance thresholds.\n\n"
        )

        f.write(
            "Window-size sensitivity tests whether "
            "inferred coupling structure depends "
            "strongly on genomic aggregation scale.\n\n"
        )

        f.write(
            "The 20-kb, 50-kb and 100-kb analyses are "
            "coarser representations constructed from "
            "the existing 10-kb binary windows; they "
            "are not independent experimental "
            "measurements.\n\n"
        )

        f.write(
            "IMPORTANT LIMITATION:\n"
        )

        f.write(
            "The Phase 12 genomic-feature table reported "
            "zero CpG-island overlap for every window. "
            "This should be checked as a feature-extraction "
            "issue before treating CpG-island control as "
            "informative in the preprint.\n"
        )

    print()
    print(
        f"Saved summary:\n"
        f"{summary_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "PHASE 13 — MODEL ROBUSTNESS AND VALIDATION"
    )

    print(
        f"Project root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nWindow matrix directory:\n"
        f"{WINDOW_DIR}"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "Seven histone marks:"
    )

    for mark in MARKS:

        print(
            f"  {mark}"
        )

    print()
    print(
        f"Bootstrap replicates: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Cross-validation folds: "
        f"{N_CV_FOLDS}"
    )

    print(
        f"Lambda_J: {LAMBDA_J}"
    )

    print(
        f"Lambda_h: {LAMBDA_H}"
    )

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    check_gradient()

    # --------------------------------------------------------
    # Load matrices
    # --------------------------------------------------------

    print_header(
        "STEP 1 — LOADING CELL-TYPE MATRICES"
    )

    data = {}

    reference_df = None

    for cell_type in CELL_TYPES:

        df, X = load_cell_matrix(
            cell_type
        )

        data[cell_type] = {
            "df": df,
            "X": X,
        }

        if reference_df is None:

            reference_df = df

    # --------------------------------------------------------
    # Verify shared windows
    # --------------------------------------------------------

    print_header(
        "STEP 2 — VERIFYING SHARED GENOMIC WINDOWS"
    )

    reference_coordinates = (
        reference_df[
            [
                "chrom",
                "start",
                "end",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    for cell_type in CELL_TYPES[1:]:

        df = data[
            cell_type
        ]["df"]

        coordinates = (
            df[
                [
                    "chrom",
                    "start",
                    "end",
                ]
            ]
            .reset_index(
                drop=True
            )
        )

        if not coordinates.equals(
            reference_coordinates
        ):

            raise ValueError(
                f"{cell_type}: genomic windows "
                "do not match H1 reference."
            )

        print(
            f"{cell_type}: "
            "windows match H1"
        )

    print()

    print(
        f"All cell types share "
        f"{len(reference_coordinates):,} windows."
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    print_header(
        "STEP 3 — BOOTSTRAP GENOMIC-WINDOW ROBUSTNESS"
    )

    bootstrap_results = {}

    original_params = {}

    for cell_type in CELL_TYPES:

        X = data[
            cell_type
        ]["X"]

        (
            params,
            bootstrap_df,
            _,
        ) = bootstrap_cell_type(
            cell_type,
            X,
            N_BOOTSTRAPS,
        )

        bootstrap_results[
            cell_type
        ] = bootstrap_df

        original_params[
            cell_type
        ] = params

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    print_header(
        "STEP 4 — CHROMOSOME-BASED CROSS-VALIDATION"
    )

    cv_results = {}

    for cell_type in CELL_TYPES:

        df = data[
            cell_type
        ]["df"]

        X = data[
            cell_type
        ]["X"]

        cv_results[
            cell_type
        ] = cross_validate_cell_type(
            cell_type,
            df,
            X,
        )

    # --------------------------------------------------------
    # Threshold sensitivity
    # --------------------------------------------------------

    print_header(
        "STEP 5 — COUPLING THRESHOLD SENSITIVITY"
    )

    threshold_results = {}

    for cell_type in CELL_TYPES:

        threshold_results[
            cell_type
        ] = threshold_sensitivity(
            cell_type,
            original_params[
                cell_type
            ],
        )

    # --------------------------------------------------------
    # Optional 200-bp threshold
    # --------------------------------------------------------

    optional_200bp_threshold_analysis()

    # --------------------------------------------------------
    # Window size
    # --------------------------------------------------------

    print_header(
        "STEP 6 — WINDOW-SIZE SENSITIVITY"
    )

    window_results = {}

    for cell_type in CELL_TYPES:

        df = data[
            cell_type
        ]["df"]

        X = data[
            cell_type
        ]["X"]

        window_results[
            cell_type
        ] = window_size_sensitivity(
            cell_type,
            df,
            X,
        )

        calculate_window_correlations(
            cell_type
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print_header(
        "STEP 7 — CREATING VALIDATION FIGURES"
    )

    make_bootstrap_figure(
        bootstrap_results
    )

    make_cv_figure(
        cv_results
    )

    make_window_size_figure(
        window_results
    )

    print(
        f"Saved figures to:\n"
        f"{FIGURE_DIR}"
    )

    # --------------------------------------------------------
    # Combined tables
    # --------------------------------------------------------

    print_header(
        "STEP 8 — COMBINING RESULTS"
    )

    all_bootstrap = pd.concat(
        bootstrap_results.values(),
        ignore_index=True,
    )

    all_bootstrap.to_csv(
        BOOTSTRAP_DIR
        / "ALL_CELL_TYPES_BOOTSTRAP.csv",
        index=False,
    )

    all_cv = pd.concat(
        cv_results.values(),
        ignore_index=True,
    )

    all_cv.to_csv(
        CV_DIR
        / "ALL_CELL_TYPES_CV.csv",
        index=False,
    )

    all_threshold = pd.concat(
        threshold_results.values(),
        ignore_index=True,
    )

    all_threshold.to_csv(
        THRESHOLD_DIR
        / "ALL_CELL_TYPES_THRESHOLD_SENSITIVITY.csv",
        index=False,
    )

    all_windows = pd.concat(
        window_results.values(),
        ignore_index=True,
    )

    all_windows.to_csv(
        WINDOW_SIZE_DIR
        / "ALL_CELL_TYPES_WINDOW_SIZE_SENSITIVITY.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "STEP 9 — WRITING PHASE 13 SUMMARY"
    )

    write_summary(
        bootstrap_results,
        cv_results,
        threshold_results,
        window_results,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PHASE 13 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Main output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "Generated:"
    )

    print(
        "  bootstrap/"
    )

    print(
        "  cross_validation/"
    )

    print(
        "  threshold_sensitivity/"
    )

    print(
        "  window_size_sensitivity/"
    )

    print(
        "  figures/"
    )

    print(
        "  PHASE13_SUMMARY.txt"
    )

    print()
    print(
        "PHASE 13 MODEL VALIDATION COMPLETE."
    )


if __name__ == "__main__":

    main()