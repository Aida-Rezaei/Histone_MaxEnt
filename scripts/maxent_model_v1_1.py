# 9
# Maximum Entropy Model
# All five cell types

# ============================================================
# PACKAGES
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import logsumexp


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "window_matrices"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "maxent"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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

marks_len = len(MARKS)


# ============================================================
# MODEL SETTINGS
# ============================================================

LAMBDA_VALUES = [
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    0.1,
    1,
    10
]

TRAIN_FRAC = 0.8

RANDOM_SEED = 42

VALIDATION_CHROMOSOME = "chr20"


# ============================================================
# PAIRWISE COUPLINGS
# ============================================================

pairs = [
    (i, j)
    for i in range(marks_len)
    for j in range(i + 1, marks_len)
]

coupling_len = len(pairs)

parameters_len = marks_len + coupling_len


assert marks_len == 7
assert coupling_len == 21
assert parameters_len == 28


# ============================================================
# ALL 2^7 = 128 POSSIBLE HISTONE STATES
# ============================================================

states = np.array(
    np.meshgrid(*[[0, 1]] * marks_len)
).T.reshape(
    -1,
    marks_len
).astype(np.float64)


# ============================================================
# CONVERT THETA INTO h AND J
# ============================================================

def parameters(theta):

    h = theta[:marks_len]

    J = np.zeros(
        (marks_len, marks_len)
    )

    coupling_values = theta[marks_len:]

    for value, (a, b) in zip(
        coupling_values,
        pairs
    ):

        J[a, b] = value
        J[b, a] = value

    return h, J


# ============================================================
# ENERGY FUNCTION
# E(sigma) = -sum_i h_i sigma_i
#            -sum_{i<j} J_ij sigma_i sigma_j
# ============================================================

def energy(sigma, h, J):

    local_energy = -np.dot(
        h,
        sigma
    )

    pairwise_energy = -np.sum(
        np.triu(J, k=1)
        * np.outer(sigma, sigma)
    )

    return (
        local_energy
        + pairwise_energy
    )


# ============================================================
# MAXIMUM ENTROPY / BOLTZMANN PROBABILITIES
# ============================================================

def state_probabilities(h, J):

    energies = np.array([
        energy(
            sigma,
            h,
            J
        )
        for sigma in states
    ])

    log_Z = logsumexp(
        -energies
    )

    probabilities = np.exp(
        -energies - log_Z
    )

    return probabilities


# ============================================================
# L2 REGULARIZATION
# ============================================================

def regularization(
    theta,
    lambda_reg
):

    h, J = parameters(theta)

    h_penalty = np.sum(
        h ** 2
    )

    J_penalty = sum(
        J[i, j] ** 2
        for i, j in pairs
    )

    return (
        lambda_reg / 2
    ) * (
        h_penalty
        + J_penalty
    )


# ============================================================
# PSEUDO-LIKELIHOOD
# ============================================================

def pseudo_log_likelihood(
    theta,
    X
):

    h, J = parameters(
        theta
    )

    fields = (
        h
        + X @ J.T
    )

    # Probability of sigma_i = 1
    log_probability_one = (
        -np.logaddexp(
            0,
            -fields
        )
    )

    # Probability of sigma_i = 0
    log_probability_zero = (
        -np.logaddexp(
            0,
            fields
        )
    )

    # Select the probability corresponding
    # to the observed binary state
    log_conditional = (
        X * log_probability_one
        + (1 - X) * log_probability_zero
    )

    return np.sum(
        log_conditional
    )


# ============================================================
# OBJECTIVE FOR MINIMIZATION
# ============================================================

def PL_regularization_diff(
    theta,
    X,
    lambda_reg
):

    log_PL = pseudo_log_likelihood(
        theta,
        X
    )

    average_log_PL = (
        log_PL
        / (
            X.shape[0]
            * marks_len
        )
    )

    penalty = regularization(
        theta,
        lambda_reg
    )

    return (
        penalty
        - average_log_PL
    )


# ============================================================
# KL DIVERGENCE
# ============================================================

def kl_divergence(
    p_observed,
    p_model
):

    mask = p_observed > 0

    return np.sum(
        p_observed[mask]
        * np.log(
            p_observed[mask]
            / p_model[mask]
        )
    )


# ============================================================
# SYNTHETIC VALIDATION
#
# This tests whether the inference procedure can recover
# known parameters from data generated from a known model.
#
# It only needs to be run once because the algorithm is
# identical for all five cell types.
# ============================================================

print("\n")
print("=" * 70)
print("SYNTHETIC PARAMETER RECOVERY")
print("=" * 70)


# ------------------------------------------------------------
# Known synthetic local fields
# ------------------------------------------------------------

h_true = np.array([
    -1.0,   # H3K27ac
     0.5,   # H3K27me3
     0.0,   # H3K36me3
    -0.25,   # H3K4me1
     1.0,   # H3K4me3
     0.25,   # H3K9ac
    -0.5    # H3K9me3
], dtype=np.float64)


# ------------------------------------------------------------
# Known synthetic coupling matrix
# ------------------------------------------------------------

J_true = np.zeros(
    (marks_len, marks_len),
    dtype=np.float64
)

J_true[0, 1] = J_true[1, 0] = 0.95
J_true[0, 2] = J_true[2, 0] = -0.43
J_true[0, 3] = J_true[3, 0] = 0.23
J_true[0, 4] = J_true[4, 0] = 0.15
J_true[0, 5] = J_true[5, 0] = -0.78
J_true[0, 6] = J_true[6, 0] = -0.10

J_true[1, 2] = J_true[2, 1] = -0.27
J_true[1, 3] = J_true[3, 1] = 0.33
J_true[1, 4] = J_true[4, 1] = -0.50
J_true[1, 5] = J_true[5, 1] = 0.60
J_true[1, 6] = J_true[6, 1] = -0.81

J_true[2, 3] = J_true[3, 2] = -0.66
J_true[2, 4] = J_true[4, 2] = -0.89
J_true[2, 5] = J_true[5, 2] = -0.05
J_true[2, 6] = J_true[6, 2] = -0.25

J_true[3, 4] = J_true[4, 3] = 0.13
J_true[3, 5] = J_true[5, 3] = 0.27
J_true[3, 6] = J_true[6, 3] = -0.91

J_true[4, 5] = J_true[5, 4] = 0.22
J_true[4, 6] = J_true[6, 4] = 0.22

J_true[5, 6] = J_true[6, 5] = 0.10


print("\nTRUE SYNTHETIC LOCAL FIELDS")

print(
    pd.DataFrame({
        "histone_mark": MARKS,
        "h_true": h_true
    }).to_string(index=False)
)


print("\nTRUE SYNTHETIC COUPLING MATRIX")

print(
    pd.DataFrame(
        J_true,
        index=MARKS,
        columns=MARKS
    ).to_string()
)


# ------------------------------------------------------------
# Generate synthetic probability distribution
# ------------------------------------------------------------

synthetic_prob = state_probabilities(
    h_true,
    J_true
)

print(
    f"\nSynthetic probability sum: "
    f"{synthetic_prob.sum()}"
)

print(
    f"Synthetic probability minimum: "
    f"{synthetic_prob.min()}"
)


# ------------------------------------------------------------
# Sample synthetic windows
# ------------------------------------------------------------

synthetic_rng = np.random.default_rng(
    seed=123
)

sampled_state_indices = (
    synthetic_rng.choice(
        len(states),
        size=100000,
        p=synthetic_prob
    )
)

X_synthetic = states[
    sampled_state_indices
]

print(
    f"\nSynthetic data shape: "
    f"{X_synthetic.shape}"
)

print(
    X_synthetic[:5]
)


# ------------------------------------------------------------
# Fit synthetic data
# ------------------------------------------------------------

theta_synthetic = np.zeros(
    parameters_len
)

synthetic_lambda = 0.0

minimize_synthetic = minimize(
    PL_regularization_diff,
    theta_synthetic,
    args=(
        X_synthetic,
        synthetic_lambda
    ),
    method="L-BFGS-B",
    options={
        "maxiter": 1000,
        "ftol": 1e-10,
        "gtol": 1e-6,
        "maxls": 50
    }
)


h_recovered, J_recovered = parameters(
    minimize_synthetic.x
)


# ------------------------------------------------------------
# Synthetic recovery metrics
# ------------------------------------------------------------

J_true_values = np.array([
    J_true[i, j]
    for i, j in pairs
])

J_recovered_values = np.array([
    J_recovered[i, j]
    for i, j in pairs
])


MAE_h = np.mean(
    np.abs(
        h_recovered
        - h_true
    )
)

RMSE_h = np.sqrt(
    np.mean(
        (
            h_recovered
            - h_true
        ) ** 2
    )
)

MAE_J = np.mean(
    np.abs(
        J_recovered_values
        - J_true_values
    )
)

RMSE_J = np.sqrt(
    np.mean(
        (
            J_recovered_values
            - J_true_values
        ) ** 2
    )
)


print(
    f"\nMAE for h: {MAE_h}"
)

print(
    f"RMSE for h: {RMSE_h}"
)

print(
    f"MAE for J: {MAE_J}"
)

print(
    f"RMSE for J: {RMSE_J}"
)

print(
    f"Synthetic optimization success: "
    f"{minimize_synthetic.success}"
)

print(
    f"Synthetic optimization message: "
    f"{minimize_synthetic.message}"
)


# ------------------------------------------------------------
# Save synthetic recovery
# ------------------------------------------------------------

synthetic_recovery = pd.DataFrame({
    "parameter_type": [
        "h",
        "h",
        "J",
        "J"
    ],
    "metric": [
        "MAE",
        "RMSE",
        "MAE",
        "RMSE"
    ],
    "value": [
        MAE_h,
        RMSE_h,
        MAE_J,
        RMSE_J
    ]
})

synthetic_recovery.to_csv(
    OUTPUT_DIR
    / "synthetic_recovery.csv",
    index=False
)


# ============================================================
# STATE INDEX
# ============================================================

state_to_index = {
    tuple(state.astype(int)): i
    for i, state in enumerate(states)
}


# ============================================================
# RUN MAXIMUM ENTROPY MODEL FOR EACH CELL TYPE
# ============================================================

for CELL_TYPE in CELL_TYPES:

    print("\n")
    print("=" * 70)
    print(f"RUNNING MAXIMUM ENTROPY MODEL: {CELL_TYPE}")
    print("=" * 70)


    # ========================================================
    # CELL-TYPE OUTPUT DIRECTORY
    # ========================================================

    CELL_OUTPUT_DIR = (
        OUTPUT_DIR
        / CELL_TYPE
    )

    CELL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # READ WINDOW MATRIX
    # ========================================================

    input_file = (
        INPUT_DIR
        / f"{CELL_TYPE}_window_matrix.parquet"
    )

    print(
        f"\nReading: {input_file}"
    )

    df = pd.read_parquet(
        input_file
    )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "chromosome",
        "start",
        "end"
    ] + MARKS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{CELL_TYPE} is missing columns: "
            f"{missing_columns}"
        )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    #
    # chr20 is held out for validation.
    # ========================================================

    train_df = df[
        df["chromosome"]
        != VALIDATION_CHROMOSOME
    ]

    validation_df = df[
        df["chromosome"]
        == VALIDATION_CHROMOSOME
    ]


    X_train = train_df[
        MARKS
    ].to_numpy(
        dtype=np.float64
    )

    validation_X = validation_df[
        MARKS
    ].to_numpy(
        dtype=np.float64
    )

    X = df[
        MARKS
    ].to_numpy(
        dtype=np.float64
    )


    N = X.shape[0]


    print(
        f"Total windows: {N}"
    )

    print(
        f"Training windows: "
        f"{X_train.shape[0]}"
    )

    print(
        f"Validation windows: "
        f"{validation_X.shape[0]}"
    )

    print(
        f"Validation chromosome: "
        f"{VALIDATION_CHROMOSOME}"
    )


    # ========================================================
    # CHECK DATA
    # ========================================================

    if X_train.shape[0] == 0:

        raise ValueError(
            f"{CELL_TYPE}: training set is empty."
        )

    if validation_X.shape[0] == 0:

        raise ValueError(
            f"{CELL_TYPE}: validation set is empty. "
            f"Check chromosome labels."
        )

    if not np.all(
        np.isin(X, [0, 1])
    ):

        raise ValueError(
            f"{CELL_TYPE}: input matrix contains "
            f"values other than 0 and 1."
        )


    # ========================================================
    # INITIAL MODEL CHECK
    # ========================================================

    theta_test = np.zeros(
        parameters_len
    )

    test_h, test_J = parameters(
        theta_test
    )

    test_fields = (
        test_h
        + validation_X @ test_J.T
    )

    test_probabilities = (
        state_probabilities(
            test_h,
            test_J
        )
    )


    print(
        f"Number of possible states: "
        f"{len(states)}"
    )

    print(
        f"Initial probability sum: "
        f"{test_probabilities.sum()}"
    )

    print(
        f"Initial minimum probability: "
        f"{test_probabilities.min()}"
    )

    print(
        f"Initial maximum probability: "
        f"{test_probabilities.max()}"
    )


    # ========================================================
    # INITIAL VALIDATION PSEUDO-LIKELIHOOD
    # ========================================================

    test_log_PL = (
        pseudo_log_likelihood(
            theta_test,
            validation_X
        )
    )

    initial_average_log_PL = (
        test_log_PL
        / (
            validation_X.shape[0]
            * marks_len
        )
    )

    print(
        f"Initial validation log "
        f"pseudo-likelihood: "
        f"{test_log_PL:.6f}"
    )

    print(
        f"Initial validation average "
        f"log pseudo-likelihood: "
        f"{initial_average_log_PL:.6f}"
    )

    print(
        f"Expected average at theta=0: "
        f"{-marks_len * np.log(2):.6f}"
    )


    # ========================================================
    # LAMBDA SEARCH
    # ========================================================

    theta_initial = np.zeros(
        parameters_len
    )

    lambda_results = []


    for lambda_reg in LAMBDA_VALUES:

        print(
            f"\n{CELL_TYPE} | lambda = "
            f"{lambda_reg}"
        )


        theta_initial = np.zeros(
            parameters_len
        )


        result = minimize(
            PL_regularization_diff,
            theta_initial,
            args=(
                X_train,
                lambda_reg
            ),
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 50
            }
        )


        print(
            f"Optimization success: "
            f"{result.success}"
        )

        print(
            f"Optimization message: "
            f"{result.message}"
        )

        print(
            f"Objective value: "
            f"{result.fun:.6f}"
        )

        print(
            f"Parameter range: "
            f"{result.x.min():.6f} "
            f"to "
            f"{result.x.max():.6f}"
        )


        # ----------------------------------------------------
        # Validation pseudo-likelihood
        # ----------------------------------------------------

        validation_log_PL = (
            pseudo_log_likelihood(
                result.x,
                validation_X
            )
        )


        validation_average_log_PL = (
            validation_log_PL
            / (
                validation_X.shape[0]
                * marks_len
            )
        )


        lambda_results.append({
            "lambda": lambda_reg,
            "validation_log_PL":
                validation_log_PL,
            "validation_average_log_PL":
                validation_average_log_PL,
            "optimization_success":
                result.success,
            "iterations":
                result.nit
        })


        print(
            f"Validation average log "
            f"pseudo-likelihood: "
            f"{validation_average_log_PL:.10f}"
        )


    # ========================================================
    # SELECT BEST LAMBDA
    # ========================================================

    lambda_results_df = pd.DataFrame(
        lambda_results
    )


    best_index = (
        lambda_results_df[
            "validation_average_log_PL"
        ].idxmax()
    )


    best_lambda = (
        lambda_results_df.loc[
            best_index,
            "lambda"
        ]
    )


    print(
        f"\nFINAL SELECTED LAMBDA "
        f"FOR {CELL_TYPE}: "
        f"{best_lambda:.10g}"
    )


    lambda_results_df.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_lambda_search.csv",
        index=False
    )


    pd.DataFrame({
        "selected_lambda": [
            best_lambda
        ]
    }).to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_selected_lambda.csv",
        index=False
    )


    # ========================================================
    # FINAL MODEL FIT
    #
    # Refit using all available windows after selecting lambda.
    # ========================================================

    theta_initial = np.zeros(
        parameters_len
    )


    pseudo_result = minimize(
        PL_regularization_diff,
        theta_initial,
        args=(
            X,
            best_lambda
        ),
        method="L-BFGS-B",
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50
        }
    )


    print(
        f"\nFinal pseudo-likelihood success: "
        f"{pseudo_result.success}"
    )

    print(
        f"Final optimization message: "
        f"{pseudo_result.message}"
    )


    h_pseudo, J_pseudo = parameters(
        pseudo_result.x
    )


    # ========================================================
    # FINAL MODEL PROBABILITIES
    # ========================================================

    final_probability = (
        state_probabilities(
            h_pseudo,
            J_pseudo
        )
    )


    print(
        "\nFinal Maximum Entropy Model"
    )

    print(
        f"Number of possible states: "
        f"{len(states)}"
    )

    print(
        f"Probability sum: "
        f"{final_probability.sum()}"
    )

    print(
        f"Min probability: "
        f"{final_probability.min()}"
    )

    print(
        f"Max probability: "
        f"{final_probability.max()}"
    )


    # ========================================================
    # LOCAL FIELDS
    # ========================================================

    local_fields = pd.DataFrame({
        "histone_mark": MARKS,
        "h": h_pseudo
    })


    print(
        "\nLOCAL FIELDS"
    )

    print(
        local_fields.to_string(
            index=False
        )
    )


    local_fields.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_local_fields.csv",
        index=False
    )


    # ========================================================
    # COUPLING MATRIX
    # ========================================================

    coupling_matrix = pd.DataFrame(
        J_pseudo,
        index=MARKS,
        columns=MARKS
    )


    print(
        "\nPAIRWISE COUPLING MATRIX"
    )

    print(
        coupling_matrix.to_string()
    )


    coupling_matrix.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_coupling_matrix.csv"
    )


    # ========================================================
    # SAVE MODEL PARAMETERS
    # ========================================================

    np.save(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_parameters_pseudolikelihood.npy",
        pseudo_result.x
    )


    # ========================================================
    # SAVE STATE PROBABILITIES
    # ========================================================

    state_table = pd.DataFrame(
        states.astype(int),
        columns=MARKS
    )

    state_table[
        "model_probability"
    ] = final_probability


    state_table.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_state_probabilities.csv",
        index=False
    )


    # ========================================================
    # EMPIRICAL STATE DISTRIBUTION
    # ========================================================

    empirical_counts = np.zeros(
        len(states)
    )


    for window in X:

        state_index = state_to_index[
            tuple(
                window.astype(int)
            )
        ]

        empirical_counts[
            state_index
        ] += 1


    empirical_probability = (
        empirical_counts
        / X.shape[0]
    )


    # ========================================================
    # EMPIRICAL VS MODEL METRICS
    # ========================================================

    MAE_distribution = np.mean(
        np.abs(
            empirical_probability
            - final_probability
        )
    )


    RMSE_distribution = np.sqrt(
        np.mean(
            (
                empirical_probability
                - final_probability
            ) ** 2
        )
    )


    max_difference = np.max(
        np.abs(
            empirical_probability
            - final_probability
        )
    )


    KL_pairwise = kl_divergence(
        empirical_probability,
        final_probability
    )


    # ========================================================
    # INDEPENDENT MODEL
    #
    # Fit the independent model directly from the empirical
    # marginal frequencies.
    # ========================================================

    empirical_marginals = (
        X.mean(axis=0)
    )


    # Protect against log(0) / log(infinity)
    # if a mark has no variation.
    clipped_marginals = np.clip(
        empirical_marginals,
        1e-12,
        1 - 1e-12
    )


    h_independent = np.log(
        clipped_marginals
        / (
            1
            - clipped_marginals
        )
    )


    independent_probability = (
        state_probabilities(
            h_independent,
            np.zeros_like(J_pseudo)
        )
    )


    KL_independent = kl_divergence(
        empirical_probability,
        independent_probability
    )


    # ========================================================
    # SAVE GOODNESS-OF-FIT METRICS
    # ========================================================

    goodness_of_fit = pd.DataFrame({
        "cell_type": [CELL_TYPE],
        "MAE_distribution": [
            MAE_distribution
        ],
        "RMSE_distribution": [
            RMSE_distribution
        ],
        "maximum_absolute_difference": [
            max_difference
        ],
        "KL_independent": [
            KL_independent
        ],
        "KL_pairwise": [
            KL_pairwise
        ],
        "pseudo_likelihood_success": [
            pseudo_result.success
        ],
        "selected_lambda": [
            best_lambda
        ]
    })


    goodness_of_fit.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_goodness_of_fit.csv",
        index=False
    )


    # ========================================================
    # SAVE EMPIRICAL + MODEL STATE PROBABILITIES
    # ========================================================

    state_comparison = pd.DataFrame(
        states.astype(int),
        columns=MARKS
    )


    state_comparison[
        "empirical_probability"
    ] = empirical_probability


    state_comparison[
        "model_probability"
    ] = final_probability


    state_comparison[
        "independent_probability"
    ] = independent_probability


    state_comparison[
        "absolute_difference"
    ] = np.abs(
        empirical_probability
        - final_probability
    )


    state_comparison.to_csv(
        CELL_OUTPUT_DIR
        / f"{CELL_TYPE}_state_probability_comparison.csv",
        index=False
    )


    # ========================================================
    # PRINT GOODNESS OF FIT
    # ========================================================

    print(
        "\nEMPIRICAL VS MODEL DISTRIBUTION"
    )

    print(
        f"Empirical probability sum: "
        f"{empirical_probability.sum()}"
    )

    print(
        f"Model probability sum: "
        f"{final_probability.sum()}"
    )

    print(
        f"MAE: "
        f"{MAE_distribution}"
    )

    print(
        f"RMSE: "
        f"{RMSE_distribution}"
    )

    print(
        f"Maximum absolute difference: "
        f"{max_difference}"
    )

    print(
        f"KL divergence - independent model: "
        f"{KL_independent}"
    )

    print(
        f"KL divergence - pairwise MaxEnt model: "
        f"{KL_pairwise}"
    )


    # ========================================================
    # FINISHED CELL TYPE
    # ========================================================

    print("\n")
    print(
        f"FINISHED: {CELL_TYPE}"
    )

    print(
        f"Results saved to: "
        f"{CELL_OUTPUT_DIR}"
    )


# ============================================================
# ALL FIVE CELL TYPES FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("ALL FIVE CELL TYPES FINISHED")
print("=" * 70)

print(
    f"\nResults directory: "
    f"{OUTPUT_DIR}"
)

for CELL_TYPE in CELL_TYPES:

    print(
        f"  {CELL_TYPE}: "
        f"{OUTPUT_DIR / CELL_TYPE}"
    )

print(
    "\nPhase 9 complete."
)