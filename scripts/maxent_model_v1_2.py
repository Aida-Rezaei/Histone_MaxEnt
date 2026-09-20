from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "window_matrices"
OUTPUT_DIR = PROJECT_ROOT / "results" / "maxent_v1_2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MARKS = [
    "H3K4me3",
    "H3K27me3",
    "H3K4me1",
    "H3K27ac",
    "H3K36me3",
    "H3K9me3",
    "H3K9ac"
]

CELL_TYPES = [
    "H1",
    "GM12878",
    "K562",
    "HepG2",
    "IMR90"
]

M = len(MARKS)

PAIRS = [
    (i, j)
    for i in range(M)
    for j in range(i + 1, M)
]

N_PARAMETERS = M + len(PAIRS)

LAMBDA_VALUES = 1e-5

def unpack_parameters(theta):
    h = theta[:M]

    J = np.zeros((M, M))

    values = theta[M:]

    for value, (i, j) in zip(values, PAIRS):
        J[i, j] = value
        J[j, i] = value

    return h, J

def energy(sigma, h, J):
    local = -np.dot(h, sigma)

    pairwise = 0.0

    for i, j in PAIRS:
        pairwise -= J[i, j] * sigma[i] * sigma[j]

    return local + pairwise

from itertools import product

STATES = np.array(
    list(product([0, 1], repeat=M)),
    dtype=np.float64
)

print("Number of configurations:", len(STATES))

def pseudo_log_likelihood(theta, X):
    h, J = unpack_parameters(theta)

    fields = h + X @ J.T

    log_p1 = -np.logaddexp(0, -fields)
    log_p0 = -np.logaddexp(0, fields)

    log_conditional = (
        X * log_p1
        + (1 - X) * log_p0
    )

    return np.sum(log_conditional)

def regularization(theta, lambda_reg):
    h, J = unpack_parameters(theta)

    h_penalty = np.sum(h ** 2)

    J_penalty = sum(
        J[i, j] ** 2
        for i, j in PAIRS
    )

    return 0.5 * lambda_reg * (
        h_penalty + J_penalty
    )

def objective(theta, X, lambda_reg):
    return (
        regularization(theta, lambda_reg)
        - pseudo_log_likelihood(theta, X)
    )

def fit_model(X, lambda_reg):

    theta0 = np.zeros(N_PARAMETERS)

    result = minimize(
        objective,
        theta0,
        args=(X, lambda_reg),
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50
        }
    )

    return result

def fit_model(X, lambda_reg):

    theta0 = np.zeros(N_PARAMETERS)

    result = minimize(
        objective,
        theta0,
        args=(X, lambda_reg),
        method="L-BFGS-B",
        options={
            "maxiter": 2000,
            "ftol": 1e-10,
            "gtol": 1e-6,
            "maxls": 50
        }
    )

    return result

best_lambda, lambda_results = select_lambda(
    X_train,
    validation_X
)

print("\nSelected lambda:", best_lambda)

lambda_results.to_csv(
    OUTPUT_DIR / f"{cell_type}_lambda_search.csv",
    index=False
)

final_result = fit_model(
    X,
    best_lambda
)

theta_final = final_result.x

h_final, J_final = unpack_parameters(theta_final)

pd.DataFrame({
    "histone_mark": MARKS,
    "h": h_final
}).to_csv(
    OUTPUT_DIR / f"{cell_type}_local_fields.csv",
    index=False
)


pd.DataFrame(
    J_final,
    index=MARKS,
    columns=MARKS
).to_csv(
    OUTPUT_DIR / f"{cell_type}_coupling_matrix.csv"
)


np.save(
    OUTPUT_DIR / f"{cell_type}_parameters.npy",
    theta_final
)


pd.DataFrame({
    "cell_type": [cell_type],
    "lambda": [best_lambda],
    "optimization_success": [final_result.success],
    "iterations": [final_result.nit]
}).to_csv(
    OUTPUT_DIR / f"{cell_type}_model_summary.csv",
    index=False
)

def model_probabilities(h, J):

    energies = np.array([
        energy(state, h, J)
        for state in STATES
    ])

    log_Z = logsumexp(-energies)

    probabilities = np.exp(
        -energies - log_Z
    )

    return probabilities

def empirical_probabilities(X):

    counts = np.zeros(len(STATES))

    state_lookup = {
        tuple(state.astype(int)): i
        for i, state in enumerate(STATES)
    }

    for row in X:

        idx = state_lookup[
            tuple(row.astype(int))
        ]

        counts[idx] += 1

    return counts / len(X)

p_empirical = empirical_probabilities(X)
p_model = model_probabilities(h_final, J_final)

print(
    "Empirical probability sum:",
    p_empirical.sum()
)

print(
    "Model probability sum:",
    p_model.sum()
)

def kl_divergence(p_obs, p_model):

    mask = p_obs > 0

    return np.sum(
        p_obs[mask]
        * np.log(
            p_obs[mask] /
            p_model[mask]
        )
    )

p_independent = np.ones(len(STATES))

for i, state in enumerate(STATES):

    probability = 1.0

    for j in range(M):

        p = X[:, j].mean()

        if state[j] == 1:
            probability *= p
        else:
            probability *= (1 - p)

    p_independent[i] = probability

p_independent /= p_independent.sum()

KL_independent = kl_divergence(
    p_empirical,
    p_independent
)

KL_pairwise = kl_divergence(
    p_empirical,
    p_model
)

fraction_explained = (
    1 -
    KL_pairwise / KL_independent
)

print(
    "KL independent:",
    KL_independent
)

print(
    "KL pairwise:",
    KL_pairwise
)

print(
    "Fraction explained:",
    fraction_explained
)

#LATER WILL CREAT scripts/plot_configuration_fit.py BUT NOT NOW
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = pd.read_csv(
    "results/maxent_v1_2/GM12878_configuration_probabilities.csv"
)

plt.figure(figsize=(6, 6))

plt.scatter(
    RESULTS["empirical_probability"],
    RESULTS["model_probability"]
)

maximum = max(
    RESULTS["empirical_probability"].max(),
    RESULTS["model_probability"].max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel("Empirical probability")
plt.ylabel("Model probability")

plt.title(
    "GM12878: empirical vs maximum-entropy probabilities"
)

plt.tight_layout()

plt.savefig(
    "figures/figure1_configuration_fit_maxent_v1_2.png",
    dpi=300
)

plt.show()

pd.DataFrame({
    "configuration": [
        "".join(map(str, state.astype(int)))
        for state in STATES
    ],
    "empirical_probability": p_empirical,
    "model_probability": p_model
}).to_csv(
    OUTPUT_DIR /
    f"{cell_type}_configuration_probabilities.csv",
    index=False
)

def run_synthetic_test(seed):

    rng = np.random.default_rng(seed)

    h_true = rng.normal(
        0,
        0.5,
        M
    )

    J_true = np.zeros((M, M))

    random_J = rng.normal(
        0,
        0.3,
        len(PAIRS)
    )

    for value, (i, j) in zip(
        random_J,
        PAIRS
    ):
        J_true[i, j] = value
        J_true[j, i] = value

    probabilities = model_probabilities(
        h_true,
        J_true
    )

    indices = rng.choice(
        len(STATES),
        size=100_000,
        p=probabilities
    )

    X_syn = STATES[indices]

    result = fit_model(
        X_syn,
        lambda_reg=0.0
    )

    h_recovered, J_recovered = (
        unpack_parameters(result.x)
    )

    true_J = np.array([
        J_true[i, j]
        for i, j in PAIRS
    ])

    recovered_J = np.array([
        J_recovered[i, j]
        for i, j in PAIRS
    ])

    h_mae = np.mean(
        np.abs(h_recovered - h_true)
    )

    J_mae = np.mean(
        np.abs(recovered_J - true_J)
    )

    h_rmse = np.sqrt(
        np.mean(
            (h_recovered - h_true) ** 2
        )
    )

    J_rmse = np.sqrt(
        np.mean(
            (recovered_J - true_J) ** 2
        )
    )

    h_r = np.corrcoef(
        h_true,
        h_recovered
    )[0, 1]

    J_r = np.corrcoef(
        true_J,
        recovered_J
    )[0, 1]

    return {
        "seed": seed,
        "h_MAE": h_mae,
        "h_RMSE": h_rmse,
        "h_correlation": h_r,
        "J_MAE": J_mae,
        "J_RMSE": J_rmse,
        "J_correlation": J_r,
        "success": result.success
    }

def run_synthetic_test(seed):

    rng = np.random.default_rng(seed)

    h_true = rng.normal(
        0,
        0.5,
        M
    )

    J_true = np.zeros((M, M))

    random_J = rng.normal(
        0,
        0.3,
        len(PAIRS)
    )

    for value, (i, j) in zip(
        random_J,
        PAIRS
    ):
        J_true[i, j] = value
        J_true[j, i] = value

    probabilities = model_probabilities(
        h_true,
        J_true
    )

    indices = rng.choice(
        len(STATES),
        size=100_000,
        p=probabilities
    )

    X_syn = STATES[indices]

    result = fit_model(
        X_syn,
        lambda_reg=0.0
    )

    h_recovered, J_recovered = (
        unpack_parameters(result.x)
    )

    true_J = np.array([
        J_true[i, j]
        for i, j in PAIRS
    ])

    recovered_J = np.array([
        J_recovered[i, j]
        for i, j in PAIRS
    ])

    h_mae = np.mean(
        np.abs(h_recovered - h_true)
    )

    J_mae = np.mean(
        np.abs(recovered_J - true_J)
    )

    h_rmse = np.sqrt(
        np.mean(
            (h_recovered - h_true) ** 2
        )
    )

    J_rmse = np.sqrt(
        np.mean(
            (recovered_J - true_J) ** 2
        )
    )

    h_r = np.corrcoef(
        h_true,
        h_recovered
    )[0, 1]

    J_r = np.corrcoef(
        true_J,
        recovered_J
    )[0, 1]

    return {
        "seed": seed,
        "h_MAE": h_mae,
        "h_RMSE": h_rmse,
        "h_correlation": h_r,
        "J_MAE": J_mae,
        "J_RMSE": J_rmse,
        "J_correlation": J_r,
        "success": result.success
    }

import pandas as pd
import matplotlib.pyplot as plt

J = pd.read_csv(
    "results/maxent/GM12878_coupling_matrix.csv",
    index_col=0
)

plt.figure(figsize=(7, 6))

plt.imshow(J.values)

plt.xticks(
    range(len(J.columns)),
    J.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(J.index)),
    J.index
)

plt.colorbar(
    label="Coupling J"
)

plt.title(
    "GM12878 pairwise histone coupling"
)

plt.tight_layout()

plt.savefig(
    "figures/GM12878_coupling_heatmap.png",
    dpi=300
)

plt.show()

