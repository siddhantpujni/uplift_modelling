"""Causal and uplift evaluation metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def ate_difference(outcome: Sequence[float], treatment: Sequence[int]) -> float:
    """Estimate the ATE by the treated-minus-control difference in means."""
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)
    if y.ndim != 1 or w.ndim != 1 or y.shape != w.shape:
        raise ValueError("outcome and treatment must be one-dimensional and have the same shape.")
    if not np.isfinite(y).all() or not np.isin(w, [0, 1]).all():
        raise ValueError("outcome must be finite and treatment must be binary.")
    if not np.any(w == 0) or not np.any(w == 1):
        raise ValueError("Both treatment groups must be present.")
    return float(y[w == 1].mean() - y[w == 0].mean())


def bootstrap_ate(
    outcome: Sequence[float],
    treatment: Sequence[int],
    *,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Return the ATE and percentile bootstrap confidence interval."""
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)
    ate_difference(y, w)

    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    attempts = 0
    max_attempts = n_bootstrap * 100
    while len(estimates) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(y), size=len(y))
        try:
            estimates.append(ate_difference(y[indices], w[indices]))
        except ValueError:
            continue
    if len(estimates) < n_bootstrap:
        raise ValueError("Could not produce enough bootstrap samples with both treatment groups.")
    interval = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": ate_difference(y, w),
        "ci_lower": float(interval[0]),
        "ci_upper": float(interval[1]),
        "n_bootstrap": n_bootstrap,
    }


def balance_table(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Calculate standardised mean differences for every baseline feature."""
    treated = data["treatment"] == 1
    rows = []
    for feature in feature_columns:
        x_treated = data.loc[treated, feature].astype(float)
        x_control = data.loc[~treated, feature].astype(float)
        pooled_sd = np.sqrt((x_treated.var(ddof=1) + x_control.var(ddof=1)) / 2)
        mean_difference = x_treated.mean() - x_control.mean()
        smd = 0.0 if pooled_sd == 0 and mean_difference == 0 else (
            np.inf if pooled_sd == 0 else mean_difference / pooled_sd
        )
        rows.append(
            {
                "feature": feature,
                "treated_mean": x_treated.mean(),
                "control_mean": x_control.mean(),
                "smd": smd,
                "absolute_smd": abs(smd),
            }
        )
    return pd.DataFrame(rows)


def uplift_curve(
    outcome: Sequence[int],
    treatment: Sequence[int],
    uplift_score: Sequence[float],
) -> pd.DataFrame:
    """Build an incremental-gain curve by descending predicted uplift."""
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)
    score = np.asarray(uplift_score, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or score.ndim != 1:
        raise ValueError("All inputs must be one-dimensional.")
    if not (len(y) == len(w) == len(score)):
        raise ValueError("All inputs must have the same length.")
    if not np.isfinite(y).all() or not np.isfinite(score).all():
        raise ValueError("outcome and uplift_score must contain finite values.")
    if not np.isin(w, [0, 1]).all() or not np.any(w == 0) or not np.any(w == 1):
        raise ValueError("treatment must be binary and contain both groups.")
    order = np.argsort(-score, kind="stable")
    ranked = pd.DataFrame({"outcome": y[order], "treatment": w[order]})
    treated = ranked["treatment"].eq(1)
    control = ~treated
    ranked["treated_n"] = treated.cumsum()
    ranked["control_n"] = control.cumsum()
    ranked["treated_outcomes"] = (ranked["outcome"] * treated).cumsum()
    ranked["control_outcomes"] = (ranked["outcome"] * control).cumsum()
    ranked["incremental_gain"] = (
        ranked["treated_outcomes"] / ranked["treated_n"].replace(0, np.nan)
        - ranked["control_outcomes"] / ranked["control_n"].replace(0, np.nan)
    ) * (ranked["treated_n"] + ranked["control_n"])
    ranked["fraction_targeted"] = (np.arange(len(ranked)) + 1) / len(ranked)
    return ranked.fillna(0)


def auuc(curve: pd.DataFrame) -> float:
    """Calculate area under the uplift curve using the trapezoid rule."""
    required = {"incremental_gain", "fraction_targeted"}
    if not required <= set(curve.columns):
        raise ValueError(f"curve must contain columns: {sorted(required)}")
    return float(np.trapezoid(curve["incremental_gain"], curve["fraction_targeted"]))


def response_curve(
    outcome: Sequence[int],
    treatment: Sequence[int],
    response_score: Sequence[float],
) -> pd.DataFrame:
    """Evaluate targeting by predicted treated response probability."""
    return uplift_curve(outcome, treatment, response_score)


def random_curve(
    outcome: Sequence[int], treatment: Sequence[int], *, random_state: int = 42
) -> pd.DataFrame:
    """Create a reproducible random-targeting benchmark curve."""
    rng = np.random.default_rng(random_state)
    return uplift_curve(outcome, treatment, rng.random(len(outcome)))


def bootstrap_auuc(
    outcome: Sequence[int],
    treatment: Sequence[int],
    uplift_score: Sequence[float],
    *,
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate AUUC and a percentile bootstrap confidence interval."""
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    y = np.asarray(outcome)
    w = np.asarray(treatment)
    score = np.asarray(uplift_score)
    if not (len(y) == len(w) == len(score)):
        raise ValueError("All inputs must have the same length.")
    rng = np.random.default_rng(random_state)
    estimates = []
    attempts = 0
    max_attempts = n_bootstrap * 100
    while len(estimates) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(y), size=len(y))
        try:
            sample = uplift_curve(y[indices], w[indices], score[indices])
        except ValueError:
            continue
        estimates.append(auuc(sample))
    if len(estimates) < n_bootstrap:
        raise ValueError("Could not produce enough bootstrap samples with both treatment groups.")
    interval = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": auuc(uplift_curve(y, w, score)),
        "ci_lower": float(interval[0]),
        "ci_upper": float(interval[1]),
        "n_bootstrap": n_bootstrap,
    }
