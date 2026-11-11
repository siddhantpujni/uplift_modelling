"""Run the reproducible first-milestone analysis on a local Criteo CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_modelling.conformal import SplitConformalCATE
from uplift_modelling.constants import FEATURE_COLUMNS, REQUIRED_COLUMNS
from uplift_modelling.data import load_criteo_csv
from uplift_modelling.metrics import (
    auuc,
    balance_table,
    bootstrap_ate,
    bootstrap_auuc,
    random_curve,
    uplift_curve,
)
from uplift_modelling.models import (
    CausalForest,
    DRLearner,
    RLearner,
    TLearner,
    TorchUpliftNetwork,
    XLearner,
)

MODEL_CLASSES = {
    "t": TLearner,
    "x": XLearner,
    "r": RLearner,
    "dr": DRLearner,
    "forest": CausalForest,
    "torch": TorchUpliftNetwork,
}


def load_stratified_slice(path: Path, nrows: int, random_state: int) -> pd.DataFrame:
    """Read a balanced slice without assuming the CSV is randomly ordered."""
    target_per_group = nrows // 2
    collected = {0: [], 1: []}
    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=100_000):
        for treatment in (0, 1):
            remaining = target_per_group - sum(len(part) for part in collected[treatment])
            if remaining > 0:
                group = chunk[chunk["treatment"] == treatment].head(remaining)
                collected[treatment].append(group)
        enough_rows = all(
            sum(len(part) for part in parts) >= target_per_group
            for parts in collected.values()
        )
        if enough_rows:
            break
    data = pd.concat([part for parts in collected.values() for part in parts], ignore_index=True)
    if len(data) < nrows:
        raise ValueError(f"Could not collect {nrows} rows with both treatment groups.")
    return data.sample(frac=1, random_state=random_state).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baseline"))
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        default="t,x,r,dr,forest",
        help="Comma-separated models: t,x,r,dr,forest,torch.",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    data = (
        load_stratified_slice(args.csv_path, args.nrows, args.random_state)
        if args.nrows is not None
        else load_criteo_csv(args.csv_path)
    )
    train, test = train_test_split(
        data, test_size=0.25, random_state=args.random_state, stratify=data["treatment"]
    )
    calibration, test = train_test_split(
        test, test_size=0.5, random_state=args.random_state, stratify=test["treatment"]
    )
    requested_models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(requested_models) - set(MODEL_CLASSES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Choose from {sorted(MODEL_CLASSES)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    balance_table(data, FEATURE_COLUMNS).to_csv(args.output_dir / "balance.csv", index=False)
    random_targeting = random_curve(
        test["conversion"], test["treatment"], random_state=args.random_state
    )
    random_targeting.to_csv(args.output_dir / "random_curve.csv", index=False)
    model_results = {}
    for model_name in requested_models:
        model_kwargs = {"random_state": args.random_state}
        if model_name in {"x", "r", "dr", "forest"}:
            model_kwargs["n_estimators"] = args.n_estimators
        if model_name == "torch":
            model_kwargs["epochs"] = args.epochs
        learner = MODEL_CLASSES[model_name](**model_kwargs).fit(
            train[FEATURE_COLUMNS], train["treatment"], train["conversion"]
        )
        test_uplift = learner.predict_uplift(test[FEATURE_COLUMNS])
        curve = uplift_curve(test["conversion"], test["treatment"], test_uplift)
        curve.to_csv(args.output_dir / f"{model_name}_uplift_curve.csv", index=False)
        interval = SplitConformalCATE(learner).fit(
            calibration[FEATURE_COLUMNS],
            calibration["treatment"],
            calibration["conversion"],
        )
        lower, upper = interval.predict_interval(test[FEATURE_COLUMNS])
        model_results[model_name] = {
            "auuc": bootstrap_auuc(
                test["conversion"],
                test["treatment"],
                test_uplift,
                n_bootstrap=500,
                random_state=args.random_state,
            ),
            "conformal_radius": float(interval.radius_),
            "conformal_interval_width": float(np.mean(upper - lower)),
            "conformal_test_coverage": interval.empirical_coverage(
                test[FEATURE_COLUMNS], test["treatment"], test["conversion"]
            ),
        }
    results = {
        "rows": len(data),
        "treatment_rate": float(data["treatment"].mean()),
        "ate_conversion": bootstrap_ate(
            data["conversion"],
            data["treatment"],
            n_bootstrap=500,
            random_state=args.random_state,
        ),
        "auuc_random_targeting": auuc(random_targeting),
        "models": model_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
