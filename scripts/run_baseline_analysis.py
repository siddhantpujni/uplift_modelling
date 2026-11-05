"""Run the reproducible first-milestone analysis on a local Criteo CSV."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from uplift_modelling.constants import FEATURE_COLUMNS, REQUIRED_COLUMNS
from uplift_modelling.data import load_criteo_csv
from uplift_modelling.metrics import (
    auuc,
    balance_table,
    bootstrap_ate,
    uplift_curve,
)
from uplift_modelling.models import TLearner


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
    pass

if __name__ == "__main__":
    main()
