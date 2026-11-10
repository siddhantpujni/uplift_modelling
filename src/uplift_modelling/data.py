"""Loading and validating the Criteo uplift data."""

from pathlib import Path

import pandas as pd

from .constants import REQUIRED_COLUMNS


def load_criteo_csv(
    path: str | Path,
    *,
    nrows: int | None = None,
    sample_fraction: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a Criteo CSV and enforce the expected schema.

    ``sample_fraction`` is useful for local development; it is applied after
    loading so that the same validation rules are used for every run.
    """
    if sample_fraction is not None and not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")

    data = pd.read_csv(path, nrows=nrows)
    validate_criteo_data(data)
    if sample_fraction is not None and sample_fraction < 1:
        data = data.sample(frac=sample_fraction, random_state=random_state)
        data = data.reset_index(drop=True)
    return data


def validate_criteo_data(data: pd.DataFrame) -> None:
    """Raise a useful error when data cannot be analysed safely."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data[REQUIRED_COLUMNS].isna().any().any():
        missing_counts = data[REQUIRED_COLUMNS].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Required columns contain missing values: {missing_counts}")

    for column in ["treatment", "conversion", "visit", "exposure"]:
        values = set(data[column].unique())
        if not values <= {0, 1}:
            raise ValueError(f"{column} must be binary, found values: {sorted(values)}")

    if data["treatment"].nunique() != 2:
        raise ValueError("treatment must contain both treated and control observations.")

    if not (data["conversion"] <= data["visit"]).all():
        raise ValueError("Conversion must not be one when visit is zero.")

    if (data.loc[data["treatment"] == 0, "exposure"] == 1).any():
        raise ValueError("Control observations must not have exposure=1.")
