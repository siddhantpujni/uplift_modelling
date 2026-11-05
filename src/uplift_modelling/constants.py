"""Dataset schema and project-wide defaults."""

FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
OUTCOME_COLUMNS = ["conversion", "visit"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["treatment", *OUTCOME_COLUMNS, "exposure"]
