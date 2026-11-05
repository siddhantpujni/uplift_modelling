# Criteo uplift modelling

This project uses the Criteo uplift dataset. The default development data is
the approximately 1.4M-row 10% split. The modelling outcome is `conversion`
and treatment is the Criteo `treatment` flag; `exposure` is retained as a
separate audit field.

## Project goal

The analysis estimates heterogeneous incremental effects of advertising in the
Criteo randomised experiment. `conversion` is the primary outcome and `visit`
is a secondary outcome. I will first test and develop using 10% of the 14M row criteo dataset to reduce computational time and then move to the full dataset after the pipeline is validated.

The repository is deliberately structured as a small Python package rather
than a notebook-only analysis. Reusable analysis code will be under [`src/`](src),
tests will be under [`tests/`](tests), and GitHub Actions runs linting and tests on
every push and pull request. Notebooks are used for general testing and results before full implementation.

## Local setup

Keep the downloaded Criteo CSV outside the repository. Then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check .
```

The first milestone implements the validated data contract, covariate balance,
bootstrap ATE confidence intervals, a T-learner baseline, and uplift-curve
metrics. Subsequent milestones will add cross-fitting, richer learners,
uncertainty for model comparisons, biased-sample experiments, and off-policy
policy evaluation.

To run the first milestone against a local download:

```bash
python scripts/run_baseline_analysis.py /path/to/criteo-uplift-v2.1.csv \
  --nrows 200000 \
  --output-dir reports/baseline
```

The generated report directory is intentionally ignored by Git. It contains
the balance table, uplift curve, and a JSON summary that can be attached to a
project write-up or published as a CI artifact later.
