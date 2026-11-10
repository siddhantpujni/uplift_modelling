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
than a notebook-only analysis. Reusable analysis code is under [`src/`](src),
tests are under [`tests/`](tests), and GitHub Actions runs linting, coverage,
tests, and package builds on every push and pull request. Notebooks are used
for narrative exploration and results, not as the only place where logic exists.

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
the balance table, uplift, response, and random-targeting curves, and a JSON
summary that can be attached to a project write-up.

## CI/CD workflows

- **CI** tests Python 3.10, 3.11, and 3.12, checks coverage, linting, and builds
  wheel/source distributions.
- **Full analysis** is manually triggered from GitHub Actions. Supply a
  downloadable dataset URL, or adapt the workflow for a private/self-hosted
  runner. Leave `nrows` blank for the full dataset; use a value such as
  `200000` for a bounded development run. Reports are uploaded as artifacts.
- **Release** runs on tags such as `v0.2.0`, reruns quality checks, builds the
  distributions, and attaches them to a GitHub Release.
- **Dependabot** checks Python and GitHub Actions dependencies monthly.

For pull requests, repository settings should require the `test-and-lint`
check before merging into `main`. The full-data workflow is intentionally not
part of pull-request CI because the dataset and model runs are expensive.
