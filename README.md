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

The analysis package now includes T-, X-, R-, and doubly robust learners, a
cross-fitted DR learner, a dependency-light causal-forest baseline, a
PyTorch shared-trunk/treatment-head network, split-conformal CATE intervals,
bootstrap AUUC uncertainty, and random-targeting benchmarks. The remaining
research milestones are biased-sample sensitivity analysis and richer
off-policy policy evaluation.

To run the first milestone against a local download:

```bash
python scripts/run_baseline_analysis.py /path/to/criteo-uplift-v2.1.csv \
  --nrows 200000 \
  --models t,x,r,dr,forest \
  --bootstrap-reps 500 \
  --output-dir reports/baseline
```

Add `--models torch` after installing the optional neural-network dependency:

```bash
python -m pip install -e ".[ml]"
python scripts/run_baseline_analysis.py /path/to/criteo-uplift-v2.1.csv \
  --nrows 200000 --models torch --epochs 30
```

The generated report directory is intentionally ignored by Git. It contains
the balance table, per-model uplift curves, the random-targeting curve, and a
JSON summary that can be attached to a project write-up.

## Running the full 14M-row dataset

The local full dataset can be run directly:

```bash
DATA=/home/sid/.cache/kagglehub/datasets/arashnic/uplift-modeling/versions/1/criteo-uplift-v2.1.csv

python scripts/run_baseline_analysis.py "$DATA" \
  --models t,x,r,dr,forest \
  --n-estimators 25 \
  --bootstrap-reps 100 \
  --output-dir reports/criteo-full
```

This command uses all rows because `--nrows` is omitted. The reduced
bootstrap count and forest size are deliberate runtime controls for the
14M-row run; they do not change the estimators. Run it from a machine with
substantial free memory and expect the forest-based learners to take much
longer than the T-learner.

When the command finishes, inspect:

```text
reports/criteo-full/summary.json
reports/criteo-full/balance.csv
reports/criteo-full/random_curve.csv
reports/criteo-full/t_uplift_curve.csv
reports/criteo-full/x_uplift_curve.csv
reports/criteo-full/r_uplift_curve.csv
reports/criteo-full/dr_uplift_curve.csv
reports/criteo-full/forest_uplift_curve.csv
```

`summary.json` contains the full-data row count, treatment rate, conversion
ATE and bootstrap interval, each model's AUUC interval, conformal radius,
interval width, and empirical test coverage. The curve files can be plotted
against `fraction_targeted` and `incremental_gain` to produce the Qini-style
figures for the report.

The PyTorch model should be run as a separate experiment rather than alongside
the five scikit-learn models:

```bash
python -m pip install -e ".[ml]"
python scripts/run_baseline_analysis.py "$DATA" \
  --models torch --epochs 10 --bootstrap-reps 100 \
  --output-dir reports/criteo-full-torch
```

For a first full-data run, use the scikit-learn command above. If it is too
slow or memory-intensive, reduce `--n-estimators` further or run
`--models t,x,dr` first; the output format remains the same. Do not compare
metrics from different holdout definitions as if they were one experiment:
each invocation creates its own deterministic 75% training, 12.5% calibration,
and 12.5% test split using `--random-state 42`.

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
