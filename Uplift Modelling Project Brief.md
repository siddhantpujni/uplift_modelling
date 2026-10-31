# Uplift Modelling on a Randomised Experiment: Project Brief

Sid | October 2026 |

## Summary

Estimate who actually changes behaviour because of a treatment, rather than who would have converted anyway, using a large public randomised ad experiment. The project combines real ML (boosting, forests, a neural net) with statistical depth (identification, cross-fitting, honest uncertainty, off-policy evaluation). Because the data comes from an RCT, the true average effect is known, which allows a controlled test of whether causal methods recover the truth when the sample is deliberately biased.

## Objectives

1. Build and compare a range of heterogeneous treatment effect (CATE) estimators on a 14M-row experiment.
2. Quantify uncertainty properly at every level: ATE, subgroup effects, individual effects and model rankings.
3. Test whether observational-style estimators (IPW, double ML) recover the RCT effect when treatment assignment is deliberately confounded, and how sensitive conclusions are to unobserved confounding.
4. Turn the best model into a budget-constrained targeting policy and evaluate it off-policy.
5. Produce a clean, reproducible repo and write-up with concrete numbers for the CV.

## Data

Criteo uplift dataset: about 14M users from randomised incrementality tests, 12 anonymised features, a treatment flag and two binary outcomes (visit and conversion). Roughly 85% of users are treated. It can be loaded with scikit-uplift, which also offers a 10% subset for fast iteration.

## Methodology

**1. Experiment validity.** Covariate balance (standardised mean differences), a classifier two-sample check that covariates cannot predict treatment, ATE with confidence intervals, regression adjustment for variance reduction, and a power / minimum detectable effect calculation.

**2. CATE models.** T-, X- and R-learners with gradient boosting; double ML with cross-fitting; a causal forest; and a PyTorch network with a shared trunk and treatment-specific outcome heads.

**3. Evaluation.** Qini curves and AUUC with bootstrap confidence intervals, predicted-vs-observed uplift calibration by decile, and nested cross-validation for tuning with a untouched final holdout.

**4. Inference.** Bootstrap or jackknife intervals for ATE and subgroup effects, conformal intervals for individual effects (weighted conformal in the style of Lei and Candes), and Benjamini-Hochberg control on any subgroup claims.

**5. Biased-sample test.** Subsample the data so treatment probability depends on covariates, then compare naive difference in means, regression adjustment, IPW with estimated propensities and double ML against the full-RCT effect. Simulate unobserved confounding by hiding a covariate used in the biasing mechanism, and compare the observed bias with what a sensitivity analysis (such as E-values or omitted-variable-bias bounds) predicts.

**6. Policy.** Choose who to treat under a budget (top k% by predicted uplift), compare against targeting by response probability and random targeting, and evaluate with doubly robust off-policy estimates on held-out data.

## Success Criteria

- Best CATE model beats random and simple T-learner baselines on AUUC, with bootstrap intervals reported.
- Conformal intervals reach empirical coverage close to nominal (report the gap).
- The biased-sample experiment reports bias for every estimator against the RCT truth, including cases where methods fail.
- Policy evaluation reports incremental visits captured at several budget levels.