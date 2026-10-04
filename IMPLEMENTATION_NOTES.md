# Implementation Notes

This document relates the implementation in `experiment-code/wluncert/` to the paper
(Sec. 3, Eqs. 1–3; Sec. 4; Sec. 5). It records how the models of the paper map to the code,
where the implemented priors differ from the equations, and which version of the evaluation
pipeline produced the reported results. File references are relative to
`experiment-code/wluncert/` and refer to the current `main` branch unless a commit is given.
Comments of the form `TODO(authors)` mark statements that depend on a decision or on information
of the authors.

## 1 Models and shared settings

### 1.1 Mapping between paper and code

| Paper | Label in `main.py` | Implementation (`models.py`) |
|---|---|---|
| HyPerf, Π̃^pp | `partial-pooling-mcmc-robust-adaptive-shrinkage` | `MCMCPartialRobustLassoAdaptiveShrinkage` |
| Π̃^np | `no-pooling-mcmc-1model` | `MCMCCombinedNoPooling` (one NumPyro model, no parameters shared across settings) |
| Π̃^cp | `cpooling-mcmc-1model` | `MCMCCombinedCompletePooling` |
| Π̂^np_Lasso | `model_lassocv_reg_no_pool` | `NoPoolingEnvModel(LassoGridSearchCV())` |
| Π̂^cp_Lasso | `model_lassocv_reg_cpool` | `CompletePoolingEnvModel(LassoGridSearchCV())` |

`LassoGridSearchCV` selects the regularization strength of scikit-learn's `Lasso` by 3-fold
grid-search cross-validation (negative MSE) over {0.001, 0.01, 0.1, 0.5, 1, 10}; with at most
three training samples, it uses `Lasso(alpha=0.5)` without cross-validation. The mapping of the
Lasso baselines is supported by the label mapping in `playground/insights-dashboard.py`
(`draw_multitask_paper_plot`) and by the exact reproduction of the per-seed Lasso values in
`supplementary-material/RQ1/RQ1-1/rq1-results.csv` (Section 3).

<!-- TODO(authors): Confirm that Π̂^np_Lasso and Π̂^cp_Lasso refer to the `model_lassocv_*`
variants and not to `model_lasso_reg_*` (fixed `Lasso(alpha=1.0)`), which `main.py` also runs. -->

### 1.2 Shared settings

- **Preprocessing** (`data.Standardizer`, fitted per setting on the training data): `StandardScaler`
  on the performance values and `MaxAbsScaler` on the option values. Predictions are transformed
  back to the original scale before evaluation.
- **MCMC** (`MCMCMultilevelPartial._fit`, `main.get_all_models`): NUTS with
  `target_accept_prob=0.9` and `init_to_median`; 3 chains, 1 000 warm-up and 1 000 retained
  samples (500/500 with `--debug`).
- **Random seeds**: per repetition `rnd`, the per-setting split seeds are drawn from
  `np.random.default_rng(rnd)` (`experiment.py`). The MCMC seed is derived from a SHA-256 hash of
  the (standardized) training performance vector (`NumPyroRegressor.array_to_seed`).

### 1.3 Code history

The definitions of the five models and of `LassoGridSearchCV` are identical, apart from
formatting, in the first code commit of this repository (`c3cdb57`, 2024-06-06) and in the
current `main`.

<!-- TODO(authors): Confirm from the history outside this repository that the definitions were
unchanged since February 2024, i.e., that the runs of 2024-05-29/30 (Section 3) used them. -->

## 2 Priors: equations vs. implementation

Notation used below:

- The paper states hyperpriors on variances (α_σ², β_σ², σ_s², σ_μ²); the implementation places
  them on standard deviations or scales.
- NumPyro's `Exponential(r)` is parameterized by the rate r; the code writes `Exponential(1/m)`,
  i.e., an exponential distribution with mean m, denoted Exp(mean m) below. For Exp(1), both
  readings coincide.
- L(μ, s) denotes a Laplace distribution with location μ and scale s (standard deviation √2·s).

| Model | Quantity (sample site) | Paper | Implementation | Remark |
|---|---|---|---|---|
| Π̃^pp | α_μ (`base-hyper`) | N(0, 1) (3a) | N(−0.5, 1) | Location shifted, see (a) |
| Π̃^pp | α_σ (`base-hyper_var`) | α_σ² ~ Exp(1) | α_σ ~ Exp(mean 1) | Used as Laplace scale |
| Π̃^pp | α_s (`base`) | N(α_μ, α_σ²) (3e) | L(α_μ, α_σ) | Laplace instead of normal |
| Π̃^pp | σ_μ (`error-hyper`) | σ_μ² ~ Exp(1) | σ_μ ~ Exp(mean 1) | Notation |
| Π̃^pp | σ_s (`error`) | σ_s² ~ Exp(σ_μ²) (3b) | σ_s = σ_μ·\|z\|, z ~ N(0, 1), i.e., HalfNormal(σ_μ) | Half-normal instead of exponential, see (b) |
| Π̃^pp | b (`influences-regularization-shrinkage`) | Exp(1) (3d) | Exp(mean 1) | As in the paper |
| Π̃^pp | β_μ (`influences-mean-hyperior`) | L(0, b) (3c) | L(0, b) | As in the paper |
| Π̃^pp | β_σ (`influences-stddevs-hyperior`) | β_σ² ~ Exp(1) | β_σ ~ Exp(mean 1) | Notation |
| Π̃^pp | β_s (`influences`) | N(β_μ, β_σ²) (3f) | N(β_μ, β_σ) | Notation |
| Π̃^np | α_s (`base`) | N(0, 1) (1a) | N(−1, 0.5) | Location and scale differ |
| Π̃^np | b | Exp(1) (1b) | none | Fixed scale, see (c) |
| Π̃^np | β_s (`influences`) | L(0, b) (1c) | L(0, 0.2/√2 ≈ 0.14) | Fixed scale, see (c) |
| Π̃^np | σ_s (`error`) | σ_s² ~ Exp(1) (1e) | σ_s ~ Exp(mean 1) | Notation |
| Π̃^cp | α (`base`) | N(0, 1) (2a) | N(−1, 1) | Location differs |
| Π̃^cp | b | Exp(1) (2b) | none | Fixed scale, see (c) |
| Π̃^cp | β (`influences`) | L(0, b) (2d) | L(0, 0.2/√2 ≈ 0.14) | Fixed scale, see (c) |
| Π̃^cp | σ (`error`) | σ² ~ Exp(1) (2e) | σ ~ Exp(mean 0.5) | Mean differs |

In all three models, the likelihood is a normal distribution parameterized by its standard
deviation (Eqs. 1f, 2f, 3h up to notation). `influences`, `base` and, for Π̃^pp,
`influences-mean-hyperior` are sampled with a non-centered parameterization
(`LocScaleReparam(0)`); this changes the parameterization used by the sampler, not the model.

**(a) Intercept hyperprior α_μ.** Performance is standardized per setting (mean 0), and all option
values are non-negative after scaling. The intercept α_s is therefore the predicted standardized
performance of the configuration with all option values equal to 0, which is typically below the
setting mean. On the evaluation data (153 settings; per setting, all configurations, preprocessing
as in Section 1.2), the intercepts of regularized linear models average approximately −0.5
(`Ridge` with penalty 1: −0.57; `LassoCV`: −0.51 to −0.53, depending on the multicollinearity
removal of Section 4); about three quarters of them are negative. Since the prior location
coincides with a statistic of the evaluation data, the prior should be regarded as informed by the
evaluation data.

<!-- TODO(authors): State in the paper that α_μ ~ N(−0.5, 1) is used and how the value was chosen. -->

**(b) Observation noise σ_s.** Both the half-normal prior and an exponential prior with mean σ_μ on
σ_s (the reading of Eq. 3b on the standard-deviation scale) are scale families in σ_μ. The
half-normal has a similar median (0.67·σ_μ vs. 0.69·σ_μ), a smaller mean (0.80·σ_μ vs. σ_μ) and
lighter tails.

**(c) Regularization of the baselines.** Π̃^np and Π̃^cp use a fixed Laplace scale of 0.2/√2 for the
option influences (prior standard deviation 0.2) and no hyperprior b, whereas Eqs. 1b/2b specify
b ~ Exp(1) (prior mean of the scale 1), and Sec. 3.3.1 states that the regularization strength is
inferred from the data. The implemented baselines thus use a stronger regularization that is not
adapted to the data. `models.py` also contains an adaptive variant
(`MCMCCombinedNoPoolingAdaptiveShrinkage`), which is not used by `main.py`.

<!-- TODO(authors): Decide whether to describe the implemented baseline priors in the paper and/or
to additionally report results for Π̃^np and Π̃^cp with priors as in Eqs. 1–2. -->

**Option scaling.** Sec. 3.1.2 describes min-max scaling to [0, 1]; the implementation uses
`MaxAbsScaler` (division by the maximum absolute value). Both coincide for binary options and for
numeric options with minimum 0. They differ for numeric options with a positive minimum: seven
numeric options of jump3r (e.g., `Highpass`, 20 000–30 000), `threads` of dconvert, and `dpi`,
`resolution`, `quality` and `indexed` of batik.

## 3 Evaluation protocol of the reported RQ1 results

Sec. 4 defines the test set as D_s^test = D_s \ ³D_s^train, i.e., all configurations of a setting
that are not part of the largest training set.

- Table 2, Figure 2 (cf. `rq1-results.pdf`) and `supplementary-material/RQ1/RQ1-1/rq1-results.csv`
  stem from two runs with the experiment ids `uncertainty-learning-2024-05-29_22-59-58-full-run`
  and `uncertainty-learning-2024-05-30_17-49-03-full-run` (30 seeds). The means over the 30 seeds
  in the CSV reproduce all pMAPE values of Table 2. The CSV additionally contains the relative
  training size 1.5, which is not reported in the paper.
- In the pipeline version of that time (contained in this repository at commit `6336e03`,
  `experiment.py`, `Replication.run`), the test set of each setting is the complete per-setting
  data set D_s (`test_list.append(env_data)`), which includes the training configurations.
- Evidence: with the data pipeline of commit `6336e03`, the per-seed values of both Lasso baselines
  in `rq1-results.csv` are reproduced with a relative deviation of at most 3·10⁻⁷ if the test set
  is D_s, whereas the test set D_s \ D_s^train yields deviations of up to 4.9 % (spot check: batik,
  lrzip, z3; seeds 0, 6, 12, 18, 24; relative training sizes 1/2, 1, 3; 90 runs).
- The current code implements the protocol of the paper: `SingleEnvData.get_split(...,
  max_train_samples_rel_opt_num=...)` (`data.py`) draws the largest training set, uses its
  complement as the test set (subsampled to at most `max_test_samples_abs` = 10 000 configurations
  per setting), and draws the smaller training sets as subsets of the largest one. In this
  repository, this was introduced with commit `15108f6` (2025-07-21).

The share of training configurations in the test set is at most 3·|O| / |D_s| per setting
(option counts as used in the 2024 runs, Section 4); it is smaller for smaller training sets.

| System | Settings | Configurations per setting | Options used | Largest training set | Max. share in test set |
|---|---|---|---|---|---|
| jump3r | 6 | 4 196 | 19 | 57 | 1.4 % |
| dconvert | 12 | 6 764 | 16 | 48 | 0.7 % |
| H2 | 8 | 1 954 | 16 | 48 | 2.5 % |
| batik | 11 | 1 919 | 10 | 30 | 1.6 % |
| xz | 12 | 1 999 | 31 | 93 | 4.7 % |
| lrzip | 13 | 191 | 11 | 33 | 17.3 % |
| x264 | 9 | 3 113 | 26 | 78 | 2.5 % |
| z3 | 12 | 1 010 | 13 | 39 | 3.9 % |
| VP9 | 35 | 302 | 16 | 48 | 15.9 % |
| x265 | 35 | 354 | 16 | 48 | 13.6 % |

<!-- TODO(authors): Decide whether to re-run RQ1 with the current protocol and whether an erratum
or a clarification of the evaluation protocol is needed. -->

## 4 Preprocessing: multicollinearity removal and option counts

The 2024 runs removed multicollinearity with `pycosa.util.remove_multicollinearity` (`data.py` at
commit `6336e03`). The current code uses `utils.remove_multicollinearity` (introduced with commit
`15108f6`), which drops constant columns and detects groups of mutually exclusive options on a
sample of 100 rows. The current routine keeps additional options for jump3r (`DualChannels`,
`NoReplayGain`), dconvert (`android`, `ceil`) and batik (`jpg`). This changes |O| and therefore the
training-set sizes α·|O| and the size of the largest training set.

| System | \|O\| in Table 1 | \|O\| used in the 2024 runs | \|O\| in the current code |
|---|---|---|---|
| jump3r | 16 | 19 | 21 |
| dconvert | 18 | 16 | 18 |
| H2 | 16 | 16 | 16 |
| batik | 10 | 10 | 11 |
| xz | 33 | 31 | 31 |
| lrzip | 11 | 11 | 11 |
| x264 | 25 | 26 | 26 |
| z3 | 12 | 13 | 13 |
| VP9 | 24 | 16 | 16 |
| x265 | 26 | 16 | 16 |

The counts used in the 2024 runs follow from `params.train_size` in `rq1-results.csv` and are
reproduced by loading the data with `pycosa.util.remove_multicollinearity`. Further differences
between Table 1 and the data: |C*| is 191 for lrzip and 1 010 for z3 (Table 1: 190 and 1 011); for
xz, the data contain 13 workloads, of which `DataAdapterXZ` excludes `artificl.tar`, so 12 settings
are used (Table 1: 13).

**Measured memory as input feature.** For xz, lrzip, x264 and z3, `measurements.csv` contains the
column `max-resident-set-size` in addition to `time`. The data adapters (`DataAdapterXZ`,
`DataAdapterX264` and subclasses) exclude memory-related columns from the list of performance
metrics but do not drop them, and `SingleEnvData.get_feature_names` treats every remaining column
as an option. Consequently, `max-resident-set-size` is an input of all models for these four
systems, in the 2024 runs and in the current code. It is included in the option counts above, and
the supplementary material for RQ2 and RQ3 contains plots for it (e.g.,
`supplementary-material/RQ3/z3/representation-matrices/representation_matrix_max-resident-set-size.pdf`).
The exact reproduction of Section 3 requires this column; without it (which also reduces |O| by
one), the reproduced per-seed values of Π̂^np_Lasso deviate by up to 14 % (lrzip) and 132 % (z3).

<!-- TODO(authors): Explain how |O|, |C*| and |S*| in Table 1 were counted, or align them with the
data used. Confirm whether `max-resident-set-size` is intended as an input; if not, assess the
effect on the results for xz, lrzip, x264 and z3. -->

## 5 RQ1.3 (TuxKConfig)

Sec. 5.3 states a test set of 10 000 configurations, 500 warm-up and 500 retained MCMC samples, and
α = 0.01 (about 120 samples per version). The run in `supplementary-material/RQ1/RQ1-3`
(`uncertainty-learning-2025-07-16_00-35-35-debug-1modelvs partial`) used 123 training
configurations per version (0.01 × 12 368 options). Its per-version calibration values
(`rel_cal_diff`, the nominal HDI level minus the empirical coverage) are integer multiples of
1/8 000, and the overall value is a multiple of 1/56 000. This is consistent with 8 000 test
configurations per kernel version (56 000 for the seven versions), not with 10 000. The current
`main.py` passes `max_test_samples_abs=10000`, which is applied per setting. The recorded fitting
time (17 842 s, about 5 h) and memory (`max_memory_mb` ≈ 70 630) agree with Sec. 5.3.2; the recorded
prediction time of 108 s refers to all 56 000 test configurations. As stated in the paper,
multicollinearity removal is skipped for this data set (`utils.remove_multicollinearity` returns
data with more than 100 columns unchanged).

<!-- TODO(authors): Confirm the number of test configurations and the MCMC settings of this run
(the run label contains "debug"; in the current `main.py`, `--debug` sets 500 warm-up and 500
retained samples, which matches the text). -->

## 6 Known issue

`modelinsights.py`, `simulate_hyperior_expectation`: the samples of the base hyper-distribution
(used by `get_base_hyperior_and_influences` for `plot_base` and `plot_single_hyperiors`) are drawn
from N(α_μ, α_σ), whereas the model uses L(α_μ, α_σ), whose standard deviation is √2·α_σ. The
plotted spread of the base hyper-distribution is therefore too narrow by a factor of √2. The option
hyper-distributions (normal in the model) are not affected, nor are model fitting, predictions and
metrics. The files `hyperiors-v1-base.pdf` and `hyperiors-v1-y-labels-base.pdf` in
`supplementary-material/RQ2/<system>/hyperpriors-vs-specific/` carry the names produced by
`plot_single_hyperiors`.

<!-- TODO(authors): Regenerate the affected plots after a fix, if they are used in the paper or the
supplementary material. -->

## 7 Reproduction hints

- The paper uses 30 repetitions (`--reps 30`); the Docker example in the README uses 5.
- With `--jobs`, the tasks must run in separate processes (process-based backend); see the fix for
  issue #1.
- With `--training-set-size x`, the largest training set has size x·|O|, so the test set is D_s minus
  that set rather than D_s \ ³D_s^train. Results comparable to the protocol of the paper require
  the full sweep of training-set sizes.
- Results of the current code are not expected to match Table 2 exactly, because the test set
  (Section 3) and the option counts (Section 4) differ from those of the 2024 runs.
