# Architecture

## Data flow (one direction only)

```
 ML_edge_vacancy_seed2026/*.csv, NEGF_all.mat     (raw NEGF output, read-only)
                │
                ▼
   config.py ─► data.py       load + validate, curve matrix, vacancy layout
                │
                ▼
            targets.py        log targets, Ion/Ioff variants, swing, power      [Stage 2]
                │
                ▼
            splits.py         3 protocols + blocked CV with embargo            [Stage 3]
                │
                ▼
            models.py         common fit/predict: baselines, ridge, GBM, GP, MLP [4-5]
            unsupervised.py   PCA + clustering                                  [6]
            evaluate.py       MAE / RMSE / R2 / skill, error maps               [4]
                │
                ▼
      run_experiments.py ──► results/*.csv ──► make_figures.py ──► figures/*.png
```

## Modules

**config.py** reads `python/config.yaml`, the single home of every constant (paths, expected row counts, physical constants, grids, seeds, CPU settings). Code receives the config dict as an argument, so there is no hidden global state.

**data.py** is the only module that opens raw files. It reads the four CSVs and checks columns, row counts, NaNs, `reliable == 1`, the seed, 20 points per curve on one shared gate grid, IV and metrics agreement, and train/test disjointness. If any check fails it raises an error instead of returning data. It also reshapes gate-point rows into an `(n_curves, 20)` matrix (`curve_matrix`) and reads the nested vacancy layout from `NEGF_all.mat` (`load_layout`).

## Glossary

- **Ion**: drain current with the gates "on", here at V_TG − V_BG = 0 (the topological edge channel conducts).
- **Ioff**: drain current with the gates "off". In the supplied data this is the current at the **last** gate point (1.9 V), not the smallest current along the sweep. See DECISIONS.md, Stage 0.
- **Ion/Ioff**: on/off ratio, a measure of how well the transistor switches. We work with log10 of it, in decades.
- **Swing (subthreshold swing)**: the gate voltage needed to reduce the current tenfold, in mV/decade. Smaller means a sharper switch.
- **Nested layout**: each vacancy level keeps all atoms removed at the level below and removes 1–2 more. Neighbouring levels are therefore almost the same device, not independent samples.
- **Embargo**: in cross-validation, training levels directly next to the held-out block are dropped too. Because of the nesting they are near-copies of the held-out devices.
- **Skill score**: 1 − MSE_model / MSE_best_naive. 0 means no better than the best naive baseline, and 1 means perfect.

## Modules added in Stage 2

**targets.py** turns the validated tables into one row per curve with every learning target in log10: Ion/Ioff (end-point and minimum versions), static power, switching energy, delay and dynamic power (DED/τ_d, one convention). It also adds the subthreshold swing. `falling_branch` cuts each sweep at its minimum, `steepest_swing` fits a shape-preserving PCHIP curve and takes the steepest slope, and `raw_grid_swing` gives the coarser upper bound. Per-gate-point log10 I is taken unchanged from the IV table.

**run_experiments.py** is the command-line entry point. Each named step writes `results/*.csv` and records its run time in `results/timings_last_run.csv`.

**make_figures.py** writes `figures/*.png`. Stage 2 adds `swing_check.png`, a 3 × 4 grid of sample curves showing the falling branch, the PCHIP fit and the steepest tangent.

## Technique: PCHIP interpolation (supporting tool, not a course model)
**Idea.** PCHIP (piecewise cubic Hermite interpolating polynomial) passes a cubic through each pair of neighbouring points. It chooses the slopes at the points so that the curve never overshoots, meaning it stays monotone wherever the data are monotone. We need a slope between grid points, and an ordinary spline would invent wiggles.
**Equation (interior node slope, weighted harmonic mean of neighbouring secants δ):**
$$ \frac{w_1+w_2}{d_k} = \frac{w_1}{\delta_{k-1}} + \frac{w_2}{\delta_k}, \quad d_k = 0 \text{ if } \delta_{k-1}\delta_k \le 0 $$
**Why here.** Currents fall by up to 5 decades between neighbouring points, and overshoot would create fake steep slopes.
**Failure mode here.** When the drop happens in one step, the steepest slope is decided by the interpolant's shape inside that step, not by data. PCHIP then reports a swing up to about 1.75× lower than the raw grid (see DECISIONS, Stage 2).

## Modules added in Stage 3

**splits.py** defines every way the 798 curves are divided. A `Split` holds two key tables (train, test), and `Split.apply(table)` selects the rows of any table by curve key, so IV rows and curve targets are split identically. `all_protocols` returns the four final splits: supplied, temperature forward-chaining (≤ 60 K → 62–78 K), vacancy block (hold out 0.8–1.2 %) and vacancy extrapolation (≤ 1.5 % → 1.6–2.0 %). `blocked_cv_folds` produces the hyperparameter-tuning folds inside one training pool. `run_experiments.py splits` writes `results/split_summary.csv`, which lists every split and fold with its sizes and ranges.

## Technique: blocked cross-validation with embargo (supervised learning: model selection)
**Idea.** Ordinary k-fold CV assumes samples are independent. Ours are not: neighbouring vacancy levels share all but 1–2 removed atoms, so a random fold would validate on near-copies of training devices and pick over-flexible models. We therefore hold out a whole block of consecutive vacancy levels. We also drop the levels next to it from training (the embargo), because they are the closest copies.
**Equation (CV estimate of the error of hyperparameter θ over K blocks B_k, embargo set E_k):**
$$ \widehat{\mathrm{Err}}(\theta) = \frac{1}{K}\sum_{k=1}^{K} \frac{1}{|B_k|}\sum_{i \in B_k} \big(y_i - \hat f_\theta^{(-B_k \cup E_k)}(x_i)\big)^2 $$
**Why here.** The final tests ask the model to predict whole unseen vacancy ranges or temperatures. Blocked CV makes the tuning task resemble that.
**Failure mode here.** The layouts are nested, so every level still contains all atoms of the levels below it. The embargo reduces this leakage but cannot remove it. With only 5 blocks, the CV estimate is also noisy.

## Modules added in Stage 4

**models.py** gives every model the same `fit(X, y)` / `predict(X)` interface. X is [T_K, vacancy] for curve-level targets and [T_K, vacancy, gate_V] for log10 I. `build_model(name, cfg, pointwise)` returns an unfitted estimator and its hyperparameter grid. `Gatewise` turns any curve-level model into an I–V model by fitting one copy per gate voltage. `KernelSelectGP` fits each candidate GP kernel and keeps the best by marginal likelihood. `tune_and_fit` runs blocked-CV grid search (only for models that have a grid) and refits on the whole training pool. A future PINN only needs fit/predict (`Regressor` protocol).

**evaluate.py** computes MAE, RMSE, MSE, R² and the skill score. Targets are already in log10 (or mV/dec), so the errors are in decades directly.

**run_experiments.py classical** runs every (protocol, vacancy encoding, target, model) job. It caches each one under `results/cache/classical_<config hash>/`, so a run resumes after interruption and a config change triggers a recompute. It writes `results/predictions_classical.csv.gz` (every test prediction, with GP std) and `results/metrics_classical.csv` (one row per job, with skill against the best naive baseline of the same task).

## Technique: naive baselines (reference point for the skill score)
**Idea.** A model is only useful if it beats a trivial guess. *Per-vacancy mean*: predict the average of the training curves at the same, or nearest, vacancy level, ignoring temperature. *Nearest neighbour*: copy the training curve closest in standardised (T, vacancy).
**Equation.** $\hat y(x) = y_{j^*},\ j^* = \arg\min_j \lVert z(x) - z(x_j)\rVert_2$, with $z$ = standardised inputs.
**Why here.** At a fixed layout the curves change slowly with T, so 1-NN is already a strong baseline for interpolation.
**Failure mode.** For an unseen vacancy level both baselines copy the nearest level, and with nested, non-monotone layouts that can be badly wrong.

## Technique: ridge and polynomial ridge (supervised learning: regression, regularisation)
**Idea.** A linear model in the inputs, or in all their products up to degree d, fitted by least squares with a penalty on large coefficients. The penalty α trades bias for variance and is chosen by blocked CV.
**Equation.** $\hat\beta = \arg\min_\beta \lVert y - \Phi\beta\rVert^2 + \alpha\lVert\beta\rVert^2 = (\Phi^\top\Phi + \alpha I)^{-1}\Phi^\top y$
**Why here.** It is the simplest smooth model of three inputs and makes a transparent reference.
**Failure mode.** The I–V curves have sharp, step-like drops (up to 5 decades per 0.1 V), and a low-degree polynomial cannot bend that fast. High degrees oscillate between grid points.

## Technique: gradient boosting (supervised learning: ensembles)
**Idea.** Add many shallow regression trees one after another, each fitted to the remaining errors of the sum so far, scaled by a learning rate. Tree depth and the number of trees are tuned by blocked CV. We use sklearn's histogram-based version.
**Equation.** $F_m(x) = F_{m-1}(x) + \nu\, h_m(x),\ h_m \approx \arg\min_h \sum_i \big(r_i^{(m)} - h(x_i)\big)^2,\ r_i^{(m)} = y_i - F_{m-1}(x_i)$
**Why here.** Trees handle steps and thresholds naturally, which suits sudden switching.
**Failure mode.** Trees predict piecewise-constant values and cannot extrapolate. Above 60 K, or beyond 1.5 % vacancy, they repeat the edge of the training data.

## Technique: Gaussian-process regression (supervised learning: kernel methods, uncertainty)
**Idea.** A prior over smooth functions, defined by a kernel that says how strongly two inputs are correlated. Conditioning on the training data gives a predictive mean and a standard deviation at every new input. The kernel's length scales (one per input) and noise level are fitted by maximising the marginal likelihood, and the same criterion picks RBF vs. Matérn-5/2.
**Equation.** $\mu(x_*) = k_*^\top (K + \sigma_n^2 I)^{-1} y,\quad \sigma^2(x_*) = k(x_*,x_*) - k_*^\top (K+\sigma_n^2 I)^{-1} k_*$
**Why here.** The data are small (≤ 638 curves per fit) and smooth in T, and an error bar is valuable. For I–V curves we fit one GP per gate voltage, which keeps every kernel matrix at most 638 × 638. Fitting on all 12,760 rows would be about 1.3 GB and slow.
**Failure mode.** With the default bounds the vacancy length scale collapsed to 10⁻⁵, a worse local optimum that treats every vacancy level as unrelated. We bound it below at 0.05 standardised units. Even then, for an unseen layout the GP reverts towards its mean with wide error bars, because neighbouring levels genuinely do not predict each other well.

## Modules added in Stage 5

**neural.py** implements a small PyTorch MLP (`MLPRegressor`) with the same sklearn interface, so it goes through the same blocked-CV grid search and evaluation as every other model. It standardises X and y internally, trains with AdamW and MSE in mini-batches, and stops early on a random 15 % of the training **curves**. Whole curves are held out, never single gate points, so early stopping does not peek at gate points of curves it trains on. It runs on CPU with one thread per fit, and several fits run in parallel. `run_experiments.py mlp` tunes width, depth and weight decay by blocked CV with the first seed, refits with 5 seeds, and writes `results/metrics_mlp.csv` (per seed) and `results/mlp_vs_classical.csv` (mean ± std over seeds next to the best classical and best naive model).

## Technique: multi-layer perceptron (neural networks)
**Idea.** A stack of linear layers with a smooth non-linearity (SiLU) between them. It learns any smooth function of (T, vacancy, gate) by gradient descent on the squared error. Width and depth set its capacity, weight decay penalises large weights, and early stopping halts training when the error on held-out curves stops falling.
**Equation (one hidden layer shown; we use 2–3).** $\hat y(x) = w_2^\top\, \sigma(W_1 x + b_1) + b_2,\quad \min_\theta \tfrac1n\sum_i (y_i - \hat y(x_i))^2 + \lambda\lVert\theta\rVert^2$
**Why here.** One network can represent a whole I–V curve, including the sharp drop, without fixing its shape in advance. For log10 I it maps (T, vacancy) to all 20 gate points at once (638 rows, not 12,760), which keeps training cheap on a CPU.
**Failure mode here.** Only 798 curves from one nested layout: the network can memorise layout-specific bumps it cannot generalise. Results also depend on the random initialisation, so we report the mean ± std over 5 seeds. With dropout 0 and noiseless data, early stopping and weight decay are the only regularisers.

## Modules added in Stage 6

**unsupervised.py** fits a PCA of the 798 log10 I–V curves (`fit_pca`, `explained_variance_table`, `component_table`) and clusters the curves in the space of the leading PCA scores (`cluster`, `cluster_scores`). It compares the clusters with a physics label: three log10 Ion/Ioff classes from `ion_ioff_classes`. It is descriptive only. It is fitted on all 798 curves, never feeds a prediction, and so does not touch the train/test protocols. `run_experiments.py unsupervised` writes `results/pca_explained_variance.csv`, `pca_components.csv`, `cluster_scores.csv` and `cluster_assignments.csv`. `make_figures.py` draws `figures/pca_clusters.png` from them.

## Technique: principal component analysis (unsupervised learning: dimensionality reduction)
**Idea.** Each I–V curve is a point in a 20-dimensional space (one axis per gate voltage). PCA finds the directions along which the curves differ most. The first component is the single curve shape that, added to the mean curve in varying amounts, explains the most variation, and so on. If a few components explain almost everything, the curves really have only a few degrees of freedom.
**Equation.** Centred data $\tilde Y = Y - \bar y$, with $\tilde Y = U S V^\top$ (SVD). Components are the rows of $V^\top$, scores $Z = \tilde Y V$, and explained-variance share $s_j^2 / \sum_i s_i^2$.
**Why here.** All 20 columns share one unit (decades), so no rescaling is needed, and the components can be read physically as curve shapes over gate voltage.
**Failure mode here.** PCA is linear. A sharp drop that moves along the gate axis from device to device needs several components to describe, so "PCs needed" overstates how complex the physics is. It is also fitted on all curves, so it describes this dataset rather than predicting new devices.

## Technique: k-means and Gaussian-mixture clustering (unsupervised learning: clustering)
**Idea.** *k-means* puts k centres in the PCA-score space and assigns each curve to the nearest one, moving each centre to the mean of its curves until nothing changes. A *Gaussian mixture* models the points as k overlapping Gaussian clouds, each with its own shape (full covariance), fitted by expectation–maximisation. We judge the result by the *silhouette* (how much closer each curve is to its own cluster than to the next) and by the *adjusted Rand index* against the Ion/Ioff classes.
**Equation (k-means objective; silhouette of point i).** $\min_{\{C_j\},\{\mu_j\}} \sum_j \sum_{i \in C_j} \lVert z_i - \mu_j \rVert^2, \qquad s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ ($a_i$: mean distance to its own cluster, $b_i$: to the nearest other cluster).
**Why here.** It asks whether the devices fall into natural groups (working, degraded, failed) without being told the labels.
**Failure mode here.** The data are not blobs. Each vacancy level traces a thin *strand* through PCA space as temperature changes, so the clusters cut the set of strands into bands of vacancy levels. The silhouette is then middling and nearly flat in k, and BIC keeps falling as k grows (noiseless data, no natural k).

## Modules added in Stage 7

**robustness.py** tests how a fitted model's error grows when its test inputs are perturbed. `input_steps` gives each input's grid step (2 K, 0.1 %, 0.1 V), `noisy` adds Gaussian noise of a chosen fraction of that step, `noise_curve` scores one model at every noise level and repeat, and `summarise` averages over repeats. `run_experiments.py robustness` writes `results/robustness_noise.csv` and `results/robustness_summary.csv`.

**plots.py** holds the plotting primitives for the report figures (`heatmap`, `predicted_vs_true`, `error_profile`, `iv_with_bands`, `robustness_lines`). Each draws on a given axis from tables read from `results/*.csv`, so figures cannot compute numbers of their own. **evaluate.comparison_table** builds `results/model_comparison.csv`: one row per protocol × target × model with MAE, RMSE, R², skill, MLP seed spread and rank.

**splits.py (changed):** each `Split` now carries a `cv_axis`. The 62–78 K protocol is tuned on blocks of consecutive temperatures, the others on vacancy blocks. `Split.fingerprint()` hashes the exact train/test curves for cache keys. Final vacancy tests now drop 1 neighbouring level from training.

## Technique: input-noise robustness (supervised learning: evaluation)
**Idea.** A surrogate should not change its answer wildly when its inputs are slightly wrong. We fit each model once on clean data, then predict the test set again with Gaussian noise added to temperature and vacancy, and watch the test error as the noise grows. Noise is measured in grid steps of each input, so "¼ step" means 0.5 K and 0.025 % vacancy.
**Equation.** $\mathrm{MAE}(\sigma) = \mathbb{E}_{\varepsilon}\big[\tfrac1n\sum_i |\hat f(x_i + \varepsilon_i) - y_i|\big],\ \varepsilon_{ij} \sim \mathcal N(0, (\sigma\,\Delta_j)^2)$, with $\Delta_j$ the grid step of input j.
**Why here.** It separates two very different reasons for a low test error: smooth physics the model has learned, or an exact look-up of the right device in a dense grid.
**Failure mode here.** Vacancy is really a *label* for one specific layout, not a continuous measurement. Noise on it partly tests "what if we picked the neighbouring layout". The test is informative about the surrogate's behaviour but is not a model of real measurement error.

## Modules added in Stage 8

**uncertainty.py** computes confidence intervals for the reported test errors from the saved predictions. `level_sums` aggregates |error| per vacancy level, `resample_levels` and `bootstrap_mae` give the cluster bootstrap, `task_bootstrap` adds paired differences against the reference model with plain and Bonferroni intervals, and `gp_coverage` measures how often the truth falls inside the GP's 68 % / 95 % intervals. `run_experiments.py uncertainty` writes `results/bootstrap_ci.csv` and `results/gp_coverage.csv` in about a second.

**ridge_numpy.py** is ridge regression written out in NumPy (`fit_ridge`, `effective_dof`, `coefficient_path`). It is a teaching companion to the sklearn model the pipeline uses and is checked against it to 1e-8.

**run_experiments.py epoch_sensitivity** re-runs selected MLP jobs with a higher epoch limit in a separate cache folder and writes `results/epoch_sensitivity.csv`. **.github/workflows/tests.yml** runs the unit tests on GitHub (tests only, read-only).

## Technique: cluster bootstrap (statistics for model evaluation)
**Idea.** To see how much a test error would change with a different but similar test set, we redraw the test set many times with replacement and recompute the error. The units we redraw must be independent. Curves of one vacancy level share one defect layout and are near-copies, so the unit is a whole vacancy level with all its curves, not a single curve.
**Equation.** For a resample $b$ of levels $\ell^{(b)}_1..\ell^{(b)}_L$: $\mathrm{MAE}^{(b)} = \sum_k S_{\ell_k^{(b)}} / \sum_k N_{\ell_k^{(b)}}$, with $S_\ell$ the summed |error| and $N_\ell$ the number of test rows at level ℓ. The 95 % interval is the 2.5th–97.5th percentile over $b$.
**Why here.** Curve-level resampling would treat 190 dependent curves as 190 independent samples and report intervals about 5× too narrow (`tests/test_uncertainty.py`).
**Failure mode here.** The vacancy tests have only 5 levels, so only 126 distinct resamples exist. The intervals are coarse and wide, and the bootstrap cannot estimate the uncertainty from the *one* layout we have.

## Technique: predictive-interval coverage (supervised learning: GP uncertainty)
**Idea.** A GP gives each prediction a standard deviation σ. If it is calibrated, the truth should fall inside ±1σ about 68 % of the time and inside ±1.96σ about 95 % of the time. We count how often it does on each test set.
**Equation.** $\widehat{\mathrm{cov}}_z = \frac1n \sum_i \mathbb 1\big[|y_i - \mu_i| \le z\,\sigma_i\big]$, with z = 1 (68.27 %) or 1.96 (95 %).
**Why here.** The proposal promised uncertainty estimates; coverage tests whether they mean anything.
**Failure mode here.** Calibration holds only where the GP has seen the layout. Beyond the trained vacancy range the coverage collapses (0–65 % at the nominal 95 %), because the GP's σ does not grow fast enough for a genuinely new layout.
