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
