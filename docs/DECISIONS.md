# Decision log

Each entry: date, stage, decision, alternatives rejected, why.

## 2026-11-17 — Stage 0 (data profile and layout proposal)

1. **Curve key = (T_K, vac_pct).** Every (T, vacancy) pair has exactly 20 gate points, and no pair appears in both train and test. We use vac_pct (the nominal level) as the curve label and keep vac_pct_actual / n_vac_atoms as alternative inputs.
   *Rejected:* keying on vac_pct_actual, because floating-point equality is fragile.
2. **Gate input = VTG_minus_VBG_V.** lambda_eV and Ez_V_per_A are exact linear rescalings of it (lambda/V = 0.78923, Ez/V = 0.188187), so including them adds no information and would give a design matrix with three collinear columns.
3. **Ioff definition (OPEN, needs your decision).** In the supplied metrics, Ioff_A is the current at the LAST gate point (V = 1.9 V), not the minimum current. 761 of 798 curves rise again after their minimum. Options are listed in the Stage 0 review.
4. **Dynamic-power convention.** SPD_W = 0.05 × Ioff and DED_J = 0.0025 × C_eff. This implies V_DS = 0.05 V and DED = C·V_DS². Also tau_d = C_eff × 0.05 / Ion. One config constant will define P_dyn = DED_J / tau_d_s.
5. **Python environment.** We will create a project venv with the CPU-only torch wheel. Python 3.13 and torch 2.10+cpu are already installed system-wide, which confirms the wheels exist for this machine.

## 2026-11-17 — Stage 1 (data loader)

You said "go" without answering the Stage 0 questions, so I applied the recommended defaults. Change any of them and I will update the code.
1. **Ioff:** both definitions are kept. The primary is `last` (the supplied value, I at 1.9 V) and the secondary is `min` (config `targets.ioff_primary/secondary`). The code for them comes in Stage 2.
2. **Swing branch:** it ends at the curve minimum, before the upturn (Stage 2).
3. **Layout:** approved as proposed, with a `python/wte2ml/` package. pytest finds it through `pythonpath` in `pyproject.toml`, so no install step is needed.
4. **Profiling scripts:** not moved yet. The loader tests now cover the same facts.

Stage 1 choices:
- The loader **fails loudly** (ValueError) on any deviation from the Stage 0 facts. *Rejected:* warnings, because they are easy to miss and let a silently changed file reach the results.
- The expected row counts and seed live in config.yaml rather than in the code.
- `vac_atoms` keeps MATLAB's 1-based indices and is never used to index a Python array. *Rejected:* converting to 0-based, because that invites off-by-one errors when comparing against the MATLAB code.
- Train/test is a `split` column on each table, so pooled protocols are a concat with no bookkeeping.

## 2026-11-17 — Stage 2 (targets and swing)

- **Swing method:** PCHIP on the falling branch (index 0 to the curve minimum), taking the steepest slope on a 2001-point grid, as you specified. The raw-grid value is saved next to it as `swing_raw_mV_per_dec`. *Rejected:* a cubic spline, which overshoots between points and could invent slopes steeper than the data. *Rejected:* a fitted straight line over a decade window, because many branches cover a decade in a single grid step.
- **Swing status: SUPPORTING result, not a main target.** The swing is set by single grid steps. On the falling branch the median steepest step drops 2.04 decades, and 441 of 798 curves have one step steeper than 2 decades (scratch check, re-derivable from `results/curve_targets.csv` together with the IV CSVs). The PCHIP swing is lower than the raw-grid swing for every curve (enforced by a test). The ratio has median 0.80 and range 0.57–0.98, so the reported swing depends on the interpolation choice by up to a factor of about 1.75.
- **Boltzmann comparison:** none of the 798 curves is below kT ln10/q, and the clean device's swing hardly depends on T (34.8, 32.3, 37.7 mV/dec at 4, 40, 78 K, from `results/curve_targets.csv`). The switching is therefore not thermionic, and kT ln10/q is a reference line, not a bound this device is fighting.
- **Ion/Ioff:** both definitions are in `curve_targets.csv`. `log10_Ion_Ioff` is the supplied end-point value (primary) and `log10_Ion_Ioff_min` uses the curve minimum.
- **Dynamic power = DED / tau_d** (config `physics.dynamic_power`), the only convention allowed in the code. Note: log10_P_dyn spans only −6.714 to −6.647 over all 798 curves (`results/curve_targets.csv`), so it is nearly constant and a weak ML target. The same is true of log10 DED and log10 tau_d, which span less than 0.5 decades.
- **Results flow:** `run_experiments.py targets` writes `results/curve_targets.csv` and `results/swing_by_vacancy.csv`. The figure script reads the data package directly only for the raw-curve check plot.

## 2026-11-17 — Stage 3 (splits and blocked CV)

Stage 2 open questions were unanswered, so the defaults apply: end-point Ion/Ioff is primary, and P_dyn, DED and tau_d are kept as targets, flagged as nearly constant.
- **Splits are defined on curves, never on rows.** A split is two sets of (T_K, vacancy level) keys, and rows are assigned by key. A test applies every split to all 15,960 IV rows and checks that each curve's 20 points stay on one side. *Rejected:* random row splits, which would put gate points of the same sweep on both sides.
- **Vacancy levels as integers** (round(vac_pct / 0.1)). This avoids float bugs such as 0.30000000000000004 ≠ 0.3, and a test covers it.
- **Protocol 3a block = 0.8–1.2 %** (5 levels × 38 T = 190 test curves). **Protocol 3b**: train ≤ 1.5 %, test 1.6–2.0 % (190 test curves). Counts are in `results/split_summary.csv`.
- **No embargo around the FINAL test blocks** (`holdout_embargo_levels: 0`). You asked for the embargo in CV. Adding one to the final hold-outs would also remove 0.7 and 1.3 % (76 curves) from training. It is one config value if you want it.
- **CV blocks:** the training pool's distinct levels are split into 5 consecutive blocks with `np.array_split` (sizes 5/4/4/4/4 for 21 levels), and an embargo of 1 level each side applies. The embargo is measured in grid steps, so it stays correct when the pool has a gap. Each fold loses about 25–30 % of its training curves (`results/split_summary.csv`, e.g. supplied fold 0 trains on 449 of the 638 pool curves).
- **Known quirk:** in the 3a pool (levels 0–0.7 and 1.3–2.0) one CV block is {0.7, 1.3, 1.4}. It is consecutive within the pool but straddles the held-out gap. Left as is, because it validates on a jump similar to the one the final test asks for. *Alternative:* never let a block cross a gap.
- **CV for protocol 2 is still over vacancy blocks,** as you specified. So hyperparameters for the temperature-extrapolation test are not tuned on a temperature-extrapolation objective. *Alternative:* forward-chaining CV over T inside 4–60 K.
- **Verified by mutation:** with the embargo disabled, `test_cv_uses_only_training_pool_and_respects_embargo` fails for all 4 protocols.

## 2026-11-17 — Stage 4 (classical models)

Stage 3 questions were unanswered, so the defaults apply: no final-split embargo, and vacancy-block CV for every protocol.
- **GP kernel chosen by log marginal likelihood, not CV** *(change from the brief, which listed "GP kernel" among the CV-tuned hyperparameters)*. One GP fit on 638 points takes 3–6 s. CV over 2 kernels × 5 folds × 20 gates would make each I–V job take about half an hour. The marginal likelihood is the GP's own principled criterion. *Rejected:* fewer gates, or subsampled data.
- **GP length scales bounded to [0.05, 100]** (standardised units). With the default bounds the vacancy length scale collapsed to 1e-5, and the log marginal likelihood was lower: 2175 vs. 2272 on the training curves at gate 1.4 V, 0.05 bound (scratch check). The bound improves the GP's own objective.
- **CV only for models with a grid** (ridge, polynomial ridge, boosting). The naive baselines and the GP have nothing to tune by CV, so their `cv_mse` is NaN.
- **log10_I runs only with vac_pct** (`experiments.point_vacancy_features`). The per-gate GP costs about 170 s per job. Both vacancy encodings are compared on all 7 curve-level targets.
- **I–V target via one model per gate voltage** for the naive baselines and the GP (`Gatewise`). Ridge, polynomial ridge and boosting use the gate as a third input instead.
- **Skill uses the better of the two naive baselines on the same test set.** This is a fixed reference, not a tuned choice.
- **Near-zero GP error on the supplied split is NOT leakage.** It was checked before reporting. No (T, vacancy) key is shared across any split, and at a fixed vacancy level log10 Ion/Ioff changes by a median of 0.0016 decades per 2 K step (90th percentile 0.11). The supplied test curves sit between training temperatures with the same layout, so the supplied split is nearly trivial. Numbers per model and protocol are in `results/metrics_classical.csv`.

## 2026-11-17 — Stage 5 (MLP)

- **New module `neural.py`** (not in the Stage 0 layout). The MLP is about 100 lines and `models.py` should keep one job. `models.build_model("mlp")` imports it, so data still flows one way.
- **Training:** AdamW with mini-batches of 512 rows (full-batch for curve-level tasks with ≤ 638 rows), up to 1,500 epochs, early stopping with patience 100 checked every 5 epochs. *Rejected:* full-batch for log10_I. In a test it was still improving at epoch 3,000 after 118 s, whereas mini-batch reached a better test MAE (0.040 vs. 0.141, supplied split, 128 × 3) in 48 s. These are scratch timings, not results.
- **Early-stopping set = 15 % of training curves, chosen at random.** Whole curves, so no gate point of a training curve is used to stop. *Rejected:* a vacancy block. Early stopping only decides when to stop, and blocked CV already does the honest hyperparameter selection.
- **SiLU activation:** smooth, so predicted curves have no ReLU kinks. **Dropout 0:** the data are noiseless simulation output.
- **Grid:** width {64, 128} × depth {2, 3} × weight decay {0, 1e-4} for curve targets. For log10_I (about 50 s per fit) weight decay is fixed at 1e-4 to fit the CPU budget (`point_grid_override`). Tuning uses seed 0, then the chosen model is refitted with seeds 0–4.
- **MLP targets:** log10_I, both Ion/Ioff definitions, swing. SPD is skipped (it equals V_DS·Ioff and has the same error as Ion/Ioff in Stage 4), as are the nearly constant P_dyn, DED and tau_d. Only the vac_pct encoding is used, because Stage 4 showed the encoding does not matter.
- **Cache keys scoped per step:** the classical cache ignores MLP-only config keys, so adding the MLP did not invalidate the 19-minute classical run (hash unchanged: b3de8b8f).
- **Parallelism:** one torch thread per fit, with CV fits and seeds run 4 at a time (`compute.n_jobs`).
- **Result check (not leakage, but checked):** the MLP's vacancy-extrapolation win on log10 Ion/Ioff holds for every seed (MAE 0.75–1.63 vs. ridge 2.12, `results/metrics_mlp.csv`). It comes from extrapolating the falling trend.
- **Compute budget EXCEEDED.** The MLP step took about 3,890 s of fitting in total (sum of `fit_seconds` in `results/metrics_mlp.csv`), of which log10_I took about 3,025 s. With the classical step (1,132 s) the full run is about 84 min against the "well under an hour" target. Fix options are in the Stage 5 review; nothing has been changed yet.
- **Memory:** the first MLP run was stopped by the system for low memory with 4 parallel fits. `compute.n_jobs` is now 2. Speed only, not results; not part of the cache key.
- **Early stopping cap:** 4 of the 80 seed fits stopped at the 1,500-epoch limit rather than by patience (`best_epochs` in `results/metrics_mlp.csv`), all on curve-level supplied-split tasks.

## 2026-11-18 — Notebook for Colab / Kaggle (requested before Stage 6)

- **`notebooks/WTe2_TIFET_pipeline.ipynb`** clones the public repository (branch `test_1`), installs only pyyaml, pytest and scikit-learn ≥ 1.5 (the platforms already ship numpy, pandas, scipy, torch and matplotlib), and calls package functions only. Every cell has a heading naming the question it answers.
- **The slow model runs are off by default** (`RERUN_CLASSICAL`, `RERUN_MLP`). The notebook reads the committed results so it finishes in about 2 min. *Rejected:* always re-running, which would take about 85 min and could hit Colab's idle timeout.
- **Verified** by executing the notebook in an empty folder, cloning from GitHub: all cells ran, 50 tests passed, and the outputs match `results/*.csv`. Total time was 96 s locally (scratch run).
- The notebook needs no torch reinstall. On a GPU runtime it still runs on CPU (`compute.device: cpu`).

## 2026-11-18 — Stage 5 revision: curve-output MLP for log10_I (your choice A)

- **log10_I MLP = one row per curve:** input (T_K, vac_pct), 20 outputs (the curve's log10 I at each gate). That is 638 training rows instead of 12,760, and matches the per-gate GP. Predictions are flattened back to (curve, gate) rows, so scoring is identical to the other models; `tests/test_runner.py` checks the alignment row by row. The `point_grid_override` (weight decay fixed for log10_I) is removed, so log10_I gets the full 8-config grid again.
- **Run time:** the MLP step took 2,017 s fresh (was about 3,890 s). The log10_I jobs took 732 s, against about 3,025 s before (`fit_seconds` in `results/metrics_mlp.csv`). Full fresh pipeline is about 53 min (classical 1,132 s + MLP 2,017 s).
- **Reproducible:** all 12 curve-level MLP jobs gave exactly the same MAE as the previous run (seeded, one torch thread per fit).
- **Accuracy trade-off for log10_I** (`results/mlp_vs_classical.csv`): supplied 0.157 ± 0.006 (per-point MLP was 0.030), temperature_forward 0.188 ± 0.054 (was 0.078), vacancy_block 1.008 ± 0.085 (was 0.983), vacancy_extrapolation 0.924 ± 0.288 (was 1.572 ± 0.800). The overall conclusion is unchanged: the MLP beats the classical models only on curve-level vacancy extrapolation.
- **Open:** on the supplied split all 5 seed fits stopped at or near the 1,500-epoch cap (best epochs 1330–1490), so the curve-output MLP is under-trained there (about 2 updates per epoch instead of about 20).

## 2026-11-18 — Option 2: I–V MLP trained to early stopping (epoch limit 1,500 → 8,000)

- **Setting:** `experiments.mlp_iv_max_epochs: 8000`, used only for the 4 log10_I MLP jobs. It is left out of the MLP cache hash (still 17f77466) and added to those jobs' cache file names (`maxep8000`), so the 12 curve-level MLP jobs were loaded from cache and are bit-identical (`changed = False` for all 12 in the comparison). The MLP now records `stop_epochs` (last epoch run) and `seeds_hit_epoch_cap` per job (`results/metrics_mlp.csv`).
- **Every seed stopped by early stopping, none by the cap** (`seeds_hit_epoch_cap` = 0 for all 4 jobs). Last epoch run per seed (`stop_epochs`): supplied 1435, 1390, 1295, 1615, 1710; temperature_forward 955, 2205, 630, 1725, 595; vacancy_block 1000, 1015, 1170, 720, 1415; vacancy_extrapolation 1220, 1330, 1515, 1105, 1530.
- **log10_I MAE, mean ± std over 5 seeds, before → after** (`results/mlp_vs_classical.csv`):
  - supplied 0.157 ± 0.006 → **0.052 ± 0.008** (better; under-training confirmed as the cause)
  - temperature_forward 0.188 ± 0.054 → 0.174 ± 0.071 (slightly better on average, larger spread)
  - vacancy_block 1.008 ± 0.085 → 1.008 ± 0.085 (identical: no fit in this job reached 1,500 epochs before, so nothing changed)
  - vacancy_extrapolation 0.924 ± 0.288 → **0.917 ± 0.289** (essentially unchanged, not worse on average). Per seed (`results/metrics_mlp.csv`): seed 0 got slightly worse (0.900 → 0.909), seeds 2 and 4 slightly better (0.902 → 0.867, 0.546 → 0.543), seeds 1 and 3 almost unchanged. Blocked CV now picks weight decay 0 instead of 1e-4 for this job.
- **The conclusion still holds:** the GP (0.001, 0.019) and boosting (0.650, 0.480) remain better than the MLP for log10_I on every protocol.
- **Run time:** the 4 log10_I jobs took 901 s (was 732 s). A fresh MLP step is 2,163 s (901 I–V + 1,262 curve-level, sum of `fit_seconds`). With the classical step (1,132 s) a fresh full run is about 3,295 s ≈ **55 min**. That is under the one-hour budget, with little margin.

## 2026-11-18 — Stage 6 (PCA and clustering)

Defaults kept for the four open questions (none affects this stage).
- **PCA on the (798, 20) log10_I matrix, centred, not scaled.** Every column is in decades, so scaling would inflate the flat on-state columns (V < 0.8 V), which carry almost no variance. Fitted on all 798 curves: descriptive only, so no split is involved.
- **Cluster on the first 6 PCs** (fewest reaching 99 % of variance, `cluster_variance_kept`). *Rejected:* clustering the raw 20 columns, which is dominated by the same directions but noisier; and 2 PCs only (91 %), which would cluster the picture rather than the data.
- **Ion/Ioff classes for comparison:** < 4 failed, 4–9 degraded, ≥ 9 working (supplied log10 Ion/Ioff). The 9-decade edge sits in a sparse region of the histogram (13 of 798 curves between 9 and 11). The 4-decade edge is a convention we chose, not a property of the data. Class sizes: 190 failed, 552 degraded, 56 working (`results/cluster_assignments.csv`).
- **Methods:** k-means and a full-covariance Gaussian mixture, k = 2–6, 10 restarts, seed 0.
- **Bug caught by a test:** `n_components_for` returned one component too many when the requested share was never reached. It is fixed. Our results were unaffected (99 % is reached at 6 of 10 components).
- **Finding (checked numerically, not only by eye):** the clusters are bands of vacancy levels. With k = 3, k-means keeps 19 of 21 levels entirely in one cluster (0.3 % and 1.3 % are split by temperature), and the Gaussian mixture keeps 20 of 21 (1.3 % split). Scratch check from `results/cluster_assignments.csv`.

## 2026-11-18 — Stage 7 (final protocol changes, robustness, figures, report notes)

Your decisions, all applied in one fresh run:
1. **Headline target = log10_Ion_Ioff** (supplied end-point Ioff). log10_Ion_Ioff_min is reported as sensitivity: it uses the lowest current of the sweep, not the current at 1.9 V.
2. **1-level embargo on the final vacancy tests** (`holdout_embargo_levels: 1`): the vacancy block now trains without 0.7 and 1.3 % (532 curves), and extrapolation without 1.5 % (570 curves) [`results/split_summary.csv`].
3. **Temperature-blocked CV for the 62–78 K test:** `Split.cv_axis = "temperature"`, 5 blocks of consecutive temperatures inside 4–60 K, 1-temperature embargo (`cv_temp_*`).
4. **GP kernel chosen by blocked CV** (RBF vs. Matérn-5/2); length scales and noise are still fitted by marginal likelihood. `KernelSelectGP` was removed. CV chose Matérn-5/2 in 29 of 32 GP jobs [`results/metrics_classical.csv`, best_params].
5. **Option B for run time:** 4 parallel classical fits, serial MLP (`compute.mlp_n_jobs: 1`), GP and MLP on all 8 targets, and `n_vac_atoms` dropped. The Stage 4 encoding finding is kept in `results/encoding_comparison_stage4.csv` (median ratio 1.0, 168 jobs).

Implementation choices:
- **Cache names include a split fingerprint** (hash of the exact train/test curves and CV axis), so a cached job is never reused for a different split. The MLP log10_I epoch limit stays in the file name (`maxep8000`).
- **Robustness step** (`wte2ml/robustness.py`, `run_experiments.py robustness`): Gaussian noise on the TEST inputs only, std = 0, ¼, ½, 1 grid step per input (2 K, 0.1 %), 5 draws, protocols supplied and vacancy_block, target log10_Ion_Ioff, 7 models. Each model is tuned and fitted once on clean data, with MLP seed 0 only. *Rejected:* one shared noise level in standardised units, because a 1-σ change in temperature and in vacancy mean very different things physically.
- **Shared `_tuned` helper** for the classical, MLP and robustness steps, so the tuning logic is not repeated.
- **Figures:** plotting primitives moved to `wte2ml/plots.py`. `make_figures.py` only reads `results/*.csv` and saves PNGs, filtering to one vacancy encoding so encodings can never mix in a figure. `results/model_comparison.csv` comes from `evaluate.comparison_table`.
- **Resuming after a low-memory stop:** the MLP was stopped by the system after 16 of 32 jobs (free RAM 0.5–0.9 GB). On restart, `--protocols` ran the vacancy block and extrapolation in separate processes, and a per-job memory log (`results/memory_log.csv`, needs `psutil`) showed a flat peak of 394 / 398 MB per process. The 16 cached jobs were reused unchanged (MLP cache key 52fd9793 unchanged).
- **Run times are summed per job** because the MLP run spans two sessions: classical 1,827 s, MLP 6,043 s, robustness 253 s, figures 49 s, about 2 h 16 min in total [`results/timings_log.csv`, `metrics_*.csv`]. That is **over the one-hour budget**, mainly because of the serial MLP on 8 targets.
- **Wording:** the 9-decade class edge lies in a *sparse region* of the histogram (13 of 798 curves between 9 and 11), not an empty gap. SPD = 0.05·Ioff duplicates Ioff. DED and τ_d are not exact copies of another modelled target (τ_d = 20·DED/Ion), and P_dyn = 0.05·Ion exactly. All verified on the supplied metrics files.

Results that changed (for the full story see `docs/REPORT_NOTES.md`):
- The MLP's Stage 5 win on vacancy extrapolation of log10 Ion/Ioff (1.080 ± 0.343) **did not survive the embargo**: 4.378 ± 3.280, against ridge 2.204 [`results/mlp_vs_classical.csv`].
- 51 of 140 curve-level MLP seed fits reached the 1,500-epoch limit (none of the 20 I–V fits did) [`results/metrics_mlp.csv`].

## 2026-11-18 — Stage 8 (uncertainty, coverage, ridge from scratch, CI, epoch-cap check)

- **Cluster bootstrap over whole vacancy levels** (your rule 1), 2,000 resamples, on the saved predictions with no refits. *Rejected:* resampling single curves, which treats 38 near-copies of one layout as independent; a test shows that gives intervals more than 5× too narrow on clustered data. Also *rejected:* a two-way bootstrap over vacancy × temperature. With 5 levels in the vacancy tests, the level clustering already dominates, and temperature curves inside a level are not independent either.
- **Paired comparisons** use the better naive baseline by test MSE (the skill reference) as reference, with the same resampled levels for both models. **192 comparisons are counted** (32 tasks × 6 models), and a Bonferroni-adjusted interval at 1 − 0.05/192 is reported next to the plain 95 % one.
- **MLP in the bootstrap = its 5-seed ensemble** (mean prediction); `model_comparison.csv` keeps the mean of the per-seed MAEs. Both are now labelled in REPORT_NOTES. The classical models agree between the two files to 1e-14.
- **GP coverage:** nominal 68.27 % (±1σ) and 95 % (±1.96σ) per protocol × target, plus the mean predictive σ.
- **Ridge from scratch** (`wte2ml/ridge_numpy.py`): closed form on standardised X with an unpenalised intercept, exactly the pipeline's StandardScaler + Ridge. The test uses the real cubic-feature task at the 4 grid α values, atol 1e-8. Coefficient path and effective dof included.
- **CI checked first** (your rule 4): the tests read only tracked files (four CSVs, `NEGF_all.mat`, config). There is no Git LFS and no test touches `results/`, the cache or figures. The workflow installs CPU torch plus `requirements.txt` and runs `pytest -q`; permissions are `contents: read`. A local fresh clone passes 75 tests.
- **`model_comparison.csv` shows both naive baselines** on every row (`MAE_naive_vac_mean`, `MAE_naive_nn`).
- **Epoch-cap check:** 4 curve-level MLP jobs with the most capped seeds, limit 1,500 → 8,000, run in their own cache folder (`mlp_epochsens_*`), so the main results are untouched. Estimated 15–30 min; actual 668 s. Result: no improvement, and 2 of 4 got worse.
- **Prediction loading moved to `evaluate.pooled_predictions`**, shared by the figures and the uncertainty step (it used to live only in `make_figures.py`).

## 2026-11-18 — Theory notebook (requested after Stage 8)

- **`notebooks/WTe2_TIFET_theory.ipynb`**: 16 sections, one per component in data-flow order. Each has *what we do*, *theory* (equations), *what is optimised* (the objective, and what chooses its settings), *why it fits / how it fails here*, then the component's figures and tables.
- **No result numbers in the text.** Every number is printed by a code cell from `results/*.csv`, so the explanations cannot drift from the data. Cells only call `wte2ml` functions or read saved files. Two plotting helpers were added to `wte2ml/plots.py` (`coefficient_path_lines`, `coverage_bars`) so the notebook draws nothing itself.
- It reads saved results only (no refits). Verified on a fresh clone: all cells ran, no errors, 14 s. Cell `id` fields were added to both notebooks for current nbformat.

## 2026-11-18 — Standalone notebook (requested after Stage 8)

Your choices: **fully standalone** (all code in cells, only the data downloaded) and **Kaggle with parallel fits** (classical and MLP, 4 at a time).
- **Generated, not hand-copied:** `python/build_standalone_notebook.py` reads the package source with `ast` and writes one markdown + code cell per function/class, in data-flow order. Module docstrings become section intros, and package imports are stripped. `config.py` is replaced by a notebook cell that embeds `config.yaml` as editable text. *Rejected:* copying code by hand, which would drift from the repository.
- **Drift guard:** `tests/test_standalone_notebook.py` regenerates the notebook and fails if it differs from the committed file. Checked by mutation: an added function made it fail, and reverting it made it pass.
- **Name collision fixed at the source:** `MODELS` existed in `models.py` (6 classical models) and `make_figures.py` (7 incl. MLP). In one namespace the later one would have changed which models the classical step runs, so the figure list is now `FIGURE_MODELS`.
- **Verified locally** (live GitHub data download, 2 parallel fits, classical/MLP/epoch results reused, everything else recomputed): all cells ran, 13 figures, 318 s. A temporary check cell recomputed ridge, boosting, GP and polynomial-ridge jobs and one 5-seed MLP job from scratch in the notebook. All matched the saved MAEs exactly (MLP max difference 0.0). All 9 recomputed result files (targets, splits, PCA, clusters, robustness, bootstrap, coverage, comparison table) are identical to `results/`.
- **Not yet measured:** the full Kaggle run time. The estimate is 1–1.5 h with 4 parallel fits.
