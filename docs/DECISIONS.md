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
