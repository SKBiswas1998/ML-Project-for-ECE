# Report notes: key numbers, failures, limitations

Every number below comes from a file in `results/`, cited in brackets. A few come from direct checks of the supplied data files and are marked as such. MAE and RMSE are in **decades of log10** (mV/dec for swing). Skill = 1 − MSE_model / MSE_best_naive on the same test set (0 = as good as the better naive baseline, negative = worse). All numbers come from one consistent fresh run on 2026-11-18 (embargoed vacancy tests, temperature-blocked CV for 62–78 K, GP kernel chosen by CV, `vac_pct` encoding only).

## 1. Data and targets
- 798 NEGF gate sweeps: 38 temperatures (4–78 K) × 21 edge-vacancy levels (0–2.0 %), 20 gate points each, one nested random layout (seed 2026) [`curve_targets.csv`, `split_summary.csv`].
- **Headline target: log10 Ion/Ioff** as supplied. Ion is the current at V_TG − V_BG = 0, and Ioff is the current at the **last** gate point (1.9 V) [`curve_targets.csv`].
- **Sensitivity variant, log10 Ion/Ioff (min):** it uses the *lowest* current along the sweep instead of the end point, so it is never smaller and differs wherever the curve turns upward again before 1.9 V.
- Power, energy and delay are fixed formulas of the supplied columns. This was verified to machine precision on `ML_metrics_*.csv`; the full spreads are from a scratch check of those files:
  - SPD = 0.05·Ioff, so **log10 SPD duplicates log10 Ioff** (not the Ion/Ioff ratio). Its errors track the ratio's only because Ion varies by just 0.067 decades over all 798 curves.
  - DED = 0.0025·C_eff and τ_d = 0.05·C_eff/Ion. **Neither is an exact copy of another modelled target**, because C_eff is not modelled. They are tied by τ_d = 20·DED/Ion, so their logs differ by 0.067 decades at most and correlate at 0.998.
  - P_dyn = DED/τ_d = 0.05·Ion exactly. It spans 0.067 decades, so it is a nearly constant target.
- **Swing is a supporting result only.** PCHIP and raw-grid values differ by a factor of 0.57–0.98, and the gate grid is 0.1 V [DECISIONS Stage 2, `curve_targets.csv`].

## 2. Evaluation protocols [`split_summary.csv`]
| Protocol | Train curves | Test curves | Tuning CV |
|---|---|---|---|
| supplied (interpolation) | 638 | 160 | 5 vacancy blocks, 1-level embargo |
| temperature_forward (train ≤ 60 K, test 62–78 K) | 609 | 189 | 5 temperature blocks, 1-temperature embargo |
| vacancy_block (test 0.8–1.2 %; 0.7 and 1.3 % embargoed) | 532 | 190 | vacancy blocks |
| vacancy_extrapolation (train ≤ 1.4 %; 1.5 % embargoed; test 1.6–2.0 %) | 570 | 190 | vacancy blocks |

## 3. Headline results: log10 Ion/Ioff [`model_comparison.csv`, `figures/model_comparison_log10_Ion_Ioff.png`]
| Model | supplied | temperature_forward | vacancy_block | vacancy_extrapolation |
|---|---|---|---|---|
| naive per-vacancy mean | 0.242 | 0.619 | **0.698** | 3.463 |
| naive nearest neighbour | 0.056 | 0.224 | 0.698 | 3.463 |
| ridge | 1.010 | 1.095 | 1.037 | **2.204** |
| polynomial ridge | 0.979 | 0.954 | 1.000 | 6.970 |
| gradient boosting | 0.176 | 0.268 | 0.743 | 3.530 |
| Gaussian process | **0.001** | **0.067** | 1.072 | 4.620 |
| MLP (mean of 5 seeds) | 0.352 | 0.555 | 0.963 | 4.378 |

- **Known layouts:** the GP is near-exact. Supplied split skill 1.00 (R² 1.00); 62–78 K skill 0.94 (R² 1.00).
- **Unseen vacancy levels:** no model beats the naive per-vacancy mean on the block. Every model has R² < 0 there, i.e. it is worse than predicting the test-set mean. On extrapolation only ridge has positive skill (0.56), and its MAE is still 2.2 decades.
- **Sensitivity (log10 Ion/Ioff min):** same pattern on known layouts (GP 0.004 supplied, 0.176 on 62–78 K). On the vacancy block the MLP is best (1.098, skill 0.67) and on extrapolation ridge (1.427, skill 0.50).

## 4. I–V curves: log10 I per gate point [`model_comparison.csv`, `figures/iv_samples_gp_bands.png`]
- Best MAE per protocol: GP 0.0005 (supplied), GP 0.019 (62–78 K), GP 0.668 (vacancy block, skill 0.20), naive per-vacancy mean 0.633 (extrapolation).
- Ridge (1.13–1.48) and polynomial ridge (0.47–0.99) cannot follow the step-like switching on any protocol.
- The GP's ±2σ band is narrow on known layouts and wide on unseen vacancy levels (figure), so it signals its own uncertainty. On extrapolation to 1.8–2.0 % its mean is still far below the truth.

## 5. Does the neural network beat simpler models? [`mlp_vs_classical.csv`, `metrics_mlp.csv`]
**No, with two exceptions.** It wins on log10 Ion/Ioff (min) on the vacancy block (1.098 ± 0.125 vs. boosting 1.192) and, by a hair, on log10 P_dyn there (0.0010 vs. 0.0011). It loses everywhere else, for example:
- log10 Ion/Ioff: 0.352 ± 0.117 (supplied, GP 0.001); 4.378 ± 3.280 (extrapolation, ridge 2.204).
- log10 I: 0.052 ± 0.008 (supplied, GP 0.0005); 4.506 ± 0.991 (extrapolation, naive 0.633).
- Swing on the vacancy block: 140.0 ± 50.4 mV/dec, against 10.2 for polynomial ridge. The seed spreads show unstable extrapolation.

**An earlier win did not survive.** Before the 1-level embargo the MLP led on vacancy extrapolation of log10 Ion/Ioff (1.080 ± 0.343, DECISIONS Stage 5). With 1.5 % removed from training it fell to 4.378 ± 3.280. That win depended on one training level.

**Training status.** The 20 I–V seed fits all stopped by early stopping (`seeds_hit_epoch_cap` = 0, limit 8,000). Of the 140 curve-level seed fits, 51 reached the 1,500-epoch limit, so some curve-level MLPs are under-trained.

## 6. Input-noise robustness [`robustness_summary.csv`, `figures/robustness_noise.png`]
Gaussian noise was added to the test inputs (T, vacancy) with a standard deviation of 0, ¼, ½ and 1 grid step (2 K, 0.1 %), 5 draws per level, on log10 Ion/Ioff.
- **Supplied split: accuracy depends on exact inputs.** GP MAE 0.001 → 0.201 (¼ step) → 0.448 (½) → 0.718 (1 step). Nearest neighbour 0.056 → 0.710 and boosting 0.176 → 0.707 at 1 step. At one grid step all flexible models converge to about 0.7 decades. The near-perfect interpolation is a look-up of the right layout, and a vacancy error of a fraction of a level picks a different layout.
- **Vacancy block: almost insensitive.** Errors are already set by the unseen layout: GP 1.072 → 1.083, boosting 0.743 → 0.745, MLP 0.623 → 0.670 at 1 step. (The MLP's clean MAE here, 0.623, is a single fit with seed 0; the 5-seed mean in section 3 is 0.963.)

## 7. Where models fail [`figures/error_map_log10_Ion_Ioff.png`, `figures/error_profiles_log10_Ion_Ioff.png`, `predictions_classical.csv.gz`]
- **Errors on unseen vacancy levels are vertical stripes:** they depend on the vacancy level (layout), not on temperature.
- **Extrapolation fails most at 1.9–2.0 %,** where Ion/Ioff collapses to about 1.2 decades. Ridge's MAE by level is 2.56, 1.43, 0.89, 3.19 and 2.96 for 1.6–2.0 %; the GP's is 4.19–5.98.
- **On 62–78 K the naive per-vacancy mean fails on the clean devices (0–0.1 %),** the only ones whose ratio changes strongly with temperature (13.4 decades at 4 K to 7.0 at 78 K for 0 %, `curve_targets.csv`). The GP handles them.

## 8. Unsupervised [`pca_explained_variance.csv`, `cluster_scores.csv`, `cluster_assignments.csv`]
- PCA: PC1 explains 83.7 % of the variation between log10 I–V curves, 2 PCs 90.8 % and 6 PCs 99.4 %.
- k-means and Gaussian-mixture clusters (k = 3) are bands of vacancy levels, not natural device classes. The silhouette is 0.47–0.52 for k = 2–6, and the ARI against the Ion/Ioff classes is 0.42 (k-means) and 0.45 (GMM).
- Class edges for that comparison: < 4, 4–9 and ≥ 9 decades. The 9-decade cut lies in a **sparse region** of the histogram (13 of 798 curves between 9 and 11 decades). The 4-decade cut is a convention we chose.

## 9. Vacancy encoding [`encoding_comparison_stage4.csv`]
- Percent vs. number of removed atoms made no difference: the median MAE ratio is 1.0 over 168 Stage 4 jobs, and atoms were better in 26 % of them. Stage 7 therefore uses `vac_pct` only. That comparison was made under the Stage 4 settings (no final embargo, vacancy-blocked CV everywhere).

## 10. Run time on a laptop CPU [`timings_log.csv`, `memory_log.csv`, `metrics_*.csv`]
| Step | Time | Notes |
|---|---|---|
| targets + splits | 0.8 s | |
| classical (192 jobs, 4 parallel fits) | 1,839 s step (1,827 s summed per job) | GP 1,686 s of it |
| MLP (32 jobs, serial) | **6,043 s summed per job** | split across sessions after a low-memory stop: supplied 1,817 s, 62–78 K 1,460 s, vacancy block 1,466 s, extrapolation 1,301 s |
| robustness | 253 s | |
| figures | 49 s | |
| **Total** | **≈ 8,170 s ≈ 2 h 16 min** | over the one-hour target; the serial MLP on all 8 targets dominates |

Peak memory per MLP process was 394 MB (vacancy block) and 398 MB (extrapolation), with no growth from job to job [`memory_log.csv`]. The low-memory stops came from the PC as a whole (0.5–0.9 GB free), not from a leak.

## 11. Limitations
1. **One nested random layout.** Every vacancy level contains all atoms of the levels below it, so levels are not independent samples. No protocol can measure how a *different* layout would behave, and blocked CV with an embargo only reduces the leakage between neighbouring levels.
2. **The supplied test split is nearly trivial:** at a fixed level, Ion/Ioff changes by a median of 0.0016 decades per 2 K step (DECISIONS Stage 4). Near-zero errors there reflect that, not model quality, and they vanish under ¼ grid step of input noise (section 6).
3. **Coarse gate grid** (20 points, 0.1 V). Single steps drop up to 5.37 decades (Stage 0 data profile, DECISIONS), so the swing depends on the interpolation method.
4. **Simulation, not experiment.** The data come from non-self-consistent NEGF (no Poisson solve) for one fixed geometry, so the surrogates reproduce that model only.
5. **Small test sets per protocol** (160–190 curves from 5–38 levels), so the differences between close models (e.g. 0.698 vs. 0.743) are not tested for significance. Stage 8 bootstrap intervals would be needed.
