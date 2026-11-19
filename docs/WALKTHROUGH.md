# Walkthrough (stage by stage)

## Stage 0: what is in the data?
We simulated a 5.7 nm × 10.5 nm 1T′-WTe₂ ribbon with NEGF. The grid covers 38 temperatures (4–78 K) × 21 edge-vacancy levels (0–2 %), which gives 798 gate sweeps of 20 points each. Three findings shaped everything that follows:
1. The vacancy levels are **nested**: one random layout, with atoms added level by level.
2. Most curves turn **upward** at the end of the sweep, and the supplied Ioff is the end-point current, not the minimum current.
3. Static power, switching energy and delay are exact formulas of Ion, Ioff and C_eff (V_DS = 0.05 V).

## Stage 1: loading the data safely
`data.py` reads the four CSV files and refuses to continue if anything differs from the facts verified in Stage 0. The checks cover row counts, missing values, the reliability flag, the seed, 20 points per curve on one shared gate grid, and train/test overlap. It also provides the curve matrix, with one row of 20 log-currents per device, and the vacancy layout. The tests check these facts and confirm that broken inputs are rejected.

## Stage 2: what do we predict?
Every target is in log10, because the off-current covers about 12 decades.
- **Ion/Ioff: two versions.** We keep two on/off ratios. One is the supplied value (current at the last gate voltage). The other uses the lowest current of the sweep, because most curves rise again near the end.
- **Power and energy.** Static power, switching energy, delay and dynamic power follow from Ion, Ioff and the capacitance by simple formulas. Dynamic power is almost constant, so it will not be an interesting target.
- **Swing.** The subthreshold swing is measured on the falling part of each curve with a smooth, non-overshooting interpolation (PCHIP). We then looked at 12 sample curves (`figures/swing_check.png`). The gate grid is coarse, and in many curves one step drops two decades or more, so swing is a supporting result.
- **Physics note.** No curve switches more steeply than the thermal limit kT ln10/q. For the clean device the swing is about the same at 4 K and 78 K, which points to gate-controlled gap opening of the edge states rather than electrons climbing over a barrier.

## Stage 3: how do we test fairly?
A model is only as trustworthy as its test. We split by whole curves, never by individual points, and use four protocols:
1. **Supplied split**: easy interpolation.
2. **Temperature forward-chaining**: learn 4–60 K, predict 62–78 K.
3. **Vacancy block**: predict 0.8–1.2 % without having seen it.
4. **Vacancy extrapolation**: learn ≤ 1.5 %, predict 1.6–2.0 %.

Model settings are chosen only inside each training pool, with blocked cross-validation. A whole block of neighbouring vacancy levels is held out, and the levels next to it are dropped too (the embargo), because nested layouts make neighbours near-copies. The tests prove that no curve crosses sides and that the embargo holds. A deliberately broken version fails them.

## Stage 4: do classical models beat a trivial guess?
*Numbers in this section are from the Stage 4 settings (no final embargo, GP kernel by marginal likelihood). The final numbers are in Stage 7 and docs/REPORT_NOTES.md.*
We compared two naive baselines (per-vacancy mean, nearest neighbour) with ridge, polynomial ridge, gradient boosting and a Gaussian process on all four protocols. MAE is in decades and all numbers are from `results/metrics_classical.csv` (vacancy encoding `vac_pct`).
- **Known layouts (supplied split, temperature forward-chaining): the GP wins clearly.** For log10 I the MAE is 0.001 (supplied) and 0.019 (62–78 K), against 0.020 and 0.057 for nearest neighbour, giving skill 1.00 and 0.89. The near-zero supplied error was checked for leakage. There is none: at a fixed vacancy level the curves barely change between neighbouring temperatures, so the supplied test set is close to trivial.
- **Unseen vacancy levels: nothing reliably beats the naive guess.** Holding out 0.8–1.2 %, the best log10 Ion/Ioff MAE is the per-vacancy mean (0.492). Boosting reaches 0.520 (skill 0.10, on MSE) and the GP 0.879 (skill −1.57). Extrapolating to 1.6–2.0 %, every model is off by about 2–4 decades in Ion/Ioff.
- **Ridge and polynomial ridge are poor everywhere for I–V curves** (log10 I MAE 0.82–1.42), because the step-like switching cannot be captured by low-order polynomials.
- **The vacancy encoding (percent vs. number of atoms) makes no difference.** The median MAE ratio across jobs is 1.0.

The lesson: the models learn temperature well but cannot predict a vacancy layout they have not seen. With one nested random layout, the data contain no information about how a *different* set of missing atoms would behave.

## Stage 5: does a neural network do better?
*Numbers in this section are from the Stage 5 settings. Under the final protocols (Stage 7) the MLP's extrapolation win disappears.*
We trained a small MLP (2–3 layers, 64–128 units) with 5 random seeds. Numbers are mean ± std over the seeds, MAE in decades, from `results/mlp_vs_classical.csv`.
- **Known layouts: no.** On the supplied split the MLP reaches log10 I MAE 0.052 ± 0.008, against the GP's 0.001. For I–V curves the MLP predicts all 20 gate points of a curve at once from (T, vacancy), and is trained until early stopping ends it (no seed reached the epoch limit). For log10 Ion/Ioff it is 0.352 ± 0.117, worse even than nearest neighbour (0.056). With only about 600 curves, the GP's built-in smoothness wins.
- **Unseen vacancy block: no.** Its log10 Ion/Ioff MAE is 0.629 ± 0.032, against 0.492 for the naive per-vacancy mean.
- **Vacancy extrapolation (1.6–2.0 %): yes, for the curve-level targets.** log10 Ion/Ioff MAE is 1.080 ± 0.343 against ridge's 2.124 (skill 0.76), and all 5 seeds beat ridge (`results/metrics_mlp.csv`). The MLP continues the downward trend of Ion/Ioff with vacancy, while the linear and naive models stay near the training average. With a single layout this is one lucky-or-not trend, not proof of learned physics. For the I–V curves the MLP is worse there (0.917 ± 0.289 vs. boosting 0.480).
- **Seeds matter.** The spread across seeds is large for extrapolation (see the per-seed rows in `results/metrics_mlp.csv`).

The honest answer to "does the neural network beat simpler models?" is mostly no. It is the best model only when extrapolating the on/off ratio beyond the trained vacancy range, and even there it errs by about 1 decade.

## Stage 6: are there natural groups of devices?
We compressed every I–V curve with PCA and clustered the devices without telling the algorithm anything about vacancy or temperature. Numbers are from `results/pca_explained_variance.csv` and `results/cluster_scores.csv`.
- **The curves are simple.** One component explains 83.7 % of the variation between curves, two explain 90.8 %, and six explain 99.4 %. PC1 measures how high the current stays above about 1 V, i.e. how leaky the off state is. PC2 captures the dip around 1.0–1.5 V and the rise after it (the upturn).
- **The clusters are vacancy bands.** With three clusters, k-means and a Gaussian mixture both find roughly clean (about 0–0.3 %), intermediate (about 0.3–1.3 %) and heavily damaged (about 1.3–2.0 %) devices, and keep almost every vacancy level whole. They match our on/off-ratio classes only moderately (adjusted Rand index 0.42 for k-means, 0.45 for the mixture). All 190 "failed" curves fall in the damaged cluster, but so do about 100 "degraded" ones.
- **There is no natural number of groups.** The silhouette stays between 0.47 and 0.52 for 2–6 clusters. In the PCA plot (`figures/pca_clusters.png`) each vacancy level is a thin strand traced out by temperature, so the algorithms are dividing strands, not separating blobs. The vacancy layout, not the temperature, determines which family a device belongs to.

## Stage 7: the final, fair comparison
We tightened the tests and re-ran everything once, consistently. The vacancy tests now also drop the neighbouring level from training. The higher-temperature test is tuned on blocks of temperatures. The GP's kernel is chosen by cross-validation. Numbers are from `results/model_comparison.csv` and `results/robustness_summary.csv`.
- **Temperature is learnable.** For unseen higher temperatures (62–78 K) the GP predicts the on/off ratio within 0.067 decades (skill 0.94).
- **An unseen vacancy layout is not.** For the held-out block 0.8–1.2 %, no model beats "use the nearest vacancy level's average" (0.698 decades), and every model is worse than simply predicting the test mean (R² < 0). Extrapolating to 1.6–2.0 %, the best model (ridge) is still off by 2.2 decades.
- **The neural network loses almost everywhere.** Its earlier extrapolation win did not survive removing one training level: 4.38 ± 3.28 decades against ridge's 2.20.
- **The near-perfect interpolation is fragile.** A temperature or vacancy error of a quarter grid step raises the GP's error from 0.001 to 0.20 decades. Its accuracy comes from recognising the exact device, not from smooth physics.
- **Where errors live.** On unseen layouts the errors form vertical stripes in the temperature × vacancy map (`figures/error_map_log10_Ion_Ioff.png`): they depend on the layout, not on temperature.

The take-home for the class: with one random defect layout, a surrogate can learn how temperature changes a known device very well. It cannot learn what a *different* defect arrangement would do, because the data never show one.

## Stage 8: how sure are we?
We put error bars on every reported error, without retraining anything. Numbers are from `results/bootstrap_ci.csv` and `results/gp_coverage.csv`.
- **Error bars must respect the layouts.** We resample whole vacancy levels, not single curves, because curves of one layout are near-copies. For the unseen-vacancy tests there are only 5 levels, so the intervals are wide: the best on/off-ratio error on the held-out block is 0.70 decades, with a 95 % interval of 0.40–0.97.
- **After correcting for 192 comparisons**, only 20 model-versus-baseline differences are significantly better, and most are the GP on known layouts.
- **The GP's own error bars are trustworthy only where it has seen the layout.** On known layouts they are, if anything, too cautious. When extrapolating to 1.6–2.0 % vacancy, the true on/off ratio falls inside its "95 %" interval only 40 % of the time.
- **Training the MLP longer does not help.** Removing the epoch limit on four of its worst jobs left the error unchanged or made it worse.
- **Checks on the code:** ridge regression re-written from scratch in NumPy matches the library to 1e-8, and an automatic GitHub check now runs all 75 tests on every push.
