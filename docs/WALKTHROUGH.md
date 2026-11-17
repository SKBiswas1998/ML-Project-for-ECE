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
We compared two naive baselines (per-vacancy mean, nearest neighbour) with ridge, polynomial ridge, gradient boosting and a Gaussian process on all four protocols. MAE is in decades and all numbers are from `results/metrics_classical.csv` (vacancy encoding `vac_pct`).
- **Known layouts (supplied split, temperature forward-chaining): the GP wins clearly.** For log10 I the MAE is 0.001 (supplied) and 0.019 (62–78 K), against 0.020 and 0.057 for nearest neighbour, giving skill 1.00 and 0.89. The near-zero supplied error was checked for leakage. There is none: at a fixed vacancy level the curves barely change between neighbouring temperatures, so the supplied test set is close to trivial.
- **Unseen vacancy levels: nothing reliably beats the naive guess.** Holding out 0.8–1.2 %, the best log10 Ion/Ioff MAE is the per-vacancy mean (0.492). Boosting reaches 0.520 (skill 0.10, on MSE) and the GP 0.879 (skill −1.57). Extrapolating to 1.6–2.0 %, every model is off by about 2–4 decades in Ion/Ioff.
- **Ridge and polynomial ridge are poor everywhere for I–V curves** (log10 I MAE 0.82–1.42), because the step-like switching cannot be captured by low-order polynomials.
- **The vacancy encoding (percent vs. number of atoms) makes no difference.** The median MAE ratio across jobs is 1.0.

The lesson: the models learn temperature well but cannot predict a vacancy layout they have not seen. With one nested random layout, the data contain no information about how a *different* set of missing atoms would behave.
