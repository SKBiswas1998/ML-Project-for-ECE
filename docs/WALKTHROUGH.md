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
