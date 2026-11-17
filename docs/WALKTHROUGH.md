# Walkthrough (stage by stage)

## Stage 0: what is in the data?
We simulated a 5.7 nm × 10.5 nm 1T′-WTe₂ ribbon with NEGF. The grid covers 38 temperatures (4–78 K) × 21 edge-vacancy levels (0–2 %), which gives 798 gate sweeps of 20 points each. Three findings shaped everything that follows:
1. The vacancy levels are **nested**: one random layout, with atoms added level by level.
2. Most curves turn **upward** at the end of the sweep, and the supplied Ioff is the end-point current, not the minimum current.
3. Static power, switching energy and delay are exact formulas of Ion, Ioff and C_eff (V_DS = 0.05 V).

## Stage 1: loading the data safely
`data.py` reads the four CSV files and refuses to continue if anything differs from the facts verified in Stage 0. The checks cover row counts, missing values, the reliability flag, the seed, 20 points per curve on one shared gate grid, and train/test overlap. It also provides the curve matrix, with one row of 20 log-currents per device, and the vacancy layout. The tests check these facts and confirm that broken inputs are rejected.
