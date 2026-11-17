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
