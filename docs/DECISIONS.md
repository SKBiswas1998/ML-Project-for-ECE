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
