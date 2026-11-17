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
