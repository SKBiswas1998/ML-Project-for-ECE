"""wte2ml: machine-learning surrogates for the 1T'-WTe2 TIFET NEGF dataset.

Data flow (no backward imports):
    config -> data -> targets -> splits -> models -> evaluate -> run_experiments -> make_figures
"""
