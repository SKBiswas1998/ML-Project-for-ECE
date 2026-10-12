1T'-WTe2 TIFET, edge vacancies, NEGF (RGF) MATLAB code  --  files
================================================================
Put everything in one folder together with your own cal_H_U_Ubar.m,
hamiltonian_TMD_device_H.m and lead_eps.m (they are not included).

Vgs_Ids_edge_vacancy_SCF.m   MAIN, self-consistent NEGF-Poisson (Schrodinger-Poisson).
                             I_DS vs V_TG-V_BG at 300/77/4 K for 0.85 % and 2.0 % edge
                             vacancies + ON/OFF carrier-density maps, width profiles and
                             the Hartree potential. All functions it needs are local
                             functions at the end of the file (MATLAB R2016b or newer).
                             Set quick_test = true for a first run (minutes), then false.
                             Add 0 to vac_list/vac_tag to recompute the pristine ribbon with
                             the same code (your old pristine files were 2q^2/h, no Poisson).

Vgs_Ids_edge_vacancy_T.m     Previous, non-self-consistent version (T(E) once, all T).
                             Faster; use it if the Hartree correction turns out negligible.
Fig_Temp_edge_vacancy.m      Fig13-style plots from the .mat files of Vgs_Ids_edge_vacancy_T.m
                             (expects vacancy levels 0, 0.85 and 2.0).

functions/                   Copies of the local functions of the SCF script as separate
                             .m files (for MATLAB older than R2016b, or to reuse elsewhere):
   rgf_spectra.m        RGF: T(E) and per-orbital a_S, a_D, A (block columns of G)
   carrier_density.m    n_i, p_i from the spectra with the local neutrality level
   poisson_setup.m      3-D finite-difference Poisson solver (grounded gates) + deposition
   point_grid.m         energy lattice of one sweep point (fine part + coarse tails)
   coupling_blocks.m    inter-slice blocks with the removed vacancy orbitals
   landauer_window.m    int T(E)[f_S - f_D] dE  (fine interpolation)
   thermal_dos.m        int D(E) sech^2 kernel  (quantum capacitance)
   discretize_rows.m    atom -> atomic row (width profile)
If you use the separate files, delete the local-function section of the script
(from '%% ===================== Local functions' to the end) or keep both: the
local functions take precedence and nothing breaks.

Outputs of the SCF script: gate_voltage_current_edge_Vacancy_<vac>_temp<T>.mat
(Vgs = lambda_v, V_TG_BG, I, n_atom, p_atom, U_H, n_scf, ...), quantum_capacitance_data_...mat,
edge_vacancy_SCF_..._all.mat (everything incl. atomic positions) and a checkpoint file.
