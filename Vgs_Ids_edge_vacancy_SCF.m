%% Vgs_Ids_edge_vacancy_SCF.m
% 1T'-WTe2 zigzag-nanoribbon TIFET with EDGE vacancies, SELF-CONSISTENT NEGF-Poisson.
%   * I_DS vs (V_TG - V_BG) at 300, 77 and 4 K for 0.85 % and 2.0 % edge vacancies
%     (add 0 to vac_list for a pristine reference computed with the same code)
%   * carrier-density maps and profiles of the ON state (lambda_v = 0) and the
%     OFF state (largest lambda_v), plus the self-consistent Hartree potential
% Needs on the path: cal_H_U_Ubar.m, hamiltonian_TMD_device_H.m, lead_eps.m
% (MATLAB R2016b or newer: implicit expansion and local functions in a script)
%
% Self-consistent loop (boxes A-F of the flow chart):
%   A  U_H = 0 (or the converged potential of the previous sweep point)
%   B  NEGF (recursive Green's function): G^R = [E - H - U - Sig_S - Sig_D]^-1,
%      G^n = G^R (Gam_S f_S + Gam_D f_D) G^A, per-orbital spectra
%      a_S(E) = diag(G Gam_S G^+), a_D(E) = diag(G Gam_D G^+), A(E) = diag[i(G - G^+)]
%   C  electron / hole split at the local neutrality level E_N(i) = E_N0 + U_H(i):
%      n_i = int_{E_N}^{inf} dE/2pi G^n_ii,  p_i = int_{-inf}^{E_N} dE/2pi [A_ii - G^n_ii]
%      (intrinsic channel: rho_i = -q (n_i - p_i); filled states below E_N are neutral)
%   D  Poisson  div(eps grad V) = -rho/eps_0 on a 3-D finite-difference grid:
%      top/bottom gates = grounded planes (Dirichlet), Neumann elsewhere,
%      eps = eps_ch inside the ribbon, eps_ox (HfO2) around it.  U_H = -q V.
%      The gate field itself stays the analytic staggered potential +/-lambda_v on
%      the two Te planes (Laplace part of the same problem), Poisson adds the
%      Hartree part from the channel charge.
%   E  converged when max|U_new - U_H| < tol_U,  F  otherwise linear mixing.
%   The E_F solver of box C is only needed for a non-linear Poisson step; the
%   charge response of the gapped intrinsic channel is weak, so the linear Poisson
%   step with mixing converges in a few iterations.
% Temperature enters through f_S, f_D (and the C_Q kernel), so the loop is run for
% every temperature (scf_reuse_U = true reuses the 300 K potential to save time).
% T(E) and the current I = (q^2/h) int T(E) [f_S - f_D] dE come from the converged
% potential.  Same Hamiltonian, leads, vacancy model, spin prefactor (q^2/h),
% energy grid and file names as Vgs_Ids_edge_vacancy_T.m.
%
% Run order: quick_test = true first (minutes), then the full run. Cost per SCF
% iteration ~ 2.5 x the old T(E)-only sweep (density needs both block columns of
% G), times 2-4 iterations, for every temperature: about a day per vacancy level
% for 20 gate points on a workstation. The run is checkpointed after every point
% (resume = true) and different vac_list values can run in separate MATLAB
% sessions (file names carry the vacancy tags). scf_reuse_U = true (one SCF at
% 300 K, potential reused at 77 and 4 K) roughly halves the time.
clc; clear; close all;
caption = 'wte2';

%% ===================== User settings =====================
T_list      = 77;               % K
vac_list    = [0 0.85 2.0];               % edge vacancy, % of all device atoms (add 0 for a pristine reference)
vac_tag     = {'0','0.85','2.0'};           % used in the file names (same order as vac_list)
seed        = 2026;                     % rerun with other seeds to average over configurations
n_edge_rows = 2;                        % outermost atomic rows per edge that may lose atoms
n_buffer    = 2;                        % vacancy-free slices next to each contact (>= 1)
U_vac       = 1e3;                      % eV, on-site energy of removed (decoupled) orbitals

ZBNR        = 12;                       % 12 -> 9 cells across, W = 5.7 nm (same as the pristine run)
Unit_Cells  = 30;                       % slices along x, L = 10.3 nm

sweep      = 'gate';                    % 'gate' : I_DS vs V_TG-V_BG at fixed V_DS (ON -> OFF)
lambda_pts = linspace(0,1.5,20);        %          lambda_v (eV) = old "Vgs"
Vds_pts    = 0.05;                      %          V
% sweep = 'drain';  lambda_pts = 0;  Vds_pts = 0:0.025:0.30;   % I_DS vs V_DS in the ON state

muS    = 3.05;                          % eV, source (lead 1); drain = muS - V_DS
Ef     = 3.015;                         % eV, equilibrium Fermi level used for C_Q
E_N0   = 3.025;                         % eV, charge-neutrality (mid-gap) level of the pristine ribbon
                                        %     at lambda = 0; must lie inside the gap (T = 2 window)
dE_f   = 2.5e-3;                        % eV, fine energy step (V_DS/dE_f must be an integer)
dE_c   = 7.5e-3;                        % eV, coarse step in the far 300 K tails (multiple of dE_f)
E_fine = 0.15;                          % eV, fine part (density and DOS) beyond mu_D and mu_S
n_kT   = 25;                            % tails kept: n_kT*kB*T beyond mu_D and mu_S (e^-25 ~ 1e-11)
eta    = 1e-6;                          % eV

% ---- electrostatics ----
t_ox   = 14;                            % Angstrom, HfO2 on each side (gives (V_TG-V_BG)/lambda = 1.26706)
eps_ox = 25;                            % HfO2
eps_ch = 1;                             % inside the WTe2 layer (same value as in the gate formula)
hx     = 0;                             % Angstrom, Poisson grid step along x (0 -> a_x/2, atoms on nodes)
hy     = 1.0;                           % Angstrom, along the width
hz     = 1.0;                           % Angstrom, vertical (rounded so that the gates sit on nodes)
pad_y  = 10;                            % Angstrom of oxide beside each ribbon edge (Neumann boundary)

% ---- self-consistency ----
n_scf_max   = 12;                       % NEGF-Poisson iterations per sweep point
tol_U       = 1e-4;                     % eV, convergence of max|U_new - U_H|
mix0        = 0.5;                      % linear mixing (halved automatically if dU grows)
U_margin    = 0.05;                     % eV, density grid always covers E_N0 +/- U_margin
scf_reuse_U = false;                    % true: converge at T_list(1) only, reuse U_H at the other T
cache_sigma = true;                     % keep the lead self-energies in memory (~2*nE*count1^2*16 B, 1.4 GB
                                        % for ZBNR = 12); false: recompute them in every SCF iteration

% ---- run control / output ----
validate         = true;                % one-shot checks of RGF, spectra and Poisson solver
quick_test       = false;               % true: coarse run to test the pipeline first
resume           = true;                % continue from the checkpoint if it matches
T_plot           = 300;                 % K, temperature of the density maps
overlay_pristine = true;                % add the existing no-vacancy curves to the combined I-V plot
pristine_file    = 'gate_voltage_current_temp%d.mat';   % files of Fig13 (fields Vgs = lambda, I)

if quick_test                            % coarse settings to check the pipeline (minutes)
    if strcmp(sweep,'gate'), lambda_pts = linspace(0,1.5,4); else, Vds_pts = Vds_pts(round(linspace(1,end,4))); end
    dE_f = 5e-3;  dE_c = 1.5e-2;  E_fine = 0.1;  n_scf_max = 6;
end
if numel(Vds_pts)    == 1, Vds_pts    = Vds_pts   *ones(size(lambda_pts)); end
if numel(lambda_pts) == 1, lambda_pts = lambda_pts*ones(size(Vds_pts));    end
npts = numel(lambda_pts);   nT = numel(T_list);   nV = numel(vac_list);
run_name  = ['edge_vacancy_SCF_' strjoin(vac_tag,'_') '_' sweep];
ckpt_file = [run_name '_checkpoint.mat'];

%% ===================== Constants =====================
q    = 1.602176634e-19;                 % C
h_pl = 6.62607015e-34;                  % J s
kB   = 8.617333262e-5;                  % eV/K
IE   = q^2/h_pl;                        % A/V; spin is explicit in the SOC basis
eps0 = 8.8541878128e-12/q*1e-10;        % e/(V*Angstrom) = 5.526e-3

%% ===================== Geometry =====================
nn       = (ZBNR-4)/4 + 1;
num_cell = ZBNR - nn;                   % cells across the width
ax = 3.49794;  by = 6.33827;  cz = 15.4319;
unit_cell_length = [ax by cz];

main_unit     = [zeros(num_cell,1), (0:num_cell-1)', zeros(num_cell,1)];   % one slice
neighbor_unit = [ones(num_cell,1),  (0:num_cell-1)', zeros(num_cell,1)];   % next slice

atom_factors = [0.00000044, 1.31473454, 6.35624338;     % 1 Te  (bottom Te plane)
                1.74897061, 4.43335644, 5.73543738;     % 2 Te  (bottom Te plane)
               -0.00000002, 5.44760823, 9.92929379;     % 3 Te  (top Te plane)
                1.74897108, 2.22834683, 9.30776171;     % 4 Te  (top Te plane)
                1.74897050, 6.06754902, 7.93340128;     % 5 W
                0.00000040, 3.81415493, 7.73232446];    % 6 W
norb = [6 6 6 6 10 10];                 % spin-orbitals per atom (Te: p x 2, W: d x 2)
off  = [0 cumsum(norb(1:end-1))];       % orbital offset of each atom inside a cell
nob  = sum(norb);                       % 44 orbitals per cell

n_cell_dev  = num_cell*Unit_Cells;
Device_unit = zeros(n_cell_dev,3);
for i = 1:Unit_Cells
    Device_unit((i-1)*num_cell+(1:num_cell),:) = main_unit + [(i-1) 0 0];
end
n_at = 6*n_cell_dev;
Device_atomic_position = zeros(n_at,3);
for c = 1:n_cell_dev
    Device_atomic_position(6*(c-1)+(1:6),:) = Device_unit(c,:).*unit_cell_length + atom_factors;
end
x_at = Device_atomic_position(:,1);  y_at = Device_atomic_position(:,2);  z_at = Device_atomic_position(:,3);
Length_Device = max(x_at) - min(x_at);  % Angstrom
L_dev = Unit_Cells*ax;                  % Angstrom, length of the N slices (DOS per length)
W_rib = num_cell*by;                    % Angstrom, ribbon width (C_Q per area)
A_atom = ax*by/6;                       % Angstrom^2 per atom (sheet density)

t_wte2      = atom_factors(3,3) - atom_factors(2,3);        % 4.19386 A (outer Te-Te)
gate_factor = (2*t_ox*eps_ch/eps_ox + t_wte2)/t_wte2;       % 1.26706 for t_ox = 14 A
fprintf('W = %.2f nm, L = %.2f nm, %d atoms, (V_TG-V_BG)/lambda = %.5f\n', ...
        W_rib/10, Length_Device/10, n_at, gate_factor);

%% ===================== Pristine Hamiltonian =====================
proj = [6 6 6 6 10 10];  HNN = 5;  Vext = 0;
atomst = repmat(1:6, num_cell, 1);
[bigH,bigU,bigUbar] = cal_H_U_Ubar(caption,main_unit,neighbor_unit,atomst,atomst,proj,HNN,Vext);
count1 = nob*num_cell;   count2 = count1*Unit_Cells;   N = Unit_Cells;
assert(size(bigH,1) == count1, 'bigH is not 44*num_cell: check ZBNR and proj');
assert(n_buffer >= 1, 'n_buffer must be >= 1 (no vacancy in the slices touching the contacts)');
H = hamiltonian_TMD_device_H(bigH,bigU,Unit_Cells);

blk = @(s) (s-1)*count1 + (1:count1);
h0 = full(H(blk(1),blk(1)));            % H_{s,s}
u0 = full(H(blk(1),blk(2)));            % H_{s,s+1}
l0 = full(H(blk(2),blk(1)));            % H_{s+1,s}
rel = @(A,B) norm(A-B,'fro')/max(norm(B,'fro'),realmin);
sm  = ceil(N/2);
assert(rel(full(H(blk(sm),blk(sm))),h0) < 1e-12 && rel(full(H(blk(sm),blk(sm+1))),u0) < 1e-12, ...
       'Device slices are not identical: the RGF below assumes one slice Hamiltonian');
assert(N < 3 || norm(full(H(blk(1),blk(3))),'fro') < 1e-10*norm(u0,'fro'), ...
       'H couples non-adjacent slices: RGF needs nearest-slice coupling');
if rel(h0,bigH) > 1e-10 || rel(u0,bigU) > 1e-10 || rel(l0,bigUbar) > 1e-10
    warning('Device blocks differ from bigH/bigU/bigUbar used for the leads: check the lambda = 0 test.');
end

% gate: +lambda on the bottom Te plane (atoms 1,2), -lambda on the top Te plane (atoms 3,4), 0 on W
chi_cell  = [ones(12,1); -ones(12,1); zeros(20,1)];
chi_slice = repmat(chi_cell, num_cell, 1);

% orbital -> atom map (global orbital index (s-1)*count1 + slice-local index)
cell_of  = ceil((1:n_at)'/6);                   % cell of each atom
type_of  = (1:n_at)' - 6*(cell_of-1);           % 1..6 = Te1 Te2 Te3 Te4 W1 W2
slice_of = Device_unit(cell_of,1) + 1;          % slice along x
atom_of_orb = zeros(count2,1);
for a = 1:n_at
    cl = cell_of(a) - (slice_of(a)-1)*num_cell;                     % cell index inside the slice
    atom_of_orb((slice_of(a)-1)*count1 + (cl-1)*nob + off(type_of(a)) + (1:norb(type_of(a)))) = a;
end
assert(all(atom_of_orb > 0), 'orbital -> atom map incomplete');

%% ===================== Edge vacancies =====================
y_rows = unique(y_at);                           % atomic rows across the width
edge_y = [y_rows(1:n_edge_rows); y_rows(end-n_edge_rows+1:end)];
cand   = find(ismember(y_at,edge_y) & slice_of > n_buffer & slice_of <= N-n_buffer);

rng(seed,'twister');
cand = cand(randperm(numel(cand)));             % one random order -> nested configurations
vac_atoms = cell(1,nV);   rm_all = cell(1,nV);   Uc_all = cell(1,nV);   Lc_all = cell(1,nV);
rm0 = repmat({zeros(1,0)},1,N);                 % no vacancy
atom_name = {'Te1','Te2','Te3','Te4','W1','W2'};
for iv = 1:nV
    Nv = round(vac_list(iv)/100*n_at);
    assert(Nv <= numel(cand), 'Not enough edge sites: increase n_edge_rows');
    vac_atoms{iv} = sort(cand(1:Nv));
    rm = rm0;                                   % removed orbitals of each slice (slice-local index)
    for a = vac_atoms{iv}(:)'
        s = slice_of(a);  t = type_of(a);
        rm{s} = [rm{s}, (cell_of(a)-1)*nob - (s-1)*count1 + off(t) + (1:norb(t))];
    end
    rm_all{iv} = rm;
    [Uc_all{iv}, Lc_all{iv}] = coupling_blocks(u0, l0, rm);    % H_{s,s+1}, H_{s+1,s} without the removed orbitals
    fprintf('vacancy %4s %%: %2d atoms (%.3f %%)', vac_tag{iv}, Nv, 100*Nv/n_at);
    for t = 1:6
        nt = nnz(type_of(vac_atoms{iv}) == t);
        if nt > 0, fprintf('  %s x%d', atom_name{t}, nt); end
    end
    fprintf('\n');
end

%% ===================== Poisson solver =====================
if hx <= 0, hx = ax/2; end
P = poisson_setup(x_at, y_at, z_at, hx, hy, hz, pad_y, t_wte2/2, t_ox, eps_ch, eps_ox, eps0);
fprintf('Poisson grid %d x %d x %d = %d nodes (h = %.2f, %.2f, %.3f A), gates at z_c +/- %.2f A\n', ...
        P.nx, P.ny, P.nz, P.nx*P.ny*P.nz, P.hx, P.hy, P.hz, P.zgate);

%% ===================== One-shot checks =====================
if validate
    % (a) RGF (both block columns) vs full sparse solve, largest vacancy set
    Vds0 = Vds_pts(1);   z = (2*muS - Vds0)/2 + 1i*eta;   lam = lambda_pts(ceil(end/2));
    [g1,~] = lead_eps(bigH,bigUbar,eta,real(z),count1,count2);
    [g2,~] = lead_eps(bigH,bigU,  eta,real(z),count1,count2);
    S1 = bigUbar*g1*bigU;   S2 = bigU*g2*bigUbar;
    rm = rm_all{nV};   pot = lam*repmat(chi_slice,1,N);
    [T_rgf, a1, a2, ad, T_bwd] = rgf_spectra(z, pot, h0, u0, l0, rm, Uc_all{nV}, Lc_all{nV}, S1, S2, U_vac, true);
    rg   = cell2mat(arrayfun(@(s) rm{s} + (s-1)*count1, 1:N, 'UniformOutput', false));
    keep = ones(count2,1);   keep(rg) = 0;   Pk = spdiags(keep,0,count2,count2);
    Hd   = sparse(H) + spdiags(pot(:),0,count2,count2);
    Hd   = Pk*Hd*Pk + sparse(rg,rg,U_vac,count2,count2);
    b1 = blk(1);   bN = blk(N);
    A = z*speye(count2) - Hd;
    A(b1,b1) = A(b1,b1) - S1;   A(bN,bN) = A(bN,bN) - S2;
    RHS = zeros(count2,count1);   RHS(bN,:) = eye(count1);
    X   = A\RHS;                                % last block column of G^r
    T_full = real(trace(1i*(S1-S1')*X(b1,:)*(1i*(S2-S2'))*X(b1,:)'));
    aD_full = real(sum((X*(1i*(S2-S2'))).*conj(X),2));     % diag(G Gam_D G^+) from the full solve
    fprintf('RGF check (%s %% vacancy, E = %.4f eV, lambda = %.3f eV):\n', vac_tag{nV}, real(z), lam);
    fprintf('   T: forward %.6f | backward %.6f | full solve %.6f\n', T_rgf, T_bwd, T_full);
    fprintf('   drain spectral density: max rel. dev. RGF vs full = %.1e\n', ...
            max(abs(a2-aD_full))/max(abs(aD_full)));
    fprintf('   sum rule max|A - a_S - a_D|/max(A) = %.1e (eta contribution)\n', max(abs(ad-a1-a2))/max(ad));
    [Uc0, Lc0] = coupling_blocks(u0, l0, rm0);
    T_0 = rgf_spectra(z, 0*pot, h0, u0, l0, rm0, Uc0, Lc0, S1, S2, U_vac, false);
    fprintf('Pristine, lambda = 0 (device = lead): T = %.6f (should be ~integer, 2 in the QSH window)\n', T_0);
    clear A RHS X Hd Pk
    % (b) Poisson: uniform sheet charge on the W atoms vs the 1-D parallel-plate solution
    q_test = -1e-3;   rho_t = zeros(n_at,1);   rho_t(type_of >= 5) = q_test;
    V_t    = P.solve(rho_t);   V_at = P.W*V_t;
    [~,ac] = min((x_at-mean(x_at)).^2 + (y_at-mean(y_at)).^2 + (type_of < 5)*1e6);   % central W atom
    sigma  = 2*q_test/(ax*by);
    V_1D   = sigma/(2*eps0)*(t_wte2/2/eps_ch + t_ox/eps_ox);
    fprintf(['Poisson check: V(centre W atom) = %.4f V | 1-D parallel-plate limit %.4f V ' ...
             '(finite ribbon width lets the field spread sideways: ~5-10 %% below the 1-D value for W = 5.7 nm)\n'], ...
            V_at(ac), V_1D);
end
clear H

%% ===================== Energy lattice and sweep points =====================
assert(abs(dE_c/dE_f - round(dE_c/dE_f)) < 1e-9, 'dE_c must be a multiple of dE_f');
m_c  = round(dE_c/dE_f);
kTs  = kB*T_list;
mT   = max(n_kT*kTs, U_margin);                            % total tail per temperature (eV)
mF   = min(E_fine, mT);                                    % fine (density) part per temperature
kbias = round(Vds_pts/dE_f);
assert(all(abs(kbias*dE_f - Vds_pts) < 1e-9), 'every V_DS/dE_f must be an integer');
% lattice E = muS + k*dE_f; all grids are subsets -> one lead self-energy cache
k_lo = -max(kbias) - ceil(max(mT)/dE_f) - m_c;   k_hi = ceil(max(mT)/dE_f) + m_c;
Sig1c = cell(1, k_hi-k_lo+1);   Sig2c = Sig1c;              % filled on demand
grid_k = cell(npts,nT,nV);   dens_k = cell(npts,nT,nV);    % lattice indices and density subset of every point
nE_show = zeros(1,nT);   nd_show = nE_show;
for iT = 1:nT
    [kk, kd] = point_grid(kbias(1), mF(iT), mT(iT), dE_f, dE_c);
    nE_show(iT) = numel(kk);   nd_show(iT) = nnz(kd);
end
fprintf('%d sweep points x %d temperatures x %d vacancy levels; energies per point: %s (density part %s)\n', ...
        npts, nT, nV, mat2str(nE_show), mat2str(nd_show));

%% ===================== Storage / checkpoint =====================
run_sig = [seed n_edge_rows n_buffer ZBNR Unit_Cells U_vac eta muS E_N0 dE_f dE_c E_fine n_kT ...
           t_ox eps_ox eps_ch hx hy hz pad_y tol_U U_margin scf_reuse_U double(strcmp(sweep,'drain'))];
Trans  = cell(npts,nT,nV);   DOSd = cell(npts,nT,nV);      % T(E) on grid_k, DOS on the density part
I_all  = zeros(npts,nT,nV);  Cq2_all = zeros(npts,nT,nV);
n_at_all = zeros(n_at,npts,nT,nV);   p_at_all = n_at_all;   U_at_all = n_at_all;
nit_all  = zeros(npts,nT,nV);   dU_all = nit_all;   Q_all = nit_all;
done = false(npts,nT,nV);
if resume && exist(ckpt_file,'file') == 2
    S = load(ckpt_file);
    if isequal(S.run_sig,run_sig) && isequal(S.lambda_pts,lambda_pts) && isequal(S.Vds_pts,Vds_pts) ...
       && isequal(S.T_list,T_list) && isequal(S.vac_list,vac_list)
        Trans = S.Trans;  DOSd = S.DOSd;  grid_k = S.grid_k;  dens_k = S.dens_k;
        I_all = S.I_all;  Cq2_all = S.Cq2_all;  n_at_all = S.n_at_all;  p_at_all = S.p_at_all;
        U_at_all = S.U_at_all;  nit_all = S.nit_all;  dU_all = S.dU_all;  Q_all = S.Q_all;  done = S.done;
        fprintf('Resuming: %d of %d points already done\n', nnz(done), numel(done));
    end
end

%% ===================== Self-consistent NEGF-Poisson sweep =====================
t_all = tic;
for iv = 1:nV
    rm = rm_all{iv};   Uc = Uc_all{iv};   Lc = Lc_all{iv};
    for iT = 1:nT
        kT = kTs(iT);
        for ip = 1:npts
            if done(ip,iT,iv), continue; end
            t_pt = tic;
            lam = lambda_pts(ip);   muD = muS - Vds_pts(ip);   mF_eff = mF(iT);
            % warm start of the Hartree potential
            if scf_reuse_U && iT > 1 && done(ip,1,iv)
                U_at = U_at_all(:,ip,1,iv);   n_it = 1;
            else
                n_it = n_scf_max;
                if ip > 1 && done(ip-1,iT,iv),     U_at = U_at_all(:,ip-1,iT,iv);
                elseif iT > 1 && done(ip,iT-1,iv), U_at = U_at_all(:,ip,iT-1,iv);
                else,                              U_at = zeros(n_at,1);
                end
            end
            pot0 = lam*repmat(chi_slice,1,N);              % gate part, count1 x N
            mix = mix0;   dU_prev = inf;   mF_last = 0;
            for it = 1:n_it
                % density grid must contain E_N0 + U_H: widen the fine part if the potential grew
                mF_eff = max(mF_eff, max(abs(U_at)) + 3*dE_f);
                [kk, kd] = point_grid(kbias(ip), mF_eff, max(mT(iT), mF_eff), dE_f, dE_c);
                E_p = muS + kk*dE_f;   Ed = E_p(kd);   nE_p = numel(kk);   nEd = nnz(kd);
                if mF_eff > mF(iT) && ceil(mF_eff/dE_f) > ceil(mF_last/dE_f)
                    fprintf('   density grid widened to +/- %.3f eV beyond the window (max|U_H| = %.3f eV)\n', mF_eff, max(abs(U_at)));
                end
                mF_last = mF_eff;
                pot = pot0 + reshape(U_at(atom_of_orb), count1, N);
                Tr = zeros(nE_p,1);   A1 = zeros(nEd,count2);   A2 = A1;   AD = A1;   jd = 0;
                for j = 1:nE_p
                    c = kk(j) - k_lo + 1;
                    inc = c >= 1 && c <= numel(Sig1c);
                    if inc && ~isempty(Sig1c{c})
                        Sig1 = Sig1c{c};   Sig2 = Sig2c{c};
                    else
                        [g1,~] = lead_eps(bigH,bigUbar,eta,E_p(j),count1,count2);    % left lead
                        [g2,~] = lead_eps(bigH,bigU,  eta,E_p(j),count1,count2);    % right lead
                        Sig1 = bigUbar*g1*bigU;   Sig2 = bigU*g2*bigUbar;
                        if cache_sigma && inc, Sig1c{c} = Sig1;  Sig2c{c} = Sig2; end
                    end
                    [Tr(j), a1, a2, ad] = rgf_spectra(E_p(j)+1i*eta, pot, h0, u0, l0, rm, Uc, Lc, ...
                                                      Sig1, Sig2, U_vac, kd(j));
                    if kd(j), jd = jd+1;  A1(jd,:) = a1.';  A2(jd,:) = a2.';  AD(jd,:) = ad.';  end
                end
                [n_orb, p_orb] = carrier_density(Ed(:), A1, A2, AD, muS, muD, kT, E_N0 + U_at(atom_of_orb));
                rho_at = accumarray(atom_of_orb, -(n_orb - p_orb), [n_at 1]);   % e per atom
                U_new  = -(P.W*P.solve(rho_at));                                 % eV, U_H = -q V
                dU = max(abs(U_new - U_at));
                if dU < tol_U || it == n_it, break; end
                if dU > dU_prev, mix = mix/2; end
                dU_prev = dU;
                U_at = U_at + mix*(U_new - U_at);
            end
            if dU >= tol_U && n_it > 1
                warning('vac %s %%, T = %d K, point %d: not converged (dU = %.1e eV)', vac_tag{iv}, T_list(iT), ip, dU);
            end
            % store (T(E), spectra and densities belong to the potential U_at)
            grid_k{ip,iT,iv} = kk;   dens_k{ip,iT,iv} = kd;
            Trans{ip,iT,iv} = Tr;
            DOSd{ip,iT,iv}  = sum(AD,2)/(2*pi*L_dev);                % states/(eV*Angstrom), both spins
            I_all(ip,iT,iv) = IE*landauer_window(E_p, Tr, muS, muD, kT, n_kT);
            Cq2_all(ip,iT,iv) = q*1e10*thermal_dos(Ed, DOSd{ip,iT,iv}, Ef, kT, n_kT);   % F/m
            n_at_all(:,ip,iT,iv) = accumarray(atom_of_orb, n_orb, [n_at 1]);
            p_at_all(:,ip,iT,iv) = accumarray(atom_of_orb, p_orb, [n_at 1]);
            U_at_all(:,ip,iT,iv) = U_at;
            nit_all(ip,iT,iv) = it;   dU_all(ip,iT,iv) = dU;   Q_all(ip,iT,iv) = sum(rho_at);
            done(ip,iT,iv) = true;
            save(ckpt_file,'run_sig','lambda_pts','Vds_pts','T_list','vac_list','Trans','DOSd','grid_k','dens_k', ...
                 'I_all','Cq2_all','n_at_all','p_at_all','U_at_all','nit_all','dU_all','Q_all','done');
            el = toc(t_all)/60;   nd = nnz(done);
            fprintf(['vac %4s %% | T = %3d K | pt %2d/%2d (lambda = %.3f eV, V_DS = %.3f V) | SCF %2d it, dU = %.1e eV, ' ...
                     'max|U_H| = %.2f meV, Q = %+.2e e | I = %.3e A | %.1f min (%.0f min left)\n'], ...
                    vac_tag{iv}, T_list(iT), ip, npts, lam, Vds_pts(ip), it, dU, 1e3*max(abs(U_at)), ...
                    Q_all(ip,iT,iv), abs(I_all(ip,iT,iv)), toc(t_pt)/60, el/nd*(numel(done)-nd));
        end
    end
end

%% ===================== Output files =====================
Vgs     = lambda_pts;                   % same meaning as before (old plot scripts multiply by 1.26706)
V_TG_BG = gate_factor*lambda_pts;       % V
Ez      = lambda_pts/t_wte2;            % V/Angstrom (thesis convention)
Vds     = Vds_pts;
if strcmp(sweep,'gate'), xs = V_TG_BG; xlab = 'V_{TG} - V_{BG} (V)'; prefix = 'gate_voltage_current_';
else,                    xs = Vds_pts; xlab = 'V_{DS} (V)';          prefix = 'drain_voltage_current_'; end
for iv = 1:nV
    vac_percent = vac_list(iv);   vac_atoms_used = vac_atoms{iv};
    for iT = 1:nT
        Temp = T_list(iT);   I = I_all(:,iT,iv).';
        Cq2 = Cq2_all(:,iT,iv).';   Cq1 = Cq2/(W_rib*1e-10);      % F/m and F/m^2
        n_atom = n_at_all(:,:,iT,iv);   p_atom = p_at_all(:,:,iT,iv);   U_H = U_at_all(:,:,iT,iv);
        n_scf = nit_all(:,iT,iv).';   dU_scf = dU_all(:,iT,iv).';   Q_channel = Q_all(:,iT,iv).';
        tag = sprintf('edge_Vacancy_%s_temp%d', vac_tag{iv}, Temp);
        save([prefix tag '.mat'], 'Vgs','V_TG_BG','Ez','Vds','I','Length_Device','Temp','vac_percent', ...
             'vac_atoms_used','seed','n_atom','p_atom','U_H','n_scf','dU_scf','Q_channel','E_N0');
        save(['quantum_capacitance_data_' tag '.mat'], 'Vgs','V_TG_BG','Cq1','Cq2','Temp','vac_percent');
    end
end
save([run_name '_all.mat'],'lambda_pts','Vds_pts','V_TG_BG','Ez','T_list','vac_list','vac_tag','vac_atoms', ...
     'seed','muS','Ef','E_N0','I_all','Cq2_all','n_at_all','p_at_all','U_at_all','nit_all','dU_all','Q_all', ...
     'Trans','DOSd','grid_k','dens_k','dE_f','Device_atomic_position','type_of','slice_of','A_atom', ...
     'L_dev','Length_Device','W_rib','n_at','gate_factor','t_ox','eps_ox','eps_ch','n_edge_rows','n_buffer');

%% ===================== Figures: I-V =====================
blue       = [0.000 0.447 0.698];   vermillion = [0.835 0.369 0.000];   green     = [0.000 0.620 0.451];
purple     = [0.459 0.439 0.702];   orange     = [0.902 0.624 0.000];   teal      = [0.000 0.502 0.502];
magenta    = [0.800 0.200 0.600];   brown      = [0.549 0.337 0.294];   nearBlack = [0.150 0.150 0.150];
colV = {blue, vermillion, green;                         % pristine (0 %)  : 300, 77, 4 K
        purple, orange, teal;                            % 0.85 %          : 300, 77, 4 K
        magenta, brown, nearBlack};                      % 2.0 %           : 300, 77, 4 K
colP = colV(1,:);
lsty = {'-','--',':'};   mrk = {'s','o','^'};             % line style -> vacancy level, marker -> temperature
lvl  = zeros(1,nV);                                       % colour/style row of each vacancy level
for iv = 1:nV
    k = find(abs([0 0.85 2.0] - vac_list(iv)) < 1e-9, 1);
    if isempty(k), k = min(iv,3); end
    lvl(iv) = k;
end
tname = arrayfun(@(T) sprintf('%d K',T), T_list, 'UniformOutput', false);
Ids = abs(real(I_all));                                   % abs(real()) before any log axis

for iv = 1:nV                                             % one figure per vacancy level (Fig13 layout)
    figure('Color',[1 1 1]);  hold on;  grid on;  box on;
    ph = [];
    for iT = 1:nT
        ph(end+1) = semilogy(xs, Ids(:,iT,iv), [lsty{lvl(iv)} mrk{iT}], 'Color', colV{lvl(iv),iT}, ...
                          'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    end
    set(gca, 'YScale', 'log');
    xlabel(xlab, 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('I_{DS} (A)', 'FontSize', 20, 'FontWeight', 'bold');
    title(sprintf('%s%% edge vacancy, self-consistent', vac_tag{iv}));
    legend(ph, strcat({'T = '}, tname), 'Location', 'best', 'FontSize', 16);
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'YMinorGrid', 'on', 'XMinorGrid', 'on', 'FontSize', 18);
    set(gcf, 'Position', [100 100 1000 700]);   xlim([min(xs) max(xs)]);
end

figure('Color',[1 1 1]);  hold on;  grid on;  box on;    % all curves together
ph = [];   lab = {};
if overlay_pristine && strcmp(sweep,'gate')
    for iT = 1:nT
        fn = sprintf(pristine_file, T_list(iT));
        if exist(fn,'file') == 2
            S  = load(fn);                                % NOTE: old pristine files: 2q^2/h, no Poisson
            ph(end+1) = semilogy(gate_factor*S.Vgs(:), abs(real(S.I(:))), ['-' mrk{iT}], 'Color', colP{iT}, ...
                                 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
            lab{end+1} = ['pristine, ' tname{iT}];
        end
    end
end
for iv = 1:nV
    for iT = 1:nT
        ph(end+1) = semilogy(xs, Ids(:,iT,iv), [lsty{lvl(iv)} mrk{iT}], 'Color', colV{lvl(iv),iT}, ...
                             'LineWidth', 1.5, 'MarkerFaceColor', 'w');
        lab{end+1} = [vac_tag{iv} '%, ' tname{iT}];
    end
end
set(gca, 'YScale', 'log');
xlabel(xlab, 'FontSize', 20, 'FontWeight', 'bold');
ylabel('I_{DS} (A)', 'FontSize', 20, 'FontWeight', 'bold');
legend(ph, lab, 'Location', 'best', 'FontSize', 13, 'NumColumns', 3);
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'YMinorGrid', 'on', 'XMinorGrid', 'on', 'FontSize', 18);
set(gcf, 'Position', [100 100 1100 750]);   xlim([min(xs) max(xs)]);

%% ===================== Figures: carrier density, ON and OFF state =====================
[~, iTp] = min(abs(T_list - T_plot));   ip_on = 1;   ip_off = npts;
if strcmp(sweep,'gate')
    st_lab = {sprintf('ON state (\\lambda_v = %.2f eV, V_{DS} = %.2f V)', lambda_pts(ip_on),  Vds_pts(ip_on)), ...
              sprintf('OFF state (\\lambda_v = %.2f eV, V_{DS} = %.2f V)', lambda_pts(ip_off), Vds_pts(ip_off))};
else
    st_lab = {sprintf('V_{DS} = %.2f V (\\lambda_v = %.2f eV)', Vds_pts(ip_on),  lambda_pts(ip_on)), ...
              sprintf('V_{DS} = %.2f V (\\lambda_v = %.2f eV)', Vds_pts(ip_off), lambda_pts(ip_off))};
end
to_cm2 = 1e16/A_atom;                                    % carriers per atom -> cm^-2 (sheet density)
for iv = 1:nV
    dens = (n_at_all(:,[ip_on ip_off],iTp,iv) + p_at_all(:,[ip_on ip_off],iTp,iv))*to_cm2;   % n + p
    cmax = max(dens(:));   if cmax <= 0, cmax = 1; end
    figure('Color',[1 1 1]);
    for k = 1:2
        subplot(2,1,k);  hold on;  box on;
        scatter(x_at/10, y_at/10, 28, dens(:,k), 'filled');
        va = vac_atoms{iv};
        if ~isempty(va), plot(x_at(va)/10, y_at(va)/10, 'kx', 'MarkerSize', 7, 'LineWidth', 1.2); end
        axis equal;  xlim([min(x_at) max(x_at)]/10 + [-0.2 0.2]);  ylim([min(y_at) max(y_at)]/10 + [-0.3 0.3]);
        caxis([0 cmax]);  cb = colorbar;  ylabel(cb, 'n + p (cm^{-2})', 'FontSize', 14);
        xlabel('x (nm)', 'FontSize', 16, 'FontWeight', 'bold');  ylabel('y (nm)', 'FontSize', 16, 'FontWeight', 'bold');
        title(sprintf('%s, %s%% edge vacancy, T = %d K', st_lab{k}, vac_tag{iv}, T_list(iTp)), 'FontSize', 14);
        set(gca, 'FontSize', 14);
    end
    set(gcf, 'Position', [100 100 1100 800]);
    fprintf('vac %s %%, T = %d K: carriers in the channel ON = %.3e (n) + %.3e (p) | OFF = %.3e (n) + %.3e (p) electrons\n', ...
            vac_tag{iv}, T_list(iTp), sum(n_at_all(:,ip_on,iTp,iv)), sum(p_at_all(:,ip_on,iTp,iv)), ...
            sum(n_at_all(:,ip_off,iTp,iv)), sum(p_at_all(:,ip_off,iTp,iv)));
end

% profile across the width: carriers per unit channel length in each atomic row
figure('Color',[1 1 1]);  hold on;  grid on;  box on;
ph = [];   lab = {};
for iv = 1:nV
    for k = 1:2
        ip = [ip_on ip_off];   ip = ip(k);
        dl = accumarray(discretize_rows(y_at, y_rows), n_at_all(:,ip,iTp,iv) + p_at_all(:,ip,iTp,iv), [numel(y_rows) 1])/L_dev;
        ph(end+1) = plot(y_rows/10, dl*10, [lsty{lvl(iv)} mrk{k}], 'Color', colV{lvl(iv),k}, 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
        if k == 1, lab{end+1} = [vac_tag{iv} '%, ON']; else, lab{end+1} = [vac_tag{iv} '%, OFF']; end
    end
end
xlabel('y (nm)', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('carriers per nm of channel (nm^{-1})', 'FontSize', 20, 'FontWeight', 'bold');
title(sprintf('Carrier density across the ribbon, T = %d K', T_list(iTp)));
legend(ph, lab, 'Location', 'best', 'FontSize', 14);
set(gca, 'FontSize', 18);   set(gcf, 'Position', [100 100 1000 700]);

% self-consistent Hartree potential of the ON state (map)
figure('Color',[1 1 1]);
for iv = 1:nV
    subplot(nV,1,iv);  hold on;  box on;
    scatter(x_at/10, y_at/10, 28, 1e3*U_at_all(:,ip_on,iTp,iv), 'filled');
    axis equal;  xlim([min(x_at) max(x_at)]/10 + [-0.2 0.2]);  ylim([min(y_at) max(y_at)]/10 + [-0.3 0.3]);
    cb = colorbar;  ylabel(cb, 'U_H (meV)', 'FontSize', 14);
    xlabel('x (nm)', 'FontSize', 16, 'FontWeight', 'bold');  ylabel('y (nm)', 'FontSize', 16, 'FontWeight', 'bold');
    title(sprintf('Hartree potential, %s, %s%% edge vacancy, T = %d K', st_lab{1}, vac_tag{iv}, T_list(iTp)), 'FontSize', 14);
    set(gca, 'FontSize', 14);
end
set(gcf, 'Position', [150 150 1100 800]);

%% ---- numbers for the text ----
fprintf('%-8s %6s %12s %12s %12s %8s %10s\n', 'vac (%)', 'T (K)', 'I_first (A)', 'I_last (A)', 'ratio', 'SCF it', 'max|U_H|');
for iv = 1:nV
    for iT = 1:nT
        fprintf('%-8s %6d %12.4e %12.4e %12.3e %8.1f %7.2f meV\n', vac_tag{iv}, T_list(iT), Ids(1,iT,iv), Ids(end,iT,iv), ...
                Ids(1,iT,iv)/Ids(end,iT,iv), mean(nit_all(:,iT,iv)), 1e3*max(max(abs(U_at_all(:,:,iT,iv)))));
    end
end

%% ===================== Local functions =====================
function [Uc, Lc] = coupling_blocks(u0, l0, rm)
% inter-slice blocks H_{s,s+1} (Uc{s}) and H_{s+1,s} (Lc{s}) with the removed orbitals decoupled
N = numel(rm);   Uc = cell(1,N-1);   Lc = cell(1,N-1);
for s = 1:N-1
    U = u0;   L = l0;
    if ~isempty(rm{s}) || ~isempty(rm{s+1})
        U(rm{s},:) = 0;     U(:,rm{s+1}) = 0;
        L(rm{s+1},:) = 0;   L(:,rm{s}) = 0;
    end
    Uc{s} = U;   Lc{s} = L;
end
end

function [T, a1, a2, Ad, T2] = rgf_spectra(z, pot, h0, u0, l0, rm, Uc, Lc, Sig1, Sig2, U_vac, need)
% Recursive Green's function for N slices with identical Hamiltonian h0 (H_ss), the
% on-site potential pot(:,s) (gate + Hartree), the orbitals rm{s} removed and the
% inter-slice blocks Uc{s} = H_{s,s+1}, Lc{s} = H_{s+1,s}.
%   T  = Tr[Gam1 G_1N Gam2 G_1N^+]              (last block column, forward sweep)
%   T2 = Tr[Gam2 G_N1 Gam1 G_N1^+]              (first block column, backward sweep; = T)
% need = true also returns, for every orbital (global index (s-1)*n + i):
%   a1 = diag(G Gam1 G^+), a2 = diag(G Gam2 G^+)  -> G^n_ii = f1 a1 + f2 a2
%   Ad = diag[i(G - G^+)] = spectral function (a1 + a2 + eta term)
N = numel(rm);   n = size(h0,1);   Id = eye(n);   dg = 1:n+1:n*n;
A0 = -h0;   A0(dg) = A0(dg) + z;                          % z - H_ss
Gam1 = 1i*(Sig1 - Sig1');   Gam2 = 1i*(Sig2 - Sig2');
gL = cell(1,N);
for s = 1:N                                               % left-connected g^L (with Sig1, Sig2 at the ends)
    A = A0;   A(dg) = A(dg) - pot(:,s).';   r = rm{s};
    if ~isempty(r), A(r,:) = 0;  A(:,r) = 0;  A(dg(r)) = z - U_vac; end
    if s == 1, A = A - Sig1; else, A = A - Lc{s-1}*(gL{s-1}*Uc{s-1}); end
    if s == N, A = A - Sig2; end
    gL{s} = A\Id;
end
% last block column G_{s,N} = g^L_s H_{s,s+1} G_{s+1,N}, and the diagonal blocks G_ss
CN = gL{N};   G = gL{N};
if need, a2 = zeros(n,N);  Ad = zeros(n,N);  Ad(:,N) = -2*imag(diag(G));  a2(:,N) = real(sum((CN*Gam2).*conj(CN),2)); end
for s = N-1:-1:1
    CN = gL{s}*(Uc{s}*CN);
    if need
        G = gL{s} + (gL{s}*Uc{s})*G*(Lc{s}*gL{s});
        Ad(:,s) = -2*imag(diag(G));
        a2(:,s) = real(sum((CN*Gam2).*conj(CN),2));
    end
end
T = real(trace(Gam1*CN*Gam2*CN'));                        % CN = G_{1,N}
if ~need, a1 = [];  a2 = [];  Ad = [];  T2 = T;  return; end
gR = cell(1,N);
for s = N:-1:1                                            % right-connected g^R
    B = A0;   B(dg) = B(dg) - pot(:,s).';   r = rm{s};
    if ~isempty(r), B(r,:) = 0;  B(:,r) = 0;  B(dg(r)) = z - U_vac; end
    if s == N, B = B - Sig2; else, B = B - Uc{s}*(gR{s+1}*Lc{s}); end
    if s == 1, B = B - Sig1; end
    gR{s} = B\Id;
end
C1 = gR{1};   a1 = zeros(n,N);   a1(:,1) = real(sum((C1*Gam1).*conj(C1),2));
for s = 2:N                                               % first block column G_{s,1} = g^R_s H_{s,s-1} G_{s-1,1}
    C1 = gR{s}*(Lc{s-1}*C1);
    a1(:,s) = real(sum((C1*Gam1).*conj(C1),2));
end
T2 = real(trace(Gam2*C1*Gam1*C1'));                       % C1 = G_{N,1}
a1 = a1(:);   a2 = a2(:);   Ad = Ad(:);
end

function [n, p] = carrier_density(Ed, A1, A2, AD, muS, muD, kT, EN)
% n_i = int_{E_N,i}^{inf} dE/2pi G^n_ii,  p_i = int_{-inf}^{E_N,i} dE/2pi [A_ii - G^n_ii]
% Ed: uniform energy column (covers E_N +/- U_margin and the Fermi tails); A1, A2, AD: nE x n_orb
f1 = 1./(1 + exp((Ed - muS)/kT));   f2 = 1./(1 + exp((Ed - muD)/kT));
Gn = f1.*A1 + f2.*A2;
Cn = cumtrapz(Ed, Gn)/(2*pi);   CA = cumtrapz(Ed, AD)/(2*pi);
dE = Ed(2) - Ed(1);   nE = numel(Ed);   no = size(A1,2);
k  = floor((EN - Ed(1))/dE) + 1;
assert(all(k >= 1 & k < nE), 'neutrality level E_N0 + U_H outside the density grid: raise U_margin or E_fine');
fr = (EN - Ed(k))/dE;
i1 = sub2ind([nE no], k, (1:no)');
CnE = Cn(i1) + fr.*(Cn(i1+1) - Cn(i1));                   % int_{-inf}^{E_N} G^n
CAE = CA(i1) + fr.*(CA(i1+1) - CA(i1));                   % int_{-inf}^{E_N} A
n = Cn(end,:).' - CnE;
p = CAE - CnE;
end

function P = poisson_setup(x_at, y_at, z_at, hx, hy, hz, pad_y, a_half, t_ox, eps_ch, eps_ox, eps0)
% 3-D finite-difference Poisson solver  div(eps grad V) = -rho/eps0  (V in volt, rho in e, lengths in A)
% grounded gate planes at z_c +/- (a_half + t_ox) (Dirichlet), Neumann on the other faces.
xg = (min(x_at)-hx):hx:(max(x_at)+hx+1e-9);
yg = (min(y_at)-pad_y):hy:(max(y_at)+pad_y+1e-9);
zc = (min(z_at)+max(z_at))/2;   nzh = max(4, round((a_half+t_ox)/hz));   hz = (a_half+t_ox)/nzh;
zg = zc + (-nzh:nzh)*hz;
xg = xg(:);  yg = yg(:);  zg = zg(:);
nx = numel(xg);  ny = numel(yg);  nz = numel(zg);  nn = nx*ny*nz;
[X,Y,Z] = ndgrid(xg,yg,zg);
inlayer = X >= min(x_at)-hx & X <= max(x_at)+hx & Y >= min(y_at)-1 & Y <= max(y_at)+1 & abs(Z-zc) <= a_half+1e-9;
eps_n = eps_ox*ones(nn,1);   eps_n(inlayer(:)) = eps_ch;
isDir = false(nx,ny,nz);   isDir(:,:,1) = true;   isDir(:,:,nz) = true;   isDir = isDir(:);
% operator
idx = reshape(1:nn,[nx ny nz]);
rows = zeros(0,1);  cols = rows;  vals = rows;  diagv = zeros(nn,1);
step = {[1 0 0], hx; [0 1 0], hy; [0 0 1], hz};
for d = 1:3
    sh = step{d,1};  h = step{d,2};
    p  = idx(1:nx-sh(1), 1:ny-sh(2), 1:nz-sh(3));   p = p(:);
    qn = idx(1+sh(1):nx, 1+sh(2):ny, 1+sh(3):nz);   qn = qn(:);
    ef = 0.5*(eps_n(p) + eps_n(qn))/h^2;
    rows = [rows; p; qn];   cols = [cols; qn; p];   vals = [vals; ef; ef];
    diagv = diagv - accumarray(p, ef, [nn 1]) - accumarray(qn, ef, [nn 1]);
end
A = sparse(rows,cols,vals,nn,nn) + spdiags(diagv,0,nn,nn);
keep = double(~isDir);
A = spdiags(keep,0,nn,nn)*A + spdiags(1-keep,0,nn,nn);      % Dirichlet rows: V = 0
[Lf,Uf,Pf,Qf,Rf] = lu(A);
% trilinear deposition / interpolation matrix W (atoms x nodes)
na = numel(x_at);
ix = min(max(floor((x_at-xg(1))/hx)+1,1),nx-1);   fx = (x_at-xg(ix))/hx;
iy = min(max(floor((y_at-yg(1))/hy)+1,1),ny-1);   fy = (y_at-yg(iy))/hy;
iz = min(max(floor((z_at-zg(1))/hz)+1,1),nz-1);   fz = (z_at-zg(iz))/hz;
r = zeros(8*na,1);  c = r;  w = r;   m = 0;
for dx = 0:1, for dy = 0:1, for dz = 0:1
    wgt = (dx*fx + (1-dx)*(1-fx)).*(dy*fy + (1-dy)*(1-fy)).*(dz*fz + (1-dz)*(1-fz));
    r(m+(1:na)) = (1:na)';   c(m+(1:na)) = sub2ind([nx ny nz], ix+dx, iy+dy, iz+dz);   w(m+(1:na)) = wgt;
    m = m + na;
end, end, end
W = sparse(r,c,w,na,nn);
scale = 1/(eps0*hx*hy*hz);
P.W = W;   P.nx = nx;   P.ny = ny;   P.nz = nz;   P.hx = hx;   P.hy = hy;   P.hz = hz;   P.zgate = a_half+t_ox;
P.xg = xg;   P.yg = yg;   P.zg = zg;   P.eps = reshape(eps_n,[nx ny nz]);
P.solve = @(rho_at) Qf*(Uf\(Lf\(Pf*(Rf\(-scale*keep.*(W'*rho_at))))));   % V at the nodes (volt)
end

function [kk, kd] = point_grid(kb, mF, mT, dE_f, dE_c)
% lattice indices k (E = muS + k*dE_f) of one sweep point: fine part mu_D - mF ... mu_S + mF
% (kd = true, density and DOS) and coarse tails out to mT (transmission only)
m_c = round(dE_c/dE_f);   kf = ceil(mF/dE_f);
kk = (-kb-kf):kf;                                          % mu_D and mu_S on the grid
nc = max(0, ceil((mT - kf*dE_f)/dE_c));
kk = [kk(1)-(nc:-1:1)*m_c, kk, kk(end)+(1:nc)*m_c];
kd = (kk >= -kb-kf) & (kk <= kf);
end

function r = discretize_rows(y, rows)
[~, r] = min(abs(y(:) - rows(:).'), [], 2);
end

function J = landauer_window(E, Tr, muS, muD, kT, n_kT)
% J = int T(E)[f_S - f_D] dE (eV); T(E) linearly interpolated on a grid finer than kB*T
lo = max(E(1), muD - n_kT*kT);   hi = min(E(end), muS + n_kT*kT);
de = min(kT/10, min(diff(E))/4);
x  = linspace(lo, hi, max(201, ceil((hi-lo)/de)+1));
fS = 1./(1 + exp((x-muS)/kT));   fD = 1./(1 + exp((x-muD)/kT));
J  = trapz(x, interp1(E(:), Tr(:), x(:)).'.*(fS - fD));
end

function Dth = thermal_dos(E, DOS, Ef, kT, n_kT)
% Dth = int D(E) F_T(E-Ef) dE, F_T = sech^2((E-Ef)/2kT)/(4kT)  ->  C_Q = q^2*Dth
lo = max(E(1), Ef - n_kT*kT);   hi = min(E(end), Ef + n_kT*kT);
de = min(kT/10, min(diff(E))/4);
x  = linspace(lo, hi, max(201, ceil((hi-lo)/de)+1));
FT = sech((x-Ef)/(2*kT)).^2/(4*kT);
Dth = trapz(x, interp1(E(:), DOS(:), x(:)).'.*FT);
end
