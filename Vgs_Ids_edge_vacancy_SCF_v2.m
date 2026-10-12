%% Vgs_Ids_edge_vacancy_SCF.m   (version 2)
% 1T'-WTe2 zigzag-nanoribbon TIFET with EDGE vacancies: SELF-CONSISTENT NEGF-Poisson,
% ungated source/drain extensions, I-V, device metrics and band/potential/charge profiles.
% Needs on the path: cal_H_U_Ubar.m, hamiltonian_TMD_device_H.m, lead_eps.m
% (MATLAB R2016b or newer; Parallel Computing Toolbox optional)
%
% Changes with respect to the first SCF version
%  1) Vacancies. Every spin-orbital of a missing atom is DELETED from the device problem:
%     its rows and columns of H_s,s, H_s,s+1, H_s+1,s are zero, it gets no gate term and no
%     Hartree term, carries no charge and no DOS. Nothing is parked at U_vac any more.
%     The validation block compares the result with the full device matrix in which those
%     rows/columns (Hamiltonian + gate) are literally set to zero.
%  2) Source / drain extensions. The n_ungated slices (unit cells) at each end carry no
%     gate term, like the leads. lambda can ramp over n_ramp slices at the two gate edges
%     (0 = abrupt edge). The gate planes of the Poisson problem cover only the gated slices
%     (gate_cover = 'channel'); over the extensions the top/bottom boundary is Neumann.
%  3) Self-consistency. Newton-Poisson step: the local charge response dq_i/dU_i (thermal
%     local DOS, from the same NEGF spectra) is put inside the Poisson operator,
%        [div eps grad - D/eps0] V_new = -(rho - D U)/eps0,   U = -qV.
%     The fixed point is the ordinary NEGF-Poisson solution. Localized vacancy/edge states
%     have a large local DOS; with plain linear mixing their charge sloshes from iteration
%     to iteration and the loop hits n_scf_max at every point (the "never finishes").
%     Newton pins them and converges in a few iterations also for the pristine ribbon.
%  4) a_S = A - a_D (sum rule) instead of a second RGF sweep: half the cost per energy; the
%     eta-broadened weight of weakly coupled states is filled from the source.
%  5) Energies run in a parfor when a parallel pool exists (serial otherwise).
%  6) Outputs: per (T, vacancy) Ion, Ioff, SS, C_Q, C_total, tau_d, DED, SPD
%     (<run_name>_metrics.csv) and, for every sweep point, the per-slice profiles of the
%     band edges, electrostatic potential and net charge (profile_*.mat) with a
%     three-panel figure (source | gated channel | drain) for the ON and OFF states.
%
% Self-consistent loop per sweep point:
%   A  U_H from the previous point (or 0)
%   B  NEGF/RGF at every energy: T(E), a_D(E) = diag(G Gam_D G+), A(E) = diag[i(G - G+)]
%   C  n_i = int_{E_N,i}^inf dE/2pi G^n_ii,  p_i = int_-inf^{E_N,i} dE/2pi [A_ii - G^n_ii],
%      E_N,i = E_N0 + U_H(i);  D_i = int dE/2pi [a_S (-df_S/dE) + a_D (-df_D/dE)]
%   D  Newton-Poisson (3-D finite differences, HfO2 around the ribbon, grounded gate planes)
%   E  converged when max|U_new - U_H| < tol_U, otherwise F: U_H <- U_H + mix (U_new - U_H)
%
% Run quick_test = true first (minutes), then the full run. Cost: ~2 s per energy per
% RGF solve for ZBNR = 12, N = 30 (serial); ~150-270 energies per SCF iteration, 3-6
% iterations per point. The run is checkpointed after every point (resume = true).
clc; clear; close all;
caption = 'wte2';

%% ===================== User settings =====================
T_list      = 300;                      % K, e.g. [300 77 4] or 4:8:77 (then set scf_reuse_U = true)
vac_list    = [0 0.85 2.0];             % edge vacancy, % of all device atoms (0 = pristine reference)
seed        = 2026;                     % vacancy configuration (rerun with other seeds to average)
n_edge_rows = 2;                        % outermost atomic rows per edge that may lose atoms
n_buffer    = 2;                        % vacancy-free slices next to each contact (>= n_ungated + n_ramp + 1)

ZBNR        = 12;                       % 12 -> 9 cells across, W = 5.7 nm
Unit_Cells  = 30;                       % slices (unit cells) along x, L = 10.5 nm
n_ungated   = 1;                        % ungated slices at the source end and at the drain end
n_ramp      = 0;                        % slices over which lambda ramps 0 -> lambda_v at each gate edge
gate_cover  = 'channel';                % Poisson gate planes above 'channel' (gated slices) or 'all'

sweep      = 'gate';                    % 'gate' : I_DS vs V_TG-V_BG at fixed V_DS (ON -> OFF)
lambda_pts = linspace(0,1.5,20);        %          lambda_v (eV) = old "Vgs"
Vds_pts    = 0.05;                      %          V
% sweep = 'drain';  lambda_pts = 0;  Vds_pts = 0:0.025:0.30;   % I_DS vs V_DS in the ON state

muS    = 3.05;                          % eV, source (lead 1); drain = muS - V_DS
Ef     = 3.015;                         % eV, equilibrium Fermi level used for C_Q
E_N0   = 3.025;                         % eV, charge-neutrality (mid-gap) level of the pristine ribbon
dE_f   = 2.5e-3;                        % eV, fine energy step (V_DS/dE_f must be an integer)
dE_c   = 7.5e-3;                        % eV, coarse step in the far tails (multiple of dE_f)
E_fine = 0.15;                          % eV, fine part (density and DOS) beyond mu_D and mu_S
n_kT   = 25;                            % tails kept: n_kT*kB*T beyond mu_D and mu_S
eta    = 1e-6;                          % eV

% ---- electrostatics ----
t_ox   = 14;                            % Angstrom, HfO2 on each side ((V_TG-V_BG)/lambda = 1.26706)
eps_ox = 25;                            % HfO2
eps_ch = 1;                             % inside the WTe2 layer (same value as in the gate formula)
hx     = 0;                             % Angstrom, Poisson grid step along x (0 -> a_x/2, atoms on nodes)
hy     = 1.0;                           % Angstrom, along the width
hz     = 1.0;                           % Angstrom, vertical (rounded so that the gates sit on nodes)
pad_y  = 10;                            % Angstrom of oxide beside each ribbon edge

% ---- self-consistency ----
n_scf_max   = 12;                       % NEGF-Poisson iterations per sweep point
tol_U       = 1e-4;                     % eV, convergence of max|U_new - U_H|
mix0        = 1.0;                      % 1 = full Newton step (halved automatically if dU grows)
dU_cap      = 0.2;                      % eV, largest change of U_H per iteration
U_margin    = 0.05;                     % eV, density grid always covers E_N0 +/- U_margin
mF_max      = 0.6;                      % eV, the run stops if |U_H| would need a wider density grid
scf_reuse_U = false;                    % true: converge at T_list(1) only, reuse U_H at the other T
cache_sigma = true;                     % keep lead self-energies in memory (~1.4 GB at 300 K, ZBNR = 12)

% ---- device metrics ----
cap_model = 'parallel';                 % 'parallel': 2*C_ox in series with C_Q; 'series': C_ox, C_Q, C_ox in series

% ---- run control / output ----
use_parfor = true;                      % energies in a parfor if a parallel pool can be opened
validate   = true;                      % one-shot checks (RGF, Poisson, Newton solver, band edges)
quick_test = false;                     % true: coarse run to test the pipeline first
resume     = true;                      % continue from the checkpoint if it matches
T_plot     = 300;                       % K, temperature of the maps and profiles
nk_band    = 41;                        % k points for the ribbon band edges of the profiles

if quick_test                           % coarse settings to check the pipeline (minutes)
    if strcmp(sweep,'gate'), lambda_pts = linspace(0,1.5,4); else, Vds_pts = Vds_pts(round(linspace(1,end,4))); end
    dE_f = 5e-3;  dE_c = 1.5e-2;  E_fine = 0.1;  n_scf_max = 6;
end
if numel(Vds_pts)    == 1, Vds_pts    = Vds_pts   *ones(size(lambda_pts)); end
if numel(lambda_pts) == 1, lambda_pts = lambda_pts*ones(size(Vds_pts));    end
npts = numel(lambda_pts);   nT = numel(T_list);   nV = numel(vac_list);
vac_tag  = arrayfun(@(v) sprintf('%g',v), vac_list, 'UniformOutput', false);
run_name = ['edge_vacancy_SCF2_' strjoin(vac_tag,'_') '_' sweep];
if quick_test, run_name = [run_name '_quick']; end
ckpt_file = [run_name '_checkpoint.mat'];
V_DD = Vds_pts(1);                      % supply voltage of tau_d, DED, SPD

%% ===================== Constants =====================
q    = 1.602176634e-19;                 % C
h_pl = 6.62607015e-34;                  % J s
kB   = 8.617333262e-5;                  % eV/K
IE   = q^2/h_pl;                        % A/V; spin is explicit in the SOC basis
eps0_SI = 8.8541878128e-12;             % F/m
eps0 = eps0_SI/q*1e-10;                 % e/(V*Angstrom) = 5.526e-3

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

N = Unit_Cells;
n_cell_dev  = num_cell*N;
Device_unit = zeros(n_cell_dev,3);
for i = 1:N
    Device_unit((i-1)*num_cell+(1:num_cell),:) = main_unit + [(i-1) 0 0];
end
n_at = 6*n_cell_dev;
Device_atomic_position = zeros(n_at,3);
for c = 1:n_cell_dev
    Device_atomic_position(6*(c-1)+(1:6),:) = Device_unit(c,:).*unit_cell_length + atom_factors;
end
x_at = Device_atomic_position(:,1);  y_at = Device_atomic_position(:,2);  z_at = Device_atomic_position(:,3);
Length_Device = max(x_at) - min(x_at);  % Angstrom
L_dev = N*ax;                           % Angstrom, length of the N slices (DOS per length)
W_rib = num_cell*by;                    % Angstrom, ribbon width
A_atom = ax*by/6;                       % Angstrom^2 per atom (sheet density)

t_wte2      = atom_factors(3,3) - atom_factors(2,3);        % 4.19386 A (outer Te-Te)
gate_factor = (2*t_ox*eps_ch/eps_ox + t_wte2)/t_wte2;       % 1.26706 for t_ox = 14 A

% slices along x: slice s holds atoms at x = (s-1)*ax + [0, dx_in]
dx_in   = max(atom_factors(:,1));                            % 1.749 A
xb      = @(s) (s-1)*ax - (ax - dx_in)/2;                    % boundary between slices s-1 and s
x_slice = ((1:N)-1)*ax + dx_in/2;                            % slice centres (Angstrom)
assert(n_ungated >= 0 && n_ramp >= 0 && 2*(n_ungated+n_ramp) < N-2, 'too many ungated/ramp slices');
g_slice = zeros(1,N);   g_slice(n_ungated+1:N-n_ungated) = 1;        % gate profile along x
for r = 1:n_ramp
    g_slice(n_ungated+r) = r/(n_ramp+1);   g_slice(N-n_ungated+1-r) = r/(n_ramp+1);
end
x_gate = [xb(n_ungated+1), xb(N-n_ungated+1)];              % gate metal covers the gated slices
L_g    = x_gate(2) - x_gate(1);                              % gated length (Angstrom)
if strcmp(gate_cover,'all'), x_gate_P = [-inf inf]; else, x_gate_P = x_gate; end
fprintf('W = %.2f nm, L = %.2f nm (gated %.2f nm, %d ungated slices per end, ramp %d), %d atoms, (V_TG-V_BG)/lambda = %.5f\n', ...
        W_rib/10, L_dev/10, L_g/10, n_ungated, n_ramp, n_at, gate_factor);

%% ===================== Pristine Hamiltonian =====================
proj = [6 6 6 6 10 10];  HNN = 5;  Vext = 0;
atomst = repmat(1:6, num_cell, 1);
[bigH,bigU,bigUbar] = cal_H_U_Ubar(caption,main_unit,neighbor_unit,atomst,atomst,proj,HNN,Vext);
count1 = nob*num_cell;   count2 = count1*N;
assert(size(bigH,1) == count1, 'bigH is not 44*num_cell: check ZBNR and proj');
assert(n_buffer >= n_ungated + n_ramp + 1, 'n_buffer must be >= n_ungated + n_ramp + 1');
H = hamiltonian_TMD_device_H(bigH,bigU,N);

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
gate_mat  = chi_slice(:)*g_slice;                 % count1 x N; gate term = lambda*gate_mat

% orbital -> atom map (global orbital index (s-1)*count1 + slice-local index)
cell_of  = ceil((1:n_at)'/6);                   % cell of each atom
type_of  = (1:n_at)' - 6*(cell_of-1);           % 1..6 = Te1 Te2 Te3 Te4 W1 W2
slice_of = Device_unit(cell_of,1) + 1;          % slice along x
atom_of_orb = zeros(count2,1);
for a = 1:n_at
    atom_of_orb((cell_of(a)-1)*nob + off(type_of(a)) + (1:norb(type_of(a)))) = a;
end
assert(all(atom_of_orb > 0), 'orbital -> atom map incomplete');

%% ===================== Edge vacancies =====================
y_rows = unique(y_at);                           % atomic rows across the width
edge_y = [y_rows(1:n_edge_rows); y_rows(end-n_edge_rows+1:end)];
s_free = max(n_buffer, n_ungated + n_ramp);      % vacancy-free slices at each end
cand   = find(ismember(y_at,edge_y) & slice_of > s_free & slice_of <= N-s_free);

rng(seed,'twister');
cand = cand(randperm(numel(cand)));             % one random order -> nested configurations
vac_atoms = cell(1,nV);   vac_orb = cell(1,nV);   kp_all = cell(1,nV);   n_vac = zeros(1,nV);
atom_name = {'Te1','Te2','Te3','Te4','W1','W2'};
for iv = 1:nV
    Nv = round(vac_list(iv)/100*n_at);
    assert(Nv <= numel(cand), 'Not enough edge sites: increase n_edge_rows');
    vac_atoms{iv} = sort(cand(1:Nv));
    g = find(ismember(atom_of_orb, vac_atoms{iv})).';     % global indices of ALL spin-orbitals of the missing atoms
    assert(numel(g) == sum(norb(type_of(vac_atoms{iv}))), 'vacancy -> orbital map is wrong');
    vac_orb{iv} = g;
    s_of = ceil(g/count1);                       % slice of each deleted orbital
    kp = cell(1,N);                              % orbitals KEPT in each slice (slice-local index)
    for s = 1:N
        kp{s} = setdiff(1:count1, g(s_of == s) - (s-1)*count1);
    end
    assert(numel(kp{1}) == count1 && numel(kp{N}) == count1, 'vacancy in a contact slice');
    kp_all{iv} = kp;   n_vac(iv) = Nv;
    fprintf('vacancy %4s %%: %2d atoms (%.3f %%), %3d orbitals deleted (H = 0, gate = 0)', ...
            vac_tag{iv}, Nv, 100*Nv/n_at, numel(g));
    for t = 1:6
        nt = nnz(type_of(vac_atoms{iv}) == t);
        if nt > 0, fprintf('  %s x%d', atom_name{t}, nt); end
    end
    fprintf('\n');
end
kp0 = repmat({1:count1},1,N);                   % pristine device

%% ===================== Poisson solver =====================
if hx <= 0, hx = ax/2; end
P = poisson_setup(x_at, y_at, z_at, hx, hy, hz, pad_y, t_wte2/2, t_ox, eps_ch, eps_ox, eps0, x_gate_P);
fprintf('Poisson grid %d x %d x %d = %d nodes (h = %.2f, %.2f, %.3f A), gates at z_c +/- %.2f A, x = %.2f .. %.2f nm\n', ...
        P.nx, P.ny, P.nz, P.nx*P.ny*P.nz, P.hx, P.hy, P.hz, P.zgate, max(x_gate_P(1),P.xg(1))/10, min(x_gate_P(2),P.xg(end))/10);

%% ===================== One-shot checks =====================
if validate
    % (a) RGF with deleted orbitals vs the full device matrix with zero rows/columns (H + gate)
    Vds0 = Vds_pts(1);   z = (2*muS - Vds0)/2 + 1i*eta;   lam = lambda_pts(ceil(end/2));
    [g1,~] = lead_eps(bigH,bigUbar,eta,real(z),count1,count2);
    [g2,~] = lead_eps(bigH,bigU,  eta,real(z),count1,count2);
    S1 = bigUbar*g1*bigU;   S2 = bigU*g2*bigUbar;   G1 = 1i*(S1-S1');   G2 = 1i*(S2-S2');
    pot = lam*gate_mat;   v = vac_orb{nV};
    [T_rgf, a1, a2, ad] = rgf_spectra(z, pot, h0, u0, l0, kp_all{nV}, S1, S2, true);
    keep = ones(count2,1);   keep(v) = 0;   Pk = spdiags(keep,0,count2,count2);
    Hd = Pk*(sparse(H) + spdiags(pot(:),0,count2,count2))*Pk;
    assert(nnz(Hd(v,:)) == 0 && nnz(Hd(:,v)) == 0, 'vacancy rows/columns of H + gate are not zero');
    b1 = blk(1);   bN = blk(N);
    A = z*speye(count2) - Hd;
    A(b1,b1) = A(b1,b1) - S1;   A(bN,bN) = A(bN,bN) - S2;
    RHS = zeros(count2,2*count1);   RHS(bN,1:count1) = eye(count1);   RHS(b1,count1+(1:count1)) = eye(count1);
    X  = A\RHS;   XN = X(:,1:count1);   X1 = X(:,count1+(1:count1));   % last and first block columns of G^r
    T_full  = real(trace(G1*XN(b1,:)*G2*XN(b1,:)'));
    aD_full = real(sum((XN*G2).*conj(XN),2));   aS_full = real(sum((X1*G1).*conj(X1),2));
    fprintf('RGF check (%s %% vacancy, E = %.4f eV, lambda = %.3f eV):\n', vac_tag{nV}, real(z), lam);
    fprintf('   T: orbitals deleted %.8f | full matrix, rows/cols + gate = 0: %.8f (rel. diff %.1e)\n', ...
            T_rgf, T_full, abs(T_rgf-T_full)/max(abs(T_full),realmin));
    fprintf('   a_D: max rel. dev. RGF vs full = %.1e | a_S (sum rule) vs explicit = %.1e of max | A - a_S - a_D = %.1e of max (eta part)\n', ...
            max(abs(a2-aD_full))/max(abs(aD_full)), max(abs(a1-aS_full))/max(abs(aS_full)), ...
            max(abs(ad-aS_full-aD_full))/max(ad));
    fprintf('   deleted orbitals: max(A) = %.1e, max|U rows of H| = %d  (both must be 0)\n', max([ad(v); 0]), nnz(Hd(v,:)));
    T_0 = rgf_spectra(z, 0*pot, h0, u0, l0, kp0, S1, S2, false);
    fprintf('Pristine, lambda = 0 (device = lead): T = %.6f (should be ~integer, 2 in the QSH window)\n', T_0);
    clear A RHS X XN X1 Hd Pk
    % (b) Poisson: uniform sheet charge on the W atoms vs the 1-D parallel-plate solution
    q_test = -1e-3;   rho_t = zeros(n_at,1);   rho_t(type_of >= 5) = q_test;
    V_t = P.solve(rho_t);   V_at = P.W*V_t;
    [~,ac] = min((x_at-mean(x_at)).^2 + (y_at-mean(y_at)).^2 + (type_of < 5)*1e6);   % central W atom
    sigma = 2*q_test/(ax*by);
    V_1D  = sigma/(2*eps0)*(t_wte2/2/eps_ch + t_ox/eps_ox);
    fprintf('Poisson check: V(centre W atom) = %.4f V | 1-D parallel-plate limit %.4f V (finite width: ~5-10 %% below)\n', V_at(ac), V_1D);
    % (c) Newton-Poisson solver with a uniform response D = 0.05 e/eV per atom
    D_t = 0.05*ones(n_at,1);   t_n = tic;
    [V_n, info] = poisson_newton(P, rho_t, D_t, zeros(n_at,1));
    r_n = P.A*V_n - P.scale*(P.keep.*(P.W'*(D_t.*(P.W*V_n)))) + P.scale*(P.keep.*(P.W'*rho_t));
    fprintf('Newton-Poisson solver: gmres flag %d, %d iterations, residual %.1e, %.1f s | V(centre) %.4f V (%.4f V without response)\n', ...
            info(1), info(2), norm(r_n)/max(norm(P.scale*(P.W'*rho_t)),realmin), toc(t_n), (P.W(ac,:)*V_n), V_at(ac));
    % (d) ribbon band edges at lambda = 0 and at the largest lambda
    [Ec_t, Ev_t] = ribbon_band_edges(bigH, bigU, chi_slice, [0 max(lambda_pts)], E_N0, nk_band);
    fprintf('Ribbon band edges (edge states included), E_N0 = %.4f eV: lambda = 0: Ev = %.4f, Ec = %.4f eV (gap %.1f meV) | lambda = %.2f: Ev = %.4f, Ec = %.4f eV (gap %.1f meV)\n', ...
            E_N0, Ev_t(1), Ec_t(1), 1e3*(Ec_t(1)-Ev_t(1)), max(lambda_pts), Ev_t(2), Ec_t(2), 1e3*(Ec_t(2)-Ev_t(2)));
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
% lattice E = muS + k*dE_f; every grid is a subset -> one lead self-energy cache
k_lo = -max(kbias) - ceil(max(max(mT),mF_max)/dE_f) - m_c;   k_hi = ceil(max(max(mT),mF_max)/dE_f) + m_c;
nC = k_hi - k_lo + 1;   E_lat = muS + (k_lo:k_hi)*dE_f;    % row
Sig1c = cell(nC,1);   Sig2c = cell(nC,1);                  % filled on demand
nE_show = zeros(1,nT);   nd_show = nE_show;
for iT = 1:nT
    [kk, kd] = point_grid(kbias(1), mF(iT), mT(iT), dE_f, dE_c);
    nE_show(iT) = numel(kk);   nd_show(iT) = nnz(kd);
end
fprintf('%d sweep points x %d temperatures x %d vacancy levels; energies per SCF iteration: %s (density part %s)\n', ...
        npts, nT, nV, mat2str(nE_show), mat2str(nd_show));

M = 0;                                                     % 0 -> parfor runs serially on the client
if use_parfor
    try, pool = gcp;  M = pool.NumWorkers; catch, M = 0; end
end
fprintf('Workers: %d\n', M);

%% ===================== Storage / checkpoint =====================
run_sig = [seed n_edge_rows n_buffer ZBNR N n_ungated n_ramp strcmp(gate_cover,'all') eta muS E_N0 dE_f dE_c ...
           E_fine n_kT t_ox eps_ox eps_ch hx hy hz pad_y tol_U U_margin dU_cap scf_reuse_U strcmp(sweep,'drain')];
Trans  = cell(npts,nT,nV);   DOSd = cell(npts,nT,nV);      % T(E) on grid_k, DOS on the density part
grid_k = cell(npts,nT,nV);   dens_k = cell(npts,nT,nV);
I_all  = zeros(npts,nT,nV);  Cq2_all = zeros(npts,nT,nV);
n_at_all = zeros(n_at,npts,nT,nV);   p_at_all = n_at_all;   U_at_all = n_at_all;   D_at_all = n_at_all;
nit_all  = zeros(npts,nT,nV);   dU_all = nit_all;   Q_all = nit_all;
done = false(npts,nT,nV);
if resume && exist(ckpt_file,'file') == 2
    S = load(ckpt_file);
    if isequal(S.run_sig,run_sig) && isequal(S.lambda_pts,lambda_pts) && isequal(S.Vds_pts,Vds_pts) ...
       && isequal(S.T_list,T_list) && isequal(S.vac_list,vac_list)
        Trans = S.Trans;  DOSd = S.DOSd;  grid_k = S.grid_k;  dens_k = S.dens_k;
        I_all = S.I_all;  Cq2_all = S.Cq2_all;  n_at_all = S.n_at_all;  p_at_all = S.p_at_all;
        U_at_all = S.U_at_all;  D_at_all = S.D_at_all;  nit_all = S.nit_all;  dU_all = S.dU_all;
        Q_all = S.Q_all;  done = S.done;
        fprintf('Resuming: %d of %d points already done\n', nnz(done), numel(done));
    else
        fprintf('Checkpoint %s does not match the settings: starting from scratch\n', ckpt_file);
    end
end

%% ===================== Self-consistent NEGF-Poisson sweep =====================
t_all = tic;
for iv = 1:nV
    kp = kp_all{iv};
    for iT = 1:nT
        kT = kTs(iT);   kT_D = max(kT, dE_f);              % broadening of the charge response (Jacobian only)
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
            pot0 = lam*gate_mat;                           % gate part, count1 x N (zero on the ungated slices)
            mix = mix0;   dU_prev = inf;   mF_last = mF(iT);   ginfo = [0 0];
            for it = 1:n_it
                % the density grid must contain E_N0 + U_H: widen the fine part if the potential grew
                mF_eff = max(mF_eff, max(abs(U_at)) + 3*dE_f);
                assert(mF_eff <= mF_max, 'max|U_H| = %.2f eV: Hartree potential out of range (raise mF_max or lower mix0 / dU_cap)', max(abs(U_at)));
                [kk, kd] = point_grid(kbias(ip), mF_eff, max(mT(iT), mF_eff), dE_f, dE_c);
                if ceil(mF_eff/dE_f) > ceil(mF_last/dE_f)
                    fprintf('   density grid widened to +/- %.3f eV beyond the window (max|U_H| = %.3f eV)\n', mF_eff, max(abs(U_at)));
                end
                mF_last = mF_eff;
                c_idx = kk - k_lo + 1;   assert(all(c_idx >= 1 & c_idx <= nC), 'energy lattice too small');
                need = false(nC,1);   need(c_idx) = true;
                dens = false(nC,1);   dens(c_idx(kd)) = true;
                pot  = pot0 + reshape(U_at(atom_of_orb), count1, N);      % gate + Hartree on every orbital
                Tr_c = zeros(nC,1);   A1_c = zeros(nC,count2);   A2_c = zeros(nC,count2);   AD_c = zeros(nC,count2);
                parfor (c = 1:nC, M)                      % energies are independent
                    if need(c)
                        if isempty(Sig1c{c})
                            [g1,~] = lead_eps(bigH,bigUbar,eta,E_lat(c),count1,count2);    % left lead
                            [g2,~] = lead_eps(bigH,bigU,  eta,E_lat(c),count1,count2);    % right lead
                            S1 = bigUbar*g1*bigU;   S2 = bigU*g2*bigUbar;
                            if cache_sigma, Sig1c{c} = S1;  Sig2c{c} = S2; end
                        else
                            S1 = Sig1c{c};   S2 = Sig2c{c};
                        end
                        [Tc, a1, a2, ad] = rgf_spectra(E_lat(c)+1i*eta, pot, h0, u0, l0, kp, S1, S2, dens(c));
                        Tr_c(c) = Tc;   A1_c(c,:) = a1.';   A2_c(c,:) = a2.';   AD_c(c,:) = ad.';
                    else
                        Tr_c(c) = 0;   A1_c(c,:) = zeros(1,count2);   A2_c(c,:) = zeros(1,count2);   AD_c(c,:) = zeros(1,count2);
                    end
                end
                E_p = E_lat(c_idx);   Tr = Tr_c(c_idx);
                cdn = c_idx(kd);   Ed = E_lat(cdn).';
                [n_orb, p_orb, D_orb] = carrier_density(Ed, A1_c(cdn,:), A2_c(cdn,:), AD_c(cdn,:), muS, muD, kT, ...
                                                        E_N0 + U_at(atom_of_orb), kT_D);
                rho_at = accumarray(atom_of_orb, -(n_orb - p_orb), [n_at 1]);   % e per atom (0 on vacancies)
                D_at   = accumarray(atom_of_orb, D_orb, [n_at 1]);              % e/eV per atom, >= 0
                [V_n, ginfo] = poisson_newton(P, rho_at, D_at, U_at);
                U_new = -(P.W*V_n);                                             % eV, U_H = -q V
                assert(all(isfinite(U_new)), 'Poisson solve returned non-finite values');
                dU = max(abs(U_new - U_at));
                if dU < tol_U || it == n_it, break; end
                if dU > dU_prev, mix = mix/2; end
                dU_prev = dU;
                U_at = U_at + mix*max(min(U_new - U_at, dU_cap), -dU_cap);
            end
            if dU >= tol_U && n_it > 1
                warning('vac %s %%, T = %g K, point %d: not converged (dU = %.1e eV)', vac_tag{iv}, T_list(iT), ip, dU);
            end
            % store (T(E), spectra and densities belong to the potential U_at)
            grid_k{ip,iT,iv} = kk;   dens_k{ip,iT,iv} = kd;
            Trans{ip,iT,iv} = Tr;
            DOSd{ip,iT,iv}  = sum(AD_c(cdn,:),2)/(2*pi*L_dev);                   % states/(eV*Angstrom), both spins
            I_all(ip,iT,iv) = IE*landauer_window(E_p, Tr, muS, muD, kT, n_kT);
            Cq2_all(ip,iT,iv) = q*1e10*thermal_dos(Ed, DOSd{ip,iT,iv}, Ef, kT, n_kT);   % F/m
            n_at_all(:,ip,iT,iv) = accumarray(atom_of_orb, n_orb, [n_at 1]);
            p_at_all(:,ip,iT,iv) = accumarray(atom_of_orb, p_orb, [n_at 1]);
            U_at_all(:,ip,iT,iv) = U_at;   D_at_all(:,ip,iT,iv) = D_at;
            nit_all(ip,iT,iv) = it;   dU_all(ip,iT,iv) = dU;   Q_all(ip,iT,iv) = sum(rho_at);
            done(ip,iT,iv) = true;
            save(ckpt_file,'run_sig','lambda_pts','Vds_pts','T_list','vac_list','Trans','DOSd','grid_k','dens_k', ...
                 'I_all','Cq2_all','n_at_all','p_at_all','U_at_all','D_at_all','nit_all','dU_all','Q_all','done');
            el = toc(t_all)/60;   nd = nnz(done);
            fprintf(['vac %4s %% | T = %3g K | pt %2d/%2d (lambda = %.3f eV, V_DS = %.3f V) | SCF %2d it, dU = %.1e eV, ' ...
                     'max|U_H| = %.1f meV, Q = %+.2e e, max D = %.2f e/eV, gmres %d it | I = %.3e A | %.1f min (~%.0f min left)\n'], ...
                    vac_tag{iv}, T_list(iT), ip, npts, lam, Vds_pts(ip), it, dU, 1e3*max(abs(U_at)), ...
                    Q_all(ip,iT,iv), max(D_at), ginfo(2), abs(I_all(ip,iT,iv)), toc(t_pt)/60, el/nd*(numel(done)-nd));
        end
    end
end

%% ===================== Band edges of the gated ribbon (for the profiles) =====================
lam_slice = g_slice(:)*lambda_pts(:).';                    % N x npts, lambda of every slice at every point
lu_all    = unique(round(lam_slice(:),10));
t_b = tic;
[Ec_all, Ev_all] = ribbon_band_edges(bigH, bigU, chi_slice, lu_all, E_N0, nk_band);
[~, loc] = ismember(round(lam_slice,10), lu_all);
Ec_slice = reshape(Ec_all(loc), N, npts);   Ev_slice = reshape(Ev_all(loc), N, npts);   % eV, pristine ribbon, no Hartree
fprintf('Ribbon band edges for %d lambda values: %.0f s\n', numel(lu_all), toc(t_b));

%% ===================== Output files =====================
Vgs     = lambda_pts;                   % same meaning as before (old plot scripts multiply by 1.26706)
V_TG_BG = gate_factor*lambda_pts;       % V
Ez      = lambda_pts/t_wte2;            % V/Angstrom (thesis convention)
Vds     = Vds_pts;
if strcmp(sweep,'gate'), xs = V_TG_BG; xlab = 'V_{TG} - V_{BG} (V)'; prefix = 'gate_voltage_current_';
else,                    xs = Vds_pts; xlab = 'V_{DS} (V)';          prefix = 'drain_voltage_current_'; end
Umean_all = zeros(N,npts,nT,nV);   Umin_all = Umean_all;   Umax_all = Umean_all;   Qs_all = Umean_all;
for iv = 1:nV
    vac_percent = vac_list(iv);   vac_atoms_used = vac_atoms{iv};
    present = true(n_at,1);   present(vac_atoms_used) = false;
    for iT = 1:nT
        Temp = T_list(iT);   I = I_all(:,iT,iv).';
        Cq2 = Cq2_all(:,iT,iv).';   Cq1 = Cq2/(W_rib*1e-10);      % F/m and F/m^2
        n_atom = n_at_all(:,:,iT,iv);   p_atom = p_at_all(:,:,iT,iv);   U_H = U_at_all(:,:,iT,iv);
        n_scf = nit_all(:,iT,iv).';   dU_scf = dU_all(:,iT,iv).';   Q_channel = Q_all(:,iT,iv).';
        tag = sprintf('edge_Vacancy_%s_temp%g', vac_tag{iv}, Temp);
        save([prefix tag '.mat'], 'Vgs','V_TG_BG','Ez','Vds','I','Length_Device','Temp','vac_percent', ...
             'vac_atoms_used','seed','n_atom','p_atom','U_H','n_scf','dU_scf','Q_channel','E_N0','g_slice');
        save(['quantum_capacitance_data_' tag '.mat'], 'Vgs','V_TG_BG','Cq1','Cq2','Temp','vac_percent');
        % per-slice profiles (source extension | gated channel | drain extension) of every sweep point
        for ip = 1:npts
            Pr = slice_profile(U_H(:,ip), p_atom(:,ip) - n_atom(:,ip), slice_of, present, N);
            Umean_all(:,ip,iT,iv) = Pr.Umean;   Umin_all(:,ip,iT,iv) = Pr.Umin;
            Umax_all(:,ip,iT,iv) = Pr.Umax;     Qs_all(:,ip,iT,iv)   = Pr.Q;
        end
        x_slice_nm = x_slice/10;   x_gate_nm = x_gate/10;   U_mean = Umean_all(:,:,iT,iv);
        U_min = Umin_all(:,:,iT,iv);   U_max = Umax_all(:,:,iT,iv);   Q_slice = Qs_all(:,:,iT,iv);
        phi_mean = -U_mean;                                           % V, electrostatic potential
        Ec_loc = Ec_slice + U_mean;   Ev_loc = Ev_slice + U_mean;   EN_loc = E_N0 + U_mean;   % eV, absolute
        save(['profile_' tag '.mat'], 'x_slice_nm','x_gate_nm','g_slice','lam_slice','Vgs','V_TG_BG','Vds', ...
             'U_mean','U_min','U_max','phi_mean','Q_slice','Ec_loc','Ev_loc','EN_loc','Ec_slice','Ev_slice','muS','E_N0','Temp','vac_percent');
    end
end
save([run_name '_all.mat'],'lambda_pts','Vds_pts','V_TG_BG','Ez','T_list','vac_list','vac_tag','vac_atoms','vac_orb', ...
     'seed','muS','Ef','E_N0','I_all','Cq2_all','n_at_all','p_at_all','U_at_all','D_at_all','nit_all','dU_all','Q_all', ...
     'Trans','DOSd','grid_k','dens_k','dE_f','Device_atomic_position','type_of','slice_of','A_atom','x_slice','x_gate', ...
     'g_slice','lam_slice','Ec_slice','Ev_slice','Umean_all','Umin_all','Umax_all','Qs_all', ...
     'L_dev','L_g','Length_Device','W_rib','n_at','gate_factor','t_ox','eps_ox','eps_ch','n_edge_rows','n_buffer','n_ungated','n_ramp');

%% ===================== Device metrics: Ion, Ioff, SS, C, tau_d, DED, SPD =====================
if strcmp(sweep,'gate') && npts >= 2
    C_ox = eps0_SI*eps_ox*(L_g*1e-10)*(W_rib*1e-10)/(t_ox*1e-10);   % F, one HfO2 layer over the gated area
    dVg  = xs(end) - xs(1);   n_fit = min(5, npts);
    rows = cell(nT*nV,1);   r = 0;
    for iv = 1:nV
        for iT = 1:nT
            Iv  = abs(real(I_all(:,iT,iv)));                % abs(real()) before any log / ratio
            Ion = Iv(1);   Ioff = Iv(end);                  % lambda = 0 (topological ON), largest lambda (OFF)
            CQ  = Cq2_all(:,iT,iv)*(L_dev*1e-10);           % F, quantum capacitance of the whole channel
            switch cap_model
                case 'parallel', CT = 1./(1/(2*C_ox) + 1./CQ);
                case 'series',   CT = 1./(2/C_ox     + 1./CQ);
                otherwise, error('cap_model must be ''parallel'' or ''series''');
            end
            C_eff  = trapz(xs(:), CT(:))/dVg;               % F, average over the gate swing
            CQ_avg = trapz(xs(:), CQ(:))/dVg;
            tau_d  = C_eff*V_DD/Ion;                        % s
            DED    = V_DD*Ion*tau_d;                        % J per switching event (= C_eff V_DD^2)
            SPD    = V_DD*Ioff;                             % W
            [SS_fit, R2, dec, SS_min] = subthreshold_swing(xs(:), Iv(:), true(npts,1), n_fit);
            r = r + 1;
            rows{r} = table(T_list(iT), vac_list(iv), n_vac(iv), Ion, Ioff, Ion/Ioff, SS_fit, R2, dec, SS_min, ...
                            CQ_avg, C_ox, C_eff, tau_d, DED, SPD, mean(nit_all(:,iT,iv)), max(dU_all(:,iT,iv)), ...
                            'VariableNames', {'T_K','vac_pct','n_vac_atoms','Ion_A','Ioff_A','Ion_Ioff','SS_fit_mVdec','R2', ...
                            'decades','SS_min_mVdec','CQ_avg_F','C_ox_F','C_eff_F','tau_d_s','DED_J','SPD_W','SCF_it_mean','dU_max_eV'});
        end
    end
    MT = vertcat(rows{:});
    writetable(MT, [run_name '_metrics.csv']);
    fprintf('\nC_ox = %.3f aF per HfO2 layer over the gated area (%s model), V_DD = %.2f V\n', C_ox*1e18, cap_model, V_DD);
    fprintf('  T(K)  vac(%%)   Ion(A)     Ioff(A)    Ion/Ioff   SS(mV/dec)  C_eff(aF)  tau_d(ps)   DED(aJ)     SPD(pW)\n');
    for k = 1:height(MT)
        fprintf('%5g  %5.2f   %9.3e  %9.3e  %9.3e  %9.1f  %9.4f  %10.4e  %10.4e  %10.4e\n', MT.T_K(k), MT.vac_pct(k), ...
                MT.Ion_A(k), MT.Ioff_A(k), MT.Ion_Ioff(k), MT.SS_fit_mVdec(k), MT.C_eff_F(k)*1e18, ...
                MT.tau_d_s(k)*1e12, MT.DED_J(k)*1e18, MT.SPD_W(k)*1e12);
    end
    fprintf('tau_d = C_eff V_DD / Ion,  DED = V_DD Ion tau_d,  SPD = V_DD Ioff  ->  %s_metrics.csv\n', run_name);
end

%% ===================== Figures: I-V =====================
blue       = [0.000 0.447 0.698];   vermillion = [0.835 0.369 0.000];   green     = [0.000 0.620 0.451];
purple     = [0.459 0.439 0.702];   orange     = [0.902 0.624 0.000];   teal      = [0.000 0.502 0.502];
magenta    = [0.800 0.200 0.600];   brown      = [0.549 0.337 0.294];   nearBlack = [0.150 0.150 0.150];
cols = {blue, vermillion, green, purple, orange, teal, magenta, brown, nearBlack};
lsty = {'-','--',':','-.'};   mrk = {'s','o','^','v','d'};   % line style -> vacancy level, marker -> temperature
tname = arrayfun(@(T) sprintf('%g K',T), T_list, 'UniformOutput', false);
Ids = abs(real(I_all));                                   % abs(real()) before any log axis

for iv = 1:nV                                             % one figure per vacancy level (Fig13 layout)
    figure('Color',[1 1 1]);  hold on;  grid on;  box on;
    ph = zeros(1,nT);
    for iT = 1:nT
        ph(iT) = semilogy(xs, Ids(:,iT,iv), [lsty{mod(iv-1,4)+1} mrk{mod(iT-1,5)+1}], 'Color', cols{mod(iT-1,9)+1}, ...
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
for iv = 1:nV
    for iT = 1:nT
        k = (iv-1)*nT + iT;
        ph(end+1) = semilogy(xs, Ids(:,iT,iv), [lsty{mod(iv-1,4)+1} mrk{mod(iT-1,5)+1}], 'Color', cols{mod(k-1,9)+1}, ...
                             'LineWidth', 1.5, 'MarkerFaceColor', 'w'); %#ok<SAGROW>
        lab{end+1} = [vac_tag{iv} '%, ' tname{iT}]; %#ok<SAGROW>
    end
end
set(gca, 'YScale', 'log');
xlabel(xlab, 'FontSize', 20, 'FontWeight', 'bold');
ylabel('I_{DS} (A)', 'FontSize', 20, 'FontWeight', 'bold');
legend(ph, lab, 'Location', 'best', 'FontSize', 13, 'NumColumns', max(1,min(3,nV)));
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'YMinorGrid', 'on', 'XMinorGrid', 'on', 'FontSize', 18);
set(gcf, 'Position', [100 100 1100 750]);   xlim([min(xs) max(xs)]);

%% ===================== Figures: band edges, potential and charge along the ribbon =====================
[~, iTp] = min(abs(T_list - T_plot));   ip_on = 1;   ip_off = npts;   ips = [ip_on ip_off];
st_name = {'ON','OFF'};
xs_nm = x_slice/10;   xg_nm = x_gate/10;   xl = [xb(1) xb(N+1)]/10;
gate_fill = [0.90 0.94 1.00];   grey = [0.5 0.5 0.5];
for iv = 1:nV
    figure('Color',[1 1 1]);
    for k = 1:2
        ip = ips(k);   lam = lambda_pts(ip);   Vd = Vds_pts(ip);
        Um = Umean_all(:,ip,iTp,iv).';   Umn = Umin_all(:,ip,iTp,iv).';   Umx = Umax_all(:,ip,iTp,iv).';
        Qs = Qs_all(:,ip,iTp,iv).';   Ecs = Ec_slice(:,ip).';   Evs = Ev_slice(:,ip).';
        ttl = sprintf('%s state: \\lambda_v = %.2f eV (V_{TG}-V_{BG} = %.2f V), V_{DS} = %.2f V | %s%% edge vacancy, T = %g K', ...
                      st_name{k}, lam, gate_factor*lam, Vd, vac_tag{iv}, T_list(iTp));
        % (1) band edges relative to the source Fermi level
        subplot(3,2,k);  hold on;  box on;
        patch([xg_nm(1) xg_nm(2) xg_nm(2) xg_nm(1)], [-20 -20 20 20], gate_fill, 'EdgeColor', 'none');
        fill([xs_nm fliplr(xs_nm)], [Ecs+Umn-muS, fliplr(Ecs+Umx-muS)], teal, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        fill([xs_nm fliplr(xs_nm)], [Evs+Umn-muS, fliplr(Evs+Umx-muS)], vermillion, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        p1 = plot(xs_nm, Ecs+Um-muS, '-o', 'Color', teal, 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', teal);
        p2 = plot(xs_nm, Evs+Um-muS, '-o', 'Color', vermillion, 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', vermillion);
        p3 = plot(xs_nm, E_N0+Um-muS, ':', 'Color', nearBlack, 'LineWidth', 1.2);
        p4 = plot(xs_nm, Ecs-muS, '--', 'Color', grey, 'LineWidth', 1.2);
        plot(xs_nm, Evs-muS, '--', 'Color', grey, 'LineWidth', 1.2);
        p5 = plot([xl(1) xg_nm(1)], [0 0], 'k-', 'LineWidth', 1.2);
        plot([xg_nm(2) xl(2)], -[Vd Vd], 'k-', 'LineWidth', 1.2);
        yl = [min([Evs+Umn-muS, -Vd]) - 0.05, max([Ecs+Umx-muS, 0]) + 0.05];
        xlim(xl);  ylim(yl);
        text(mean([xl(1) xg_nm(1)]), yl(2)-0.02*diff(yl), 'SOURCE', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold', 'Color', grey);
        text(mean(xg_nm), yl(2)-0.02*diff(yl), 'GATED CHANNEL', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold', 'Color', grey);
        text(mean([xg_nm(2) xl(2)]), yl(2)-0.02*diff(yl), 'DRAIN', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontWeight', 'bold', 'Color', grey);
        ylabel('E - \mu_S (eV)', 'FontSize', 13, 'FontWeight', 'bold');
        title(ttl, 'FontSize', 11);
        if k == 1
            legend([p1 p2 p3 p4 p5], {'CB edge (self-consistent)', 'VB edge (self-consistent)', 'neutrality level E_N + U_H', ...
                   'band edges without Hartree term', '\mu_S, \mu_D'}, 'Location', 'southwest', 'FontSize', 9);
        end
        set(gca, 'FontSize', 12);
        % (2) electrostatic potential
        subplot(3,2,2+k);  hold on;  box on;
        patch([xg_nm(1) xg_nm(2) xg_nm(2) xg_nm(1)], [-20 -20 20 20], gate_fill, 'EdgeColor', 'none');
        fill([xs_nm fliplr(xs_nm)], [-Umx, fliplr(-Umn)], blue, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        plot(xs_nm, -Um, '-o', 'Color', blue, 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', blue);
        yl = [min(-Umx) max(-Umn)];   yl = yl + [-1 1]*max(0.1*diff(yl), 1e-3);
        xlim(xl);  ylim(yl);
        ylabel('\phi = -U_H/q (V)', 'FontSize', 13, 'FontWeight', 'bold');
        title('Self-consistent electrostatic potential (shading: range across the ribbon sites)', 'FontSize', 11);
        set(gca, 'FontSize', 12);
        % (3) net charge per slice
        subplot(3,2,4+k);  hold on;  box on;
        patch([xg_nm(1) xg_nm(2) xg_nm(2) xg_nm(1)], [-1e3 -1e3 1e3 1e3], gate_fill, 'EdgeColor', 'none');
        plot(xl, [0 0], '-', 'Color', grey);
        plot(xs_nm, Qs, '-o', 'Color', purple, 'LineWidth', 1.8, 'MarkerSize', 4, 'MarkerFaceColor', purple);
        yl = [min(Qs) max(Qs)];   yl = yl + [-1 1]*max(0.1*diff(yl), 1e-4);
        xlim(xl);  ylim(yl);
        ylabel('net charge per slice (e)', 'FontSize', 13, 'FontWeight', 'bold');
        xlabel('position along the ribbon (nm)', 'FontSize', 13, 'FontWeight', 'bold');
        title(sprintf('Net charge = p - n per slice (total %+.3e e)', sum(Qs)), 'FontSize', 11);
        set(gca, 'FontSize', 12);
    end
    set(gcf, 'Position', [60 60 1500 950]);
end

%% ===================== Figures: carrier density maps, ON and OFF state =====================
st_lab = {sprintf('ON state (\\lambda_v = %.2f eV, V_{DS} = %.2f V)', lambda_pts(ip_on),  Vds_pts(ip_on)), ...
          sprintf('OFF state (\\lambda_v = %.2f eV, V_{DS} = %.2f V)', lambda_pts(ip_off), Vds_pts(ip_off))};
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
        plot([xg_nm(1) xg_nm(1)], [min(y_at) max(y_at)]/10 + [-0.3 0.3], 'k--');
        plot([xg_nm(2) xg_nm(2)], [min(y_at) max(y_at)]/10 + [-0.3 0.3], 'k--');
        axis equal;  xlim([min(x_at) max(x_at)]/10 + [-0.2 0.2]);  ylim([min(y_at) max(y_at)]/10 + [-0.3 0.3]);
        caxis([0 cmax]);  cb = colorbar;  ylabel(cb, 'n + p (cm^{-2})', 'FontSize', 14);
        xlabel('x (nm)', 'FontSize', 16, 'FontWeight', 'bold');  ylabel('y (nm)', 'FontSize', 16, 'FontWeight', 'bold');
        title(sprintf('%s, %s%% edge vacancy, T = %g K (dashed: gate edges, x: missing atoms)', st_lab{k}, vac_tag{iv}, T_list(iTp)), 'FontSize', 13);
        set(gca, 'FontSize', 14);
    end
    set(gcf, 'Position', [100 100 1100 800]);
end

% self-consistent Hartree potential of the ON state (map)
figure('Color',[1 1 1]);
for iv = 1:nV
    subplot(nV,1,iv);  hold on;  box on;
    present = true(n_at,1);   present(vac_atoms{iv}) = false;
    scatter(x_at(present)/10, y_at(present)/10, 28, 1e3*U_at_all(present,ip_on,iTp,iv), 'filled');
    axis equal;  xlim([min(x_at) max(x_at)]/10 + [-0.2 0.2]);  ylim([min(y_at) max(y_at)]/10 + [-0.3 0.3]);
    cb = colorbar;  ylabel(cb, 'U_H (meV)', 'FontSize', 14);
    xlabel('x (nm)', 'FontSize', 16, 'FontWeight', 'bold');  ylabel('y (nm)', 'FontSize', 16, 'FontWeight', 'bold');
    title(sprintf('Hartree potential, %s, %s%% edge vacancy, T = %g K', st_lab{1}, vac_tag{iv}, T_list(iTp)), 'FontSize', 13);
    set(gca, 'FontSize', 14);
end
set(gcf, 'Position', [150 150 1100 800]);

%% ---- numbers for the text ----
fprintf('\n%-8s %6s %12s %12s %12s %8s %10s\n', 'vac (%)', 'T (K)', 'I_first (A)', 'I_last (A)', 'ratio', 'SCF it', 'max|U_H|');
for iv = 1:nV
    for iT = 1:nT
        fprintf('%-8s %6g %12.4e %12.4e %12.3e %8.1f %7.2f meV\n', vac_tag{iv}, T_list(iT), Ids(1,iT,iv), Ids(end,iT,iv), ...
                Ids(1,iT,iv)/Ids(end,iT,iv), mean(nit_all(:,iT,iv)), 1e3*max(max(abs(U_at_all(:,:,iT,iv)))));
    end
end

%% ===================== Local functions =====================
function [T, a1, a2, Ad] = rgf_spectra(z, pot, h0, u0, l0, kp, Sig1, Sig2, need)
% Recursive Green's function over N slices with the pristine slice blocks h0 = H_s,s,
% u0 = H_s,s+1, l0 = H_s+1,s, the on-site potential pot(:,s) (gate + Hartree) and the
% orbitals kp{s} KEPT in slice s (the spin-orbitals of missing atoms are deleted: their
% rows/columns of H, their gate and Hartree entries are zero, they take no part in
% transport, charge or DOS). Slices 1 and N must be complete (contacts).
%   T  = Tr[Gam1 G_1N Gam2 G_1N^+]
%   need = true also returns, per orbital (global index (s-1)*n + i, zero for deleted ones):
%   a2 = diag(G Gam2 G^+), Ad = diag[i(G - G^+)], a1 = Ad - a2 (sum rule)  ->  G^n_ii = f1 a1 + f2 a2
N = numel(kp);   n = size(h0,1);
Gam2 = 1i*(Sig2 - Sig2');
gL = cell(1,N);   Uc = cell(1,N-1);   Lc = cell(1,N-1);
for s = 1:N                                               % left-connected g^L (Sig1, Sig2 at the ends)
    k = kp{s};   m = numel(k);
    if m == n, A = -h0;        d = z - pot(:,s);
    else,      A = -h0(k,k);   d = z - pot(k,s);
    end
    A(1:m+1:end) = A(1:m+1:end) + d.';                   % z - H_ss - pot
    if s == 1, A = A - Sig1; else, A = A - Lc{s-1}*(gL{s-1}*Uc{s-1}); end
    if s == N, A = A - Sig2; end
    gL{s} = A\eye(m);
    if s < N
        k2 = kp{s+1};
        if m == n && numel(k2) == n, Uc{s} = u0;        Lc{s} = l0;
        else,                        Uc{s} = u0(k,k2);  Lc{s} = l0(k2,k);
        end
    end
end
% last block column G_sN = g^L_s H_s,s+1 G_s+1,N and the diagonal blocks G_ss
CN = gL{N};   G = gL{N};
a2 = zeros(n,N);   Ad = zeros(n,N);
if need
    Ad(kp{N},N) = -2*imag(diag(G));   a2(kp{N},N) = real(sum((CN*Gam2).*conj(CN),2));
end
for s = N-1:-1:1
    CN = gL{s}*(Uc{s}*CN);
    if need
        G = gL{s} + (gL{s}*Uc{s})*G*(Lc{s}*gL{s});
        Ad(kp{s},s) = -2*imag(diag(G));
        a2(kp{s},s) = real(sum((CN*Gam2).*conj(CN),2));
    end
end
Gam1 = 1i*(Sig1 - Sig1');
T  = real(sum(sum((Gam1*CN).*(Gam2*CN').')));           % Tr[Gam1 G_1N Gam2 G_1N^+]
a1 = max(Ad - a2, 0);
a1 = a1(:);   a2 = a2(:);   Ad = Ad(:);
end

function [n, p, D] = carrier_density(Ed, A1, A2, AD, muS, muD, kT, EN, kT_D)
% n_i = int_{E_N,i}^inf dE/2pi G^n_ii,  p_i = int_-inf^{E_N,i} dE/2pi [A_ii - G^n_ii],
% D_i = dq_i/dU_i = int dE/2pi [a_S (-df_S/dE) + a_D (-df_D/dE)] >= 0 (broadened with kT_D)
% Ed: uniform energy column (covers E_N +/- U_margin and the Fermi tails); A1, A2, AD: nE x n_orb
f1 = 1./(1 + exp((Ed - muS)/kT));   f2 = 1./(1 + exp((Ed - muD)/kT));
Gn = f1.*A1 + f2.*A2;
Cn = cumtrapz(Ed, Gn)/(2*pi);   CA = cumtrapz(Ed, AD)/(2*pi);
dE = Ed(2) - Ed(1);   nE = numel(Ed);   no = size(A1,2);
k  = floor((EN - Ed(1))/dE) + 1;
assert(all(k >= 1 & k < nE), 'neutrality level E_N0 + U_H outside the density grid: raise U_margin or E_fine');
fr = (EN - Ed(k))/dE;
i1 = sub2ind([nE no], k, (1:no)');
CnE = Cn(i1) + fr.*(Cn(i1+1) - Cn(i1));                   % int_-inf^{E_N} G^n
CAE = CA(i1) + fr.*(CA(i1+1) - CA(i1));                   % int_-inf^{E_N} A
n = Cn(end,:).' - CnE;
p = CAE - CnE;
w1 = sech((Ed - muS)/(2*kT_D)).^2/(4*kT_D);   w2 = sech((Ed - muD)/(2*kT_D)).^2/(4*kT_D);
D = (trapz(Ed, w1.*A1 + w2.*A2)/(2*pi)).';
end

function P = poisson_setup(x_at, y_at, z_at, hx, hy, hz, pad_y, a_half, t_ox, eps_ch, eps_ox, eps0, x_gate)
% 3-D finite-difference Poisson solver  div(eps grad V) = -rho/eps0  (V in volt, rho in e, lengths in A)
% grounded gate planes at z_c +/- (a_half + t_ox) for x inside x_gate (Dirichlet), Neumann elsewhere.
xg = (min(x_at)-hx):hx:(max(x_at)+hx+1e-9);
yg = (min(y_at)-pad_y):hy:(max(y_at)+pad_y+1e-9);
zc = (min(z_at)+max(z_at))/2;   nzh = max(4, round((a_half+t_ox)/hz));   hz = (a_half+t_ox)/nzh;
zg = zc + (-nzh:nzh)*hz;
xg = xg(:);  yg = yg(:);  zg = zg(:);
nx = numel(xg);  ny = numel(yg);  nz = numel(zg);  nn = nx*ny*nz;
[X,Y,Z] = ndgrid(xg,yg,zg);
inlayer = X >= min(x_at)-hx & X <= max(x_at)+hx & Y >= min(y_at)-1 & Y <= max(y_at)+1 & abs(Z-zc) <= a_half+1e-9;
eps_n = eps_ox*ones(nn,1);   eps_n(inlayer(:)) = eps_ch;
xin = xg >= x_gate(1) & xg <= x_gate(2);                   % nodes under the gate metal
isDir = false(nx,ny,nz);   isDir(xin,:,1) = true;   isDir(xin,:,nz) = true;   isDir = isDir(:);
assert(any(isDir), 'no gate nodes: check x_gate');
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
lusolve = @(b) Qf*(Uf\(Lf\(Pf*(Rf\b))));
P.A = A;   P.W = W;   P.keep = keep;   P.scale = scale;   P.lusolve = lusolve;
P.solve = @(rho_at) lusolve(-scale*(keep.*(W'*rho_at)));   % V at the nodes (volt), no charge response
P.nx = nx;   P.ny = ny;   P.nz = nz;   P.hx = hx;   P.hy = hy;   P.hz = hz;   P.zgate = a_half+t_ox;
P.xg = xg;   P.yg = yg;   P.zg = zg;   P.eps = reshape(eps_n,[nx ny nz]);
end

function [V, info] = poisson_newton(P, rho_at, D_at, U_at)
% Newton-Poisson step: rho(U_new) ~ rho + D (U_new - U), U_new = -V  =>
%   [A - s W' diag(D) W] V = -s W' (rho - D.*U),   s = 1/(eps0 hx hy hz)
% solved by GMRES preconditioned with the LU factors of the plain operator A.
% The fixed point (U_new = U) is the ordinary Poisson solution whatever D is.
s = P.scale;   keep = P.keep;   W = P.W;
b = -s*(keep.*(W'*(rho_at - D_at.*U_at)));
if ~any(D_at)
    V = P.lusolve(b);   info = [0 0];   return;
end
afun = @(v) P.A*v - s*(keep.*(W'*(D_at.*(W*v))));
[V, flag, relres, iter] = gmres(afun, b, 50, 1e-9, 10, P.lusolve);
info = [flag, (iter(1)-1)*50 + iter(2)];
if flag ~= 0, warning('Newton-Poisson gmres: flag %d, relative residual %.1e', flag, relres); end
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

function [Ec, Ev] = ribbon_band_edges(bigH, bigU, chi, lam_vals, E_N0, nk)
% band edges of the infinite pristine ribbon with the staggered potential lam*chi
% (edge states included): Ec = lowest level above E_N0, Ev = highest level below E_N0, k in [0, pi]
bigH = full(bigH);   bigU = full(bigU);   kk = linspace(0, pi, nk);
nl = numel(lam_vals);   Ec = zeros(1,nl);   Ev = zeros(1,nl);
for il = 1:nl
    ec = inf;   ev = -inf;   Vl = diag(lam_vals(il)*chi(:));
    for ik = 1:nk
        Hk = bigH + bigU*exp(1i*kk(ik)) + bigU'*exp(-1i*kk(ik)) + Vl;
        e  = eig((Hk + Hk')/2);
        if any(e > E_N0), ec = min(ec, min(e(e > E_N0))); end
        if any(e < E_N0), ev = max(ev, max(e(e < E_N0))); end
    end
    Ec(il) = ec;   Ev(il) = ev;
end
end

function Pr = slice_profile(U, rho, slice_of, present, N)
% per-slice mean / min / max of the Hartree potential over the atoms that are present,
% and the net charge (e) of the slice
Pr.Umean = zeros(1,N);   Pr.Umin = zeros(1,N);   Pr.Umax = zeros(1,N);   Pr.Q = zeros(1,N);
for s = 1:N
    i = (slice_of == s) & present;
    Pr.Umean(s) = mean(U(i));   Pr.Umin(s) = min(U(i));   Pr.Umax(s) = max(U(i));
    Pr.Q(s) = sum(rho(slice_of == s));
end
end

function [SS_fit, R2, dec, SS_min, iw] = subthreshold_swing(Vg, Id, ok, n_fit)
% SS (mV/decade) from a linear fit of log10(I) vs V_G over the steepest n_fit-point window
% (chosen automatically among reliable points); the device turns OFF with increasing V_G.
y  = log10(max(Id, realmin));
SS_fit = NaN;   R2 = NaN;   dec = NaN;   SS_min = NaN;   iw = [];
n = numel(Vg);
if n < 2, return; end
dy = diff(y)./diff(Vg);
good = ok(1:end-1) & ok(2:end) & dy < 0;
if any(good), SS_min = -1000/min(dy(good)); end
best = -inf;
for i0 = 1:n-n_fit+1
    i = i0:i0+n_fit-1;
    if ~all(ok(i)), continue; end
    Pf = polyfit(Vg(i), y(i), 1);
    if Pf(1) >= 0, continue; end
    yf = polyval(Pf, Vg(i));
    r2 = 1 - sum((y(i)-yf).^2)/max(sum((y(i)-mean(y(i))).^2), realmin);
    if -Pf(1) > best
        best = -Pf(1);   SS_fit = 1000/best;   R2 = r2;   dec = y(i(1)) - y(i(end));   iw = i;
    end
end
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
