%% Vgs_Ids_edge_vacancy_T.m
% 1T'-WTe2 zigzag-nanoribbon TIFET: I_DS vs (V_TG - V_BG) with EDGE vacancies
% Temperatures: 300, 77, 4 K  |  edge vacancy: 0 (reference), 0.85, 2.0 %
% Coherent NEGF solved with the recursive Green's function (RGF) method.
% Same Wannier Hamiltonian, leads and gate model as Vgs_Ids_vacancy.m.
% Needs on the path: cal_H_U_Ubar.m, hamiltonian_TMD_device_H.m, lead_eps.m
%
% Corrections with respect to Vgs_Ids_vacancy.m
%  1) Vacancy -> orbital map. Old: quotient = floor(n/6) and the deletion always
%     started at the first orbital of the cell, so a Te4 vacancy deleted Te1, a W1
%     vacancy deleted Te1 + 4 of the 6 Te2 spin-orbitals, and atom 6k hit the next
%     cell. Now all spin-orbitals of the missing atom (and only those) are removed.
%  2) Energy grid. Old: 10 points in [mu_D, mu_S] = 42 % of the 300 K Fermi window
%     (82 % at 77 K) and no thermionic OFF current. Now the grid covers
%     +/- n_kT*kB*T(300 K) around the window (fine steps close to it).
%  3) Temperature. kB = 8.617333e-5 eV/K (old 8.3333e-5); kT = kB*T in the Fermi
%     functions AND in the C_Q kernel (old: fixed 0.0259 eV). T(E) of coherent
%     transport does not depend on temperature, so it is solved once and reused
%     at 300, 77 and 4 K with the SAME vacancy configuration.
%  4) Spin. The 44-orbital SOC basis already contains both spins -> I = (q^2/h)*int T
%     (old 2q^2/h counted spin twice: currents halve, ON/OFF ratio and SS unchanged).
%  5) Fixed seed, nested configurations (0.85 % set inside the 2.0 % set), no vacancy
%     in the n_buffer slices next to each contact, randperm(length(max_indices)) bug gone.
%  6) RGF instead of the dense 11880 x 11880 inversion; lead self-energies once per
%     energy; unused Gn, Gp removed; real(T); trapz; C_Q per area divides by the width.
clc; clear; close all;
caption = 'wte2';

%% ===================== User settings =====================
T_list      = [300 77 4];               % K
vac_list    = [0 0.85 2.0];             % edge vacancy, % of all device atoms (0 = pristine reference)
vac_tag     = {'0','0.85','2.0'};       % used in the file names
seed        = 2026;                     % rerun with other seeds to average over configurations
n_edge_rows = 2;                        % outermost atomic rows per edge that may lose atoms
n_buffer    = 2;                        % vacancy-free slices next to each contact (>= 1)
U_vac       = 1e3;                      % eV, on-site energy of removed (decoupled) orbitals

ZBNR        = 12;                       % 12 -> 9 cells across, W = 5.7 nm (same as the pristine run)
Unit_Cells  = 30;                       % slices along x, L = 10.3 nm
lambda_list = linspace(0,1.5,20);       % staggered potential lambda_v (eV) = old "Vgs"

Vds    = 0.05;                          % V
muS    = 3.05;                          % eV, source (lead 1)
muD    = muS - Vds;                     % eV, drain  (lead 2) = 3.00
Ef     = 3.015;                         % eV, equilibrium Fermi level used for C_Q
dE_f   = 2.5e-3;                        % eV, fine step up to E_fine outside [mu_D, mu_S]
dE_c   = 7.5e-3;                        % eV, coarse step in the far 300 K tails
E_fine = 0.15;                          % eV
n_kT   = 25;                            % tails kept: n_kT*kB*T beyond mu_D and mu_S (e^-25 ~ 1e-11)
eta    = 1e-6;                          % eV

t_ox = 14; eps_ox = 25; eps_wte2 = 1;   % HfO2 thickness (Angstrom) and permittivities

compute_DOS  = true;                    % false: ~35 % faster, no C_Q
validate_rgf = true;                    % RGF vs full sparse solve (largest vacancy set) + lambda = 0 test
quick_test   = false;                   % true: coarse run to test the pipeline first
resume       = true;                    % continue from the checkpoint if it matches

if quick_test, lambda_list = linspace(0,1.5,4); dE_f = 1e-2; dE_c = 2e-2; end
run_name  = ['edge_vacancy_' strjoin(vac_tag,'_')];      % separate sessions per vac_list do not collide
ckpt_file = [run_name '_checkpoint.mat'];

%% ===================== Constants =====================
q    = 1.602176634e-19;                 % C
h_pl = 6.62607015e-34;                  % J s
kB   = 8.617333262e-5;                  % eV/K
IE   = q^2/h_pl;                        % A/V; spin is explicit in the SOC basis

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
Length_Device = max(Device_atomic_position(:,1)) - min(Device_atomic_position(:,1));  % Angstrom
L_dev = Unit_Cells*ax;                  % Angstrom, length of the N slices (DOS per length)
W_rib = num_cell*by;                    % Angstrom, ribbon width (C_Q per area)

t_wte2      = atom_factors(3,3) - atom_factors(2,3);        % 4.19386 A (outer Te-Te)
gate_factor = (2*t_ox*eps_wte2/eps_ox + t_wte2)/t_wte2;     % 1.26706 for t_ox = 14 A
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

%% ===================== Edge vacancies =====================
cell_of  = ceil((1:n_at)'/6);                   % cell of each atom
type_of  = (1:n_at)' - 6*(cell_of-1);           % 1..6 = Te1 Te2 Te3 Te4 W1 W2
slice_of = Device_unit(cell_of,1) + 1;          % slice along x
y_at     = Device_atomic_position(:,2);
y_rows   = unique(y_at);                         % atomic rows across the width
edge_y   = [y_rows(1:n_edge_rows); y_rows(end-n_edge_rows+1:end)];
cand     = find(ismember(y_at,edge_y) & slice_of > n_buffer & slice_of <= N-n_buffer);

rng(seed,'twister');
cand = cand(randperm(numel(cand)));             % one random order -> nested configurations
nV = numel(vac_list);   vac_atoms = cell(1,nV);   rm_all = cell(1,nV);
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
    fprintf('vacancy %4s %%: %2d atoms (%.3f %%)', vac_tag{iv}, Nv, 100*Nv/n_at);
    for t = 1:6
        nt = nnz(type_of(vac_atoms{iv}) == t);
        if nt > 0, fprintf('  %s x%d', atom_name{t}, nt); end
    end
    fprintf('\n');
end

%% ===================== One-shot checks =====================
if validate_rgf
    z   = (muS+muD)/2 + 1i*eta;   lam = lambda_list(ceil(end/2));
    [g1,~] = lead_eps(bigH,bigUbar,eta,real(z),count1,count2);
    [g2,~] = lead_eps(bigH,bigU,  eta,real(z),count1,count2);
    S1 = bigUbar*g1*bigU;   S2 = bigU*g2*bigUbar;
    rm = rm_all{nV};
    T_rgf = rgf_T_trG(z,lam,h0,u0,l0,chi_slice,rm,S1,S2,U_vac,false);
    rg   = cell2mat(arrayfun(@(s) rm{s} + (s-1)*count1, 1:N, 'UniformOutput', false));
    keep = ones(count2,1);   keep(rg) = 0;   P = spdiags(keep,0,count2,count2);
    Hd   = sparse(H) + spdiags(lam*repmat(chi_slice,N,1),0,count2,count2);
    Hd   = P*Hd*P + sparse(rg,rg,U_vac,count2,count2);
    b1 = blk(1);   bN = blk(N);
    A = z*speye(count2) - Hd;
    A(b1,b1) = A(b1,b1) - S1;   A(bN,bN) = A(bN,bN) - S2;
    RHS = zeros(count2,count1);   RHS(bN,:) = eye(count1);
    X   = A\RHS;                                % last-slice columns of G^r
    G1N = X(b1,:);
    T_full = real(trace(1i*(S1-S1')*G1N*(1i*(S2-S2'))*G1N'));
    fprintf('RGF check (%s %% vacancy, E = %.4f eV, lambda = %.3f eV): T_RGF = %.6f | T_full = %.6f\n', ...
            vac_tag{nV}, real(z), lam, T_rgf, T_full);
    T_0 = rgf_T_trG(z,0,h0,u0,l0,chi_slice,rm0,S1,S2,U_vac,false);
    fprintf('Pristine, lambda = 0 (device = lead): T = %.6f (should be ~integer, 2 in the QSH window)\n', T_0);
    clear A RHS X G1N Hd P
end
clear H

%% ===================== Energy grid =====================
kT_max = kB*max(T_list);
k_bias = round((muS-muD)/dE_f);
assert(abs(k_bias*dE_f-(muS-muD)) < 1e-9, 'Vds/dE_f must be an integer');
k_fine = round(E_fine/dE_f);
E_f    = muD + (-k_fine:k_bias+k_fine)*dE_f;                % fine part, mu_D and mu_S on the grid
n_c    = max(0, ceil((n_kT*kT_max - k_fine*dE_f)/dE_c));   % coarse points per side
E      = [E_f(1)-(n_c:-1:1)*dE_c, E_f, E_f(end)+(1:n_c)*dE_c];
nE = numel(E);   nL = numel(lambda_list);
assert(E(1) <= Ef-n_kT*kT_max && E(end) >= Ef+n_kT*kT_max, 'grid does not cover the C_Q kernel');
fprintf('%d energies (%.3f to %.3f eV) x %d gate points x %d vacancy levels\n', nE, E(1), E(end), nL, nV);

%% ===================== NEGF sweep: T(E,lambda,vac) and DOS =====================
run_sig = [seed n_edge_rows n_buffer ZBNR Unit_Cells U_vac eta muS Vds compute_DOS];
Trans = zeros(nE,nL,nV);   DOS = zeros(nE,nL,nV);   iE0 = 1;
if resume && exist(ckpt_file,'file') == 2
    S = load(ckpt_file);
    if isequal(S.E,E) && isequal(S.lambda_list,lambda_list) && ...
       isequal(S.vac_list,vac_list) && isequal(S.run_sig,run_sig)
        Trans = S.Trans;  DOS = S.DOS;  iE0 = S.iE_done + 1;
        if iE0 > nE, fprintf('All energies already in %s\n', ckpt_file);
        else,        fprintf('Resuming at energy point %d/%d\n', iE0, nE); end
    end
end

t0 = tic;
for iE = iE0:nE                                  % energies are independent
    [g1,~] = lead_eps(bigH,bigUbar,eta,E(iE),count1,count2);    % left lead
    [g2,~] = lead_eps(bigH,bigU,  eta,E(iE),count1,count2);    % right lead
    Sig1 = bigUbar*g1*bigU;
    Sig2 = bigU*g2*bigUbar;
    for iv = 1:nV
        for il = 1:nL
            [Trans(iE,il,iv), trG] = rgf_T_trG(E(iE)+1i*eta, lambda_list(il), h0, u0, l0, ...
                                       chi_slice, rm_all{iv}, Sig1, Sig2, U_vac, compute_DOS);
            DOS(iE,il,iv) = -imag(trG)/(pi*L_dev);   % states/(eV*Angstrom), both spins
        end
    end
    if iE == iE0 || mod(iE,10) == 0 || iE == nE
        iE_done = iE;
        save(ckpt_file,'E','lambda_list','vac_list','run_sig','Trans','DOS','iE_done');
        el = toc(t0)/60;
        fprintf('E %3d/%3d (%.4f eV) | %.1f min elapsed | ~%.1f min left\n', ...
                iE, nE, E(iE), el, el*(nE-iE)/(iE-iE0+1));
    end
end

%% ===================== Current and quantum capacitance for each T =====================
Vgs     = lambda_list;                  % same meaning as before (old plot scripts multiply by 1.26706)
V_TG_BG = gate_factor*lambda_list;      % V
Ez      = lambda_list/t_wte2;           % V/Angstrom (thesis convention)
T_top   = max(Trans(:));                % most channels seen anywhere on the grid
for iT = 1:numel(T_list)
    Temp = T_list(iT);   kT = kB*Temp;
    fw = @(x) 1./(1+exp((x-muS)/kT)) - 1./(1+exp((x-muD)/kT));
    I_floor = 10*IE*kT*T_top*(fw(E(1)) + fw(E(end)));   % A, currents below this depend on the grid range
    for iv = 1:nV
        I = zeros(1,nL);   Do = zeros(1,nL);
        for il = 1:nL
            I(il)  = IE*landauer_window(E, Trans(:,il,iv), muS, muD, kT, n_kT);   % A
            Do(il) = thermal_dos(E, DOS(:,il,iv), Ef, kT, n_kT);             % states/(eV*A)
        end
        Cq2 = q*1e10*Do;                % F/m   (per unit channel length)
        Cq1 = Cq2/(W_rib*1e-10);        % F/m^2 (per unit area)
        vac_percent = vac_list(iv);   vac_atoms_used = vac_atoms{iv};
        tag = sprintf('edge_Vacancy_%s_temp%d', vac_tag{iv}, Temp);
        save(['gate_voltage_current_' tag '.mat'], 'Vgs','V_TG_BG','Ez','I','Length_Device', ...
             'Temp','vac_percent','vac_atoms_used','seed');
        save(['quantum_capacitance_data_' tag '.mat'], 'Vgs','V_TG_BG','Cq1','Cq2','Temp','vac_percent');
        fprintf('T = %3d K | vac = %4s %% | I(0 V) = %.3e A | I(%.2f V) = %.3e A | ratio = %.2e | floor %.0e A\n', ...
                Temp, vac_tag{iv}, abs(I(1)), V_TG_BG(end), abs(I(end)), abs(I(1))/abs(I(end)), I_floor);
    end
end
save([run_name '_NEGF_all.mat'],'E','Trans','DOS','lambda_list','V_TG_BG','Ez','T_list', ...
     'vac_list','vac_tag','vac_atoms','seed','muS','muD','Ef','Vds','L_dev','Length_Device', ...
     'W_rib','n_at','gate_factor','t_ox','eps_ox','n_edge_rows','n_buffer');

%% ===================== Local functions =====================
function [T, trG] = rgf_T_trG(z, lam, h0, u0, l0, chi, rm, Sig1, Sig2, U_vac, doDOS)
% Recursive Green's function for N identical slices (h0 = H_ss, u0 = H_{s,s+1},
% l0 = H_{s+1,s}) with the orbitals rm{s} removed and the gate term lam*chi.
% T = Tr[Gam1 G_1N Gam2 G_1N'];  trG = Tr[G^r] over the device (only if doDOS).
N  = numel(rm);   n = size(h0,1);   Id = eye(n);   dg = 1:n+1:n*n;
A0 = -h0;   A0(dg) = A0(dg) + (z - lam*chi(:)).';     % z - H_ss - lambda*chi
gL = cell(1,N);   X = cell(1,N-1);   Lc = cell(1,N-1);
for s = 1:N
    A = A0;   r = rm{s};
    if ~isempty(r)                                    % missing atom: decoupled, far off-resonance
        A(r,:) = 0;   A(:,r) = 0;   A(dg(r)) = z - U_vac;
    end
    if s == 1
        A = A - Sig1;
    else
        A = A - Lc{s-1}*X{s-1};                       % H_{s,s-1} gL_{s-1} H_{s-1,s}
    end
    if s == N, A = A - Sig2; end
    gL{s} = A\Id;                                     % left-connected Green's function
    if s == 1
        G1N = gL{1};
    else
        G1N = (G1N*Uc)*gL{s};                         % G_{1,s} = G_{1,s-1} H_{s-1,s} gL_s
    end
    if s < N
        Uc = u0;   Lc{s} = l0;
        if ~isempty(r) || ~isempty(rm{s+1})
            Uc(r,:) = 0;            Uc(:,rm{s+1}) = 0;       % H_{s,s+1}
            Lc{s}(rm{s+1},:) = 0;   Lc{s}(:,r) = 0;          % H_{s+1,s}
        end
        X{s} = gL{s}*Uc;
    end
end
Gam1 = 1i*(Sig1 - Sig1');   Gam2 = 1i*(Sig2 - Sig2');
T = real(trace(Gam1*G1N*Gam2*G1N'));
trG = 0;
if doDOS
    G = gL{N};   trG = trace(G);
    for s = N-1:-1:1
        G   = gL{s} + X{s}*G*(Lc{s}*gL{s});           % G_{s,s}
        trG = trG + trace(G);
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
