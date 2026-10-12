function R = Vgs_Ids_edge_vacancy_ML
%% Vgs_Ids_edge_vacancy_ML.m
% 1T'-WTe2 zigzag-nanoribbon TIFET with EDGE vacancies.
% For every temperature and vacancy level: I_DS vs (V_TG - V_BG), quantum and total
% gate capacitance, intrinsic delay tau_d, dynamic energy dissipation (DED) and static
% power dissipation (SPD). Results are written as separate TRAIN and TEST files for ML.
%
% Non-self-consistent coherent NEGF (no Poisson solver), recursive Green's function (RGF).
% Same Wannier Hamiltonian, leads, gate model and vacancy sampling as Vgs_Ids_edge_vacancy_T.m.
% Needs on the path: cal_H_U_Ubar.m, hamiltonian_TMD_device_H.m, lead_eps.m
%
% Vacancy model: every spin-orbital of a missing atom is DELETED from the device problem,
% i.e. its rows and columns of H_s,s, H_s,s+1 and H_s+1,s are zero AND its diagonal entry of
% the gate matrix lambda*chi is zero. (Vgs_Ids_edge_vacancy_T.m also zeroed those hoppings and
% skipped the gate term, but parked the decoupled orbital at U_vac = 1000 eV.) Deleting leaves
% no artificial level and gives the same T(E) as the full device matrix with those rows,
% columns and gate entries literally set to 0 -- the validation block checks this.
%
% Coherent T(E) does not depend on temperature: T(E,lambda) and the DOS are solved once per
% vacancy configuration and reused for every T in T_list.
%
% Usage:  R = Vgs_Ids_edge_vacancy_ML;     (everything is also saved in out_dir)
caption = 'wte2';

%% ===================== User settings =====================
T_list      = 4:2:78;                   % K -> 4, 12, ..., 76  (use [4:8:76 77] to include 77 K)
vac_list    = 0:0.1:2.0;               % edge vacancy, % of all device atoms (0 = pristine)
seed        = 2026;                     % vacancy configuration (rerun with other seeds for more data)
n_edge_rows = 2;                        % outermost atomic rows per edge that may lose atoms
n_buffer    = 2;                        % vacancy-free slices next to each contact (>= 1)

ZBNR        = 12;                       % 9 cells across, W = 5.70 nm
Unit_Cells  = 30;                       % slices along x, L = 10.49 nm (atomic span 10.32 nm)
lambda_list = linspace(0,1.5,20);       % staggered potential lambda_v (eV) = old "Vgs"

Vds    = 0.05;                          % V
muS    = 3.05;                          % eV, source (lead 1)
muD    = muS - Vds;                     % eV, drain  (lead 2) = 3.00
Ef     = 3.015;                         % eV, equilibrium Fermi level used for C_Q
dE_f   = 2.5e-3;                        % eV, fine step up to E_fine outside [mu_D, mu_S]
dE_c   = 7.5e-3;                        % eV, coarse step in the far tails
E_fine = 0.15;                          % eV
n_kT   = 30;                            % tails kept: n_kT*kB*max(T) beyond mu_D, mu_S (e^-30 ~ 1e-13)
eta    = 1e-6;                          % eV

t_ox = 14; eps_ox = 25; eps_wte2 = 1;   % HfO2 thickness (Angstrom) and permittivities

% ---- device metrics ----
cap_model = 'parallel';                 % 'parallel': 2*C_ox in series with C_Q        (october 11 script)
                                        % 'series'  : C_ox, C_Q and C_ox all in series (thesis slide)
V_DD = Vds;                             % supply in tau_d = C*V_DD/Ion and DED = V_DD*Ion*tau_d

% ---- machine-learning split: whole (T, vac) curves go either to TRAIN or to TEST ----
test_frac  = 0.20;                      % fraction of the (T, vac) curves held out for testing
split_seed = 7;
test_pairs = [];                        % optional [T vac; T vac; ...] to choose the test curves yourself

use_parfor    = true;                   % parfor over energies if the Parallel Computing Toolbox exists
validate      = true;                   % deleted orbitals vs full matrix with zero rows/cols + pristine test
resume        = true;                   % continue from the checkpoint in out_dir
save_per_case = true;                   % old-format .mat per (T, vac) for the plot / SS scripts
quick_test    = false;                  % true: coarse run to test the pipeline first

if quick_test
    lambda_list = linspace(0,1.5,4);  vac_list = [0 1 2];  dE_f = 1e-2;  dE_c = 2e-2;
end
vac_tag = arrayfun(@(v) sprintf('%.2f',v), vac_list, 'UniformOutput', false);
out_dir = sprintf('ML_edge_vacancy_seed%d', seed);
if quick_test, out_dir = [out_dir '_quick']; end
if ~exist(out_dir,'dir'), mkdir(out_dir); end
ckpt_file = fullfile(out_dir,'NEGF_checkpoint.mat');

%% ===================== Constants =====================
q    = 1.602176634e-19;                 % C
h_pl = 6.62607015e-34;                  % J s
kB   = 8.617333262e-5;                  % eV/K
eps0 = 8.8541878128e-12;                % F/m
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
for ic = 1:Unit_Cells
    Device_unit((ic-1)*num_cell+(1:num_cell),:) = main_unit + [(ic-1) 0 0];
end
n_at = 6*n_cell_dev;
Device_atomic_position = zeros(n_at,3);
for c = 1:n_cell_dev
    Device_atomic_position(6*(c-1)+(1:6),:) = Device_unit(c,:).*unit_cell_length + atom_factors;
end
Length_Device = max(Device_atomic_position(:,1)) - min(Device_atomic_position(:,1));  % Angstrom
L_dev = Unit_Cells*ax;                  % Angstrom, length of the N gated slices (DOS per length)
W_rib = num_cell*by;                    % Angstrom, ribbon width

t_wte2      = atom_factors(3,3) - atom_factors(2,3);        % 4.19386 A (outer Te-Te)
gate_factor = (2*t_ox*eps_wte2/eps_ox + t_wte2)/t_wte2;     % 1.26706 for t_ox = 14 A
V_TG_BG     = gate_factor*lambda_list;                      % V
Ez          = lambda_list/t_wte2;                           % V/Angstrom (thesis convention)
fprintf('W = %.2f nm, L = %.2f nm, %d atoms, (V_TG-V_BG)/lambda = %.5f\n', ...
        W_rib/10, L_dev/10, n_at, gate_factor);

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
nV = numel(vac_list);
vac_atoms = cell(1,nV);   vac_orb = cell(1,nV);   kp_all = cell(1,nV);   n_vac = zeros(1,nV);
atom_name = {'Te1','Te2','Te3','Te4','W1','W2'};
for iv = 1:nV
    Nv = round(vac_list(iv)/100*n_at);
    assert(Nv <= numel(cand), 'Not enough edge sites: increase n_edge_rows');
    vac_atoms{iv} = sort(cand(1:Nv));
    g = zeros(1,0);                              % global indices of ALL spin-orbitals of the missing atoms
    for a = vac_atoms{iv}(:)'
        t = type_of(a);
        g = [g, (cell_of(a)-1)*nob + off(t) + (1:norb(t))]; %#ok<AGROW>
    end
    g = sort(g);
    assert(numel(unique(g)) == numel(g) && numel(g) == sum(norb(type_of(vac_atoms{iv}))), ...
           'vacancy -> orbital map is wrong');
    vac_orb{iv} = g;
    s_of = ceil(g/count1);                       % slice of each deleted orbital
    kp = cell(1,N);                              % orbitals KEPT in each slice (slice-local index)
    for s = 1:N
        kp{s} = setdiff(1:count1, g(s_of == s) - (s-1)*count1);
    end
    assert(numel(kp{1}) == count1 && numel(kp{N}) == count1, 'vacancy in a contact slice');
    kp_all{iv} = kp;
    n_vac(iv)  = Nv;
    fprintf('vacancy %s %%: %2d atoms (%.3f %%), %3d orbitals with H = 0 and gate = 0 |', ...
            vac_tag{iv}, Nv, 100*Nv/n_at, numel(g));
    for t = 1:6
        nt = nnz(type_of(vac_atoms{iv}) == t);
        if nt > 0, fprintf(' %s x%d', atom_name{t}, nt); end
    end
    fprintf('\n');
end
vac_actual = 100*n_vac/n_at;                    % realised vacancy percentage
kp0 = repmat({1:count1},1,N);                   % pristine device

%% ===================== Validation =====================
if validate
    iv  = nV;   z = (muS+muD)/2 + 1i*eta;   lam = lambda_list(ceil(end/2));
    [Sig1,Sig2,Gam1,Gam2] = lead_sigma(real(z),eta,bigH,bigU,bigUbar,count1,count2);
    T_rgf = rgf_T_trG(z,lam,h0,u0,l0,chi_slice,kp_all{iv},Sig1,Sig2,Gam1,Gam2,false);
    % full device matrix H + gate, with the rows/columns of the missing atoms literally set to 0
    v    = vac_orb{iv};
    keep = ones(count2,1);   keep(v) = 0;   P = spdiags(keep,0,count2,count2);
    Hd   = sparse(H) + spdiags(lam*repmat(chi_slice,N,1),0,count2,count2);
    Hd   = P*Hd*P;
    assert(nnz(Hd(v,:)) == 0 && nnz(Hd(:,v)) == 0, 'vacancy rows/columns of H + gate are not zero');
    b1 = blk(1);   bN = blk(N);
    A  = z*speye(count2) - Hd;
    A(b1,b1) = A(b1,b1) - Sig1;   A(bN,bN) = A(bN,bN) - Sig2;
    RHS = zeros(count2,count1);   RHS(bN,:) = eye(count1);
    Xf  = A\RHS;                                 % last-slice columns of G^r
    G1N = Xf(b1,:);
    T_full = real(trace(Gam1*G1N*Gam2*G1N'));
    fprintf(['Check (%s %% vacancy, E = %.4f eV, lambda = %.3f eV): T (orbitals deleted) = %.8f | ', ...
             'T (full H, rows/cols + gate = 0) = %.8f | rel. diff %.1e\n'], vac_tag{iv}, real(z), lam, ...
             T_rgf, T_full, abs(T_rgf-T_full)/max(abs(T_full),realmin));
    T_0 = rgf_T_trG(z,0,h0,u0,l0,chi_slice,kp0,Sig1,Sig2,Gam1,Gam2,false);
    fprintf('Pristine, lambda = 0 (device = lead): T = %.6f (should be ~integer, 2 in the QSH window)\n', T_0);
    clear A RHS Xf G1N Hd P
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
nE = numel(E);   nL = numel(lambda_list);   nT = numel(T_list);
assert(E(1) <= Ef-n_kT*kT_max && E(end) >= Ef+n_kT*kT_max, 'grid does not cover the C_Q kernel');
fprintf('%d energies (%.3f to %.3f eV) x %d gate points x %d vacancy levels = %d RGF solves, reused for %d temperatures\n', ...
        nE, E(1), E(end), nL, nV, nE*nL*nV, nT);

%% ===================== NEGF sweep: T(E,lambda,vac) and DOS =====================
run_sig = [seed n_edge_rows n_buffer ZBNR Unit_Cells eta muS Vds];
Trans = zeros(nE,nL,nV);   DOS = zeros(nE,nL,nV);   done = false(nE,1);
if resume && exist(ckpt_file,'file') == 2
    S = load(ckpt_file);
    assert(isequal(S.E,E) && isequal(S.lambda_list,lambda_list) && isequal(S.vac_list,vac_list) && ...
           isequal(S.run_sig,run_sig), '%s was made with other settings: rename or delete it.', ckpt_file);
    Trans = S.Trans;   DOS = S.DOS;   done = S.done;
    fprintf('Resuming: %d of %d energies already done\n', nnz(done), nE);
end

M = 0;                                           % 0 -> parfor runs as an ordinary loop on the client
if use_parfor
    try
        pool = gcp;   M = pool.NumWorkers;
    catch
        M = 0;                                   % no Parallel Computing Toolbox
    end
end
fprintf('Workers: %d\n', M);
chunk = max(8, 3*M);                             % energies between checkpoints
todo  = find(~done).';
t0 = tic;   n0 = nnz(done);
for c0 = 1:chunk:numel(todo)
    idx = todo(c0:min(c0+chunk-1,end));
    Tc = zeros(nL,nV,numel(idx));   Dc = Tc;
    parfor (k = 1:numel(idx), M)                 % energies are independent
        [Tk, Dk] = one_energy(E(idx(k)), eta, bigH, bigU, bigUbar, count1, count2, ...
                              lambda_list, h0, u0, l0, chi_slice, kp_all, L_dev);
        Tc(:,:,k) = Tk;   Dc(:,:,k) = Dk;
    end
    Trans(idx,:,:) = permute(Tc,[3 1 2]);   DOS(idx,:,:) = permute(Dc,[3 1 2]);
    done(idx) = true;
    save(ckpt_file,'E','lambda_list','vac_list','run_sig','Trans','DOS','done');
    el = toc(t0)/60;
    fprintf('%3d/%d energies | %.1f min elapsed | ~%.1f min left\n', ...
            nnz(done), nE, el, el*(nE-nnz(done))/max(nnz(done)-n0,1));
end
save(fullfile(out_dir,'NEGF_all.mat'),'E','Trans','DOS','lambda_list','V_TG_BG','Ez','T_list', ...
     'vac_list','vac_tag','vac_atoms','vac_orb','n_vac','vac_actual','seed','muS','muD','Ef','Vds', ...
     'L_dev','Length_Device','W_rib','n_at','gate_factor','t_ox','eps_ox','n_edge_rows','n_buffer');

%% ===================== Current and capacitances for every (T, vac) =====================
T_top   = max(Trans(:));                         % most channels seen anywhere on the grid
I_all   = zeros(nL,nT,nV);   Do_all = zeros(nL,nT,nV);   I_floor = zeros(1,nT);
for iT = 1:nT
    kT = kB*T_list(iT);
    fw = @(x) 1./(1+exp((x-muS)/kT)) - 1./(1+exp((x-muD)/kT));
    I_floor(iT) = 10*IE*kT*T_top*(fw(E(1)) + fw(E(end)));   % A, currents below this depend on the grid range
    for iv = 1:nV
        for il = 1:nL
            I_all(il,iT,iv)  = IE*landauer_window(E, Trans(:,il,iv), muS, muD, kT, n_kT);   % A
            Do_all(il,iT,iv) = thermal_dos(E, DOS(:,il,iv), Ef, kT, n_kT);             % states/(eV*A)
        end
    end
end
I_all   = real(I_all);
Cq2_all = q*1e10*Do_all;                         % F/m   C_Q per unit channel length (whole width)
Cq1_all = Cq2_all/(W_rib*1e-10);                 % F/m^2 C_Q per unit area
CQ_all  = Cq2_all*(L_dev*1e-10);                 % F     C_Q of the whole channel
C_ox    = eps0*eps_ox*(L_dev*1e-10)*(W_rib*1e-10)/(t_ox*1e-10);   % F, one HfO2 layer over L_dev x W_rib
switch cap_model                                 % both terms are absolute capacitances (F)
    case 'parallel', CT_all = 1./(1/(2*C_ox) + 1./CQ_all);
    case 'series',   CT_all = 1./(2/C_ox     + 1./CQ_all);
    otherwise, error('cap_model must be ''parallel'' or ''series''');
end

%% ===================== Device metrics for every (T, vac) =====================
Ion = zeros(nT,nV);   Ioff = Ion;   C_eff = Ion;   CQ_avg = Ion;
dV  = V_TG_BG(end) - V_TG_BG(1);
for iv = 1:nV
    for iT = 1:nT
        Iv = abs(real(I_all(:,iT,iv)));
        Ion(iT,iv)    = Iv(1);                   % lambda = 0: topological ON state
        Ioff(iT,iv)   = Iv(end);                 % largest lambda: trivial OFF state
        C_eff(iT,iv)  = trapz(V_TG_BG, CT_all(:,iT,iv).')/dV;   % F, dQ/dV over the gate swing
        CQ_avg(iT,iv) = trapz(V_TG_BG, CQ_all(:,iT,iv).')/dV;   % F
    end
end
tau_d = C_eff*V_DD./Ion;                         % s
DED   = V_DD*Ion.*tau_d;                         % J  (= C_eff*V_DD^2)
SPD   = V_DD*Ioff;                               % W

fprintf('\nC_ox = %.3f aF per HfO2 layer (t_ox = %.2f nm, %s model)\n', C_ox*1e18, t_ox/10, cap_model);
for iT = 1:nT
    fprintf('T = %3g K: currents below %.1e A depend on the energy-grid range\n', T_list(iT), I_floor(iT));
end
fprintf('\n  T(K)  vac(%%)   Ion(A)     Ioff(A)    Ion/Ioff   C_eff(aF)  tau_d(ps)   DED(aJ)     SPD(pW)\n');
for iv = 1:nV
    for iT = 1:nT
        fprintf('%5g  %5.2f   %9.3e  %9.3e  %9.3e  %9.4f  %10.3e  %10.3e  %10.3e\n', T_list(iT), ...
                vac_list(iv), Ion(iT,iv), Ioff(iT,iv), Ion(iT,iv)/Ioff(iT,iv), C_eff(iT,iv)*1e18, ...
                tau_d(iT,iv)*1e12, DED(iT,iv)*1e18, SPD(iT,iv)*1e12);
    end
end

%% ===================== Machine-learning dataset: TRAIN / TEST =====================
[TT, VV] = ndgrid(T_list, vac_list);             % nT x nV, one entry per I-V curve
is_test  = false(nT,nV);
if isempty(test_pairs)                           % random interior curves -> test is pure interpolation
    inner = TT > min(T_list) & TT < max(T_list) & VV > min(vac_list) & VV < max(vac_list);
    ci = find(inner);
    rng(split_seed,'twister');
    is_test(ci(randperm(numel(ci), min(numel(ci), round(test_frac*nT*nV))))) = true;
else
    for r = 1:size(test_pairs,1)
        hit = abs(TT-test_pairs(r,1)) < 1e-9 & abs(VV-test_pairs(r,2)) < 1e-9;
        assert(any(hit(:)), 'test pair (%g K, %g %%) is not on the grid', test_pairs(r,1), test_pairs(r,2));
        is_test = is_test | hit;
    end
end
fprintf('\nTest curves (T K, vac %%):');
fprintf(' (%g, %.2f)', [TT(is_test).'; VV(is_test).']);
fprintf('\n');

T_col = T_list(:);   v_col = vac_list(:);   va_col = vac_actual(:);   nv_col = n_vac(:);
l_col = lambda_list(:);   Vg_col = V_TG_BG(:);   Ez_col = Ez(:);   fl_col = I_floor(:);

% one row per (gate point, T, vac)
[iL3, iT3, iV3] = ndgrid(1:nL, 1:nT, 1:nV);
iL3 = iL3(:);   iT3 = iT3(:);   iV3 = iV3(:);
I_abs = abs(real(I_all(:)));
IV = table(T_col(iT3), v_col(iV3), va_col(iV3), nv_col(iV3), l_col(iL3), Vg_col(iL3), Ez_col(iL3), ...
           I_abs, log10(max(I_abs,realmin)), Cq2_all(:), Cq1_all(:), CQ_all(:), CT_all(:), ...
           I_abs > fl_col(iT3), repmat(seed,numel(iL3),1), ...
           'VariableNames', {'T_K','vac_pct','vac_pct_actual','n_vac_atoms','lambda_eV','VTG_minus_VBG_V', ...
                             'Ez_V_per_A','I_A','log10_I','Cq_F_per_m','Cq_F_per_m2','CQ_F','CT_F', ...
                             'reliable','seed'});
test_row = is_test(sub2ind([nT nV], iT3, iV3));

% one row per (T, vac)
[iT2, iV2] = ndgrid(1:nT, 1:nV);
iT2 = iT2(:);   iV2 = iV2(:);
MT = table(T_col(iT2), v_col(iV2), va_col(iV2), nv_col(iV2), Ion(:), Ioff(:), Ion(:)./Ioff(:), ...
           log10(Ion(:)./Ioff(:)), CQ_avg(:), C_eff(:), tau_d(:), DED(:), SPD(:), repmat(seed,numel(iT2),1), ...
           'VariableNames', {'T_K','vac_pct','vac_pct_actual','n_vac_atoms','Ion_A','Ioff_A','Ion_Ioff', ...
                             'log10_Ion_Ioff','CQ_avg_F','C_eff_F','tau_d_s','DED_J','SPD_W','seed'});
test_m = is_test(:);

IV_train = IV(~test_row,:);   IV_test = IV(test_row,:);
M_train  = MT(~test_m,:);     M_test  = MT(test_m,:);
writetable(IV_train, fullfile(out_dir,'ML_IV_train.csv'));
writetable(IV_test,  fullfile(out_dir,'ML_IV_test.csv'));
writetable(M_train,  fullfile(out_dir,'ML_metrics_train.csv'));
writetable(M_test,   fullfile(out_dir,'ML_metrics_test.csv'));
test_curves = [TT(is_test) VV(is_test)];
save(fullfile(out_dir,'ML_dataset.mat'),'IV_train','IV_test','M_train','M_test','is_test','test_curves', ...
     'T_list','vac_list','lambda_list','V_TG_BG','split_seed','test_frac','seed','cap_model','C_ox', ...
     'V_DD','Vds','t_ox','eps_ox','L_dev','W_rib');
fprintf('TRAIN: %d I-V points, %d curves | TEST: %d I-V points, %d curves  ->  %s\n', ...
        height(IV_train), height(M_train), height(IV_test), height(M_test), out_dir);

%% ===================== Old-format files per (T, vac) =====================
if save_per_case
    pc = fullfile(out_dir,'per_case');
    if ~exist(pc,'dir'), mkdir(pc); end
    Vgs = lambda_list;                           % old plot scripts multiply by 1.26706
    for iv = 1:nV
        for iT = 1:nT
            I   = I_all(:,iT,iv).';   Cq1 = Cq1_all(:,iT,iv).';   Cq2 = Cq2_all(:,iT,iv).';
            Temp = T_list(iT);   vac_percent = vac_list(iv);   vac_atoms_used = vac_atoms{iv};
            tg = sprintf('edge_Vacancy_%s_temp%g', vac_tag{iv}, Temp);
            save(fullfile(pc,['gate_voltage_current_' tg '.mat']), 'Vgs','V_TG_BG','Ez','I', ...
                 'Length_Device','Temp','vac_percent','vac_atoms_used','seed');
            save(fullfile(pc,['quantum_capacitance_data_' tg '.mat']), 'Vgs','V_TG_BG','Cq1','Cq2', ...
                 'Temp','vac_percent');
        end
    end
end

R.T_list = T_list;     R.vac_list = vac_list;   R.V_TG_BG = V_TG_BG;   R.I = I_all;
R.Cq2 = Cq2_all;       R.CT = CT_all;           R.metrics = MT;        R.IV = IV;
R.is_test = is_test;   R.out_dir = out_dir;
end

%% ===================== Local functions =====================
function [Tk, Dk] = one_energy(Ek, eta, bigH, bigU, bigUbar, count1, count2, lambda_list, ...
                               h0, u0, l0, chi, kp_all, L_dev)
% T(E) and DOS (states/(eV*Angstrom), both spins) at one energy for all gate and vacancy points
[Sig1,Sig2,Gam1,Gam2] = lead_sigma(Ek, eta, bigH, bigU, bigUbar, count1, count2);
nL = numel(lambda_list);   nV = numel(kp_all);
Tk = zeros(nL,nV);   Dk = zeros(nL,nV);
for iv = 1:nV
    for il = 1:nL
        [Tk(il,iv), trG] = rgf_T_trG(Ek+1i*eta, lambda_list(il), h0, u0, l0, chi, kp_all{iv}, ...
                                     Sig1, Sig2, Gam1, Gam2, true);
        Dk(il,iv) = -imag(trG)/(pi*L_dev);
    end
end
end

function [Sig1, Sig2, Gam1, Gam2] = lead_sigma(E, eta, bigH, bigU, bigUbar, count1, count2)
% pristine semi-infinite leads (lambda = 0), same as Vgs_Ids_edge_vacancy_T.m
[g1,~] = lead_eps(bigH,bigUbar,eta,E,count1,count2);    % left lead
[g2,~] = lead_eps(bigH,bigU,  eta,E,count1,count2);     % right lead
Sig1 = bigUbar*g1*bigU;    Sig2 = bigU*g2*bigUbar;
Gam1 = 1i*(Sig1 - Sig1');  Gam2 = 1i*(Sig2 - Sig2');
end

function [T, trG] = rgf_T_trG(z, lam, h0, u0, l0, chi, kp, Sig1, Sig2, Gam1, Gam2, doDOS)
% Recursive Green's function over N slices. h0 = H_s,s, u0 = H_s,s+1, l0 = H_s+1,s of the
% pristine slice; gate term lam*chi on the diagonal. kp{s} = orbitals KEPT in slice s: the
% spin-orbitals of missing atoms are deleted, i.e. their rows/columns of H and their gate
% entries are zero and they take no part in transport or in the DOS.
% T = Tr[Gam1 G_1N Gam2 G_1N'];  trG = Tr[G^r] over the device (only if doDOS).
N = numel(kp);   n = size(h0,1);
gL = cell(1,N);   X = cell(1,N-1);   Lc = cell(1,N-1);
for s = 1:N
    k = kp{s};   m = numel(k);
    if m == n                                    % no vacancy in this slice
        A = -h0;          d = z - lam*chi;
    else                                         % vacancy orbitals deleted
        A = -h0(k,k);     d = z - lam*chi(k);
    end
    A(1:m+1:end) = A(1:m+1:end) + d.';          % z - H_ss - lam*chi
    if s == 1
        A = A - Sig1;                            % slices 1 and N are vacancy-free (n_buffer >= 1)
    else
        A = A - Lc{s-1}*X{s-1};                  % H_s,s-1 gL_s-1 H_s-1,s
    end
    if s == N, A = A - Sig2; end
    gL{s} = inv(A);                              % left-connected Green's function
    if s == 1
        G1N = gL{1};
    else
        G1N = (G1N*Uc)*gL{s};                    % G_1,s = G_1,s-1 H_s-1,s gL_s
    end
    if s < N
        k2 = kp{s+1};
        if m == n && numel(k2) == n
            Uc = u0;          Lc{s} = l0;
        else
            Uc = u0(k,k2);    Lc{s} = l0(k2,k);  % H_s,s+1 and H_s+1,s without the deleted orbitals
        end
        X{s} = gL{s}*Uc;
    end
end
T = real(sum(sum((Gam1*G1N).*(Gam2*G1N').')));  % Tr[Gam1 G_1N Gam2 G_1N']
trG = 0;
if doDOS
    G = gL{N};   trG = trace(G);
    for s = N-1:-1:1
        G   = gL{s} + X{s}*G*(Lc{s}*gL{s});      % G_s,s
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
