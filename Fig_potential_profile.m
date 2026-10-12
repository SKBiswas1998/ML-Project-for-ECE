%% Fig_potential_profile.m
% Potential profile along the channel for several gate voltages, from the
% self-consistent NEGF-Poisson results of Vgs_Ids_edge_vacancy_SCF.m (same folder).
%  (a) potential energy of the top and bottom Te sublayers, U = -/+lambda_v + U_H
%      (gate-induced staggered potential + self-consistent Hartree potential)
%  (b) Hartree potential U_H = -qV, V = Poisson solution for the channel charge
%      (average over the cross-section of the ribbon)
%  (c) carrier density along the channel (sheet density, cm^-2)
% x = 0 is the source end of the channel. Grey: source/drain leads (ungated
% pristine ribbon, U = 0; their density is not part of the NEGF-Poisson solution).
% Works while the SCF run is still going: it then uses the finished points of
% the checkpoint file.
clc; clear; close all;

%% ===================== Settings =====================
run_name = 'edge_vacancy_SCF_0.85_2.0_gate';   % run_name of the SCF run (file prefix)
V_show   = [0 0.5 1.0 1.5 1.9];   % V: V_TG - V_BG (gate sweep) or V_DS (drain sweep); nearest computed points
T_plot   = 300;                   % K
vac_show = [0.85 2.0];            % edge vacancy in % (one figure each)
L_lead   = 2;                     % nm of source/drain lead drawn on each side (0: channel only)
rho_mode = 'total';               % 'total': n + p  |  'net': p - n (charge density in the Poisson equation)
save_png = true;                  % potential_profile_vac<..>_T<..>.png, 300 dpi

%% ===================== Colours =====================
blue       = [0.000 0.447 0.698];   vermillion = [0.835 0.369 0.000];   green     = [0.000 0.620 0.451];
purple     = [0.459 0.439 0.702];   orange     = [0.902 0.624 0.000];   teal      = [0.000 0.502 0.502];
magenta    = [0.800 0.200 0.600];   brown      = [0.549 0.337 0.294];   nearBlack = [0.150 0.150 0.150];
cols = {blue, vermillion, green, purple, orange, teal, magenta, brown, nearBlack};   % one per voltage

%% ===================== Load =====================
vars = {'lambda_pts','Vds_pts','T_list','vac_list','U_at_all','n_at_all','p_at_all','nit_all'};
if exist([run_name '_all.mat'],'file') == 2
    S = load([run_name '_all.mat'], vars{:}, 'Device_atomic_position','type_of','W_rib','gate_factor','vac_atoms');
    pos = S.Device_atomic_position;   type_of = S.type_of;   W_rib = S.W_rib;
    gate_factor = S.gate_factor;      vac_atoms = S.vac_atoms;
elseif exist([run_name '_checkpoint.mat'],'file') == 2          % run still going
    S = load([run_name '_checkpoint.mat'], vars{:}, 'run_sig');
    % run_sig = [seed n_edge_rows n_buffer ZBNR Unit_Cells U_vac eta muS E_N0 dE_f dE_c E_fine n_kT t_ox eps_ox eps_ch ...]
    [pos, type_of, W_rib, t_wte2] = wte2_geometry(S.run_sig(4), S.run_sig(5));
    gate_factor = (2*S.run_sig(14)*S.run_sig(16)/S.run_sig(15) + t_wte2)/t_wte2;
    vac_atoms = cell(1, numel(S.vac_list));                     % not stored in the checkpoint
    fprintf('Run not finished: %d of %d points available\n', nnz(S.nit_all), numel(S.nit_all));
else
    error('No %s_all.mat or %s_checkpoint.mat here: check run_name and the current folder', run_name, run_name);
end
done = S.nit_all > 0;
lam  = S.lambda_pts(:);   Vds = S.Vds_pts(:);
gate_sweep = numel(unique(lam)) > 1;
if gate_sweep, sv = gate_factor*lam;  sv_tex = 'V_{TG} - V_{BG}';  sv_txt = 'V_TG-V_BG (V)';
else,          sv = Vds;              sv_tex = 'V_{DS}';           sv_txt = 'V_DS (V)';       end

%% ===================== Atomic columns along x =====================
n_at = size(pos,1);
x_nm = (pos(:,1) - min(pos(:,1)))/10;                % nm from the source end of the channel
[xc, ~, col] = unique(round(x_nm*1e4)/1e4);         % atomic columns, a_x/2 apart
xc = xc(:);   col = col(:);   ncol = numel(xc);
dxc = mean(diff(xc));   Lch = xc(end);
bot = type_of(:) <= 2;                               % Te1, Te2: bottom Te sublayer (+lambda_v)
top = type_of(:) == 3 | type_of(:) == 4;             % Te3, Te4: top Te sublayer    (-lambda_v)
col_area = W_rib*dxc*10;                             % Angstrom^2 per atomic column
nL   = round(L_lead/dxc);
xS   = (-nL:-1)'*dxc;   xD = Lch + (1:nL)'*dxc;      % lead columns (U = 0)
xall = [xS; xc; xD];    z_ld = zeros(nL,1);
xl   = [xall(1) xall(end)] + [-1 1]*dxc/2;
mid  = xc > 0.25*Lch & xc < 0.75*Lch;                % central half of the channel (table)
avg  = @(v) mean(v(~isnan(v)));

%% ===================== One figure per vacancy level =====================
[~, iT] = min(abs(S.T_list - T_plot));   Temp = S.T_list(iT);
for vs = vac_show(:)'
    iv = find(abs(S.vac_list - vs) < 1e-9, 1);
    if isempty(iv), warning('%g %% vacancy is not in this run', vs); continue; end
    avail = find(done(:,iT,iv));
    if isempty(avail), warning('no finished point for %g %%, %d K yet', vs, Temp); continue; end
    [~, id] = min(abs(sv(avail) - V_show(:)'), [], 1);   % nearest computed point to each V_show
    ips = unique(avail(id));   ips = ips(1:min(end,numel(cols)));   nk = numel(ips);

    ok = true(n_at,1);   ok(vac_atoms{iv}) = false;       % no potential "on" a missing atom
    Ut = zeros(ncol,nk);   Ub = Ut;   UH = Ut;   rho = Ut;
    for k = 1:nk
        ip = ips(k);   U = S.U_at_all(:,ip,iT,iv);        % Hartree potential per atom (eV)
        Ub(:,k) = accumarray(col(bot&ok),  lam(ip) + U(bot&ok), [ncol 1], @mean, NaN);
        Ut(:,k) = accumarray(col(top&ok), -lam(ip) + U(top&ok), [ncol 1], @mean, NaN);
        UH(:,k) = accumarray(col(ok), U(ok), [ncol 1], @mean, NaN);
        n = S.n_at_all(:,ip,iT,iv);   p = S.p_at_all(:,ip,iT,iv);
        if strcmp(rho_mode,'net'), d = p - n; else, d = n + p; end
        rho(:,k) = accumarray(col, d, [ncol 1])/col_area*1e16;        % cm^-2
    end
    UHm = 1e3*UH;                                         % meV
    lab = cell(1,nk);
    for k = 1:nk
        lab{k} = sprintf('%s = %.2f V', sv_tex, sv(ips(k)));
        if gate_sweep && lam(ips(k)) == 0,       lab{k} = [lab{k} ' (ON)'];  end
        if gate_sweep && ips(k) == numel(lam),   lab{k} = [lab{k} ' (OFF)']; end
    end

    figure('Color',[1 1 1]);   set(gcf, 'Position', [100 60 1000 1050]);

    % ---- (a) potential energy of the Te sublayers ----
    ya = [min([Ut(:); Ub(:); 0]) max([Ut(:); Ub(:); 0])];
    ya = ya + [-1 1]*0.10*max(diff(ya), 0.05);
    pa = axes('Position', [0.12 0.70 0.56 0.26]);   hold on;   box on;
    lead_patch(xl, dxc, Lch, nL, ya);
    for k = 1:nk
        p_top = plot(xall, [z_ld; Ut(:,k); z_ld], '-',  'Color', cols{k}, 'LineWidth', 1.8);   % top Te
        p_bot = plot(xall, [z_ld; Ub(:,k); z_ld], '--', 'Color', cols{k}, 'LineWidth', 1.8);   % bottom Te
    end
    xlim(xl);   ylim(ya);   set(pa, 'XTickLabel', [], 'FontSize', 16);
    ylabel('U (eV)', 'FontSize', 18, 'FontWeight', 'bold');
    text(0.015, 0.90, '(a)', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');
    if nL > 0
        text(mean([xl(1) -dxc/2]),     ya(2) - 0.08*diff(ya), 'S', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
        text(mean([Lch+dxc/2 xl(2)]),  ya(2) - 0.08*diff(ya), 'D', 'HorizontalAlignment', 'center', 'FontSize', 16, 'FontWeight', 'bold');
    end
    title(sprintf('%g%% edge vacancy, T = %d K', vs, Temp), 'FontSize', 16);

    % legend in an invisible axes to the right of panel (a), so the three panels stay aligned
    pl = axes('Position', [0.70 0.70 0.29 0.26], 'Visible', 'off');   hold on;
    hl = [];
    for k = 1:nk, hl(end+1) = plot(NaN, NaN, '-', 'Color', cols{k}, 'LineWidth', 1.8); end
    hl(end+1) = plot(NaN, NaN, 'k-',  'LineWidth', 1.8);
    hl(end+1) = plot(NaN, NaN, 'k--', 'LineWidth', 1.8);
    ll = [lab, {'top Te sublayer', 'bottom Te sublayer'}];
    if ~isempty(vac_atoms{iv})
        hl(end+1) = plot(NaN, NaN, 'kx', 'MarkerSize', 9, 'LineWidth', 1.5);   ll{end+1} = 'edge vacancy';
    end
    set(pl, 'Visible', 'off');
    legend(hl, ll, 'Location', 'northwest', 'FontSize', 13);

    % ---- (b) Hartree potential (Poisson) ----
    yb = [min([UHm(:); 0]) max([UHm(:); 0])];
    yb = yb + [-1 1]*0.12*max(diff(yb), 0.1);
    pb = axes('Position', [0.12 0.405 0.56 0.26]);   hold on;   box on;
    lead_patch(xl, dxc, Lch, nL, yb);
    for k = 1:nk
        plot(xall, [z_ld; UHm(:,k); z_ld], '-', 'Color', cols{k}, 'LineWidth', 1.8);
    end
    xlim(xl);   ylim(yb);   set(pb, 'XTickLabel', [], 'FontSize', 16);
    ylabel('U_H (meV)', 'FontSize', 18, 'FontWeight', 'bold');
    text(0.015, 0.90, '(b)', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

    % ---- (c) carrier density ----
    if strcmp(rho_mode,'net'), yc = [min(rho(:)) max(rho(:))];   yc = yc + [-1 1]*0.10*max(diff(yc), eps);
    else,                      yc = [0 1.12*max(max(rho(:)), eps)];   end
    pc = axes('Position', [0.12 0.11 0.56 0.26]);   hold on;   box on;
    lead_patch(xl, dxc, Lch, nL, yc);
    for k = 1:nk
        plot(xc, rho(:,k), '-', 'Color', cols{k}, 'LineWidth', 1.5);
    end
    va = vac_atoms{iv};
    if ~isempty(va)
        plot(x_nm(va), yc(1) + 0.06*diff(yc)*ones(numel(va),1), 'kx', 'MarkerSize', 9, 'LineWidth', 1.5);
    end
    xlim(xl);   ylim(yc);   set(pc, 'FontSize', 16);
    xlabel('x (nm)', 'FontSize', 18, 'FontWeight', 'bold');
    if strcmp(rho_mode,'net'), ylabel('p - n (cm^{-2})', 'FontSize', 18, 'FontWeight', 'bold');
    else,                      ylabel('n + p (cm^{-2})', 'FontSize', 18, 'FontWeight', 'bold'); end
    text(0.015, 0.90, '(c)', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

    if save_png
        print(gcf, sprintf('potential_profile_vac%s_T%d.png', strrep(sprintf('%g',vs),'.','p'), Temp), '-dpng', '-r300');
    end

    % ---- numbers for the text (central half of the channel) ----
    fprintf('\n%g %% edge vacancy, T = %d K (central half of the channel)\n', vs, Temp);
    fprintf('%15s %11s %11s %11s %15s %14s\n', sv_txt, 'lambda (eV)', 'U_top (eV)', 'U_bot (eV)', 'max|U_H| (meV)', 'rho (cm^-2)');
    for k = 1:nk
        fprintf('%15.3f %11.3f %11.4f %11.4f %15.3f %14.3e\n', sv(ips(k)), lam(ips(k)), avg(Ut(mid,k)), ...
                avg(Ub(mid,k)), max(abs(UHm(:,k))), avg(rho(mid,k)));
    end
end

%% ===================== Local functions =====================
function lead_patch(xl, dxc, Lch, nL, yl)
% grey boxes for the source (x < 0) and drain (x > L) leads
if nL < 1, return; end
g = [0.92 0.92 0.92];
patch([xl(1) -dxc/2 -dxc/2 xl(1)],        [yl(1) yl(1) yl(2) yl(2)], g, 'EdgeColor', 'none');
patch([Lch+dxc/2 xl(2) xl(2) Lch+dxc/2],  [yl(1) yl(1) yl(2) yl(2)], g, 'EdgeColor', 'none');
end

function [pos, type_of, W_rib, t_wte2] = wte2_geometry(ZBNR, Unit_Cells)
% atom positions of Vgs_Ids_edge_vacancy_SCF.m (used when only the checkpoint exists)
nn = (ZBNR-4)/4 + 1;   num_cell = ZBNR - nn;
ax = 3.49794;   by = 6.33827;
af = [0.00000044, 1.31473454, 6.35624338;     % Te1
      1.74897061, 4.43335644, 5.73543738;     % Te2
     -0.00000002, 5.44760823, 9.92929379;     % Te3
      1.74897108, 2.22834683, 9.30776171;     % Te4
      1.74897050, 6.06754902, 7.93340128;     % W1
      0.00000040, 3.81415493, 7.73232446];    % W2
pos = zeros(6*num_cell*Unit_Cells, 3);   c = 0;
for i = 1:Unit_Cells
    for j = 1:num_cell
        pos(6*c+(1:6),:) = af + repmat([(i-1)*ax, (j-1)*by, 0], 6, 1);
        c = c + 1;
    end
end
type_of = repmat((1:6)', num_cell*Unit_Cells, 1);
W_rib = num_cell*by;   t_wte2 = af(3,3) - af(2,3);
end
