%% Fig_Temp_edge_vacancy.m
% I_DS vs (V_TG - V_BG) at 300, 77, 4 K for pristine, 0.85 % and 2.0 % edge vacancies.
% Reads the .mat files written by Vgs_Ids_edge_vacancy_T.m
clc; clear; close all;

T_list   = [300 77 4];
vac_tag  = {'0','0.85','2.0'};
vac_name = {'No vacancy','0.85% edge vacancy','2.0% edge vacancy'};

% ---- colours (RGB) ----
blue       = [0.000 0.447 0.698];
vermillion = [0.835 0.369 0.000];
green      = [0.000 0.620 0.451];
purple     = [0.459 0.439 0.702];
orange     = [0.902 0.624 0.000];
teal       = [0.000 0.502 0.502];
magenta    = [0.800 0.200 0.600];
brown      = [0.549 0.337 0.294];
nearBlack  = [0.150 0.150 0.150];
col  = {blue,    vermillion, green;         % pristine : 300, 77, 4 K
        purple,  orange,     teal;          % 0.85 %   : 300, 77, 4 K
        magenta, brown,      nearBlack};    % 2.0 %    : 300, 77, 4 K
lsty = {'-','--',':'};                      % line style  -> vacancy level
mrk  = {'s','o','^'};                       % marker      -> temperature

% ---- load (abs(real()) before any log) ----
V = cell(3,3);   Ids = cell(3,3);
for iv = 1:3
    for iT = 1:3
        S = load(sprintf('gate_voltage_current_edge_Vacancy_%s_temp%d.mat', vac_tag{iv}, T_list(iT)));
        V{iv,iT}   = S.V_TG_BG;              % already 1.26706*lambda_v
        Ids{iv,iT} = abs(real(S.I));
    end
end

%% ---- one figure per vacancy level (same layout as Fig13) ----
for iv = 1:3
    figure('Color',[1 1 1]);  hold on;  grid on;  box on;
    p1 = semilogy(V{iv,1}, Ids{iv,1}, [lsty{iv} mrk{1}], 'Color', col{iv,1}, 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    p2 = semilogy(V{iv,2}, Ids{iv,2}, [lsty{iv} mrk{2}], 'Color', col{iv,2}, 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    p3 = semilogy(V{iv,3}, Ids{iv,3}, [lsty{iv} mrk{3}], 'Color', col{iv,3}, 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
    set(gca, 'YScale', 'log');           % semilogy after 'hold on' keeps a linear axis
    xlabel('V_{TG} - V_{BG} (V)', 'FontSize', 20, 'FontWeight', 'bold');
    ylabel('I_{DS} (A)', 'FontSize', 20, 'FontWeight', 'bold');
    title(vac_name{iv});
    legend([p1 p2 p3], {'T = 300 K','T = 77 K','T = 4 K'}, 'Location', 'best', 'FontSize', 16);
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'YMinorGrid', 'on', 'XMinorGrid', 'on', 'FontSize', 18);
    set(gcf, 'Position', [100 100 1000 700]);
    xlim([min(V{iv,1}) max(V{iv,1})]);
end

%% ---- all 9 curves ----
figure('Color',[1 1 1]);  hold on;  grid on;  box on;
p1 = semilogy(V{1,1}, Ids{1,1}, '-s',  'Color', blue,       'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p2 = semilogy(V{1,2}, Ids{1,2}, '-o',  'Color', vermillion, 'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p3 = semilogy(V{1,3}, Ids{1,3}, '-^',  'Color', green,      'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p4 = semilogy(V{2,1}, Ids{2,1}, '--s', 'Color', purple,     'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p5 = semilogy(V{2,2}, Ids{2,2}, '--o', 'Color', orange,     'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p6 = semilogy(V{2,3}, Ids{2,3}, '--^', 'Color', teal,       'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p7 = semilogy(V{3,1}, Ids{3,1}, ':s',  'Color', magenta,    'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p8 = semilogy(V{3,2}, Ids{3,2}, ':o',  'Color', brown,      'LineWidth', 1.5, 'MarkerFaceColor', 'w');
p9 = semilogy(V{3,3}, Ids{3,3}, ':^',  'Color', nearBlack,  'LineWidth', 1.5, 'MarkerFaceColor', 'w');
set(gca, 'YScale', 'log');
xlabel('V_{TG} - V_{BG} (V)', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('I_{DS} (A)', 'FontSize', 20, 'FontWeight', 'bold');
lg = legend([p1 p2 p3 p4 p5 p6 p7 p8 p9], ...
       {'300 K, pristine','77 K, pristine','4 K, pristine', ...
        '300 K, 0.85%','77 K, 0.85%','4 K, 0.85%', ...
        '300 K, 2.0%','77 K, 2.0%','4 K, 2.0%'}, ...
       'Location', 'best', 'FontSize', 13);
try, set(lg, 'NumColumns', 3); end        % R2018a or newer
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.5, 'YMinorGrid', 'on', 'XMinorGrid', 'on', 'FontSize', 18);
set(gcf, 'Position', [100 100 1100 750]);
xlim([min(V{1,1}) max(V{1,1})]);

%% ---- quick numbers for the text ----
fprintf('%-18s %6s %12s %12s %12s\n', 'case', 'T (K)', 'I_ON (A)', 'I_OFF (A)', 'ON/OFF');
for iv = 1:3
    for iT = 1:3
        Ion = Ids{iv,iT}(1);   Ioff = Ids{iv,iT}(end);
        fprintf('%-18s %6d %12.4e %12.4e %12.3e\n', vac_name{iv}, T_list(iT), Ion, Ioff, Ion/Ioff);
    end
end
