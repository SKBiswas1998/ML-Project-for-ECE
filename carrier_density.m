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
