function J = landauer_window(E, Tr, muS, muD, kT, n_kT)
% J = int T(E)[f_S - f_D] dE (eV); T(E) linearly interpolated on a grid finer than kB*T
lo = max(E(1), muD - n_kT*kT);   hi = min(E(end), muS + n_kT*kT);
de = min(kT/10, min(diff(E))/4);
x  = linspace(lo, hi, max(201, ceil((hi-lo)/de)+1));
fS = 1./(1 + exp((x-muS)/kT));   fD = 1./(1 + exp((x-muD)/kT));
J  = trapz(x, interp1(E(:), Tr(:), x(:)).'.*(fS - fD));
end
