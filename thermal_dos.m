function Dth = thermal_dos(E, DOS, Ef, kT, n_kT)
% Dth = int D(E) F_T(E-Ef) dE, F_T = sech^2((E-Ef)/2kT)/(4kT)  ->  C_Q = q^2*Dth
lo = max(E(1), Ef - n_kT*kT);   hi = min(E(end), Ef + n_kT*kT);
de = min(kT/10, min(diff(E))/4);
x  = linspace(lo, hi, max(201, ceil((hi-lo)/de)+1));
FT = sech((x-Ef)/(2*kT)).^2/(4*kT);
Dth = trapz(x, interp1(E(:), DOS(:), x(:)).'.*FT);
end
