function [kk, kd] = point_grid(kb, mF, mT, dE_f, dE_c)
% lattice indices k (E = muS + k*dE_f) of one sweep point: fine part mu_D - mF ... mu_S + mF
% (kd = true, density and DOS) and coarse tails out to mT (transmission only)
m_c = round(dE_c/dE_f);   kf = ceil(mF/dE_f);
kk = (-kb-kf):kf;                                          % mu_D and mu_S on the grid
nc = max(0, ceil((mT - kf*dE_f)/dE_c));
kk = [kk(1)-(nc:-1:1)*m_c, kk, kk(end)+(1:nc)*m_c];
kd = (kk >= -kb-kf) & (kk <= kf);
end
