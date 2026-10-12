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
