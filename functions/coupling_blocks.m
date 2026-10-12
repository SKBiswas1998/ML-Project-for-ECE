function [Uc, Lc] = coupling_blocks(u0, l0, rm)
% inter-slice blocks H_{s,s+1} (Uc{s}) and H_{s+1,s} (Lc{s}) with the removed orbitals decoupled
N = numel(rm);   Uc = cell(1,N-1);   Lc = cell(1,N-1);
for s = 1:N-1
    U = u0;   L = l0;
    if ~isempty(rm{s}) || ~isempty(rm{s+1})
        U(rm{s},:) = 0;     U(:,rm{s+1}) = 0;
        L(rm{s+1},:) = 0;   L(:,rm{s}) = 0;
    end
    Uc{s} = U;   Lc{s} = L;
end
end
