function P = poisson_setup(x_at, y_at, z_at, hx, hy, hz, pad_y, a_half, t_ox, eps_ch, eps_ox, eps0)
% 3-D finite-difference Poisson solver  div(eps grad V) = -rho/eps0  (V in volt, rho in e, lengths in A)
% grounded gate planes at z_c +/- (a_half + t_ox) (Dirichlet), Neumann on the other faces.
xg = (min(x_at)-hx):hx:(max(x_at)+hx+1e-9);
yg = (min(y_at)-pad_y):hy:(max(y_at)+pad_y+1e-9);
zc = (min(z_at)+max(z_at))/2;   nzh = max(4, round((a_half+t_ox)/hz));   hz = (a_half+t_ox)/nzh;
zg = zc + (-nzh:nzh)*hz;
xg = xg(:);  yg = yg(:);  zg = zg(:);
nx = numel(xg);  ny = numel(yg);  nz = numel(zg);  nn = nx*ny*nz;
[X,Y,Z] = ndgrid(xg,yg,zg);
inlayer = X >= min(x_at)-hx & X <= max(x_at)+hx & Y >= min(y_at)-1 & Y <= max(y_at)+1 & abs(Z-zc) <= a_half+1e-9;
eps_n = eps_ox*ones(nn,1);   eps_n(inlayer(:)) = eps_ch;
isDir = false(nx,ny,nz);   isDir(:,:,1) = true;   isDir(:,:,nz) = true;   isDir = isDir(:);
% operator
idx = reshape(1:nn,[nx ny nz]);
rows = zeros(0,1);  cols = rows;  vals = rows;  diagv = zeros(nn,1);
step = {[1 0 0], hx; [0 1 0], hy; [0 0 1], hz};
for d = 1:3
    sh = step{d,1};  h = step{d,2};
    p  = idx(1:nx-sh(1), 1:ny-sh(2), 1:nz-sh(3));   p = p(:);
    qn = idx(1+sh(1):nx, 1+sh(2):ny, 1+sh(3):nz);   qn = qn(:);
    ef = 0.5*(eps_n(p) + eps_n(qn))/h^2;
    rows = [rows; p; qn];   cols = [cols; qn; p];   vals = [vals; ef; ef];
    diagv = diagv - accumarray(p, ef, [nn 1]) - accumarray(qn, ef, [nn 1]);
end
A = sparse(rows,cols,vals,nn,nn) + spdiags(diagv,0,nn,nn);
keep = double(~isDir);
A = spdiags(keep,0,nn,nn)*A + spdiags(1-keep,0,nn,nn);      % Dirichlet rows: V = 0
[Lf,Uf,Pf,Qf,Rf] = lu(A);
% trilinear deposition / interpolation matrix W (atoms x nodes)
na = numel(x_at);
ix = min(max(floor((x_at-xg(1))/hx)+1,1),nx-1);   fx = (x_at-xg(ix))/hx;
iy = min(max(floor((y_at-yg(1))/hy)+1,1),ny-1);   fy = (y_at-yg(iy))/hy;
iz = min(max(floor((z_at-zg(1))/hz)+1,1),nz-1);   fz = (z_at-zg(iz))/hz;
r = zeros(8*na,1);  c = r;  w = r;   m = 0;
for dx = 0:1, for dy = 0:1, for dz = 0:1
    wgt = (dx*fx + (1-dx)*(1-fx)).*(dy*fy + (1-dy)*(1-fy)).*(dz*fz + (1-dz)*(1-fz));
    r(m+(1:na)) = (1:na)';   c(m+(1:na)) = sub2ind([nx ny nz], ix+dx, iy+dy, iz+dz);   w(m+(1:na)) = wgt;
    m = m + na;
end, end, end
W = sparse(r,c,w,na,nn);
scale = 1/(eps0*hx*hy*hz);
P.W = W;   P.nx = nx;   P.ny = ny;   P.nz = nz;   P.hx = hx;   P.hy = hy;   P.hz = hz;   P.zgate = a_half+t_ox;
P.xg = xg;   P.yg = yg;   P.zg = zg;   P.eps = reshape(eps_n,[nx ny nz]);
P.solve = @(rho_at) Qf*(Uf\(Lf\(Pf*(Rf\(-scale*keep.*(W'*rho_at))))));   % V at the nodes (volt)
end
