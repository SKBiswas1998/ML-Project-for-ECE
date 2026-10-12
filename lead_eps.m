%function for calculating lead Green's function
function [G,countG]=lead_eps(h1,u,eta,st,endcount1,endcount2)
countG=0;
step=100;%large so that only one energy
countk=1000;%no of iterations
err=1e-3;%convergence error
ud=ctranspose(u);
for E1=st:step:st
    countG=countG+1;

    E2(countG)=E1+i*eta;
    E3(:,:,countG)=E2(countG)*eye(endcount1);
    
    for k=1:countk
        if k==1
            epss(:,:,countG,k)=h1;
            eps(:,:,countG,k)=h1;
            alpha(:,:,countG,k)=u;
            beta(:,:,countG,k)=ud;
        else
        epss(:,:,countG,k)=epss(:,:,countG,k-1)+alpha(:,:,countG,k-1)*(E3(:,:,countG)-eps(:,:,countG,k-1))^(-1)*beta(:,:,countG,k-1);
        eps(:,:,countG,k)=eps(:,:,countG,k-1)+beta(:,:,countG,k-1)*(E3(:,:,countG)-eps(:,:,countG,k-1))^(-1)*alpha(:,:,countG,k-1)+alpha(:,:,countG,k-1)*(E3(:,:,countG)-eps(:,:,countG,k-1))^(-1)*beta(:,:,countG,k-1) ;
        alpha(:,:,countG,k)=alpha(:,:,countG,k-1)*(E3(:,:,countG)-eps(:,:,countG,k-1))^(-1)*alpha(:,:,countG,k-1);
        beta(:,:,countG,k)=beta(:,:,countG,k-1)*(E3(:,:,countG)-eps(:,:,countG,k-1))^(-1)*beta(:,:,countG,k-1);
        if abs(max(max(alpha(:,:,countG,k))))<err && abs(max(max(beta(:,:,countG,k))))<err
%            if norm(alpha(:,:,countG,k-1))<err && norm(beta(:,:,countG,k-1))<err 
            aa=epss(:,:,countG,k);
            break;
        end
        end
    end
%     G(:,:,count)=(E(:,:,count)-epss(:,:,count,k))^(-1);
    G(:,:,countG)=(E3(:,:,countG)-aa)^(-1);
%     [v(:,:,countG), G(:,:,countG)]=eig(G(:,:,countG));
    
% [v(:,:,countG), G(:,:,countG)]=eig(G(:,:,countG));
 end