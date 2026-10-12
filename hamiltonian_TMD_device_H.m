function [H]=hamiltonian_TMD_device_H(h,u,Unit_Cells)
a=eye(Unit_Cells);
b=diag(ones(1,Unit_Cells-1),1);
c1=diag(ones(1,Unit_Cells-1),-1);
H=kron(a,h)+kron(b,u)+kron(c1,u');



%%%% kron function is so useful 