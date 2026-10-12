function[H_1,store_ind]=dismantle_hamiltonian(caption,proj)

sum_proj=sum(proj);
[HB,unique_trans_vec]=extract_hamil_from_wannier(caption);
size_trans=length(unique_trans_vec(:,1));
H_1=cell(size_trans,sum_proj);

for hb=1:size_trans
    H=HB{hb,1};
    proj_end1=0;
    proj_start1=1;
    inc=0;
    store_ind=[];
    for at=1:length(proj)
        proj_end=0;
    				proj_start=1;
        proj_end1=proj_end1+proj(at);
        for at1=1:length(proj)
            %proj_new=1;
            inc=inc+1;
            proj_end=proj_end+proj(at1);
            H_1{hb,inc}=H(proj_start1:proj_end1,proj_start:proj_end);
            proj_start=proj_start+proj(at1);
            store_ind(inc,:)=[at,at1];
        end
        proj_start1=proj_start1+proj(at);
    end
    hb;
end
end