function[bigH,bigU,bigUbar]=cal_H_U_Ubar(caption,cell_pos,cell_pos1,atomst,atomst1,proj,HNN,Vext)

   %%%

   %caption= name comes before _hr.dat

   %cell_pos= lattice vector of unitcells making nano ribbon(N by 3 matrix)

    %cell_pos1= lattice vector of unitcells making nanoribbon ajdoint to nanoribbon(N by 3 matrix)

    %atomst= atoms located at each cell_pos(N by no. of atom matrix)

    %proj=number of orbital  centered around each atom(1 by no. of atom matrix)

    %HNN=highest neighbour to include(can be any value between 1 to infinite)

    %Vext= Ef

    passivate=0;

[HB,unique_trans_vec]=extract_hamil_from_wannier(caption);

[cellr,cellc]=size(cell_pos);

[H_1,store_ind]=dismantle_hamiltonian(caption,proj);


bigU=[];

bigH=[];


xu=0;

gd=0;

p_end1=0;

p_start1=1;

for i=1:cellr

    home_cell=cell_pos(i,:);

    p_endd=0;

    p_start=1;

    for j=1:cellr

        away_cell=cell_pos(j,:);

        dis_lat_vec=away_cell-home_cell;

        if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

            [check,dr]=ismember(dis_lat_vec,unique_trans_vec,'rows');

            if check

                    %if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

                p_e1=0;

                p_s11=1;

                p_e11=0;

                bigHh=[];

                for h=1:length(proj)

                    p_s=1;

    																	p_e=0;

                    for g=1:length(proj)


                        atom_index=[atomst(i,h), atomst(j,g)];

                        [check1,hr]=ismember(atom_index,store_ind,'rows');


                        if check1 

                            H=H_1{dr,hr};

                         			[Hzr,Hzc]=size(H);

                        				p_e11=p_e1+Hzr;


                        				p_e=p_e+Hzc;

                        				col_range1=[p_s,p_e];

                        				row_range1=[p_s11,p_e11];

                        				bigHh(p_s11:p_e11,p_s:p_e)=H;

                        				p_s=p_s+Hzc;

                        end 

                     end

                     %if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                    % end

                end

                [Hzrr,Hzcc]=size(bigHh);

               		p_end=p_end1+Hzrr;

                p_endd=p_endd+Hzcc;  

                final_range= [p_start1,p_end;p_start,p_endd];

                bigH(p_start1:p_end,p_start:p_endd)=bigHh;

        									p_start=p_start+Hzcc;

            elseif check==0

                    %if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

                 	p_e1=0;

			p_s11=1;

p_e11=0;

bigHh=[];

                 for h=1:length(proj)

                     p_s=1;

    																	p_e=0;

                     for g=1:length(proj)

                         atom_index=[atomst(i,h), atomst(j,g)];

                         [check1,hr]=ismember(atom_index,store_ind,'rows');

                         if check1 

                             atominn=[atomst(i,h), atomst(j,g)];

                        				 %H=H_1{dr,hr};

                         				Hzr=proj(atominn(1));

                         Hzc=proj(atominn(2));

                         H=zeros(Hzr,Hzc);

                        p_e11=p_e1+Hzr;


                        p_e=p_e+Hzc;

                        col_range1=[p_s,p_e];

                        row_range1=[p_s11,p_e11];

                        bigHh(p_s11:p_e11,p_s:p_e)=H;

                        p_s=p_s+Hzc;

                         end 

                     end

                      % if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                     %end

                 end

                      [Hzrr,Hzcc]=size(bigHh);

               								 p_end=p_end1+Hzrr;

                        p_endd=p_endd+Hzcc;  

                       final_range= [p_start1,p_end;p_start,p_endd];


                        bigH(p_start1:p_end,p_start:p_endd)=bigHh;

      						  p_start=p_start+Hzcc;

                   % end

             end


           else

               p_e1=0;

			p_s11=1;

p_e11=0;

bigHh=[];

                 for h=1:length(proj)

                     p_s=1;

    																	p_e=0;

                     for g=1:length(proj)

                         atom_index=[atomst(i,h), atomst(j,g)];

                         [check1,hr]=ismember(atom_index,store_ind,'rows');

                         if check1 

                             atominn=[atomst(i,h), atomst(j,g)];

                        				 %H=H_1{dr,hr};

                         				Hzr=proj(atominn(1));

                         Hzc=proj(atominn(2));

                         H=zeros(Hzr,Hzc);

                        p_e11=p_e1+Hzr;


                        p_e=p_e+Hzc;

                        col_range1=[p_s,p_e];

                        row_range1=[p_s11,p_e11];

                        bigHh(p_s11:p_e11,p_s:p_e)=H;

                        p_s=p_s+Hzc;

                         end 

                     end

                      % if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                     %end

                 end

                      [Hzrr,Hzcc]=size(bigHh);

               								 p_end=p_end1+Hzrr;

                        p_endd=p_endd+Hzcc;  

                       final_range= [p_start1,p_end;p_start,p_endd];


                        bigH(p_start1:p_end,p_start:p_endd)=bigHh;

      						  p_start=p_start+Hzcc;

           end


    end

    p_start1=p_end+1;

    p_end1=p_end;

    end


diag(bigH);




[cell1r,cell1c]=size(cell_pos1);


p_end1=0;

p_start1=1;

for i=1:cellr

    home_cell=cell_pos(i,:);

    p_endd=0;

    p_start=1;

    for j=1:cell1r

        away_cell=cell_pos1(j,:);

        dis_lat_vec=away_cell-home_cell;

           if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

           [ check,dr]=ismember(dis_lat_vec,unique_trans_vec,'rows');

             if check

                    %if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

                 %if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2)<=HNN

			p_e1=0;

			p_s11=1;

p_e11=0;

bigHh=[];

                 for h=1:length(proj)

                     p_s=1;

    																	p_e=0;

                     for g=1:length(proj)

                         atom_index=[atomst(i,h), atomst1(j,g)];

                         [check1,hr]=ismember(atom_index,store_ind,'rows');

                         if check1 

                             atom_index=[atomst(i,h), atomst1(j,g)];

                        				 H=H_1{dr,hr};

                         				[Hzr,Hzc]=size(H);

                        p_e11=p_e1+Hzr;

                        gd=gd+1;

                        p_e=p_e+Hzc;

                        [i,h;j,g];

                        col_range1=[p_s,p_e];

                        row_range1=[p_s11,p_e11];

                        bigHh(p_s11:p_e11,p_s:p_e)=H;

                        p_s=p_s+Hzc;

                       % bigHh((h-1)*Hzr+1:h*Hzr,(g-1)*Hzc+1:g*Hzc)=H;

                         end 

                     end

                     %if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                     %end

                 end

                  %[Hzrr,Hzcc]=size(bigHh)

                 % bigH((i-1)*Hzrr+1:i*Hzrr,(j-1)*Hzcc+1:j*Hzcc)=bigHh;

                      [Hzrr,Hzcc]=size(bigHh);

               								 p_end=p_end1+Hzrr;

                        gd=gd+1;

                        p_endd=p_endd+Hzcc;  

                       final_range= [p_start1,p_end;p_start,p_endd];

                        %bigH((i-1)*Hzrr+1:i*Hzrr,(j-1)*Hzcc+1:j*Hzcc)=bigHh;

                        bigU(p_start1:p_end,p_start:p_endd)=bigHh;

        p_start=p_start+Hzcc;


             else

                   % if abs(dis_lat_vec(1)) <= HNN || abs(dis_lat_vec(2))<=HNN

                 	p_e1=0;

			p_s11=1;

p_e11=0;

bigHh=[];

                 for h=1:length(proj)

                     p_s=1;

    																	p_e=0;

                     for g=1:length(proj)

                         atom_index=[atomst(i,h), atomst1(j,g)];

                         [check1,hr]=ismember(atom_index,store_ind,'rows');

                         if check1 


                        				 atominn=[atomst(i,h), atomst1(j,g)];

                        				 %H=H_1{dr,hr};

                         				Hzr=proj(atominn(1));

                         Hzc=proj(atominn(2));


                         H=zeros(Hzr,Hzc);

                        p_e11=p_e1+Hzr;


                        p_e=p_e+Hzc;

                        col_range1=[p_s,p_e];

                        row_range1=[p_s11,p_e11];

                        bigHh(p_s11:p_e11,p_s:p_e)=H;

                        p_s=p_s+Hzc;

                         end 

                     end

                       %if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                     %end

                 end

                      [Hzrr,Hzcc]=size(bigHh);

               								 p_end=p_end1+Hzrr;

                        p_endd=p_endd+Hzcc;  

                       final_range= [p_start1,p_end;p_start,p_endd];


                        bigU(p_start1:p_end,p_start:p_endd)=bigHh;

      						  p_start=p_start+Hzcc;

                   % end

             end

           else

               p_e1=0;

			p_s11=1;

p_e11=0;

bigHh=[];

                 for h=1:length(proj)

                     p_s=1;

    																	p_e=0;

                     for g=1:length(proj)

                         atom_index=[atomst(i,h), atomst1(j,g)];

                         [check1,hr]=ismember(atom_index,store_ind,'rows');

                         if check1 

                             atominn=[atomst(i,h), atomst1(j,g)];

                        				 %H=H_1{dr,hr};

                         				Hzr=proj(atominn(1));

                         Hzc=proj(atominn(2));

                         H=zeros(Hzr,Hzc);

                        p_e11=p_e1+Hzr;


                        p_e=p_e+Hzc;

                        col_range1=[p_s,p_e];

                        row_range1=[p_s11,p_e11];

                        bigHh(p_s11:p_e11,p_s:p_e)=H;

                        p_s=p_s+Hzc;

                         end 

                     end

                      % if check1

                     p_s11=p_e11+1;

                     p_e1=p_e11;

                     %end

                 end

                      [Hzrr,Hzcc]=size(bigHh);

               								 p_end=p_end1+Hzrr;

                        p_endd=p_endd+Hzcc;  

                       final_range= [p_start1,p_end;p_start,p_endd];


                        bigU(p_start1:p_end,p_start:p_endd)=bigHh;

      						  p_start=p_start+Hzcc;

           end

    end

    p_start1=p_end+1;

    p_end1=p_end;

    end



if exist('Vext','var')

			bigH=bigH+Vext;

end


bigUbar=bigU';

end