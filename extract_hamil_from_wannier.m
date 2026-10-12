function[HB,unique_trans_vec]=extract_hamil_from_wannier(capti)
caption=sprintf('%s_hr.dat',capti);
% Read the hr.dat file and extract the relevant information
file_ID_read = fopen(caption,'r');

% Read the data from the file
data_in = textscan(file_ID_read,'%f %f %f %f %f %f %f');

% Close the file
fclose(file_ID_read);

% Extract the translational vectors, orbital indices and matrix elements from the data
trans_vec = [data_in{1}, data_in{2}, data_in{3}];
orb_indices = [data_in{4}, data_in{5}];
matrix_elements = data_in{6} + 1i*data_in{7};

% Get the unique translational vectors
unique_trans_vec = unique(trans_vec, 'rows');
HB=cell(length(unique_trans_vec),1);
% Loop over each unique translational vector
for i = 1:size(unique_trans_vec, 1)
    
    % Get the indices of the elements with the current translational vector
    curr_trans_vec_indices = find(ismember(trans_vec, unique_trans_vec(i, :), 'rows'));
    
    % Get the orbital indices and matrix elements for the current translational vector
    curr_orb_indices = orb_indices(curr_trans_vec_indices, :);
    curr_matrix_elements = matrix_elements(curr_trans_vec_indices);
    
    % Create a matrix for the current translational vector with the orbital indices as row and column indices
    curr_matrix = zeros(max(curr_orb_indices(:,1)), max(curr_orb_indices(:,2)));
    curr_matrix(sub2ind(size(curr_matrix), curr_orb_indices(:,1), curr_orb_indices(:,2))) = curr_matrix_elements;
    HB{i,1}=curr_matrix;
    % Write the matrix to a text file with a name based on the translational vector
    %file_name = sprintf('%s_%d%d%d.txt',Nam, unique_trans_vec(i,1), unique_trans_vec(i,2), unique_trans_vec(i,3));
    %file_ID_wr = fopen(file_name, 'w');
   % for j = 1:size(curr_matrix, 1)
%         fprintf(fileID, '%f\t', curr_matrix(j, :));
%         fprintf(fileID, '\n');
      %   temp_curr=num2str(curr_matrix(j,:));
        % fprintf(file_ID_wr, '%s', temp_curr);
        % fprintf(file_ID_wr, '\n');
   % end
    %fclose(file_ID_wr);
    
end
end
