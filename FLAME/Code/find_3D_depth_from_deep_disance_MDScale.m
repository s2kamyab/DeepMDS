clear;
close all;
% file_list = dir('E:\thesis_phd\MDS\FLAME\results\pred_dissim_ours\');
file_list = dir('../data/ComA_lmks\train\pred_dissim_ours\');
file_list = file_list(3:end);
for k =1 : length(file_list)
    disp(k);
    D = zeros(51);
    t=importdata(['../data/ComA_lmks\train\pred_dissim_ours\', file_list(k).name]);%spred_3d_sim(k,:);
    max_dist = max(t)-min(t);
    for i = 1 : 51
        for j = 1 : 51
            D(i,j)=t(51*(i-1)+j) ;
            D(j,i) = D(i,j);
        end
        D(i,i) = 0;
    end
    
    K =abs(D);%nearestSPD(D);%-1/2 * H * D * H;
    Y = real(mdscale(K,3));
    save(['../data/ComA_lmks\train\\pred_x3d_ours\',file_list(k).name], 'Y')
    
end
