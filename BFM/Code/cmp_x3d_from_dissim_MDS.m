clear;
close all;
for k =1 : 445%size(pred_3d_sim,1)
    disp(k);
    D = zeros(72);
    t=importdata(['..\results\pred_dissim_ours\', mat2str(k), '.mat']);%spred_3d_sim(k,:);
    max_dist = max(t)-min(t);
    for i = 1 : 72
        for j = 1 : 72
            D(i,j)=t(72*(i-1)+j) ;
            D(j,i) = D(i,j);
        end
        D(i,i) = 0;
    end
    
    K =abs(D);%nearestSPD(D);%-1/2 * H * D * H;
    Y = real(mdscale(K,3));
    save(['..\results\pred_x3d_ours\',mat2str(k),'.mat'], 'Y')
    
end
