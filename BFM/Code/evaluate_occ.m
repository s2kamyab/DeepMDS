clear;
close all;
addpath('PublicMM1')
addpath('PublicMM1\matlab');
% addpath('E:\Kamyab_data\paper_1_matlabCode');
load 01_MorphableModel;
load landmarks_me_72
gt_x3d_path = '..\data\Occluded\test\x3d\';
gt_alpha_path = '..\data\Occluded\test\alpha\';
pred_x3d_ours_path = '..\results\pred_x3d_ours\';
pred_x3d_PAMI = '..\results\pred_pami_occ\';
pred_x3d_aldrian = '..\results\pred_aldrian_occ\';
pred_x3d_mobileFace = '..\results\pred_mobileface_occ\';


for i = 1%:445
    i
    gt = importdata([gt_x3d_path , mat2str(i), '.mat']);
    gt = (gt - min(gt(:)))/(max(gt(:))-min(gt(:)));
    
    pred_ours = importdata([pred_x3d_ours_path , mat2str(i), '.mat']);
    [d,pred_ours,tr] = procrustes(gt,pred_ours);
    depthCorr_matrix = corr(gt', pred_ours');
    depthCorr_matrix_trace(i) = sum(diag(depthCorr_matrix));
    MSE_ours(i) = immse(gt , pred_ours);
    
    pred_pami = importdata([pred_x3d_PAMI , mat2str(i-1), '.mat']);
    pred_pami = (pred_pami - min(pred_pami))/(max(pred_pami)-min(pred_pami));
    pred_pami = [gt(:,1:2),pred_pami'];%reshape(pred_pami, [31,3]);
    [d,pred_pami,tr] = procrustes(gt,pred_pami);
    depthCorr_matrix_pami = corr(gt', pred_pami');
    depthCorr_matrix_trace_pami(i) = sum(diag(depthCorr_matrix_pami));
    MSE_pami(i) = immse(gt , double(pred_pami));
    
    pred_mobileFace_alpha = importdata([pred_x3d_mobileFace , mat2str(i), '.mat.mat']);
    pred_mobileFace = coef2object(  pred_mobileFace_alpha' , shapeMU,shapePC, shapeEV );
    pred_mobileFace = (pred_mobileFace - min(pred_mobileFace))/(max(pred_mobileFace)-min(pred_mobileFace));
    pred_mobileFace = reshape(pred_mobileFace' , [3,53490])';
    pred_mobileFace =  pred_mobileFace(landmarks,:);
    [d,pred_mobileFace,tr] = procrustes(gt,pred_mobileFace);
    depthCorr_matrix_mf = corr(gt', pred_mobileFace');
    depthCorr_matrix_trace_mf(i) = sum(diag(depthCorr_matrix_mf));
    MSE_mf(i) = immse(gt , double(pred_mobileFace));
    
    
    
    pred_aldrian = importdata([pred_x3d_aldrian  , mat2str(i), '.mat']);
    pred_aldrian = (pred_aldrian - min(pred_aldrian(:)))/(max(pred_aldrian(:))-min(pred_aldrian(:)));
    [d,pred_aldrian,tr] = procrustes(gt,pred_aldrian);
    depthCorr_matrix_aldrian  = corr(gt', pred_aldrian');
    depthCorr_matrix_trace_aldrian (i) = sum(diag(depthCorr_matrix_aldrian ));
    MSE_aldrian(i) = immse(gt , double(pred_aldrian));
end
avg_DC_ours = mean(depthCorr_matrix_trace)% mean_depthCorr = ttt/10;
avg_DC_PAMI = mean(depthCorr_matrix_trace_pami)% mean_depthCorr = ttt/10;
avg_DC_depthnet = mean(depthCorr_matrix_trace_mf)% mean_depthCorr = ttt/10;
avg_DC_aldrian = mean(depthCorr_matrix_trace_aldrian)% mean_depthCorr = ttt/10;
avgMSE_ours = mean(MSE_ours)
avgMSE_pami = mean(MSE_pami)
avgMSE_depth = mean(MSE_mf)
avgMSE_aldrian = mean(MSE_aldrian)

for i = 1:5
    idx=randi(445)
    gt = importdata([gt_x3d_path , mat2str(idx), '.mat']);
    gt = (gt - min(gt(:)))/(max(gt(:))-min(gt(:)));
    pred_ours = importdata([pred_x3d_ours_path , mat2str(idx), '.mat']);
    [d,pred_ours,tr] = procrustes(gt,pred_ours);
    
    pred_pami = importdata([pred_x3d_PAMI , mat2str(i-1), '.mat']);
    pred_pami = (pred_pami - min(pred_pami))/(max(pred_pami)-min(pred_pami));
    pred_pami = [gt(:,1:2),pred_pami'];%reshape(pred_pami, [31,3]);
    [d,pred_pami,tr] = procrustes(gt,pred_pami);
    
    pred_mobileFace_alpha = importdata([pred_x3d_mobileFace , mat2str(i), '.mat.mat']);
    pred_mobileFace = coef2object(  pred_mobileFace_alpha' , shapeMU,shapePC, shapeEV );
    pred_mobileFace = (pred_mobileFace - min(pred_mobileFace))/(max(pred_mobileFace)-min(pred_mobileFace));
    pred_mobileFace = reshape(pred_mobileFace' , [3,53490])';
    pred_mobileFace =  pred_mobileFace(landmarks,:);
    [d,pred_mobileFace,tr] = procrustes(gt,pred_mobileFace);
    
    
    pred_aldrian = importdata([pred_x3d_aldrian  , mat2str(i), '.mat']);
    pred_aldrian = (pred_aldrian - min(pred_aldrian(:)))/(max(pred_aldrian(:))-min(pred_aldrian(:)));
    [d,pred_aldrian,tr] = procrustes(gt,pred_aldrian);
    figure;
    subplot(3,3,1); skelVisualise_shima_BFM(gt);title('GT');grid on;
    subplot(3,3,2);skelVisualise_shima_BFM(pred_pami);title('PAMI');grid on;
    subplot(3,3,4); skelVisualise_shima_BFM(pred_mobileFace);title('MobileFace');grid on;
    subplot(3,3,5); skelVisualise_shima_BFM(pred_aldrian);title('Aldrian');grid on;
    
    subplot(3,3,6); handle = skelVisualise_shima_BFM(pred_ours);title('Ours');view(3);
    hold off
end
