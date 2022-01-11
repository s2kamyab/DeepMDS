%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                          Generate Training Data for MDS + deep kernel

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
close all;
addpath('PublicMM1')
addpath('PublicMM1\matlab');
% addpath('E:\Kamyab_data\paper_1_matlabCode');
load 01_MorphableModel;
% load mean_landmarks_besel_72;
[model msz] = load_model();
load landmarks_me_72.mat landmarks; % index of 72 landmarks in besel face model in landmarks
num_of_landmarks = 72;
num_trdata = 5000;
C = [1,0,0,0;0,1,0,0;0,0,0,0];
no_occ_x2d = '..\data\No_occlusion\train\x2d\';
no_occ_x3d = '..\data\No_occlusion\train\x3d\';
no_occ_alpha = '..\data\No_occlusion\train\alpha\';
no_occ_beta = '..\data\No_occlusion\train\beta\';
no_occ_img = '..\data\No_occlusion\train\img\';

occ_x2d = '..\data\Occluded\train\x2d\';
occ_x3d = '..\data\Occluded\train\x3d\';
occ_alpha = '..\data\Occluded\train\alpha\';
occ_beta = '..\data\Occluded\train\beta\';
occ_img = '..\data\Occluded\train\img\';
theta = [-45,0,45];
rotation_axis = [0,1,0];
do_save=1;
% figure;
%% x2d, alpha
for iter = 1 : num_trdata
    iter
    alpha = randn(msz.n_shape_dim, 1);
    beta  = randn(msz.n_tex_dim, 1);
    shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
    tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );
    shape2 = reshape(double(shape) , 3 , length(shape)/3)';
    tex2 = reshape(double(tex) , 3 , length(shape)/3)';
    img = capture_2d_img(struct('shape', shape2, 'texture', tex2),tl);
    axis1=gca;
    x3d = shape2(landmarks , :);
    x3d = (x3d - min(x3d(:)))/(max(x3d(:))-min(x3d(:)));
    % figure; subplot(1,2,1);skelVisualise_shima_BFM(x3d);subplot(1,2,2);skelVisualise_shima_BFM_2d(x3d(:,1:2));
    %% save orthographic data
    if do_save==1
        save([no_occ_x3d,sprintf('%d.mat', iter)], 'x3d');
        save([no_occ_x2d,sprintf('%d.mat', iter)], 'x3d');
        save([no_occ_img,sprintf('%d.mat', iter)], 'img');
        save([no_occ_alpha,sprintf('%d.mat', iter)], 'alpha');
        save([no_occ_beta,sprintf('%d.mat', iter)], 'beta');
    end
    %%
    rr=rand;
    if rr < 0.3
        %% perspective projectopn with random parameters
        [az,el] = view(axis1);
        fov = camva(axis1);
        T = viewmtx(az+rand*90-45,el-rand*30,5*rand*fov,campos(axis1)+0*rand);
        landmarks_3d_hom = [shape2, ones(size(shape2,1),1)];
        projected = [T*landmarks_3d_hom']';
        
        img = capture_2d_img(struct('shape', projected(:,1:3), 'texture', tex2),tl);
        
        projected = projected(landmarks,1:2);
        projected = (projected - min(projected(:)))/(max(projected(:)) - min(projected(:)));
        
%         figure;subplot(1,3,1);imshow(img/255);
%         subplot(1,3,2);skelVisualise_shima_BFM_2d(projected);
%         subplot(1,3,3);skelVisualise_shima_BFM_2d(x3d(:,1:2));  title('GT');
        x2d = projected;
    elseif rr < 0.6 % posed
        h = randi(3);
        if theta(h) ~=0
            R = rotationmat3D(theta(h) , rotation_axis);
        else
            R = 1;
        end
        shape3 = shape2*R;
        img = capture_2d_img(struct('shape', shape3, 'texture', tex2),tl);
        %             figure;imshow(img/255);
        
        x2d = shape3(landmarks,1:2);
        x2d = (x2d - min(x2d(:)))/(max(x2d(:)) - min(x2d(:)));
        %     skelVisualise_shima_BFM_2d(x2d);
    else % orthographic
        x2d = x3d(:,1:2);
        x2d = (x2d - min(x2d(:)))/(max(x2d(:)) - min(x2d(:)));
    end
    %% save occluded data
    if do_save==1
        save([occ_x3d,sprintf('%d.mat', iter)], 'x3d');
        save([occ_img,sprintf('%d.mat', iter)], 'img');
        save([occ_x2d,sprintf('%d.mat', iter)], 'x2d');
        save([occ_alpha,sprintf('%d.mat', iter)], 'alpha');
        save([occ_beta,sprintf('%d.mat', iter)], 'beta');
    end
    close all;
end


%% Test data
num_tstdata = 500;
no_occ_x2d = '..\data\No_occlusion\test\x2d\';
no_occ_x3d = '..\data\No_occlusion\test\x3d\';
no_occ_alpha = '..\data\No_occlusion\test\alpha\';
no_occ_beta = '..\data\No_occlusion\test\beta\';
no_occ_img = '..\data\No_occlusion\test\img\';

occ_x2d = '..\data\Occluded\test\x2d\';
occ_x3d = '..\Occluded\test\x3d\';
occ_alpha = '..\data\Occluded\test\alpha\';
occ_beta = '..\data\Occluded\test\beta\';
occ_img = '..\data\Occluded\test\img\';% figure;
%% x2d, alpha
for iter = 1 : num_tstdata
    iter
    alpha = randn(msz.n_shape_dim, 1);
    beta  = randn(msz.n_tex_dim, 1);
    %     load alpha
    %     load beta
    shape  = coef2object( alpha, model.shapeMU, model.shapePC, model.shapeEV );
    tex    = coef2object( beta,  model.texMU,   model.texPC,   model.texEV );
    shape2 = reshape(double(shape) , 3 , length(shape)/3)';
    tex2 = reshape(double(tex) , 3 , length(shape)/3)';
    img = capture_2d_img(struct('shape', shape2, 'texture', tex2),tl);
    axis1=gca;
    x3d = shape2(landmarks , :);
    x3d = (x3d - min(x3d(:)))/(max(x3d(:))-min(x3d(:)));
    % skelVisualise_shima_BFM(landmarks_3d);
    %% save orthographic data
    if do_save==1
        save([no_occ_x3d,sprintf('%d.mat', iter)], 'x3d');
        save([no_occ_x2d,sprintf('%d.mat', iter)], 'x3d');
        save([no_occ_img,sprintf('%d.mat', iter)], 'img');
        save([no_occ_alpha,sprintf('%d.mat', iter)], 'alpha');
        save([no_occ_beta,sprintf('%d.mat', iter)], 'beta');
    end
    %%
    rr=rand;
    if rr < 0.3
        [az,el] = view(axis1);
        fov = camva(axis1);
        T = viewmtx(az+rand*90-45,el-rand*30,5*rand*fov,campos(axis1)+0*rand);
        landmarks_3d_hom = [shape2, ones(size(shape2,1),1)];
        projected = [T*landmarks_3d_hom']';
        
        img = capture_2d_img(struct('shape', projected(:,1:3), 'texture', tex2),tl);
        
        projected = projected(landmarks,1:2);
        projected = (projected - min(projected(:)))/(max(projected(:)) - min(projected(:)));
%         
%         figure;subplot(1,3,1);imshow(img/255);
%         subplot(1,3,2);skelVisualise_shima_BFM_2d(projected);
%         subplot(1,3,3);skelVisualise_shima_BFM_2d(x3d(:,1:2));  title('GT');
        x2d = projected;
%         ;imshow(img/255);
%         x2d = projected;
    elseif rr < 0.6 % posed
        h = randi(3);
        if theta(h) ~=0
            R = rotationmat3D(theta(h) , rotation_axis);
        else
            R = 1;
        end
        shape3 = shape2*R;
        img = capture_2d_img(struct('shape', shape3, 'texture', tex2),tl);
        %             figure;imshow(img/255);
        
        x2d = shape3(landmarks,1:2);
        x2d = (x2d - min(x2d(:)))/(max(x2d(:)) - min(x2d(:)));
        %     skelVisualise_shima_BFM_2d(x2d);
    else % orthographic
        x2d = x3d(:,1:2);
        x2d = (x2d - min(x2d(:)))/(max(x2d(:)) - min(x2d(:)));
    end
    %% save occluded data
    if do_save == 1
        save([occ_x3d,sprintf('%d.mat', iter)], 'x3d');
        save([occ_img,sprintf('%d.mat', iter)], 'img');
        save([occ_x2d,sprintf('%d.mat', iter)], 'x2d');
        save([occ_alpha,sprintf('%d.mat', iter)], 'alpha');
        save([occ_beta,sprintf('%d.mat', iter)], 'beta');
    end
    %     close all;
end

