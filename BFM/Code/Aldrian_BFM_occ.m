close all;
% my_addPath;
addpath('PublicMM1')
addpath('PublicMM1\matlab');
load 01_MorphableModel;
% load mean_landmarks_besel;
load landmarks_me_72
[model msz] = load_model();
mean_shape = double(shapeMU);%160470*1
mean_shape2 = reshape(mean_shape , 3 , length(mean_shape)/3 )';
mean_landmarks = mean_shape2(landmarks,:);
indexes = [1:length(mean_shape)]';
indexes = reshape(indexes , 3 , length(indexes)/3)';
newIndexes = indexes(landmarks , :);
newIndexes = newIndexes';
newIndexes = newIndexes(:);
U = double(shapePC); % 160470*199
U = U(newIndexes , :);
T = double(texPC);% 160470*199
T = T(newIndexes , :);
mean_texture = double(texMU);
meanFace = struct();
meanFace.shape = mean_shape2;
num_of_landmarks = 72;
landmark_path = '..\data\Occluded\test\x2d\';
landmarks_files = dir(landmark_path);
landmarks_files = landmarks_files(3:end);
pred_path = '..\results\pred_aldrian\';
for iii = 1 : length(landmarks_files)
    iii
    pts_2d = importdata([landmark_path,mat2str(iii),'.mat']);
%     pts_2d = (pts_2d - min(pts_2d))/(max(pts_2d)-min(pts_2d));
    pts_2d = reshape(pts_2d,[72,2]);
    pts_2d = pts_2d(:,1:2);
    %% import out of sample faces in order to compute noise and generalization error
    load out_of_sample_faces_besel; % mean face is the first element in out_of-_sample_faces already
    N = num_of_landmarks;
    shape_param_vars = std(U)';
    %% finding the index of sparse feature points
%     indexes = [1:length(mean_shape)]';
%     indexes = reshape(indexes , size(meanFace.shape'))';
% %     mean_landmarks = mean_landmarks';
%     featurePointsInds = [];
    n1 = 1;
    % figure;
    for i = 1 :3: 3*N
%         dists = d(repmat(mean_landmarks(i , :) , size(meanFace.shape , 1) , 1) , meanFace.shape , 1);
%         [~ , ind] = min(dists);
%         featurePointsInds = [featurePointsInds , indexes(ind , :)];
        V_h(n1:n1+2, :) = U(i:i+2 , :);
        V_h(n1+3 , :) = 0;
        n1 = n1 + 4;
        %     plot3(meanFace.shape(ind,1) , meanFace.shape(ind,2) ,meanFace.shape(ind,3) , '*r');
        %     hold on;
        %     plot3(mean_landmarks(i ,1) , mean_landmarks(i ,2) ,mean_landmarks(i ,3) , '-ok');
    end
    % sigma = sqrt(sigma);
    %% modeling generalization error and noise
    sigma_2_3d = 0;
    for k = 2 : length(out_of_sample_faces)
        tt = out_of_sample_faces{k}.shape';
        vi = tt(:);
        vi = vi(newIndexes);
%         vi = vi(featurePointsInds);
        % out of sample reconstruction
        COEF = object2coef(vi, mean_shape(newIndexes), U, shapeEV);
        COEF = COEF ./ shape_param_vars;
        vi_prim = U*COEF + mean_shape(newIndexes);
%         vi_prim = vi_prim(featurePointsInds);
        ei = (vi - vi_prim).^2;
        e_hat = ei;%(featurePointsInds);
        e_hat = reshape(e_hat , 3,length(e_hat)/3)';
        sigma_2_3d = sigma_2_3d + e_hat ;
    end
    sigma_2_3d = sigma_2_3d / (length(out_of_sample_faces) - 1);
    
    
    
    %% |============================|
    %% |    modeling Shape          |
    %% |============================|
    %%
    X = [mean_landmarks' ;ones(1 , size(mean_landmarks , 1))]';
    x = [pts_2d';ones(1 , size(pts_2d , 1))]';
    %
    % calculating normalization matrices so that the mean = 0 and avg RMS = sqrt(2) , sqrt(3)
    m_bar = mean(x);
    s = sqrt(2)/ mean(sqrt(sum((x - repmat(m_bar , size(x ,1) , 1)).^2 , 2)));
    T2 = [s , 0 , (-s)*m_bar(1);...
        0 , s , (-s)*m_bar(2); ...
        0 , 0 ,1];
    x_norm = transpose(T2*x');
    
    t2 = mean_shape(newIndexes);
%     t2 = t2(featurePointsInds , :);
    t2 = reshape(t2' , 3,N)';
    % t2 = mean_landmarks;
    t2 = [t2 , ones(N , 1)]';
    t2 = t2(:);% this is v_bar in the paper
    y = reshape(x' , 3*N , 1);
%     figure;
    dpi = get(0, 'ScreenPixelsPerInch');
    inches = 1 ./ dpi;
    pixel_mm = inches * 25.4;
    for iter = 1 : 50
        m_bar = mean(X);
        s = sqrt(3)/ mean(sqrt(sum((X - repmat(m_bar , size(X ,1) , 1)).^2 , 2)));
        U2 = [s , 0 , 0 ,  (-s)*m_bar(1);...
            0 ,   s ,0 , (-s)*m_bar(2); ...
            0 ,    0 , s , (-s)*m_bar(3);...
            0 , 0, 0 ,1];
        X_norm = transpose(U2*X');
        %     mean(sqrt(abs(X_norm(: , 1)).^2 + abs(X_norm(: , 2)).^2 + abs(X_norm(: , 3)).^ 2)) %mean RMS
        %     (1/size(X_norm , 1))*sum(X_norm(:,1:3)) % mean
        
        
        C_tilda = transpose(inv(X_norm'*X_norm)*X_norm'*x_norm);% Least Square solution of: argmin_c |x - c_tildaX|^2
        C = inv(T2) * C_tilda * U2;
%             tt = C*X';
%             hold off;
%             plot(tt(1,:) , tt(2,:) , '*r');hold on;plot(x(:,1) , x(:,2) , 'ok');
%             pause(0.001);
        %% Calculating shape parameters
        P = zeros(3*N , 4*N);
        N1 = 1;
        N2 = 1;
        for n = 1 : N
            P(N1 : N1 + 2 , N2 : N2 + 3) = C;
            N1 = N1 + 3;
            N2 = N2 + 4;
        end
        A = P * V_h;        
        b = P * t2 - y;
        %% modeling generalization error and noise
        %     C(logical(eye(size(C)))) = 1;
        C_bar = C;
        C_bar(1,end) = 0; % removing translational component
        C_bar(2,end) = 0;
        
        %     sigma_2_3d_t = reshape(sigma_2_3d , 3,N)';
        hom_sigma_2_3d = [sigma_2_3d , ones(N , 1)];
        temp = C_bar * hom_sigma_2_3d'+ ((sqrt(3)*pixel_mm)^2);
        sigma_2_2d = temp(:);
        omega = eye(3*N);
        omega(logical(eye(size(omega)))) = 1 ./ sigma_2_2d;% set diagonal elemets to error
        
        %%
        lambda = 1e-7;
        t1 = A'*omega*A;
        t3 = A'*omega'*b;
        t11 = t1+lambda*eye(size(t1));
        cs = -inv(t11)*(t3);% variance normalized shape parameters
        cs = (cs - mean(cs))/std(cs);
        newShape = coef2object(cs, mean_shape(newIndexes), U, model.shapeEV);
        newShape = double(newShape);
        newShape = reshape(newShape' , [3,72])';
        X = [newShape'; ones(1 , size(mean_landmarks , 1))]';
%             plot3(newShape(: , 1) , newShape(:, 2) , newShape(: , 3) , '.r');
        %     pause(0.01);
    end
    save([pred_path,mat2str(iii),'.mat'], 'newShape');
end
