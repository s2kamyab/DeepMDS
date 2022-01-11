load rec_3d_media;
close all;
figure;

for i = 1 : 468
    plot3(rec_3d_media(i,1) ,rec_3d_media(i,2) , rec_3d_media(i,3) , '.k', 'MarkerSize' , 20);
    text(rec_3d_media(i,1)+eps ,rec_3d_media(i,2)+eps , rec_3d_media(i,3)+eps , mat2str(i));
    hold on;
%     pause;
end

mediapipe_51_lmks = [47,64,106,67,108,297,335,294,301,384,169,198,6,2,219,167,95,291,393,...
    34,161,159,174,154,145,363,386,388,360,374,381,62,41,74,12,304,271,293,322,316,17,181,91,77,81,14,312,309,403,15,179];
save mediapipe_51_lmks  mediapipe_51_lmks