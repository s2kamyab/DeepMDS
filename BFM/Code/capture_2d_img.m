function cropped_input_image = capture_2d_img(sub,tl)
figure;plot3dFace(sub , tl);
saveas(gcf,'temp.png')
input_image = double(imread('temp.png'));
%input_image = input_image / 255;
input_image_1 = reshape(input_image(:,:,1) , [size(input_image,1),size(input_image , 2)]);
input_image_2 = reshape(input_image(:,:,2), [size(input_image,1),size(input_image , 2)]);
input_image_3 = reshape(input_image(:,:,3), [size(input_image,1),size(input_image , 2)]);

[tx , ty] = find( input_image_1 ~= 255);
input_image_1(: , 1:min(ty)) = [];
input_image_1(: , max(ty)-min(ty):end) = [];
input_image_1(1:min(tx), :) = [];
input_image_1(max(tx)-min(tx):end, :) = [];
input_image_1 = imresize(input_image_1 , [500,500]);

[tx , ty] = find( input_image_2 ~= 255);
input_image_2(: , 1:min(ty)) = [];
input_image_2(: , max(ty)-min(ty):end) = [];
input_image_2(1:min(tx), :) = [];
input_image_2(max(tx)-min(tx):end, :) = [];
input_image_2 = imresize(input_image_2 , [500,500]);

[tx , ty] = find( input_image_3 ~= 255);
input_image_3(: , 1:min(ty)) = [];
input_image_3(: , max(ty)-min(ty):end) = [];
input_image_3(1:min(tx), :) = [];
input_image_3(max(tx)-min(tx):end, :) = [];
input_image_3 = imresize(input_image_3 , [500,500]);
%         figure;imshow(input_image_3./255);
%         figure;imshow(input_image_2./255);
%         figure;imshow(input_image_1./255);
cropped_input_image=[];
cropped_input_image(:,:,1) = reshape(input_image_1 , [size(input_image_1),1]);
cropped_input_image(:,:,2) = reshape(input_image_2 , [size(input_image_2),1]);
cropped_input_image(:,:,3) = reshape(input_image_3 , [size(input_image_3),1]);
%figure;imshow(cropped_input_image./255);
cropped_input_image = imresize(cropped_input_image , [96,96]);
% figure;imshow(cropped_input_image./255);

