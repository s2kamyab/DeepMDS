function handle = skelVisualise_shima_BFM_2d(vals)

% SKELVISUALISE For drawing a skel representation of 3-D data.
% FORMAT
% DESC draws a skeleton representation in a 3-D plot.
% ARG channels : the channels to update the skeleton with.
% ARG skel : the skeleton structure.
% RETURN handle : a vector of handles to the plotted structure.
%
% SEEALSO : skelModify
%
% COPYRIGHT : Neil D. Lawrence, 2005, 2006
  
% MOCAP

% if nargin<3
%   padding = 0;
% end
% channels = [channels zeros(1, padding)];
% vals = skel2xyz(skel, channels);
% load('86\x3d\i.mat','i');
% save(['86/x3d/',mat2str(i),'.mat'],'vals');
% i=i+1;
% save '86\x3d\i.mat' i

% connect = skelConnectionMatrix(skel);

% indices = find(connect);
% [I, J] = ind2sub(size(connect), indices);
handle(1) = plot(vals(:, 1),vals(:, 2), '.');
axis ij % make sure the left is on the left.
set(handle(1), 'markersize', 20);
%/~
%set(handle(1), 'visible', 'off')
%~/
hold on
grid on
a={[1,3,12,4,1],[59,62,6,64,63,62], [64,15,13,9,14 19,29],[11,18,23,22,20,11],...
    [36,35,38,47,40,36],[65,24,25,26,27,28,29,30],[66,42,45,49],[49,44,39,29]...
    [66,67,68,60,52,66],[46,53,57,55,46],...
    [7,8,10,17,21,31,37,41,48,50,51,43,33,16,8,7],[7,8,32,50,51],[33,34,70,71]};
for j = 1 : length(a)
    a1 = a{j};
for i = 1:length(a1)-1
  handle(i) = line([vals(a1(i), 1) vals(a1(i+1), 1)], ...,
              [vals(a1(i), 2) vals(a1(i+1), 2)]);
  set(handle(i), 'linewidth', 3);
end
end
  
%   handle(2) = line([vals(3, 1) vals(12, 1)], ...
%               [vals(3, 3) vals(12, 3)], ...
%               [vals(3, 2) vals(12, 2)]);
%   set(handle(1), 'linewidth', 6);
%   
%   handle(2) = line([vals(3, 1) vals(12, 1)], ...
%               [vals(3, 3) vals(12, 3)], ...
%               [vals(3, 2) vals(12, 2)]);
%   set(handle(1), 'linewidth', 6);
% end




axis equal
xlabel('x')
ylabel('z')
zlabel('y')
axis on
