function handle = skelVisualise_shima_BFM(vals)

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
handle(1) = plot3(vals(:, 1), vals(:, 3), vals(:, 2), '.');
axis ij % make sure the left is on the left.
set(handle(1), 'markersize', 15);
%/~
%set(handle(1), 'visible', 'off')
%~/
hold on
grid on
a={1:5,6:10, 11:14,15:19, [20:25,20]...
    [26:31,26],[32:43,32], [44:51,44], [14,17],[14,15],[14,19]};
for j = 1 : length(a)
    a1 = a{j};
for i = 1:length(a1)-1
  handle(i) = line([vals(a1(i), 1) vals(a1(i+1), 1)], ...
              [vals(a1(i), 3) vals(a1(i+1), 3)], ...
              [vals(a1(i), 2) vals(a1(i+1), 2)]);
  set(handle(i), 'linewidth', 2);
end
end
 view(3) 
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
