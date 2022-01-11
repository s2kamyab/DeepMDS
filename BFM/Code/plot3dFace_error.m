function t = plot3dFace_error(sub_t , sub_p , tl)
%% plotting the 3d shape
% The solution is to use Delaunay triangulation. Let's look at some
% info about the "tri" variable.
% if max(points_3D(:))>1
%     points_3D = mapminmax(points_3D , 0 , 1);
% end
%  figure;
t = sqrt(sum((sub_t.shape - sub_p.shape).^2 , 2));
% t = mapminmax(t , 0 , 1);
tri = tl;% delaunay(sub.shape(:, 1), sub.shape(:, 2));
% Plot it with TRISURF
% if max(t) > 1
%     h = trisurf(tri, sub_t.shape(:, 1),sub_t.shape(:, 2), sub_t.shape(:, 3) , 'FaceVertexCData', t./max(t), 'FaceColor' , 'interp', 'EdgeColor','interp');%, 'LineWidth',5);%'CDataMapping','scaled'););
% else
    h = trisurf(tri, sub_t.shape(:, 1),sub_t.shape(:, 2), sub_t.shape(:, 3) , 'FaceVertexCData', t, 'FaceColor' , 'interp', 'EdgeColor','interp');%, 'LineWidth',5);%'CDataMapping','scaled'););
% end
axis vis3d
axis tight
% view([-50 , -30 , 300]);
view([0 , 0 , 300]);
drawnow;
camlight right
% set(gca , 'Projection' , 'perspective');

% Clean it up
axis off
% l = light('Position',[-50 -15 29]);
% l = light('Position',[0 0.5 0.5]);
% set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp
colorbar EastOutside;
colormap jet
caxis([0,1.2e4]);
% h.Clim=[0,10e4];
% title('MSE-heatmap');

