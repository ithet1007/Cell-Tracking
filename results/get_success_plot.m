function [ s_rate ] = get_success_plot( truths,predicts, threshold )
w = 32;
h = 32;

size_h = size(truths,2);
size_w = size(threshold,2);
s_rates = zeros(size_w, size_h);
s_rate = zeros(size_w,1);
ys = zeros(4,size_h);
for i = 1:size_h
    p = predicts(i,:);
    M = [p(1) p(3) p(4); p(2) p(5) p(6)];
    corners = [ 1,-w/2,-h/2; 1,w/2,h/2;]';
    y = M*corners;
    y = y(:);
    ys(:,i) = y;
end
area_x = (truths(3,:)-truths(1,:)).*(truths(4,:)-truths(2,:));
area_y = (ys(3,:)-ys(1,:)).*(ys(4,:)-ys(2,:));
ws = [truths(1,:); truths(3,:); ys(1,:); ys(3,:)];
hs = [truths(2,:); truths(4,:); ys(2,:); ys(4,:)];
ws = reject_minmax(ws);
hs = reject_minmax(hs);
s_areas = abs(ws(1,:)-ws(2,:)).*abs(hs(1,:)-hs(2,:));
for i = 1:size_w
    s_rates(i,:) = (s_areas./(area_x+area_y-s_areas))>threshold(1,i);
    s_rate(i,1) = sum(s_rates(i,:)==1)/size_h;
end
end





function [Y] = reject_minmax(X)
leng = size(X, 2);
Y = zeros(size(X,1)-2, leng );
for i = 1:leng
    cur_x = X(:,i);
    x_max = max(cur_x);
    x_min = min(cur_x);
    cur_x(cur_x==x_max)=[];
    cur_x(cur_x==x_min)=[];
    Y(:,i) = cur_x;
end

end
