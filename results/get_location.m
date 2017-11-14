function [ source_minerror ] = get_location(truths,predicts, threshold)
size_w = size(threshold,2);
x1 = (truths(1,:)+truths(3,:))/2;
x2 = (truths(2,:)+truths(4,:))/2;
y1 = predicts(:,1)';
y2 = predicts(:,2)';
source_errors = sqrt((x1-y1).^2 + (x2-y2).^2);
source_minerror = sum(source_errors,2)/size_w;

