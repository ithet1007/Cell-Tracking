function [ average_center_location_errors ] = get_precision_plot(truths,predicts, threshold)
size_w = size(threshold,2);
x1 = (truths(1,:)+truths(3,:))/2;
x2 = (truths(2,:)+truths(4,:))/2;
y1 = predicts(:,1)';
y2 = predicts(:,2)';
average_center_location_errors = zeros(size_w,1);
source_errors = sqrt((x1-y1).^2 + (x2-y2).^2);
for j = 1:size_w 
    truth = 0;
    for i = 1:size(source_errors,2)
        if source_errors(1,i)<threshold(1,j)
            truth = truth+1;
        end
    end
    average_center_location_errors(j,:) = truth/size(source_errors,2);
end
end

