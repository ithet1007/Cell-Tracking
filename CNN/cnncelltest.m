function [result_o] = cnncelltest(net, x)
    %  feedforward
    net.task = 'assist';
    net = cnnff(net, x);
    result = net.o;
    result_o(1,:) = result(1,:);
    result_o(2,:) = result(2,:) + result(3,:);
    
    for i = 1:size(result_o,1)
       for j = 1:size(result_o,2)
          result_o(i,j) = r_3(result_o(i,j));
       end
    end
end

function [y] = r_1(x)
    if x>0.5
        y=1;
    else 
        y=0;
    end
end

function [y] = r_2(x)
    y = x;
end

function [y] = r_3(x)
    if x>0.8
        y=0.8;
    elseif x<0.2
        y = 0.2;
    else
        y = x;
    end
end

function [y] = r_4(x)
    if x>=0.5
        y=0.8;
    elseif x<0.5
        y = 0.2;
    end
end

function [y] = r(x)
    y = 1;
end
