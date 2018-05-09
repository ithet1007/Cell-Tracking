function [result] = cnncelltest(net, x)
    %  feedforward
    net.task = 'assist';
    net = cnnff(net, x);
    result = net.o;
    
    for i = 1:size(result,1)
       for j = 1:size(result,2)
          if result(i,j)>0.8
              result(i,j) = 0.8;
          elseif result(i,j)<0.1
              result(i,j) = 0.1;
          end
       end
    end
end
