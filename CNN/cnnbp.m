function net = cnnbp(net, y)
    n = numel(net.layers);
    %   error
    net.e = net.o - y;
    %%  backprop deltas

    cellresult = net.cellresult;
    binalresult(1,:) = cellresult(1,:);
    binalresult(2,:) = cellresult(2,:) + cellresult(3,:);
    switch net.optfun
        case 'sigm'
            net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
            net.od = net.e .* (net.o .* (1 - net.o));           
        case 'softmax'
            switch net.task
                case 'main'
                    net.L = -sum(sum(y .*log(net.o))) / size(net.e, 2) + net.lambda/2*sum(sum(net.ffW.^2, 1), 2);
                case 'assist'
                    net.L = -sum(sum(y .*log(net.o))) / size(net.e, 2) + net.lambda/2*sum(sum(net.ffW.^2, 1), 2);                    
            end
            net.od = net.e;
    end
      

    net.fvd = (net.ffW' * net.od);              %  feature vector delta
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has sigm function
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv));
    end

    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's')
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                     z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    %%  calc gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end
    switch net.task
        case 'main'
            net.dffW = (net.od .* binalresult)*(net.fv)' / size(net.od, 2) + net.lambda.*net.ffW;
            net.dffb = mean(net.od, 2);
        case 'assist'
            net.dffW = net.od * (net.fv)' / size(net.od, 2) + net.lambda.*net.ffW;
            net.dffb = mean(net.od, 2);
    end

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end
