function net = cnnff(net, x)
    n = numel(net.layers);
    net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.layers{l - 1}.a{i}, net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.layers{l}.a{j} = sigm(z + net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    switch net.task
        case 'assist'
            switch net.optfun
                case 'sigm'
                    net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
                case 'softmax'
                    z = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));
                    z = exp(bsxfun(@minus, z, max(z,[],1)));
                    net.o = bsxfun(@rdivide, z, sum(z, 1)); 
            end  
        case 'main'
            cellresult = net.cellresult;
            binalresult(1,:) = cellresult(1,:);
            binalresult(2,:) = cellresult(2,:) + cellresult(3,:);
            switch net.optfun
                case 'sigm'
                    net.o = sigm(binalresult.*(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2))));
                case 'softmax'
                    z = net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2));
                    z = binalresult.*z;
                    z = exp(bsxfun(@minus, z, max(z,[],1)));
                    net.o = bsxfun(@rdivide, z, sum(z, 1)); 
            end              
    end

end
